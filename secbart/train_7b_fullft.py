"""Standalone bottleneck-token cotrain trainer (HF, no verl/FSDP).

verl's FSDP path fails to update tied bottleneck-token embedding rows (plain
torch verifies grads are large); this trainer reproduces the exact cotrain
objective (vulcode compression + seccode repair + token-level security head)
and saves a merged HF model plus security_head.pt, so the standard eval
pipeline works unchanged. Use --lora_rank 0 for full fine-tuning (0.5B),
>0 for LoRA (7B).
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from secbart.bottleneck_token_common import (
    _process_sample,
    BottleneckQFormer,
    SecurityHead,
    VULN_TOK,
    VULN_TOKS,
    SECU_TOK,
    JFIX_TOK,
    JKEEP_TOK,
    FUNC_ANAL_TOK,
    VULN_ANAL_TOK,
    SECU_IMPL_TOK,
    THINK_TOK,
    THINK_FUNC_ANAL_TOK,
    THINK_VULN_ANAL_TOK,
    THINK_SECU_IMPL_TOK,
    CLS_NEUTRAL,
    CLS_SAFE,
    CLS_UNSAFE,
    CLS_IGNORE,
)


class RawSFTDataset(Dataset):
    def __init__(self, path, max_samples=-1, seed=42, extra_path=None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if extra_path and Path(extra_path).exists():
            data = data + json.loads(Path(extra_path).read_text(encoding="utf-8"))
        if max_samples > 0 and max_samples < len(data):
            rng = np.random.default_rng(seed=seed)
            idx = rng.choice(len(data), size=max_samples, replace=False)
            data = [data[int(i)] for i in idx]
        # 保留全部字段：think 布局（--think_mode plain/block/triple）需要
        # think / func_anal / vuln_anal / secu_impl 字段（general 样本无则走
        # 默认布局，混合训练）。
        self._items = list(data)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def collate_fn(batch, tokenizer, cfg):
    processed = [
        _process_sample(
            s, tokenizer, cfg["n_vuln"], cfg["max_length"], cfg["truncation"],
            cfg["vulcode_loss_weight"], cfg["cls_neutral_weight"],
            cfg["label_mode"], cfg["skip_modes"], cfg["label_align"],
            cfg["overlap_ratio"], cfg.get("vulcode_see_input", False),
            cfg.get("seccode_see_vulcode", False),
            cfg.get("multi_vuln", False),
            cfg.get("interleave", False), cfg.get("keep_up", 1.0),
            cfg.get("judge", False),
            cfg.get("vuln_anal_vis", True), cfg.get("seccode_cond", False),
            cfg.get("think_mode", "none"),
        )
        for s in batch
    ]
    max_seq = max(p[0].shape[0] for p in processed)
    B = len(processed)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((B, max_seq), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, max_seq), dtype=torch.long)
    loss_mask = torch.zeros((B, max_seq), dtype=torch.float)
    attention_mask = torch.zeros((B, 1, max_seq, max_seq), dtype=torch.bool)
    cls_labels = torch.full((B, max_seq), -100, dtype=torch.long)
    cls_weights = torch.zeros((B, max_seq), dtype=torch.float)
    inp_lens = torch.zeros(B, dtype=torch.long)
    for i, (ids, mask, pos, lm, cl, cw, L) in enumerate(processed):
        T = ids.shape[0]
        input_ids[i, :T] = ids
        position_ids[i, :T] = pos
        loss_mask[i, :T] = lm
        attention_mask[i, 0, :T, :T] = mask
        cls_labels[i, :T] = cl
        cls_weights[i, :T] = cw
        inp_lens[i] = L
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "position_ids": position_ids, "loss_mask": loss_mask,
        "cls_labels": cls_labels, "cls_weights": cls_weights,
        "inp_lens": inp_lens,
    }


def _forward_cotrain(model, qformer, ids, attn, pos, inp_lens, n_vuln, device):
    """One cotrain forward; with ``qformer`` enabled this is two-pass:
    pass 1 encodes input+<vuln> per sample and runs the QFormer (explicit
    cross-attention compression), pass 2 runs the full sequence with the
    refined <vuln> states injected as input embeddings. Mask / positions /
    loss targets are identical to the single-pass path."""
    if qformer is None:
        return model(input_ids=ids, attention_mask=attn, position_ids=pos,
                     use_cache=False, output_hidden_states=True)
    B = ids.shape[0]
    q_states = []
    for bi in range(B):
        L_i = int(inp_lens[bi])
        T1 = L_i + n_vuln
        ids1 = ids[bi, :T1].unsqueeze(0)
        pos1 = torch.arange(T1, dtype=torch.long, device=device).unsqueeze(0)
        mask1 = torch.tril(
            torch.ones((1, 1, T1, T1), dtype=torch.bool, device=device))
        out1 = model.model(input_ids=ids1, attention_mask=mask1,
                           position_ids=pos1, use_cache=False,
                           output_hidden_states=True)
        h1 = out1.hidden_states[-1]  # (1, T1, H)
        qs = qformer(h1[:, :L_i], h1[:, L_i:T1])  # (1, N, H)
        q_states.append(qs[0])
    emb = model.model.embed_tokens(ids)
    for bi in range(B):
        L_i = int(inp_lens[bi])
        emb[bi, L_i:L_i + n_vuln] = q_states[bi]
    return model(inputs_embeds=emb, attention_mask=attn, position_ids=pos,
                 use_cache=False, output_hidden_states=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_train", type=int, default=-1)
    ap.add_argument("--max_val", type=int, default=525)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--micro_batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--emb_lr_mult", type=float, default=10.0,
                    help="learning-rate multiplier for the embedding matrix "
                         "(incl. new bottleneck rows); 1.0 = same as base lr")
    ap.add_argument("--emb_init", choices=["mean", "random", "keep"], default="mean",
                    help="init the new <vuln>/<secu> rows: mean of pretrained "
                         "embedding rows (+tiny noise), keep random init, or "
                         "keep (warm start: 从已含训练过瓶颈行的 merged 模型加载，"
                         "不动 embedding/lm_head 行)")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup_frac", type=float, default=0.1,
                    help="线性 warmup 占总步数比例 (默认 0.1, 与改前一致)。"
                         "续训(多 epoch 阶梯)时传 0 可避免每段重跑 LR 爬坡的伪影。")
    ap.add_argument("--export_epoch_ckpts", action="store_true",
                    help="每个 epoch 结束额外导出 merged_hf_model_ep{N} / security_head_ep{N}.pt "
                         "(默认关闭, 关闭时行为与改前逐字一致)")
    ap.add_argument("--epoch_offset", type=int, default=0,
                    help="续训时已完成的 epoch 数: 导出的 ckpt 编号从 offset+1 起 (默认 0)")
    ap.add_argument("--freeze_layers", type=int, default=0,
                    help="抗遗忘: 冻结前 N 层 transformer block (0=全参训练)。"
                         "底层保基座先验, 只让上层适配安全目标; 与 --lora_rank 互斥使用。")
    ap.add_argument("--n_vuln", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--sec_up", type=float, default=1.0)
    ap.add_argument("--sequential", action="store_true",
                    help="顺序课程：DataLoader 不 shuffle，文件行序 = 训练顺序（行序即课程）")
    ap.add_argument("--lr_map", type=str, default=None,
                    help="顺序课程段切学习率：\"step:lr,...\"（该全局步起生效，升序；"
                         "首个边界前用 --lr）；缺省恒为 --lr")
    ap.add_argument("--sec_up_map", type=str, default=None,
                    help="顺序课程段切 sec_up：\"step:sec_up,...\"，同上；缺省恒为 --sec_up")
    ap.add_argument("--judge", action="store_true",
                    help="两段条件生成：`<secu>` 前插 `<jfix>`/`<jkeep>` 判定 token "
                         "（训练真实标签；与 interleave 互斥）")
    ap.add_argument("--extra_data", type=str, default=None,
                    help="追加训练数据（如失败修复错题本 json），与原数据拼接")
    ap.add_argument("--vul_fix_w", type=float, default=1.0,
                    help="token 级安全标签：vulcode 段 UNSAFE（漏洞）token 的 "
                         "NLL 乘数（>1 让压缩瓶颈优先保真漏洞细节，与 keep_up "
                         "保'不变部分'对称）；1.0 = 不生效")
    ap.add_argument("--vul_ul", type=float, default=0.0)
    ap.add_argument("--cls_weight", type=float, default=0.1)
    ap.add_argument("--vulcode_loss_weight", type=float, default=0.5,
                    help="vul-code 段的辅助 NLL 权重（默认 0.5 = 旧行为）；0 = 不训练模型复现脆弱代码")
    ap.add_argument("--neutral_weight", type=float, default=0.1)
    ap.add_argument("--cls_focal_gamma", type=float, default=0.0,
                    help="focal-loss gamma for the security-head CE loss "
                         "(0.0 = plain CE); down-weights easy/high-confidence "
                         "tokens to focus on hard/rare ones")
    ap.add_argument("--cls_pos_weight", type=float, default=1.0,
                    help="extra multiplier on SAFE/UNSAFE class CE for the "
                         "security head (vs neutral); 1.0 = plain class weights")
    ap.add_argument("--cls_label_smooth", type=float, default=0.0,
                    help="label smoothing for the security-head CE (0.0 = off)")
    ap.add_argument("--head_n_layers", type=int, default=1,
                    help="aggregate the last K decoder layers' hidden states "
                         "(mean) as the security-head input; DeepGuard-style "
                         "multi-layer aggregation")
    ap.add_argument("--qformer_layers", type=int, default=0,
                    help="QFormer decoder layers for explicit bottleneck "
                         "compression (0=off, default implicit causal <vuln>)")
    ap.add_argument("--qformer_heads", type=int, default=8)
    ap.add_argument("--qformer_dropout", type=float, default=0.1)
    ap.add_argument("--label_mode", default="ast")
    ap.add_argument("--label_align", default="difflib")
    ap.add_argument("--overlap_ratio", type=float, default=0.0)
    ap.add_argument("--vulcode_see_input", action="store_true",
                    help="ablation: let vulcode attend to the input prompt "
                         "(default: input blocked -> bottleneck compression)")
    ap.add_argument("--seccode_see_vulcode", action="store_true",
                    help="ablation: let seccode/eos rows attend to vulcode cols "
                         "(default: vulcode blocked -> repair info via <vuln> only; "
                         "训练有/推理无的不对称对照)")
    ap.add_argument("--multi_vuln", action="store_true",
                    help="L53: each bottleneck slot uses its own token "
                         "<vuln1>..<vulnN> instead of <vuln> repeated N times")
    ap.add_argument("--interleave", action="store_true",
                    help="chunk-interleaved 布局（PIC+EPL 的 vLLM 一致实现）："
                         "input 切 N 块交错插 <vuln1..N>，标准 causal 即局部感受野，"
                         "训练/推理掩码天然一致（强制 multi-vuln token）")
    ap.add_argument("--keep_up", type=float, default=1.0,
                    help="PRepair 最小编辑：seccode 保留（未变化）token 的 NLL 权重")
    ap.add_argument("--vuln_anal_vis", type=int, default=1, choices=[0, 1],
                    help="三段式布局：vuln_anal 段生成时能否看 vulcode 原文 "
                         "(1=看，训练有/推理无的不对称对照；0=只靠 <vuln> 压缩状态，"
                         "与推理一致)")
    ap.add_argument("--seccode_cond", action="store_true",
                    help="三段式布局：seccode/<secu> 行挖掉 FA/VA/SI 三段文本列，"
                         "只留 <func_anal>/<vuln_anal>/<secu_impl> token 作条件信号")
    ap.add_argument("--icae_weight", type=float, default=0.0,
                    help="ICAE 重建 loss 权重：<vuln> 段 hidden 均值回归 vulcode 段 "
                         "hidden 均值（aux MSE，目标 detach；仅默认布局）")
    ap.add_argument("--sec_mod", type=float, default=0.0,
                    help="安全头软调制：seccode NLL ×(1+λ·max(0,p_safe−p_unsafe))，"
                         "权重 stop-grad")
    ap.add_argument("--sec_head", type=str, default=None,
                    help="冻结加载的安全头（离线 PPO 去自举）：从该 path 加载 "
                         "SecurityHead state_dict（如基线 security_head.pt），"
                         "requires_grad=False + eval 模式作纯外部 reward；"
                         "配合 --sec_mod（线性）或 --sec_rwr（指数）加权")
    ap.add_argument("--sec_rwr", type=float, default=0.0,
                    help="RWR 指数加权：seccode NLL ×exp(max(0,p_safe−p_unsafe)/β)，"
                         "权重 stop-grad；β>0 启用（与 --sec_mod 互斥，sec_mod 优先）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit", "paged_adamw8bit"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--think_mode", choices=["none", "plain", "block", "triple", "front"],
                    default="none",
                    help="隐式思维链布局：#3 plain = think 前置全 causal；"
                         "#6 front = 链前置生成目标 inp|<think>链</think>|<vuln>*N|"
                         "vulcode|<secu>|seccode（seccode 见 input+think+<vuln>+<secu>，"
                         "挖 vulcode 列；vulcode 仍盲；链 loss=0.5，两段式推理）；"
                         "#4 block = <think>*4 双层瓶颈（think 从摘要重建，"
                         "vulcode 只见 <think>+<vuln>）；#5 triple = 三个 think "
                         "token 各自生成 func_anal/vuln_anal/secu_impl 一段。"
                         "样本含对应字段才启用，general 样本走默认布局")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"[seed] seed={args.seed} (torch/np/random fixed; DataLoader generator pinned, deterministic)")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    extra_toks = VULN_TOKS[: args.n_vuln] if (args.multi_vuln or args.interleave) else []
    if args.judge:
        extra_toks = extra_toks + [JFIX_TOK, JKEEP_TOK]
    # 三段式思维链 token 恒注册（数据含 func_anal 字段即用；无三段样本不受影响）
    extra_toks = extra_toks + [FUNC_ANAL_TOK, VULN_ANAL_TOK, SECU_IMPL_TOK]
    # 隐式思维链 token 恒注册（think_mode 布局用；无 think 样本不受影响）
    extra_toks = extra_toks + [THINK_TOK, THINK_FUNC_ANAL_TOK,
                               THINK_VULN_ANAL_TOK, THINK_SECU_IMPL_TOK]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [VULN_TOK, SECU_TOK] + extra_toks})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    # Keep embeddings TIED (Qwen2.5-Coder default). Untying is unnecessary in
    # the standalone trainer (it was only a verl/FSDP workaround) and broke
    # checkpointing: with a shared lm_head/embed object, save_pretrained omits
    # lm_head.weight, so reloading produced a random lm_head (garbage output).
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if args.lora_rank == 0:
        # Full FT works for both tied (0.5B) and untied (7B) bases. For untied
        # bases lm_head.weight is an independent matrix; both it and the
        # embedding participate in the main optimizer (id-dedup keeps both).
        if model.config.tie_word_embeddings:
            assert model.lm_head.weight is model.model.embed_tokens.weight
        else:
            print("[standalone] untied base: emb + lm_head are separate, both trainable")

    # New bottleneck rows: mean-init (proven to train faster/more stably than
    # random for vocabulary expansion) or keep random.
    v_id = tokenizer.convert_tokens_to_ids(VULN_TOK)
    s_id = tokenizer.convert_tokens_to_ids(SECU_TOK)
    v_ids = (
        [tokenizer.convert_tokens_to_ids(t) for t in VULN_TOKS[: args.n_vuln]]
        if args.multi_vuln else [v_id]
    )
    # Capture the embedding weight BEFORE PEFT wrapping: get_peft_model moves
    # embed_tokens into a ModulesToSaveWrapper and the attribute path changes.
    emb_w = model.model.embed_tokens.weight
    if args.emb_init == "mean":
        with torch.no_grad():
            pretrained_rows = emb_w[: min(v_id, s_id)]
            mean_row = pretrained_rows.mean(dim=0)
            noise = torch.randn_like(mean_row) * 0.02
            for rid in v_ids:
                emb_w[rid] = mean_row + noise
            emb_w[s_id] = mean_row - noise
        print(f"[standalone] bottleneck rows mean-initialized "
              f"(vuln{'1..' + str(len(v_ids)) if args.multi_vuln else ''}/secu "
              f"anti-symmetric noise)")
    else:
        print(f"[standalone] bottleneck rows keep existing "
              f"({'random init' if args.emb_init == 'random' else 'warm-start loaded'})")

    if not model.config.tie_word_embeddings and args.emb_init != "keep":
        # 7B/untied base: keep lm_head's new rows consistent with the emb rows.
        # warm start（keep）时 emb/lm_head 行都是训练过的，跳过同步
        with torch.no_grad():
            for rid in v_ids:
                model.lm_head.weight[rid] = emb_w[rid]
            model.lm_head.weight[s_id] = emb_w[s_id]
        print("[standalone] lm_head bottleneck rows synced")

    if args.lora_rank > 0:
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank, target_modules="all-linear",
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.enable_input_require_grads()
        # 瓶颈行(新 <vuln>/<secu>)必须能学: 让 embed 可导(才有 grad), 但**不进主优化器**,
        # 只由下面的 row-restricted Adam 更新那几行 -> 预训练 embedding 行严格不动 (抗遗忘关键).
        # 刻意不用 modules_to_save=["embed_tokens"]: peft 会 deepcopy 该模块, forward 走副本,
        # 而 row-only 更新打在原模块上 => 静默失效, 且整张 embed 矩阵会被主优化器按 LoRA 的 lr 更新.
        _emb_mod = model.get_input_embeddings()
        _emb_mod.weight.requires_grad_(True)
        _same = _emb_mod.weight is emb_w
        print(f"[standalone] LoRA rank {args.lora_rank} "
              f"(embed trainable but excluded from main optimizer; row-only Adam; "
              f"embed_identity_ok={_same})")
        if not _same:
            print("[standalone][WARN] embed param identity mismatch: row-only emb update "
                  "may be ineffective")
        model = model.to(torch.bfloat16)  # keep LoRA adapters in bf16 (matches base)
    else:
        print("[standalone] full fine-tuning")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = model.to(device)
    model.train()

    hidden = model.config.hidden_size
    head = SecurityHead(hidden_size=hidden).to(torch.bfloat16).to(device)
    head.train()
    if args.sec_head:
        # 离线 PPO：加载已收敛安全头并冻结，作纯外部 reward（无自举）。
        # eval() 关掉 dropout，reward 输出稳定；参数不进 optimizer。
        head.load_state_dict(torch.load(args.sec_head, map_location="cpu"))
        head.requires_grad_(False)
        head.eval()
        print(f"[standalone] security head FROZEN (external reward) from {args.sec_head}")
    if args.interleave and args.qformer_layers > 0:
        raise RuntimeError("--interleave 与 qformer 互斥（qformer 按连续 vuln 段定位）")
    if args.judge and args.interleave:
        raise RuntimeError("--judge 与 --interleave 互斥（judge 按连续 vuln/vulcode 段定位）")
    qformer = None
    if args.qformer_layers > 0:
        qformer = BottleneckQFormer(
            hidden_size=hidden, n_queries=args.n_vuln,
            n_layers=args.qformer_layers, n_heads=args.qformer_heads,
            dropout=args.qformer_dropout,
        ).to(torch.bfloat16).to(device)
        qformer.train()
        print(f"[standalone] QFormer: {args.qformer_layers} layers, "
              f"{args.qformer_heads} heads, {args.n_vuln} queries")
    # Embeddings stay in the MAIN optimizer at the base LR. ONLY the two new
    # bottleneck rows additionally get a high-LR Adam update (own state), so
    # the pretrained embedding matrix is not disturbed (a whole-matrix 10x LR
    # trashed the pretrained rows and made the model output gibberish).
    # Dedup params: tied lm_head.weight IS embed_tokens.weight (one object).
    if args.freeze_layers > 0:
        n_frozen = 0
        for _name, _p in model.named_parameters():
            if _name.startswith("model.layers."):
                try:
                    _idx = int(_name.split(".")[2])
                except (IndexError, ValueError):
                    continue
                if _idx < args.freeze_layers:
                    _p.requires_grad_(False)
                    n_frozen += 1
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[standalone] freeze_layers={args.freeze_layers}: "
              f"froze {n_frozen} tensors; trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params")
    params = list({id(p): p for p in model.parameters() if p.requires_grad}.values())
    if args.lora_rank > 0:
        _before = len(params)
        params = [p for p in params if p is not emb_w]
        print(f"[standalone] main optimizer excludes embed matrix "
              f"({_before} -> {len(params)} param tensors); embed rows update = row-only Adam")
    params = params + [p for p in head.parameters() if p.requires_grad]
    if qformer is not None:
        params = params + list(qformer.parameters())
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01,
                                        betas=(0.9, 0.95))
        print("[standalone] optimizer = AdamW8bit")
    elif args.optimizer == "paged_adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=0.01,
                                             betas=(0.9, 0.95))
        print("[standalone] optimizer = PagedAdamW8bit")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01,
                                      betas=(0.9, 0.95))
        print("[standalone] optimizer = AdamW")
    emb_extra_state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    print(f"[standalone] emb_lr_mult={args.emb_lr_mult} (rows-only extra Adam)")

    cfg = {
        "n_vuln": args.n_vuln, "max_length": args.max_length, "truncation": "right",
        "vulcode_loss_weight": args.vulcode_loss_weight, "cls_neutral_weight": args.neutral_weight,
        "label_mode": args.label_mode, "skip_modes": "comment+string",
        "label_align": args.label_align, "overlap_ratio": args.overlap_ratio,
        "vulcode_see_input": args.vulcode_see_input,
        "seccode_see_vulcode": args.seccode_see_vulcode,
        "multi_vuln": args.multi_vuln,
        "interleave": args.interleave, "keep_up": args.keep_up,
        "judge": args.judge,
        "vuln_anal_vis": bool(args.vuln_anal_vis),
        "seccode_cond": args.seccode_cond,
        "think_mode": args.think_mode,
    }
    train_ds = RawSFTDataset(args.train_data, args.max_train, args.seed,
                             extra_path=args.extra_data)
    val_ds = RawSFTDataset(args.val_data, args.max_val, args.seed)
    collate = functools.partial(collate_fn, tokenizer=tokenizer, cfg=cfg)
    loader = DataLoader(train_ds, batch_size=args.train_batch_size,
                        shuffle=not args.sequential,
                        num_workers=4, collate_fn=collate, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))
    total_steps = len(loader) * args.epochs
    warmup = int(args.warmup_frac * total_steps)

    def _step_table(spec, default):
        """'4202:5e-6,5452:3e-6' → 每步取值数组（下标 = step）。缺省恒为 default。"""
        arr = [default] * (total_steps + 1)
        if spec:
            for tok in spec.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                st, val = tok.split(":")
                st, val = int(st), float(val)
                if not (1 <= st <= total_steps):
                    raise SystemExit(f"[curriculum] step {st} 越界 (1..{total_steps})")
                for s in range(st, total_steps + 1):
                    arr[s] = val
        return arr

    lr_arr = _step_table(args.lr_map, args.lr)
    sec_arr = _step_table(args.sec_up_map, args.sec_up)
    if args.sequential:
        print(f"[curriculum] sequential 行序=课程 steps={total_steps} "
              f"lr_map={args.lr_map} sec_up_map={args.sec_up_map}", flush=True)

    loss_fct = nn.CrossEntropyLoss(reduction="none")

    def _cls_ce(logits, targets):
        """Security-head CE with optional label smoothing + class up-weight."""
        if args.cls_label_smooth > 0.0:
            n_cls = logits.shape[-1]
            logp = torch.log_softmax(logits, dim=-1)
            tgt = targets.clamp(min=0)  # CLS_IGNORE(-100) -> 0, masked below
            smooth = args.cls_label_smooth / (n_cls - 1)
            ce = -(1.0 - args.cls_label_smooth) * logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) \
                 - smooth * logp.sum(-1)
            ce = ce * (targets != CLS_IGNORE).float()
        else:
            ce = loss_fct(logits, targets)
        if args.cls_pos_weight != 1.0:
            posw = torch.where((targets == CLS_SAFE) | (targets == CLS_UNSAFE),
                               args.cls_pos_weight, 1.0)
            ce = ce * posw
        return ce

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[standalone] total_steps={total_steps} trainable={sum(p.numel() for p in params)}")
    step = 0
    log_rows = []
    csv_path = out_dir / "train_log.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step,loss,vul,sec,cls,cls_acc,ul,time\n")
    for epoch in range(args.epochs):
        for batch in loader:
            step += 1
            if lr_arr[step] != lr_arr[step - 1] or sec_arr[step] != sec_arr[step - 1]:
                print(f"[curriculum] step {step}: lr={lr_arr[step]:g} "
                      f"sec_up={sec_arr[step]:g}", flush=True)
            t0 = time.time()
            optimizer.zero_grad()
            n_micro = max(1, args.train_batch_size // args.micro_batch_size)
            acc_loss = acc_vul = acc_sec = acc_cls = acc_ul = 0.0
            acc_cls_acc = 0.0
            n_mb_done = 0
            for mb in range(n_micro):
                sl = slice(mb * args.micro_batch_size, (mb + 1) * args.micro_batch_size)
                ids = batch["input_ids"][sl].to(device)
                if ids.shape[0] == 0:
                    continue
                n_mb_done += 1
                attn = batch["attention_mask"][sl].to(device)
                pos = batch["position_ids"][sl].to(device)
                lm = batch["loss_mask"][sl][:, 1:].contiguous().reshape(-1).to(device)
                cl = batch["cls_labels"][sl].to(device)
                cw = batch["cls_weights"][sl].to(device)
                il = batch["inp_lens"][sl].to(device)
                labels = ids[:, 1:].contiguous()

                out = _forward_cotrain(model, qformer, ids, attn, pos, il,
                                       args.n_vuln, device)
                logits = out.logits[..., :-1, :].contiguous()
                loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
                loss = loss * lm
                shift_cls = cl[:, 1:].contiguous().view(-1)
                if sec_arr[step] != 1.0:
                    boost = torch.where(shift_cls == CLS_SAFE, sec_arr[step], 1.0).to(loss.device)
                    loss = loss * boost
                if args.vul_fix_w != 1.0:
                    # token 级安全标签（vul_fix_w）：vulcode 段 UNSAFE（漏洞）
                    # token 的 NLL 加权——压缩瓶颈优先保真漏洞细节。
                    vul_w = torch.where(
                        (lm > 0) & (lm <= 0.5) & (shift_cls == CLS_UNSAFE),
                        args.vul_fix_w, 1.0).to(loss.device)
                    loss = loss * vul_w
                ul_loss = torch.zeros((), device=loss.device)
                if args.vul_ul > 0:
                    probs = torch.softmax(logits.view(-1, model.config.vocab_size), dim=-1)
                    target_p = probs.gather(-1, labels.view(-1).unsqueeze(-1)).squeeze(-1)
                    ul = -torch.log(1.0 - target_p.clamp(max=1.0 - 1e-4) + 1e-8)
                    ul_mask = ((shift_cls == CLS_UNSAFE) & (lm > 0) & (lm <= 0.5)).float()
                    n_ul = ul_mask.sum()
                    if n_ul > 0:
                        ul_loss = (ul * ul_mask).sum() / n_ul
                valid = lm.sum()
                gen_loss = loss.sum() / valid.clamp(min=1.0)

                if args.head_n_layers > 1:
                    hs_agg = torch.stack(out.hidden_states[-args.head_n_layers:], dim=0).mean(dim=0)
                else:
                    hs_agg = out.hidden_states[-1]
                cls_logits = head(hs_agg)
                cls_ce = _cls_ce(cls_logits.reshape(-1, 3), cl.reshape(-1))
                if args.cls_focal_gamma > 0:
                    cls_probs = torch.softmax(cls_logits.reshape(-1, 3), dim=-1)
                    cls_idx = cl.reshape(-1).clamp(min=0)  # CLS_IGNORE(-100) -> 0, later zeroed by cw
                    cls_pt = cls_probs.gather(-1, cls_idx.unsqueeze(-1)).squeeze(-1)
                    cls_focal_w = (1.0 - cls_pt).pow(args.cls_focal_gamma)
                    cls_focal_w = cls_focal_w * (cl.reshape(-1) != CLS_IGNORE).float()
                    cls_ce = cls_ce * cls_focal_w
                cls_ce = cls_ce * cw.reshape(-1).to(cls_ce.device)
                cls_loss = cls_ce.sum() / cw.sum().clamp(min=1.0)
                cls_acc = ((cls_logits.argmax(-1) == cl) & (cw > 0)).float().sum() / cw.gt(0).sum().clamp(min=1.0)

                icae_loss = torch.zeros((), device=loss.device)
                if args.icae_weight > 0:
                    # ICAE 重建（仅默认布局；interleave 的 vuln 段分散不适用）：
                    # <vuln> 段 hidden 均值回归 vulcode 段 hidden 均值（目标 detach）。
                    Bs = hs_agg.shape[0]
                    vh = torch.stack(
                        [hs_agg[b, il[b]:il[b] + args.n_vuln].mean(0) for b in range(Bs)])
                    vc_m = batch["loss_mask"][sl].to(device) == 0.5  # vulcode 段
                    vc = torch.stack([hs_agg[b, vc_m[b]].mean(0) for b in range(Bs)])
                    icae_loss = (vh - vc.detach()).pow(2).mean()
                if args.sec_mod > 0 or args.sec_rwr > 0:
                    # 安全头加权（stop-grad，RWR 风格）：预测目标 token k 时，若安全头
                    # 在位置 k 输出 p_safe > p_unsafe，则上调该位置 NLL（模型越自信是
                    # 安全修复，越要求严格对齐目标）。只作用于 seccode 段（lm > 0.5）。
                    # 头冻结（--sec_head）时即为离线 PPO：纯外部 reward，无自举。
                    cls_shift = cls_logits[:, :-1].reshape(-1, 3)  # target-aligned
                    cls_pt = torch.softmax(cls_shift, dim=-1)[:, [CLS_SAFE, CLS_UNSAFE]]
                    delta = torch.clamp(
                        cls_pt[:, 0] - cls_pt[:, 1], min=0.0).detach()
                    if args.sec_mod > 0:
                        sm_w = 1.0 + args.sec_mod * delta          # 线性（secmod 形式）
                    else:
                        sm_w = torch.exp(delta / args.sec_rwr)     # RWR 指数 exp(Δ/β)
                    loss = loss * torch.where(lm > 0.5, sm_w, 1.0)
                    if step % 200 == 0 and mb == 0:
                        # 诊断：冻结头置信 Δ 与显式 token 标签一致性（seccode 段）
                        sec_m = lm > 0.5
                        d_all = delta[sec_m]
                        c_all = shift_cls.view(-1)[sec_m]
                        if d_all.numel() > 0:
                            for lab, nm in ((CLS_SAFE, "SAFE"),
                                            (CLS_UNSAFE, "UNSAFE"),
                                            (CLS_NEUTRAL, "NEUTRAL")):
                                sub = d_all[c_all == lab]
                                if sub.numel():
                                    print(f"  [诊断] step {step}: {nm} "
                                          f"Δmean={sub.mean().item():.3f} "
                                          f"n={sub.numel()}", flush=True)

                total = gen_loss + args.cls_weight * cls_loss + args.vul_ul * ul_loss \
                    + args.icae_weight * icae_loss
                total.backward()

                vul_mask = (lm > 0) & (lm <= 0.5)
                sec_mask = lm > 0.5
                acc_loss += total.item()
                acc_vul += (loss * vul_mask).sum().item() / vul_mask.sum().clamp(min=1).item() if vul_mask.sum() > 0 else 0
                acc_sec += (loss * sec_mask).sum().item() / sec_mask.sum().clamp(min=1).item() if sec_mask.sum() > 0 else 0
                acc_cls += cls_loss.item()
                acc_ul += ul_loss.item()
                acc_cls_acc += cls_acc.item()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            lr_now = lr_arr[step]
            lr_scale = step / max(1, warmup) if step <= warmup else 1.0
            if lr_scale < 1.0 or args.lr_map:
                for g in optimizer.param_groups:
                    g["lr"] = lr_now * lr_scale
            optimizer.step()
            if args.emb_lr_mult > 1.0:
                with torch.no_grad():
                    grad = emb_w.grad
                    if grad is not None:
                        lr_emb = lr_now * (args.emb_lr_mult - 1.0) * lr_scale
                        beta1, beta2, eps = 0.9, 0.95, 1e-8
                        for row_id in v_ids + [s_id]:
                            g = grad[row_id].float()
                            if row_id not in emb_extra_state:
                                emb_extra_state[row_id] = (
                                    torch.zeros_like(g), torch.zeros_like(g))
                            exp_avg, exp_avg_sq = emb_extra_state[row_id]
                            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                            bc1 = 1.0 - beta1 ** step
                            bc2 = 1.0 - beta2 ** step
                            step_size = lr_emb * (exp_avg / bc1) / (
                                exp_avg_sq.sqrt() / math.sqrt(bc2) + eps)
                            emb_w[row_id] -= step_size

            acc_loss /= max(1, n_mb_done); acc_vul /= max(1, n_mb_done); acc_sec /= max(1, n_mb_done)
            acc_cls /= max(1, n_mb_done); acc_ul /= max(1, n_mb_done); acc_cls_acc /= max(1, n_mb_done)
            log_rows.append((step, acc_loss, acc_vul, acc_sec, acc_cls, acc_ul, acc_cls_acc))
            with csv_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{acc_loss:.5f},{acc_vul:.5f},{acc_sec:.5f},"
                        f"{acc_cls:.5f},{acc_cls_acc:.5f},{acc_ul:.5f},{time.time()-t0:.2f}\n")
            if step % 5 == 0 or step == 1:
                print(f"step {step}/{total_steps} loss={acc_loss:.3f} vul={acc_vul:.3f} "
                      f"sec={acc_sec:.3f} cls={acc_cls:.3f} cls_acc={acc_cls_acc:.3f} "
                      f"ul={acc_ul:.3f} time={time.time()-t0:.1f}s", flush=True)

        if args.export_epoch_ckpts and args.lora_rank == 0:
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            _ep_no = args.epoch_offset + epoch + 1
            ep_dir = out_dir / f"merged_hf_model_ep{_ep_no}"
            print(f"[standalone] exporting epoch {_ep_no} ckpt -> {ep_dir}", flush=True)
            model.save_pretrained(ep_dir)
            tokenizer.save_pretrained(ep_dir)
            torch.save(head.state_dict(), out_dir / f"security_head_ep{_ep_no}.pt")
            model.train()
            head.train()
            if qformer is not None:
                qformer.train()

    # ---- final validation pass ----
    # 2026-09-02：14B 全参跑满 5045 步，却在 final validation 前向上 OOM
    # (78.88G/79.15G used, 245M free)：训练结束后 optimizer 状态与梯度缓冲
    # 仍驻留显存，val 与 save 都拿不到内存，权重全丢。训练已结束，先显式释放。
    try:
        del optimizer
    except NameError:
        pass
    try:
        model.zero_grad(set_to_none=True)
    except Exception:
        pass
    # 2026-09-01：think4(2048) 训练峰值后 22.6G 碎片堵住 reserved 池，
    # val/save 阶段 cross_entropy OOM（3750 步白跑）。先释放缓存再进 val。
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("[standalone] running final validation...")
    model.eval()
    head.eval()
    if qformer is not None:
        qformer.eval()
    val_loss = val_cls_loss = val_cls_acc = 0.0
    n_val = 0
    # 2026-09-02：显存兜底——若释放后仍不宽裕，直接跳过 val，保证权重能存下来。
    _val_bs = 8 if torch.cuda.mem_get_info()[0] > int(20 * 2**30) else 0
    if _val_bs == 0:
        print("[standalone] WARN: 显存不足, 跳过 final validation 直接保存", flush=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, _val_bs), shuffle=False,
                            num_workers=2, collate_fn=collate) if _val_bs else []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pos = batch["position_ids"].to(device)
            lm = batch["loss_mask"][:, 1:].contiguous().reshape(-1).to(device)
            cl = batch["cls_labels"].to(device)
            cw = batch["cls_weights"].to(device)
            il = batch["inp_lens"].to(device)
            labels = ids[:, 1:].contiguous()
            out = _forward_cotrain(model, qformer, ids, attn, pos, il,
                                   args.n_vuln, device)
            logits = out.logits[..., :-1, :].contiguous()
            loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
            loss = loss * lm
            valid = lm.sum()
            val_loss += (loss.sum() / valid.clamp(min=1.0)).item()
            if args.head_n_layers > 1:
                hs_agg = torch.stack(out.hidden_states[-args.head_n_layers:], dim=0).mean(dim=0)
            else:
                hs_agg = out.hidden_states[-1]
            cls_logits = head(hs_agg)
            cls_ce = _cls_ce(cls_logits.reshape(-1, 3), cl.reshape(-1))
            cls_ce = cls_ce * cw.reshape(-1).to(cls_ce.device)
            val_cls_loss += (cls_ce.sum() / cw.sum().clamp(min=1.0)).item()
            acc = ((cls_logits.argmax(-1) == cl) & (cw > 0)).float().sum() / cw.gt(0).sum().clamp(min=1.0)
            val_cls_acc += acc.item()
            n_val += 1
    if n_val:
        val_metrics = {
            "val/loss": val_loss / n_val,
            "val/cls_loss": val_cls_loss / n_val,
            "val/cls_acc": val_cls_acc / n_val,
            "emb_lr_mult": args.emb_lr_mult,
            "emb_init": args.emb_init,
        }
        print(f"[standalone] final val: loss={val_metrics['val/loss']:.4f} "
              f"cls_loss={val_metrics['val/cls_loss']:.4f} cls_acc={val_metrics['val/cls_acc']:.4f}")
        (out_dir / "val_metrics.json").write_text(
            json.dumps(val_metrics, indent=2), encoding="utf-8")
    model.train()
    head.train()
    if qformer is not None:
        qformer.train()

    # ---- save ----
    # 2026-09-01：val 后同样先释放碎片，防 save_pretrained 序列化 OOM
    gc.collect()
    torch.cuda.empty_cache()
    print("[standalone] saving...")
    if args.lora_rank > 0:
        merged = model.merge_and_unload()
    else:
        merged = model
    merged.save_pretrained(out_dir / "merged_hf_model")
    tokenizer.save_pretrained(out_dir / "merged_hf_model")
    torch.save(head.state_dict(), out_dir / "security_head.pt")
    if qformer is not None:
        torch.save(qformer.state_dict(), out_dir / "qformer.pt")
        (out_dir / "qformer_config.json").write_text(json.dumps({
            "hidden_size": hidden, "n_queries": args.n_vuln,
            "n_layers": args.qformer_layers, "n_heads": args.qformer_heads,
            "dropout": args.qformer_dropout,
        }), encoding="utf-8")
    gs = out_dir / "global_step_final"
    gs.mkdir(exist_ok=True)
    torch.save(head.state_dict(), gs / "security_head.pt")
    if qformer is not None:
        torch.save(qformer.state_dict(), gs / "qformer.pt")
        (gs / "qformer_config.json").write_text(json.dumps({
            "hidden_size": hidden, "n_queries": args.n_vuln,
            "n_layers": args.qformer_layers, "n_heads": args.qformer_heads,
            "dropout": args.qformer_dropout,
        }), encoding="utf-8")
    print(f"[standalone] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
