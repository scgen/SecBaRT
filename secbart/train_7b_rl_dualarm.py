"""#64 (09-10) RL 双臂生成: <vuln> 段生成 vulcode 臂, 与 seccode 反向奖励。

fork train_7b_rl_docker_func.py (func_safety 单臂) — 布局与掩码语义的源码依据:
bottleneck_token_common.py _process_sample: SFT 序列 = input | <vuln>*N | vulcode |
<secu> | seccode | eos; 掩码 = vulcode 行禁看 input (<vuln> 态被迫携带漏洞语义 =
压缩监督), seccode 行禁看 vulcode 列; loss = vulcode aux 0.5 + seccode 主 1.0。
→ RL 现状只生成 seccode 槽, vulcode 槽 (漏洞侧生成段) 无监督 → <vuln> 语义塌缩。
#64 = 双臂对偶采样, 双槽都生成都接同一套真实双测试:

  sec 臂 (现状不变): prompt+<vuln>*N+<secu> → gen seccode  → 双测试 func✓sec✓ 正奖
  vul 臂 (新):       prompt+<vuln>*N      → gen vulcode  → 双测试 func✓sec✗ 正奖
                     (生成时因果只见 <vuln> 态, 与 SFT 压缩监督同构; 若自吐 <secu>
                      token 则截断 — SFT 从未教模型输出 <secu>, 仅防御)

奖励: 与预期一致才正更新 (硬门): r_sec = func_s·sec_s; r_vul = func_v·(1−sec_v);
安全轴反向 = vul 臂 gate 用 (1−sec)。可选塑形: --token_head 时 sec 臂复刻三合一
func_safety (r=func·sec·(1+w·score)); vul 臂 --vul_shaping w 用反号 score
(score=p_safe−p_unsafe, vul 臂要 d<0 → r_vul=func·(1−sec)·(1+w·max(0,−d)) 段级)。
GRPO 组内 advantage 按 (臂, 任务) 各自归一。空生成 −1。

warm-start (spec 3): --warm_pool vulpool jsonl {id,prompt,vul_code} (探针产物,
cap✓sec✗ 双测试认证) → RL 前 SFT 克隆 epoch (seq=prompt+<vuln>*N+vul_code+eos,
只暖 vul 臂槽), 后放开自产。

数据: 默认 --dataset secodeplt_filtered (filtered-test_cases.json 400 = RL update
池同源, RL 禁 CWEval 评测集)。种子/全量规则: --lora_rank 0 fullft; 判据 func_sec@1
> 55.56 (func_safety_s768) 才上三合一比较; 副指标 = vul 臂检出率 >0 + 两臂功能双过率。
用法: python -m secbart.train_7b_rl_dualarm --seed_model <merge|s768>
    --output_dir DIR [--warm_pool vulpool_base_k8/vul_pool.jsonl --warm_epochs 2] ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from secbart.bottleneck_token_common import SECU_TOK
from secbart.train_7b_rl_docker_func import (
    LANG_NAME, MAX_NEW, SECPLT_CASES, SecCodePLTDataset, _btoks_ids,
    docker_secplt_eval,
)
from secbart.utils import extract_code

FILT_CASES = os.path.join(os.path.dirname(SECPLT_CASES), "filtered-test_cases.json")
EVAL_ROOT = "/tmp/rl_dual_eval"
MAX_LEN = 1024


class ScalarTokenHead(nn.Module):
    """Frozen scalar security head used by the P2T-style GRPO arm."""

    def __init__(self, hidden_size, n_layers=1, activation="sigmoid"):
        super().__init__()
        self.n_layers = max(1, int(n_layers))
        self.activation = activation
        if self.n_layers == 1:
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 256), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(256, 1),
            )
        else:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_size) for _ in range(self.n_layers)])
            self.layer_weights = nn.Parameter(torch.ones(self.n_layers))
            self.net = nn.Sequential(
                nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 1),
            )

    def encode(self, hidden_states):
        if self.n_layers == 1:
            return hidden_states[-1]
        selected = hidden_states[-self.n_layers:]
        weights = torch.softmax(self.layer_weights, dim=0)
        return sum(
            weight.to(hidden.dtype) * norm(hidden)
            for weight, norm, hidden in zip(weights, self.layer_norms, selected)
        )

    def forward(self, hidden_states):
        score = self.net(self.encode(hidden_states)).squeeze(-1)
        return torch.sigmoid(score) if self.activation == "sigmoid" else score


class VulPoolDataset(Dataset):
    """warm-start 克隆池 (vulpool 产物: {id, prompt, vul_code} 双测试认证 cap✓sec✗)。"""

    def __init__(self, path, max_rows=-1, seed=42):
        rows = [json.loads(l) for l in open(path)]
        if max_rows > 0:
            import random
            rows = random.Random(seed).sample(rows, max_rows)
        print(f"[vulpool] warm-start 克隆行 {len(rows)} (vul 臂槽)")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="每臂每任务采样条数 (总 rollout = 2×B×k)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_beta", type=float, default=0.05)
    ap.add_argument("--lora_rank", type=int, default=0, help="RL 种子全量版规则: 0=fullft")
    ap.add_argument("--dataset", default="secodeplt_filtered",
                    choices=["secodeplt", "secodeplt_filtered"],
                    help="训练任务池 (RL 禁 CWEval 评测集)")
    ap.add_argument("--max_tasks", type=int, default=-1)
    ap.add_argument("--n_vuln", type=int, default=4)
    ap.add_argument("--vulcode_see_input", action="store_true",
                    help="vul 臂 vulcode 行可看 input (SFT 同款消融开关); "
                         "默认 False = 与 SFT 种子一致, 屏蔽 input 只走 <vuln> 压缩槽")
    ap.add_argument("--max_length", type=int, default=MAX_LEN)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--token_head", default=None, help="sec 臂三合一塑形头 (v4, 可选)")
    ap.add_argument("--p2t_token", action="store_true",
                    help="P2T-style token advantage: redistribute signed outcome reward "
                         "with a frozen scalar security head")
    ap.add_argument("--p2t_alpha", type=float, default=0.1,
                    help="weight of the P2T token reward in token advantage")
    ap.add_argument("--p2t_omega", type=float, default=0.6,
                    help="weight of head-derived token weights; [0,1]")
    ap.add_argument("--p2t_temperature", type=float, default=1.0,
                    help="softmax temperature multiplier for signed head scores")
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="save a rolling resumable checkpoint every N steps; 0=disabled")
    ap.add_argument("--resume_from", default=None,
                    help="load policy weights from a rolling checkpoint directory")
    ap.add_argument("--resume_step", type=int, default=-1,
                    help="step represented by --resume_from; -1 reads checkpoint_state.json")
    ap.add_argument("--paged_optimizer", action="store_true",
                    help="use PagedAdamW8bit; default AdamW8bit avoids host UVM paging")
    ap.add_argument("--ref_chunk_size", type=int, default=4,
                    help="reference-model batch size for chunked log-prob computation")
    ap.add_argument("--no_save", action="store_true",
                    help="调试运行结束时不导出完整模型")
    ap.add_argument("--tkh_agg", default="mean", choices=["mean", "min"])
    ap.add_argument("--w_head", type=float, default=0.5)
    ap.add_argument("--vul_shaping", type=float, default=0.0,
                    help="vul 臂塑形权重: r_vul=func·(1−sec)·(1+w·max(0,−score)); 0=纯反向硬门")
    ap.add_argument("--warm_pool", default=None, help="vulpool jsonl → RL 前克隆暖 vul 槽")
    ap.add_argument("--warm_epochs", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.seed_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    secu_id = tok.convert_tokens_to_ids(SECU_TOK)
    assert secu_id is not None, "seed 模型词表需含 <secu> (btok SFT 扩展词表)"

    ref = AutoModelForCausalLM.from_pretrained(
        args.seed_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    policy_path = args.resume_from or args.seed_model
    policy = AutoModelForCausalLM.from_pretrained(
        policy_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    if args.lora_rank > 0:
        raise SystemExit("双臂 RL 用全量版规则: --lora_rank 0 (fullft)")
    policy.gradient_checkpointing_enable()
    policy.train()
    import bitsandbytes as bnb
    optimizer_cls = bnb.optim.PagedAdamW8bit if args.paged_optimizer else bnb.optim.AdamW8bit
    opt = optimizer_cls([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    print(f"[optimizer] {optimizer_cls.__name__}", flush=True)
    if args.resume_from:
        optimizer_path = Path(args.resume_from) / "optimizer.pt"
        if optimizer_path.exists():
            opt.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            print(f"[resume] optimizer={optimizer_path}", flush=True)
        else:
            print(f"[resume] optimizer checkpoint missing: {optimizer_path}; "
                  "continuing with fresh optimizer state", flush=True)

    token_head = None
    if args.token_head:
        sd = torch.load(args.token_head, map_location=device)
        head_cfg = {}
        cfg_path = Path(args.token_head).parent / "cotrain_config.json"
        if cfg_path.exists():
            try:
                head_cfg = json.loads(cfg_path.read_text())
            except Exception:
                head_cfg = {}
        if args.p2t_token:
            out_dim = int(sd.get("net.4.weight", torch.empty(1, 1)).shape[0])
            if out_dim != 1:
                raise SystemExit("--p2t_token 需要单值 scalar security head (out_dim=1)")
            token_head = ScalarTokenHead(
                policy.config.hidden_size,
                n_layers=int(head_cfg.get("head_n_layers", 1)),
                activation=head_cfg.get("head_activation", "sigmoid"),
            )
        elif any(k.startswith("net.0") for k in sd):
            net = nn.Sequential(nn.LayerNorm(policy.config.hidden_size),
                                nn.Linear(policy.config.hidden_size, 256), nn.GELU(),
                                nn.Dropout(0.1), nn.Linear(256, 2))
            token_head = nn.Module()
            token_head.net = net
        else:
            net = nn.Linear(policy.config.hidden_size, 2)
            token_head = nn.Module()
            token_head.net = net
        token_head.load_state_dict(sd)
        token_head = token_head.to(torch.bfloat16).to(device).eval()
        for p in token_head.parameters():
            p.requires_grad = False
        if args.p2t_token:
            if not 0.0 <= args.p2t_omega <= 1.0:
                raise SystemExit("--p2t_omega 必须在 [0,1]")
            print(f"[reward] P2T-style scalar TKH {args.token_head} "
                  f"alpha={args.p2t_alpha} omega={args.p2t_omega} "
                  f"temperature={args.p2t_temperature} | 双臂反向", flush=True)
        else:
            print(f"[reward] sec 臂三合一 tkh {args.token_head} agg={args.tkh_agg} "
                  f"w={args.w_head} | vul 臂塑形 w={args.vul_shaping}", flush=True)
    else:
        print("[reward] 纯硬门: r_sec=func·sec, r_vul=func·(1−sec)", flush=True)

    cases_path = SECPLT_CASES if args.dataset == "secodeplt" else FILT_CASES
    ds = SecCodePLTDataset(cases_path, args.max_tasks, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "rl_log.csv"
    resumed_step = args.resume_step
    if resumed_step < 0 and args.resume_from:
        state_path = Path(args.resume_from) / "checkpoint_state.json"
        if state_path.exists():
            resumed_step = int(json.loads(state_path.read_text()).get("step", 0))
    if resumed_step < 0:
        resumed_step = 0
    log_f = log_path.open("a" if resumed_step else "w")
    if resumed_step == 0:
        log_f.write("step,r_sec,r_vul,func_s,sec_s,func_v,sec_v,kl,adv_std,empty,p2t_sum_err,p2t_abs_mean\n")
    if resumed_step:
        print(f"[resume] policy={policy_path} from step {resumed_step}", flush=True)

    # ---- warm-start: 克隆 vul 臂槽 (池 = 双测试认证 cap✓sec✗) ----
    if args.warm_pool:
        warm_ds = VulPoolDataset(args.warm_pool)
        w_loader = DataLoader(warm_ds, batch_size=8, shuffle=True, collate_fn=lambda x: x)
        for ep in range(args.warm_epochs):
            tot, acc = 0, 0.0
            for rows in w_loader:
                seqs, starts = [], []
                for r in rows:
                    pre = _btoks_ids(r["prompt"], args.n_vuln, tok, args.max_length)
                    cids = tok(r["vul_code"] or "", add_special_tokens=False,
                               truncation=True,
                               max_length=args.max_length - len(pre))["input_ids"]
                    seqs.append(pre + cids)
                    starts.append(len(pre))
                ml = max(len(x) for x in seqs)
                pad = torch.full((len(seqs), ml), tok.pad_token_id,
                                 dtype=torch.long, device=device)
                for i, s in enumerate(seqs):
                    pad[i, :len(s)] = torch.tensor(s, device=device)
                am = pad != tok.pad_token_id
                out = policy(pad, attention_mask=am)
                lg = torch.log_softmax(out.logits[:, :-1], dim=-1)
                tgt = pad[:, 1:]
                lm = (tgt != tok.pad_token_id).float()
                logp = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * lm
                # 只对 gen 段 (pre 之后) 反传; pre 内 <vuln> 已入词表无需模仿
                gen_mask = torch.zeros_like(lm)
                for i, st in enumerate(starts):
                    gen_mask[i, st - 1:] = 1.0
                nll = -(logp * gen_mask).sum(1) / gen_mask.sum(1).clamp(min=1)
                loss = nll.mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
                acc += loss.item()
                tot += 1
                del out, lg
            print(f"[warm] epoch {ep+1}/{args.warm_epochs} clone-loss={acc/max(tot,1):.4f}",
                  flush=True)

    # ---- 双臂 GRPO ----
    step = resumed_step
    while step < args.steps:
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            collate_fn=lambda x: x)
        for metas in loader:
            if step >= args.steps:
                break
            step += 1
            B = len(metas)
            # ---- rollout: 每任务每臂 k 条 (sec 臂现状布局 / vul 臂无 <secu> 前缀) ----
            policy.eval()
            raw_sec = []          # (B, k)
            raw_vul = []          # (B, k)
            lens_sec, lens_vul = [], []
            with torch.no_grad():
                for m in metas:
                    s_pre = _btoks_ids(m["prompt"], args.n_vuln, tok, args.max_length)
                    v_pre = (tok(m["prompt"], add_special_tokens=False, truncation=True,
                                 max_length=args.max_length)["input_ids"]
                             + [tok.convert_tokens_to_ids("<vuln>")] * args.n_vuln)[:args.max_length]
                    s_t = torch.tensor([s_pre], device=device)
                    v_t = torch.tensor([v_pre], device=device)
                    outs = policy.generate(
                        s_t, do_sample=True, temperature=args.temp, top_p=0.95,
                        max_new_tokens=MAX_NEW, num_return_sequences=args.k,
                        pad_token_id=tok.pad_token_id, use_cache=True)
                    cs = []
                    for o in outs:
                        g = o.tolist()
                        lens_sec.append(len(g) - len(s_pre))
                        text = tok.decode(g[len(s_pre):], skip_special_tokens=True)
                        code = extract_code(text, lang=LANG_NAME.get(m["lang"], "Python"))
                        cs.append(code if code and code.strip() else "")
                    raw_sec.append(cs)
                    outs_v = policy.generate(
                        v_t, do_sample=True, temperature=args.temp, top_p=0.95,
                        max_new_tokens=MAX_NEW, num_return_sequences=args.k,
                        pad_token_id=tok.pad_token_id, use_cache=True)
                    cv = []
                    for o in outs_v:
                        g = o.tolist()
                        if secu_id in g[len(v_pre):]:
                            cut = g.index(secu_id)           # 自吐 <secu> → 截断 (防御)
                            g = g[:cut]
                        lens_vul.append(len(g) - len(v_pre))
                        text = tok.decode(g[len(v_pre):], skip_special_tokens=True)
                        code = extract_code(text, lang=LANG_NAME.get(m["lang"], "Python"))
                        cv.append(code if code and code.strip() else "")
                    raw_vul.append(cv)
            policy.train()

            # ---- 写双槽 raw + 宿主双跑 (generated_{i}=sec 槽, generated_v{i}=vul 槽) ----
            eval_dir = Path(EVAL_ROOT) / f"eval_rl_{step}"
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
            for i in range(args.k):
                for b, m in enumerate(metas):
                    p = eval_dir / f"generated_{i}" / m["raw_rel"]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(raw_sec[b][i])
                    pv = eval_dir / f"generated_v{i}" / m["raw_rel"]
                    pv.parent.mkdir(parents=True, exist_ok=True)
                    pv.write_text(raw_vul[b][i])
            print(f"[step {step}] 双槽判定 {B}x{args.k}×2 (sec+{chr(118)}ul) ...", flush=True)
            func_map, sec_map = docker_secplt_eval(str(eval_dir), step, metas)
            shutil.rmtree(eval_dir, ignore_errors=True)
            fs_t = torch.zeros(B * args.k, device=device)
            ss_t = torch.zeros(B * args.k, device=device)
            fv_t = torch.zeros(B * args.k, device=device)
            sv_t = torch.zeros(B * args.k, device=device)
            miss = 0
            for b, m in enumerate(metas):
                rel_key = os.path.splitext(m["raw_rel"])[0]  # secodeplt: {id}.py → {id}
                for i in range(args.k):
                    ks = f"generated_{i}/{rel_key}"
                    kv = f"generated_v{i}/{rel_key}"
                    if ks in func_map:
                        fs_t[b * args.k + i] = 1.0 if func_map[ks] else 0.0
                        ss_t[b * args.k + i] = 1.0 if sec_map.get(ks) else 0.0
                    else:
                        miss += 1
                    if kv in func_map:
                        fv_t[b * args.k + i] = 1.0 if func_map[kv] else 0.0
                        sv_t[b * args.k + i] = 1.0 if sec_map.get(kv) else 0.0
                    else:
                        miss += 1
            if miss:
                print(f"[step {step}] WARN {miss} 条无判定 (空/编译失败)", flush=True)

            # ---- 完整序列 forward: logp, arm-major 行序 (R 行 sec 槽 + R 行 vul 槽),
            #     与 r_all = cat([r_sec, r_vul]) 对齐; GRPO 组 = view(2B, k) 每行 (臂,任务) ----
            seqs, starts, vul_blk = [], [], []
            for b, m in enumerate(metas):
                s_pre = _btoks_ids(m["prompt"], args.n_vuln, tok, args.max_length)
                for i in range(args.k):
                    cs = tok(raw_sec[b][i] or "", add_special_tokens=False,
                             truncation=True,
                             max_length=args.max_length - len(s_pre))["input_ids"]
                    seqs.append(s_pre + cs)
                    starts.append(len(s_pre))
                    # sec 臂布局 = input|<vuln>*N|<secu>|seccode, 无 vulcode 块;
                    # SFT 的 <secu>/seccode 行本就只看 input+<vuln>+<secu> ⇒ 普通因果即同义
                    vul_blk.append(None)
            for b, m in enumerate(metas):
                pids = tok(m["prompt"], add_special_tokens=False, truncation=True,
                           max_length=args.max_length)["input_ids"]
                v_pre = (pids
                         + [tok.convert_tokens_to_ids("<vuln>")] * args.n_vuln)[:args.max_length]
                for i in range(args.k):
                    cv = tok(raw_vul[b][i] or "", add_special_tokens=False,
                             truncation=True,
                             max_length=args.max_length - len(v_pre))["input_ids"]
                    seqs.append(v_pre + cv)
                    starts.append(len(v_pre))
                    # (input 末位, <vuln> 块末位): vulcode 行屏蔽落在 [0, inp_end) 的列
                    vul_blk.append((min(len(pids), args.max_length), len(v_pre)))
            max_len = max(len(x) for x in seqs)
            padded = torch.full((len(seqs), max_len), tok.pad_token_id,
                                dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                padded[i, :len(s)] = torch.tensor(s, device=device)
            seg_len = int((padded != tok.pad_token_id).sum(1).max())
            seg = padded[:, :seg_len]
            # ---- 注意力掩码: 与 SFT (_process_sample) 逐位一致 ----
            # SFT 布局 input | <vuln>*N | vulcode | <secu> | seccode, vulcode 行**看不到 input**
            # (bottleneck_token_common.py:453-455, cols >= inp_end, 只走 <vuln> 压缩槽)。
            # 位置编码两边都是自然位: SFT vulcode 段 = arange, <secu> 起回卷到 vuln_end;
            # 本脚本 sec 臂 <secu> 落在 index vuln_end、vul 臂 vulcode 落在 vuln_end+i ⇒ 同值。
            attn = torch.ones((len(seqs), 1, seg_len, seg_len),
                              dtype=torch.bool, device=device)
            attn.tril_()
            if not args.vulcode_see_input:
                for i, blk in enumerate(vul_blk):
                    if blk is None:
                        continue
                    ie, ve = blk
                    if ve > ie:
                        attn[i, 0, ve:, :ie] = False      # vulcode 行: input 列全遮
            # 2026-09-27 SC2 修复: SC2 基座 config 带 attention/embedding/residual_dropout=0.1
            # 且走 nn.functional.dropout(..., training=self.training) + 模块属性(不是 nn.Dropout
            # 模块)。policy 在 train() 态算 logp、ref 是 eval() ⇒ step-1 KL ~31(应≈0), KL 惩罚
            # 0.3*31≈9 盖过策略梯度、分数是噪声。修法: 打分前临时把这些属性置 0。
            # 不能改成 policy.eval(): HF 的 gradient checkpointing 只在 training 态生效,
            # 切了会全量激活 ⇒ OOM(实测 79GiB)。Qwen/CL 无这些属性 ⇒ 行为逐位不变。
            import contextlib as _ctx
            @_ctx.contextmanager
            def _no_dropout(mod):
                saved = []
                for m in mod.modules():
                    for a in ("attention_dropout", "residual_dropout", "embedding_dropout"):
                        v = getattr(m, a, None)
                        if isinstance(v, float) and v > 0:
                            saved.append((m, a, v)); setattr(m, a, 0.0)
                try:
                    yield
                finally:
                    for m, a, v in saved:
                        setattr(m, a, v)
            with _no_dropout(policy):
                outs = policy(seg, attention_mask=attn,
                              output_hidden_states=token_head is not None)
            outs.logits = torch.log_softmax(outs.logits, dim=-1)
            lg = outs.logits[:, :-1]
            tgt = seg[:, 1:]
            logp_tok = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            logp_tok = logp_tok * (tgt != tok.pad_token_id).float()
            st_t = torch.tensor(starts, device=device)
            gen_mask = torch.arange(seg_len - 1, device=device).unsqueeze(0) >= \
                (st_t - 1).unsqueeze(1)
            gen_mask = gen_mask & (tgt != tok.pad_token_id).bool()
            seq_logp = (logp_tok * gen_mask.float()).sum(1)
            # The reference vocabulary is large; keeping logits for all rollout
            # rows at once can exceed the remaining GPU memory even in bf16.
            # Chunking changes no objective terms and avoids a transient OOM.
            ref_logp = torch.empty_like(logp_tok)
            ref_chunk_size = max(1, int(args.ref_chunk_size))
            with torch.inference_mode():
                for ref_start in range(0, seg.size(0), ref_chunk_size):
                    ref_end = min(seg.size(0), ref_start + ref_chunk_size)
                    ref_outs = ref(seg[ref_start:ref_end],
                                   attention_mask=attn[ref_start:ref_end])
                    ref_lg = torch.log_softmax(ref_outs.logits[:, :-1], dim=-1)
                    ref_tgt = tgt[ref_start:ref_end]
                    ref_chunk_logp = ref_lg.gather(
                        -1, ref_tgt.unsqueeze(-1)).squeeze(-1)
                    ref_logp[ref_start:ref_end] = ref_chunk_logp * \
                        (ref_tgt != tok.pad_token_id).float()
                    del ref_outs, ref_lg, ref_tgt, ref_chunk_logp
            kl_pen = ((ref_logp - logp_tok).square() * gen_mask.float()).sum(1)

            # ---- 奖励 (行序 = arm-major: 前 R 行 sec, 后 R 行 vul) ----
            R = B * args.k
            r_sec = fs_t * ss_t                       # func✓sec✓
            r_vul = fv_t * (1.0 - sv_t)               # func✓sec✗ (安全轴反向)
            if token_head is not None and not args.p2t_token:
                h = outs.hidden_states[-1]
                sec_mask = torch.arange(seg_len, device=device).unsqueeze(0) >= \
                    st_t.unsqueeze(1)
                sec_mask = sec_mask & (seg != tok.pad_token_id)
                with torch.no_grad():
                    head_net = getattr(token_head, "net", token_head)
                    lt = head_net(h)
                    pt = torch.softmax(lt.float(), dim=-1)
                    d = pt[..., 0] - pt[..., 1]
                    if args.tkh_agg == "min":
                        score = d.masked_fill(~sec_mask, 1.0).min(1).values
                    else:
                        score = (d * sec_mask).sum(1) / sec_mask.sum(1).clamp(min=1.0)
                r_sec = r_sec * (1.0 + args.w_head * score[:R].clamp(-1.0, 1.0))
                if args.vul_shaping > 0:
                    # vul 臂要 d<0 (判 UNSAFE): 只奖负侧幅度, 正侧 (判 SAFE) 无 credit
                    vs = torch.clamp(-score[R:], 0.0, 1.0)
                    r_vul = r_vul * (1.0 + args.vul_shaping * vs)
            sec_empty = torch.tensor([l <= 0 for l in lens_sec], device=device)
            vul_empty = torch.tensor([l <= 0 for l in lens_vul], device=device)
            r_sec = r_sec + torch.where(sec_empty, -1.0, 0.0)
            r_vul = r_vul + torch.where(vul_empty, -1.0, 0.0)
            r_all = torch.cat([r_sec, r_vul])
            empty_t = torch.cat([sec_empty, vul_empty])

            p2t_reward = None
            p2t_sum_err = 0.0
            p2t_abs_mean = 0.0
            if args.p2t_token:
                outcome = torch.where(empty_t, -torch.ones_like(r_all),
                                      2.0 * r_all - 1.0).detach()
                valid = gen_mask
                lengths = valid.float().sum(1, keepdim=True).clamp(min=1.0)
                arm_sign = torch.cat([
                    torch.ones(R, device=device),
                    -torch.ones(R, device=device),
                ]).view(-1, 1)
                with torch.enable_grad():
                    attr_hidden = outs.hidden_states[-1].detach().requires_grad_(True)
                    p_safe_attr = token_head([attr_hidden]).float()
                    signed_attr = (2.0 * p_safe_attr - 1.0) * arm_sign
                    segment_score = (signed_attr[:, 1:] * valid.float()).sum(1) / lengths.squeeze(1)
                    grad_hidden = torch.autograd.grad(
                        segment_score.sum(), attr_hidden, retain_graph=False,
                        create_graph=False, allow_unused=False,
                    )[0]
                    attribution = (
                        grad_hidden * attr_hidden.detach()
                    ).sum(-1)[:, 1:]
                    del attr_hidden, grad_hidden, p_safe_attr, signed_attr
                # Keep the attribution normalization in fp32.  The policy/head
                # may run in bf16, but bf16 softmax loses enough mass to make
                # the redistributed reward visibly non-conservative.
                logits = (args.p2t_temperature * attribution.float()).masked_fill(~valid, -1e9)
                weights = torch.softmax(logits, dim=1) * valid.float()
                uniform = valid.float() / lengths
                p2t_reward = outcome.unsqueeze(1) * (
                    (1.0 - args.p2t_omega) * uniform + args.p2t_omega * weights)
                p2t_sum_err = float((p2t_reward.sum(1) - outcome).abs().max().item())
                p2t_abs_mean = float(p2t_reward.abs().sum().item()
                                     / valid.float().sum().clamp(min=1.0).item())
                del attribution, logits, weights, uniform

            # ---- GRPO: 组 = (臂, 任务) 的 k 条, 各自归一 ----
            grpo_reward = (outcome if args.p2t_token else r_all).view(2 * B, args.k)
            rg = grpo_reward
            mu = rg.mean(1, keepdim=True)
            sd = rg.std(1, keepdim=True)
            adv = ((rg - mu) / (sd + 1e-4)).view(-1).detach()
            if args.p2t_token:
                adv_tok = adv.unsqueeze(1) + args.p2t_alpha * p2t_reward
                loss_pg = -(adv_tok * logp_tok * gen_mask.float()).sum() / \
                    gen_mask.float().sum().clamp(min=1.0)
            else:
                loss_pg = -(adv * seq_logp).mean()
            loss = loss_pg + args.kl_beta * kl_pen.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            log_f.write(f"{step},{r_sec.mean().item():.4f},{r_vul.mean().item():.4f},"
                        f"{fs_t.mean().item():.4f},{ss_t.mean().item():.4f},"
                        f"{fv_t.mean().item():.4f},{sv_t.mean().item():.4f},"
                        f"{kl_pen.mean().item():.4f},{sd.mean().item():.4f},"
                        f"{empty_t.float().mean().item():.3f},{p2t_sum_err:.6e},"
                        f"{p2t_abs_mean:.6e}\n")
            log_f.flush()
            print(f"step {step}/{args.steps} r_sec={r_sec.mean().item():.4f} "
                  f"r_vul={r_vul.mean().item():.4f} | sec func/safe="
                  f"{fs_t.mean().item():.3f}/{ss_t.mean().item():.3f} | vul func/safe="
                  f"{fv_t.mean().item():.3f}/{sv_t.mean().item():.3f} "
                  f"empty={empty_t.sum().item()} kl={kl_pen.mean().item():.4f}", flush=True)
            del outs, lg, logp_tok, ref_logp
            torch.cuda.empty_cache()

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                ckpt = out_dir / "checkpoint_latest"
                tmp_ckpt = out_dir / "checkpoint_latest.tmp"
                if tmp_ckpt.exists():
                    shutil.rmtree(tmp_ckpt)
                tmp_ckpt.mkdir(parents=True)
                policy.save_pretrained(tmp_ckpt)
                tok.save_pretrained(tmp_ckpt)
                torch.save(opt.state_dict(), tmp_ckpt / "optimizer.pt")
                (tmp_ckpt / "checkpoint_state.json").write_text(
                    json.dumps({"step": step}, indent=2) + "\n")
                if ckpt.exists():
                    shutil.rmtree(ckpt)
                tmp_ckpt.rename(ckpt)
                print(f"[checkpoint] step {step} -> {ckpt}", flush=True)

    log_f.close()
    if not args.no_save:
        policy.save_pretrained(out_dir / "merged_hf_model")
        tok.save_pretrained(out_dir / "merged_hf_model")
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
