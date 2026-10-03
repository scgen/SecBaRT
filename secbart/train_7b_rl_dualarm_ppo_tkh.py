"""Dual-arm PPO with token-head rewards.

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

奖励: 终点任务奖励保持双臂定义 r_sec = func_s·sec_s、r_vul = func_v·(1−sec_v)。
当给出 --token_head 时，只对安全臂的生成 token 加即时奖励 w_head·z_t，其中
z_t 是同一响应内标准化后的 p_safe-p_unsafe。终点奖励与逐 token 奖励共同进入
GAE；PPO 对每个生成 token 使用自己的 advantage。空生成 −1。

warm-start (spec 3): --warm_pool vulpool jsonl {id,prompt,vul_code} (探针产物,
cap✓sec✗ 双测试认证) → RL 前 SFT 克隆 epoch (seq=prompt+<vuln>*N+vul_code+eos,
只暖 vul 臂槽), 后放开自产。

数据: 默认 --dataset secodeplt_filtered (filtered-test_cases.json 400 = RL update
池同源, RL 禁 CWEval 评测集)。种子/全量规则: --lora_rank 0 fullft; 判据 func_sec@1
> 55.56 (func_safety_s768) 才上三合一比较; 副指标 = vul 臂检出率 >0 + 两臂功能双过率。
这是实验专用 fork，不修改共享的 GRPO dualarm 入口。
"""
from __future__ import annotations

import argparse
import json
import gzip
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from secbart.bottleneck_token_common import SECU_TOK
from secbart.train_7b_rl_docker_func import (
    LANG_NAME, MAX_NEW, SECPLT_CASES, SecCodePLTDataset, _btoks_ids,
    docker_secplt_eval,
)
from secbart.utils import extract_code

FILT_CASES = os.path.join(os.path.dirname(SECPLT_CASES), "filtered-test_cases.json")
EVAL_ROOT = "/tmp/rl_dual_eval"
MAX_LEN = 1024


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
    ap.add_argument("--ref_8bit", action="store_true",
                    help="freeze reference policy in 8-bit to fit fullft PPO on one 80GB GPU")
    ap.add_argument("--tkh_agg", default="mean", choices=["mean", "min"])
    ap.add_argument("--w_head", type=float, default=0.5)
    ap.add_argument("--vul_shaping", type=float, default=0.0,
                    help="保留旧参数兼容性；PPO-TKH 实验固定为 0。")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--tkh_token_norm", default="raw_centered", choices=["none", "raw_centered", "raw_norm"],
                    help="TKH 即时奖励的响应内归一化；raw_centered = 安全分 - 0.5, "
                         "'没有安全相关 token' 恰好是 0 奖励。")
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

    ref_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if args.ref_8bit:
        ref_kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
        )
    ref = AutoModelForCausalLM.from_pretrained(args.seed_model, **ref_kwargs)
    if not args.ref_8bit:
        ref = ref.to(device)
    ref.eval()
    print(f"[memory] reference={'8bit' if args.ref_8bit else 'bf16'}", flush=True)
    policy = AutoModelForCausalLM.from_pretrained(
        args.seed_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    if args.lora_rank > 0:
        raise SystemExit("双臂 RL 用全量版规则: --lora_rank 0 (fullft)")
    policy.gradient_checkpointing_enable()
    policy.train()
    import bitsandbytes as bnb
    opt = bnb.optim.PagedAdamW8bit([p for p in policy.parameters() if p.requires_grad],
                                   lr=args.lr)
    value_head = nn.Linear(policy.config.hidden_size, 1).to(torch.bfloat16).to(device)
    value_opt = torch.optim.AdamW(value_head.parameters(), lr=args.lr)

    token_head = None
    token_head_out = None
    if args.token_head:
        sd = torch.load(args.token_head, map_location=device)
        # 输出维从 checkpoint 读: 2 = 旧双类头 (p_safe/p_vuln, 取 p0-p1);
        # 1 = cotrain 的 (0,1) 回归头 (sigmoid 已烘在 forward 里, 输出即安全分)。
        # 两种都能原地装, 旧 2 类路径逐字不变。
        if any(k.startswith("net.0") for k in sd):
            out_dim = int(sd["net.4.weight"].shape[0])
            net = nn.Sequential(nn.LayerNorm(policy.config.hidden_size),
                                nn.Linear(policy.config.hidden_size, 256), nn.GELU(),
                                nn.Dropout(0.1), nn.Linear(256, out_dim))
        else:
            out_dim = int(sd["weight"].shape[0])
            net = nn.Linear(policy.config.hidden_size, out_dim)
        token_head_out = out_dim
        token_head = nn.Module()
        token_head.net = net
        token_head.load_state_dict(sd)
        token_head = token_head.to(torch.bfloat16).to(device).eval()
        for p in token_head.parameters():
            p.requires_grad = False
        print(f"[reward] PPO token TKH {args.token_head} w={args.w_head} "
              f"norm={args.tkh_token_norm}; 双臂(vul臂反向)", flush=True)
    else:
        print("[reward] 纯硬门: r_sec=func·sec, r_vul=func·(1−sec)", flush=True)

    cases_path = SECPLT_CASES if args.dataset == "secodeplt" else FILT_CASES
    ds = SecCodePLTDataset(cases_path, args.max_tasks, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "rl_log.csv").open("w")
    log_f.write("step,r_sec,r_vul,func_s,sec_s,func_v,sec_v,tkh_abs,kl,adv_std,clip_frac,empty\n")

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
    step = 0
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
            # Rollout policy and value baseline are frozen before the PPO update.
            # 显存纪律 (09-15 01:1x 真实 batch 下 OOM 后按共享 GRPO 入口 :735-749 改):
            # ① log_softmax **原地**覆盖 logits, 不再多留一份 (B,T,V) 中间量
            #    (batch4×k4×双臂 = 32 行 × 1024 × 152k × 2B ≈ 10 GB/份);
            # ② 同一时刻**只允许一个**全词表 forward 存活 —— old_outs 用完立刻 del,
            #    ref_outs 算完 ref_logp 立刻 del, 再开 policy 的带梯度 forward。
            #    头的逐 token 分在 old_outs 还活着时就地取走 (head_d, 只有 (B,L) 大小)。
            with torch.no_grad():
                old_outs = policy(seg, attention_mask=attn, output_hidden_states=True)
                old_outs.logits = torch.log_softmax(old_outs.logits, dim=-1)
                old_lg = old_outs.logits[:, :-1]
                tgt = seg[:, 1:]
                old_logp = old_lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                old_logp = old_logp * (tgt != tok.pad_token_id).float()
                old_hs = old_outs.hidden_states[-1]
                v_old = value_head(old_hs.detach()).squeeze(-1)
                if token_head is not None:
                    _net = getattr(token_head, "net", token_head)
                    if token_head_out == 1:
                        # cotrain 回归头: 直接取 (0,1) 安全分, 不做任何变换。
                        # 下游 raw_centered 只做 -0.5 平移 ⇒ 线性重标定不影响
                        # 奖励方向, 无需换算。
                        head_d = _net(old_hs).squeeze(-1).float()
                    else:
                        _pt = torch.softmax(_net(old_hs).float(), dim=-1)
                        head_d = _pt[..., 0] - _pt[..., 1]     # p(safe) - p(vuln): 安全代码→正值，漏洞代码→负值
                else:
                    head_d = None
                del old_outs, old_lg, old_hs
                ref_outs = ref(seg, attention_mask=attn)
                ref_outs.logits = torch.log_softmax(ref_outs.logits, dim=-1)
                ref_lg = ref_outs.logits[:, :-1]
                ref_logp = ref_lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                ref_logp = ref_logp * (tgt != tok.pad_token_id).float()
                del ref_outs, ref_lg

            outs = policy(seg, attention_mask=attn, output_hidden_states=True)
            outs.logits = torch.log_softmax(outs.logits, dim=-1)   # 原地, 见上
            lg = outs.logits[:, :-1]
            tgt = seg[:, 1:]
            logp_tok = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            logp_tok = logp_tok * (tgt != tok.pad_token_id).float()
            v_new = value_head(outs.hidden_states[-1].detach()).squeeze(-1)
            st_t = torch.tensor(starts, device=device)
            gen_mask = torch.arange(seg_len - 1, device=device).unsqueeze(0) >= \
                (st_t - 1).unsqueeze(1)
            gen_mask = gen_mask & (tgt != tok.pad_token_id).bool()
            seq_logp = (logp_tok * gen_mask.float()).sum(1)
            kl_pen = ((ref_logp - logp_tok).square() * gen_mask.float()).sum(1)

            # ---- 奖励 (行序 = arm-major: 前 R 行 sec, 后 R 行 vul) ----
            R = B * args.k
            r_sec = fs_t * ss_t                       # func✓sec✓
            r_vul = fv_t * (1.0 - sv_t)               # func✓sec✗ (安全轴反向)
            tkh_reward = torch.zeros_like(v_old, dtype=torch.float32)
            if token_head is not None:
                sec_mask = torch.arange(seg_len, device=device).unsqueeze(0) >= \
                    st_t.unsqueeze(1)
                sec_mask = sec_mask & (seg != tok.pad_token_id)
                with torch.no_grad():
                    # head_d 在上面的 old_outs no_grad 块里就取好了 (那边一算完就 del
                    # old_outs, 省一份全词表 forward), 这里只做裁剪 + 平移。
                    d = head_d
                    code_d = d[:, :-1]
                    if args.tkh_token_norm == "raw_centered":
                        code_d = code_d - 0.5
                    elif args.tkh_token_norm == "raw_norm":
                        denom = gen_mask.float().sum(1, keepdim=True).clamp(min=1.0)
                        code_d = (code_d - 0.5) / denom
                    # 双臂都给 token 奖励：sec 臂 d>0.5→正，vul 臂 d<0.5→正(反号)
                    tkh_reward[:R, :-1] = args.w_head * code_d[:R] * gen_mask[:R].float()
                    tkh_reward[R:, :-1] = -args.w_head * code_d[R:] * gen_mask[R:].float()
            sec_empty = torch.tensor([l <= 0 for l in lens_sec], device=device)
            vul_empty = torch.tensor([l <= 0 for l in lens_vul], device=device)
            r_sec = r_sec + torch.where(sec_empty, -1.0, 0.0)
            r_vul = r_vul + torch.where(vul_empty, -1.0, 0.0)
            r_all = torch.cat([r_sec, r_vul])
            empty_t = torch.cat([sec_empty, vul_empty])

            # ---- PPO: token rewards + terminal dualarm outcome reward -> GAE ----
            Bk = seg.shape[0]
            gen_span = torch.cat([gen_mask, torch.zeros(Bk, 1, dtype=torch.bool,
                                                         device=device)], dim=1)
            r_pt = tkh_reward
            token_positions = torch.arange(seg_len - 1, device=device).unsqueeze(0)
            last_j = (token_positions * gen_mask.long()).max(1).values
            r_pt[torch.arange(Bk, device=device), last_j] += r_all.detach().float()
            v_next = torch.zeros_like(v_old)
            v_next[:, :-1] = v_old[:, 1:]
            in_next = torch.zeros_like(gen_span)
            in_next[:, :-1] = gen_span[:, 1:]
            deltas = (r_pt + args.gamma * v_next * in_next.float() - v_old) * gen_span.float()
            adv = torch.zeros_like(deltas)
            acc = torch.zeros(Bk, device=device)
            for t in range(seg_len - 1, -1, -1):
                acc = deltas[:, t] + args.gamma * args.lam * acc
                adv[:, t] = acc
            adv_tok = adv[:, :-1] * gen_mask.float()
            ratio = (logp_tok - old_logp.detach()).exp()
            clip_ratio = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps)
            surr = torch.minimum(ratio * adv_tok, clip_ratio * adv_tok)
            n_gen = gen_mask.float().sum().clamp(min=1.0)
            loss_p = -surr.sum() / n_gen
            value_target = (adv + v_old).detach()
            n_value = gen_span.float().sum().clamp(min=1.0)
            loss_v = ((v_new - value_target).square() * gen_span.float()).sum() / n_value
            loss = loss_p + args.vf_coef * loss_v + args.kl_beta * kl_pen.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            value_opt.step()
            value_opt.zero_grad()
            adv_seq = adv_tok.sum(1)
            clip_frac = (((ratio - 1).abs() > args.clip_eps) & gen_mask).float().sum() / n_gen
            log_f.write(f"{step},{r_sec.mean().item():.4f},{r_vul.mean().item():.4f},"
                        f"{fs_t.mean().item():.4f},{ss_t.mean().item():.4f},"
                        f"{fv_t.mean().item():.4f},{sv_t.mean().item():.4f},"
                        f"{tkh_reward[:R].abs().sum().item() / max(int(gen_mask[:R].sum().item()), 1):.4f},"
                        f"{kl_pen.mean().item():.4f},{adv_seq.std().item():.4f},{clip_frac.item():.4f},"
                        f"{empty_t.float().mean().item():.3f}\n")
            log_f.flush()
            print(f"step {step}/{args.steps} r_sec={r_sec.mean().item():.4f} "
                  f"r_vul={r_vul.mean().item():.4f} | sec func/safe="
                  f"{fs_t.mean().item():.3f}/{ss_t.mean().item():.3f} | vul func/safe="
                  f"{fv_t.mean().item():.3f}/{sv_t.mean().item():.3f} "
                  f"tkh={tkh_reward[:R].abs().mean().item():.4f} empty={empty_t.sum().item()} "
                  f"kl={kl_pen.mean().item():.4f} clip={clip_frac.item():.3f}", flush=True)
            # old_outs / ref_outs / old_lg / ref_lg 已在各自 no_grad 块里 del 过
            # (:372 / :378); 这里只能删此刻仍存在的名字, 否则 UnboundLocalError
            # —— 09-15 01:19 与 01:20 两次真实 batch 起跑都死在这一行。
            
            # ---- 逐 token 落盘 (每 step 一份 gzip jsonl) ----
            dump_dir = out_dir / "token_dumps"
            dump_dir.mkdir(exist_ok=True)
            dump_path = dump_dir / f"step_{step:04d}.jsonl.gz"
            R = B * args.k
            with gzip.open(dump_path, "wt", encoding="utf-8") as dump_f:
                for arm_idx, arm_name in enumerate(["sec", "vul"]):
                    offset = arm_idx * R
                    for b in range(B):
                        for i in range(args.k):
                            row = offset + b * args.k + i
                            ids = seg[row].tolist()
                            try:
                                first_pad = ids.index(tok.pad_token_id)
                                ids = ids[:first_pad]
                            except ValueError:
                                pass
                            tokens = [tok.decode([t], skip_special_tokens=False) for t in ids]
                            gm = gen_mask[row].tolist()[:len(ids)-1]
                            tkh = tkh_reward[row].tolist()[:len(ids)-1]
                            adv = adv_tok[row].tolist()[:len(ids)-1]
                            rec = {
                                "step": step, "arm": arm_name, "task_idx": b, "sample_idx": i,
                                "ids": ids, "tokens": tokens, "gen_start": starts[row],
                                "gen_mask": gm, "tkh_reward": tkh, "adv_tok": adv,
                                "r_seq": r_all[row].item(),
                                "func": (fs_t if arm_name=="sec" else fv_t)[b*args.k+i].item(),
                                "sec": (ss_t if arm_name=="sec" else sv_t)[b*args.k+i].item(),
                                "empty": empty_t[row].item(),
                            }
                            dump_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step % 50 == 0:
                print(f"[dump] step {step} → {dump_path} ({os.path.getsize(dump_path)/1e6:.2f} MB)",
                      flush=True)
            
            del outs, lg, logp_tok, old_logp, ref_logp
            del v_old, v_new, adv, adv_tok, ratio, r_pt, tkh_reward
            torch.cuda.empty_cache()

    log_f.close()
    policy.save_pretrained(out_dir / "merged_hf_model")
    tok.save_pretrained(out_dir / "merged_hf_model")
    torch.save(value_head.state_dict(), out_dir / "ppo_value_head.pt")
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
