#!/usr/bin/env python3
"""RL 用逐 token 安全头 v4 (fork of train_token_head.py, 09-08 留痕, 用户令: wcstatic
基线上从任何方向优化 token head 对 RL 的精确度, 跑完接 RL func_seq --token_head)。

数据: build_token_labels_rl.py 产物 (wcstatic_synthref_merge 池, 每对出 pos/neg 两条
同 prompt 的 code 行, pair_id 相同, labels 对齐 code token; 仅 pos 有 1 病灶 span)。

与 v3 的差异 (v3 探针全败根因逐一处理):
  1. 特征布局 = RL 同款 btoks 布局 (prompt|<vuln>*4|<secu>|code), 只存 code 段特征 —
     v3 在裸代码 causal 特征上训, RL 里作用在 btoks rollout 段 = 域错配。
  2. 目标函数 = token CE + λ_seg·行级 any-unsafe BCE (log(1-Π(1-p1))),
     直接对准 RL 消费的统计量 (min/flag 聚合 = 段内有没有洞) — v3 只训 token CE,
     概率在校准层面与聚合脱节 (clean 0.143 / mean/min 无分离)。
  3. 可换架构 linear(复现 v3 对照)|mlp(LN+256+dropout), 全在冻结 wcstatic 特征上。
判据 (沿用 train_token_head.py 上线门槛, RL 口径): 存在 θ 使 vul 命中 ≥0.9 且
sec 误报 ≤0.2 (flag); 附 mean/min 聚合分离度 (min_d: sec 行 <−0.2 占比 ≤0.2)。
best 保存按: 达门槛者优先 (fp 最小), 否则 val 行级 any-score AUC 优先。

v5 改动 (09-09, 用户令新优化策略; 四代败因复盘 #58): H2 的 span 0.26 vs hit 0.60
= 激活落点错位 (行级 smax 摊到病灶外 token); H1d 过火 (fp 0.47) = hi/lo 裕量失配;
RL 双臂负收益 = fp 噪声 + 病灶外激活无信号。→
  - 损失换 span-条件化 hinge (--loss span, 默认): vul 行**病灶内** (labels==1)
    smax ≥ hi=0.75 且**病灶外** ≤ lo=0.3; sec 行整行 ≤ lo=0.3 (v4 是 0.45)。
    病灶外压制把激活钉在病灶位 (对齐 E2 probe 语义与 RL min-d 聚合)。
  - 特征层可换 (--feat_layer, 默认 -1=末层): E2 probe 峰在 L16-20 → 对照臂 20。
RL 有效判据 (09-09 收严): sec 行 min_d<−0.2 占比 ≤0.1 且 vul **span** 命中 ≥0.6。

用法:
  python3 -m secbart.train_token_head_rl --data .../wcstatic_synthref_merge/token_labels_v7_ord.jsonl \
      --model .../btok_7b_fullft_wcstatic_synthref_merge/merged_hf_model \
      --out_dir .../btok_7b_fullft_wcstatic_synthref_merge/seq_head_v4_token \
      --gpu 3 --archs mlp,linear --epochs 40 --lr 1e-3 --batch_size 32 --seed 0 --cap_pairs 10000
"""
import argparse, json, os, random, shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from secbart.bottleneck_token_common import VULN_TOK, SECU_TOK

PAD_LAB = -100
# v7 三分类 (ord_cls) → 整数掩码: 0=irrel 1=vuln(side=0 病灶) 2=safe(side=1 防御)
CLS_ID = {"irrel": 0, "vuln": 1, "safe": 2}


def _cls_tensor(r, n):
    """行内三分类 → long 张量; 无 ord_cls 键 (旧标签) 返回 None。"""
    oc = r.get("ord_cls")
    if oc is None:
        return None
    return torch.tensor([CLS_ID.get(x, 0) for x in oc[:n]], dtype=torch.long)


class TokenHead(nn.Module):
    """逐 token 头 (逐位置独立分类, 卖点不丢): linear = v3 对照; mlp = LN+256+dropout。"""

    def __init__(self, hidden_size: int, arch: str = "mlp"):
        super().__init__()
        if arch == "linear":
            self.net = nn.Linear(hidden_size, 2)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2),
            )

    def forward(self, feats):            # (B, L, H) -> (B, L, 2)
        return self.net(feats)


def load_rows(path, cap_pairs, seed):
    pos, neg = [], []
    for line in open(path):
        r = json.loads(line)
        (pos if r["side"] == 0 else neg).append(r)
    pairs = sorted({r["pair_id"] for r in pos})
    random.Random(seed).shuffle(pairs)
    if cap_pairs:
        pairs = pairs[:cap_pairs]
    keep = set(pairs)
    pos = [r for r in pos if r["pair_id"] in keep]
    neg = [r for r in neg if r["pair_id"] in keep]
    return pos, neg, pairs


def _cache_path(cache_dir, tag, layer, args):
    """缓存键 = 数据文件(路径+大小+mtime) + 模型 + max_len + 层。数据改了键就变,
    不会静默读到旧特征 (这是一切缓存类改动最容易出的错)。"""
    import hashlib
    key = hashlib.md5(
        f"{args.data}|{os.path.getsize(args.data)}|{os.path.getmtime(args.data)}|"
        f"{args.model}|{args.max_len}|{layer}".encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"{tag}_L{layer}_{key}.pt")


def extract(model, tok, rows, device, max_len, tag, layer=-1):
    """btoks 布局前向, 只存 code 段特征 (bf16 CPU)。返回 list[feat], list[lab], H。
    layer: output_hidden_states 索引 (v5: 可存中间层对照臂; -1=末层)。"""
    v = tok.convert_tokens_to_ids(VULN_TOK)
    s = tok.convert_tokens_to_ids(SECU_TOK)
    feats, labs, plens, wts, clss = [], [], [], [], []
    model.eval()
    batch = []
    tlen = 0
    with torch.no_grad():
        for r in rows:
            cids = tok(r["code"], add_special_tokens=False,
                       truncation=True, max_length=1020)["input_ids"]
            lab = torch.tensor(r["labels"][: len(cids)], dtype=torch.long)
            # 09-11 v4 标签 (多票一致) 可带每 token 重要性 weights; 无该键 → 退化为
            # 二值 (= lab.float()) → --use_weights 在老标签上等价旧行为。
            wsrc = r.get("weights")
            wt = (torch.tensor([float(x) for x in wsrc[: len(cids)]], dtype=torch.float32)
                  if wsrc else lab.float())
            pre_len = max(8, max_len - 5 - len(cids))
            pre = tok(r["prompt"], add_special_tokens=False,
                      truncation=True, max_length=pre_len)["input_ids"]
            batch.append((pre + [v] * 4 + [s] + cids, lab, wt, len(cids),
                          _cls_tensor(r, len(cids))))
            tlen += len(batch[-1][0])
            if len(batch) >= 8 or tlen > 6000:
                _flush(model, tok, batch, device, feats, labs, plens, wts, clss, layer)
                batch, tlen = [], 0
        if batch:
            _flush(model, tok, batch, device, feats, labs, plens, wts, clss, layer)
    n_tok = sum(plens)
    h = feats[0].shape[-1]
    print(f"[extract {tag} hs{layer}] rows={len(rows)} code_tokens={n_tok} "
          f"~{n_tok * h * 2 / 1e9:.1f}GB bf16", flush=True)
    return feats, labs, wts, clss


def _flush(model, tok, batch, device, feats, labs, plens, wts, clss, layer=-1):
    ms = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), ms), tok.pad_token_id, dtype=torch.long)
    for i, (ids_, lab, wt, clen, _) in enumerate(batch):
        ids[i, : len(ids_)] = torch.tensor(ids_)
    out = model(input_ids=ids.to(device), output_hidden_states=True)
    h = out.hidden_states[layer]         # (B, ms, H) bf16
    for i, (ids_, lab, wt, clen, cls_) in enumerate(batch):
        # 09-10 修致命错位: 原 `h[i, -clen:]` 只在「本行 = batch 最长」时才是 code 段。
        # ids 是**右填充** (ids[i,:len(ids_)]=...), 故短行的末 clen 个状态落在填充区 ——
        # 实测 86.8% 的行错位 (53.7% 切片完全在填充区, 中位位移 224 tok) →
        # 头拿不到与标签对齐的特征, 这是 v4/v5 全臂 AUC≈0.50 的唯一根因。
        # 正确切片 = 该行真实 token 区间 [li-clen, li), 与填充方式和同批长度无关。
        li = len(ids_)
        seg = h[i, li - clen:li]
        feats.append(seg.cpu())          # bf16
        labs.append(lab)
        wts.append(wt)
        plens.append(clen)
        clss.append(cls_)


def pad_batch(X, y, W, C, idxs, device):
    Ls = [y[i].shape[0] for i in idxs]
    L = max(Ls)
    H = X[0].shape[-1]
    xp = torch.zeros(len(idxs), L, H, dtype=torch.float32, device=device)
    yp = torch.full((len(idxs), L), PAD_LAB, dtype=torch.long, device=device)
    wp = torch.zeros(len(idxs), L, dtype=torch.float32, device=device)
    cp = torch.zeros(len(idxs), L, dtype=torch.long, device=device)   # 0=irrel (填充区同为 0)
    for k, i in enumerate(idxs):
        xp[k, : Ls[k]] = X[i].float()
        yp[k, : Ls[k]] = y[i]
        wp[k, : Ls[k]] = W[i]
        if C[i] is not None:
            cp[k, : Ls[k]] = C[i]
    return xp, yp, wp, cp


def row_smax_p1_hinge(logits, yb, m, beta=25.0, lo=0.45, hi=0.7):
    """行级 hinge on smooth-max p1 —— RL min-d 聚合的可微代理 (H1c 迭代3)。

    H1 (λ1) 败因 = token-CE mean 淹没行 BCE (梯度 <0.1%)。H1b (λ50) 败因 = 目标错配:
    any-BCE 1-Π(1-p1) 可被弥散弱 p1 满足 (200 tok×0.05 → any_p≈0.994), 而 RL 消费
    min d = min(1−2·p1) 要求**单个**强 token。→ 直接监督行内最大 p1:
    smax = logsumexp(β·p1, 掩码)/β; vul 行推 smax≥hi=0.7 (⟺ min d≤−0.4), sec 行压
    smax≤lo=0.45 (⟺ min d≥0.1) —— 阈值对齐 RL τ±0.2 判决边界且留裕量。β=25:
    单 token 0.7 → smax 0.7; 200 tok 弥散 0.05 → smax≈0.26 (不再计为命中)。
    p1 用 softmax(与 RL 部署一致, 非 sigmoid)。"""
    p1 = torch.softmax(logits, dim=-1)[..., 1]
    off = torch.where(m, torch.zeros_like(p1), torch.full_like(p1, -1e4))
    smax = torch.logsumexp(beta * p1 + off, dim=-1) / beta
    row_tgt = (m & (yb == 1)).any(1).float()
    pos = (row_tgt * F.relu(hi - smax)).mean()
    neg = ((1.0 - row_tgt) * F.relu(smax - lo)).mean()
    return pos + neg


def row_span_hinge(logits, yb, m, beta=25.0, hi=0.75, lo=0.3):
    """v5 span-条件化 hinge (09-09): 修复 H2 span 0.26 vs hit 0.60 的落点错位。

    vul 行: 病灶 (yb==1) 内 smax ≥ hi (激活必须落在病灶位, 非行内任意处);
      病灶外 token 压制 ≤ lo (防行尾/语法 token 抢激活 — RL 消费 min-d 时
      病灶外高 p1 与病灶内高 p1 等权, 但 E2/消融证明只有病灶位激活有信号)。
    sec 行: 整行压 ≤ lo (无病灶, 全行即"病灶外"; v4 lo 0.45 → 0.3 抗过火)。
    空集防 -inf: 非 vul 行 span 项回填整行 p1 (值有限) 且权重 0 → 贡献恒 0;
      全行病灶行 out 项 logsumexp 空 → smax=-inf → relu(-inf-lo)=0 无惩罚。
    p1 用 softmax (与 RL 部署一致, 非 sigmoid)。"""
    p1 = torch.softmax(logits, dim=-1)[..., 1]
    NEG = -1e4
    is_vul = (m & (yb == 1)).any(1)
    span_p = torch.where(is_vul[:, None],
                         torch.where((yb == 1) & m, p1, torch.full_like(p1, NEG)),
                         p1)   # vul 行只对病灶 logsumexp; 非 vul 行回填整行 (求有限)
    smax_span = torch.logsumexp(beta * span_p, dim=-1) / beta
    out_p = torch.where((yb != 1) & m, p1, torch.full_like(p1, NEG))
    smax_out = torch.logsumexp(beta * out_p, dim=-1) / beta
    w_v = is_vul.float()
    pos = (w_v * F.relu(hi - smax_span)).sum() / w_v.sum().clamp(min=1.0)
    neg = F.relu(smax_out - lo).mean()   # 空 out 行 (全行病灶): smax=-inf → relu=0
    return pos + neg


def row_ordinal_hinge(logits, cp, m, beta=25.0, margin=0.3):
    """v7 版本内成对序数 hinge (09-12): 只在**同一版本内部**比较, 永不跨版本配对。

    动机: 现有头只被训练成"标注检测器" —— side=1 (修复版) 的 labels 全零, 头从没学过
    "防御代码 = 安全", 于是 B = P(safe > irrel) 被结构性压到 0.16~0.44, H&T 上限 ~0.65,
    RL 里加防御代码的 token 反被判低分。本项直接监督两个版本内的序数关系:

      side=0 行 (含 vuln): min_{irrel} d > max_{vuln} d      → L_A
      side=1 行 (含 safe): min_{safe}  d > max_{irrel} d     → L_B

    其中 d = p_safe − p_unsafe = 1 − 2·p1 ∈ [−1,1] (与 RL 部署同源)。
    跨版本比较被完全排除 ⇒ 不存在"整条 task 版拿高分"的序列级捷径, C 由 A/B 顺带抬升。
    soft-min/soft-max 用 logsumexp (与 row_span_hinge 同风格同温), 空类回填 -1e4 且按行
    权重归零, 保证空集不产生 -inf 也不贡献梯度。"""
    d = 1.0 - 2.0 * torch.softmax(logits, dim=-1)[..., 1]
    NEG = -1e4

    def smax(sel):
        return torch.logsumexp(torch.where(sel & m, beta * d, torch.full_like(d, NEG)),
                               dim=-1) / beta

    def smin(sel):
        return -torch.logsumexp(torch.where(sel & m, -beta * d, torch.full_like(d, NEG)),
                                dim=-1) / beta

    vul = (cp == 1) & m
    saf = (cp == 2) & m
    irr = (cp == 0) & m
    rowA = (vul.any(1) & irr.any(1)).float()
    rowB = (saf.any(1) & irr.any(1)).float()
    lA = F.relu(smax(vul) - smin(irr) + margin)      # 违反 min_irrel > max_vuln 的量
    lB = F.relu(smax(irr) - smin(saf) + margin)      # 违反 min_safe  > max_irrel 的量
    return ((lA * rowA).sum() / rowA.sum().clamp(min=1.0)
            + (lB * rowB).sum() / rowB.sum().clamp(min=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--archs", default="mlp,linear",
                    help="逗号分隔 (linear|mlp): 共享一次特征提取分训, 全局最优写固定名")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lam_seg", type=float, default=5.0, help="行级 smooth-max-p1 hinge 权重")
    ap.add_argument("--w_unsafe", type=float, default=3.0, help="token CE 类权重")
    ap.add_argument("--use_weights", action="store_true",
                    help="v4 标签: 用 weights 键 (每 token 票数权重) 缩放正样本 CE; "
                         "缺 weights 键时等价于关闭")
    ap.add_argument("--beta", type=float, default=25.0,
                    help="row_smax_p1_hinge 平滑温度 (H1d: 25→40 抗弥散过火)")
    ap.add_argument("--val_pairs", type=float, default=0.15)
    ap.add_argument("--cap_pairs", type=int, default=10000, help="0=全部")
    ap.add_argument("--loss", default="span", choices=["span", "smax", "ordinal"],
                    help="span=v5 病灶位条件化 hinge (默认); smax=v4 行级 smooth-max; "
                         "ordinal=v7 版本内成对序数 (需 --data 带 ord_cls 键)")
    ap.add_argument("--lam_ord", type=float, default=5.0, help="ordinal hinge 权重")
    ap.add_argument("--margin_ord", type=float, default=0.3,
                    help="ordinal 序数间隔 (d 值域 [-1,1]; 0.3 = 要求清晰分离)")
    ap.add_argument("--save_ckpt_every", type=int, default=0,
                    help="09-12: >0 时在每个评估点另存 token_head_rl_{arch}_ep{N}.pt。"
                         "默认 0 = 旧行为 (只存 gate_score 最优的那个)。"
                         "**ordinal 损失下必开**: gate_score 是按旧二值 flag 口径算的 "
                         "(1−fp)+hit(+0.5 过门), 与序数目标弱相关, 只存它等于让 epoch 选择 "
                         "被一个无关指标决定; 开此项后可离线按 H&T 挑最佳 epoch")
    ap.add_argument("--hi", type=float, default=0.75, help="vul 病灶内 smax 目标下限")
    ap.add_argument("--lo", type=float, default=0.3,
                    help="病灶外 (vul) / 整行 (sec) smax 上限")
    ap.add_argument("--feat_layer", type=int, default=-1,
                    help="output_hidden_states 索引 (默认 -1=末层; E2 probe 峰对照臂 20)")
    ap.add_argument("--feat_cache", default=None,
                    help="特征缓存目录: 抽一次存 /tmp 或 NFS, 之后同 (数据/模型/max_len/层) "
                         "的变体直接读盘 ⇒ 单轮从 ~35 min 降到分钟级 (只跑 MLP)")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"

    pos, neg, all_pairs = load_rows(args.data, args.cap_pairs, args.seed)
    n_val = max(1, int(len(all_pairs) * args.val_pairs))
    random.Random(args.seed).shuffle(all_pairs)
    val_pairs = set(all_pairs[:n_val])
    tr = [r for r in pos + neg if r["pair_id"] not in val_pairs]
    va = [r for r in pos + neg if r["pair_id"] in val_pairs]
    n_pos = sum(1 for r in tr if r["side"] == 0)
    print(f"[data] pairs={len(all_pairs)} val_pairs={n_val} train_rows={len(tr)} "
          f"(pos={n_pos}/neg={len(tr) - n_pos}) val_rows={len(va)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # 09-12: 特征缓存。冻结 7B 的前向是每轮 30+ min 的全部成本, 而它只取决于
    # (数据文件, 模型, max_len, feat_layer) —— 与要扫的 loss/beta/margin/lam_ord 无关。
    # 落盘一次后每次变体只需跑 MLP (分钟级), 扫描才可能快。
    cache = {}
    if args.feat_cache:
        os.makedirs(args.feat_cache, exist_ok=True)
        for tag in ("train", "val"):
            p = _cache_path(args.feat_cache, tag, args.feat_layer, args)
            if os.path.exists(p):
                cache[tag] = torch.load(p, map_location="cpu")
                print(f"[cache] hit {tag} ← {p}", flush=True)
    need = [t for t in ("train", "val") if t not in cache]
    if need:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=device)
        for tag, rows_ in (("train", tr), ("val", va)):
            if tag in cache:
                continue
            cache[tag] = extract(model, tok, rows_, device, args.max_len, tag,
                                 args.feat_layer)
            if args.feat_cache:
                p = _cache_path(args.feat_cache, tag, args.feat_layer, args)
                torch.save(cache[tag], p)
                print(f"[cache] write {tag} → {p}", flush=True)
        del model
        torch.cuda.empty_cache()
    else:
        print("[cache] train/val 全部命中, 不加载基座 (省 ~30 min + 15GB 显存)", flush=True)
    X_tr, y_tr, W_tr, C_tr = cache["train"]
    X_va, y_va, W_va, C_va = cache["val"]
    if args.loss == "ordinal" and any(c is None for c in C_tr + C_va):
        raise SystemExit("--loss ordinal 需要 --data 带 ord_cls 键 "
                         "(token_labels_v7_ord.jsonl); 当前数据缺该键")
    if args.loss == "ordinal":
        n_v = sum(int((c == 1).sum()) for c in C_tr if c is not None)
        n_s = sum(int((c == 2).sum()) for c in C_tr if c is not None)
        n_i = sum(int((c == 0).sum()) for c in C_tr if c is not None)
        print(f"[ord] train 三分类 token: vuln={n_v} safe={n_s} irrel={n_i}", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def run_epoch(X, y, W, C, head, opt, w1, train, ep):
        idx = list(range(len(X)))
        if train:
            random.Random(args.seed + ep).shuffle(idx)
        tot, n = 0.0, 0
        for s in range(0, len(idx), args.batch_size):
            idxs = idx[s:s + args.batch_size]
            xp, yp, wp, cp = pad_batch(X, y, W, C, idxs, device)
            logits = head(xp)
            m = yp != PAD_LAB
            if args.use_weights:
                # 每 token 重要性: 正样本按票数权重缩放 (核心 1.0 / 支持 0.4), 负样本恒 1
                ce_tok = F.cross_entropy(logits[m], yp[m], weight=w1, reduction="none")
                rel = torch.where(yp[m] == 1, wp[m], torch.ones_like(wp[m]))
                ce = (ce_tok * rel).sum() / rel.sum().clamp(min=1e-6)
            else:
                ce = F.cross_entropy(logits[m], yp[m], weight=w1)
            if args.loss == "span":
                loss = ce + args.lam_seg * row_span_hinge(
                    logits, yp, m, beta=args.beta, hi=args.hi, lo=args.lo)
            elif args.loss == "ordinal":
                loss = ce + args.lam_ord * row_ordinal_hinge(
                    logits, cp, m, beta=args.beta, margin=args.margin_ord)
            else:
                loss = ce + args.lam_seg * row_smax_p1_hinge(logits, yp, m, beta=args.beta)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(idxs)
            n += len(idxs)
        return tot / n

    archs = args.archs.split(",")
    all_ev = {}
    for arch in archs:
        head = TokenHead(X_tr[0].shape[-1], arch).to(device).float()
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
        w1 = torch.tensor([1.0, args.w_unsafe], device=device)
        best = None
        for ep in range(1, args.epochs + 1):
            tl = run_epoch(X_tr, y_tr, W_tr, C_tr, head, opt, w1, True, ep)
            vl = run_epoch(X_va, y_va, W_va, C_va, head, opt, w1, False, ep)
            if ep % 5 == 0 or ep == args.epochs:
                ev = evaluate(X_va, y_va, head, device)
                print(f"[{arch}] ep{ep} tl={tl:.4f} vl={vl:.4f} {ev['short']}", flush=True)
                if args.save_ckpt_every and ep % args.save_ckpt_every == 0:
                    torch.save(head.state_dict(),
                               Path(args.out_dir) / f"token_head_rl_{arch}_ep{ep}.pt")
                score = ev["gate_score"]
                if best is None or score > best[0]:
                    best = (score, ev, ep)
                    torch.save(head.state_dict(),
                               Path(args.out_dir) / f"token_head_rl_{arch}.pt")
            else:
                print(f"[{arch}] ep{ep} tl={tl:.4f} vl={vl:.4f}", flush=True)
        _, ev, eph = best
        # 09-10: 记口径 (RL wrapper 会校验 feat_layer==-1 —— RL 侧固定 hidden_states[-1],
        # 层不一致 = 域错配, 必须能在报告里查出来; 数据/损失也记, 用于亚臂溯源)。
        ev.update({"best_ep": eph, "arch": arch, "lam_seg": args.lam_seg,
                   "lam_ord": args.lam_ord, "margin_ord": args.margin_ord,
                   "use_weights": bool(args.use_weights),
                   "feat_layer": args.feat_layer, "loss": args.loss,
                   "hi": args.hi, "lo": args.lo,
                   "data": Path(args.data).name})
        all_ev[arch] = ev
        print(f"[best {arch}] ep{eph} score={best[0]}", flush=True)
    win = max(all_ev, key=lambda a: all_ev[a]["gate_score"])
    shutil.copyfile(Path(args.out_dir) / f"token_head_rl_{win}.pt",
                    Path(args.out_dir) / "token_head_rl.pt")
    plain = {k: v for k, v in all_ev[win].items() if not isinstance(v, (dict, list))}
    json.dump(plain, open(Path(args.out_dir) / "report_rl.json", "w"),
              indent=1, ensure_ascii=False)
    json.dump(all_ev, open(Path(args.out_dir) / "report_rl_all.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"[winner] {win} ep{all_ev[win]['best_ep']} score={all_ev[win]['gate_score']} "
          f"-> {args.out_dir}/token_head_rl.pt", flush=True)
    print(f"[report] {json.dumps(plain, ensure_ascii=False)[:900]}", flush=True)


def evaluate(X, y, head, device):
    """RL 口径: θ 扫 flag (行内 any p1>θ) 的 sec 误报/vul 命中/span 命中; min 聚合分离。"""
    pks, vul, labs = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xp = torch.zeros(min(64, len(X) - i), max(y[j].shape[0] for j in range(i, min(i + 64, len(X)))),
                             X[0].shape[-1], dtype=torch.float32, device=device)
            for k, j in enumerate(range(i, min(i + 64, len(X)))):
                xp[k, : y[j].shape[0]] = X[j].float()
            lg = head(xp)
            p1 = torch.softmax(lg, dim=-1)[..., 1]  # 与 RL 部署一致 (softmax 非 sigmoid)
            for k, j in enumerate(range(i, min(i + 64, len(X)))):
                L = y[j].shape[0]
                pks.append(p1[k, :L].cpu())
                labs.append(y[j])
                vul.append(bool((y[j] == 1).any()))
    sec_i = [i for i, v in enumerate(vul) if not v]
    vul_i = [i for i, v in enumerate(vul) if v]
    rows = {  # d = p_safe - p_unsafe 的 min (RL min 聚合)
        i: (pks[i], labs[i], (1.0 - 2.0 * pks[i]).min().item()) for i in range(len(pks))}

    best_gate = None
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        fp = sum(1 for i in sec_i if rows[i][0].max() > th) / max(1, len(sec_i))
        hit = sum(1 for i in vul_i if rows[i][0].max() > th) / max(1, len(vul_i))
        span = sum(1 for i in vul_i
                   if (rows[i][0][rows[i][1] == 1] > th).any()) / max(1, len(vul_i))
        if best_gate is None or (hit >= 0.9 and fp < best_gate[1]) or \
           (hit > best_gate[2] and best_gate[2] < 0.9):
            best_gate = (th, fp, hit, span)
    th, fp, hit, span = best_gate
    minS = sum(1 for i in sec_i if rows[i][2] < -0.2) / max(1, len(sec_i))
    minV = sum(1 for i in vul_i if rows[i][2] < -0.2) / max(1, len(vul_i))
    gate_ok = bool(hit >= 0.9 and fp <= 0.2)
    try:
        from sklearn.metrics import roc_auc_score
        surv = []
        for i in range(len(pks)):
            p = pks[i].clamp(1e-6, 1.0)
            surv.append(float(1.0 - torch.exp(torch.log(1.0 - p).sum())))
        auc = float(roc_auc_score(vul, surv))
    except Exception:
        auc = 0.0
    ev = {"th": float(th), "sec_fp": round(fp, 4), "vul_hit": round(hit, 4),
          "vul_span_hit": round(span, 4), "min_sec_lt02": round(minS, 4),
          "min_vul_lt02": round(minV, 4), "row_auc": auc,
          "n_sec": len(sec_i), "n_vul": len(vul_i), "gate_ok": gate_ok}
    ev["gate_score"] = round((1.0 - fp) + min(hit, 1.0) + (0.5 if gate_ok else 0.0), 4)
    ev["short"] = (f"fp={fp:.2f} hit={hit:.2f} span={span:.2f} minS={minS:.2f} "
                   f"minV={minV:.2f} auc={auc:.3f} gate={'OK' if gate_ok else 'no'}")
    return ev


if __name__ == "__main__":
    main()
