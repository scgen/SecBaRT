"""Shared bottleneck-token components (framework-agnostic).

This module holds the pieces shared by the standalone HF cotrain trainer
(``train_7b_fullft.py``), the eval servers/scripts and the data-prep utilities:

* bottleneck/special-token constants (``<vuln>`` / ``<secu>``) and security
  class labels (neutral/unsafe/safe),
* the token-level security head (``SecurityHead``) attached to shared hidden
  states,
* per-sample processing / batching (``_process_sample``, ``bottleneck_token_collate``),
* the BPE-level edit-distance labeler (``token_diff_labels``),
* the loss-curve plotter (``plot_loss_curve``).

The marker-era verl/FSDP trainer that used to live here has been removed;
training now runs only through the standalone HF trainer.
"""
from __future__ import annotations

import difflib

import torch
import torch.nn as nn

from secbart import ast_labeling as _ast_labeling
from secbart.ast_labeling import ast_token_diff_labels, dp_diff_labels

VULN_TOK = "<vuln>"
SECU_TOK = "<secu>"
JFIX_TOK = "<jfix>"    # 两段条件生成：需要修复（vulcode 含 UNSAFE）
JKEEP_TOK = "<jkeep>"  # 两段条件生成：无需修复（安全/通用）
# 三段式思维链布局（L 用户方案）：<func_anal> 在 input 后 <vuln> 前（功能分析），
# <vuln_anal> 在 vulcode 后（漏洞分析），<secu_impl> 在其后 <secu> 前（安全实现规划）
FUNC_ANAL_TOK = "<func_anal>"
VULN_ANAL_TOK = "<vuln_anal>"
SECU_IMPL_TOK = "<secu_impl>"
# L53: multi-token variant — 每个瓶颈位置用独立 token <vuln1>..<vulnN>
# （默认仍为单个 <vuln> 重复 N 次；multi_vuln=True 时启用）
VULN_TOKS = [f"<vuln{i}>" for i in range(1, 9)]
# 隐式思维链（think）布局（用户方案 #4/#5）：
# - <think>*N 块：input 之后的 4 个摘要 token，think 文本从该块重建（loss 0.5），
#   推理时 think 文本不生成（全压缩），<think> 块代替显式分析。
# - <think_func_anal>/<think_vuln_anal>/<think_secu_impl>：#5 三个独立摘要 token，
#   各自负责生成 func_anal/vuln_anal/secu_impl 一段（各 loss 0.5）。
THINK_TOK = "<think>"
THINK_FUNC_ANAL_TOK = "<think_func_anal>"
THINK_VULN_ANAL_TOK = "<think_vuln_anal>"
THINK_SECU_IMPL_TOK = "<think_secu_impl>"
# 注意：Qwen2.5-Coder-7B 词表无 <think>，注册后落在 151667-151670 槽位。

CLS_IGNORE = -100
CLS_NEUTRAL = 0
CLS_UNSAFE = 1
CLS_SAFE = 2


def token_diff_labels(vul_ids, sec_ids):
    """Edit-distance alignment of vul/sec token id lists.

    Returns (vul_labels, sec_labels) with values in {0,1,2}:
      vul side: D/S -> CLS_UNSAFE(1), U -> CLS_NEUTRAL(0)
      sec side: A/S -> CLS_SAFE(2),    U -> CLS_NEUTRAL(0)
    """
    A, B = vul_ids, sec_ids
    if A == B:
        return [CLS_NEUTRAL] * len(A), [CLS_NEUTRAL] * len(B)
    vul_labels, sec_labels = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, A, B, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            vul_labels.extend([CLS_NEUTRAL] * (i2 - i1))
            sec_labels.extend([CLS_NEUTRAL] * (j2 - j1))
        elif tag == "delete":
            vul_labels.extend([CLS_UNSAFE] * (i2 - i1))
        elif tag == "insert":
            sec_labels.extend([CLS_SAFE] * (j2 - j1))
        elif tag == "replace":
            vul_labels.extend([CLS_UNSAFE] * (i2 - i1))
            sec_labels.extend([CLS_SAFE] * (j2 - j1))
    return vul_labels, sec_labels


class SecurityHead(nn.Module):
    """Lightweight token-level security classifier on shared hidden states."""

    def __init__(self, hidden_size, num_classes=3, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


class SeqSecurityHead(nn.Module):
    """Sequence-level security head (1-query attention pooling).

    Layout: ``input | <vuln>*N | vulcode | <secu> | seccode`` — the head
    consumes the *full* seccode segment's hidden states (all tokens visible,
    unlike the token-level :class:`SecurityHead`, whose position-k output only
    sees the causal prefix). Emits one overall SAFE/UNSAFE/NEUTRAL judgment,
    intended as the sequence-level reward gate in RL, trained against
    external-judgment (docker func_sec) distilled labels.

    A single learnable query cross-attends over the segment (weighted pool).
    The byproduct attention weights ``w_k`` are the per-token *contribution*
    to the overall judgment — a global-context per-token signal, unlike the
    local confidence delta of the token head. No query/key/value projections:
    the 7B causal hidden states already encode full-prefix information, so
    direct dot-product pooling keeps the head ~1M params and easy to fit.
    """

    def __init__(self, hidden_size, num_classes=3, dropout=0.1, init_scale=0.02):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.query, std=init_scale)
        self.ln = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, hidden_states, seg_mask=None):
        """hidden_states: (B, Ls, H) seccode-segment hidden states.
        seg_mask: (B, Ls) bool, True = valid position (padding filtered).
        Returns (logits (B, 3), attn_weights (B, Ls))."""
        B, Ls, H = hidden_states.shape
        q = self.query.expand(B, -1, -1)                    # (B, 1, H)
        scores = q @ hidden_states.transpose(-2, -1) / (H ** 0.5)  # (B, 1, Ls)
        if seg_mask is not None:
            scores = scores.masked_fill(~seg_mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(scores, dim=-1)                   # (B, 1, Ls)
        pooled = (w @ hidden_states).squeeze(1)             # (B, H)
        logits = self.mlp(self.ln(pooled))                  # (B, 3)
        return logits, w.squeeze(1)


class BottleneckQFormer(nn.Module):
    """BLIP-2/ICAE-style explicit bottleneck compressor (``<vuln>`` queries).

    ``n_queries`` learnable queries cross-attend to the input prompt's
    last-layer hidden states (encoder features), then self-attend + FFN,
    producing refined ``(B, N, H)`` bottleneck states. During cotrain these
    states replace the ``<vuln>`` input embeddings in the generation stream,
    so compression is explicit (cross-attention pooling) instead of relying
    only on causal LM attention through the last input token.

    ``h_vuln`` (the model's own ``<vuln>`` hidden states) is added as a
    residual so the module starts from the already-working implicit
    compression and only refines it (warm start, keeps ablation fair).
    """

    def __init__(self, hidden_size, n_queries=4, n_layers=2, n_heads=8,
                 dropout=0.1, init_scale=0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.zeros(n_queries, hidden_size))
        nn.init.normal_(self.queries, std=init_scale)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=n_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, h_input, h_vuln=None):
        """h_input: (B, L, H) last-layer hidden states of the input prompt.
        h_vuln: (B, N, H) last-layer hidden states of the <vuln> block.
        Returns refined bottleneck states (B, N, H)."""
        B = h_input.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        if h_vuln is not None:
            q = q + h_vuln
        for layer in self.layers:
            q = layer(tgt=q, memory=h_input)
        return self.ln(q)


def _process_sample(
    sample, tokenizer, n_vuln, max_length, truncation, vulcode_loss_weight,
    cls_neutral_weight, cls_label_mode="ast", cls_label_skip_modes="comment+string",
    cls_label_align="difflib", cls_label_overlap_ratio=0.0,
    vulcode_see_input=False,
    seccode_see_vulcode=False,
    multi_vuln=False,
    interleave=False,
    keep_up=1.0,
    judge=False,
    vuln_anal_vis=True,
    seccode_cond=False,
    think_mode="none",
):
    """Build raw (unpadded) tensors for one sample.

    Returns input_ids (1D long), attention_mask (T, T) bool (True=attend),
    position_ids (1D long), loss_mask (1D float), cls_labels (1D long),
    cls_weights (1D float). cls_labels/cls_weights are token-level security
    targets (CLS_IGNORE outside vulcode/seccode; CLS_UNSAFE/CLS_SAFE/CLS_NEUTRAL
    inside), with neutral tokens down-weighted by ``cls_neutral_weight``.

    loss_mask follows verl's *target-aligned* convention: ``loss_mask[k]``
    weights the loss of predicting target token ``k`` (from position k-1),
    matching the trainer's ``loss_mask[:, 1:]`` slicing.
    """
    vuln_id = tokenizer.convert_tokens_to_ids(VULN_TOK)
    secu_id = tokenizer.convert_tokens_to_ids(SECU_TOK)
    eos_id = tokenizer.eos_token_id

    inp_ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
    vul_ids = tokenizer(sample["vul-code"], add_special_tokens=False)["input_ids"]
    sec_ids = tokenizer(sample["sec-code"], add_special_tokens=False)["input_ids"]

    L, N, F = len(inp_ids), n_vuln, len(vul_ids)
    S = len(sec_ids)
    T = L + N + F + 1 + S + 1  # input + vuln*N + vulcode + <secu> + seccode + eos

    if T > max_length:
        if truncation == "error":
            raise RuntimeError(f"sequence length {T} > max_length {max_length}")
        elif truncation == "right":
            # Prefer cutting the seccode tail (keeps the structural prefix and
            # the eos), then the input prefix, then the vulcode head (keeps the
            # vulcode tail adjacent to <secu>). The mask/positions/loss are
            # recomputed from the reduced lengths below.
            if L + N + F + 2 + S > max_length:
                S = max(0, max_length - (L + N + F + 2))
                sec_ids = sec_ids[:S]
            if L + N + F + 2 > max_length:
                L = max(0, max_length - (N + F + 2))
                inp_ids = inp_ids[-L:] if L else []
            if N + F + 2 > max_length:
                F = max(0, max_length - (N + 2))
                vul_ids = vul_ids[-F:] if F else []
            if L + N + F + 2 > max_length:
                raise RuntimeError(
                    f"cannot fit even the compressed skeleton (vuln*{N}+vulcode+<secu>+eos) "
                    f"in max_length={max_length}"
                )
            T = L + N + F + 1 + S + 1
        elif truncation == "left":
            # Drop input from the front only; keep the whole structural prefix.
            drop = T - max_length
            if drop > L:
                raise RuntimeError("left truncation would cut into vuln/vulcode")
            inp_ids = inp_ids[drop:]
            L = len(inp_ids)
            T = L + N + F + 1 + S + 1
        else:
            raise ValueError(f"unknown truncation mode: {truncation}")

    vuln_ids = (
        [tokenizer.convert_tokens_to_ids(t) for t in VULN_TOKS[:N]]
        if multi_vuln else [vuln_id] * N
    )

    # token 级安全标签（vulcode 段 UNSAFE/SAFE/NEUTRAL），judge 模式的「需要修复」
    # 判定需要它，故提前到序列构造之前（与分支后放置互斥，仅计算一次）。
    if cls_label_mode == "precomputed":
        # 外部预计算标签（data-prep 产出，如 LLM-as-judge / difflib 三分类）：
        # 0.0=UNSAFE / 0.5=NEUTRAL / 1.0=SAFE，直接映射到 CLS_* 编号。
        # 长度已由 data-prep 用同一 tokenizer 对齐；截断时按 vul 尾 / sec 头切片。
        def _to_cls(x):
            x = float(x)
            if x <= 0.25:
                return CLS_UNSAFE
            if x >= 0.75:
                return CLS_SAFE
            return CLS_NEUTRAL
        vul_labels = [_to_cls(x) for x in sample["vul-token-labels"]]
        sec_labels = [_to_cls(x) for x in sample["sec-token-labels"]]
        if len(vul_labels) != F:
            vul_labels = vul_labels[-F:] if F else []
        if len(sec_labels) != S:
            sec_labels = sec_labels[:S]
    elif cls_label_mode == "ast":
        lang = sample.get("language")
        if lang is not None:
            lang = _ast_labeling.LANGUAGE_ALIASES.get(lang, lang)
        else:
            lang = _ast_labeling.detect_language(sample["vul-code"])
        vul_labels, sec_labels = ast_token_diff_labels(
            sample["vul-code"], sample["sec-code"], tokenizer,
            vul_ids, sec_ids,
            language=lang,
            skip_modes=cls_label_skip_modes,
            align=cls_label_align,
            overlap_ratio=cls_label_overlap_ratio,
        )
        # ast labels cover the untruncated code; slice to the truncated ids
        if len(vul_labels) != F:
            vul_labels = vul_labels[-F:] if F else []
        if len(sec_labels) != S:
            sec_labels = sec_labels[:S]
    elif cls_label_align == "dp":
        # BPE ids + exact edit-distance DP (matches the reward evaluator).
        vul_labels, sec_labels = dp_diff_labels(vul_ids, sec_ids)
    else:
        vul_labels, sec_labels = token_diff_labels(vul_ids, sec_ids)
    vul_t = torch.tensor(vul_labels, dtype=torch.long)
    sec_t = torch.tensor(sec_labels, dtype=torch.long)

    rows = torch.arange(T).unsqueeze(1)  # (T,1)
    cols = torch.arange(T).unsqueeze(0)  # (1,T)
    causal = cols <= rows
    if interleave:
        # chunk-interleaved 布局（PIC+EPL 的 vLLM 一致实现，server 端 --interleave）：
        #   b0 <v1> b1 <v2> ... b_{N-1} <vN> [vulcode] <secu> [seccode] eos
        # input 均匀切 N 块，<vuln_i> 紧跟块 i 末尾。标准 causal 掩码下每个
        # <vuln_i> 渐进吸收 chunk0..i（PIC 局部感受野），块位置即位置对应
        # （EPL），训练/推理掩码天然一致；vulcode 行仍看不到 input（强制压缩）。
        # 注意：interleave 强制 <vuln1..N>，且与 qformer/icae 互斥（按连续
        # vuln 段定位的组件在交错布局下会错位）。
        inv = [tokenizer.convert_tokens_to_ids(t) for t in VULN_TOKS[:N]]
        chunk = max(1, -(-L // N))  # ceil(L/N)
        blocks = [inp_ids[i * chunk : (i + 1) * chunk] for i in range(N)]
        vpos, bpos, off = [], [], 0
        for b in blocks:
            bpos.append(off)
            off += len(b)
            vpos.append(off)
            off += 1
        ids = []
        for b, v in zip(blocks, inv):
            ids.extend(b)
            ids.append(v)
        input_ids = torch.tensor(ids + vul_ids + [secu_id] + sec_ids + [eos_id],
                                 dtype=torch.long)
        # 边界：vuln 段末 = <vN> 后一个位置（vulcode 起点）
        vuln_end_last = vpos[-1] + 1
        inp_end = vuln_end_last
        vuln_end = vuln_end_last
        vul_end = vuln_end_last + F
        secu_idx = vul_end
        sec_end = secu_idx + 1 + S
        eos_idx = T - 1
        # 掩码：默认即标准 causal（训练/推理一致）；仅两处调整 ——
        # vulcode 行看不到 input（强制压缩），<secu> 之后挖掉 vulcode 列。
        inp_cols = torch.zeros(T, dtype=torch.bool)
        for s, e in zip(bpos, vpos):
            inp_cols[s:e] = True
        vucols = torch.zeros(T, dtype=torch.bool)
        vucols[vuln_end_last:vul_end] = True
        mask = causal
        mask[vuln_end_last:vul_end] = causal[vuln_end_last:vul_end] & ~inp_cols
        mask[secu_idx:] = causal[secu_idx:] & ~vucols
        # RoPE 位置：<secu> 起紧贴 vuln 段末（跳过 vulcode），与推理一致
        position_ids = torch.arange(T, dtype=torch.long)
        position_ids[secu_idx : eos_idx + 1] = torch.arange(
            vuln_end_last, vuln_end_last + S + 2, dtype=torch.long
        )
        # Loss 权重：vulcode 段 aux、seccode+eos 主 loss（与默认布局相同）
        loss_mask = torch.zeros(T, dtype=torch.float)
        loss_mask[vuln_end_last:vul_end] = vulcode_loss_weight
        loss_mask[secu_idx + 1 : eos_idx + 1] = 1.0
    else:
        # 隐式思维链布局（用户方案 #3/#4/#5；think_mode 训练级开关，样本含
        # think/func_anal 字段才启用，否则走默认布局——general 样本混合训练）。
        # 注意：think 分支与 func_anal 分支一样必须 return（否则继续执行会
        # 落入 judge/默认布局覆盖 input_ids/mask）。
        if think_mode in ("plain", "block", "triple", "front") and (
                ("think" in sample) if think_mode != "triple"
                else ("func_anal" in sample)):
            if think_mode == "plain":
                (input_ids, mask, position_ids, loss_mask, T,
                 vuln_end, vul_end, secu_idx, sec_end, L) = _build_think_plain(
                    inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                    max_length, vulcode_loss_weight)
            elif think_mode == "front":
                (input_ids, mask, position_ids, loss_mask, T,
                 vuln_end, vul_end, secu_idx, sec_end, L) = _build_think_front(
                    inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                    max_length, vulcode_loss_weight)
            elif think_mode == "block":
                (input_ids, mask, position_ids, loss_mask, T,
                 vuln_end, vul_end, secu_idx, sec_end, L) = _build_think_block(
                    inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                    max_length, vulcode_loss_weight)
            else:
                (input_ids, mask, position_ids, loss_mask, T,
                 vuln_end, vul_end, secu_idx, sec_end, L) = _build_think_triple(
                    inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                    max_length, vulcode_loss_weight)
            cls_labels = torch.full((T,), CLS_IGNORE, dtype=torch.long)
            cls_weights = torch.zeros(T, dtype=torch.float)
            sec_t_cut = sec_t[:sec_end - secu_idx - 1]  # think 布局截断后的 seccode 标签
            cls_labels[vuln_end:vul_end] = vul_t
            cls_labels[secu_idx + 1:sec_end] = sec_t_cut
            cls_weights[vuln_end:vul_end] = torch.where(
                vul_t == CLS_NEUTRAL, cls_neutral_weight, 1.0)
            cls_weights[secu_idx + 1:sec_end] = torch.where(
                sec_t_cut == CLS_NEUTRAL, cls_neutral_weight, 1.0)
            if keep_up > 1.0:
                keep = sec_t_cut == CLS_NEUTRAL
                loss_mask[secu_idx + 1:sec_end] *= torch.where(keep, keep_up, 1.0)
            return input_ids, mask, position_ids, loss_mask, cls_labels, cls_weights, L
        if "func_anal" in sample:
            # 三段式思维链（样本带 func_anal 字段即启用；与 interleave/judge 互斥）
            (input_ids, mask, position_ids, loss_mask, T,
             vuln_end, vul_end, secu_idx, sec_end, L) = _build_three_stage(
                inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                max_length, vulcode_loss_weight, multi_vuln,
                vuln_anal_vis, seccode_cond)
            cls_labels = torch.full((T,), CLS_IGNORE, dtype=torch.long)
            cls_weights = torch.zeros(T, dtype=torch.float)
            sec_t_cut = sec_t[:sec_end - secu_idx - 1]  # 三段式截断后的 seccode 标签
            cls_labels[vuln_end:vul_end] = vul_t
            cls_labels[secu_idx + 1 : sec_end] = sec_t_cut
            cls_weights[vuln_end:vul_end] = torch.where(
                vul_t == CLS_NEUTRAL, cls_neutral_weight, 1.0)
            cls_weights[secu_idx + 1 : sec_end] = torch.where(
                sec_t_cut == CLS_NEUTRAL, cls_neutral_weight, 1.0)
            if keep_up > 1.0:
                keep = sec_t_cut == CLS_NEUTRAL
                loss_mask[secu_idx + 1 : sec_end] *= torch.where(keep, keep_up, 1.0)
            return input_ids, mask, position_ids, loss_mask, cls_labels, cls_weights, L
        # 两段条件生成（judge）：`<secu>` 前插判定 token —— vulcode 段含
        # UNSAFE 标签 → <jfix>（需要修复），否则 <jkeep>。训练用真实标签；
        # 推理侧固定 <jfix>（btoks 只用于修复任务）。
        if judge:
            judge_id = tokenizer.convert_tokens_to_ids(
                JFIX_TOK if (vul_t == CLS_UNSAFE).any() else JKEEP_TOK)
            j_ids = [judge_id]
        else:
            j_ids = []
        input_ids = torch.tensor(
            inp_ids + vuln_ids + vul_ids + j_ids + [secu_id] + sec_ids + [eos_id],
            dtype=torch.long,
        )
        Tj = input_ids.shape[0]

        # Segment boundaries (indices in the spliced sequence).
        # inp [0,L) | vuln [L,L+N) | vulcode [L+N,L+N+F) | [judge] | secu | seccode | eos
        inp_end = L
        vuln_end = L + N
        vul_end = L + N + F
        j_idx = vul_end if judge else None
        secu_idx = vul_end + (1 if judge else 0)
        sec_end = secu_idx + 1 + S
        eos_idx = Tj - 1

        # 4D-style attention mask (True = attend), causal within each block.
        mask = torch.zeros((Tj, Tj), dtype=torch.bool)
        rows_j = torch.arange(Tj).unsqueeze(1)
        cols_j = torch.arange(Tj).unsqueeze(0)
        causal_j = cols_j <= rows_j
        base = ((cols_j < inp_end)
                | ((cols_j >= inp_end) & (cols_j <= vuln_end - 1)))
        if judge:
            base = base | (cols_j == j_idx)  # judge 是生成条件信号，seccode 可看

        # input rows: causal within input
        mask[:inp_end] = causal_j[:inp_end] & (cols_j <= inp_end - 1)
        # vuln rows: input + vuln prefix
        mask[inp_end:vuln_end] = causal_j[inp_end:vuln_end] & (cols_j <= vuln_end - 1)
        # vulcode rows: vuln + vulcode prefix (input blocked -> compression).
        # vulcode_see_input=True: 允许 vulcode 看 input（消融：压缩是否必需）
        if vulcode_see_input:
            mask[vuln_end:vul_end] = causal_j[vuln_end:vul_end]
        else:
            mask[vuln_end:vul_end] = (cols_j >= inp_end) & causal_j[vuln_end:vul_end]
        # judge row（若启用）: input + vuln + self（与 <secu> 同理）
        if judge:
            mask[j_idx] = base & (cols_j == j_idx) | (cols_j < inp_end) \
                | ((cols_j >= inp_end) & (cols_j <= vuln_end - 1)) | (cols_j == j_idx)
        # <secu> row: input + vuln + (judge) + self (matches inference prefill)
        mask[secu_idx] = base | (cols_j == secu_idx)
        # seccode rows: input + vuln + (judge) + <secu> + seccode prefix (vulcode blocked)
        mask[secu_idx + 1 : sec_end] = (
            base | (cols_j == secu_idx)
            | ((cols_j >= secu_idx + 1) & causal_j[secu_idx + 1 : sec_end])
        )
        # eos row: input + vuln + (judge) + <secu> + full seccode (still no vulcode)
        mask[eos_idx] = (
            base | (cols_j == secu_idx)
            | ((cols_j >= secu_idx + 1) & (cols_j <= sec_end - 1))
        )
        # seccode_see_vulcode=True: 消融放开 seccode/eos 行对 vulcode 列的封锁
        # (对称变体: 训练 vulcode+seccode 都优化 NTP, 推理也是 input+<vuln> 先生成
        # vulcode 再 <secu>+seccode 两段式; <secu> 行保持原样)
        if seccode_see_vulcode:
            vucols_r = (cols_j >= vuln_end) & (cols_j < vul_end)
            mask[secu_idx + 1 : sec_end] = mask[secu_idx + 1 : sec_end] | vucols_r
            mask[eos_idx] = mask[eos_idx] | vucols_r

        # RoPE positions: judge 落在 vuln 块末（vuln_end），<secu> 及之后紧贴其后
        # （vulcode 跳过），与推理 prefill（input+vuln*N+[judge]+<secu>）一致。
        # seccode_see_vulcode=True: 不跳 —— 两段式推理 stage-2 的 prefill 里
        # vulcode 是实 token（自然位置 L+N..L+N+F-1），<secu> 落在 L+N+F;
        # 回卷会让 seccode→input/<vuln>/vulcode 的相对距离整体错位 F, 保持自然
        # 位置才与训练逐位一致（champion/A 的 chat 推理无 vulcode 才需要回卷）。
        position_ids = torch.arange(Tj, dtype=torch.long)
        pos_start = vuln_end + (1 if judge else 0)
        if not seccode_see_vulcode:
            position_ids[secu_idx : eos_idx + 1] = torch.arange(
                pos_start, pos_start + S + 2, dtype=torch.long
            )
            if judge:
                position_ids[j_idx] = vuln_end
        # seccode_see_vulcode 下 judge 保持自然位置（vul_end）

        # Loss weights (target-aligned): vulcode tokens are an auxiliary loss
        # (vulcode_loss_weight, default 0.5); seccode tokens and eos are the main
        # loss (1.0). input / vuln / <secu> / judge are never trained.
        loss_mask = torch.zeros(Tj, dtype=torch.float)
        loss_mask[vuln_end : vul_end] = vulcode_loss_weight
        loss_mask[secu_idx + 1 : eos_idx + 1] = 1.0
        T = Tj  # 供后续标签放置使用

    cls_labels = torch.full((T,), CLS_IGNORE, dtype=torch.long)
    cls_weights = torch.zeros(T, dtype=torch.float)
    cls_labels[vuln_end:vul_end] = vul_t
    cls_labels[secu_idx + 1 : sec_end] = sec_t
    cls_weights[vuln_end:vul_end] = torch.where(
        vul_t == CLS_NEUTRAL, cls_neutral_weight, 1.0
    )
    cls_weights[secu_idx + 1 : sec_end] = torch.where(
        sec_t == CLS_NEUTRAL, cls_neutral_weight, 1.0
    )

    # PRepair 最小编辑（keep_up>1.0）：seccode 中「保留」token（未变化部分）
    # 的 NLL 权重上调，鼓励模型少改写、只做最小安全编辑。
    if keep_up > 1.0:
        keep = sec_t == CLS_NEUTRAL
        loss_mask[secu_idx + 1 : sec_end] *= torch.where(keep, keep_up, 1.0)

    return input_ids, mask, position_ids, loss_mask, cls_labels, cls_weights, L


def _build_three_stage(inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                       max_length, vulcode_loss_weight, multi_vuln,
                       vuln_anal_vis, seccode_cond):
    """三段式思维链布局（L 方案）。样本含 func_anal 字段时由 _process_sample 调用。

    序列: inp <func_anal> [FA] <vuln>*N vulcode <vuln_anal> [VA]
          <secu_impl> [SI] <secu> seccode eos
    FA/VA/SI 均为生成目标（loss 1.0）。vulcode 行仍看不到 input+FA（强制压缩）。
    - vuln_anal_vis=1: VA 段可看 vulcode 原文（训练有、推理无——不对称对照）；
      =0: VA 只靠 <vuln> 压缩状态（与推理一致）。
    - seccode_cond=1: <secu>/seccode/eos 行挖掉 FA/VA/SI 全部文本列，只留
      <func_anal>/<vuln_anal>/<secu_impl> 三个 token 作条件信号（类似 judge）。
    SI 行两种开关下均看不到 vulcode（VA 已显式分析，规划基于分析结果）。
    返回 (input_ids, mask, position_ids, loss_mask, T, vuln_end, vul_end,
           secu_idx, sec_end, L)。sec_ids 可能被截断（S 由 secu_idx/sec_end 隐含）。
    """
    fa_ids = tokenizer(sample["func_anal"], add_special_tokens=False)["input_ids"]
    va_ids = tokenizer(sample["vuln_anal"], add_special_tokens=False)["input_ids"]
    si_ids = tokenizer(sample["secu_impl"], add_special_tokens=False)["input_ids"]
    fa_tok = tokenizer.convert_tokens_to_ids(FUNC_ANAL_TOK)
    va_tok = tokenizer.convert_tokens_to_ids(VULN_ANAL_TOK)
    si_tok = tokenizer.convert_tokens_to_ids(SECU_IMPL_TOK)
    secu_id = tokenizer.convert_tokens_to_ids(SECU_TOK)
    eos_id = tokenizer.eos_token_id
    N, F = n_vuln, len(vul_ids)
    # 截断优先级：seccode 尾 → SI 尾 → VA 尾 → FA 尾 → input 头（骨架必留）
    T = len(inp_ids) + 1 + len(fa_ids) + N + F + 1 + len(va_ids) + 1 + len(si_ids) + 1 + len(sec_ids) + 1
    if T > max_length:
        for ids in (sec_ids, si_ids, va_ids, fa_ids):
            over = T - max_length
            if over <= 0:
                break
            cut = min(len(ids), over)
            if cut:
                del ids[len(ids) - cut:]
            T = (len(inp_ids) + 1 + len(fa_ids) + N + F + 1 + len(va_ids)
                 + 1 + len(si_ids) + 1 + len(sec_ids) + 1)
        if T > max_length:
            over = T - max_length
            if over > len(inp_ids):
                raise RuntimeError("三段式骨架超 max_length")
            inp_ids = inp_ids[over:]
            T = (len(inp_ids) + 1 + len(fa_ids) + N + F + 1 + len(va_ids)
                 + 1 + len(si_ids) + 1 + len(sec_ids) + 1)
    L, A, VA, SI, S = len(inp_ids), len(fa_ids), len(va_ids), len(si_ids), len(sec_ids)
    vuln_ids = ([tokenizer.convert_tokens_to_ids(t) for t in VULN_TOKS[:N]]
                if multi_vuln else [tokenizer.convert_tokens_to_ids(VULN_TOK)] * N)
    input_ids = torch.tensor(
        inp_ids + [fa_tok] + fa_ids + vuln_ids + vul_ids +
        [va_tok] + va_ids + [si_tok] + si_ids + [secu_id] + sec_ids + [eos_id],
        dtype=torch.long)
    # 段边界
    fa_idx = L
    fa_end = L + 1 + A
    vuln_end = fa_end + N
    vul_end = vuln_end + F
    va_idx = vul_end
    va_end = vul_end + 1 + VA
    si_idx = va_end
    si_end = va_end + 1 + SI
    secu_idx = si_end
    sec_end = secu_idx + 1 + S
    eos_idx = T - 1
    # 掩码（True=attend）：causal 基准
    rows_j = torch.arange(T).unsqueeze(1)
    cols_j = torch.arange(T).unsqueeze(0)
    causal_j = cols_j <= rows_j
    vucols = (cols_j >= vuln_end) & (cols_j < vul_end)   # vulcode 列
    fatxt = (cols_j > fa_idx) & (cols_j < fa_end)        # FA 文本列
    vatxt = (cols_j > va_idx) & (cols_j < va_end)        # VA 文本列
    sitxt = (cols_j > si_idx) & (cols_j < si_end)        # SI 文本列
    mask = causal_j.clone()
    # vulcode 行：vuln+vulcode 前缀（挖 input+FA，强制压缩）
    mask[vuln_end:vul_end] = causal_j[vuln_end:vul_end] & (cols_j >= fa_end)
    # VA 行：vis=0 时挖 vulcode（只靠压缩状态）；vis=1 全 causal（看原文）
    if not vuln_anal_vis:
        mask[va_idx:va_end] = causal_j[va_idx:va_end] & ~vucols
    # SI/<secu>/seccode/eos 行：一律看不到 vulcode
    mask[si_idx:] = causal_j[si_idx:] & ~vucols
    if seccode_cond:
        # <secu> 及之后：再挖掉 FA/VA/SI 三段文本列，只留三个 token 作信号
        mask[secu_idx:] = mask[secu_idx:] & ~fatxt & ~vatxt & ~sitxt
    # 位置：自然 arange（FA/VA/SI 真实文本连续占位；vulcode 也占位）
    position_ids = torch.arange(T, dtype=torch.long)
    # 损失：FA/VA/SI/seccode+eos 主 loss(1.0)，vulcode aux（target-aligned）
    loss_mask = torch.zeros(T, dtype=torch.float)
    loss_mask[fa_idx + 1:fa_end] = 1.0
    loss_mask[vuln_end:vul_end] = vulcode_loss_weight
    loss_mask[va_idx + 1:va_end] = 1.0
    loss_mask[si_idx + 1:si_end] = 1.0
    loss_mask[secu_idx + 1:eos_idx + 1] = 1.0
    return (input_ids, mask, position_ids, loss_mask, T,
            vuln_end, vul_end, secu_idx, sec_end, L)


# ---------------------------------------------------------------------------
# 隐式思维链（think）布局（用户方案 #3/#4/#5）。
# 统一设计原则：推理 prefill 中存在的块（input、think 摘要 token、<vuln>*N、
# <secu>）构成 seccode 的 base 可见集（去掉训练专属块后 mask 退化为标准
# causal，与 vLLM 推理一致）；训练专属块（think 文本、vulcode）只被自身重建
# 行可见，且只 attend 其摘要 token。RoPE 位置：think 文本/vulcode 用训练连续
# 位置，摘要 token/<vuln>/<secu> 回卷到推理 prefill 位置（<vuln> 块位置 =
# L + think 块长，<secu> 位置 = L + think 块长 + N）。
# ---------------------------------------------------------------------------

def _think_truncate(inp_ids, vul_ids, sec_ids, thk_idss, tokenizer, n_vuln,
                    max_length, order):
    """think 布局截断：按 order 优先级（seccode → think 文本 → input）切尾。
    骨架（think token 块 + <vuln>*N + vulcode + <secu> + eos）必留。
    返回 (inp_ids, sec_ids, thk_idss)。"""
    N, F = n_vuln, len(vul_ids)
    N, F = n_vuln, len(vul_ids)

    def total():
        return len(inp_ids) + sum(len(x) for x in thk_idss) + N + F + 1 \
            + len(sec_ids) + 1

    T = total()
    if T <= max_length:
        return inp_ids, sec_ids, thk_idss
    for ids in ([sec_ids] + thk_idss + [inp_ids]):
        over = T - max_length
        if over <= 0:
            break
        cut = min(len(ids), over)
        if cut:
            del ids[len(ids) - cut:]
        T = total()
    if T > max_length:
        over = T - max_length
        if over > len(inp_ids):
            raise RuntimeError("think 骨架超 max_length")
        inp_ids = inp_ids[over:]
    return inp_ids, sec_ids, thk_idss


def _build_think_plain(inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                       max_length, vulcode_loss_weight):
    """#3 显式思维链基线：input + think + <vuln>*N + vulcode + <secu> + seccode，
    全 causal（普通掩码注意力）。think 文本是显式输入（推理时由 doubao 分析
    注入），只比默认布局多一个 think 段（loss 0.5）。
    RoPE：think/<vuln> 自然位置（推理也存在），vulcode 占位，<secu> 回卷到
    L + len(think) + N（推理 prefill 中 <secu> 的位置）。"""
    thk_ids = tokenizer(sample["think"], add_special_tokens=False)["input_ids"]
    secu_id = tokenizer.convert_tokens_to_ids(SECU_TOK)
    eos_id = tokenizer.eos_token_id
    N, F = n_vuln, len(vul_ids)
    inp_ids, sec_ids, thk_idss = _think_truncate(
        inp_ids, vul_ids, sec_ids, [thk_ids], tokenizer, n_vuln, max_length, "text")
    thk_ids = thk_idss[0]
    L, A, S = len(inp_ids), len(thk_ids), len(sec_ids)
    vuln_id = tokenizer.convert_tokens_to_ids(VULN_TOK)
    vuln_ids = [vuln_id] * N
    input_ids = torch.tensor(
        inp_ids + thk_ids + vuln_ids + vul_ids + [secu_id] + sec_ids + [eos_id],
        dtype=torch.long)
    T = input_ids.shape[0]
    inp_end, thk_end = L, L + A
    vuln_end, vul_end = thk_end + N, thk_end + N + F
    secu_idx, sec_end, eos_idx = vul_end, vul_end + 1 + S, T - 1
    # 全 causal（普通掩码注意力）
    mask = torch.arange(T).unsqueeze(1) >= torch.arange(T).unsqueeze(0)
    position_ids = torch.arange(T, dtype=torch.long)
    pos_start = L + A + N
    position_ids[secu_idx:eos_idx + 1] = torch.arange(
        pos_start, pos_start + S + 2, dtype=torch.long)
    loss_mask = torch.zeros(T, dtype=torch.float)
    loss_mask[thk_end - A:thk_end] = vulcode_loss_weight  # think 段 0.5
    loss_mask[vuln_end:vul_end] = vulcode_loss_weight
    loss_mask[secu_idx + 1:eos_idx + 1] = 1.0
    return (input_ids, mask, position_ids, loss_mask, T,
            vuln_end, vul_end, secu_idx, sec_end, L)


def _build_think_block(inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                       max_length, vulcode_loss_weight):
    """#4 <think>*4 双层瓶颈：input + <think>*4 + think + <vuln>*N + vulcode
    + <secu> + seccode。think 文本只 attend <think>*4 块（从 4-token 摘要重建，
    loss 0.5）；<vuln> 行 attend input+<think>*4（推理 prefill 一致）；
    vulcode 行只 attend <think>*4+<vuln>*4（不含 input/think 文本）；
    <secu>/seccode 行 base = input+<think>*4+<vuln>*4（推理 causal 一致）。
    RoPE：<think>*4 = L..L+3；think 文本训练连续；<vuln> 回卷 L+4..L+4+N-1；
    vulcode 从 L+4+N 起；<secu> = L+4+N。"""
    thk_ids = tokenizer(sample["think"], add_special_tokens=False)["input_ids"]
    secu_id = tokenizer.convert_tokens_to_ids(SECU_TOK)
    eos_id = tokenizer.eos_token_id
    N, F = n_vuln, len(vul_ids)
    inp_ids, sec_ids, thk_idss = _think_truncate(
        inp_ids, vul_ids, sec_ids, [thk_ids], tokenizer, n_vuln, max_length, "text")
    thk_ids = thk_idss[0]
    L, A, S = len(inp_ids), len(thk_ids), len(sec_ids)
    vuln_id = tokenizer.convert_tokens_to_ids(VULN_TOK)
    thk_tok = tokenizer.convert_tokens_to_ids(THINK_TOK)
    vuln_ids = [vuln_id] * N
    input_ids = torch.tensor(
        inp_ids + [thk_tok] * 4 + thk_ids + vuln_ids + vul_ids + [secu_id]
        + sec_ids + [eos_id], dtype=torch.long)
    T = input_ids.shape[0]
    inp_end, thk_end = L, L + 4
    thk_txt_end = thk_end + A
    vuln_end, vul_end = thk_txt_end + N, thk_txt_end + N + F
    secu_idx, sec_end, eos_idx = vul_end, vul_end + 1 + S, T - 1
    rows_j = torch.arange(T).unsqueeze(1)
    cols_j = torch.arange(T).unsqueeze(0)
    causal_j = cols_j <= rows_j
    mask = torch.zeros((T, T), dtype=torch.bool)
    # input rows: causal
    mask[:inp_end] = causal_j[:inp_end] & (cols_j <= inp_end - 1)
    # <think>*4 rows: causal（input + <think> 前缀）
    mask[inp_end:thk_end] = causal_j[inp_end:thk_end] & (cols_j <= thk_end - 1)
    # think rows: <think>*4 + think 前缀（input 被 block）
    mask[thk_end:thk_txt_end] = (cols_j >= inp_end) & (cols_j < thk_txt_end) \
        & causal_j[thk_end:thk_txt_end]
    # <vuln> rows: input + <think>*4 + <vuln> 前缀（无 think 文本）
    mask[thk_txt_end:vuln_end] = (cols_j < thk_end) \
        | ((cols_j >= thk_txt_end) & causal_j[thk_txt_end:vuln_end])
    # vulcode rows: <think>*4 + <vuln>*4 + 自身前缀（无 input/think 文本）
    mask[vuln_end:vul_end] = ((cols_j >= inp_end) & (cols_j < thk_end)) \
        | ((cols_j >= thk_txt_end) & causal_j[vuln_end:vul_end])
    # <secu> row: input + <think>*4 + <vuln>*4 + self（推理 prefill）
    mask[secu_idx] = (cols_j < thk_end) \
        | ((cols_j >= thk_txt_end) & (cols_j < vuln_end)) | (cols_j == secu_idx)
    # seccode rows: base + <secu> + seccode 前缀
    mask[secu_idx + 1:sec_end] = (cols_j < thk_end) \
        | ((cols_j >= thk_txt_end) & (cols_j < vuln_end)) | (cols_j == secu_idx) \
        | ((cols_j >= secu_idx + 1) & causal_j[secu_idx + 1:sec_end])
    # eos row: base + <secu> + 全部 seccode
    mask[eos_idx] = (cols_j < thk_end) \
        | ((cols_j >= thk_txt_end) & (cols_j < vuln_end)) | (cols_j == secu_idx) \
        | ((cols_j >= secu_idx + 1) & (cols_j <= sec_end - 1))
    # RoPE：think 文本训练连续；<vuln>/<secu> 回卷推理位置
    position_ids = torch.arange(T, dtype=torch.long)
    position_ids[thk_end:thk_txt_end] = torch.arange(  # think 文本 = <think> 后
        thk_end, thk_end + A, dtype=torch.long)
    pos_start = L + 4 + N  # 推理 prefill 中 <secu> 的位置
    position_ids[thk_txt_end:vuln_end] = torch.arange(  # <vuln> 回卷
        L + 4, L + 4 + N, dtype=torch.long)
    position_ids[vuln_end:vul_end] = torch.arange(  # vulcode：推理首个输出位置起
        pos_start, pos_start + F, dtype=torch.long)
    position_ids[secu_idx:eos_idx + 1] = torch.arange(
        pos_start, pos_start + S + 2, dtype=torch.long)
    loss_mask = torch.zeros(T, dtype=torch.float)
    loss_mask[thk_end:thk_txt_end] = vulcode_loss_weight  # think 0.5
    loss_mask[vuln_end:vul_end] = vulcode_loss_weight
    loss_mask[secu_idx + 1:eos_idx + 1] = 1.0
    return (input_ids, mask, position_ids, loss_mask, T,
            vuln_end, vul_end, secu_idx, sec_end, L)


def _build_think_front(inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                       max_length, vulcode_loss_weight, think_loss_weight=None,
                       vulcode_see_think=False):
    """链前置布局（用户方案 09-10）：``inp + <think> 链 </think> + <vuln>*N
    + vulcode + <secu> + seccode``。与 cotlbl 的差别只在链的位置——链从
    「seccode 段前缀」移到「input 侧」，于是生成顺序变成 链 → 压缩 →
    (盲)vulcode → 代码。链是生成目标（loss think_loss_weight，默认同
    vulcode 的 0.5）。

    掩码（用户口径 09-10）::

        seccode 行: input + think + <vuln>*N + <secu> + 自身前缀，**挖 vulcode 列**
        <secu>  行: 同上（不含 seccode）
        vulcode 行: 只 <vuln>*N + 自身前缀（挖 input/think）
                     —— 保持冠军的盲生成语义；--vulcode_see_think 可放开
        think   行: causal（链由 input 生成）
        <vuln>  行: causal（摘要由 input+think 生成）

    RoPE：inp/think/<vuln> 取自然位置；vulcode 占位 pos_start=L+A+N；
    <secu>+seccode+eos 回卷到 pos_start 起——即推理 prefill 中 <secu> 的位
    （推理没有 vulcode 段，故训练必须回卷才与推理逐位一致）。

    推理（两段式，server--think_front）：prompt → 生成 ``<think>链</think>``
    → harness 追加 ``<vuln>*N <secu>`` → 生成 seccode。"""
    if think_loss_weight is None:
        think_loss_weight = vulcode_loss_weight
    thk_ids = tokenizer(sample["think"], add_special_tokens=False)["input_ids"]
    # 与 cotlbl 数据同构的段文本：<think>\n链\n</think>\n\n（<think> 是特殊
    # token，tokenize 时按 vocab 映射到 151670；</think> 是普通文本多 token）
    thk_seg = ([tokenizer.convert_tokens_to_ids(THINK_TOK)]
               + thk_ids + tokenizer("</think>\n\n", add_special_tokens=False)["input_ids"])
    secu_id = tokenizer.convert_tokens_to_ids(SECU_TOK)
    eos_id = tokenizer.eos_token_id
    N, F = n_vuln, len(vul_ids)
    inp_ids, sec_ids, thk_idss = _think_truncate(
        inp_ids, vul_ids, sec_ids, [thk_seg], tokenizer, n_vuln, max_length, "text")
    thk_seg = thk_idss[0]
    L, A, S = len(inp_ids), len(thk_seg), len(sec_ids)
    vuln_id = tokenizer.convert_tokens_to_ids(VULN_TOK)
    vuln_ids = [vuln_id] * N
    input_ids = torch.tensor(
        inp_ids + thk_seg + vuln_ids + vul_ids + [secu_id] + sec_ids + [eos_id],
        dtype=torch.long)
    T = input_ids.shape[0]
    inp_end = L
    thk_end, thk_txt_end = inp_end + 1, inp_end + A   # <think> 与链文本/闭标签
    vuln_end = thk_txt_end + N
    vul_end = vuln_end + F
    secu_idx, sec_end, eos_idx = vul_end, vul_end + 1 + S, T - 1

    rows_j = torch.arange(T).unsqueeze(1)
    cols_j = torch.arange(T).unsqueeze(0)
    causal_j = cols_j <= rows_j

    mask = torch.zeros((T, T), dtype=torch.bool)
    mask[:inp_end] = causal_j[:inp_end]
    mask[inp_end:thk_txt_end] = causal_j[inp_end:thk_txt_end]        # think: 全 causal
    mask[thk_txt_end:vuln_end] = causal_j[thk_txt_end:vuln_end]      # <vuln>: 全 causal
    # vulcode: 只 <vuln> 块 + 自身前缀（盲生成，看不到 input/think）
    base_vu = (cols_j >= thk_txt_end) & (cols_j < vuln_end)
    own_vu = (cols_j >= vuln_end) & causal_j[vuln_end:vul_end]
    if vulcode_see_think:
        mask[vuln_end:vul_end] = causal_j[vuln_end:vul_end]   # 放开: 全 causal
    else:
        mask[vuln_end:vul_end] = base_vu | own_vu
    # <secu>/seccode/eos: input + think + <vuln> 块 + <secu> + 自身前缀（挖 vulcode）
    keep_cols = (cols_j < vuln_end)          # input/think/<vuln> 全段
    mask[secu_idx] = keep_cols | (cols_j == secu_idx)
    mask[secu_idx + 1:sec_end] = keep_cols | (cols_j == secu_idx) \
        | ((cols_j >= secu_idx + 1) & causal_j[secu_idx + 1:sec_end])
    mask[eos_idx] = keep_cols | (cols_j == secu_idx) \
        | ((cols_j >= secu_idx + 1) & (cols_j <= sec_end - 1))

    # RoPE：自然位置到 <vuln> 段末；vulcode 占位；<secu> 起回卷到推理 prefill 位
    position_ids = torch.arange(T, dtype=torch.long)
    pos_start = L + A + N
    position_ids[vuln_end:vul_end] = torch.arange(pos_start, pos_start + F)
    position_ids[secu_idx:eos_idx + 1] = torch.arange(
        pos_start, pos_start + S + 2, dtype=torch.long)
    loss_mask = torch.zeros(T, dtype=torch.float)
    loss_mask[inp_end:thk_txt_end] = think_loss_weight    # 链（生成目标）
    loss_mask[vuln_end:vul_end] = vulcode_loss_weight     # vulcode 辅助
    loss_mask[secu_idx + 1:eos_idx + 1] = 1.0             # seccode + eos 主 loss
    return (input_ids, mask, position_ids, loss_mask, T,
            vuln_end, vul_end, secu_idx, sec_end, L)


def _build_think_triple(inp_ids, vul_ids, sec_ids, sample, tokenizer, n_vuln,
                        max_length, vulcode_loss_weight):
    """#5 三段 think token：input + <think_func_anal> + func_anal +
    <think_vuln_anal> + vuln_anal + <think_secu_impl> + secu_impl + <vuln>*N +
    vulcode + <secu> + seccode。三个 think token 各自 attend input（独立摘要）；
    每段文本只 attend 自己的 token（各 loss 0.5）；vulcode 只 attend 三 think
    token + <vuln>*N；<secu>/seccode base = input + 三 think token + <vuln>*N
    （推理 causal 一致）。RoPE：三 think token = L..L+2；文本训练连续；
    <vuln> 回卷 L+3..；<secu> = L+3+N。"""
    fa_ids = tokenizer(sample["func_anal"], add_special_tokens=False)["input_ids"]
    va_ids = tokenizer(sample["vuln_anal"], add_special_tokens=False)["input_ids"]
    si_ids = tokenizer(sample["secu_impl"], add_special_tokens=False)["input_ids"]
    secu_id = tokenizer.convert_tokens_to_ids(SECU_TOK)
    eos_id = tokenizer.eos_token_id
    N, F = n_vuln, len(vul_ids)
    inp_ids, sec_ids, thk_idss = _think_truncate(
        inp_ids, vul_ids, sec_ids, [fa_ids, va_ids, si_ids], tokenizer,
        n_vuln, max_length, "text")
    fa_ids, va_ids, si_ids = thk_idss
    L, A, VA, SI, S = (len(inp_ids), len(fa_ids), len(va_ids),
                       len(si_ids), len(sec_ids))
    vuln_id = tokenizer.convert_tokens_to_ids(VULN_TOK)
    tfa = tokenizer.convert_tokens_to_ids(THINK_FUNC_ANAL_TOK)
    tva = tokenizer.convert_tokens_to_ids(THINK_VULN_ANAL_TOK)
    tsi = tokenizer.convert_tokens_to_ids(THINK_SECU_IMPL_TOK)
    vuln_ids = [vuln_id] * N
    input_ids = torch.tensor(
        inp_ids + [tfa] + fa_ids + [tva] + va_ids + [tsi] + si_ids
        + vuln_ids + vul_ids + [secu_id] + sec_ids + [eos_id], dtype=torch.long)
    T = input_ids.shape[0]
    # 段边界（训练序列）
    inp_end = L
    tfa_idx, fa_start, fa_end = L, L + 1, L + 1 + A
    tva_idx, va_start, va_end = fa_end, fa_end + 1, fa_end + 1 + VA
    tsi_idx, si_start, si_end = va_end, va_end + 1, va_end + 1 + SI
    vuln_end, vul_end = si_end + N, si_end + N + F
    secu_idx, sec_end, eos_idx = vul_end, vul_end + 1 + S, T - 1
    rows_j = torch.arange(T).unsqueeze(1)
    cols_j = torch.arange(T).unsqueeze(0)
    causal_j = cols_j <= rows_j
    # 三 think token 的序列 index（被文本隔开：L、fa_end、va_end）
    thk_idx = [tfa_idx, tva_idx, tsi_idx]
    thk_cols = ((cols_j == tfa_idx) | (cols_j == tva_idx) | (cols_j == tsi_idx))
    mask = torch.zeros((T, T), dtype=torch.bool)
    # input rows: causal
    mask[:inp_end] = causal_j[:inp_end] & (cols_j <= inp_end - 1)
    # 三 think token rows: input + 位置 ≤ 自己的 think token + self（无文本列）。
    # 注意用列 index 显式列举（tfa/tva/tsi 的 index 被文本隔开，不能用位置
    # 做 <= 比较——RoPE 位置 L/L+1/L+2 与 index 不一一对应）。
    mask[tfa_idx] = (cols_j < inp_end) | (cols_j == tfa_idx)
    mask[tva_idx] = (cols_j < inp_end) | (cols_j == tfa_idx) | (cols_j == tva_idx)
    mask[tsi_idx] = (cols_j < inp_end) | (cols_j == tfa_idx) \
        | (cols_j == tva_idx) | (cols_j == tsi_idx)
    # FA rows: 只 attend <think_func_anal> + FA 前缀
    mask[fa_start:fa_end] = (cols_j == tfa_idx) \
        | ((cols_j >= fa_start) & causal_j[fa_start:fa_end])
    # VA rows: 只 attend <think_vuln_anal> + VA 前缀
    mask[va_start:va_end] = (cols_j == tva_idx) \
        | ((cols_j >= va_start) & causal_j[va_start:va_end])
    # SI rows: 只 attend <think_secu_impl> + SI 前缀
    mask[si_start:si_end] = (cols_j == tsi_idx) \
        | ((cols_j >= si_start) & causal_j[si_start:si_end])
    # <vuln> rows: input + 三 think token + <vuln> 前缀（无文本）
    mask[si_end:vuln_end] = (cols_j < inp_end) | thk_cols \
        | ((cols_j >= si_end) & causal_j[si_end:vuln_end])
    # vulcode rows: 三 think token + <vuln>*N + 自身前缀
    mask[vuln_end:vul_end] = thk_cols \
        | ((cols_j >= si_end) & causal_j[vuln_end:vul_end])
    # <secu>/seccode/eos rows: input + 三 think token + <vuln>*N + <secu> + causal
    base = (cols_j < inp_end) | thk_cols | ((cols_j >= si_end) & (cols_j < vuln_end))
    mask[secu_idx] = base | (cols_j == secu_idx)
    mask[secu_idx + 1:sec_end] = base | (cols_j == secu_idx) \
        | ((cols_j >= secu_idx + 1) & causal_j[secu_idx + 1:sec_end])
    mask[eos_idx] = base | (cols_j == secu_idx) \
        | ((cols_j >= secu_idx + 1) & (cols_j <= sec_end - 1))
    # RoPE：三 think token = L..L+2（推理 prefill 位置，显式设置——文本段
    # 位置改写会覆盖它们）；三段文本训练连续（不与其他 token 冲突）；
    # <vuln> 回卷 L+3..；vulcode/<secu>/seccode 从 L+3+N 起（推理 prefill
    # 中 <secu> 的位置）。
    position_ids = torch.arange(T, dtype=torch.long)
    position_ids[tfa_idx] = L
    position_ids[tva_idx] = L + 1
    position_ids[tsi_idx] = L + 2
    position_ids[fa_start:fa_end] = torch.arange(
        inp_end + 3, inp_end + 3 + A, dtype=torch.long)
    position_ids[va_start:va_end] = torch.arange(
        inp_end + 3 + A, inp_end + 3 + A + VA, dtype=torch.long)
    position_ids[si_start:si_end] = torch.arange(
        inp_end + 3 + A + VA, inp_end + 3 + A + VA + SI, dtype=torch.long)
    pos_start = L + 3 + N
    position_ids[si_end:vuln_end] = torch.arange(
        L + 3, L + 3 + N, dtype=torch.long)  # <vuln> 回卷
    position_ids[vuln_end:vul_end] = torch.arange(
        pos_start, pos_start + F, dtype=torch.long)
    position_ids[secu_idx:eos_idx + 1] = torch.arange(
        pos_start, pos_start + S + 2, dtype=torch.long)
    loss_mask = torch.zeros(T, dtype=torch.float)
    loss_mask[fa_start:fa_end] = vulcode_loss_weight
    loss_mask[va_start:va_end] = vulcode_loss_weight
    loss_mask[si_start:si_end] = vulcode_loss_weight
    loss_mask[vuln_end:vul_end] = vulcode_loss_weight
    loss_mask[secu_idx + 1:eos_idx + 1] = 1.0
    return (input_ids, mask, position_ids, loss_mask, T,
            vuln_end, vul_end, secu_idx, sec_end, L)


def bottleneck_token_collate(
    batch, tokenizer, n_vuln, max_length, truncation, vulcode_loss_weight,
    cls_neutral_weight, cls_label_mode="ast", cls_label_skip_modes="comment+string",
    cls_label_align="difflib", cls_label_overlap_ratio=0.0,
    interleave=False, keep_up=1.0, judge=False,
    vuln_anal_vis=True, seccode_cond=False, think_mode="none",
):
    """Collate raw-string samples into a padded training batch.

    Returns a dict with input_ids (B,S), attention_mask (B,1,S,S) bool,
    position_ids (B,S), loss_mask (B,S) float, cls_labels (B,S) long,
    cls_weights (B,S) float.
    """
    processed = [
        _process_sample(
            s, tokenizer, n_vuln, max_length, truncation, vulcode_loss_weight,
            cls_neutral_weight, cls_label_mode, cls_label_skip_modes,
            cls_label_align, cls_label_overlap_ratio,
            multi_vuln=multi_vuln, interleave=interleave, keep_up=keep_up,
            judge=judge, vuln_anal_vis=vuln_anal_vis, seccode_cond=seccode_cond,
            think_mode=think_mode,
        )
        for s in batch
    ]
    max_seq = max(p[0].shape[0] for p in processed)
    B = len(processed)
    pad_id = tokenizer.pad_token_id

    # Attention mask is (B, 1, S, S): torch 2.5.x SDPA path in transformers
    # 4.57 needs an explicit broadcastable head dimension; (B, S, S) triggers
    # an invalid internal expand (5D -> 4D).
    input_ids = torch.full((B, max_seq), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, max_seq), dtype=torch.long)
    loss_mask = torch.zeros((B, max_seq), dtype=torch.float)
    attention_mask = torch.zeros((B, 1, max_seq, max_seq), dtype=torch.bool)
    cls_labels = torch.full((B, max_seq), CLS_IGNORE, dtype=torch.long)
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
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "cls_labels": cls_labels,
        "cls_weights": cls_weights,
        "inp_lens": inp_lens,
    }


def plot_loss_curve(metrics, output_dir):
    """Loss curves for overall / vulcode / seccode / cls, smoothed like verl's."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [
        "train/loss",
        "train/loss_vulcode",
        "train/loss_seccode",
        "train/cls_loss",
    ]
    colors = {
        "train/loss": "b",
        "train/loss_vulcode": "r",
        "train/loss_seccode": "g",
        "train/cls_loss": "m",
    }
    labels = {
        "train/loss": "overall",
        "train/loss_vulcode": "vulcode (aux)",
        "train/loss_seccode": "seccode (main)",
        "train/cls_loss": "cls (security head)",
    }
    window = 10

    plt.figure(figsize=(10, 5))
    for key in keys:
        values = [m.get(key) for m in metrics if m.get(key) is not None]
        if not values:
            continue
        sampled_steps = range(window, len(values) + 1, window)
        sampled = [
            sum(values[i - window : i]) / window for i in sampled_steps
        ]
        plt.plot(sampled_steps, sampled, color=colors[key], marker="o",
                 markersize=4, label=labels[key])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Bottleneck-token SFT Training Loss (Smoothed window=10)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()
