"""vLLM OpenAI-compatible server for bottleneck-token (btoks) models.

Inference layout (identical to training inference):
    input + <vuln>*N + <secu>  ->  seccode

Key fact (verified numerically): with no vulcode block at inference, the
btoks attention mask is EXACTLY the standard causal mask.  So we only need to
append the <vuln>*N + <secu> token ids to the raw prompt and let vLLM generate
with its normal causal attention + paged KV cache.  This removes the O(T^2)
HF loop (use_cache=False) and enables vLLM's continuous batching.

For plain models (--plain) no tokens are appended.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m secbart.vllm_bottleneck_server \
        --model_path .../merged_hf_model --port 8300 --n_vuln 4
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
import uvicorn
from fastapi import FastAPI, Request

from secbart.bottleneck_token_common import (
    SECU_TOK, VULN_TOK, VULN_TOKS, JFIX_TOK,
    FUNC_ANAL_TOK, VULN_ANAL_TOK, SECU_IMPL_TOK,
    THINK_TOK, THINK_FUNC_ANAL_TOK, THINK_VULN_ANAL_TOK, THINK_SECU_IMPL_TOK,
)

app = FastAPI()
ENGINE = None
TOKENIZER = None
N_VULN = 4
PLAIN = False
CHAT = False
MULTI_VULN = False
INTERLEAVE = False
JUDGE = False
THREE_STAGE = False
TWO_STAGE = False
RESP_BOTH_MARKER = "\n<<<VULCODE_END>>>\n"  # two_stage 返回 = vulcode + marker + seccode (两段都进 resp)
NO_SECU = False
THINK_BLOCK = False
THINK_TRIPLE = False
THINK_FRONT = False
DEBUG_STREAM = False
VUL_MAX_TOKENS = 768  # stage-1 vulcode budget (train p90 ~500 tok, max 3828 chars)
THINK_MAX_TOKENS = 512  # think_front stage-1 链预算 (train 链 p90 ~330 tok)
MODEL_ID = "Qwen2.5-Coder-0.5B"
DEFAULT_RP = 1.0


def _suffix_ids():
    v = TOKENIZER.convert_tokens_to_ids(VULN_TOK)
    s = TOKENIZER.convert_tokens_to_ids(SECU_TOK)
    if v is None or s is None:
        raise RuntimeError(
            "bottleneck serving requires tokenizer entries for "
            f"{VULN_TOK} and {SECU_TOK}; use --plain for a raw base model "
            "or serve a merged_hf_model that includes the bottleneck tokens"
        )
    if MULTI_VULN:
        vs = [TOKENIZER.convert_tokens_to_ids(t) for t in VULN_TOKS[:N_VULN]]
        if any(i is None for i in vs):
            raise RuntimeError(
                "multi-vuln serving requires <vuln1>..<vulnN> in the tokenizer"
            )
        return vs + [s]
    j = TOKENIZER.convert_tokens_to_ids(JFIX_TOK) if JUDGE else None
    if JUDGE and j is None:
        raise RuntimeError(
            "judge serving requires <jfix> in the tokenizer "
            "(serve a merged_hf_model trained with --judge)"
        )
    # 两段条件生成：固定插 <jfix>（btoks 只用于修复任务，输入恒为待修漏洞代码）
    if NO_SECU:
        # 实验2：<vuln> 直出——不带 <secu>，模型直接生成（vulcode 分布），
        # 测两阶段流水线 stage1（草案生成）质量。
        return [v] * N_VULN
    return [v] * N_VULN + ([j] if JUDGE else []) + [s]


def _think_block_suffix():
    """实验4：<think>*4 双层瓶颈压缩推理——input + <think>*4 + <vuln>*N + <secu>。
    think 文本不生成（全压缩），<think>*4 块代替显式分析。"""
    thk = TOKENIZER.convert_tokens_to_ids(THINK_TOK)
    v = TOKENIZER.convert_tokens_to_ids(VULN_TOK)
    s = TOKENIZER.convert_tokens_to_ids(SECU_TOK)
    if thk is None or v is None or s is None:
        raise RuntimeError(
            "--think_block requires <think>/<vuln>/<secu> in the tokenizer")
    return [thk] * 4 + [v] * N_VULN + [s]


def _think_triple_suffix():
    """实验5：三段 think token 压缩推理——input + <think_func_anal>
    <think_vuln_anal> <think_secu_impl> + <vuln>*N + <secu>。三段分析不生成。"""
    toks = [TOKENIZER.convert_tokens_to_ids(t) for t in
            (THINK_FUNC_ANAL_TOK, THINK_VULN_ANAL_TOK, THINK_SECU_IMPL_TOK)]
    v = TOKENIZER.convert_tokens_to_ids(VULN_TOK)
    s = TOKENIZER.convert_tokens_to_ids(SECU_TOK)
    if any(i is None for i in toks) or v is None or s is None:
        raise RuntimeError(
            "--think_triple requires <think_func_anal>/<think_vuln_anal>/"
            "<think_secu_impl>/<vuln>/<secu> in the tokenizer")
    return toks + [v] * N_VULN + [s]


def _three_stage_suffix():
    """三段式思维链布局（推理无 vulcode）：input <func_anal> <vuln>*N
    <vuln_anal> <secu_impl> <secu>，FA/VA/SI 文本由模型依次生成。"""
    toks = [TOKENIZER.convert_tokens_to_ids(t) for t in
            (FUNC_ANAL_TOK, VULN_TOK, VULN_ANAL_TOK, SECU_IMPL_TOK, SECU_TOK)]
    if any(i is None for i in toks):
        raise RuntimeError(
            "three-stage serving requires <func_anal>/<vuln>/<vuln_anal>/"
            "<secu_impl>/<secu> in the tokenizer "
            "(serve a merged_hf_model trained with three-stage data)")
    v = toks[1]
    return [toks[0]] + [v] * N_VULN + toks[2:]


def _interleave_toks(toks):
    """chunk-interleaved 布局（与训练 --interleave 一致）：input 均匀切 N 块，
    交错插 <vuln1..N>，末尾 + <secu>。标准 causal 即局部感受野（PIC+EPL）。"""
    vs = [TOKENIZER.convert_tokens_to_ids(t) for t in VULN_TOKS[:N_VULN]]
    s = TOKENIZER.convert_tokens_to_ids(SECU_TOK)
    if any(i is None for i in vs) or s is None:
        raise RuntimeError(
            "interleave serving requires <vuln1>..<vulnN> and <secu> in the tokenizer"
        )
    chunk = max(1, -(-len(toks) // N_VULN))
    out = []
    for i in range(N_VULN):
        out.extend(toks[i * chunk:(i + 1) * chunk])
        out.append(vs[i])
    return out + [s]


def _strip_think(text: str) -> str:
    """删除成对 <think>...</think> 段（模型自产思维链，评测提取代码前剥掉）。

    只删成对的：不闭合的 <think>（生成被截断）保留原样，避免误删代码。
    评测链 get_code_from 只提取 ``` 代码块，think 分析里的 ``` 引用会
    污染提取——统一在生成端剥掉。CWEval raw 文件只含生成段，不受影响。
    """
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


async def _generate(prompt: str, max_tokens: int, temperature: float,
                    top_p: float, stop, n: int, repetition_penalty: float = 1.0):
    """Submit one request to vLLM's shared async scheduler.

    Every HTTP request gets its own generator, but AsyncLLMEngine schedules all
    live generators together. This keeps the API shape of the HF server while
    enabling vLLM continuous batching for benchmark workers.
    """
    from vllm import SamplingParams
    if CHAT:
        try:
            prompt = TOKENIZER.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    toks = TOKENIZER(prompt, add_special_tokens=False)["input_ids"]

    if THINK_FRONT:
        # 链前置 (#6, 2026-09-10): 两段式推理, 与训练 --think_mode front 逐位对齐。
        #   stage-1: input                         -> <think> 链 (stop 于 </think>)
        #   stage-2: input + <think> 链 </think> + <vuln>*N + <secu> -> seccode
        # 关键：stage-2 的 <secu> 位置 = L + A + N = 训练 layout 里 <secu> 的**回卷位**
        # (vulcode 长度不影响它), 且推理没有 vulcode 段 → 训练的 block-causal 掩码
        # (seccode 挖掉 vulcode 列) 恰好退化成标准 causal = vLLM 原生行为。
        vid = TOKENIZER.convert_tokens_to_ids(VULN_TOK)
        sid = TOKENIZER.convert_tokens_to_ids(SECU_TOK)
        if vid is None or sid is None:
            raise RuntimeError(
                "--think_front requires <vuln>/<secu> in the tokenizer")
        cids = TOKENIZER("</think>\n\n", add_special_tokens=False)["input_ids"]
        p1 = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p, max_tokens=THINK_MAX_TOKENS, n=n,
            stop=["</think>"], include_stop_str_in_output=False,
            skip_special_tokens=False, stop_token_ids=None,
            repetition_penalty=repetition_penalty,
        )
        p2 = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p, max_tokens=max_tokens, n=1,
            stop=stop or None, stop_token_ids=None,
            repetition_penalty=repetition_penalty,
        )
        out_texts = []
        for _ in range(max(1, n)):
            gid = []
            async for out in ENGINE.generate(
                {"prompt_token_ids": toks}, p1,
                request_id=f"btoktf-{uuid.uuid4().hex}",
            ):
                for o in out.outputs:
                    if o.token_ids:
                        gid = list(o.token_ids)
            # vLLM 的 token_ids 含 stop 串 (文本已截, token 未截) → 按 token 精确去尾;
            # 尾部不匹配则退回 decode→重编码 (skip_special_tokens=False, 保住 <think>)
            cclose = TOKENIZER("</think>", add_special_tokens=False)["input_ids"]
            if len(gid) >= len(cclose) and gid[-len(cclose):] == cclose:
                gid = gid[:-len(cclose)]
                cut = "exact"
            else:
                gid = TOKENIZER(TOKENIZER.decode(gid), add_special_tokens=False)["input_ids"]
                cut = "reenc"
            toks2 = toks + gid + cids + [vid] * N_VULN + [sid]
            s2, final_fr = "", None
            async for out in ENGINE.generate(
                {"prompt_token_ids": toks2}, p2,
                request_id=f"btoktf-{uuid.uuid4().hex}",
            ):
                final_fr = out
                for o in out.outputs:
                    if o.text:
                        s2 = o.text
            print(f"[tf-dbg] think_tok={len(gid)} cut={cut} "
                  f"think_head={TOKENIZER.decode(gid[:12])!r} "
                  f"stage2 len={len(s2)} "
                  f"finish={final_fr and final_fr.outputs[0].finish_reason}",
                  flush=True)
            out_texts.append(_strip_think(s2) if not CHAT else s2)
        return out_texts[: max(1, n)]

    if TWO_STAGE:
        # 两段式无压缩基线：input + <vuln>*N -> vulcode（停在 <secu>/<vuln>），
        # 再 input + <vuln>*N + vulcode + <secu> -> seccode。
        # 全程标准 causal mask（secu 段可 attend 到 vulcode 全文）——训练是
        # block-causal 压缩，推理不压缩，验证"压缩"在推理端是否必要。
        vids = [TOKENIZER.convert_tokens_to_ids(VULN_TOK)] * N_VULN
        sid = TOKENIZER.convert_tokens_to_ids(SECU_TOK)
        if vids[0] is None or sid is None:
            raise RuntimeError("--two_stage requires <vuln>/<secu> in the tokenizer")
        p1 = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p, max_tokens=VUL_MAX_TOKENS, n=n,
            stop=stop or None,
            stop_token_ids=[sid, vids[0]],
            repetition_penalty=repetition_penalty,
        )
        vulcodes = []
        for i in range(n):
            vtext = ""
            async for out in ENGINE.generate(
                {"prompt_token_ids": toks + vids}, p1,
                request_id=f"btok2s-{uuid.uuid4().hex}",
            ):
                # 流式：最后 chunk 可能为空（EOS 结束信号），累积非空 chunk
                for o in out.outputs:
                    if o.text:
                        vtext = o.text
            vulcodes.append(vtext)
        # stage 2: concatenate vulcode text verbatim (no mask restriction)
        p2 = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p, max_tokens=max_tokens, n=1,
            stop=stop or None, stop_token_ids=None,
            repetition_penalty=repetition_penalty,
        )
        out_texts = []
        for vulcode in vulcodes:
            toks2 = (toks + vids
                     + TOKENIZER(vulcode, add_special_tokens=False)["input_ids"]
                     + [sid])
            s2 = ""
            final_fr = None
            async for out in ENGINE.generate(
                {"prompt_token_ids": toks2}, p2,
                request_id=f"btok2s-{uuid.uuid4().hex}",
            ):
                final_fr = out
                for o in out.outputs:
                    if o.text:
                        s2 = o.text
            print(f"[ts-dbg] n_vuln={N_VULN} stage1 vulcode len={len(vulcode)} "
                  f"stage2 out len={len(s2)} finish={final_fr and final_fr.outputs[0].finish_reason} "
                  f"imend_in_vulcode={'<|im_end|>' in vulcode} "
                  f"vulcode head={vulcode[:120]!r}", flush=True)
            s2_clean = _strip_think(s2) if not CHAT else s2
            out_texts.append(vulcode + RESP_BOTH_MARKER + s2_clean)
        return out_texts[: max(1, n)]

    if not PLAIN:
        try:
            if THINK_BLOCK:
                toks = toks + _think_block_suffix()
            elif THINK_TRIPLE:
                toks = toks + _think_triple_suffix()
            elif THREE_STAGE:
                toks = toks + _three_stage_suffix()
            elif INTERLEAVE:
                toks = _interleave_toks(toks)
            else:
                toks = toks + _suffix_ids()
        except RuntimeError:
            if not CHAT:
                raise
            # chat-only instruct models (no btoks in vocab) run without suffix
    params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p, max_tokens=max_tokens, n=n,
        stop=stop or None, stop_token_ids=None,
        repetition_penalty=repetition_penalty,
    )
    texts = [""] * max(1, n)
    request_id = f"btok-{uuid.uuid4().hex}"
    n_chunks = 0
    async for out in ENGINE.generate(
        {"prompt_token_ids": toks}, params, request_id=request_id
    ):
        # 流式：最后一个 chunk 在自然 EOS 停止时为空（结束信号），
        # 不能取 final 覆盖——累积每个 chunk（chunk.text 是累计文本）。
        for i, o in enumerate(out.outputs):
            if o.text:
                texts[i] = o.text
        n_chunks += 1
        if DEBUG_STREAM and (n_chunks <= 3 or n_chunks % 100 == 0):
            print(f"[strm] {request_id} chunk#{n_chunks} "
                  f"len={len(out.outputs[0].text)} "
                  f"fr={out.outputs[0].finish_reason}", flush=True)
    if DEBUG_STREAM:
        print(f"[strm] {request_id} DONE chunks={n_chunks} "
              f"outlen={len(texts[0])}", flush=True)
    # chat 端点不剥 think：客户端（gen_analysis 等）显式要求模型输出
    # <think>...</think> 包裹的分析（ANALYSIS_ONLY_PROMPT tag_instr），
    # 剥掉成对 think 会把整个分析剥成空。非 chat（CWEval 代码生成）保留剥除。
    if not CHAT:
        texts = [_strip_think(t) for t in texts]
    return texts[: max(1, n)]


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = ""
    for m in messages:  # last non-system message == coding prompt (same as HF server)
        if m.get("role") != "system":
            prompt = m.get("content", "") or ""
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
    temperature = float(body.get("temperature", 0.0))
    top_p = float(body.get("top_p", 1.0))
    n = int(body.get("n", 1))
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    repetition_penalty = float(body.get("repetition_penalty", DEFAULT_RP))
    texts = await _generate(prompt, max_tokens, temperature, top_p, stop, n, repetition_penalty)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {"index": i, "message": {"role": "assistant", "content": t},
             "finish_reason": "stop"}
            for i, t in enumerate(texts)
        ],
    }


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
    temperature = float(body.get("temperature", 0.0))
    top_p = float(body.get("top_p", 1.0))
    n = int(body.get("n", 1))
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    repetition_penalty = float(body.get("repetition_penalty", DEFAULT_RP))
    texts = await _generate(prompt, max_tokens, temperature, top_p, stop, n, repetition_penalty)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:16]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": i, "text": t, "finish_reason": "stop"} for i, t in enumerate(texts)],
    }


def main():
    global ENGINE, TOKENIZER, N_VULN, PLAIN, CHAT, MULTI_VULN, INTERLEAVE, JUDGE, THREE_STAGE, TWO_STAGE, NO_SECU, THINK_BLOCK, THINK_TRIPLE, THINK_FRONT, MODEL_ID, DEBUG_STREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--n_vuln", type=int, default=4)
    ap.add_argument("--multi_vuln", action="store_true",
                    help="L53: use <vuln1>..<vulnN> instead of <vuln> x N")
    ap.add_argument("--interleave", action="store_true",
                    help="chunk-interleaved 布局（与训练 --interleave 一致）")
    ap.add_argument("--judge", action="store_true",
                    help="两段条件生成：<secu> 前固定插 <jfix>（与 --interleave 互斥）")
    ap.add_argument("--three_stage", action="store_true",
                    help="三段式思维链布局（与训练 three-stage 数据一致）："
                         "suffix = <func_anal> <vuln>*N <vuln_anal> <secu_impl> <secu>，"
                         "FA/VA/SI 文本由模型生成")
    ap.add_argument("--two_stage", action="store_true",
                    help="两段式无压缩基线：先生成 vulcode（<vuln>*N 后，stop 于 "
                         "<secu>），再拼接 vulcode 全文 + <secu> 生成 seccode；"
                         "全程标准 causal（推理不压缩）")
    ap.add_argument("--no_secu", action="store_true",
                    help="实验2：suffix 只有 <vuln>*N（不带 <secu>），模型直接"
                         "生成代码（vulcode 分布，两阶段流水线 stage1 质量）")
    ap.add_argument("--think_block", action="store_true",
                    help="实验4：suffix = <think>*4 + <vuln>*N + <secu>（<think>"
                         " 块代替显式分析，全压缩推理）")
    ap.add_argument("--think_triple", action="store_true",
                    help="实验5：suffix = <think_func_anal> <think_vuln_anal> "
                         "<think_secu_impl> + <vuln>*N + <secu>（三段分析不生成）")
    ap.add_argument("--think_front", action="store_true",
                    help="#6 链前置两段式推理（配 --think_mode front 训练）："
                         "stage-1 input -> <think>链</think>；stage-2 "
                         "input+链+<vuln>*N+<secu> -> seccode")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model_name", default="Qwen2.5-Coder-0.5B")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--chat", action="store_true",
                    help="apply tokenizer chat template (instruct models)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    global DEFAULT_RP
    try:
        import json as _json
        with open(f"{args.model_path}/generation_config.json") as _f:
            DEFAULT_RP = float(_json.load(_f).get("repetition_penalty", 1.0))
        print(f"[server] generation_config repetition_penalty={DEFAULT_RP}", flush=True)
    except Exception as _e:
        print(f"[server] no generation_config rp found, default 1.0", flush=True)
    MODEL_ID = args.model_name
    N_VULN = args.n_vuln
    PLAIN = args.plain
    MULTI_VULN = args.multi_vuln
    INTERLEAVE = args.interleave
    JUDGE = args.judge
    CHAT = args.chat
    THREE_STAGE = args.three_stage
    TWO_STAGE = args.two_stage
    NO_SECU = args.no_secu
    THINK_BLOCK = args.think_block
    THINK_TRIPLE = args.think_triple
    THINK_FRONT = args.think_front
    if THINK_FRONT and (TWO_STAGE or THREE_STAGE or INTERLEAVE or JUDGE
                        or THINK_BLOCK or THINK_TRIPLE or NO_SECU):
        raise RuntimeError("--think_front 与其他布局 flag 互斥")
    if JUDGE and INTERLEAVE:
        raise RuntimeError("--judge 与 --interleave 互斥")
    if THREE_STAGE and (JUDGE or INTERLEAVE):
        raise RuntimeError("--three_stage 与 --judge/--interleave 互斥")
    if TWO_STAGE and (JUDGE or THREE_STAGE or INTERLEAVE):
        raise RuntimeError("--two_stage 与 --judge/--three_stage/--interleave 互斥")
    if THINK_BLOCK and THINK_TRIPLE:
        raise RuntimeError("--think_block 与 --think_triple 互斥")
    if (THINK_BLOCK or THINK_TRIPLE) and (TWO_STAGE or THREE_STAGE or INTERLEAVE or JUDGE):
        raise RuntimeError("--think_block/--think_triple 与 --two_stage/--three_stage/--interleave/--judge 互斥")
    if NO_SECU and (JUDGE or THREE_STAGE or INTERLEAVE or TWO_STAGE or THINK_BLOCK or THINK_TRIPLE):
        raise RuntimeError("--no_secu 与其他布局 flag 互斥")
    print(f"loading async vLLM {args.model_path} (plain={PLAIN}, chat={CHAT}, n_vuln={N_VULN}, interleave={INTERLEAVE}, judge={JUDGE}, three_stage={THREE_STAGE}, two_stage={TWO_STAGE}, no_secu={NO_SECU}, think_block={THINK_BLOCK}, think_triple={THINK_TRIPLE}, think_front={THINK_FRONT}) ...", flush=True)
    # Validate the vocabulary before allocating an engine/GPU. This catches a
    # raw base checkpoint accidentally passed without --plain immediately.
    TOKENIZER = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if not PLAIN:
        try:
            if THINK_BLOCK:
                _think_block_suffix()
            elif THINK_TRIPLE:
                _think_triple_suffix()
            elif INTERLEAVE:
                _interleave_toks([])
            else:
                _suffix_ids()
        except RuntimeError:
            if not CHAT:
                raise
            print("[server] chat-only model (no btoks); suffix skipped", flush=True)
    DEBUG_STREAM = os.environ.get("DEBUG_STREAM", "0") == "1"
    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        task="generate",
        enforce_eager=False,
        limit_mm_per_prompt=None,
    )
    ENGINE = AsyncLLMEngine.from_engine_args(engine_args)
    v = TOKENIZER.convert_tokens_to_ids(VULN_TOK)
    s = TOKENIZER.convert_tokens_to_ids(SECU_TOK)
    print(f"tokenizer ready; <vuln>={v} <secu>={s} (plain={PLAIN})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
