#!/usr/bin/env python3
"""CWEval 119 题 × 2 版的 LLM token 三分类标注 (CWEval token 级评测 步骤 2)。

用户要求 (09-12 原文): "再用大模型分别对两个代码的 token 进行标注, 哪些是安全 / 漏洞 / 不相关",
"安全 token 用绿色, 颜色越深表示越安全, 漏洞 token 用红色, 颜色越深表示越漏洞,
 不相关 token 用灰色, 颜色越深表示越不相关"。

分类体系:
  vuln  (红) — **漏洞版**里承载该 CWE 的 token (未校验的值到达危险操作、缺失检查的 sink、
               不安全 API 参数、被削弱的密码学参数)
  safe  (绿) — **安全版**里实施该 CWE 防护的 token (校验/转义/边界/权限检查、安全 API、净化后的实参)
  irrel (灰) — 两版里未被引用的 token (补集, 不送模型判定)

strength:
  被引用的 token = 模型给的 weight (1.0 = 就是病灶/防护本身, 0.5 = 路径上的支撑上下文)
  irrel token    = 距离衰减: 与最近被引用 token 的字符距离 / IRREL_D0, 截到 [0,1]
                   (离任何标注越远 = 越不相关 = HTML 里灰越深; 规则写进 HTML 图例)

复用件 (与 annotate_flaw_spans_llm.py 同仓库同 API 路径):
  - ARK_ENDPOINTS/ARK_KEY/ARK_MODEL/parse_json ← gen_wcstatic_cot3 (同 annotate_flaw_spans_llm:33)
  - Verifier ← annotate_flaw_spans_llm (:132-167) —— enc() 拿 offset_mapping 做字符→token 映射;
    其 resolve() 只用并集且丢弃类别, 故本脚本在其上扩一个 tag() 返回 {tok_idx: weight}
  - **thinking 必须 disabled**: annotate_flaw_spans_llm.call_api 的实测 (同一请求 137s→2.9s,
    completion 15,253 tok 里 15,191 是 reasoning)。本脚本的 call_llm 完整复刻该请求约定
    (含 429 退避), 单独实现是因为返回 schema 是第三种 (secure/vulnerable 双侧), 不适用
    其 kind="pairs"/"fix" 两种解析分支。

用法 (repository root; 需 ARK_KEY 环境变量):
  python3 -m secbart.annotate.annotate_cweval_tokens_llm --gen [--limit N]   # 生成 (多线程, 断点续跑)
  python3 -m secbart.annotate.annotate_cweval_tokens_llm --assemble          # 组装 token_labels_cweval_llm.jsonl
env: CWEVAL_WORKERS (默认 8), CWEVAL_BATCH (每请求题数, 默认 1 — 每题两版正文大)
输出: $OUT/<pool.jsonl, token_labels_cweval_llm.jsonl>
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from secbart.annotate import cweval_pairs as V  # noqa: E402
from secbart.annotate.annotate_flaw_spans_llm import Verifier  # noqa: E402
from secbart.annotate.llm_client import ENDPOINTS, KEY, MODEL, parse_json  # noqa: E402

import requests  # noqa: E402

# [release] paths are environment-driven; CWEVAL_REPO points at the CWEval checkout
ROOT = os.environ.get(
    "SECBART_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.environ.get("SECBART_CWEVAL_TOK_DIR", os.path.join(ROOT, "cweval_tkh"))
POOL = os.path.join(OUT, "cweval_tokens_pool.jsonl")
ASSEMBLED = os.path.join(OUT, "token_labels_cweval_llm.jsonl")
CASES_JSON = os.environ.get(
    "CWEVAL_CASES_JSON",
    os.path.join(os.environ.get("CWEVAL_REPO", os.path.join(ROOT, "third_party", "CWEval")),
                 "cweval", "cases.json"))
CODE_MAX = 2048       # TokenTagger.MAX; 共享模块 Verifier 的 1020 会截掉 15/238 版 (最长 1374)
PROMPT_CAP = 1200     # CWEval code_prompt 实测 ~818 字符
IRREL_D0 = 60         # irrel 距离衰减尺度 (字符)

SYS = ("You are a vulnerability analyst. You are given ONE programming task and TWO "
       "implementations of it: a SECURE reference and a VULNERABLE variant that differs from it "
       "by a minimal edit. For EACH implementation separately, point at the minimal token spans "
       "that carry the security property. You answer with verbatim quotes only, no commentary, "
       "no fixes, no line numbers.")

ONE = """=== CASE {i} ===
LANGUAGE: {lang}
CWE: {cwe}
TASK the code must implement:
{prompt}

--- IMPLEMENTATION A (SECURE reference) ---
```{lang}
{sec}
```

--- IMPLEMENTATION B (VULNERABLE variant) ---
```{lang}
{vul}
```
"""

TAIL = """
For EACH case above, fill BOTH lists:
- "secure": the minimal spans INSIDE implementation A that IMPLEMENT the defense against the CWE
  (the validation / escaping / bounds / permission check, the safe API call, the sanitized
  argument). If A has no such check for this CWE, leave the list empty.
- "vulnerable": the minimal spans INSIDE implementation B that CARRY the flaw (the unvalidated
  value reaching a dangerous operation, the sink where a check is missing, the unsafe API
  argument, the weakened parameter).
Rules:
- Every quote must be a SHORT **verbatim substring copied exactly** from the implementation it
  belongs to. A "secure" quote must occur in A; a "vulnerable" quote must occur in B. Never
  paraphrase, never add "..." or line numbers, never span a line break if a smaller expression
  carries the meaning.
- Prefer the smallest expression that carries the meaning (the argument, the value, the call),
  not the enclosing statement or function.
- NEVER quote comments, whitespace, or imports.
- weight 1.0 = this span IS the defense / the flaw; 0.5 = supporting context on its path.
- Tokens you do NOT quote are shown to a human as IRRELEVANT, so quote everything that matters.
Reply with ONLY a JSON object:
{"cases": [{"case": <i>, "cwe": "CWE-XXX" or null,
  "secure":     {"spans": [{"quote": "<verbatim from A>", "weight": 1.0, "why": "<=10 words"}]},
  "vulnerable": {"spans": [{"quote": "<verbatim from B>", "weight": 1.0, "why": "<=10 words"}]}},
  ... in the same order]}"""


def call_llm(prompt, retry=6, temperature=0.1, max_tokens=1600, system=None):
    """同一请求约定见 annotate_flaw_spans_llm.call_api: headers=Bearer, timeout=240,
    429 指数退避, 且 **thinking 必须 disabled** (否则 reasoning 吃掉 99.6% 的 completion)。"""
    ep = ENDPOINTS[0]
    last = None
    for attempt in range(retry + 1):
        try:
            r = requests.post(
                ep, headers={"Authorization": f"Bearer {KEY}"},
                json={"model": MODEL,
                      "messages": [{"role": "system", "content": system or SYS},
                                   {"role": "user", "content": prompt}],
                      "temperature": temperature, "max_tokens": max_tokens,
                      "thinking": {"type": "disabled"}},
                timeout=240)
            if r.status_code == 429:
                last = "429 rate-limit"
                time.sleep(min(90, 6 * 2 ** attempt) + attempt * 3)
                continue
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(3 + attempt * 4)
                continue
            obj = parse_json(r.json()["choices"][0]["message"]["content"])
            if obj and isinstance(obj.get("cases"), list):
                return obj
            last = "parse-fail"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(3 + attempt * 4)
    return {"err": last}


class TokenTagger(Verifier):
    """Verifier 的按类扩展: 与 resolve() 同样做 verbatim find + offset 重叠映射,
    但返回 {tok_idx: weight} (同一 token 被多 span 命中取最大权重), 供三分类与深浅着色。

    enc() 覆写只为放宽 CODE_MAX (共享模块的 1020 会截掉 238 版里的 15 版, 最长 1374)
    —— 不动 annotate_flaw_spans_llm.py 本身 (共享脚本, 其 1020 是给原任务的)。
    评测侧若头的输入窗口更窄, 按自身窗口截前 N 个 token 对齐即可, cls/strength 是逐 token 数组。
    """

    MAX = 2048

    def enc(self, code):
        e = self.tok(code, add_special_tokens=False, truncation=True,
                     max_length=self.MAX, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]

    def tag(self, code, spans, min_weight=0.0):
        ids, off = self.enc(code)
        truncated = len(ids) >= self.MAX
        hits, keep = {}, []
        for sp in spans or []:
            q = (sp.get("quote") or "").strip()
            w = float(sp.get("weight") or 1.0)
            if not q or w < min_weight:
                continue
            pos, start = [], code.find(q)
            while start >= 0:
                pos.append(start)
                start = code.find(q, start + 1)
            if not pos:
                continue                       # 非逐字 → 丢 (与 Verifier.resolve 同口径)
            for p in pos:
                a, b = p, p + len(q)
                for t, (ta, tb) in enumerate(off):
                    if tb > ta and ta < b and tb > a:
                        hits[t] = max(hits.get(t, 0.0), w)
            keep.append({"quote": q, "weight": w, "why": (sp.get("why") or "")[:80],
                         "n_occ": len(pos)})
        return ids, hits, keep, truncated


def load_cases():
    """119 题的 (case_id, language, origin, task 正文, unsafe 正文, code_prompt, cwe)。"""
    prom = {}
    if os.path.exists(CASES_JSON):
        for k, v in json.loads(open(CASES_JSON).read()).items():
            prom[os.path.basename(k)] = v.get("code_prompt") or ""
    out = []
    for c in V.find_cases():
        tpath = c.src_dir / c.task_name
        upath = c.src_dir / c.unsafe_name
        vul = c.unsafe_text if c.unsafe_text is not None else (
            upath.read_text(encoding="utf-8", errors="replace") if upath.exists() else "")
        if not vul.strip():
            continue
        num = c.case_id.split("_")[1]
        out.append({"case_id": c.case_id, "language": c.language, "origin": c.origin,
                    "group": c.group, "test_name": c.test_name,
                    "secure": tpath.read_text(encoding="utf-8", errors="replace"),
                    "vulnerable": vul,
                    "prompt": prom.get(c.task_name, "")[:PROMPT_CAP],
                    "cwe": f"CWE-{int(num)}"})
    return out


def build_prompt(batch):
    parts = []
    for i, r in enumerate(batch):
        parts.append(ONE.format(i=i, lang=r["language"], cwe=r["cwe"],
                                prompt=r["prompt"], sec=r["secure"], vul=r["vulnerable"]))
    return "".join(parts) + TAIL


def gen(limit=0):
    os.makedirs(OUT, exist_ok=True)
    tg = TokenTagger()
    cases = load_cases()
    done = set()
    if os.path.exists(POOL):
        for line in open(POOL):
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done.add(r["case_id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in cases if r["case_id"] not in done]
    if limit:
        todo = todo[:limit]
    nb = int(os.environ.get("CWEVAL_BATCH", "1"))
    workers = int(os.environ.get("CWEVAL_WORKERS", "8"))
    print(f"[cw-tok] cases={len(cases)} done={len(done)} todo={len(todo)} "
          f"batch={nb} workers={workers}", flush=True)
    batches = [todo[i:i + nb] for i in range(0, len(todo), nb)]
    lock = threading.Lock()
    stats = {"ok": 0, "none": 0, "err": 0, "tok": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call_llm, build_prompt(b)): b for b in batches}
        for k, fut in enumerate(as_completed(futs), 1):
            b = futs[fut]
            res = fut.result()
            recs = []
            if "err" in res:
                recs = [{"case_id": r["case_id"], "err": res["err"][:200], "ok": False} for r in b]
                with lock:
                    stats["err"] += len(b)
            else:
                by = {}
                for p in res["cases"]:
                    try:
                        by[int(p.get("case"))] = p
                    except Exception:  # noqa: BLE001
                        pass
                for i, r in enumerate(b):
                    p = by.get(i, {})
                    sids, shits, skeep, strunc = tg.tag(
                        r["secure"], (p.get("secure") or {}).get("spans"))
                    vids, vhits, vkeep, vtrunc = tg.tag(
                        r["vulnerable"], (p.get("vulnerable") or {}).get("spans"))
                    if not shits and not vhits:
                        recs.append({"case_id": r["case_id"], "ok": False,
                                     "err": "no-verbatim-span-either-side"})
                        with lock:
                            stats["none"] += 1
                    else:
                        recs.append({"case_id": r["case_id"], "ok": True,
                                     "cwe": p.get("cwe") or r["cwe"],
                                     "secure_spans": skeep, "secure_tok": sorted(shits),
                                     "secure_w": {str(t): shits[t] for t in shits},
                                     "vul_spans": vkeep, "vul_tok": sorted(vhits),
                                     "vul_w": {str(t): vhits[t] for t in vhits},
                                     "n_sec": len(sids), "n_vul": len(vids),
                                     "trunc_sec": strunc, "trunc_vul": vtrunc})
                        with lock:
                            stats["ok"] += 1
                            stats["tok"] += len(shits) + len(vhits)
            with lock:
                with open(POOL, "a") as f:
                    for rec in recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if k % 20 == 0:
                    el = time.time() - t0
                    print(f"[cw-tok] {k}/{len(batches)} ok={stats['ok']} none={stats['none']} "
                          f"err={stats['err']} {stats['ok']/max(el,1e-9):.2f} case/s", flush=True)
    print(f"[cw-tok] DONE ok={stats['ok']} none={stats['none']} err={stats['err']} "
          f"tokens={stats['tok']}", flush=True)


def irrel_strength(n, hits, off):
    """irrel token 的强度 = 与最近被引用 token 的字符距离 / IRREL_D0, 截到 [0,1]。"""
    starts = [off[t][0] for t in hits]
    out = [0.0] * n
    for t in range(n):
        if t in hits:
            continue
        ta, tb = off[t]
        if tb <= ta:
            continue
        d = min((abs(ta - s) for s in starts), default=IRREL_D0)
        out[t] = min(1.0, d / IRREL_D0)
    return out


def assemble():
    pool = {}
    for line in open(POOL):
        r = json.loads(line)
        if r.get("ok"):
            pool[r["case_id"]] = r
    tg = TokenTagger()
    rows, st = [], {"kept": 0, "nopool": 0, "empty": 0, "trunc": 0}
    for c in V.find_cases():
        p = pool.get(c.case_id)
        if p is None:
            st["nopool"] += 1
            continue
        tpath = c.src_dir / c.task_name
        upath = c.src_dir / c.unsafe_name
        sec = tpath.read_text(encoding="utf-8", errors="replace")
        vul = c.unsafe_text if c.unsafe_text is not None else (
            upath.read_text(encoding="utf-8", errors="replace") if upath.exists() else "")
        for variant, code, tok_key, w_key, cls_hit in (
                ("task", sec, "secure_tok", "secure_w", "safe"),
                ("unsafe", vul, "vul_tok", "vul_w", "vuln")):
            ids, off = tg.enc(code)
            hits = {int(t): float(v) for t, v in (p.get(w_key) or {}).items() if int(t) < len(ids)}
            if p.get("trunc_sec" if variant == "task" else "trunc_vul"):
                st["trunc"] += 1
            if not hits:
                st["empty"] += 1
                continue
            cls = [cls_hit if t in hits else "irrel" for t in range(len(ids))]
            strength = [0.0] * len(ids)
            for t, w in hits.items():
                strength[t] = w
            ir = irrel_strength(len(ids), hits, off)
            for t in range(len(ids)):
                if cls[t] == "irrel":
                    strength[t] = ir[t]
            rows.append({
                "case_id": c.case_id, "language": c.language, "origin": c.origin,
                "group": c.group, "variant": variant, "code": code,
                "ids": ids, "cls": cls, "strength": [round(x, 3) for x in strength],
                # 兼容键: labels/weights 语义不变 (labels=1 为被标注 token; 未标记者权重 0),
                # 训练侧 train_token_head_rl.py:93-98 读这两个键, 新增的 cls/strength 不干扰
                "labels": [1 if t in hits else 0 for t in range(len(ids))],
                "weights": [round(strength[t] if t in hits else 0.0, 3) for t in range(len(ids))],
                "spans": p.get(f"{'secure' if variant == 'task' else 'vul'}_spans"),
                "cwe": p.get("cwe"),
            })
            st["kept"] += 1
    with open(ASSEMBLED, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = {}
    for r in rows:
        per.setdefault(r["variant"], []).append(sum(1 for x in r["cls"] if x != "irrel"))
    print(f"[assemble] {ASSEMBLED} rows={len(rows)} | {st}")
    for v, ln in per.items():
        if ln:
            ln.sort()
            print(f"[assemble] {v} 标注 token 数 min/med/p90/max = {ln[0]}/{ln[len(ln)//2]}/"
                  f"{ln[int(len(ln)*0.9)]}/{ln[-1]} (n={len(ln)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.gen:
        gen(a.limit)
    elif a.assemble:
        assemble()
    else:
        ap.print_help()
