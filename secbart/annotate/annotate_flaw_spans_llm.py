#!/usr/bin/env python3
"""#66/tkh 数据层 v3 (09-10 用户令: 「你可以任意调用大模型来优化训练数据 /
例如加强最小编辑, 或者直接生成每个 token 的重要性」)。

v1 标签 = vul/sec 的 token 级 difflib replace/delete 全段 → 实测噪声:
  - 96,006 个被标病灶的 token 是注释/标点/空白 类噪声 (10,274/14,971 对里有);
  - **1,568 对 (10.5%) 的病灶全部由噪声组成** —— 整条样本在教头"注释=危险";
  - 大段重写里 sec 的副作用改动 (重命名/加 include/缩进) 也被标成病灶。
本脚本让 LLM 直接给出**最小病灶跨度** (verbatim 引用, 可逐字校验) 及每跨度权重
  = 用户说的「直接生成每个 token 的重要性」的稀疏可验证形式。

产出 (append 续跑, 可断点):
  $D/wcstatic_synthref_merge/flaw_spans_pool.jsonl
    {pair_id, spans:[{quote, weight, why, tok_idx:[...], n_tok}], raw, err}
  $D/wcstatic_synthref_merge/token_labels_v3_llm.jsonl  (--assemble, 与 v1 同 schema
    + spans/weights 审计键; train_token_head_rl.py 直读)

用法:
  python3 sh/data/annotate_flaw_spans_llm.py --gen [--limit N]     # 生成 (多线程)
  python3 sh/data/annotate_flaw_spans_llm.py --assemble [--max-edit R] [--min-weight W]
env: ARK_KEY/ARK_BASE/ARK_MODEL (同 gen_wcstatic_cot3); FLAW_WORKERS (默认 8),
     FLAW_BATCH (每请求对数, 默认 3)
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from secbart.annotate.llm_client import ENDPOINTS, KEY, MODEL, parse_json  # noqa: E402

# [release] paths/credentials are environment-driven instead of workspace-hardcoded
SRC = os.environ.get("SECBART_SFT_JSON", "")
THINK = os.environ.get("SECBART_THINK_JSONL", "")

import requests  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

D = os.environ.get("SECBART_WORK_DIR") or (os.path.dirname(SRC) if SRC else os.getcwd())
POOL = os.environ.get("SECBART_FLAW_POOL", f"{D}/flaw_spans_pool.jsonl")
OUT = os.environ.get("SECBART_FLAW_OUT", f"{D}/token_labels_v3_llm.jsonl")
TOK_PATH = os.environ.get("SECBART_TOKENIZER", "Qwen/Qwen2.5-Coder-7B")
CODE_MAX = 1020
TASK_CAP = 1000
VUL_CAP = 2200
SEC_CAP = 1500

SYS = ("You are a vulnerability analyst. Given a task, a vulnerable implementation "
       "and its corrected version, you point at the MINIMAL tokens of the vulnerable "
       "implementation that carry the flaw. You answer with verbatim quotes only, no "
       "commentary, no fixes, no line numbers.")

ONE = """=== PAIR {i} ===
TASK the assistant had to implement:
{prompt}

VULNERABLE implementation:
```{lang}
{vul}
```
CORRECTED implementation (reference only — never quote from it):
```{lang}
{sec}
```
"""

TAIL = """
For EACH pair above, output the minimal token span(s) of the VULNERABLE implementation
that carry the security flaw — the root cause an attacker exploits (an unvalidated value
reaching a dangerous operation, a missing bounds/ownership/permission check, an unsafe API
argument, a hardcoded secret, a weak crypto parameter, an injection sink).
Rules:
- 1-3 spans per pair, each a SHORT **verbatim substring copied exactly** from that pair's
  vulnerable code. Copy it character-for-character; never paraphrase, never add "..." or
  line numbers, never span a line break if a smaller expression carries the flaw.
- Prefer the smallest expression that carries the flaw (the argument, the value, the call),
  not the enclosing statement or function.
- NEVER quote comments, whitespace, imports/includes, or anything from the corrected code.
- If the flaw is a MISSING check, quote the dangerous operation or the value that reaches
  it (the sink), NOT the check the corrected version adds.
- weight 1.0 = this span IS the flaw; 0.5 = supporting context on the flaw's path.
- If a pair has no security flaw you can name, give it an empty spans list.
Reply with ONLY a JSON object:
{"pairs": [{"pair": <i>, "cwe": "CWE-XXX" or null,
  "spans": [{"quote": "<verbatim from the vulnerable code>", "weight": 1.0,
             "why": "<=10 words"}]}, ... in the same order]}"""


def call_api(prompt, retry=6, temperature=0.1, max_tokens=900,
             system=None, kind="pairs"):
    """kind="pairs" = 多对 span 问法 (默认, SYS);  kind="fix" = 单点因果问法 (调用方传 system,
    期望 {"found","quote","cwe","why"})。09-11 加: 原先解析硬编码 `pairs` 键 → v4 fix 模式
    **100% parse-fail** (每个 job 还白烧 7 次重试), 见 annotate_flaw_votes.py --mode fix。"""
    ep = ENDPOINTS[0]
    last = None
    for attempt in range(retry + 1):
        try:
            # 09-10 关键: **必须关 thinking** —— 实测同一请求 137s→2.9s (47×):
            # 开思考时 completion=15,253 tok 里 15,191 是 reasoning (可见答案仅 158 字符),
            # 8 并发也只能 3 对/分钟, ETA 83h; 关掉后 reasoning=0、质量不变。
            # 同 gen_cot_full_en.py 的 stage-2 通道 (ark 原生字段, 非 SDK extra_body)。
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
            if kind == "fix":
                # 单点问法: 原样回传 {"found","quote","cwe","why"}, 由调用方按 fix 语义处理
                if obj and ("quote" in obj or "found" in obj):
                    return obj
            elif obj and isinstance(obj.get("pairs"), list):
                return {"pairs": obj["pairs"]}
            last = "parse-fail"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(3 + attempt * 4)
    return {"err": last}


class Verifier:
    """quote → 字符区间 → token 索引 (逐字校验, 找不到就整条失败)。"""

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(TOK_PATH)

    def enc(self, code):
        e = self.tok(code, add_special_tokens=False, truncation=True,
                     max_length=CODE_MAX, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]

    def resolve(self, code, spans, min_weight=0.0):
        """返回 (tok_idx 列表, 保留的 spans, 失败原因)。"""
        ids, off = self.enc(code)
        keep, idx = [], set()
        for sp in spans:
            q = (sp.get("quote") or "").strip()
            w = float(sp.get("weight") or 1.0)
            if not q or w < min_weight:
                continue
            pos, start = [], code.find(q)
            while start >= 0:
                pos.append(start)
                start = code.find(q, start + 1)
            if not pos:
                continue                      # 该 quote 非逐字 → 丢
            for p in pos:
                a, b = p, p + len(q)
                for t, (ta, tb) in enumerate(off):
                    if tb > ta and ta < b and tb > a:
                        idx.add(t)
            keep.append({"quote": q, "weight": w, "why": (sp.get("why") or "")[:80],
                         "n_occ": len(pos)})
        if not keep:
            return None, None, "no-verbatim-span"
        return sorted(idx), keep, ""


def load_pairs():
    """v1 标签 → 去重后的 pair 表 (prompt/code 取自 pos 行)。"""
    rows = [json.loads(l) for l in open(f"{D}/token_labels_rl_pairs.jsonl")]
    seen, out = set(), []
    for r in rows:
        if r["side"] == 0 and r["pair_id"] not in seen:
            seen.add(r["pair_id"])
            out.append(r)
    return out


def build_prompt(batch, lang=""):
    parts = []
    for i, r in enumerate(batch):
        parts.append(ONE.format(i=i, prompt=(r["prompt"] or "")[:TASK_CAP],
                                vul=r["code"][:VUL_CAP], sec="", lang=lang))
    return "".join(parts) + TAIL


def gen(limit=0):
    ver = Verifier()
    pairs = load_pairs()
    done = set()
    if os.path.exists(POOL):
        for line in open(POOL):
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done.add(r["pair_id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in pairs if r["pair_id"] not in done]
    if limit:
        todo = todo[:limit]
    nb = int(os.environ.get("FLAW_BATCH", "3"))
    workers = int(os.environ.get("FLAW_WORKERS", "8"))
    print(f"[flaw] pairs={len(pairs)} done={len(done)} todo={len(todo)} "
          f"batch={nb} workers={workers}", flush=True)
    batches = [todo[i:i + nb] for i in range(0, len(todo), nb)]
    lock = threading.Lock()
    stats = {"ok": 0, "none": 0, "err": 0, "tok": 0}
    t0 = time.time()

    def work(b):
        return b, call_api(build_prompt(b))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, b) for b in batches]
        for k, fut in enumerate(as_completed(futs), 1):
            b, res = fut.result()
            recs = []
            if "err" in res:
                recs = [{"pair_id": r["pair_id"], "err": res["err"][:200], "ok": False}
                        for r in b]
                with lock:
                    stats["err"] += len(b)
            else:
                by = {}
                for p in res["pairs"]:
                    try:
                        by[int(p.get("pair"))] = p
                    except Exception:  # noqa: BLE001
                        pass
                for i, r in enumerate(b):
                    p = by.get(i, {})
                    idx, keep, why = ver.resolve(r["code"], p.get("spans") or [])
                    if idx is None:
                        recs.append({"pair_id": r["pair_id"], "ok": False, "err": why})
                        with lock:
                            stats["none"] += 1
                    else:
                        recs.append({"pair_id": r["pair_id"], "ok": True, "cwe": p.get("cwe"),
                                     "spans": keep, "tok_idx": idx,
                                     "n_tok": len(idx), "code_len": len(r["code"])})
                        with lock:
                            stats["ok"] += 1
                            stats["tok"] += len(idx)
            with lock:
                with open(POOL, "a") as f:
                    for rec in recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if k % 50 == 0:
                    el = time.time() - t0
                    print(f"[flaw] {k}/{len(batches)} pairs_ok={stats['ok']} "
                          f"no_span={stats['none']} err={stats['err']} "
                          f"{stats['ok']/el:.2f} pair/s", flush=True)
    print(f"[flaw] DONE ok={stats['ok']} no_span={stats['none']} err={stats['err']} "
          f"tokens={stats['tok']}", flush=True)


def assemble(max_edit=0.55, min_weight=0.0, drop_no_span=True):
    from difflib import SequenceMatcher
    ver = Verifier()
    pool = {}
    for line in open(POOL):
        r = json.loads(line)
        pool[r["pair_id"]] = r
    rows = [json.loads(l) for l in open(f"{D}/token_labels_rl_pairs.jsonl")]
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], {})[r["side"]] = r
    out, st = [], {"kept": 0, "no_span": 0, "edit": 0, "empty": 0}
    for pid, d in by.items():
        p = pool.get(pid)
        if p is None or not p.get("ok"):
            st["no_span" if (p and not p.get("ok")) else "no_span"] += 1
            if drop_no_span:
                continue
        pos, neg = d[0], d[1]
        vids, _ = ver.enc(pos["code"])
        _, off = ver.enc(pos["code"])
        sid = ver.enc(neg["code"])[0]
        r = SequenceMatcher(None, vids, sid, autojunk=False)
        changed = sum((i2 - i1) + (j2 - j1) for t, i1, i2, j1, j2 in r.get_opcodes()
                      if t != "equal")
        if changed / max(1, len(vids) + len(sid)) > max_edit:
            st["edit"] += 1
            continue
        lab = [0] * len(vids)
        wts = [0.0] * len(vids)
        idx = [t for t in (p.get("tok_idx") or []) if t < len(vids)]
        wmax = max([sp["weight"] for sp in (p.get("spans") or [])] or [1.0])
        for t in idx:
            lab[t] = 1
            wts[t] = wmax
        if sum(lab) == 0:
            st["empty"] += 1
            continue
        out.append({"pair_id": pid, "prompt": pos["prompt"], "side": 0,
                    "code": pos["code"], "labels": lab, "weights": wts,
                    "spans": p.get("spans"), "cwe": p.get("cwe")})
        out.append({"pair_id": pid, "prompt": pos["prompt"], "side": 1,
                    "code": neg["code"], "labels": [0] * len(sid)})
        st["kept"] += 1
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[assemble] {OUT} pairs={st['kept']} rows={len(out)} | {st}")
    ln = sorted(sum(1 for x in r["labels"] if x) for r in out if r["side"] == 0)
    if ln:
        print(f"[assemble] 病灶 token 数 min/med/p90/max = {ln[0]}/{ln[len(ln)//2]}/"
              f"{ln[int(len(ln)*0.9)]}/{ln[-1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-edit", type=float, default=0.55)
    ap.add_argument("--min-weight", type=float, default=0.0)
    a = ap.parse_args()
    if a.gen:
        gen(a.limit)
    elif a.assemble:
        assemble(a.max_edit, a.min_weight)
    else:
        ap.print_help()
