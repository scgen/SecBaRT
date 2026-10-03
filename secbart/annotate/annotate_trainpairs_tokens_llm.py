#!/usr/bin/env python3
"""wcstatic_synthref_merge 训练对的 **三分类 token 标注** (irrel / vuln / safe), 09-12。

为什么需要
----------
头动物园的离线表显示两个口径正交:
  H&T (版本内三项序数 A/B/C) 榜首 v5_span_l20 = 0.6534,
  bin_AUC (病灶 vs 非病灶)   榜首 v3_llm      = 0.9276。
四个训练期指标 (vul_hit / vul_span_hit / row_auc) 与 bin_AUC 正相关 (+0.46~+0.73)、
与 H&T/ord_exc **负相关**。根因在数据: 训练对 `side=1` (修复版) 的 `labels` **全零**,
头从没学过"防御代码 = 安全" —— B = P(safe > irrel) 被结构性压到 0.16~0.44,
这正是 H&T 上限 ~0.65 的直接原因, 也是 14 个序列级 RL 臂 Δ≈0 的同源解释
(加防御代码的 token 反被头判低分)。

本脚本用 **与 CWEval 评测同一口径**(同一套 prompt 模板 + 同一个 TokenTagger) 给
13,998 个训练对的两版各标三类, 产 `token_labels_v7_ord.jsonl`, 供
`train_token_head_rl.py --loss ordinal` 做**版本内成对序数**训练。

不改任何共享脚本
----------------
`call_llm` / `TokenTagger` / `SYS` / `ONE` / `TAIL` / `irrel_strength` 全部 **import**
自 `annotate_cweval_tokens_llm.py`(该文件一字不动), 本文件只换 `load_cases()` 与
`assemble()` 两头。

用法 (repository root)
-----------------
  python3 -m secbart.annotate.annotate_trainpairs_tokens_llm --gen            # 抽标注 (断点续跑)
  python3 -m secbart.annotate.annotate_trainpairs_tokens_llm --assemble       # 组装成训练标签
env: ORD_WORKERS (默认 16), ORD_BATCH (每请求 case 数, 默认 4)
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
ROOT = os.path.dirname(os.path.dirname(_HERE))

# 复用 CWEval 标注器的全部基础设施 (该文件不改) —— 口径一致是本任务的全部意义
from secbart.annotate.annotate_cweval_tokens_llm import (  # noqa: E402
    call_llm, TokenTagger, SYS, ONE, TAIL, irrel_strength,
)

# [release] data paths are environment-driven (default: <repo>/data/wcstatic_synthref_merge)
D = os.environ.get("SECBART_WORK_DIR", os.path.join(ROOT, "data", "wcstatic_synthref_merge"))
SRC_LABELS = os.environ.get("SECBART_PAIR_LABELS",
                            os.path.join(D, "token_labels_v5_v3fix.jsonl"))  # 13,998 pairs
POOL = os.environ.get("SECBART_ORD_POOL", os.path.join(D, "ord_pool.jsonl"))
ASSEMBLED = os.environ.get("SECBART_ORD_OUT", os.path.join(D, "token_labels_v7_ord.jsonl"))
IRREL_D0 = 60          # 与 CWEval 标注器同尺度 (字符)

# 训练对没有任务描述 (`prompt` 字段是字面量 '...'), 用占位符说明, 不编造
NO_TASK = "(not provided — infer the required behavior from the two implementations)"


def guess_lang(code):
    """训练对无 language 列, 从代码轻量推断 (只进 prompt 的 ``` 围栏, 不影响标签)."""
    if "<?php" in code or "$_GET" in code or "$_POST" in code:
        return "php"
    if re.search(r"^\s*#include\s*[<\"]", code, re.M) or "std::" in code:
        return "cpp"
    if re.search(r"^\s*(def|import|from)\s+\w", code, re.M) or "self." in code:
        return "python"
    if re.search(r"\bfn\s+\w+\s*[(<]", code) or "let mut " in code:
        return "rust"
    if "public class " in code or "System.out" in code:
        return "java"
    if re.search(r"\bfunc\s+\w+\s*\(", code) or "package main" in code:
        return "go"
    if "function " in code or "=>" in code or re.search(r"\b(const|let|var)\s+\w", code):
        return "javascript"
    return "c"


def load_cases(cap=0):
    """13,998 个训练对 → 标注用例 (A=修复版 side=1, B=漏洞版 side=0)。"""
    byp = {}
    for line in open(SRC_LABELS):
        r = json.loads(line)
        byp.setdefault(r["pair_id"], {})[r["side"]] = r
    pids = sorted(byp)
    if cap:
        pids = pids[:cap]
    out = []
    for pid in pids:
        p = byp[pid]
        if 0 not in p or 1 not in p:
            continue
        v, s = p[0]["code"], p[1]["code"]
        if not v.strip() or not s.strip():
            continue
        lang = guess_lang(s) or guess_lang(v)
        out.append({"case_id": f"tp{pid}", "pair_id": pid, "language": lang,
                    "secure": s, "vulnerable": v, "prompt": NO_TASK,
                    "cwe": p[0].get("cwe") or "unknown"})
    return out


def build_prompt(batch):
    parts = []
    for i, r in enumerate(batch):
        parts.append(ONE.format(i=i, lang=r["language"], cwe=r["cwe"],
                                prompt=r["prompt"], sec=r["secure"], vul=r["vulnerable"]))
    return "".join(parts) + TAIL


def gen(limit=0, cap=0):
    tg = TokenTagger()
    cases = load_cases(cap)
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
    nb = int(os.environ.get("ORD_BATCH", "4"))
    workers = int(os.environ.get("ORD_WORKERS", "16"))
    print(f"[ord] cases={len(cases)} done={len(done)} todo={len(todo)} "
          f"batch={nb} workers={workers}", flush=True)
    batches = [todo[i:i + nb] for i in range(0, len(todo), nb)]
    lock = threading.Lock()
    stats = {"ok": 0, "none": 0, "err": 0, "n_sec": 0, "n_vul": 0, "empty_sec": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call_llm, build_prompt(b)): b for b in batches}
        for k, fut in enumerate(as_completed(futs), 1):
            b = futs[fut]
            res = fut.result()
            recs = []
            if "err" in res:
                recs = [{"case_id": r["case_id"], "err": res["err"][:200], "ok": False}
                        for r in b]
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
                        recs.append({"case_id": r["case_id"], "pair_id": r["pair_id"],
                                     "ok": True, "lang": r["language"],
                                     "cwe": p.get("cwe") or r["cwe"],
                                     "secure_spans": skeep, "secure_tok": sorted(shits),
                                     "secure_w": {str(t): shits[t] for t in shits},
                                     "vul_spans": vkeep, "vul_tok": sorted(vhits),
                                     "vul_w": {str(t): vhits[t] for t in vhits},
                                     "n_sec": len(sids), "n_vul": len(vids),
                                     "trunc_sec": strunc, "trunc_vul": vtrunc})
                        with lock:
                            stats["ok"] += 1
                            stats["n_sec"] += len(shits)
                            stats["n_vul"] += len(vhits)
                            stats["empty_sec"] += 0 if shits else 1
            with lock:
                with open(POOL, "a") as f:
                    for rec in recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if k % 20 == 0:
                    el = time.time() - t0
                    print(f"[ord] {k}/{len(batches)} ok={stats['ok']} none={stats['none']} "
                          f"err={stats['err']} sec没标={stats['empty_sec']} "
                          f"{stats['ok']/max(el,1e-9):.2f} case/s", flush=True)
    print(f"[ord] DONE ok={stats['ok']} none={stats['none']} err={stats['err']} "
          f"sec_tok={stats['n_sec']} vul_tok={stats['n_vul']} "
          f"sec空={stats['empty_sec']} 用时={time.time()-t0:.0f}s", flush=True)


def assemble(src=None, out=None):
    """原标签 + 三分类 → token_labels_v7_ord.jsonl (原键一字不动, 只新增 ord_*)。"""
    src = src or SRC_LABELS
    out_path = out or ASSEMBLED
    pool = {}
    for line in open(POOL):
        r = json.loads(line)
        if r.get("ok"):
            pool[r["case_id"]] = r
    tg = TokenTagger()
    rows, st = [], {"kept": 0, "nopool": 0, "no_sec": 0, "no_vul": 0, "both": 0}
    for line in open(src):
        r = json.loads(line)
        p = pool.get(f"tp{r['pair_id']}")
        if p is None:
            st["nopool"] += 1
            continue
        ids, off = tg.enc(r["code"])
        hit_tok = p["secure_tok"] if r["side"] == 1 else p["vul_tok"]
        cls_hit = "safe" if r["side"] == 1 else "vuln"
        wts = p["secure_w"] if r["side"] == 1 else p["vul_w"]
        hits = {int(t): float(v) for t, v in wts.items() if int(t) < len(ids)}
        if not hits:
            st["no_sec" if r["side"] == 1 else "no_vul"] += 1
        cls = [cls_hit if t in hits else "irrel" for t in range(len(ids))]
        strength = [0.0] * len(ids)
        for t, w in hits.items():
            strength[t] = w
        ir = irrel_strength(len(ids), hits, off)
        for t in range(len(ids)):
            if cls[t] == "irrel":
                strength[t] = ir[t]
        out = dict(r)
        out.update({
            "ord_cls": cls,
            "ord_strength": [round(x, 3) for x in strength],
            "ord_ids": ids,
            "ord_src": "llm_v7_pairwise",
        })
        rows.append(out)
        st["kept"] += 1
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    byp = {}
    for r in rows:
        byp.setdefault(r["pair_id"], {})[r["side"]] = r
    for p in byp.values():
        if len(p) == 2:
            has_s = any(x == "safe" for x in p[1]["ord_cls"])
            has_v = any(x == "vuln" for x in p[0]["ord_cls"])
            st["both"] += 1 if (has_s and has_v) else 0
    print(f"[assemble] {out_path} rows={len(rows)} pairs={len(byp)} | {st}")
    print(f"[assemble] 可用对(两版各自有标注) = {st['both']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=int, default=0, help="只处理前 N 对 (0=全部)")
    ap.add_argument("--src", default="", help="assemble 输入 (默认 token_labels_v5_v3fix.jsonl)")
    ap.add_argument("--out", default="", help="assemble 输出 (默认 token_labels_v7_ord.jsonl)")
    a = ap.parse_args()
    if a.gen:
        gen(a.limit, a.cap)
    elif a.assemble:
        assemble(a.src or None, a.out or None)
    else:
        ap.print_help()
