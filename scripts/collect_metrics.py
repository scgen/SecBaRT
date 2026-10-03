#!/usr/bin/env python3
"""Collect SecBaRT evaluation results into a single ``metric.json``.

Reads the CWEval ``report.json`` produced by
``third_party/cweval/official_reeval.sh`` and, when present, the EvalPlus
``result.json`` files for HumanEval+ and MBPP+.

Usage:
    python scripts/collect_metrics.py --eval-dir work/eval/cweval/direct \
        --functional-root work/eval --out work/metric.json
"""
import argparse
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="CWEval direct/ directory")
    ap.add_argument("--functional-root", default="", help="root containing humanevalplus/ and mbppplus/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metrics = {}
    cweval = read_json(Path(args.eval_dir) / "report.json")
    if cweval:
        metrics["cweval"] = cweval

    if args.functional_root:
        root = Path(args.functional_root)
        for key, sub in (("humanevalplus", "humanevalplus"), ("mbppplus", "mbppplus")):
            res = read_json(root / sub / "result.json")
            if not res:
                continue
            # EvalPlus result.json: {model: {"pass@1": ...}} or {"pass@1": ...}
            if isinstance(res, dict) and "pass@1" not in res:
                vals = [v for v in res.values() if isinstance(v, dict) and "pass@1" in v]
                if vals:
                    res = vals[0]
            if isinstance(res, dict) and "pass@1" in res:
                metrics[key] = {"pass@1": round(float(res["pass@1"]) * 100, 2)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
