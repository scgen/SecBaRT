#!/usr/bin/env python3
# In-container patch for the official CWEval reeval harness: when the model output
# has no fenced code block, strip a natural-language/Chinese analysis preamble
# before using the raw text as code (cot models emit think-style analysis first).
import sys

P = "/home/ubuntu/CWEval/cweval/evaluate.py"
H = "/tmp/cweval_strip_helper.py"
src = open(P, encoding="utf-8").read()
helper = open(H, encoding="utf-8").read()

if "_strip_analysis_preamble" in src:
    print("PATCH-ALREADY")
    sys.exit(0)

old = "            raw_code = raw_str\n"
new = "            raw_code = _strip_analysis_preamble(raw_str)\n"
if old not in src:
    raise SystemExit("PATCH-FAIL: anchor not found: raw_code = raw_str")
src = src.replace(old, new, 1)

# insert helper BEFORE the __main__ fire block so it exists when pipeline runs
marker = "\nif __name__ == '__main__':\n"
assert marker in src, "PATCH-FAIL: __main__ marker not found"
src = src.replace(marker, "\n" + helper.rstrip() + "\n" + marker, 1)
open(P, "w", encoding="utf-8").write(src)
print("PATCH-OK")
