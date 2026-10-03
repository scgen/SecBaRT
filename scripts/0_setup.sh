#!/usr/bin/env bash
# Install the Python dependencies and report which external assets are present.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
PIP="${PIP:-$PYTHON -m pip}"

$PIP install -r requirements.txt

echo
echo "Dependency check:"
$PYTHON - <<'PY'
import importlib
mods = ["torch", "transformers", "peft", "datasets", "vllm", "fastapi", "uvicorn", "numpy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  [ok]   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  [MISS] {m}: {type(e).__name__}: {e}")
PY

echo
echo "External assets (download instructions in DATA.md):"
for p in data/wcstatic_synthref_merge/train-sft.json \
         data/wcstatic_synthref_merge/token_labels_v7_ord.jsonl \
         models/Qwen2.5-Coder-7B \
         third_party/SecCodePLT_Plus/filtered-test_cases.json \
         third_party/CWEval/benchmark; do
  if [ -e "$p" ]; then echo "  [ok]   $p"; else echo "  [MISS] $p"; fi
done

command -v docker >/dev/null 2>&1 && echo "  [ok]   docker" || echo "  [MISS] docker (required for CWEval scoring)"
echo
echo "Docker image for CWEval scoring: co1lin/cweval:latest"
