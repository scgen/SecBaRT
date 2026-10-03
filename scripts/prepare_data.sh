#!/usr/bin/env bash
# Download the prepared data assets and the base model.
#
# Override the hosting repositories if you mirror them:
#   SECBART_DATA_REPO=my-org/SecBaRT-data bash scripts/prepare_data.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

DATA_REPO="${SECBART_DATA_REPO:-<ORG>/SecBaRT-data}"
SCPLT_REPO="${SECBART_SCPLT_REPO:-<ORG>/SecBaRT-SecCodePLT-plus}"
BASE_MODEL_REPO="${BASE_MODEL_REPO:-Qwen/Qwen2.5-Coder-7B}"

command -v huggingface-cli >/dev/null 2>&1 || die "huggingface-cli not found (pip install huggingface_hub)"

mkdir -p data/wcstatic_synthref_merge third_party/SecCodePLT_Plus models

log "downloading supervised pool and token labels"
huggingface-cli download "${DATA_REPO}" --local-dir data/wcstatic_synthref_merge \
  train-sft.json token_labels_v7_ord.jsonl

log "downloading SecCodePLT+ RL cases"
huggingface-cli download "${SCPLT_REPO}" --local-dir third_party/SecCodePLT_Plus \
  filtered-test_cases.json tests.tar.gz 2>/dev/null || \
  huggingface-cli download "${SCPLT_REPO}" --local-dir third_party/SecCodePLT_Plus \
    filtered-test_cases.json

log "downloading base model"
huggingface-cli download "${BASE_MODEL_REPO}" --local-dir models/Qwen2.5-Coder-7B

log "cloning CWEval"
if [ ! -d third_party/CWEval/.git ]; then
  git clone https://github.com/Co1lin/CWEval third_party/CWEval
fi

log "done; run bash scripts/0_setup.sh to verify"
