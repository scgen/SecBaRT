#!/usr/bin/env bash
# Download the prepared data assets and the base model.
#
# The data is split into three Hugging Face dataset repositories; each can be
# mirrored independently:
#   SECBART_SFT_REPO     (default scgen/SecBaRT-sft)
#   SECBART_LABELS_REPO  (default scgen/SecBaRT-token-labels)
#   SECBART_SCPLT_REPO   (default scgen/SecBaRT-rl-tasks)
#
# Repositories that are not published yet are reported and skipped, so the
# script is usable while the release is being staged.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SFT_REPO="${SECBART_SFT_REPO:-${SECBART_DATA_REPO:-scgen/SecBaRT-sft}}"
LABELS_REPO="${SECBART_LABELS_REPO:-scgen/SecBaRT-token-labels}"
SCPLT_REPO="${SECBART_SCPLT_REPO:-scgen/SecBaRT-rl-tasks}"
BASE_MODEL_REPO="${BASE_MODEL_REPO:-Qwen/Qwen2.5-Coder-7B}"

HF="${HF_CLI:-hf}"
command -v "${HF}" >/dev/null 2>&1 || die "huggingface CLI not found (pip install -U huggingface_hub)"

mkdir -p data/wcstatic_synthref_merge third_party/SecCodePLT_Plus models

try_download() {  # try_download <repo> <file> <local-dir>
  local repo="$1" file="$2" dir="$3"
  if "${HF}" download "${repo}" "${file}" --repo-type dataset --local-dir "${dir}" 2>/dev/null; then
    log "downloaded ${repo}:${file}"
  else
    log "SKIP ${repo}:${file} (repository or file not published yet)"
  fi
}

log "downloading supervised pool"
try_download "${SFT_REPO}" train-sft.json data/wcstatic_synthref_merge

log "downloading token labels"
try_download "${LABELS_REPO}" token_labels_v7_ord.jsonl data/wcstatic_synthref_merge

log "downloading SecCodePLT+ RL cases"
try_download "${SCPLT_REPO}" filtered-test_cases.json third_party/SecCodePLT_Plus

log "downloading base model"
"${HF}" download "${BASE_MODEL_REPO}" --local-dir models/Qwen2.5-Coder-7B

log "cloning CWEval"
if [ ! -d third_party/CWEval/.git ]; then
  git clone https://github.com/Co1lin/CWEval third_party/CWEval
fi

log "done; run bash scripts/0_setup.sh to verify"
