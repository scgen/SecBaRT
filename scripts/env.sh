#!/usr/bin/env bash
# Shared configuration for every SecBaRT entry script.  Source it; do not run it.
#
# All paths default to locations inside this repository so that a fresh clone
# works out of the box once the data/checkpoints have been placed (or linked)
# as described in DATA.md.  Every variable can be overridden from the
# environment, e.g.  MODEL_DIR=/models/Qwen2.5-Coder-7B bash scripts/1_sft.sh
set -euo pipefail

SECBART_ROOT="${SECBART_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SECBART_ROOT
export PYTHONPATH="${SECBART_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python3}"
export PYTHON
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORK_DIR="${WORK_DIR:-${SECBART_ROOT}/work}"
export WORK_DIR
mkdir -p "${WORK_DIR}"

# ---- models and data -------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-${SECBART_ROOT}/models/Qwen2.5-Coder-7B}"   # HF base model
SFT_DATA="${SFT_DATA:-${SECBART_ROOT}/data/wcstatic_synthref_merge/train-sft.json}"
TOKEN_LABELS="${TOKEN_LABELS:-${SECBART_ROOT}/data/wcstatic_synthref_merge/token_labels_v7_ord.jsonl}"
SECPLT_CASES="${SECPLT_CASES:-${SECBART_ROOT}/third_party/SecCodePLT_Plus/filtered-test_cases.json}"
CWEVAL_REPO="${CWEVAL_REPO:-${SECBART_ROOT}/third_party/CWEval}"
export MODEL_DIR SFT_DATA TOKEN_LABELS SECPLT_CASES CWEVAL_REPO

# ---- outputs ---------------------------------------------------------------
SFT_OUT="${SFT_OUT:-${WORK_DIR}/sft_wcstatic_synthref_merge}"
HEAD_OUT="${HEAD_OUT:-${WORK_DIR}/token_head_v7_ord}"
RL_OUT="${RL_OUT:-${WORK_DIR}/rl_token_reward}"
export SFT_OUT HEAD_OUT RL_OUT

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_file() { [ -f "$1" ] || die "missing $2: $1"; }
need_dir()  { [ -d "$1" ] || die "missing $2: $1"; }
