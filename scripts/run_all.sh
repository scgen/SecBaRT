#!/usr/bin/env bash
# Run the full pipeline: SFT -> token head -> dual-arm RL -> CWEval evaluation.
# Each stage can be skipped by setting SKIP_<STAGE>=1.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

[ "${SKIP_SFT:-0}" = "1" ]   || bash "${HERE}/1_sft.sh"
[ "${SKIP_HEAD:-0}" = "1" ]  || bash "${HERE}/2_token_head.sh"
[ "${SKIP_RL:-0}" = "1" ]    || bash "${HERE}/3_rl_token_reward.sh"
[ "${SKIP_EVAL:-0}" = "1" ]  || bash "${HERE}/4_eval_cweval.sh"
