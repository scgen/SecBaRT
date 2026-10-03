#!/usr/bin/env bash
# Stage III: dual-arm reinforcement learning on execution feedback.
#
# Two arms share the policy: the secure arm reads prompt + bottleneck tokens +
# secure sentinel and maximises functional-and-secure programs; the vulnerable
# arm reads prompt + bottleneck tokens and maximises functional-but-vulnerable
# programs.  Rewards are computed by running the generated programs against the
# SecCodePLT+ functional and security tests.  Advantages are group-normalised
# (GRPO); this is the configuration behind the main result checkpoint.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SEED_MODEL="${SEED_MODEL:-${SFT_OUT}/merged_hf_model}"
need_dir  "${SEED_MODEL}" "SFT seed model"
need_file "${SECPLT_CASES}" "SecCodePLT+ RL task file"

mkdir -p "${RL_OUT}"
log "dual-arm RL: dataset=${RL_DATASET:-secodeplt_filtered} steps=${RL_STEPS:-768} out=${RL_OUT}"

SECPLT_CASES="${SECPLT_CASES}" \
"${PYTHON}" -m secbart.train_7b_rl_dualarm \
  --seed_model "${SEED_MODEL}" --output_dir "${RL_OUT}" \
  --dataset "${RL_DATASET:-secodeplt_filtered}" --lora_rank 0 \
  --steps "${RL_STEPS:-768}" --batch "${RL_BATCH:-4}" --k "${RL_K:-4}" \
  --lr "${RL_LR:-3e-6}" --kl_beta "${RL_KL:-0.3}" --temp "${RL_TEMP:-0.8}" \
  --seed "${SEED:-42}" --n_vuln 4

log "dual-arm RL done: ${RL_OUT}/merged_hf_model"
