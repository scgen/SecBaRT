#!/usr/bin/env bash
# Optional SecBaRT variant: dual-arm PPO that adds the frozen token head's
# position-aligned security reward to the secure arm (Section "Token-Level
# Reward Computation and Integration").  The main table's final row uses this
# variant; the pure sequence-level configuration is scripts/3_rl_dualarm.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SEED_MODEL="${SEED_MODEL:-${RL_OUT}/merged_hf_model}"
TOKEN_HEAD="${TOKEN_HEAD:-${HEAD_OUT}/token_head_rl.pt}"
PPO_OUT="${PPO_OUT:-${WORK_DIR}/rl_dualarm_tkh_s768}"
need_dir  "${SEED_MODEL}" "dual-arm RL seed model (run scripts/3_rl_dualarm.sh first)"
need_file "${TOKEN_HEAD}" "frozen token head (run scripts/2_token_head.sh first)"
need_file "${SECPLT_CASES}" "SecCodePLT+ RL task file"

log "dual-arm PPO + token head: out=${PPO_OUT} w_head=${W_HEAD:-0.5}"

# The wrapper only adds an ndarray-safe serialiser for one SecCodePLT+ case
# (bcce7d57); arguments are passed through unchanged.
SECPLT_CASES="${SECPLT_CASES}" \
"${PYTHON}" -m secbart.run_rl_ppo_tkh_ndsafe \
  --seed_model "${SEED_MODEL}" --token_head "${TOKEN_HEAD}" --output_dir "${PPO_OUT}" \
  --dataset "${RL_DATASET:-secodeplt_filtered}" --steps "${RL_STEPS:-768}" \
  --batch "${RL_BATCH:-1}" --k "${RL_K:-4}" --lr "${RL_LR:-3e-6}" \
  --kl_beta "${RL_KL:-0.3}" --lora_rank 0 --seed "${SEED:-42}" --temp 0.8 \
  --w_head "${W_HEAD:-0.5}" --gamma 1.0 --lam 0.95 --clip_eps 0.2 --vf_coef 0.5 \
  --tkh_token_norm raw_centered --ref_8bit

log "dual-arm PPO + token head done: ${PPO_OUT}/merged_hf_model"
