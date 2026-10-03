#!/usr/bin/env bash
# Stage III: dual-arm RL that starts from the SFT seed and adds the frozen
# token head's position-aligned security reward to the secure arm.  The
# sequence-level outcome reward alone is not enough: it cannot tell which
# generation decisions were responsible for the outcome, so the token-level
# reward is what turns the executable feedback into local credit.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SEED_MODEL="${SEED_MODEL:-${SFT_OUT}/merged_hf_model}"
TOKEN_HEAD="${TOKEN_HEAD:-${HEAD_OUT}/token_head_rl.pt}"
PPO_OUT="${PPO_OUT:-${RL_OUT}}"
need_dir  "${SEED_MODEL}" "SFT seed model (run scripts/1_sft.sh first)"
need_file "${TOKEN_HEAD}" "frozen token head (run scripts/2_token_head.sh first)"
need_file "${SECPLT_CASES}" "SecCodePLT+ RL task file"

log "token-reward dual-arm RL: seed=${SEED_MODEL} out=${PPO_OUT} w_head=${W_HEAD:-0.5}"

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

log "token-reward dual-arm RL done: ${PPO_OUT}/merged_hf_model"
