#!/usr/bin/env bash
# Stage II: train the token-level security scoring head on the frozen SFT seed.
#
# The head is a one-hidden-layer MLP (LayerNorm -> Linear -> GELU -> Dropout ->
# scalar) trained with the ordinal objective over the three-valued labels
# (0 = unsafe, 0.5 = neutral, 1 = safe).  It is frozen afterwards and reused
# during RL; no separate reward model is trained.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SEED_MODEL="${SEED_MODEL:-${SFT_OUT}/merged_hf_model}"
need_file "${TOKEN_LABELS}" "token-label file (token_labels_v7_ord.jsonl)"
need_dir  "${SEED_MODEL}" "SFT seed model"

mkdir -p "${HEAD_OUT}"
log "token head: data=${TOKEN_LABELS} seed=${SEED_MODEL} out=${HEAD_OUT}"

"${PYTHON}" -m secbart.train_token_head_rl \
  --data "${TOKEN_LABELS}" --model "${SEED_MODEL}" --out_dir "${HEAD_OUT}" \
  --gpu 0 --archs mlp --epochs "${HEAD_EPOCHS:-40}" --lr "${HEAD_LR:-1e-3}" \
  --batch_size "${HEAD_BATCH:-32}" --seed "${SEED:-42}" --cap_pairs 0 \
  --loss ordinal --lam_ord 5.0 --margin_ord 0.3 --save_ckpt_every 5

cp -f "${HEAD_OUT}/token_head_rl_mlp_ep40.pt" "${HEAD_OUT}/token_head_rl.pt"
printf 'ordinal\nepoch=40\n' > "${HEAD_OUT}/LINEAGE.txt"
log "token head done: ${HEAD_OUT}/token_head_rl.pt"
