#!/usr/bin/env bash
# Stage I: bottleneck supervised fine-tuning on the 40,360-triple pool.
#
# Trains the vulnerable reconstruction (through the bottleneck) and the secure
# generation jointly.  The resulting ``merged_hf_model`` is the seed for both
# the token head (Stage II) and dual-arm RL (Stage III).
set -euo pipefail
source "$(dirname "$0")/env.sh"

need_file "${SFT_DATA}" "SFT training file (train-sft.json)"
need_dir  "${MODEL_DIR}" "base model directory (Qwen2.5-Coder-7B)"

# A validation split is only used for loss logging; the released pipeline
# trains for one epoch and always keeps the final checkpoint.  If you have the
# original held-out file, pass VAL_DATA=/path/to/test-sft.json instead.
VAL_DATA="${VAL_DATA:-${WORK_DIR}/val_525.json}"
if [ ! -f "${VAL_DATA}" ]; then
  log "creating a 525-example validation split from ${SFT_DATA}"
  "${PYTHON}" - "${SFT_DATA}" "${VAL_DATA}" <<'PY'
import json, random, sys
rows = json.load(open(sys.argv[1]))
rng = random.Random(42)
rng.shuffle(rows)
json.dump(rows[:525], open(sys.argv[2], "w"), ensure_ascii=False)
print(f"wrote {sys.argv[2]} ({min(525, len(rows))} rows)")
PY
fi

mkdir -p "${SFT_OUT}"
log "SFT: gpu=${CUDA_VISIBLE_DEVICES} out=${SFT_OUT}"

"${PYTHON}" -m secbart.train_7b_fullft \
  --model_path "${MODEL_DIR}" \
  --train_data "${SFT_DATA}" \
  --val_data "${VAL_DATA}" \
  --output_dir "${SFT_OUT}" \
  --max_train -1 --max_val 525 \
  --train_batch_size "${SFT_BATCH:-8}" --micro_batch_size "${SFT_MICRO:-1}" \
  --lr "${SFT_LR:-1e-5}" --emb_lr_mult 50.0 --emb_init mean \
  --lora_rank "${SFT_LORA_RANK:-0}" --n_vuln 4 --max_length 1024 --sec_up 3.0 \
  --optimizer "${SFT_OPTIMIZER:-paged_adamw8bit}" \
  --seed "${SEED:-42}"

log "SFT done: ${SFT_OUT}/merged_hf_model"
