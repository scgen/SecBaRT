#!/usr/bin/env bash
# Stage IV evaluation (main result): generate on CWEval-119 with the bottleneck
# server and score with the official CWEval harness.
#
# Usage: bash scripts/4_eval_cweval.sh [model_path] [gpu] [port]
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL_PATH="${1:-${RL_OUT}/merged_hf_model}"
GPU="${2:-${CUDA_VISIBLE_DEVICES}}"
PORT="${3:-8123}"
[ -d "${MODEL_PATH}" ] || MODEL_PATH="${SFT_OUT}/merged_hf_model"
[ -d "${MODEL_PATH}" ] || MODEL_PATH="${MODEL_DIR}"
need_dir "${MODEL_PATH}" "model to evaluate"
need_dir "${CWEVAL_REPO}/benchmark" "CWEval checkout (third_party/CWEval)"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-Coder-7B}"
EVAL_ROOT="${EVAL_ROOT:-${WORK_DIR}/eval}"
EVAL_DIR="${EVAL_ROOT}/cweval/direct"
mkdir -p "${EVAL_DIR}"
SERVER_LOG="${EVAL_ROOT}/vllm_cweval.log"

log "CWEval generation: model=${MODEL_PATH} gpu=${GPU} port=${PORT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m secbart.vllm_bottleneck_server \
  --model_path "${MODEL_PATH}" --port "${PORT}" --n_vuln "${N_VULN:-4}" \
  --model_name "${MODEL_NAME}" --chat \
  --gpu_memory_utilization "${GPU_MEM:-0.9}" > "${SERVER_LOG}" 2>&1 &
SPID=$!
trap 'kill ${SPID} >/dev/null 2>&1 || true' EXIT

ready=0
for _ in $(seq 1 600); do
  if curl -s -m 3 "http://127.0.0.1:${PORT}/v1/models" | grep -q "${MODEL_NAME}"; then
    if grep -q "tokenizer ready" "${SERVER_LOG}"; then ready=1; break; fi
  fi
  kill -0 "${SPID}" 2>/dev/null || { echo "server died:"; tail -30 "${SERVER_LOG}"; exit 1; }
  sleep 3
done
[ "${ready}" -eq 1 ] || { echo "vLLM server startup timeout"; tail -30 "${SERVER_LOG}"; exit 1; }
log "server ready"

"${PYTHON}" scripts/cweval_gen.py \
  --cweval-repo "${CWEVAL_REPO}" --eval-path "${EVAL_DIR}" \
  --model-name "${MODEL_NAME}" --api-port "${PORT}" \
  --num-proc "${NUM_PROC:-16}" --n 1 --temperature 0 --prompt-name direct

kill "${SPID}" >/dev/null 2>&1 || true
SPID=""

log "official CWEval scoring in docker"
bash third_party/cweval/official_reeval.sh "${EVAL_DIR}"

"${PYTHON}" scripts/collect_metrics.py --eval-dir "${EVAL_DIR}" \
  --out "$(dirname "${MODEL_PATH}")/metric.json" || true
log "CWEval done: ${EVAL_DIR}/report.json"
