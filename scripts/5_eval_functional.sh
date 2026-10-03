#!/usr/bin/env bash
# Optional functional evaluation on HumanEval+ and MBPP+ (EvalPlus).
#
# These numbers are reported next to the security results to show that the
# security training does not damage general code-generation ability.  The
# script uses the EvalPlus CLI against the same OpenAI-compatible bottleneck
# server; install it with ``pip install evalplus``.
#
# Usage: bash scripts/5_eval_functional.sh [model_path] [gpu] [port]
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL_PATH="${1:-${RL_OUT}/merged_hf_model}"
GPU="${2:-${CUDA_VISIBLE_DEVICES}}"
PORT="${3:-8124}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-Coder-7B}"
EVAL_ROOT="${EVAL_ROOT:-${WORK_DIR}/eval}"
mkdir -p "${EVAL_ROOT}"

command -v evalplus >/dev/null 2>&1 || die "evalplus not installed (pip install evalplus)"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m secbart.vllm_bottleneck_server \
  --model_path "${MODEL_PATH}" --port "${PORT}" --n_vuln "${N_VULN:-4}" \
  --model_name "${MODEL_NAME}" --chat \
  --gpu_memory_utilization "${GPU_MEM:-0.9}" > "${EVAL_ROOT}/vllm_functional.log" 2>&1 &
SPID=$!
trap 'kill ${SPID} >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 600); do
  curl -s -m 3 "http://127.0.0.1:${PORT}/v1/models" | grep -q "${MODEL_NAME}" && break
  kill -0 "${SPID}" 2>/dev/null || { tail -30 "${EVAL_ROOT}/vllm_functional.log"; exit 1; }
  sleep 3
done

for ds in humaneval mbpp; do
  out="${EVAL_ROOT}/${ds}plus"
  mkdir -p "${out}"
  log "EvalPlus ${ds}+"
  evalplus.evaluate --model "${MODEL_NAME}" --dataset "${ds}" \
    --backend openai --base-url "http://127.0.0.1:${PORT}/v1" \
    --greedy --output-dir "${out}" || \
    die "EvalPlus run failed; see the CLI help for the backend flags of your version"
done

"${PYTHON}" scripts/collect_metrics.py --eval-dir "${EVAL_ROOT}/cweval/direct" \
  --functional-root "${EVAL_ROOT}" --out "${WORK_DIR}/metric.json" || true
log "functional evaluation done: ${EVAL_ROOT}"
