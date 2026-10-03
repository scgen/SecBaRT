#!/usr/bin/env bash
# Score existing CWEval raw generations with the official harness.
#
# The official harness (github.com/Co1lin/CWEval) runs inside the public
# docker image ``co1lin/cweval:latest``.  This script stages
# ``<src_eval_dir>/generated_*`` into the container, drives the harness stage by
# stage (parse -> compile -> tests -> merge -> report) with per-stage timeouts,
# and writes ``res_all.json`` + ``report.json`` back to the host.
#
# Usage: bash third_party/cweval/official_reeval.sh <src_eval_dir> [container]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:?usage: official_reeval.sh <src_eval_dir> [container]}"
CONTAINER="${2:-secbart_cweval_reeval}"
IMAGE="${CWEVAL_IMAGE:-co1lin/cweval:latest}"
NUM_PROC="${NUM_PROC:-8}"
CT_DST=/home/ubuntu/CWEval/evals/eval_reeval

SRC="$(realpath -m "${SRC}")"
[ -d "${SRC}/generated_0" ] || { echo "missing ${SRC}/generated_0" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 || docker pull "${IMAGE}"

STAGE_ROOT="$(mktemp -d)"
trap 'rm -rf "${STAGE_ROOT}"; docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true' EXIT

if [ "${ALL_SAMPLES:-0}" = "1" ]; then
  for gd in "${SRC}"/generated_*; do
    [ -d "${gd}" ] || continue
    mkdir -p "${STAGE_ROOT}/$(basename "${gd}")"
    (cd "${gd}" && find . -name '*_raw.*' -exec cp --parents {} "${STAGE_ROOT}/$(basename "${gd}")/" \;)
  done
else
  mkdir -p "${STAGE_ROOT}/generated_0"
  (cd "${SRC}/generated_0" && find . -name '*_raw.*' -exec cp --parents {} "${STAGE_ROOT}/generated_0/" \;)
fi
echo "[reeval] staged $(find "${STAGE_ROOT}" -name '*_raw.*' | wc -l) raw files"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run --name "${CONTAINER}" --rm -d --net host "${IMAGE}" tail -f /dev/null >/dev/null
docker exec "${CONTAINER}" bash -c "rm -rf ${CT_DST} && mkdir -p ${CT_DST}"
for sd in "${STAGE_ROOT}"/generated_*; do
  [ -d "${sd}" ] || continue
  docker cp "${sd}" "${CONTAINER}:${CT_DST}/$(basename "${sd}")"
done
docker exec -u root "${CONTAINER}" bash -c "chown -R ubuntu:ubuntu ${CT_DST}"
docker cp "${HERE}/cweval_stages_driver.py" "${CONTAINER}:/tmp/cweval_stages_driver.py"

PY_IN=/home/ubuntu/miniforge3/envs/cweval/bin/python
CT_PRE="source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && cd /home/ubuntu/CWEval && source .env"
GO_PROXY="${GOPROXY_GO:-https://goproxy.cn,direct}"
CONTAINER_RC=0
stage() {  # stage <timeout_s> <driver args>
  local t="$1"; shift
  docker exec -u ubuntu -e GOPROXY="${GO_PROXY}" "${CONTAINER}" bash -c \
    "${CT_PRE}; timeout -k 10 ${t} ${PY_IN} /tmp/cweval_stages_driver.py $1"
}
kill_orphans() {
  docker exec -u root "${CONTAINER}" bash -c \
    "pkill -f multiprocessing.spawn >/dev/null 2>&1; pkill -f pytest >/dev/null 2>&1; true" || true
}

stage 900  "--stage parse"   || echo "WARN: parse rc=$?"
stage 2400 "--stage compile" || echo "WARN: compile rc=$?"
mapfile -t BATCHES < <(stage 180 "--stage print-batches --num_proc ${NUM_PROC}" | grep -E '^evals/')
echo "[reeval] ${#BATCHES[@]} test batches (<=${NUM_PROC} dirs each)"
i=0
for b in "${BATCHES[@]}"; do
  echo "[reeval] test batch ${i}"
  if ! stage 1500 "--stage tests --dirs \"${b}\""; then
    kill_orphans
    for d in ${b}; do
      stage 600 "--stage tests --dirs \"${d}\"" || {
        kill_orphans
        stage 1800 "--stage one-dir --dirs \"${d}\"" || echo "WARN: dir ${d} rc=$?"
      }
    done
  fi
  i=$((i + 1))
done
kill_orphans
stage 600 "--stage merge --allow-missing-dirs" || CONTAINER_RC=$?
stage 900 "--stage report" || true

docker cp "${CONTAINER}:${CT_DST}/res_all.json" "${SRC}/res_all.json"

"${PYTHON:-python3}" - "${SRC}/res_all.json" "${SRC}/report.json" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))
n = len(res)
func = sum(1 for v in res.values() if (v.get("functional") or [False])[0])
sec = sum(1 for v in res.values() if (v.get("secure") or [False])[0])
fs = sum(1 for v in res.values() if (v.get("func_secure") or [False])[0])
report = {"harness": "official", "n_tasks": n,
          "func@1": round(func / n * 100, 2) if n else None,
          "sec@1": round(sec / n * 100, 2) if n else None,
          "func_sec@1": round(fs / n * 100, 2) if n else None,
          "correct_sec@1": round(fs / func * 100, 2) if func else None,
          "n_func": func}
json.dump(report, open(sys.argv[2], "w"), indent=2)
print(json.dumps(report, indent=2))
PY
echo "[reeval] wrote ${SRC}/res_all.json and ${SRC}/report.json (driver rc=${CONTAINER_RC})"
