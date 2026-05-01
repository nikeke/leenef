#!/usr/bin/env bash
# Run B4 MLP panels (MNIST, Fashion, CIFAR-10) in parallel.
#
# Each dataset spawns sl_phase_runner with OMP_NUM_THREADS=4 to share
# the 12-core CPU.  Logs land under results/sl_clean/logs/.
#
# Usage:
#   bash benchmarks/sl_clean/run_b4_panels.sh [seeds]
#   SEEDS=0-4 bash benchmarks/sl_clean/run_b4_panels.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/venv/bin/python}"
SEEDS_ARG="${1:-${SEEDS:-0-4}}"

LOG_DIR="${REPO_ROOT}/results/sl_clean/logs"
mkdir -p "${LOG_DIR}"

declare -a EXPS=(b4_mlp_mnist b4_mlp_fashion b4_mlp_cifar10)
declare -a PIDS=()

for exp in "${EXPS[@]}"; do
  out_log="${LOG_DIR}/${exp}.log"
  echo "[run_b4_panels] launching ${exp} (seeds=${SEEDS_ARG}) -> ${out_log}"
  (
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      SEEDS="${SEEDS_ARG}" PYTHON="${PYTHON}" \
      bash benchmarks/sl_clean/sl_hp_driver.sh "${exp}"
  ) > "${out_log}" 2>&1 &
  PIDS+=($!)
done

echo "[run_b4_panels] launched PIDs: ${PIDS[*]}"
echo "[run_b4_panels] sentinel will be touched on success: results/sl_clean/b4_panels_done"
rm -f "${REPO_ROOT}/results/sl_clean/b4_panels_done"

ALL_OK=1
for pid in "${PIDS[@]}"; do
  if wait "${pid}"; then
    echo "[run_b4_panels] PID ${pid} OK"
  else
    echo "[run_b4_panels] PID ${pid} FAILED"
    ALL_OK=0
  fi
done

if [ "${ALL_OK}" -eq 1 ]; then
  date -Is > "${REPO_ROOT}/results/sl_clean/b4_panels_done"
  echo "[run_b4_panels] all panels complete"
  exit 0
else
  echo "[run_b4_panels] one or more panels FAILED"
  exit 1
fi
