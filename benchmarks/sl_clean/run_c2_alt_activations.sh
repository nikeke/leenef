#!/usr/bin/env bash
# Run only the new C2 activation cells (step, square) on MNIST,
# Fashion-MNIST, and CIFAR-10 in parallel.
#
# Uses the sibling leenef venv for dependencies while forcing imports from
# this worktree through PYTHONPATH. Logs land under results/sl_clean/logs/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/amnesia/Persistent/src/leenef/venv/bin/python}"
SEEDS_ARG="${1:-${SEEDS:-0-9}}"
VALUES_ARG="${VALUES:-step+g0.5_2.0,step+g1.0_1.0,step+g0.1_5.0,square+g0.5_2.0,square+g1.0_1.0,square+g0.1_5.0}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${REPO_ROOT}/results/sl_clean/logs"
DONE="${REPO_ROOT}/results/sl_clean/c2_alt_activations_done"
mkdir -p "${LOG_DIR}"
rm -f "${DONE}"

declare -a EXPS=(c2_activation_mnist c2_activation_fashion c2_activation_cifar10)
declare -a PIDS=()

for exp in "${EXPS[@]}"; do
  out_log="${LOG_DIR}/c2_alt_${exp}.log"
  echo "[run_c2_alt_activations] launching ${exp} (seeds=${SEEDS_ARG}; values=${VALUES_ARG}) -> ${out_log}"
  (
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      SEEDS="${SEEDS_ARG}" VALUES="${VALUES_ARG}" PYTHON="${PYTHON}" \
      bash benchmarks/sl_clean/sl_hp_driver.sh "${exp}"
  ) > "${out_log}" 2>&1 &
  PIDS+=($!)
done

echo "[run_c2_alt_activations] launched PIDs: ${PIDS[*]}"
echo "[run_c2_alt_activations] sentinel will be touched on success: ${DONE#${REPO_ROOT}/}"

ALL_OK=1
for pid in "${PIDS[@]}"; do
  if wait "${pid}"; then
    echo "[run_c2_alt_activations] PID ${pid} OK"
  else
    echo "[run_c2_alt_activations] PID ${pid} FAILED"
    ALL_OK=0
  fi
done

if [ "${ALL_OK}" -eq 1 ]; then
  date -Is > "${DONE}"
  echo "[run_c2_alt_activations] all panels complete"
  exit 0
fi

echo "[run_c2_alt_activations] one or more panels FAILED"
exit 1
