#!/usr/bin/env bash
# Driver for B3 encoder-strategy panels (MNIST + Fashion + CIFAR-10).
# Runs the three datasets in parallel with bounded threads each.
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-$PWD/venv/bin/python}"
RESULTS_DIR="results/sl_clean"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOGS_DIR}"

run_one() {
  local exp="$1"
  local log="${LOGS_DIR}/b3_${exp#b3_strategies_}.log"
  echo "[$(date -u +%H:%M:%S)] starting ${exp} -> ${log}"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTHON="${PYTHON}" SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh "${exp}" \
    > "${log}" 2>&1
  echo "[$(date -u +%H:%M:%S)] ${exp} done"
}

run_one b3_strategies_mnist &
PID_MNIST=$!
run_one b3_strategies_fashion &
PID_FASHION=$!
run_one b3_strategies_cifar10 &
PID_CIFAR=$!

echo "B3 launched: mnist=${PID_MNIST} fashion=${PID_FASHION} cifar=${PID_CIFAR}"
wait "${PID_MNIST}" || echo "mnist failed"
wait "${PID_FASHION}" || echo "fashion failed"
wait "${PID_CIFAR}" || echo "cifar failed"

touch "${RESULTS_DIR}/b3_panels_done"
echo "[$(date -u +%H:%M:%S)] all B3 panels finished"
