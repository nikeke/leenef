#!/usr/bin/env bash
# Run B7 follow-up panels.  Two 9-cell × 10-seed biases panels in parallel:
#   - biases_cifar10_centered  (per-channel-centred CIFAR-10 — DC-fold falsifier #1)
#   - biases_mnist_decentered  (DC-shifted MNIST           — DC-fold falsifier #2)
#
# Each panel is independent; running them concurrently saturates the box
# better than serial (each chunked-Cholesky path runs ~6-thread BLAS).
#
# A sentinel file results/sl_clean/b7_panels_done is touched at completion.
# The sister script `wait_for_b7_panels.sh` polls that file and detects
# log stalls.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-$PWD/venv/bin/python}"
LOGDIR="results/sl_clean/logs"
mkdir -p "$LOGDIR"

DRIVER="benchmarks/sl_clean/sl_hp_driver.sh"

ts() { date -u +'[%Y-%m-%dT%H:%M:%SZ]'; }
echo "$(ts) panel-driver start (B7)"

# Limit each child to ~6 BLAS threads so two children together fit in 12 cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"

echo "$(ts) launching biases_cifar10_centered + biases_mnist_decentered (parallel)"
PYTHON="$PYTHON" SEEDS=0-9 bash "$DRIVER" biases_cifar10_centered \
    > "$LOGDIR/b7_cifar10_centered.log" 2>&1 &
PID_CIFAR=$!
PYTHON="$PYTHON" SEEDS=0-9 bash "$DRIVER" biases_mnist_decentered \
    > "$LOGDIR/b7_mnist_decentered.log" 2>&1 &
PID_MNIST=$!
echo "$(ts) cifar10_centered pid=$PID_CIFAR  mnist_decentered pid=$PID_MNIST"

wait $PID_CIFAR
RC_CIFAR=$?
echo "$(ts) biases_cifar10_centered done (rc=$RC_CIFAR)"
wait $PID_MNIST
RC_MNIST=$?
echo "$(ts) biases_mnist_decentered done (rc=$RC_MNIST)"

if [[ $RC_CIFAR -eq 0 && $RC_MNIST -eq 0 ]]; then
    touch results/sl_clean/b7_panels_done
    echo "$(ts) panel-driver complete (B7) — sentinel touched"
else
    echo "$(ts) panel-driver FAILED (B7): cifar=$RC_CIFAR mnist=$RC_MNIST" >&2
    exit 1
fi
