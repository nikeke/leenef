#!/usr/bin/env bash
# Run B2 final panels.  Strategy:
#   - Phase A (sequential):  streaming_smnist_row alone (fast, ~15 min;
#       it has 10 seeds × 3 n_neurons = 30 cells; let it have the box).
#   - Phase B (parallel):    streaming_smnist_permuted + streaming_solve_modes
#       run concurrently.  Each phase saturates ~5 of 12 cores via BLAS,
#       so two phases together push us toward ~10 cores instead of leaving
#       7 cores idle for ~3 h.
#
# A sentinel file results/sl_clean/b2_panels_done is touched at completion.
# The sister script `wait_for_b2_panels.sh` polls that file (or detects log
# stalls) so the agent can sleep on a foreground watcher and receive a
# completion notification.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-$PWD/venv/bin/python}"
LOGDIR="results/sl_clean/logs"
mkdir -p "$LOGDIR"

DRIVER="benchmarks/sl_clean/sl_hp_driver.sh"

ts() { date -u +'[%Y-%m-%dT%H:%M:%SZ]'; }
echo "$(ts) panel-driver start"

# Phase A: row sMNIST.
echo "$(ts) launching streaming_smnist_row (sequential)"
PYTHON="$PYTHON" SEEDS=0-9 bash "$DRIVER" streaming_smnist_row \
    > "$LOGDIR/b2_row.log" 2>&1
echo "$(ts) streaming_smnist_row done"

# Phase B: permuted sMNIST + solve-mode equivalence in parallel.
echo "$(ts) launching streaming_smnist_permuted + streaming_solve_modes (parallel)"
PYTHON="$PYTHON" SEEDS=0-4 bash "$DRIVER" streaming_smnist_permuted \
    > "$LOGDIR/b2_permuted.log" 2>&1 &
PID_PERMUTED=$!
PYTHON="$PYTHON" SEEDS=0-4 bash "$DRIVER" streaming_solve_modes \
    > "$LOGDIR/b2_modes.log" 2>&1 &
PID_MODES=$!
echo "$(ts) permuted pid=$PID_PERMUTED modes pid=$PID_MODES"
wait $PID_PERMUTED
echo "$(ts) streaming_smnist_permuted done (rc=$?)"
wait $PID_MODES
echo "$(ts) streaming_solve_modes done (rc=$?)"

touch results/sl_clean/b2_panels_done
echo "$(ts) panel-driver complete"
