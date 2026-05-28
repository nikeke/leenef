#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
RESULT_DIR="$ROOT/results/rl_paper/dense"
LOG_DIR="$RESULT_DIR/logs"
SENTINEL="$RESULT_DIR/dense_chain_done"
mkdir -p "$LOG_DIR"
rm -f "$SENTINEL"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-3}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$OMP_NUM_THREADS}"

run_phase() {
    local phase="$1"
    local seeds="$2"
    local parallel="$3"
    local episodes="${4:-}"
    local log="$LOG_DIR/${phase}.log"

    echo
    echo "============================================================"
    echo "[$(date -Is)] phase=${phase} seeds=${seeds} parallel=${parallel} episodes=${episodes:-default}"
    echo "============================================================"

    local args=(
        "$PYTHON" -u benchmarks/run_rl_paper_dense.py
        --phase "$phase"
        --seeds $seeds
        --parallel "$parallel"
        --output "$RESULT_DIR/${phase}.json"
    )
    if [[ -n "$episodes" ]]; then
        args+=(--episodes "$episodes")
    fi
    "${args[@]}" 2>&1 | tee "$log"
}

echo "[$(date -Is)] Paper III dense RL chain starting"
echo "root=$ROOT"
echo "python=$PYTHON"
echo "commit=$(git rev-parse HEAD)"
echo "code_status=$(git status --short | wc -l) changed-paths"
echo "threads=OMP:$OMP_NUM_THREADS MKL:$MKL_NUM_THREADS OPENBLAS:$OPENBLAS_NUM_THREADS NUMEXPR:$NUMEXPR_NUM_THREADS TORCH:$TORCH_NUM_THREADS"

run_phase cartpole_sanity "${CARTPOLE_SEEDS:-0 1 2 3 4 5 6 7 8 9}" "${CARTPOLE_PARALLEL:-4}" "${CARTPOLE_EPISODES:-}"
run_phase bootstrap_lunar "${BOOTSTRAP_SEEDS:-0 1 2 3 4}" "${BOOTSTRAP_PARALLEL:-4}" "${BOOTSTRAP_EPISODES:-}"
run_phase lunar_dqn "${DQN_SEEDS:-0 1 2 3 4 5 6 7 8 9}" "${DQN_PARALLEL:-4}" "${DQN_EPISODES:-}"
run_phase lunar_final "${LUNAR_FINAL_SEEDS:-0 1 2 3 4 5 6 7 8 9}" "${LUNAR_FINAL_PARALLEL:-4}" "${LUNAR_FINAL_EPISODES:-}"
run_phase ensemble_lunar "${ENSEMBLE_SEEDS:-0 1 2}" "${ENSEMBLE_PARALLEL:-1}" "${ENSEMBLE_EPISODES:-}"

date -Is > "$SENTINEL"
echo "[$(date -Is)] Paper III dense RL chain complete; sentinel=$SENTINEL"
