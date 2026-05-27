#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
RESULT_DIR="$ROOT/results/rl_sparse/geometry"
LOG_DIR="$RESULT_DIR/logs"
SENTINEL="$RESULT_DIR/geometry_chain_done"
WIDTH="${WIDTH:-2000}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPISODES="${EPISODES:-300}"
EVAL_EVERY="${EVAL_EVERY:-25}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
CHAIN_SEEDS="${CHAIN_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CHAIN_EPISODES="${CHAIN_EPISODES:-200}"

mkdir -p "$LOG_DIR"
rm -f "$SENTINEL"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-6}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$OMP_NUM_THREADS}"

run_sparse_chain() {
    local log="$LOG_DIR/sparse_chain.log"
    echo
    echo "============================================================"
    echo "[$(date -Is)] SparseChain readout geometry probe"
    echo "============================================================"
    "$PYTHON" -u benchmarks/run_sparse_chain.py \
        --n-states 5 10 20 \
        --seeds $CHAIN_SEEDS \
        --episodes "$CHAIN_EPISODES" \
        --output "$RESULT_DIR/sparse_chain_geometry.json" \
        2>&1 | tee "$log"
}

run_mode() {
    local mode="$1"
    local log="$LOG_DIR/mountaincar_${mode}_n${WIDTH}.log"
    local output="$RESULT_DIR/mountaincar_${mode}_n${WIDTH}.json"
    echo
    echo "============================================================"
    echo "[$(date -Is)] MountainCar readout_mode=${mode} width=${WIDTH} seeds=${SEEDS}"
    echo "============================================================"
    "$PYTHON" -u benchmarks/run_rl_sparse.py \
        --env MountainCar-v0 \
        --episodes "$EPISODES" \
        --eval-every "$EVAL_EVERY" \
        --eval-episodes "$EVAL_EPISODES" \
        --seeds $SEEDS \
        --variants nef_sf_hindsight \
        --warmup-episodes 10 \
        --goal-horizon 25 \
        --goal-mode random_direction \
        --goal-action-mode commit \
        --eval-goal 0.5,0.0 \
        --goal-readout-mode "$mode" \
        --config-override "n_neurons=$WIDTH" \
        --config-override "forget_factor=0.999" \
        --device cpu \
        --skip-agent-diagnostics \
        --output "$output" \
        2>&1 | tee "$log"
}

echo "[$(date -Is)] sparse RL readout geometry chain starting"
echo "root=$ROOT"
echo "python=$PYTHON"
echo "commit=$(git rev-parse HEAD)"
echo "code_status=$(git status --short | wc -l) changed-paths"
echo "threads=OMP:$OMP_NUM_THREADS MKL:$MKL_NUM_THREADS OPENBLAS:$OPENBLAS_NUM_THREADS NUMEXPR:$NUMEXPR_NUM_THREADS TORCH:$TORCH_NUM_THREADS"
echo "result_dir=$RESULT_DIR"

run_sparse_chain

fail=0
run_mode dot &
pid_dot=$!
run_mode cosine &
pid_cosine=$!

if ! wait "$pid_dot"; then
    fail=1
fi
if ! wait "$pid_cosine"; then
    fail=1
fi
if (( fail != 0 )); then
    echo "[$(date -Is)] ERROR: dot/cosine stage failed" >&2
    exit "$fail"
fi

run_mode future_cosine

date -Is > "$SENTINEL"
echo "[$(date -Is)] sparse RL readout geometry chain complete; sentinel=$SENTINEL"
