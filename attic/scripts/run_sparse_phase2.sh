#!/usr/bin/env bash
# Phase-2 sparse-reward driver, parallel edition.
#
# Launches the four remaining experiments concurrently as separate
# python processes, each pinned to a single BLAS / OMP thread so we
# stay around 4 cores total (≈33% of 12, comfortably under the 50%
# target the user set for our own work).
#
# Logs land in results/rl_sparse/logs/*.log, JSON in results/rl_sparse/.
# PIDs are recorded in results/rl_sparse/logs/phase2.pids for monitoring.
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH="src:."
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PY=/home/amnesia/Persistent/src/leenef/venv/bin/python
RES=results/rl_sparse
LOG=$RES/logs
mkdir -p "$LOG"
PIDFILE=$LOG/phase2.pids
: > "$PIDFILE"

start() {
    local name=$1; shift
    echo "[$(date +%H:%M:%S)] launching $name" >&2
    "$@" >"$LOG/${name}.log" 2>&1 &
    local pid=$!
    echo "$pid $name" >> "$PIDFILE"
    echo "  pid=$pid  log=$LOG/${name}.log" >&2
}

# 1) Acrobot stability tune: forget_factor 0.999 -> 0.9999.
start acrobot_hindsight_ff9999 \
    $PY -u benchmarks/run_rl_sparse.py \
        --env Acrobot-v1 \
        --episodes 300 --eval-every 25 --eval-episodes 5 \
        --seeds 0 1 2 \
        --variants nef_sf_hindsight \
        --warmup-episodes 20 \
        --goal-buffer-size 1024 --goal-horizon 50 \
        --goal-action-mode commit \
        --eval-goal=-1,0,1,0,0,0 \
        --config-override n_neurons=2000 \
        --config-override forget_factor=0.9999 \
        --output $RES/acrobot_hindsight_evalgoal_ff9999.json

# 2) LunarLander dense regression — nef_fqi @ n=2000 (parity with hindsight).
start lunarlander_fqi_n2000 \
    $PY -u benchmarks/run_rl_sparse.py \
        --env LunarLander-v3 \
        --episodes 300 --eval-every 25 --eval-episodes 5 \
        --seeds 0 1 2 \
        --variants nef_fqi \
        --warmup-episodes 20 \
        --config-override n_neurons=2000 \
        --output $RES/lunarlander_fqi_n2000.json

# 3) LunarLander dense regression — nef_sf_hindsight @ n=2000.
start lunarlander_hindsight_n2000 \
    $PY -u benchmarks/run_rl_sparse.py \
        --env LunarLander-v3 \
        --episodes 300 --eval-every 25 --eval-episodes 5 \
        --seeds 0 1 2 \
        --variants nef_sf_hindsight \
        --warmup-episodes 20 \
        --goal-buffer-size 1024 --goal-horizon 50 \
        --config-override n_neurons=2000 \
        --output $RES/lunarlander_hindsight_n2000.json

# 4) MiniGrid DoorKey-5x5: sparse reward, last-state hindsight, no eval_goal.
start minigrid_doorkey5_hindsight \
    $PY -u benchmarks/run_rl_sparse.py \
        --env MiniGrid-DoorKey-5x5-v0 \
        --episodes 400 --eval-every 25 --eval-episodes 5 \
        --seeds 0 1 2 \
        --variants nef_sf_hindsight \
        --warmup-episodes 20 \
        --goal-buffer-size 1024 --goal-horizon 50 \
        --goal-action-mode commit \
        --config-override n_neurons=2000 \
        --output $RES/minigrid_doorkey5_hindsight.json

echo
echo "Launched $(wc -l < "$PIDFILE") jobs.  PIDs:"
cat "$PIDFILE"
