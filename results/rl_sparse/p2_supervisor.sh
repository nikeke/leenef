#!/usr/bin/env bash
set -uo pipefail

WAIT_PID="${WAIT_PID:-}"
RESULT_DIR="${RESULT_DIR:-results/rl_sparse}"
WORKERS_PER_SWEEP="${WORKERS_PER_SWEEP:-2}"
BLAS_THREADS="${BLAS_THREADS:-2}"

mkdir -p "${RESULT_DIR}"

SUPERVISOR_LOG="${RESULT_DIR}/sparse_p2_supervisor.log"
ACROBOT_LOG="${RESULT_DIR}/sparse_p2_acrobot_sf.log"
MOUNTAINCAR_LOG="${RESULT_DIR}/sparse_p2_mountaincar_explore.log"

export OMP_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"
export TORCH_NUM_THREADS="${BLAS_THREADS}"

{
  echo "======================================================================"
  echo "Sparse P2 supervisor start: $(date -Is)"
  echo "wait_pid=${WAIT_PID:-none}"
  echo "workers_per_sweep=${WORKERS_PER_SWEEP}"
  echo "blas_threads=${BLAS_THREADS}"
  echo "approx_cpu_threads=$((2 * WORKERS_PER_SWEEP * BLAS_THREADS))"
  echo "======================================================================"
} >>"${SUPERVISOR_LOG}"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    echo "Waiting for P1 pid ${WAIT_PID}: $(date -Is)" >>"${SUPERVISOR_LOG}"
    sleep 60
  done
fi

source ../leenef/venv/bin/activate

python -u benchmarks/run_sparse_reward_sweep.py \
  --phase sparse_p2_acrobot_sf \
  --envs Acrobot-v1 \
  --configs "${RESULT_DIR}/configs/p2_acrobot_sf.json" \
  --seeds 0 1 2 3 4 \
  --episodes 500 \
  --eval-every 50 \
  --eval-episodes 10 \
  --warmup-episodes 10 \
  --parallel "${WORKERS_PER_SWEEP}" \
  --base-config results/rl_clean/base_config.json \
  --output "${RESULT_DIR}/sparse_p2_acrobot_sf.json" \
  >>"${ACROBOT_LOG}" 2>&1 &
acrobot_pid=$!

python -u benchmarks/run_sparse_reward_sweep.py \
  --phase sparse_p2_mountaincar_explore \
  --envs MountainCar-v0 \
  --configs "${RESULT_DIR}/configs/p2_mountaincar_explore.json" \
  --seeds 0 1 2 3 4 5 \
  --episodes 1000 \
  --eval-every 100 \
  --eval-episodes 10 \
  --warmup-episodes 20 \
  --parallel "${WORKERS_PER_SWEEP}" \
  --base-config results/rl_clean/base_config.json \
  --output "${RESULT_DIR}/sparse_p2_mountaincar_explore.json" \
  >>"${MOUNTAINCAR_LOG}" 2>&1 &
mountaincar_pid=$!

echo "Started Acrobot P2 pid=${acrobot_pid}: $(date -Is)" >>"${SUPERVISOR_LOG}"
echo "Started MountainCar P2 pid=${mountaincar_pid}: $(date -Is)" >>"${SUPERVISOR_LOG}"

acrobot_status=0
mountaincar_status=0
wait "${acrobot_pid}" || acrobot_status=$?
wait "${mountaincar_pid}" || mountaincar_status=$?

{
  echo "Acrobot P2 exit=${acrobot_status}: $(date -Is)"
  echo "MountainCar P2 exit=${mountaincar_status}: $(date -Is)"
  echo "Sparse P2 supervisor end: $(date -Is)"
  echo "======================================================================"
} >>"${SUPERVISOR_LOG}"

if [[ "${acrobot_status}" -ne 0 || "${mountaincar_status}" -ne 0 ]]; then
  exit 1
fi
