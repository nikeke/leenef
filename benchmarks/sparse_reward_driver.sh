#!/usr/bin/env bash
set -euo pipefail

PHASE="${PHASE:-sparse_p1_mechanisms}"
RESULT_DIR="${RESULT_DIR:-results/rl_sparse}"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/${PHASE}.log}"
OUTPUT_FILE="${OUTPUT_FILE:-${RESULT_DIR}/${PHASE}.json}"
WORKERS="${WORKERS:-3}"
BLAS_THREADS="${BLAS_THREADS:-2}"

mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"
export TORCH_NUM_THREADS="${BLAS_THREADS}"

{
  echo "======================================================================"
  echo "Sparse reward driver start: $(date -Is)"
  echo "phase=${PHASE}"
  echo "output=${OUTPUT_FILE}"
  echo "workers=${WORKERS}"
  echo "blas_threads=${BLAS_THREADS}"
  echo "approx_cpu_threads=$((WORKERS * BLAS_THREADS))"
  echo "======================================================================"
} >>"${LOG_FILE}"

source ../leenef/venv/bin/activate

python -u benchmarks/run_sparse_reward_sweep.py \
  --phase "${PHASE}" \
  --envs MountainCar-v0 Acrobot-v1 \
  --seeds 0 1 2 \
  --episodes 300 \
  --eval-every 50 \
  --eval-episodes 10 \
  --warmup-episodes 5 \
  --parallel "${WORKERS}" \
  --base-config results/rl_clean/base_config.json \
  --output "${OUTPUT_FILE}" \
  >>"${LOG_FILE}" 2>&1

{
  echo "======================================================================"
  echo "Sparse reward driver end: $(date -Is)"
  echo "======================================================================"
} >>"${LOG_FILE}"
