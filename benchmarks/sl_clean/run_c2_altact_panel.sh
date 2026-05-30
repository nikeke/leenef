#!/usr/bin/env bash
# Alt-activation real-task panel (companion to docs/activation_kernel_theory.md).
#
# Runs the full set of "small" nonlinearities under the SSUP-I section 5.4
# protocol (n=5000, hypersphere encoders, data-driven biases, trace-scaled
# alpha=1e-3) on MNIST / Fashion-MNIST / CIFAR-10, 10 seeds.
#
# IMPORTANT: this writes to SEPARATE result files
# (results/sl_clean/c2_altact_{dataset}_panel.json) via a custom --phase, so
# the published c2_activation_*_panel.json panels are left untouched.  All new
# rows carry the current code_identity.
#
# Uses the leenef venv for deps while forcing imports from this worktree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/venv/bin/python}"
SEEDS_ARG="${SEEDS:-0-9}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# References (re-run under the current code_identity) + new activations at the
# main gain, plus a small/wide gain sweep for the RBF-family pair cos/gauss.
CELLS="${VALUES:-abs+g0.5_2.0,relu+g0.5_2.0,bent_line+g0.5_2.0,leaky_relu+g0.5_2.0,softplus+g0.5_2.0,square+g0.5_2.0,double_fold+g0.5_2.0,sqrt_abs+g0.5_2.0,cos+g0.5_2.0,gauss+g0.5_2.0,cos+g0.05_0.2,gauss+g0.05_0.2,cos+g0.1_5.0,gauss+g0.1_5.0}"

LOG_DIR="${REPO_ROOT}/results/sl_clean/logs"
DONE="${REPO_ROOT}/results/sl_clean/c2_altact_panel_done"
mkdir -p "${LOG_DIR}"
rm -f "${DONE}"

declare -a EXPS=(c2_activation_mnist c2_activation_fashion c2_activation_cifar10)
declare -a PIDS=()

for exp in "${EXPS[@]}"; do
  dataset="${exp#c2_activation_}"
  phase="c2_altact_${dataset}"
  base="${REPO_ROOT}/results/sl_clean/base_config_${exp}.json"
  out="${REPO_ROOT}/results/sl_clean/c2_altact_${dataset}_panel.json"
  log="${LOG_DIR}/c2_altact_${dataset}.log"
  echo "[run_c2_altact] launching ${phase} (seeds=${SEEDS_ARG}) -> ${out}"
  (
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      ${PYTHON} -u -m benchmarks.sl_clean.sl_phase_runner \
        --experiment "${exp}" \
        --phase      "${phase}" \
        --knob       cell \
        --values     "${CELLS}" \
        --seeds      "${SEEDS_ARG}" \
        --base-config "${base}" \
        --out         "${out}" \
        --log         "${log}"
  ) > "${LOG_DIR}/c2_altact_${dataset}.driver.log" 2>&1 &
  PIDS+=($!)
done

echo "[run_c2_altact] launched PIDs: ${PIDS[*]}"

ALL_OK=1
for pid in "${PIDS[@]}"; do
  if wait "${pid}"; then
    echo "[run_c2_altact] PID ${pid} OK"
  else
    echo "[run_c2_altact] PID ${pid} FAILED"
    ALL_OK=0
  fi
done

if [ "${ALL_OK}" -eq 1 ]; then
  date -Is > "${DONE}"
  echo "[run_c2_altact] all panels complete"
  exit 0
fi

echo "[run_c2_altact] one or more panels FAILED"
exit 1
