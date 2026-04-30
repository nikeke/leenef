#!/usr/bin/env bash
# Driver template for sl_clean phases.
#
# This is the SL counterpart to attic/benchmarks/rl_hp_driver.sh.
# Each phase is a single-knob sweep that:
#   1. runs the phase to completion (resumable),
#   2. analyzes the result against the live base_config,
#   3. logs everything under results/sl_clean/logs/.
#
# Phases are intentionally laid out as plain shell so an unattended run
# can be tailed via `tail -f results/sl_clean/logs/<phase>.log`.
#
# Usage:
#   bash benchmarks/sl_clean/sl_hp_driver.sh smoke
#
# The argument selects which experiment block to run.  Add new blocks
# below as headline-claim experiments come online (B1, B2, ...).

set -euo pipefail

EXPERIMENT="${1:-smoke}"
RESULTS_DIR="results/sl_clean"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOGS_DIR}"

PYTHON="${PYTHON:-python}"

run_phase() {
  local phase="$1"; shift
  local knob="$1"; shift
  local values="$1"; shift
  local seeds="$1"; shift
  local base_config="$1"; shift
  local exp="${1:-${EXPERIMENT}}"

  local out="${RESULTS_DIR}/${phase}.json"
  local log="${LOGS_DIR}/${phase}.log"

  echo "================================================================"
  echo "  PHASE ${phase}  exp=${exp}  knob=${knob}  values=${values}"
  echo "  base_config=${base_config}  out=${out}"
  echo "================================================================"

  ${PYTHON} -u -m benchmarks.sl_clean.sl_phase_runner \
      --experiment "${exp}" \
      --phase      "${phase}" \
      --knob       "${knob}" \
      --values     "${values}" \
      --seeds      "${seeds}" \
      --base-config "${base_config}" \
      --out         "${out}" \
      --log         "${log}"

  ${PYTHON} -u -m benchmarks.sl_clean.analyze_sl_sweep \
      --experiment  "${exp}" \
      --knob        "${knob}" \
      --sweep-json  "${out}" \
      --base-config "${base_config}"
}

case "${EXPERIMENT}" in
  smoke)
    BASE="${RESULTS_DIR}/base_config_smoke.json"
    run_phase smoke_n_neurons       n_neurons      "100,200,400"     "0,1,2,3,4" "${BASE}" smoke
    run_phase smoke_data_driven_bias data_driven_biases "true,false" "0,1,2,3,4" "${BASE}" smoke
    ;;

  # New blocks land here as B-phase / C-phase experiments are written.

  *)
    echo "unknown experiment block: ${EXPERIMENT}" >&2
    echo "available: smoke" >&2
    exit 2
    ;;
esac
