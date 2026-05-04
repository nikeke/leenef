#!/usr/bin/env bash
# Driver for cl_clean phases.  Mirrors benchmarks/sl_clean/sl_hp_driver.sh.
#
# Each phase is a single-knob sweep that:
#   1. runs to completion (resumable),
#   2. analyzes the result (where adoption applies),
#   3. logs everything under results/cl_clean/logs/.
#
# Usage:
#   bash benchmarks/cl_clean/cl_hp_driver.sh smoke
#   PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
#       bash benchmarks/cl_clean/cl_hp_driver.sh nef_continual_split_mnist
#
# The argument selects which experiment block to run.
#
# **Driver-edit rule (Paper I lesson):** never edit this script while a
# phase that uses it is in flight.  Add new experiments via Python
# registration and a fresh top-level case clause; if a paragraph below
# is currently being interpreted by bash, edits to it can corrupt the
# in-flight run with an ``unexpected EOF`` parse error.

set -euo pipefail

EXPERIMENT="${1:-smoke}"
RESULTS_DIR="results/cl_clean"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOGS_DIR}"

PYTHON="${PYTHON:-python}"
SEEDS_DEFAULT="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"

# Translate ``0-9`` → ``0,1,2,3,4,5,6,7,8,9`` for convenience.
case "${SEEDS_DEFAULT}" in
  *-*)
    LO="${SEEDS_DEFAULT%%-*}"
    HI="${SEEDS_DEFAULT##*-}"
    SEEDS_DEFAULT="$(seq -s, "${LO}" "${HI}")"
    ;;
esac

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

  ${PYTHON} -u -m benchmarks.cl_clean.cl_phase_runner \
      --experiment "${exp}" \
      --phase      "${phase}" \
      --knob       "${knob}" \
      --values     "${values}" \
      --seeds      "${seeds}" \
      --base-config "${base_config}" \
      --out         "${out}" \
      --log         "${log}"
}

run_phase_and_analyze() {
  run_phase "$@"
  local phase="$1"
  local knob="$2"
  local exp="${6:-${EXPERIMENT}}"
  local base_config="$5"
  local out="${RESULTS_DIR}/${phase}.json"

  ${PYTHON} -u -m benchmarks.cl_clean.analyze_cl_sweep \
      --experiment  "${exp}" \
      --knob        "${knob}" \
      --sweep-json  "${out}" \
      --base-config "${base_config}"
}

case "${EXPERIMENT}" in
  smoke)
    BASE="${RESULTS_DIR}/base_config_smoke.json"
    run_phase_and_analyze smoke_n_neurons n_neurons "100,200,400" "${SEEDS_DEFAULT}" "${BASE}" smoke
    ;;

  nef_continual_split_mnist|nef_continual_permuted_mnist|nef_continual_split_cifar10|nef_continual_split_cifar100)
    # Phase B1/B2 — H1 panel (joint-equivalence + cross-method baseline).
    # Single factorial phase with up to 7 cells × N seeds (3 NEF + 4 MLP).
    # The standard adoption analyzer does not run on this phase
    # (knob 'cell' is enumerative); the panel verdict is produced by
    # analyze_cl_panel.py.
    BASE="${RESULTS_DIR}/base_config_${EXPERIMENT}.json"
    PHASE_NAME="${EXPERIMENT}_panel"
    CELLS="${CELLS:-nef_accumulate,nef_joint,nef_reset,mlp_finetune,mlp_joint,mlp_ewc,mlp_er}"
    run_phase "${PHASE_NAME}" cell "${CELLS}" \
              "${SEEDS_DEFAULT}" "${BASE}" "${EXPERIMENT}"
    touch "${RESULTS_DIR}/${PHASE_NAME}_done"
    ;;

  *)
    echo "unknown experiment: ${EXPERIMENT}" >&2
    exit 1
    ;;
esac
