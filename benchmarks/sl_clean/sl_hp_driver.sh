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

  biases_mnist|biases_fashion|biases_cifar10)
    # B1 — H1 panel.  Single factorial phase with 9 cells × N seeds.
    # The standard adoption analyzer does NOT run on this phase
    # (knob 'cell' is enumerative, not adoptable); the panel verdict
    # is produced by analyze_biases_panel.py instead.
    case "${EXPERIMENT}" in
      biases_mnist)   BASE="${RESULTS_DIR}/base_config_biases_mnist.json"   ; PHASE_NAME="biases_mnist_panel"    ;;
      biases_fashion) BASE="${RESULTS_DIR}/base_config_biases_fashion.json" ; PHASE_NAME="biases_fashion_panel"  ;;
      biases_cifar10) BASE="${RESULTS_DIR}/base_config_biases_cifar10.json" ; PHASE_NAME="biases_cifar10_panel"  ;;
    esac

    SEEDS="${SEEDS:-0-9}"
    CELLS="data_driven+hypersphere,data_driven+gaussian,data_driven+sparse,random_normal+hypersphere,random_normal+gaussian,random_normal+sparse,zero+hypersphere,zero+gaussian,zero+sparse"

    out="${RESULTS_DIR}/${PHASE_NAME}.json"
    log="${LOGS_DIR}/${PHASE_NAME}.log"
    echo "================================================================"
    echo "  PHASE ${PHASE_NAME}  exp=${EXPERIMENT}  cells=${CELLS}  seeds=${SEEDS}"
    echo "================================================================"
    ${PYTHON} -u -m benchmarks.sl_clean.sl_phase_runner \
        --experiment "${EXPERIMENT}" \
        --phase      "${PHASE_NAME}" \
        --knob       cell \
        --values     "${CELLS}" \
        --seeds      "${SEEDS}" \
        --base-config "${BASE}" \
        --out         "${out}" \
        --log         "${log}"

    ${PYTHON} -u -m benchmarks.sl_clean.analyze_biases_panel \
        --results "${out}" \
        --out     "${RESULTS_DIR}/${PHASE_NAME}_summary.json"
    ;;

  # New blocks land here as B-phase / C-phase experiments are written.

  streaming_smnist_row|streaming_smnist_permuted)
    # B2 — H4 streaming-NEF panel.  Knob is n_neurons; 10 seeds.
    case "${EXPERIMENT}" in
      streaming_smnist_row)      BASE="${RESULTS_DIR}/base_config_streaming_smnist_row.json"      ;;
      streaming_smnist_permuted) BASE="${RESULTS_DIR}/base_config_streaming_smnist_permuted.json" ;;
    esac

    SEEDS="${SEEDS:-0-9}"
    # Row defaults: 2k/4k/8k.  Permuted defaults: 2k/4k (8k is too slow on CPU).
    if [ "${EXPERIMENT}" = "streaming_smnist_row" ]; then
      VALUES="${VALUES:-2000,4000,8000}"
    else
      VALUES="${VALUES:-2000,4000}"
    fi

    run_phase "${EXPERIMENT}_n_neurons" n_neurons "${VALUES}" "${SEEDS}" "${BASE}" "${EXPERIMENT}"
    ;;

  streaming_solve_modes)
    # B2 — H3 equivalence: batch_acc vs accumulate vs woodbury at fixed n.
    BASE="${RESULTS_DIR}/base_config_streaming_solve_modes.json"
    SEEDS="${SEEDS:-0-4}"
    VALUES="${VALUES:-batch_acc,accumulate,woodbury}"
    run_phase streaming_solve_modes solve_mode "${VALUES}" "${SEEDS}" "${BASE}" "${EXPERIMENT}"
    ;;

  *)
    echo "unknown experiment block: ${EXPERIMENT}" >&2
    echo "available: smoke biases_mnist biases_fashion biases_cifar10 streaming_smnist_row streaming_smnist_permuted streaming_solve_modes" >&2
    exit 2
    ;;
esac
