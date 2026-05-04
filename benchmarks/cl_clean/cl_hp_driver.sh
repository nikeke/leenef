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

# DETACH=1 re-launches this driver fully detached via setsid (matches the
# sl_clean Paper-I pattern).  Stdout/stderr go to a per-experiment log so
# the launching shell can exit without taking the run down.  Watchdogs
# in the sandbox cannot survive CLI restarts; sweeps must be launched
# from the host or at least with `setsid ... < /dev/null & disown`.
if [[ "${DETACH:-0}" == "1" && -z "${CL_DETACHED:-}" ]]; then
  DETACH_LOG="${LOGS_DIR}/${EXPERIMENT}_detached.log"
  echo "detaching driver for ${EXPERIMENT}; log=${DETACH_LOG}"
  CL_DETACHED=1 setsid bash "$0" "$@" \
      </dev/null >>"${DETACH_LOG}" 2>&1 &
  disown || true
  sleep 1
  pgrep -af "cl_hp_driver.sh ${EXPERIMENT}" | grep -v "DETACH=1" || true
  exit 0
fi

normalize_seeds() {
  local raw="$1"
  case "${raw}" in
    *-*)
      local lo="${raw%%-*}"
      local hi="${raw##*-}"
      seq -s, "${lo}" "${hi}"
      ;;
    *)
      echo "${raw}"
      ;;
  esac
}

# Translate ``0-9`` → ``0,1,2,3,4,5,6,7,8,9`` for convenience.
SEEDS_DEFAULT="$(normalize_seeds "${SEEDS_DEFAULT}")"

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
    if run_phase "${PHASE_NAME}" cell "${CELLS}" \
              "${SEEDS_DEFAULT}" "${BASE}" "${EXPERIMENT}"; then
      touch "${RESULTS_DIR}/${PHASE_NAME}_done"
    else
      echo "phase ${PHASE_NAME} had cell errors; sentinel NOT touched" >&2
      exit 1
    fi
    ;;

  b3_capacity)
    # Phase B3 — capacity matrix.  Defaults follow the plan:
    # Permuted-MNIST: 5 seeds over a 5×5 n_neurons × n_tasks grid.
    # Permuted-CIFAR-10: 3 seeds over a smaller confirmation grid.
    B3_MNIST_SEEDS="$(normalize_seeds "${B3_MNIST_SEEDS:-0-4}")"
    B3_CIFAR_SEEDS="$(normalize_seeds "${B3_CIFAR_SEEDS:-0-2}")"
    B3_MNIST_N="${B3_MNIST_N:-500,1000,2000,5000,10000}"
    B3_CIFAR_N="${B3_CIFAR_N:-1000,2000,5000,10000}"
    BASE="${RESULTS_DIR}/base_config_b3_capacity.json"
    for T in ${B3_MNIST_TASKS:-5 10 20 50 100}; do
      run_phase "b3_capacity_permuted_mnist_t${T}" n_neurons "${B3_MNIST_N}" \
                "${B3_MNIST_SEEDS}" "${BASE}" "nef_capacity_permuted_mnist_t${T}"
    done
    for T in ${B3_CIFAR_TASKS:-5 10 20}; do
      run_phase "b3_capacity_permuted_cifar10_t${T}" n_neurons "${B3_CIFAR_N}" \
                "${B3_CIFAR_SEEDS}" "${BASE}" "nef_capacity_permuted_cifar10_t${T}"
    done

    # Joint-equivalence spot checks at the four MNIST grid extremes.
    for T in 5 100; do
      for N in 500 10000; do
        SPOT_BASE="${RESULTS_DIR}/base_config_b3_capacity_spot_n${N}.json"
        run_phase "b3_capacity_spot_permuted_mnist_t${T}_n${N}" cell \
                  "nef_accumulate,nef_joint" "${B3_CIFAR_SEEDS}" \
                  "${SPOT_BASE}" "nef_capacity_permuted_mnist_t${T}"
      done
    done
    touch "${RESULTS_DIR}/b3_capacity_done"
    ;;

  b4_centers)
    # Phase B4 — center-adaptation falsifier.
    B4_SEEDS="$(normalize_seeds "${B4_SEEDS:-0-4}")"
    BASE="${RESULTS_DIR}/base_config_b4_centers.json"
    run_phase b4_centers_split_mnist centers_mode "none,first_task,all_tasks,growing" \
              "${B4_SEEDS}" "${BASE}" nef_continual_split_mnist
    run_phase b4_centers_split_cifar10 centers_mode "none,first_task,all_tasks,growing" \
              "${B4_SEEDS}" "${BASE}" nef_continual_split_cifar10
    touch "${RESULTS_DIR}/b4_centers_done"
    ;;

  b5_solvers)
    # Phase B5 — solver mode × fixed-alpha regime.
    B5_SEEDS="$(normalize_seeds "${B5_SEEDS:-0-4}")"
    BASE="${RESULTS_DIR}/base_config_b5_solvers.json"
    SOLVER_ALPHA_CELLS="${SOLVER_ALPHA_CELLS:-accumulate_a0.001,accumulate_a0.01,accumulate_a0.1,accumulate_a1,accumulate_a10,accumulate_a100,accumulate_a1000,woodbury_batch_a0.001,woodbury_batch_a0.01,woodbury_batch_a0.1,woodbury_batch_a1,woodbury_batch_a10,woodbury_batch_a100,woodbury_batch_a1000,woodbury_online_a0.001,woodbury_online_a0.01,woodbury_online_a0.1,woodbury_online_a1,woodbury_online_a10,woodbury_online_a100,woodbury_online_a1000}"
    for EXP in nef_continual_split_mnist nef_continual_permuted_mnist nef_continual_split_cifar10; do
      run_phase "b5_solvers_${EXP#nef_continual_}" solver_alpha_cell "${SOLVER_ALPHA_CELLS}" \
                "${B5_SEEDS}" "${BASE}" "${EXP}"
    done
    touch "${RESULTS_DIR}/b5_solvers_done"
    ;;

  *)
    echo "unknown experiment: ${EXPERIMENT}" >&2
    exit 1
    ;;
esac
