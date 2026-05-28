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

  biases_mnist|biases_fashion|biases_cifar10|biases_cifar10_centered|biases_mnist_decentered)
    # B1 — H1 panel.  Single factorial phase with 9 cells × N seeds.
    # The standard adoption analyzer does NOT run on this phase
    # (knob 'cell' is enumerative, not adoptable); the panel verdict
    # is produced by analyze_biases_panel.py instead.
    case "${EXPERIMENT}" in
      biases_mnist)             BASE="${RESULTS_DIR}/base_config_biases_mnist.json"             ; PHASE_NAME="biases_mnist_panel"             ;;
      biases_fashion)           BASE="${RESULTS_DIR}/base_config_biases_fashion.json"           ; PHASE_NAME="biases_fashion_panel"           ;;
      biases_cifar10)           BASE="${RESULTS_DIR}/base_config_biases_cifar10.json"           ; PHASE_NAME="biases_cifar10_panel"           ;;
      biases_cifar10_centered)  BASE="${RESULTS_DIR}/base_config_biases_cifar10_centered.json"  ; PHASE_NAME="biases_cifar10_centered_panel"  ;;
      biases_mnist_decentered)  BASE="${RESULTS_DIR}/base_config_biases_mnist_decentered.json"  ; PHASE_NAME="biases_mnist_decentered_panel"  ;;
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

  b3_strategies_mnist|b3_strategies_fashion|b3_strategies_cifar10)
    # B3 — encoder-strategy ablation. 7 cells × N seeds. Same shape as
    # the biases panel (knob='cell' is enumerative; no adoption).
    case "${EXPERIMENT}" in
      b3_strategies_mnist)   BASE="${RESULTS_DIR}/base_config_b3_strategies_mnist.json"   ; PHASE_NAME="b3_strategies_mnist_panel"   ;;
      b3_strategies_fashion) BASE="${RESULTS_DIR}/base_config_b3_strategies_fashion.json" ; PHASE_NAME="b3_strategies_fashion_panel" ;;
      b3_strategies_cifar10) BASE="${RESULTS_DIR}/base_config_b3_strategies_cifar10.json" ; PHASE_NAME="b3_strategies_cifar10_panel" ;;
    esac

    SEEDS="${SEEDS:-0-9}"
    CELLS="${VALUES:-hypersphere,gaussian,sparse,receptive_field,whitened,class_contrast,local_pca}"

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

    ${PYTHON} -u -m benchmarks.sl_clean.analyze_b3_panel \
        --results "${out}" \
        --out     "${RESULTS_DIR}/${PHASE_NAME}_summary.json"
    ;;

  b4_mlp_mnist|b4_mlp_fashion|b4_mlp_cifar10)
    # B4 — MLP baseline grid (12 cells × N seeds).
    case "${EXPERIMENT}" in
      b4_mlp_mnist)   BASE="${RESULTS_DIR}/base_config_b4_mlp_mnist.json"   ; PHASE_NAME="b4_mlp_mnist_panel"   ;;
      b4_mlp_fashion) BASE="${RESULTS_DIR}/base_config_b4_mlp_fashion.json" ; PHASE_NAME="b4_mlp_fashion_panel" ;;
      b4_mlp_cifar10) BASE="${RESULTS_DIR}/base_config_b4_mlp_cifar10.json" ; PHASE_NAME="b4_mlp_cifar10_panel" ;;
    esac

    SEEDS="${SEEDS:-0-4}"
    CELLS="${VALUES:-d1_w256_lr0.001,d1_w256_lr0.003,d1_w512_lr0.001,d1_w512_lr0.003,d1_w1024_lr0.001,d1_w1024_lr0.003,d2_w256_lr0.001,d2_w256_lr0.003,d2_w512_lr0.001,d2_w512_lr0.003,d2_w1024_lr0.001,d2_w1024_lr0.003}"

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

    ${PYTHON} -u -m benchmarks.sl_clean.analyze_b4_panel \
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

  streaming_alpha_sweep)
    # B2.alpha — calibrate regularisation: alpha_scale × alpha × n_neurons.
    # Cells encode 'alpha_scale+alpha_nominal+n_neurons'.
    BASE="${RESULTS_DIR}/base_config_streaming_alpha_sweep.json"
    SEEDS="${SEEDS:-0-4}"
    if [ -z "${VALUES:-}" ]; then
      cells=""
      for scale in fixed trace; do
        for a in 1e-6 1e-5 1e-4 1e-3 1e-2; do
          for n in 4000 8000; do
            cells="${cells:+${cells},}${scale}+${a}+${n}"
          done
        done
      done
      VALUES="${cells}"
    fi
    run_phase streaming_alpha_sweep cell "${VALUES}" "${SEEDS}" "${BASE}" "${EXPERIMENT}"
    ;;

  b5_lstm_smnist_row|b5_lstm_smnist_permuted)
    # B5 — LSTM gradient-trained baseline (sMNIST). 18 cells × N seeds.
    case "${EXPERIMENT}" in
      b5_lstm_smnist_row)      BASE="${RESULTS_DIR}/base_config_b5_lstm_smnist_row.json"      ; PHASE_NAME="b5_lstm_smnist_row_panel"      ;;
      b5_lstm_smnist_permuted) BASE="${RESULTS_DIR}/base_config_b5_lstm_smnist_permuted.json" ; PHASE_NAME="b5_lstm_smnist_permuted_panel" ;;
    esac

    SEEDS="${SEEDS:-0-4}"
    CELLS="${VALUES:-h64_l1_lr0.0003,h64_l1_lr0.001,h64_l1_lr0.003,h128_l1_lr0.0003,h128_l1_lr0.001,h128_l1_lr0.003,h256_l1_lr0.0003,h256_l1_lr0.001,h256_l1_lr0.003,h64_l2_lr0.0003,h64_l2_lr0.001,h64_l2_lr0.003,h128_l2_lr0.0003,h128_l2_lr0.001,h128_l2_lr0.003,h256_l2_lr0.0003,h256_l2_lr0.001,h256_l2_lr0.003}"

    out="${RESULTS_DIR}/${PHASE_NAME}.json"
    log="${LOGS_DIR}/${PHASE_NAME}.log"
    echo "================================================================"
    echo "  PHASE ${PHASE_NAME}  exp=${EXPERIMENT}  seeds=${SEEDS}"
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

    ${PYTHON} -u -m benchmarks.sl_clean.analyze_b5_panel \
        --results "${out}" \
        --out     "${RESULTS_DIR}/${PHASE_NAME}_summary.json"
    ;;

  b6_esn_smnist_row|b6_esn_smnist_permuted)
    # B6 — ESN gradient-free reservoir baseline (sMNIST).
    case "${EXPERIMENT}" in
      b6_esn_smnist_row)      BASE="${RESULTS_DIR}/base_config_b6_esn_smnist_row.json"      ; PHASE_NAME="b6_esn_smnist_row_panel"      ;;
      b6_esn_smnist_permuted) BASE="${RESULTS_DIR}/base_config_b6_esn_smnist_permuted.json" ; PHASE_NAME="b6_esn_smnist_permuted_panel" ;;
    esac

    SEEDS="${SEEDS:-0-4}"
    CELLS="${VALUES:-n500_sr0.7_leak0.3,n500_sr0.7_leak1.0,n500_sr0.95_leak0.3,n500_sr0.95_leak1.0,n1000_sr0.7_leak0.3,n1000_sr0.7_leak1.0,n1000_sr0.95_leak0.3,n1000_sr0.95_leak1.0,n2000_sr0.7_leak0.3,n2000_sr0.7_leak1.0,n2000_sr0.95_leak0.3,n2000_sr0.95_leak1.0}"

    out="${RESULTS_DIR}/${PHASE_NAME}.json"
    log="${LOGS_DIR}/${PHASE_NAME}.log"
    echo "================================================================"
    echo "  PHASE ${PHASE_NAME}  exp=${EXPERIMENT}  seeds=${SEEDS}"
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

    ${PYTHON} -u -m benchmarks.sl_clean.analyze_b6_panel \
        --results "${out}" \
        --out     "${RESULTS_DIR}/${PHASE_NAME}_summary.json"
    ;;

  c1_scaling_mnist|c1_scaling_fashion|c1_scaling_cifar10|\
c2_activation_mnist|c2_activation_fashion|c2_activation_cifar10|\
c3_ensemble_cifar10|\
c4_multilayer_mnist|c4_multilayer_cifar10|\
c5_recurrent_smnist|\
c6_convnef_mnist|c6_convnef_cifar10|\
c7_housing)
    # Phase C — supporting reruns. Each one is an enumerative knob='cell'
    # panel; no adoption analyzer (matches biases / b3 / b4 layout).  The
    # default cells / seeds are listed below per experiment; override via
    # the VALUES / SEEDS environment variables.
    BASE="${RESULTS_DIR}/base_config_${EXPERIMENT}.json"
    PHASE_NAME="${EXPERIMENT}_panel"

    case "${EXPERIMENT}" in
      c1_scaling_*)
        DEFAULT_CELLS="n500,n1000,n2000,n5000,n10000,n20000"
        DEFAULT_SEEDS="0-9"
        ;;
      c2_activation_*)
        DEFAULT_CELLS="abs+g0.5_2.0,abs+g1.0_1.0,abs+g0.1_5.0,relu+g0.5_2.0,relu+g1.0_1.0,relu+g0.1_5.0,tanh+g0.5_2.0,tanh+g1.0_1.0,tanh+g0.1_5.0,step+g0.5_2.0,step+g1.0_1.0,step+g0.1_5.0,square+g0.5_2.0,square+g1.0_1.0,square+g0.1_5.0"
        DEFAULT_SEEDS="0-9"
        ;;
      c3_ensemble_cifar10)
        DEFAULT_CELLS="m1+hypersphere,m1+receptive_field,m5+hypersphere,m5+receptive_field,m10+hypersphere,m10+receptive_field,m20+hypersphere,m20+receptive_field"
        DEFAULT_SEEDS="0-4"
        ;;
      c4_multilayer_*)
        DEFAULT_CELLS="greedy,hybrid,target_prop,end_to_end,hybrid_e2e"
        DEFAULT_SEEDS="0-4"
        ;;
      c5_recurrent_smnist)
        DEFAULT_CELLS="end_to_end,target_prop"
        DEFAULT_SEEDS="0-4"
        ;;
      c6_convnef_*)
        DEFAULT_CELLS="default"
        DEFAULT_SEEDS="0-4"
        ;;
      c7_housing)
        DEFAULT_CELLS="n500,n1000,n2000,n5000,n10000"
        DEFAULT_SEEDS="0-2"
        ;;
    esac

    CELLS="${VALUES:-${DEFAULT_CELLS}}"
    SEEDS="${SEEDS:-${DEFAULT_SEEDS}}"

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
    # No phase-C-specific analyzer yet; per-cell aggregation handled by
    # docs/sl_clean_sweeps.md note + downstream paper figure scripts.
    ;;

  *)
    echo "unknown experiment block: ${EXPERIMENT}" >&2
    echo "available: smoke biases_mnist biases_fashion biases_cifar10 biases_cifar10_centered biases_mnist_decentered b3_strategies_mnist b3_strategies_fashion b3_strategies_cifar10 b4_mlp_mnist b4_mlp_fashion b4_mlp_cifar10 b5_lstm_smnist_row b5_lstm_smnist_permuted b6_esn_smnist_row b6_esn_smnist_permuted streaming_smnist_row streaming_smnist_permuted streaming_solve_modes streaming_alpha_sweep c1_scaling_{mnist,fashion,cifar10} c2_activation_{mnist,fashion,cifar10} c3_ensemble_cifar10 c4_multilayer_{mnist,cifar10} c5_recurrent_smnist c6_convnef_{mnist,cifar10} c7_housing" >&2
    exit 2
    ;;
esac
