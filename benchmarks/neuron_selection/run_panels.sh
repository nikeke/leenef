#!/usr/bin/env bash
# PYTHON is required. Optional: SEEDS=0-9, THREADS=4, GROWTH_SHARDS=2,
# DATASETS="FashionMNIST CIFAR10 MNIST". Growth shards use contiguous seed slices.
# Use SMOKE=1 for the small panel grids and ALLOW_DIRTY=1 only for a local probe.
# Override comma-separated growth arms with MNIST_ARMS, FASHIONMNIST_ARMS,
# or CIFAR10_ARMS. MNIST/FashionMNIST default to all arms; CIFAR10 defaults
# to random,phd,phd_classshuffle,phd_randkeep,phd_patch (random in both families).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

out=results/neuron_selection
mkdir -p "$out/logs"
exec 9>"$out/driver.lock"
flock -n 9 || { echo "A neuron-selection panel driver is already running." >&2; exit 1; }
rm -f -- "$out/panels_done" "$out/panels_failed"

: "${PYTHON:?Set PYTHON to the venv Python executable}"
if [[ ! -d data ]]; then
    echo "Missing dataset root: $PWD/data (link or populate it before starting)." >&2
    exit 2
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if ! NS_IDENTITY=$("$PYTHON" -c \
    'from benchmarks.sl_clean.sl_repro import code_identity; print(code_identity() or "")') ||
    [[ -z "$NS_IDENTITY" ]]; then
    echo "Cannot determine the checkout's code identity." >&2
    exit 2
fi
if [[ "$NS_IDENTITY" == *-dirty* && "${ALLOW_DIRTY:-0}" != 1 ]]; then
    echo "Refusing dirty code identity: $NS_IDENTITY (use a clean worktree or ALLOW_DIRTY=1)." >&2
    exit 2
fi
export NS_IDENTITY

SEEDS=${SEEDS:-0-9}
THREADS=${THREADS:-4}
GROWTH_SHARDS=${GROWTH_SHARDS:-2}
read -r -a datasets <<< "${DATASETS:-FashionMNIST CIFAR10 MNIST}"
declare -A dataset_arms=()
manifest_arms=()
for dataset in "${datasets[@]}"; do
    case "$dataset" in
        MNIST) arms=${MNIST_ARMS:-} ;;
        FashionMNIST) arms=${FASHIONMNIST_ARMS:-} ;;
        CIFAR10) arms=${CIFAR10_ARMS:-random,phd,phd_classshuffle,phd_randkeep,phd_patch} ;;
        *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
    esac
    dataset_arms[$dataset]=$arms
    manifest_arms+=("$dataset=$arms")
done
smoke=()
if [[ "${SMOKE:-0}" == 1 ]]; then
    smoke=(--smoke)
fi
manifest="$out/logs/run_manifest.txt"
shard_list=$("$PYTHON" -m benchmarks.neuron_selection.common manifest "$manifest" \
    --identity "$NS_IDENTITY" --seeds "$SEEDS" --threads "$THREADS" --shards "$GROWTH_SHARDS" \
    "${smoke[@]}" "${manifest_arms[@]}")
mapfile -t shards <<< "$shard_list"

analyze() {
    local label=$1 status
    shift
    "$PYTHON" -u -m benchmarks.neuron_selection.analyze "$@" \
        --identity "$NS_IDENTITY" --exploratory MNIST || {
        status=$?
        echo "analyze failed for $label (exit $status)" >&2
        printf '{"event":"analyze_failed","summary":"%s","status":%s}\n' "$label" "$status" \
            >> "$manifest" || echo "could not append analyzer failure to $manifest" >&2
    }
}

all_files=()
for dataset in "${datasets[@]}"; do
    arms=${dataset_arms[$dataset]}
    growth_args=()
    if [[ -n "$arms" ]]; then
        growth_args=(--arms "$arms")
    fi

    prune_file="$out/prune_$dataset.json"
    growth_files=()
    "$PYTHON" -u -m benchmarks.neuron_selection.prune_panel \
        --dataset "$dataset" --seeds "$SEEDS" --threads "$THREADS" "${smoke[@]}" \
        --out "$prune_file" >> "$out/logs/prune_$dataset.log" 2>&1 &
    declare -A children=(["$!"]="prune_$dataset")
    for i in "${!shards[@]}"; do
        stem="growth_${dataset}_s$i"
        growth_files+=("$out/$stem.json")
        "$PYTHON" -u -m benchmarks.neuron_selection.growth_panel \
            --dataset "$dataset" --seeds "${shards[$i]}" --threads "$THREADS" "${smoke[@]}" \
            --out "$out/$stem.json" "${growth_args[@]}" \
            >> "$out/logs/$stem.log" 2>&1 &
        children[$!]=$stem
    done
    remaining=${#children[@]}
    while ((remaining > 0)); do
        status=0
        wait -n -p child "${!children[@]}" 2>/dev/null || status=$?
        if [[ -z "${child:-}" ]]; then
            # Bash can omit a job already reaped by the shell from wait -n.
            for child in "${!children[@]}"; do break; done
            status=0
            wait "$child" || status=$?
        fi
        label=${children[$child]}
        unset 'children[$child]'
        remaining=$((remaining - 1))
        if ((status != 0)); then
            printf 'dataset=%s child=%s status=%s\n' "$dataset" "$label" "$status" \
                > "$out/.panels_failed.$$"
            mv -- "$out/.panels_failed.$$" "$out/panels_failed"
            cat "$out/panels_failed" >&2
            if ((remaining > 0)); then
                kill -TERM "${!children[@]}" 2>/dev/null || true
                for pid in "${!children[@]}"; do wait "$pid" 2>/dev/null || true; done
            fi
            exit "$status"
        fi
    done
    analyze "$dataset" "$prune_file" "${growth_files[@]}" \
        > "$out/logs/summary_$dataset.md" 2>&1
    all_files+=("$prune_file" "${growth_files[@]}")
done

analyze all "${all_files[@]}" > "$out/logs/summary_all.md" 2>&1
touch "$out/panels_done"
