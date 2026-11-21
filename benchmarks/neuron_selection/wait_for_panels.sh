#!/usr/bin/env bash
# Run after starting the driver, with the same PYTHON (default: ./venv/bin/python).
# Whole seconds: STALL_SECONDS=2700, POLL_SECONDS=300 (result/status interval).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
shopt -s nullglob

stall=${STALL_SECONDS:-2700}
poll=${POLL_SECONDS:-300}
out=results/neuron_selection
manifest="$out/logs/run_manifest.txt"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
started=$(date +%s)
fail_if_marked() {
    if [[ -f "$out/panels_failed" ]]; then
        cat "$out/panels_failed" >&2
        exit 1
    fi
}

while true; do
    fail_if_marked
    status=0
    progress=$("${PYTHON:-$PWD/venv/bin/python}" -m benchmarks.neuron_selection.common \
        status "$manifest") || status=$?
    fail_if_marked
    if ((status == 0)); then
        printf '[%s] panels done: %s\n' "$(date -Is)" "$progress"
        exit 0
    elif ((status != 2)); then
        printf '[%s] invalid or incomplete results: %s\n' "$(date -Is)" "$progress" >&2
        exit 1
    fi
    now=$(date +%s)
    latest=$started
    for file in "$manifest" "$out"/prune_*.json "$out"/growth_*_s[0-9]*.json \
        "$out"/logs/prune_*.log "$out"/logs/growth_*_s[0-9]*.log "$out"/logs/summary_*.md; do
        mtime=$(stat -c '%Y' -- "$file" 2>/dev/null) || continue
        if ((mtime > latest)); then
            latest=$mtime
        fi
    done
    age=$((now - latest))
    if ((age >= stall)); then
        printf '[%s] stalled: no updates for %ss; %s\n' "$(date -Is)" "$age" "$progress" >&2
        exit 1
    fi
    printf '[%s] running: last activity %ss ago; %s\n' "$(date -Is)" "$age" "$progress"
    for ((elapsed=0; elapsed<poll; elapsed++)); do
        sleep 1
        fail_if_marked
    done
done
