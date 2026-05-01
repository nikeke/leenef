#!/usr/bin/env bash
# Watch B2 panel runs.  Foreground; intended to be invoked by the agent's
# bash tool with a long initial_wait so the agent gets a completion
# notification when this exits.
#
# Exits 0 when results/sl_clean/b2_panels_done exists.
# Exits 1 if all watched logs go silent for STALL_SECS (= 1800 by default,
# tightened to 1200 for cheaper sweeps).  Reports state every POLL_SECS.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGDIR="results/sl_clean/logs"
DONE="results/sl_clean/b2_panels_done"
POLL_SECS="${POLL_SECS:-60}"
STALL_SECS="${STALL_SECS:-1800}"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

last_progress=$(date +%s)

while true; do
    if [[ -f "$DONE" ]]; then
        echo "[$(ts)] done sentinel found — all panels finished"
        exit 0
    fi

    # Most-recent mtime across the panel logs (only files that exist).
    newest_mtime=0
    newest_log=""
    for log in "$LOGDIR"/b2_row.log "$LOGDIR"/b2_permuted.log "$LOGDIR"/b2_modes.log; do
        [[ -f "$log" ]] || continue
        m=$(stat -c %Y "$log")
        if (( m > newest_mtime )); then
            newest_mtime=$m
            newest_log=$log
        fi
    done

    now=$(date +%s)
    if (( newest_mtime > 0 )); then
        if (( newest_mtime > last_progress )); then
            last_progress=$newest_mtime
        fi
        idle=$(( now - last_progress ))
        last_line=$(tail -n 1 "$newest_log" 2>/dev/null || echo "")
        echo "[$(ts)] watching ($idle s idle); last: $last_line"
        if (( idle > STALL_SECS )); then
            echo "[$(ts)] STALLED — no log progress for ${STALL_SECS}s; check ps + logs"
            ps -ef | grep -E "sl_phase|sl_hp_driver|run_b2_panels" | grep -v grep || true
            exit 1
        fi
    else
        echo "[$(ts)] no logs yet"
    fi

    sleep "$POLL_SECS"
done
