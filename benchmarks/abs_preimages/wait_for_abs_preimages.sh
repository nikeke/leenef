#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/results/abs_preimages"
PROGRESS="$RESULT_DIR/abs_preimages_progress.tsv"
HEARTBEAT="$RESULT_DIR/abs_preimages_heartbeat"
DONE="$RESULT_DIR/abs_preimages_done"
FAILED="$RESULT_DIR/abs_preimages_failed"
LOG="$RESULT_DIR/abs_preimages_runner.log"

POLL_SECONDS="${POLL_SECONDS:-15}"
STALL_SECONDS="${STALL_SECONDS:-900}"
MODE="${1:---next-progress}"

completed_count() {
  [[ -f "$PROGRESS" ]] || {
    echo 0
    return
  }
  awk -F '\t' '$2 == "END" {count++} END {print count + 0}' "$PROGRESS"
}

total_count() {
  [[ -f "$PROGRESS" ]] || {
    echo 0
    return
  }
  awk -F '\t' '$2 == "START" {total=$3} END {print total + 0}' "$PROGRESS"
}

started_at() {
  [[ -f "$PROGRESS" ]] || return
  awk -F '\t' '$2 == "START" {print $1; exit}' "$PROGRESS"
}

last_line() {
  [[ -f "$PROGRESS" ]] || return
  tail -n 1 "$PROGRESS"
}

summary() {
  local completed total elapsed avg remaining eta start_epoch now_epoch
  completed="$(completed_count)"
  total="$(total_count)"
  if [[ "$total" -eq 0 ]]; then
    echo "Waiting for runner to create progress file..."
    return
  fi
  start_epoch="$(date -d "$(started_at)" +%s 2>/dev/null || date +%s)"
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - start_epoch))
  if [[ "$completed" -gt 0 ]]; then
    avg=$((elapsed / completed))
    remaining=$(((total - completed) * avg))
    eta="$(date -u -d "@$((now_epoch + remaining))" +"%Y-%m-%dT%H:%M:%SZ")"
  else
    remaining=0
    eta="unknown"
  fi
  echo "Progress: $completed/$total complete; elapsed ${elapsed}s; ETA $eta"
  echo "Last event: $(last_line)"
  if [[ -f "$LOG" ]]; then
    echo "Recent log:"
    tail -n 12 "$LOG"
  fi
}

initial_completed="$(completed_count)"

while true; do
  if [[ -f "$FAILED" ]]; then
    summary
    echo "Runner failed: $(cat "$FAILED")"
    exit 1
  fi
  if [[ -f "$DONE" ]]; then
    summary
    echo "Runner completed."
    exit 0
  fi
  current_completed="$(completed_count)"
  if [[ "$MODE" == "--next-progress" && "$current_completed" -gt "$initial_completed" ]]; then
    summary
    exit 0
  fi
  if [[ -f "$HEARTBEAT" ]]; then
    now="$(date +%s)"
    heartbeat_mtime="$(stat -c %Y "$HEARTBEAT")"
    if (( now - heartbeat_mtime > STALL_SECONDS )); then
      summary
      echo "Runner appears stalled: heartbeat older than ${STALL_SECONDS}s."
      exit 1
    fi
  fi
  sleep "$POLL_SECONDS"
done
