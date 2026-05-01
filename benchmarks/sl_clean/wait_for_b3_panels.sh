#!/usr/bin/env bash
# Watchdog for B3 panels. Foreground sync; auto-notify on completion.
set -euo pipefail
cd "$(dirname "$0")/../.."

POLL_SECS="${POLL_SECS:-300}"
STALL_SECS="${STALL_SECS:-2400}"
SENTINEL="results/sl_clean/b3_panels_done"
LOGS_DIR="results/sl_clean/logs"

declare -A LAST_SIZE
declare -A LAST_CHANGE
NOW=$(date -u +%s)
for f in mnist fashion cifar10; do
  log="${LOGS_DIR}/b3_${f}.log"
  LAST_SIZE[$f]=0
  LAST_CHANGE[$f]=$NOW
done

while [ ! -e "${SENTINEL}" ]; do
  NOW=$(date -u +%s)
  echo "===== watchdog tick $(date -u +%H:%M:%S) ====="
  for f in mnist fashion cifar10; do
    log="${LOGS_DIR}/b3_${f}.log"
    if [ -e "${log}" ]; then
      sz=$(stat -c%s "${log}")
      if [ "${sz}" != "${LAST_SIZE[$f]}" ]; then
        LAST_SIZE[$f]=$sz
        LAST_CHANGE[$f]=$NOW
      fi
      stall=$((NOW - LAST_CHANGE[$f]))
      tail -1 "${log}" 2>/dev/null | sed "s|^|  ${f}: |"
      if [ "${stall}" -gt "${STALL_SECS}" ]; then
        echo "STALL ALERT: ${f} log unchanged for ${stall}s"
      fi
    else
      echo "  ${f}: (no log yet)"
    fi
  done
  sleep "${POLL_SECS}"
done

echo "===== B3 panels done at $(date -u +%H:%M:%S) ====="
