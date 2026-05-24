#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/results/rl_paper/dense"
LOG_DIR="$RESULT_DIR/logs"
SENTINEL="$RESULT_DIR/dense_chain_done"
STALL_SECONDS="${STALL_SECONDS:-7200}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REPORT_EVERY="${REPORT_EVERY:-300}"

last_report=0

latest_mtime() {
    find "$RESULT_DIR" -type f \( -name '*.json' -o -name '*.log' \) -printf '%T@\n' 2>/dev/null \
        | sort -nr \
        | head -n 1
}

progress_report() {
    python3 - <<'PY'
import json
from pathlib import Path

base = Path("results/rl_paper/dense")
for path in sorted(base.glob("*.json")):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"{path.name}: unreadable ({exc})")
        continue
    runs = data.get("runs", []) if isinstance(data, dict) else []
    done = sum(1 for r in runs if "error" not in r)
    errs = sum(1 for r in runs if "error" in r)
    labels = sorted({str(r.get("label")) for r in runs})
    print(f"{path.name}: rows={len(runs)} ok={done} errors={errs} labels={','.join(labels)}")
PY
}

mkdir -p "$LOG_DIR"
echo "[$(date -Is)] watching dense RL chain"
echo "sentinel=$SENTINEL"
echo "stall_seconds=$STALL_SECONDS poll_seconds=$POLL_SECONDS"

while true; do
    now=$(date +%s)
    if [[ -f "$SENTINEL" ]]; then
        echo "[$(date -Is)] dense RL chain sentinel detected"
        progress_report
        exit 0
    fi

    latest="$(latest_mtime || true)"
    if [[ -n "$latest" ]]; then
        latest_int="${latest%.*}"
        age=$((now - latest_int))
    else
        age=0
    fi

    if (( now - last_report >= REPORT_EVERY )); then
        echo
        echo "[$(date -Is)] dense RL progress; latest_file_age=${age}s"
        progress_report
        last_report=$now
    fi

    if [[ -n "${latest:-}" ]] && (( age > STALL_SECONDS )); then
        echo "[$(date -Is)] ERROR: dense RL chain appears stalled; latest file age ${age}s" >&2
        progress_report >&2
        exit 1
    fi

    sleep "$POLL_SECONDS"
done
