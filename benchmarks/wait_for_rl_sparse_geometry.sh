#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/results/rl_sparse/geometry"
SENTINEL="$RESULT_DIR/geometry_chain_done"
STALL_SECONDS="${STALL_SECONDS:-7200}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REPORT_EVERY="${REPORT_EVERY:-300}"
EXPECTED_SEEDS="${EXPECTED_SEEDS:-10}"
EXPECTED_MODES="${EXPECTED_MODES:-dot cosine future_cosine}"
PARALLEL_MODES="${PARALLEL_MODES:-2}"
WIDTH="${WIDTH:-2000}"

last_report=0

latest_mtime() {
    find "$RESULT_DIR" -type f \( -name '*.json' -o -name '*.log' \) -printf '%T@\n' 2>/dev/null \
        | sort -nr \
        | awk 'NR == 1 {print; exit}'
}

progress_report() {
    EXPECTED_SEEDS="$EXPECTED_SEEDS" \
    EXPECTED_MODES="$EXPECTED_MODES" \
    PARALLEL_MODES="$PARALLEL_MODES" \
    WIDTH="$WIDTH" \
    python3 - <<'PY'
import json
import math
import os
import statistics
import time
from pathlib import Path

base = Path("results/rl_sparse/geometry")
expected_seeds = int(os.environ["EXPECTED_SEEDS"])
modes = os.environ["EXPECTED_MODES"].split()
parallel_modes = max(1, int(os.environ["PARALLEL_MODES"]))
width = int(os.environ["WIDTH"])
total_expected = expected_seeds * len(modes)
done_total = 0
times = []

for mode in modes:
    path = base / f"mountaincar_{mode}_n{width}.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        print(f"{path.name}: rows=0/{expected_seeds}")
        continue
    except Exception as exc:
        print(f"{path.name}: unreadable ({exc})")
        continue
    runs = data.get("runs", []) if isinstance(data, dict) else []
    ok = [r for r in runs if "error" not in r]
    errs = [r for r in runs if "error" in r]
    vals = [float(r["robust_eval"]) for r in ok if r.get("robust_eval") is not None]
    row_times = [float(r["time_s"]) for r in ok if r.get("time_s") is not None]
    done_total += len(ok)
    times.extend(row_times)
    if vals:
        se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        metric = f" robust={statistics.mean(vals):.1f} +/- {se:.1f}"
    else:
        metric = ""
    mean_time = f" mean_row_s={statistics.mean(row_times):.0f}" if row_times else ""
    print(
        f"{path.name}: rows={len(ok)}/{expected_seeds} errors={len(errs)}"
        f"{metric}{mean_time}"
    )

chain = base / "sparse_chain_geometry.json"
if chain.exists():
    try:
        data = json.loads(chain.read_text())
        rows = data.get("runs", []) if isinstance(data, dict) else []
        print(f"{chain.name}: rows={len(rows)}")
    except Exception as exc:
        print(f"{chain.name}: unreadable ({exc})")
else:
    print(f"{chain.name}: not written")

remaining = max(0, total_expected - done_total)
if times and remaining:
    eta_s = remaining * statistics.mean(times) / parallel_modes
    print(f"ETA_from_completed_rows: ~{eta_s / 3600:.2f} h ({remaining} rows remaining)")
elif remaining == 0:
    print("ETA_from_completed_rows: complete")
else:
    print("ETA_from_completed_rows: waiting for first completed row")

latest = max(
    (p.stat().st_mtime for p in base.glob("logs/*.log") if p.exists()),
    default=None,
)
if latest is not None:
    print(f"latest_log_age_s={int(time.time() - latest)}")
PY
}

mkdir -p "$RESULT_DIR"
echo "[$(date -Is)] watching sparse RL readout geometry chain"
echo "sentinel=$SENTINEL"
echo "stall_seconds=$STALL_SECONDS poll_seconds=$POLL_SECONDS"

while true; do
    now=$(date +%s)
    if [[ -f "$SENTINEL" ]]; then
        echo "[$(date -Is)] sparse RL readout geometry sentinel detected"
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
        echo "[$(date -Is)] sparse RL readout geometry progress; latest_file_age=${age}s"
        progress_report
        last_report=$now
    fi

    if [[ -n "${latest:-}" ]] && (( age > STALL_SECONDS )); then
        echo "[$(date -Is)] ERROR: sparse RL readout geometry appears stalled; latest file age ${age}s" >&2
        progress_report >&2
        exit 1
    fi

    sleep "$POLL_SECONDS"
done
