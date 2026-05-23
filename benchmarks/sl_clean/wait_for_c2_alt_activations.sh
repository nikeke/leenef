#!/usr/bin/env bash
# Watch the C2 alt-activation run. Foreground; exits when every expected
# row for the current code identity is present, or when progress stalls.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/amnesia/Persistent/src/leenef/venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

LOGDIR="results/sl_clean/logs"
DONE="results/sl_clean/c2_alt_activations_done"
POLL_SECS="${POLL_SECS:-60}"
STALL_SECS="${STALL_SECS:-1800}"
SEEDS_ARG="${SEEDS:-0-9}"
VALUES_ARG="${VALUES:-step+g0.5_2.0,step+g1.0_1.0,step+g0.1_5.0,square+g0.5_2.0,square+g1.0_1.0,square+g0.1_5.0}"
START_TS="$(date +%s)"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

count_rows() {
  "${PYTHON}" - "$SEEDS_ARG" "$VALUES_ARG" <<'PY'
import json
import sys
from pathlib import Path

from benchmarks.sl_clean.sl_repro import code_identity

seeds_raw, values_raw = sys.argv[1:3]

def parse_seeds(raw: str) -> list[int]:
    if "-" in raw and "," not in raw:
        lo, hi = raw.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(tok) for tok in raw.split(",") if tok.strip()]

seeds = parse_seeds(seeds_raw)
values = [tok.strip() for tok in values_raw.split(",") if tok.strip()]
expected = len(seeds) * len(values)
identity = code_identity()
panels = {
    "mnist": Path("results/sl_clean/c2_activation_mnist_panel.json"),
    "fashion": Path("results/sl_clean/c2_activation_fashion_panel.json"),
    "cifar10": Path("results/sl_clean/c2_activation_cifar10_panel.json"),
}
total_done = 0
total_expected = expected * len(panels)
parts = []
for name, path in panels.items():
    if path.exists():
        with open(path) as f:
            rows = json.load(f)
    else:
        rows = {}
    done = 0
    errors = 0
    for row in rows.values():
        if row.get("code_identity") != identity:
            continue
        if row.get("value") not in values:
            continue
        if row.get("seed") not in seeds:
            continue
        if "error" in row:
            errors += 1
        elif row.get("metrics", {}).get("test_accuracy") is not None:
            done += 1
    total_done += done
    parts.append(f"{name}={done}/{expected} errors={errors}")
print(f"{total_done}/{total_expected} " + " ".join(parts))
PY
}

last_progress=$(date +%s)
last_done=-1

while true; do
  progress="$(count_rows)"
  done_rows="${progress%%/*}"
  total_rows_rest="${progress#*/}"
  total_rows="${total_rows_rest%% *}"
  now="$(date +%s)"

  if (( done_rows > last_done )); then
    last_done="${done_rows}"
    last_progress="${now}"
  fi

  elapsed=$(( now - START_TS ))
  if (( done_rows > 0 )); then
    eta=$(( elapsed * (total_rows - done_rows) / done_rows ))
  else
    eta=-1
  fi

  eta_msg="warming up"
  if (( eta >= 0 )); then
    eta_msg="$(( eta / 60 ))m remaining"
  fi
  echo "[$(ts)] progress ${progress}; elapsed=$(( elapsed / 60 ))m; eta=${eta_msg}"

  if (( done_rows >= total_rows )); then
    if [[ -f "$DONE" ]]; then
      echo "[$(ts)] done sentinel found and all expected rows are current"
    else
      echo "[$(ts)] all expected rows are current; done sentinel not yet written"
    fi
    exit 0
  fi

  newest_mtime=0
  newest_log=""
  for log in "$LOGDIR"/c2_alt_c2_activation_*.log "$LOGDIR"/c2_activation_*_panel.log; do
    [[ -f "$log" ]] || continue
    m="$(stat -c %Y "$log")"
    if (( m > newest_mtime )); then
      newest_mtime="$m"
      newest_log="$log"
    fi
  done

  if (( newest_mtime > last_progress )); then
    last_progress="$newest_mtime"
  fi
  idle=$(( now - last_progress ))
  if (( idle > STALL_SECS )); then
    echo "[$(ts)] STALLED — no row/log progress for ${STALL_SECS}s"
    [[ -n "$newest_log" ]] && tail -n 20 "$newest_log" || true
    exit 1
  fi

  sleep "$POLL_SECS"
done
