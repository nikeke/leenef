#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/results/abs_preimages"
mkdir -p "$RESULT_DIR"

LOG="$RESULT_DIR/abs_preimages_runner.log"
PROGRESS="$RESULT_DIR/abs_preimages_progress.tsv"
HEARTBEAT="$RESULT_DIR/abs_preimages_heartbeat"
DONE="$RESULT_DIR/abs_preimages_done"
FAILED="$RESULT_DIR/abs_preimages_failed"

PYTHON="${PYTHON:-$ROOT/../leenef/venv/bin/python}"
LEENEF_DATA_ROOT="${LEENEF_DATA_ROOT:-$ROOT/../leenef/data}"
export LEENEF_DATA_ROOT
CPU_THREADS="${CPU_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS}"

rm -f "$DONE" "$FAILED"
: > "$LOG"
: > "$PROGRESS"
touch "$HEARTBEAT"

steps=(
  "sign-freedom:benchmarks/abs_preimages/sign_freedom_experiment.py"
  "visualizations:benchmarks/abs_preimages/visualize_preimages.py"
  "canonical-prototypes:benchmarks/abs_preimages/canonical_prototypes.py"
  "nullspace-radius:benchmarks/abs_preimages/nullspace_radius_sweep.py"
  "prototype-initialization:benchmarks/abs_preimages/prototype_initialization_experiment.py"
  "pdf-build:docs/build_ssup_pdf.sh --verbose docs/abs_preimage_geometry.md docs/abs_preimage_geometry.pdf"
)

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

record() {
  printf '%s\t%s\n' "$(ts)" "$*" >> "$PROGRESS"
  touch "$HEARTBEAT"
}

run_step() {
  local index="$1"
  local label="$2"
  local command="$3"
  local start end duration
  start="$(date +%s)"
  record "BEGIN	$index	$label"
  {
    echo
    echo "===== $(ts) BEGIN $index/$TOTAL $label ====="
    if [[ "$label" == "pdf-build" ]]; then
      bash ${command#*:}
      if grep -q "Missing character:" "$LOG"; then
        echo "Missing-character warnings found in PDF build log" >&2
        return 1
      fi
    else
      "$PYTHON" "$command"
    fi
    echo "===== $(ts) END $index/$TOTAL $label ====="
  } >> "$LOG" 2>&1
  end="$(date +%s)"
  duration=$((end - start))
  record "END	$index	$label	$duration"
}

TOTAL="${#steps[@]}"
record "START	$TOTAL	$PYTHON	$LEENEF_DATA_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  record "FAIL	0	setup	0"
  echo "Python executable not found or not executable: $PYTHON" > "$FAILED"
  exit 1
fi

for i in "${!steps[@]}"; do
  step="${steps[$i]}"
  label="${step%%:*}"
  command="${step#*:}"
  if ! run_step "$((i + 1))" "$label" "$command"; then
    record "FAIL	$((i + 1))	$label	0"
    echo "$label failed; see $LOG" > "$FAILED"
    exit 1
  fi
done

record "DONE	$TOTAL"
touch "$DONE"
