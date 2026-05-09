"""Resumable phase runner for ``cl_clean`` sweeps.

A *phase* is a single-knob sweep over a list of values, each evaluated
across a list of seeds.  The runner is idempotent on the tuple

    (namespace, runner_version, experiment, code_identity,
     config_with_overridden_knob, seed, run_context)

so an interrupted phase can be resumed by re-invoking the same command.

This is a verbatim mirror of ``benchmarks.sl_clean.sl_phase_runner``,
pointed at the ``cl_clean`` namespace.  The CL-specific machinery
(multi-task streams, sequential ``partial_fit``, EWC/ER baselines)
lives inside each experiment's ``run_fn``; the runner stays
experiment-agnostic.

Usage example::

    python -m benchmarks.cl_clean.cl_phase_runner \\
        --experiment smoke \\
        --phase smoke_n_neurons \\
        --knob n_neurons \\
        --values 100,200,400 \\
        --seeds 0,1,2 \\
        --base-config results/cl_clean/base_config_smoke.json \\
        --out results/cl_clean/smoke_n_neurons.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .cl_repro import (
    LOGS_DIR,
    RESULTS_DIR,
    can_reuse_result,
    code_identity,
    expected_config_with_knob,
    result_metadata,
)
from .experiments import get as get_experiment


def _value_tag(v: object) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _parse_values(raw: str) -> list[object]:
    out: list[object] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "true":
            out.append(True)
            continue
        if tok.lower() == "false":
            out.append(False)
            continue
        if tok.lower() == "none":
            out.append(None)
            continue
        try:
            out.append(int(tok))
            continue
        except ValueError:
            pass
        try:
            out.append(float(tok))
            continue
        except ValueError:
            pass
        out.append(tok)
    return out


def _parse_seeds(raw: str) -> list[int]:
    if "-" in raw and "," not in raw:
        a, b = raw.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(t) for t in raw.split(",") if t.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def run_phase(
    *,
    experiment: str,
    phase: str,
    knob: str,
    values: list[object],
    seeds: list[int],
    base_config: dict,
    out_path: Path,
    run_context: dict | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Execute ``phase`` and write its rows to ``out_path``.

    Returns the result dict.  Callers can inspect rows for ``error`` keys
    to detect cell failures; the CLI ``main`` exits non-zero if any cell
    in the requested run failed (cached rows from prior runs do not
    affect the exit status of the current invocation).
    """
    spec = get_experiment(experiment)
    expected_run_context = dict(spec.default_run_context())
    if run_context:
        expected_run_context.update(run_context)

    results = _load_json(out_path)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")

    log(
        f"phase={phase} experiment={experiment} knob={knob} "
        f"values={values} seeds={seeds} code_identity={code_identity()}"
    )

    total = len(values) * len(seeds)
    done = 0
    n_errors = 0
    for value in values:
        config = expected_config_with_knob(base_config, knob, value)
        for seed in seeds:
            done += 1
            cell_key = f"{phase}_{_value_tag(value)}_s{seed}"
            existing = results.get(cell_key)
            if existing and can_reuse_result(
                existing,
                experiment=experiment,
                expected_config=config,
                expected_seed=seed,
                expected_run_context=expected_run_context,
            ):
                log(f"  [{done}/{total}] skip   {cell_key}  (cached)")
                continue
            log(f"  [{done}/{total}] run    {cell_key}")
            t0 = time.perf_counter()
            try:
                metrics = spec.run_fn(config, seed, expected_run_context)
                row: dict[str, Any] = {
                    **result_metadata(
                        config=config,
                        seed=seed,
                        experiment=experiment,
                        run_context=expected_run_context,
                        primary_metric=spec.primary_metric,
                        higher_is_better=spec.higher_is_better,
                    ),
                    "phase": phase,
                    "knob": knob,
                    "value": value,
                    "label": cell_key,
                    "metrics": metrics,
                    "wall_time_s": round(time.perf_counter() - t0, 4),
                }
            except Exception as exc:
                n_errors += 1
                row = {
                    "phase": phase,
                    "knob": knob,
                    "value": value,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                log(f"  [{done}/{total}] ERROR {cell_key}: {row['error']}")
            results[cell_key] = row
            _atomic_write(out_path, results)
    if n_errors:
        log(f"phase {phase} finished with {n_errors} cell error(s) — see {out_path}")
    else:
        log(f"phase {phase} done — {len(results)} rows in {out_path}")
    # Report error count via a wrapping dict so callers can detect failures
    # without polluting the on-disk JSON.
    return {"results": results, "n_errors": n_errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--knob", required=True)
    ap.add_argument("--values", required=True, help="Comma-separated values")
    ap.add_argument("--seeds", required=True, help="Comma-separated seeds or A-B range")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)

    base_path = Path(args.base_config)
    if not base_path.exists():
        print(f"error: base config not found: {base_path}", file=sys.stderr)
        return 2
    with open(base_path) as f:
        base_config = json.load(f)

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.phase}.json"
    log_path = Path(args.log) if args.log else LOGS_DIR / f"{args.phase}.log"
    summary = run_phase(
        experiment=args.experiment,
        phase=args.phase,
        knob=args.knob,
        values=_parse_values(args.values),
        seeds=_parse_seeds(args.seeds),
        base_config=base_config,
        out_path=out_path,
        log_path=log_path,
    )
    return 1 if summary["n_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
