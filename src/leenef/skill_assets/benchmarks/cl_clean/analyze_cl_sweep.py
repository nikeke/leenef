"""Aggregate one ``cl_clean`` sweep file and decide adoption.

Mirrors ``attic/benchmarks/analyze_rl_sweep.py``.  The adoption rule is
the same:

    adopt new value v* iff
        mean(metric | v=v*) - mean(metric | v=base) > sqrt(SE_v*² + SE_base²)

For losses or other "lower is better" metrics, set ``--minimize`` or
rely on the experiment's ``higher_is_better`` flag.

The script never deletes data: it only updates ``base_config.json`` when
a winner clears the bar, and prints a per-value table either way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .cl_repro import is_current_sweep_row
from .experiments import get as get_experiment


def _tag(v: object) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def aggregate(
    sweep: dict,
    *,
    experiment: str,
    knob: str,
    base_config: dict,
    expected_run_context: dict,
    metric: str,
) -> dict[str, dict]:
    by_value: dict[str, dict] = {}
    for r in sweep.values():
        if not is_current_sweep_row(
            r,
            experiment=experiment,
            base_config=base_config,
            knob=knob,
            expected_run_context=expected_run_context,
        ):
            continue
        m = r.get("metrics", {}).get(metric)
        if m is None:
            continue
        tag = _tag(r["value"])
        by_value.setdefault(tag, {"value": r["value"], "samples": []})
        by_value[tag]["samples"].append(float(m))

    stats = {}
    for tag, d in by_value.items():
        arr = np.asarray(d["samples"], dtype=float)
        n = arr.size
        mean = float(arr.mean()) if n else float("nan")
        se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        stats[tag] = {
            "value": d["value"],
            "n": n,
            "mean": mean,
            "se": se,
            "min": float(arr.min()) if n else float("nan"),
            "max": float(arr.max()) if n else float("nan"),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--knob", required=True)
    ap.add_argument("--sweep-json", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--metric", default=None, help="Override primary metric")
    ap.add_argument("--minimize", action="store_true", help="Lower is better")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    spec = get_experiment(args.experiment)
    metric = args.metric or spec.primary_metric
    higher_is_better = (not args.minimize) and spec.higher_is_better

    base_path = Path(args.base_config)
    if not base_path.exists():
        print(f"error: base config not found: {base_path}", file=sys.stderr)
        return 2
    with open(base_path) as f:
        base = json.load(f)

    sweep_path = Path(args.sweep_json)
    if not sweep_path.exists():
        print(f"error: sweep file not found: {sweep_path}", file=sys.stderr)
        return 2
    with open(sweep_path) as f:
        sweep = json.load(f)

    expected_run_context = spec.default_run_context()
    stats = aggregate(
        sweep,
        experiment=args.experiment,
        knob=args.knob,
        base_config=base,
        expected_run_context=expected_run_context,
        metric=metric,
    )
    if not stats:
        print(
            f"[analyze] no rows matched experiment={args.experiment} knob={args.knob} "
            f"in {sweep_path}"
        )
        return 0

    sign = -1 if not higher_is_better else 1
    ordered = sorted(stats.items(), key=lambda kv: -sign * kv[1]["mean"])
    direction = "↑" if higher_is_better else "↓"

    print("=" * 78)
    print(f"  ANALYSIS  experiment={args.experiment}  knob={args.knob}")
    print(f"  metric={metric}  ({direction} better)   sweep={sweep_path}")
    print("=" * 78)
    print(f"  {'value':>14}  {'n':>3}  {'mean':>10}  {'± SE':>8}  {'min':>10}  {'max':>10}")
    for tag, s in ordered:
        se_str = f"{s['se']:.4f}" if not np.isnan(s["se"]) else "  nan"
        print(
            f"  {tag:>14}  {s['n']:>3}  {s['mean']:>10.4f}  ± {se_str}  "
            f"{s['min']:>10.4f}  {s['max']:>10.4f}"
        )

    current_value = base.get(args.knob)
    current_tag = _tag(current_value)
    winner_tag, winner = ordered[0]
    print(f"\n  winner:       {args.knob}={winner_tag}  (mean={winner['mean']:.4f})")
    print(
        f"  current base: {args.knob}={current_tag}  "
        f"(in sweep: {'yes' if current_tag in stats else 'no'})"
    )

    if current_tag not in stats:
        print("  → current base not in sweep values; keeping base unchanged")
        return 0
    if winner["n"] < args.min_seeds:
        print(f"  → winner has only {winner['n']} seeds (<{args.min_seeds}); keeping base")
        return 0
    if winner_tag == current_tag:
        print("  → base is already the winner; no change")
        return 0

    cur = stats[current_tag]
    delta = sign * (winner["mean"] - cur["mean"])
    se_comb = float(np.sqrt(winner["se"] ** 2 + cur["se"] ** 2))
    if np.isnan(se_comb) or se_comb == 0:
        print("  → SE_combined is NaN or zero (need ≥2 seeds per value); keeping base")
        return 0
    ratio = delta / se_comb
    print(f"  Δ = {delta:.4f}   SE_combined = {se_comb:.4f}   ratio = {ratio:.2f}")
    if delta > se_comb:
        print(f"  ✅ ADOPT: {args.knob}: {current_value!r} → {winner['value']!r}")
        if not args.dry_run:
            base[args.knob] = winner["value"]
            tmp = str(base_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(base, f, indent=2, sort_keys=True)
            os.replace(tmp, base_path)
            print(f"     wrote {base_path}")
    else:
        print("  → Δ within 1 SE_combined; keeping base unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
