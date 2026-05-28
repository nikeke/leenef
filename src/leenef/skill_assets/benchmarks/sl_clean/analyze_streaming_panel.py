"""Panel analyzer for the B2 streaming sMNIST experiments.

Three experiments share this analyzer:

* ``streaming_smnist_row``  — knob ``n_neurons`` (H4 panel: row).
* ``streaming_smnist_permuted`` — knob ``n_neurons`` (H4 panel: permuted).
* ``streaming_solve_modes`` — knob ``solve_mode`` (H3 equivalence).

For the n_neurons sweeps it computes:

* per-cell ``mean ± SE`` of ``test_accuracy``;
* paired-by-seed deltas between adjacent ``n_neurons`` values
  (e.g. 4000 − 2000, 8000 − 4000), with their SE and σ-ratio so we
  can read off the marginal value of doubling neurons.

For the solve_modes equivalence panel it computes the paired-by-seed
gap between modes and an "equivalence supported" verdict if the gap is
within a configurable tolerance (default 0.2 pp).

This script never mutates any base config.  Outputs are printed to
stdout and optionally written as JSON via ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

from ._streaming_exp import STREAMING_SOLVE_MODES
from .sl_repro import RESULT_NAMESPACE

_KNOB_BY_EXPERIMENT = {
    "streaming_smnist_row": "n_neurons",
    "streaming_smnist_permuted": "n_neurons",
    "streaming_solve_modes": "solve_mode",
}


def _is_data_row(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and "error" not in row
        and row.get("result_namespace") == RESULT_NAMESPACE
        and row.get("experiment") in _KNOB_BY_EXPERIMENT
        and "metrics" in row
    )


def _se(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def aggregate(rows: Iterable[dict]) -> dict:
    """Group rows by (experiment, knob_value, seed) → per-cell stats."""
    by_exp: dict[str, dict] = {}
    n_used = 0
    for row in rows:
        if not _is_data_row(row):
            continue
        exp = row["experiment"]
        knob = _KNOB_BY_EXPERIMENT[exp]
        value = row["config"][knob]
        seed = int(row["seed"])
        acc = float(row["metrics"]["test_accuracy"])
        per_exp = by_exp.setdefault(exp, {"knob": knob, "by_value": {}})
        cell = per_exp["by_value"].setdefault(value, {"by_seed": {}})
        cell["by_seed"][seed] = acc
        n_used += 1
    for per_exp in by_exp.values():
        for cell in per_exp["by_value"].values():
            accs = list(cell["by_seed"].values())
            cell["acc"] = accs
            cell["n"] = len(accs)
            cell["mean"] = mean(accs) if accs else float("nan")
            cell["se"] = _se(accs)
    return {"by_experiment": by_exp, "raw_count": n_used}


def _sorted_values(per_exp: dict) -> list:
    keys = list(per_exp["by_value"].keys())
    if per_exp["knob"] == "solve_mode":
        order = {m: i for i, m in enumerate(STREAMING_SOLVE_MODES)}
        return sorted(keys, key=lambda v: order.get(v, len(order)))
    try:
        return sorted(keys, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(keys, key=str)


def paired_deltas(per_exp: dict) -> list[dict]:
    """For each consecutive (a, b) pair of knob values, compute paired-by-seed b - a."""
    values = _sorted_values(per_exp)
    deltas = []
    for a, b in zip(values[:-1], values[1:]):
        sa = per_exp["by_value"][a]["by_seed"]
        sb = per_exp["by_value"][b]["by_seed"]
        common = sorted(set(sa) & set(sb))
        ds = [sb[s] - sa[s] for s in common]
        d_mean = mean(ds) if ds else float("nan")
        d_se = _se(ds)
        ratio = d_mean / d_se if d_se and d_se == d_se and d_se != 0 else float("nan")
        deltas.append(
            {
                "from": a,
                "to": b,
                "n": len(ds),
                "delta_mean": d_mean,
                "delta_se": d_se,
                "sigma_ratio": ratio,
            }
        )
    return deltas


def equivalence_panel(per_exp: dict, *, tol: float = 0.002) -> dict:
    """All pairwise paired-by-seed |gaps| ≤ tol → equivalence supported."""
    values = _sorted_values(per_exp)
    pairs: list[dict] = []
    worst_abs = 0.0
    enough = True
    for i, a in enumerate(values):
        for b in values[i + 1 :]:
            sa = per_exp["by_value"][a]["by_seed"]
            sb = per_exp["by_value"][b]["by_seed"]
            common = sorted(set(sa) & set(sb))
            ds = [sb[s] - sa[s] for s in common]
            if len(ds) < 3:
                enough = False
            d_mean = mean(ds) if ds else float("nan")
            d_se = _se(ds)
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "n": len(ds),
                    "delta_mean": d_mean,
                    "delta_se": d_se,
                }
            )
            if d_mean == d_mean:
                worst_abs = max(worst_abs, abs(d_mean))
    if not pairs or not enough:
        verdict = "inconclusive"
    elif worst_abs <= tol:
        verdict = "equivalence_supported"
    else:
        verdict = "gap_present"
    return {"tol": tol, "worst_abs_delta": worst_abs, "verdict": verdict, "pairs": pairs}


def _fmt_pct(x: float) -> str:
    if x != x:
        return "  n/a"
    return f"{100 * x:6.2f}"


def _fmt_pp(x: float) -> str:
    if x != x:
        return "  n/a"
    return f"{100 * x:+6.2f}"


def print_report(agg: dict, *, equiv_tol: float) -> None:
    print(f"\nB2 streaming panel — {agg['raw_count']} rows aggregated\n")
    for exp_name in sorted(agg["by_experiment"]):
        per_exp = agg["by_experiment"][exp_name]
        knob = per_exp["knob"]
        values = _sorted_values(per_exp)
        print(f"\n=== {exp_name}  (knob={knob}) ===\n")
        print("  Per-cell test_accuracy (mean ± SE  [n]):\n")
        for v in values:
            cell = per_exp["by_value"][v]
            print(
                f"    {knob}={v!s:<12s} "
                f"{_fmt_pct(cell['mean'])}% ± {_fmt_pct(cell['se']).strip()}%  "
                f"[n={cell['n']}]"
            )
        if knob == "solve_mode":
            panel = equivalence_panel(per_exp, tol=equiv_tol)
            print(f"\n  H3 equivalence (tol = {100 * panel['tol']:.2f}pp):  {panel['verdict']}")
            print(f"  Worst |Δ| = {100 * panel['worst_abs_delta']:.3f} pp")
            print("  Pairwise paired-by-seed deltas:")
            for p in panel["pairs"]:
                print(
                    f"    {p['a']:<12s} → {p['b']:<12s}  "
                    f"Δ {_fmt_pp(p['delta_mean']).strip():>7s} pp  "
                    f"SE {_fmt_pp(p['delta_se']).strip():>7s} pp  [n={p['n']}]"
                )
        else:
            deltas = paired_deltas(per_exp)
            if deltas:
                print("\n  Marginal effect of doubling n_neurons (paired by seed):")
                for d in deltas:
                    sigma = d["sigma_ratio"]
                    sigma_s = f"{sigma:6.1f}σ" if sigma == sigma else "  n/a"
                    print(
                        f"    {d['from']!s:<6s} → {d['to']!s:<6s}  "
                        f"Δ {_fmt_pp(d['delta_mean']).strip():>7s} pp  "
                        f"SE {_fmt_pp(d['delta_se']).strip():>7s} pp  "
                        f"ratio {sigma_s}  [n={d['n']}]"
                    )
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more streaming_*.json files to aggregate.",
    )
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument(
        "--equiv-tol",
        type=float,
        default=0.002,
        help="Max |Δ| between solve modes to declare equivalence (default 0.002 = 0.2pp).",
    )
    args = ap.parse_args(argv)

    rows: list[dict] = []
    for path in args.results:
        with open(path) as f:
            data = json.load(f)
        rows.extend(data.values() if isinstance(data, dict) else data)

    agg = aggregate(rows)
    print_report(agg, equiv_tol=args.equiv_tol)
    if args.out:
        out = {"aggregate": agg, "equiv_tol": args.equiv_tol}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True, default=lambda o: None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
