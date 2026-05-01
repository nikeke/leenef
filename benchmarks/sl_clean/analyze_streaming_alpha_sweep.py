"""Analyzer for the streaming_alpha_sweep experiment.

Cells encode 'alpha_scale+alpha_nominal+n_neurons'.  We group by
``(n_neurons, alpha_scale, alpha_nominal)`` → list of seed accuracies and
report:

* per-cell mean ± SE of ``test_accuracy``;
* per-cell median ``alpha_eff`` (so trace-scaled curves are anchored on
  the actual ridge they applied);
* per-(n, scale) accuracy-vs-alpha curve and best alpha;
* cross-(n) consistency check: does a single (scale, alpha) win at both
  n=4000 and n=8000, or do we need n-dependent defaults?

The verdict block recommends a single ``(alpha_scale, alpha_nominal)``
default for the streaming module based on:

1. accuracy at the best n and at the worst n
2. flatness across n (SE-pooled gap |acc(n=8k) − acc(n=4k)|)
3. preference for trace-scaling when competitive (handles future n).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from .sl_repro import RESULT_NAMESPACE


def _se(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def _is_data_row(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and "error" not in row
        and row.get("result_namespace") == RESULT_NAMESPACE
        and row.get("experiment") == "streaming_alpha_sweep"
        and "metrics" in row
    )


def _parse_cell(cell: str) -> tuple[str, float, int]:
    scale, alpha_nominal, n_neurons = cell.split("+")
    return scale, float(alpha_nominal), int(n_neurons)


def aggregate(rows: Iterable[dict]) -> dict:
    """Group rows by (n_neurons, alpha_scale, alpha_nominal) → per-cell stats."""
    cells: dict[tuple[int, str, float], dict] = {}
    n_used = 0
    for row in rows:
        if not _is_data_row(row):
            continue
        cell_str = str(row["config"]["cell"])
        scale, alpha_nominal, n_neurons = _parse_cell(cell_str)
        seed = int(row["seed"])
        metrics = row["metrics"]
        diag = metrics.get("diagnostics", {})
        acc = float(metrics["test_accuracy"])
        alpha_eff = diag.get("alpha_eff")
        key = (n_neurons, scale, alpha_nominal)
        cell = cells.setdefault(
            key,
            {
                "n_neurons": n_neurons,
                "alpha_scale": scale,
                "alpha_nominal": alpha_nominal,
                "by_seed": {},
                "alpha_eff_by_seed": {},
            },
        )
        cell["by_seed"][seed] = acc
        if alpha_eff is not None:
            cell["alpha_eff_by_seed"][seed] = float(alpha_eff)
        n_used += 1
    out_cells = []
    for (n_neurons, scale, alpha_nominal), cell in cells.items():
        accs = list(cell["by_seed"].values())
        eff = list(cell["alpha_eff_by_seed"].values())
        cell["accs"] = accs
        cell["n_seeds"] = len(accs)
        cell["mean"] = mean(accs) if accs else float("nan")
        cell["se"] = _se(accs)
        cell["alpha_eff_median"] = median(eff) if eff else float("nan")
        out_cells.append(cell)
    out_cells.sort(key=lambda c: (c["n_neurons"], c["alpha_scale"], c["alpha_nominal"]))
    return {"cells": out_cells, "raw_count": n_used}


def _curves_by_group(cells: list[dict]) -> dict[tuple[int, str], list[dict]]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for c in cells:
        key = (c["n_neurons"], c["alpha_scale"])
        grouped.setdefault(key, []).append(c)
    for key in grouped:
        grouped[key].sort(key=lambda c: c["alpha_nominal"])
    return grouped


def best_per_group(cells: list[dict]) -> dict[tuple[int, str], dict]:
    grouped = _curves_by_group(cells)
    best: dict[tuple[int, str], dict] = {}
    for key, curve in grouped.items():
        viable = [c for c in curve if c["mean"] == c["mean"]]
        if not viable:
            continue
        winner = max(viable, key=lambda c: c["mean"])
        best[key] = winner
    return best


def _paired_gap(a_cell: dict, b_cell: dict) -> tuple[float, float]:
    sa, sb = a_cell["by_seed"], b_cell["by_seed"]
    common = sorted(set(sa) & set(sb))
    ds = [sb[s] - sa[s] for s in common]
    if not ds:
        return float("nan"), float("nan")
    return mean(ds), _se(ds)


def consistency_across_n(cells: list[dict]) -> list[dict]:
    """For each (alpha_scale, alpha_nominal): paired gap acc(n=8k)−acc(n=4k)."""
    by_sa: dict[tuple[str, float], dict[int, dict]] = {}
    for c in cells:
        key = (c["alpha_scale"], c["alpha_nominal"])
        by_sa.setdefault(key, {})[c["n_neurons"]] = c
    rows = []
    for (scale, alpha_nominal), per_n in sorted(by_sa.items()):
        if 4000 in per_n and 8000 in per_n:
            a, b = per_n[4000], per_n[8000]
            d_mean, d_se = _paired_gap(a, b)
            rows.append(
                {
                    "alpha_scale": scale,
                    "alpha_nominal": alpha_nominal,
                    "acc_4k": a["mean"],
                    "acc_8k": b["mean"],
                    "delta_8k_minus_4k": d_mean,
                    "delta_se": d_se,
                    "min_acc": min(a["mean"], b["mean"]),
                }
            )
    return rows


def recommend_default(cells: list[dict]) -> dict:
    """Pick a single (scale, alpha) that maximises the worst-case mean across n.

    Tie-break (within 1 SE) on flatness across n, then on simpler (trace
    is more principled, so prefer trace when tied).
    """
    cons = consistency_across_n(cells)
    if not cons:
        return {"verdict": "inconclusive", "reason": "missing data at one n"}
    by_min = sorted(cons, key=lambda r: -r["min_acc"])
    top_min = by_min[0]["min_acc"]
    pool = [
        r for r in by_min if (top_min - r["min_acc"]) <= max(0.002, 2 * (r["delta_se"] or 0.0))
    ]
    pool.sort(key=lambda r: (abs(r["delta_8k_minus_4k"]), r["alpha_scale"] != "trace"))
    winner = pool[0]
    return {
        "verdict": "recommended_default",
        "alpha_scale": winner["alpha_scale"],
        "alpha_nominal": winner["alpha_nominal"],
        "min_acc": winner["min_acc"],
        "delta_8k_minus_4k": winner["delta_8k_minus_4k"],
        "delta_se": winner["delta_se"],
        "ties_considered": len(pool),
    }


def _fmt_pct(x: float) -> str:
    if x != x:
        return "  n/a"
    return f"{100 * x:6.2f}"


def _fmt_pp(x: float) -> str:
    if x != x:
        return "  n/a"
    return f"{100 * x:+6.2f}"


def print_report(agg: dict) -> None:
    cells = agg["cells"]
    print(f"\nB2 streaming alpha sweep — {agg['raw_count']} rows aggregated\n")

    grouped = _curves_by_group(cells)
    for n_neurons, scale in sorted(grouped):
        curve = grouped[(n_neurons, scale)]
        print(f"\n=== n_neurons={n_neurons}, alpha_scale={scale} ===\n")
        print("  alpha_nominal       acc%       SE   alpha_eff(median)   n_seeds")
        for c in curve:
            print(
                f"  {c['alpha_nominal']:>10.0e}    "
                f"{_fmt_pct(c['mean'])}   {_fmt_pct(c['se'])}   "
                f"{c['alpha_eff_median']:14.4g}      {c['n_seeds']}"
            )

    print("\n=== Best per (n_neurons, alpha_scale) ===\n")
    print("  n_neurons  scale     alpha_nominal     acc%       SE   alpha_eff(median)")
    for key, w in sorted(best_per_group(cells).items()):
        print(
            f"  {w['n_neurons']:>9}  {w['alpha_scale']:<6}  "
            f"{w['alpha_nominal']:>10.0e}      "
            f"{_fmt_pct(w['mean'])}   {_fmt_pct(w['se'])}   "
            f"{w['alpha_eff_median']:14.4g}"
        )

    print("\n=== Cross-n consistency (paired delta acc(8k) − acc(4k)) ===\n")
    cons = consistency_across_n(cells)
    print("  scale     alpha_nominal   acc%(4k)   acc%(8k)   Δ(pp)    SE     min_acc%")
    for r in cons:
        print(
            f"  {r['alpha_scale']:<6}  {r['alpha_nominal']:>10.0e}     "
            f"{_fmt_pct(r['acc_4k'])}    {_fmt_pct(r['acc_8k'])}   "
            f"{_fmt_pp(r['delta_8k_minus_4k'])}   {_fmt_pp(r['delta_se'])}   "
            f"{_fmt_pct(r['min_acc'])}"
        )

    rec = recommend_default(cells)
    print("\n=== Recommended streaming default ===\n")
    if rec["verdict"] != "recommended_default":
        print(f"  verdict: {rec['verdict']}  reason: {rec.get('reason')}")
    else:
        print(f"  alpha_scale   : {rec['alpha_scale']}")
        print(f"  alpha_nominal : {rec['alpha_nominal']:.0e}")
        print(f"  worst-case acc: {_fmt_pct(rec['min_acc'])}%")
        print(
            f"  Δ(8k − 4k)    : {_fmt_pp(rec['delta_8k_minus_4k'])}  ± {_fmt_pp(rec['delta_se'])}"
        )
        print(f"  ties pooled   : {rec['ties_considered']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to streaming_alpha_sweep.json (phase output)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional summary JSON output path",
    )
    args = parser.parse_args(argv)

    data = json.loads(args.results.read_text())
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    agg = aggregate(rows)
    print_report(agg)

    if args.out is not None:
        rec = recommend_default(agg["cells"])
        summary = {
            "raw_count": agg["raw_count"],
            "cells": agg["cells"],
            "best_per_group": {
                f"n={k[0]},scale={k[1]}": v for k, v in best_per_group(agg["cells"]).items()
            },
            "consistency_across_n": consistency_across_n(agg["cells"]),
            "recommendation": rec,
        }
        args.out.write_text(json.dumps(summary, indent=2, default=float))
        print(f"\nWrote summary → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
