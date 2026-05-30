"""Aggregate the alt-activation panel into mean +/- SE per cell per dataset.

Reads ``results/sl_clean/c2_altact_{dataset}_panel.json`` and writes a compact
``c2_altact_{dataset}_summary.json`` plus a combined ``c2_altact_summary.json``,
and prints a table.  This is the analyzer behind
``docs/activation_kernel_theory.md`` section 8.3.

Rows are grouped by *committed* content-id (the part of ``code_identity``
before any ``-dirty-`` suffix): the dirty suffix only reflects uncommitted
analysis/launcher scaffolding, not the computation code, so rows that share a
content-id ran identical encoders, solver, harness, and activations.  Pure
stdlib (no torch), so it stays fast under CPU contention.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "sl_clean"

DATASETS = ("mnist", "fashion", "cifar10")

# Display order: references, kink family, smooth/odd relatives, even/RBF
# family at the main gain, then the cos/gauss gain sweep.
ORDER = (
    "abs+g0.5_2.0",
    "relu+g0.5_2.0",
    "leaky_relu+g0.5_2.0",
    "bent_line+g0.5_2.0",
    "softplus+g0.5_2.0",
    "tanh+g0.5_2.0",
    "square+g0.5_2.0",
    "double_fold+g0.5_2.0",
    "sqrt_abs+g0.5_2.0",
    "cos+g0.5_2.0",
    "gauss+g0.5_2.0",
    "cos+g0.05_0.2",
    "gauss+g0.05_0.2",
    "cos+g0.1_5.0",
    "gauss+g0.1_5.0",
)


def _content_id(row: dict) -> str:
    return str(row.get("code_identity", "")).split("-dirty-")[0]


def _mean_se(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n), n


def _aggregate(path: Path) -> dict:
    rows = json.load(open(path))
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    ident = collections.Counter(_content_id(r) for r in rows).most_common(1)[0][0]
    cells: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        if _content_id(r) != ident:
            continue
        acc = r.get("metrics", {}).get("test_accuracy")
        if acc is not None:
            cells[r["value"]].append(acc)
    summary = {}
    for cell, accs in cells.items():
        m, se, n = _mean_se(accs)
        summary[cell] = {"mean": m, "se": se, "n": n}
    return {"code_identity": ident, "cells": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    combined = {}
    for ds in datasets:
        path = RESULTS_DIR / f"c2_altact_{ds}_panel.json"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        agg = _aggregate(path)
        summary = agg["cells"]
        json.dump(
            {"dataset": ds, **agg},
            open(RESULTS_DIR / f"c2_altact_{ds}_summary.json", "w"),
            indent=2,
        )
        combined[ds] = summary
        print(f"\n=== {ds}  (content_id {agg['code_identity'][:16]}) ===")
        keys = [c for c in ORDER if c in summary]
        keys += [c for c in sorted(summary) if c not in ORDER]
        for cell in keys:
            s = summary[cell]
            print(f"  {cell:24s}  {s['mean']:.4f} +/- {s['se']:.4f}  (n={s['n']})")

    json.dump(combined, open(RESULTS_DIR / "c2_altact_summary.json", "w"), indent=2)
    print("\n[c2_altact] wrote c2_altact_{dataset}_summary.json + c2_altact_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
