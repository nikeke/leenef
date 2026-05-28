"""Analyze a B5 LSTM panel (cf. analyze_b4_panel.py)."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_CELL_RE = re.compile(r"h(\d+)_l(\d+)_lr([0-9.]+(?:e-?\d+)?)")


def _parse(cell: str) -> tuple[int, int, float]:
    m = _CELL_RE.fullmatch(cell)
    if not m:
        raise ValueError(f"unrecognised cell: {cell!r}")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def aggregate(rows: list[dict]) -> list[dict]:
    by_cell: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "metrics" not in row:
            continue
        cfg = row.get("config", {}) or {}
        cell = cfg.get("cell")
        if cell is None:
            continue
        by_cell.setdefault(cell, []).append(row)

    out = []
    for cell, cells in sorted(by_cell.items()):
        accs = [r["metrics"]["test_accuracy"] for r in cells]
        n = len(accs)
        mean = sum(accs) / n
        var = sum((a - mean) ** 2 for a in accs) / max(1, n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n) if n > 1 else 0.0
        hidden, layers, lr = _parse(cell)
        wall = sum(r["metrics"]["diagnostics"]["wall_sec"] for r in cells) / n
        epochs = sum(r["metrics"]["diagnostics"]["epochs_run"] for r in cells) / n
        params = cells[0]["metrics"]["diagnostics"]["n_params"]
        out.append(
            {
                "cell": cell,
                "hidden": hidden,
                "layers": layers,
                "lr": lr,
                "n_seeds": n,
                "mean_acc": mean,
                "se_acc": se,
                "mean_wall_sec": wall,
                "mean_epochs": epochs,
                "n_params": params,
                "by_seed": {r["seed"]: r["metrics"]["test_accuracy"] for r in cells},
            }
        )
    return out


def pick_winner(agg: list[dict]) -> tuple[dict, list[dict]]:
    best = max(agg, key=lambda c: c["mean_acc"])
    pool = []
    for c in agg:
        se_comb = math.sqrt(best["se_acc"] ** 2 + c["se_acc"] ** 2)
        margin = max(0.002, 2.0 * se_comb)
        if best["mean_acc"] - c["mean_acc"] <= margin:
            pool.append(c)
    recommended = min(pool, key=lambda c: c["n_params"])
    return recommended, pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    raw = json.loads(args.results.read_text())
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    agg = aggregate(rows)
    if not agg:
        print(f"no data in {args.results}", file=sys.stderr)
        return 1

    print(f"=== B5 LSTM panel — {args.results} ===\n")
    print(f"Per-cell mean test acc ± SE (n=seeds), {len(agg)} cells:\n")
    print(
        f"  {'cell':<22}{'mean%':>8}{'SE pp':>8}{'n':>4}{'epochs':>8}{'wall_s':>9}{'params':>10}"
    )
    for c in sorted(agg, key=lambda c: -c["mean_acc"]):
        print(
            f"  {c['cell']:<22}"
            f"{c['mean_acc'] * 100:>8.3f}"
            f"{c['se_acc'] * 100:>8.3f}"
            f"{c['n_seeds']:>4}"
            f"{c['mean_epochs']:>8.1f}"
            f"{c['mean_wall_sec']:>9.1f}"
            f"{c['n_params']:>10}"
        )

    rec, pool = pick_winner(agg)
    print()
    print(f"Tie pool (within max(0.2pp, 2·SE_combined) of the top mean): {len(pool)} cell(s)")
    for c in sorted(pool, key=lambda c: -c["mean_acc"]):
        print(
            f"  - {c['cell']:<22} mean={c['mean_acc'] * 100:.3f}% SE={c['se_acc'] * 100:.3f}pp params={c['n_params']}"
        )
    print()
    print(
        f"Recommended (smallest in pool): {rec['cell']} "
        f"({rec['mean_acc'] * 100:.3f}% ± {rec['se_acc'] * 100:.3f}pp, "
        f"params={rec['n_params']})"
    )

    summary = {
        "results_path": str(args.results),
        "n_cells": len(agg),
        "cells": agg,
        "tie_pool": [c["cell"] for c in pool],
        "recommended": rec,
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
