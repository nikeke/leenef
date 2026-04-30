"""Panel analyzer for the B1 ``biases_mnist`` factorial sweep.

Unlike :mod:`benchmarks.sl_clean.analyze_sl_sweep`, this analyzer does
**not** treat ``cell`` as a single knob to adopt.  H1 is a *factorial
panel claim*:

1. Without data-driven biases, encoder families differ by ~2-3 %.
2. With data-driven biases, encoder families collapse to within ~0.5 %.

The script computes:

* per-cell ``mean ± SE`` of ``test_accuracy``;
* paired-by-seed pairwise encoder differences within each bias mode;
* ``max_pairwise_gap`` per bias mode (mean over seed-paired max gaps);
* a verdict per bias mode ("collapse supported" / "gap present" /
  "inconclusive") using a simple decision rule documented inline;
* delta vs adoption-rule SE for the headline contrasts
  ``data_driven`` ↔ ``random_normal`` per encoder.

Outputs are printed to stdout and optionally written as JSON via
``--out``.  This script never mutates the base config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

from .experiments import (  # type: ignore[reportPrivateUsage]
    _BIAS_MODES,
    _ENCODER_STRATEGIES,
    _parse_biases_cell,
)
from .sl_repro import RESULT_NAMESPACE


def _is_data_row(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and "error" not in row
        and row.get("result_namespace") == RESULT_NAMESPACE
        and row.get("experiment") == "biases_mnist"
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
    """Group rows by (bias_mode, encoder_strategy, seed) → per-cell stats.

    Returns:
        {
          "by_cell": {bm: {es: {"acc": [..], "mean": .., "se": .., "n": ..,
                              "by_seed": {seed: acc}}}},
          "raw_count": int,
        }
    """
    by_cell: dict[str, dict[str, dict]] = {
        bm: {es: {"by_seed": {}} for es in _ENCODER_STRATEGIES} for bm in _BIAS_MODES
    }
    n_used = 0
    for row in rows:
        if not _is_data_row(row):
            continue
        try:
            bm, es = _parse_biases_cell(str(row["config"]["cell"]))
        except Exception:
            continue
        seed = int(row["seed"])
        acc = float(row["metrics"]["test_accuracy"])
        by_cell[bm][es]["by_seed"][seed] = acc
        n_used += 1
    for bm in _BIAS_MODES:
        for es in _ENCODER_STRATEGIES:
            cell = by_cell[bm][es]
            accs = list(cell["by_seed"].values())
            cell["acc"] = accs
            cell["n"] = len(accs)
            cell["mean"] = mean(accs) if accs else float("nan")
            cell["se"] = _se(accs)
    return {"by_cell": by_cell, "raw_count": n_used}


def _paired_max_gaps(by_cell_for_bm: dict[str, dict]) -> tuple[list[float], list[int]]:
    """For each seed present in *all three* encoder strategies, compute
    ``max(acc) - min(acc)`` across encoders, paired by seed.

    Returns (gaps, seeds_used).
    """
    seeds_per_es = [set(by_cell_for_bm[es]["by_seed"]) for es in _ENCODER_STRATEGIES]
    common = sorted(set.intersection(*seeds_per_es)) if seeds_per_es else []
    gaps = []
    for s in common:
        vals = [by_cell_for_bm[es]["by_seed"][s] for es in _ENCODER_STRATEGIES]
        gaps.append(max(vals) - min(vals))
    return gaps, common


def panel_summary(agg: dict, *, collapse_threshold: float = 0.005) -> dict:
    """Build the H1 panel verdicts.

    The decision rule per bias mode:

    * **collapse supported** if
      ``mean(max_pairwise_gap) + SE ≤ collapse_threshold``;
    * **gap present** if
      ``mean(max_pairwise_gap) - SE ≥ 2 × collapse_threshold``
      (a simple "robustly above ~1 %" rule);
    * otherwise **inconclusive**.
    """
    out: dict = {"collapse_threshold": collapse_threshold, "by_bias_mode": {}}
    for bm in _BIAS_MODES:
        gaps, seeds = _paired_max_gaps(agg["by_cell"][bm])
        bm_summary: dict = {
            "n_paired_seeds": len(seeds),
            "max_gap_mean": mean(gaps) if gaps else float("nan"),
            "max_gap_se": _se(gaps),
            "max_gap_min": min(gaps) if gaps else float("nan"),
            "max_gap_max": max(gaps) if gaps else float("nan"),
        }
        if not gaps or len(gaps) < 3:
            verdict = "inconclusive"
        elif bm_summary["max_gap_mean"] + bm_summary["max_gap_se"] <= collapse_threshold:
            verdict = "collapse_supported"
        elif bm_summary["max_gap_mean"] - bm_summary["max_gap_se"] >= 2 * collapse_threshold:
            verdict = "gap_present"
        else:
            verdict = "inconclusive"
        bm_summary["verdict"] = verdict
        out["by_bias_mode"][bm] = bm_summary
    return out


def headline_contrasts(agg: dict) -> dict:
    """Per-encoder ``data_driven - random_normal`` paired-by-seed deltas."""
    contrasts: dict = {}
    for es in _ENCODER_STRATEGIES:
        dd = agg["by_cell"]["data_driven"][es]["by_seed"]
        rn = agg["by_cell"]["random_normal"][es]["by_seed"]
        common = sorted(set(dd) & set(rn))
        deltas = [dd[s] - rn[s] for s in common]
        contrasts[es] = {
            "n": len(deltas),
            "delta_mean": mean(deltas) if deltas else float("nan"),
            "delta_se": _se(deltas),
            "data_driven_mean": agg["by_cell"]["data_driven"][es]["mean"],
            "random_normal_mean": agg["by_cell"]["random_normal"][es]["mean"],
        }
    return contrasts


def _fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "  n/a"
    return f"{100 * x:5.2f}"


def _fmt_pp(x: float) -> str:
    if x != x:
        return "  n/a"
    return f"{100 * x:+5.2f}"


def print_report(agg: dict, panel: dict, contrasts: dict) -> None:
    print(f"\nB1 biases_mnist panel — {agg['raw_count']} rows aggregated\n")
    print("Per-cell test_accuracy (mean ± SE  [n]):\n")
    header = "  " + "{:18s}".format("bias_mode \\ encoder")
    for es in _ENCODER_STRATEGIES:
        header += " {:>16s}".format(es)
    print(header)
    for bm in _BIAS_MODES:
        line = "  " + "{:18s}".format(bm)
        for es in _ENCODER_STRATEGIES:
            cell = agg["by_cell"][bm][es]
            n = cell["n"]
            if n == 0:
                line += " {:>16s}".format("—")
            else:
                line += " {:>9s}±{:>4s} [{:>2d}]".format(
                    _fmt_pct(cell["mean"]), _fmt_pct(cell["se"]).strip(), n
                )
        print(line)

    print("\nMax pairwise encoder gap per bias mode (paired by seed):\n")
    print(
        "  {:18s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>5s}  {}".format(
            "bias_mode", "mean", "se", "min", "max", "n", "verdict"
        )
    )
    for bm in _BIAS_MODES:
        s = panel["by_bias_mode"][bm]
        print(
            "  {:18s}  {:>9s}%  {:>9s}%  {:>9s}%  {:>9s}%  {:>5d}  {}".format(
                bm,
                _fmt_pct(s["max_gap_mean"]).strip(),
                _fmt_pct(s["max_gap_se"]).strip(),
                _fmt_pct(s["max_gap_min"]).strip(),
                _fmt_pct(s["max_gap_max"]).strip(),
                s["n_paired_seeds"],
                s["verdict"],
            )
        )
    print(f"  (collapse_threshold = {100 * panel['collapse_threshold']:.2f}%)")

    print("\nHeadline contrasts: data_driven − random_normal (paired by seed):\n")
    print(
        "  {:14s}  {:>9s}  {:>9s}  {:>9s}  {:>5s}".format(
            "encoder", "Δ mean", "Δ SE", "ratio", "n"
        )
    )
    for es in _ENCODER_STRATEGIES:
        c = contrasts[es]
        ratio = (
            (c["delta_mean"] / c["delta_se"])
            if c["delta_se"] and c["delta_se"] == c["delta_se"]
            else float("nan")
        )
        print(
            "  {:14s}  {:>8s}pp  {:>8s}pp  {:>9.2f}  {:>5d}".format(
                es,
                _fmt_pp(c["delta_mean"]).strip(),
                _fmt_pp(c["delta_se"]).strip(),
                ratio if ratio == ratio else float("nan"),
                c["n"],
            )
        )
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--results", required=True, help="Path to biases_mnist.json")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.005,
        help="Max-gap threshold for 'collapse supported' (default 0.005 = 0.5pp)",
    )
    args = ap.parse_args(argv)

    with open(args.results) as f:
        data = json.load(f)
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    agg = aggregate(rows)
    panel = panel_summary(agg, collapse_threshold=args.collapse_threshold)
    contrasts = headline_contrasts(agg)
    print_report(agg, panel, contrasts)
    if args.out:
        out = {"aggregate": agg, "panel": panel, "contrasts": contrasts}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True, default=lambda o: None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
