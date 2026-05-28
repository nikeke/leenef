"""Panel analyzer for the B3 encoder-strategy ablation.

Knob ``cell`` enumerates the 7 strategies; bias_mode is fixed to
``data_driven``.  This analyzer computes:

* per-cell ``mean ± SE`` of ``test_accuracy`` and ``encode_fit_time_s``;
* paired-by-seed Δ vs the ``hypersphere`` baseline for each other
  strategy, plus the ``Δ / sqrt(SE_a² + SE_b²)`` z-score;
* a verdict per strategy ("strictly above hypersphere" /
  "strictly below" / "ties hypersphere") under the standard
  rl_clean adoption rule (|z| ≥ 1).

Outputs printed to stdout and (optionally) written as JSON via ``--out``.
This script never mutates the base config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

from .experiments import _B3_STRATEGIES  # type: ignore[reportPrivateUsage]
from .sl_repro import RESULT_NAMESPACE

_BASELINE = "hypersphere"
_B3_EXPERIMENTS = {
    "b3_strategies_mnist",
    "b3_strategies_fashion",
    "b3_strategies_cifar10",
}


def _is_data_row(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and "error" not in row
        and row.get("result_namespace") == RESULT_NAMESPACE
        and row.get("experiment") in _B3_EXPERIMENTS
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
    by_cell: dict[str, dict] = {s: {"by_seed": {}, "fit_by_seed": {}} for s in _B3_STRATEGIES}
    raw = 0
    for r in rows:
        if not _is_data_row(r):
            continue
        raw += 1
        cell = r.get("config", {}).get("cell")
        if cell not in _B3_STRATEGIES:
            continue
        seed = int(r["seed"])
        m = r["metrics"]
        by_cell[cell]["by_seed"][seed] = float(m.get("test_accuracy", float("nan")))
        by_cell[cell]["fit_by_seed"][seed] = float(m.get("encode_fit_time_s", float("nan")))

    summary: dict[str, dict] = {}
    for s, d in by_cell.items():
        accs = sorted(d["by_seed"].items())
        if not accs:
            continue
        vals = [v for _, v in accs]
        fits = [d["fit_by_seed"].get(seed, float("nan")) for seed, _ in accs]
        summary[s] = {
            "mean": mean(vals),
            "se": _se(vals),
            "n": len(vals),
            "fit_mean": mean(fits) if fits else float("nan"),
            "by_seed": dict(accs),
        }
    return {"by_strategy": summary, "raw_count": raw}


def pairwise_vs_baseline(by_strategy: dict[str, dict]) -> dict[str, dict]:
    if _BASELINE not in by_strategy:
        return {}
    base = by_strategy[_BASELINE]
    base_seeds = base["by_seed"]
    out: dict[str, dict] = {}
    for s, d in by_strategy.items():
        if s == _BASELINE:
            continue
        # paired-by-seed
        common = sorted(set(d["by_seed"]) & set(base_seeds))
        if len(common) < 2:
            continue
        diffs = [d["by_seed"][k] - base_seeds[k] for k in common]
        m = mean(diffs)
        sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
        se_paired = sd / math.sqrt(len(diffs))
        # also unpaired-style adoption SE for reporting
        se_unpaired = math.sqrt(d["se"] ** 2 + base["se"] ** 2)
        z = m / se_paired if se_paired > 0 else float("nan")
        if abs(z) < 1:
            verdict = f"ties {_BASELINE}"
        elif z > 0:
            verdict = f"strictly above {_BASELINE}"
        else:
            verdict = f"strictly below {_BASELINE}"
        out[s] = {
            "delta_pp": m * 100,
            "se_paired_pp": se_paired * 100,
            "se_unpaired_pp": se_unpaired * 100,
            "z": z,
            "n": len(common),
            "verdict": verdict,
        }
    return out


def render(summary: dict, vs_base: dict, sweep_path: str) -> str:
    lines: list[str] = []
    lines.append(f"=== B3 encoder-strategy panel — {sweep_path} ===")
    lines.append("")
    lines.append(f"Per-strategy mean ± SE (n=seeds), bias_mode=data_driven, baseline={_BASELINE}:")
    lines.append("")
    lines.append(f"  {'strategy':<18s} {'mean%':>8s} {'SE pp':>7s} {'n':>3s} {'fit_s':>7s}")
    for s in _B3_STRATEGIES:
        d = summary.get(s)
        if d is None:
            lines.append(f"  {s:<18s} (no data)")
            continue
        lines.append(
            f"  {s:<18s} {d['mean'] * 100:>8.3f} {d['se'] * 100:>7.3f} "
            f"{d['n']:>3d} {d['fit_mean']:>7.2f}"
        )
    lines.append("")
    lines.append(f"Δ vs {_BASELINE}, paired-by-seed, z = Δ / SE_paired:")
    lines.append("")
    lines.append(f"  {'strategy':<18s} {'Δ pp':>8s} {'SE pp':>7s} {'z':>7s} {'n':>3s}  verdict")
    for s in _B3_STRATEGIES:
        if s == _BASELINE:
            continue
        d = vs_base.get(s)
        if d is None:
            lines.append(f"  {s:<18s} (no comparison)")
            continue
        lines.append(
            f"  {s:<18s} {d['delta_pp']:>+8.3f} {d['se_paired_pp']:>7.3f} "
            f"{d['z']:>+7.2f} {d['n']:>3d}  {d['verdict']}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    raw = json.loads(args.results.read_text())
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    agg = aggregate(rows)
    vs_base = pairwise_vs_baseline(agg["by_strategy"])

    text = render(agg["by_strategy"], vs_base, str(args.results))
    print(text)

    if args.out is not None:
        out_payload = {
            "results_path": str(args.results),
            "raw_count": agg["raw_count"],
            "by_strategy": {
                s: {
                    "mean": d["mean"],
                    "se": d["se"],
                    "n": d["n"],
                    "fit_mean": d["fit_mean"],
                }
                for s, d in agg["by_strategy"].items()
            },
            "vs_baseline": vs_base,
            "baseline": _BASELINE,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_payload, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
