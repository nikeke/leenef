#!/usr/bin/env python3
"""Plot broadness sweep aggregates with min-max bars.

Usage:
    python benchmarks/plot_broadness_results.py
    python benchmarks/plot_broadness_results.py \
        --inputs results/rl/broadness_lunarlander_corrected.json \
                 results/rl/broadness_lunarlander_6000_edges.json

The script is intentionally tolerant of partial sweep files: it aggregates
whatever completed runs are already present and can be rerun later as more
results are added to the same JSON files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

DEFAULT_INPUTS = [
    Path("results/rl/broadness_lunarlander_corrected.json"),
    Path("results/rl/broadness_lunarlander_6000_edges.json"),
]

OUTDIR = Path("results/rl")
COLORS = {
    4000: "#0072B2",
    6000: "#D55E00",
    8000: "#009E73",
}


def _load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for item in data.values():
            if "error" in item:
                continue
            config = item.get("config", {})
            n_neurons = config.get("n_neurons")
            broadness = item.get("broadness")
            if n_neurons is None or broadness is None:
                continue
            records.append(item)
    return records


def _aggregate(records: list[dict], metric: str) -> dict[int, list[dict]]:
    grouped: dict[int, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in records:
        value = item.get(metric)
        if value is None:
            continue
        n_neurons = int(item["config"]["n_neurons"])
        broadness = float(item["broadness"])
        grouped[n_neurons][broadness].append(float(value))

    out: dict[int, list[dict]] = {}
    for n_neurons, by_broadness in grouped.items():
        rows = []
        for broadness in sorted(by_broadness):
            values = by_broadness[broadness]
            rows.append(
                {
                    "broadness": broadness,
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
            )
        out[n_neurons] = rows
    return out


def _plot_metric(ax: "Axes", aggregated: dict[int, list[dict]], metric: str) -> None:
    for n_neurons in sorted(aggregated):
        rows = aggregated[n_neurons]
        x = np.array([row["broadness"] for row in rows], dtype=float)
        y = np.array([row["mean"] for row in rows], dtype=float)
        ymin = np.array([row["min"] for row in rows], dtype=float)
        ymax = np.array([row["max"] for row in rows], dtype=float)
        yerr = np.vstack([y - ymin, ymax - y])
        color = COLORS.get(n_neurons, "#444444")
        label = f"{n_neurons} neurons"
        counts = {row["count"] for row in rows}
        if len(counts) == 1 and next(iter(counts)) < 3:
            label += f" (partial, n={next(iter(counts))})"
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            markersize=5,
            linewidth=1.8,
            label=label,
        )

    ax.set_title(metric.replace("_", " "))
    ax.set_xlabel("Broadness")
    ax.set_ylabel("Eval return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot broadness sweep summaries")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUTS],
        help="Broadness JSON files to aggregate",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTDIR),
        help="Directory for output PNGs",
    )
    args = parser.parse_args()

    records = _load_records([Path(path) for path in args.inputs])
    if not records:
        raise SystemExit("No broadness sweep records found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["best_eval", "final_eval"]:
        aggregated = _aggregate(records, metric)
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        _plot_metric(ax, aggregated, metric)
        fig.suptitle("Broadness sweeps on LunarLander-v3")
        path = output_dir / f"broadness_{metric}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
