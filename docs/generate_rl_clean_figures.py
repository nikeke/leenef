#!/usr/bin/env python3
"""Generate clean RL sweep figures under docs/figures."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIR = Path("docs") / "figures"
P3B_RESULT_PATH = Path("results") / "rl_clean" / "p3b_h1_grid_lunarlander_v3.json"
P3B_FIGURE_PATH = FIGURE_DIR / "rl_clean_p3b_h1_grid.png"


def _mean_se(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance) / math.sqrt(len(values))


def _parse_grid_name(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"n(\d+)_k(\d+)", name)
    if not match:
        return (10**9, 10**9)
    return int(match.group(2)), int(match.group(1))


def plot_p3b_h1_grid() -> None:
    """Plot P3b config-grid robust-eval means without current-code filtering."""
    if not P3B_RESULT_PATH.exists():
        return
    rows = list(json.loads(P3B_RESULT_PATH.read_text()).values())
    by_name: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict) or "error" in row:
            continue
        if row.get("result_namespace") != "rl_clean" or row.get("runner_version") != "rl-clean-v2":
            continue
        if (row.get("run_context") or {}).get("phase") != "p3b_h1_grid":
            continue
        robust = row.get("robust_eval")
        if robust is None:
            continue
        name = str(row.get("config_name") or row.get("value"))
        by_name.setdefault(name, []).append(float(robust))

    if not by_name:
        return

    names = sorted(by_name, key=_parse_grid_name)
    means: list[float] = []
    ses: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    color_by_step = {50: "#4a6aa8", 100: "#b34e1e", 200: "#538a4e"}
    for name in names:
        mean, se = _mean_se(by_name[name])
        n_neurons, n_step = _parse_grid_name(name)
        means.append(mean)
        ses.append(se)
        labels.append(f"{n_step} step\n{k_label(n_neurons)}")
        colors.append(color_by_step.get(n_step, "#777777"))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(len(names)))
    ax.bar(x, means, yerr=ses, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("robust_eval mean +/- SE")
    ax.set_title("Clean P3b H1 grid: horizon x capacity (LunarLander-v3)")
    ax.grid(axis="y", alpha=0.25)

    for i, (mean, se, values) in enumerate(zip(means, ses, (by_name[name] for name in names))):
        ax.text(
            i, mean + se + 4, f"{mean:.1f}\nn={len(values)}", ha="center", va="bottom", fontsize=8
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_by_step[step], ec="black", linewidth=0.5)
        for step in sorted(color_by_step)
    ]
    ax.legend(handles, [f"n_step={step}" for step in sorted(color_by_step)], loc="upper right")
    fig.tight_layout()
    tmp = P3B_FIGURE_PATH.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=130)
    plt.close(fig)
    tmp.replace(P3B_FIGURE_PATH)


def k_label(n_neurons: int) -> str:
    return f"{n_neurons} neurons"


def main() -> None:
    from benchmarks.plot_rl_sweeps import main as plot_main

    plot_main()
    plot_p3b_h1_grid()


if __name__ == "__main__":
    main()
