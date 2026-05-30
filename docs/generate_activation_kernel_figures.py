"""Generate figures for docs/activation_kernel_theory.md.

Reads the local worktree's `results/sl_clean/*.json` artifacts and writes PNGs
to `docs/figures/`, matching the paper-series Markdown convention:
`![caption](figures/name.png)`.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sl_clean"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _mean_se(xs: list[float]) -> tuple[float, float]:
    if len(xs) <= 1:
        return mean(xs), 0.0
    return mean(xs), stdev(xs) / math.sqrt(len(xs))


def _save(fig: plt.Figure, name: str) -> Path:
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_activation_hermite_spectrum() -> Path:
    data = _load_json(RESULTS / "hermite_mode_recovery.json")
    coefficients = data["coefficients"]
    r2 = data["r2"]
    activations = [
        "linear",
        "abs",
        "relu",
        "cos",
        "cos_sin",
        "square_wave",
        "floor_wave",
        "box",
        "gelu",
        "silu",
    ]
    activations = [a for a in activations if a in coefficients]
    modes = [1, 2, 3]
    width = 0.24
    x = list(range(len(activations)))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=True)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for j, mode in enumerate(modes):
        axes[0].bar(
            [i + (j - 1) * width for i in x],
            [coefficients[a][mode] ** 2 for a in activations],
            width=width,
            color=colors[j],
            label=f"$a_{mode}^2$",
        )
        axes[1].bar(
            [i + (j - 1) * width for i in x],
            [max(0.0, r2[a][f"k{mode}"]["mean"]) for a in activations],
            width=width,
            color=colors[j],
            label=f"$R^2(k={mode})$",
        )

    axes[0].set_ylabel("Hermite energy")
    axes[1].set_ylabel("held-out $R^2$ (clipped at 0)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(activations, rotation=35, ha="right")
    axes[0].legend(ncol=3, loc="upper right")
    axes[1].legend(ncol=3, loc="upper right")
    axes[0].set_title("Activation spectrum predicts which Hermite modes the solve can recover")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    return _save(fig, "activation_hermite_spectrum")


def figure_activation_kink_angle() -> Path:
    datasets = ["mnist", "fashion", "cifar10"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for ds in datasets:
        rows = _load_json(RESULTS / f"kink_angle_{ds}_panel.json")
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in rows.values():
            if "theta_over_pi" in row and "test_accuracy" in row:
                grouped[float(row["theta_over_pi"])].append(float(row["test_accuracy"]))
        xs = sorted(grouped)
        ys = [_mean_se(grouped[x])[0] for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=ds)
    ax.axvline(0.50, color="0.5", linestyle="--", linewidth=1)
    ax.axvline(0.75, color="0.5", linestyle="--", linewidth=1)
    ax.axvline(1.00, color="0.1", linestyle=":", linewidth=1.5)
    ax.text(0.50, ax.get_ylim()[0], "abs", va="bottom", ha="center", fontsize=8)
    ax.text(0.75, ax.get_ylim()[0], "relu", va="bottom", ha="center", fontsize=8)
    ax.text(1.00, ax.get_ylim()[0], "linear", va="bottom", ha="center", fontsize=8)
    ax.set_xlabel(r"origin angle $\theta/\pi$")
    ax.set_ylabel("test accuracy")
    ax.set_title("Kink angle: flat equivalence class with a sharp linear-collapse dip")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(fig, "activation_kink_angle")


def figure_activation_altact_panel() -> Path:
    summary = _load_json(RESULTS / "c2_altact_summary.json")
    cells = [
        ("abs", "abs+g0.5_2.0"),
        ("gauss", "gauss+g0.5_2.0"),
        ("cos_sin", "cos_sin+g0.5_2.0"),
        ("gelu", "gelu+g0.5_2.0"),
        ("square_wave", "square_wave+g0.5_2.0"),
        ("square_wave wide", "square_wave+g0.1_5.0"),
        ("box wide", "box+g0.1_5.0"),
        ("floor_wave", "floor_wave+g0.5_2.0"),
        ("floor_wave wide", "floor_wave+g0.1_5.0"),
    ]
    datasets = ["mnist", "fashion", "cifar10"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    for ax, ds in zip(axes, datasets, strict=True):
        ds_summary = summary[ds]
        available = [(label, cell) for label, cell in cells if cell in ds_summary]
        labels = [label for label, _ in available]
        vals = [ds_summary[cell]["mean"] for _, cell in available]
        errs = [ds_summary[cell]["se"] for _, cell in available]
        ax.bar(range(len(vals)), vals, yerr=errs, capsize=2, color="#4c78a8")
        ax.set_title(ds)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(max(0.0, min(vals) - 0.03), min(1.0, max(vals) + 0.015))
    axes[0].set_ylabel("test accuracy")
    fig.suptitle("Activation panel: gain-robust, RBF, and digital cells")
    return _save(fig, "activation_altact_panel")


def main() -> None:
    for path in [
        figure_activation_hermite_spectrum(),
        figure_activation_kink_angle(),
        figure_activation_altact_panel(),
    ]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
