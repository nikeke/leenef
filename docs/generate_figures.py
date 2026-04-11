#!/usr/bin/env python3
"""Generate key figures for the leenef technical report.

Usage:
    python docs/generate_figures.py          # generate all figures
    python docs/generate_figures.py --show   # also display interactively

Outputs PNGs to docs/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path(__file__).parent / "figures"

# ── color palette (colorblind-friendly) ─────────────────────────────────
C_NEF = "#0072B2"  # blue
C_MLP = "#D55E00"  # vermilion
C_ENS = "#009E73"  # green
C_GREY = "#999999"
C_LSTM = "#CC79A7"  # pink
DATASETS = ["MNIST", "Fashion-MNIST", "CIFAR-10"]


def _save(fig: plt.Figure, name: str, show: bool = False) -> None:
    path = OUTDIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  saved {path}")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Speed-accuracy Pareto: NEF vs MLP
# ═══════════════════════════════════════════════════════════════════════
def fig_speed_accuracy(show: bool = False) -> None:
    """Scatter comparing best NEF configs to MLP across all 3 datasets."""
    # (dataset_idx, label, accuracy, time_s, color, marker)
    points = [
        # MLP baselines
        (0, "MLP", 98.5, 83, C_MLP, "s"),
        (1, "MLP", 89.7, 83, C_MLP, "s"),
        (2, "MLP", 52.7, 83, C_MLP, "s"),
        # Best NEF
        (0, "NEF RF 12k", 98.5, 52, C_NEF, "o"),
        (1, "NEF RF 12k", 89.7, 59, C_NEF, "o"),
        (2, "NEF 10×3k", 58.4, 53, C_ENS, "^"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    fig.suptitle("NEF single-layer RF vs gradient-trained MLP", fontsize=12, y=1.02)

    # Per-dataset y-axis ranges that give breathing room without distortion
    ylims = [(96, 100), (88.5, 90.5), (50, 60)]

    for ax, ds_idx, ds_name, ylim in zip(axes, range(3), DATASETS, ylims):
        subset = [p for p in points if p[0] == ds_idx]
        for i, (_, label, acc, t, col, mk) in enumerate(subset):
            ax.scatter(t, acc, c=col, marker=mk, s=120, zorder=5, label=label)
            # Offset vertically when points are close
            va = "bottom" if i == 0 else "top"
            ax.annotate(
                f"  {acc:.1f}% / {t}s",
                (t, acc),
                fontsize=8,
                va=va,
                ha="left",
                xytext=(5, 8 if i == 0 else -8),
                textcoords="offset points",
            )
        ax.set_title(ds_name, fontsize=10)
        ax.set_xlabel("Training time (s)")
        ax.set_xlim(40, 95)
        ax.set_ylim(*ylim)
        if ds_idx == 0:
            ax.set_ylabel("Test accuracy (%)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "speed_accuracy", show)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — Alpha tuning curves
# ═══════════════════════════════════════════════════════════════════════
def fig_alpha_tuning(show: bool = False) -> None:
    """Effect of Tikhonov α on test accuracy for each dataset."""
    # MNIST: 12000n, RF patch=10
    mnist_alpha = [5e-4, 1e-3, 5e-3, 1e-2, 2e-2]
    mnist_acc = [98.5, 98.5, 98.4, 98.3, 98.1]

    # Fashion: 12000n, RF patch=5
    fash_alpha = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    fash_acc = [89.3, 89.6, 89.7, 89.7, 89.5, 89.4]

    # CIFAR-10: 10×3000 RF patch=5
    cifar_alpha = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    cifar_acc = [58.4, 58.3, 58.2, 57.6, 56.9]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    fig.suptitle("Regularization (α) tuning — the decisive lever", fontsize=12, y=1.02)

    configs = [
        ("MNIST (12k neurons, patch 10)", mnist_alpha, mnist_acc, 98.5),
        ("Fashion (12k neurons, patch 5)", fash_alpha, fash_acc, 89.7),
        ("CIFAR-10 (10×3k, patch 5)", cifar_alpha, cifar_acc, 52.7),
    ]

    for ax, (title, alphas, accs, mlp_baseline) in zip(axes, configs):
        ax.semilogx(alphas, accs, "o-", color=C_NEF, linewidth=2, markersize=6)
        ax.axhline(
            mlp_baseline,
            color=C_MLP,
            linestyle="--",
            linewidth=1.5,
            label=f"MLP baseline ({mlp_baseline}%)",
        )
        best_idx = int(np.argmax(accs))
        ax.scatter(
            [alphas[best_idx]],
            [accs[best_idx]],
            c="gold",
            edgecolors="black",
            s=120,
            zorder=10,
            label="Best α",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Tikhonov α")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "alpha_tuning", show)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — Neuron × Patch heatmap (MNIST)
# ═══════════════════════════════════════════════════════════════════════
def fig_neuron_patch_heatmap(show: bool = False) -> None:
    """Heatmap of accuracy for the MNIST neuron-patch sweep (α=0.01)."""
    neurons = [2000, 3000, 4000, 5000]
    patches = [3, 5, 7, 10, 12, 14, 16, 18]

    # NaN for cells not measured
    data = np.array(
        [
            [95.7, 96.5, 96.9, 97.1, np.nan, np.nan, np.nan, np.nan],
            [96.3, 97.0, 97.3, 97.4, 97.4, 97.2, 97.2, 97.0],
            [np.nan, np.nan, 97.6, 97.8, 97.7, 97.5, 97.3, np.nan],
            [96.8, 97.4, 97.7, 97.8, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(
        data, cmap="YlOrRd", aspect="auto", vmin=np.nanmin(data) - 0.2, vmax=np.nanmax(data) + 0.1
    )

    ax.set_xticks(range(len(patches)))
    ax.set_xticklabels(patches)
    ax.set_yticks(range(len(neurons)))
    ax.set_yticklabels(neurons)
    ax.set_xlabel("Patch size")
    ax.set_ylabel("Neurons")
    ax.set_title("MNIST accuracy (%) — neuron × patch sweep (α = 0.01)")

    for i in range(len(neurons)):
        for j in range(len(patches)):
            v = data[i, j]
            if not np.isnan(v):
                color = "white" if v > 97.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )

    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout()
    _save(fig, "neuron_patch_heatmap", show)


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — Single-layer scaling: accuracy & time vs neuron count
# ═══════════════════════════════════════════════════════════════════════
def fig_neuron_scaling(show: bool = False) -> None:
    """Accuracy saturation and quadratic time scaling with neuron count."""
    neurons = [500, 1000, 2000, 5000, 10_000, 20_000, 30_000]
    mnist = [92.1, 94.2, 95.7, 97.0, 97.8, 98.1, 98.3]
    fashion = [82.6, 84.8, 85.9, 87.2, 88.0, 89.1, 89.8]
    cifar = [43.7, 46.2, 48.3, 50.0, 50.8, 51.4, 51.8]
    times = [0.5, 0.8, 2, 10, 43, 140, 394]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    for accs, label, color in [
        (mnist, "MNIST", C_NEF),
        (fashion, "Fashion", C_ENS),
        (cifar, "CIFAR-10", C_MLP),
    ]:
        ax1.semilogx(neurons, accs, "o-", label=label, color=color, linewidth=2)

    ax1.set_xlabel("Neurons")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("Accuracy vs neuron count")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.loglog(neurons, times, "o-", color=C_NEF, linewidth=2)
    ns = np.array(neurons, dtype=float)
    ax2.loglog(ns, 2 * (ns / 2000) ** 2, "--", color=C_GREY, linewidth=1, label="O(n²) reference")
    ax2.set_xlabel("Neurons")
    ax2.set_ylabel("Training time (s)")
    ax2.set_title("Time scaling (CPU)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "neuron_scaling", show)


# ═══════════════════════════════════════════════════════════════════════
# Figure 5 — Method comparison overview
# ═══════════════════════════════════════════════════════════════════════
def fig_method_comparison(show: bool = False) -> None:
    """Bar chart: all training strategies + MLP baseline across 3 datasets."""
    methods = [
        "Linear",
        "NEF 1-layer",
        "NEF-greedy",
        "NEF-hybrid",
        "NEF-TP",
        "NEF-E2E",
        "Hybrid→E2E",
        "NEF RF+α*",
        "MLP (SGD)",
    ]
    mnist = [85.3, 95.7, 95.0, 98.6, 98.6, 98.5, 98.6, 98.5, 98.5]
    fashion = [81.0, 85.9, 85.4, 90.2, 90.1, 90.3, 90.6, 89.7, 89.7]
    cifar = [39.6, 47.8, 45.6, 52.7, 51.0, 58.5, 58.4, 58.4, 52.7]

    x = np.arange(len(methods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w, mnist, w, label="MNIST", color=C_NEF, alpha=0.85)
    ax.bar(x, fashion, w, label="Fashion", color=C_ENS, alpha=0.85)
    ax.bar(x + w, cifar, w, label="CIFAR-10", color=C_MLP, alpha=0.85)

    # Highlight the NEF RF+α and MLP columns
    for idx in [7, 8]:
        ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.08, color="gold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Training strategy comparison (* = best single-layer RF config per dataset)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(30, 102)

    fig.tight_layout()
    _save(fig, "method_comparison", show)


# ═══════════════════════════════════════════════════════════════════════
# Figure 6 — StreamNEF: accuracy vs neuron count at different windows
# ═══════════════════════════════════════════════════════════════════════
def fig_streaming_neuron_scaling(show: bool = False) -> None:
    """StreamNEF accuracy vs neuron count for different window sizes."""
    neurons = [2000, 4000, 6000, 8000, 10000]

    # Best α per configuration (from sweep table)
    w7 = [97.0, 97.8, np.nan, np.nan, np.nan]
    w10 = [97.2, 98.1, 98.4, 98.6, 98.6]
    w14_pts = ([4000], [98.1])  # sparse data for window=14

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        neurons[:2],
        [v for v in w7 if not np.isnan(v)],
        "s-",
        color=C_ENS,
        linewidth=2,
        label="Window = 7",
    )
    ax.plot(neurons, w10, "o-", color=C_NEF, linewidth=2, label="Window = 10")
    ax.plot(*w14_pts, "^", color=C_MLP, markersize=10, label="Window = 14")

    ax.axhline(98.3, color=C_LSTM, linestyle="--", linewidth=1.5, label="LSTM-128 (98.3%)")
    ax.axhline(97.2, color=C_GREY, linestyle=":", linewidth=1, label="StreamNEF 2000n (23s)")

    ax.set_xlabel("Neurons")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("StreamNEF sMNIST-row: accuracy vs neuron count")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(96.5, 99.0)

    fig.tight_layout()
    _save(fig, "streaming_neuron_scaling", show)


# ═══════════════════════════════════════════════════════════════════════
# Figure 7 — GPU speedup: Woodbury vs Accumulate
# ═══════════════════════════════════════════════════════════════════════
def fig_gpu_speedup(show: bool = False) -> None:
    """Bar chart: Woodbury vs Accumulate timing on T4 GPU."""
    configs = ["2000n w=10", "8000n w=10"]
    woodbury = [8.1, 93.2]
    accumulate = [1.1, 8.3]
    speedups = [w / a for w, a in zip(woodbury, accumulate)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(configs))
    w = 0.35
    ax1.bar(x - w / 2, woodbury, w, label="Woodbury (float64)", color=C_MLP, alpha=0.85)
    ax1.bar(x + w / 2, accumulate, w, label="Accumulate (float32)", color=C_NEF, alpha=0.85)

    for i, (wb, ac) in enumerate(zip(woodbury, accumulate)):
        ax1.text(i - w / 2, wb + 1, f"{wb}s", ha="center", fontsize=9)
        ax1.text(i + w / 2, ac + 1, f"{ac}s", ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylabel("Training time (s)")
    ax1.set_title("T4 GPU: Woodbury vs Accumulate")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, speedups, 0.5, color=C_ENS, alpha=0.85)
    for i, s in enumerate(speedups):
        ax2.text(i, s + 0.3, f"{s:.0f}×", ha="center", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.set_ylabel("Speedup factor")
    ax2.set_title("Accumulate speedup over Woodbury")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "gpu_speedup", show)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    args = parser.parse_args()

    matplotlib.use("Agg" if not args.show else "TkAgg")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Generating figures …")
    fig_speed_accuracy(args.show)
    fig_alpha_tuning(args.show)
    fig_neuron_patch_heatmap(args.show)
    fig_neuron_scaling(args.show)
    fig_method_comparison(args.show)
    fig_streaming_neuron_scaling(args.show)
    fig_gpu_speedup(args.show)
    print("Done.")


if __name__ == "__main__":
    main()
