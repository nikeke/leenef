"""Generate benchmark visualisation plots."""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data (from benchmark runs — abs + hypersphere + data-driven biases)
# ---------------------------------------------------------------------------

NEURONS = [500, 1000, 2000, 5000, 10000, 20000, 30000]

# Single-layer: neuron scaling
SCALE = {
    "MNIST": [92.1, 94.3, 95.5, 96.9, 97.4, 97.9, 98.3],
    "Fashion-MNIST": [82.6, 84.7, 85.7, 87.1, 88.4, 89.3, 89.8],
    "CIFAR-10": [43.7, 45.9, 47.8, 50.4, 51.0, 51.5, 51.8],
}

# Bias effect (2000 neurons, abs activation): encoder × random/data bias
ENCODERS = ["hypersphere", "gaussian", "sparse"]
BIAS_EFFECT = {
    "MNIST": {"random": [93.4, 96.0, 95.6], "data": [95.6, 95.7, 95.6]},
    "Fashion-MNIST": {"random": [84.1, 86.0, 86.0], "data": [85.9, 86.0, 85.6]},
    "CIFAR-10": {"random": [45.9, 47.3, 47.5], "data": [48.3, 47.5, 48.2]},
}

# Multi-layer: strategy comparison
STRATEGIES = [
    "Linear",
    "NEFLayer",
    "Greedy",
    "Hybrid",
    "Target Prop",
    "TP→E2E",
    "Hybrid→E2E",
    "E2E",
    "MLP",
]
MULTI = {
    "MNIST": [85.3, 95.6, 95.1, 98.5, 98.6, 98.6, 98.6, 98.4, 98.1],
    "Fashion-MNIST": [81.0, 85.5, 85.5, 90.0, 90.1, 90.6, 91.0, 90.3, 90.2],
    "CIFAR-10": [39.6, 47.8, 45.8, 51.7, 51.0, 58.5, 58.1, 57.8, 54.6],
}
MULTI_TIME = {
    "MNIST": [2, 2, 3, 315, 371, 470, 412, 240, 84],
    "Fashion-MNIST": [2, 2, 3, 316, 376, 463, 410, 239, 82],
    "CIFAR-10": [14, 3, 3, 343, 373, 497, 475, 319, 142],
}

# Multi-layer: activation effect (hybrid, hypersphere + data biases)
ACTIVATIONS_ML = ["relu", "abs", "lif_rate", "softplus"]
HYBRID_ACT = {
    "MNIST": [97.8, 97.5, 95.2, 94.0],
    "Fashion-MNIST": [88.7, 88.0, 85.1, 84.1],
}


def plot_neuron_scaling():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds, accs in SCALE.items():
        ax.plot(NEURONS, accs, "o-", label=ds, linewidth=2, markersize=6)
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Single-layer NEF: scaling with neuron count\n(abs + hypersphere + data biases)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(NEURONS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(OUT / "neuron_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'neuron_scaling.png'}")


def plot_bias_effect():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(ENCODERS))
    width = 0.35
    for ax, (ds, data) in zip(axes, BIAS_EFFECT.items()):
        ax.bar(x - width / 2, data["random"], width, label="Random bias", color="#7faedb")
        ax.bar(x + width / 2, data["data"], width, label="Data bias", color="#2d6da3")
        ax.set_xticks(x)
        ax.set_xticklabels(ENCODERS)
        ax.set_title(ds)
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        lo = min(min(data["random"]), min(data["data"]))
        ax.set_ylim(lo - 2, max(max(data["random"]), max(data["data"])) + 1.5)
        for i, (r, d) in enumerate(zip(data["random"], data["data"])):
            ax.text(i - width / 2, r + 0.2, f"{r:.1f}", ha="center", fontsize=8)
            ax.text(i + width / 2, d + 0.2, f"{d:.1f}", ha="center", fontsize=8)
    fig.suptitle(
        "Effect of data-driven biases (2000 neurons, abs activation)", fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "bias_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / 'bias_effect.png'}")


def plot_strategy_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = np.arange(len(STRATEGIES))
    width = 0.25
    for i, (ds, accs) in enumerate(MULTI.items()):
        ax.bar(x + i * width, accs, width, label=ds)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Strategy comparison\n(abs + hypersphere + data biases)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(STRATEGIES, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for ds in ["MNIST", "Fashion-MNIST"]:
        points = [
            (fit_time, acc, strat)
            for fit_time, acc, strat in zip(MULTI_TIME[ds], MULTI[ds], STRATEGIES, strict=True)
            if fit_time is not None
        ]
        ax.scatter(
            [fit_time for fit_time, _, _ in points], [acc for _, acc, _ in points], s=80, zorder=3
        )
        for fit_time, acc, strat in points:
            ax.annotate(
                strat,
                (fit_time, acc),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xlabel("Fit time (s)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Speed–accuracy trade-off")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "strategy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'strategy_comparison.png'}")


def plot_activation_multilayer():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(ACTIVATIONS_ML))
    width = 0.35
    for i, (ds, accs) in enumerate(HYBRID_ACT.items()):
        ax.bar(x + i * width, accs, width, label=ds)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Hybrid multi-layer: activation effect\n(hypersphere + data biases)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ACTIVATIONS_ML)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(78, 99)
    fig.tight_layout()
    fig.savefig(OUT / "activation_multilayer.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'activation_multilayer.png'}")


if __name__ == "__main__":
    print("Generating plots...")
    plot_neuron_scaling()
    plot_bias_effect()
    plot_strategy_comparison()
    plot_activation_multilayer()
    print("Done.")
