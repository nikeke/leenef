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
    "MNIST": [92.0, 94.1, 95.5, 96.8, 97.5, 98.0, 98.1],
    "Fashion-MNIST": [82.8, 84.4, 86.1, 87.3, 88.5, 89.3, 89.6],
    "CIFAR-10": [43.9, 45.3, 48.5, 50.4, 50.9, 51.4, 51.5],
}

# Bias effect (2000 neurons, abs activation): encoder × random/data bias
ENCODERS = ["hypersphere", "gaussian", "sparse"]
BIAS_EFFECT = {
    "MNIST": {"random": [92.8, 95.8, 95.8], "data": [95.5, 95.5, 95.7]},
    "Fashion-MNIST": {"random": [83.8, 86.1, 86.2], "data": [86.1, 86.1, 86.2]},
    "CIFAR-10": {"random": [45.2, 47.4, 47.2], "data": [48.5, 48.5, 48.7]},
}

# Multi-layer: strategy comparison
STRATEGIES = ["Linear", "NEFLayer", "Greedy", "Hybrid", "Hybrid→E2E", "E2E", "MLP"]
MULTI = {
    "MNIST": [85.3, 95.5, 95.0, 98.6, 98.6, 98.5, 98.4],
    "Fashion-MNIST": [81.0, 86.1, 85.7, 90.3, 90.9, 90.6, 89.6],
    "CIFAR-10": [39.6, 48.5, 45.5, 52.3, 58.4, 58.4, 53.4],
}
MULTI_TIME = {
    "MNIST": [2, 2, 3, 355, 501, 259, 87],
    "Fashion-MNIST": [2, 2, 3, 340, 457, 263, 89],
    "CIFAR-10": [15, 3, 3, 375, 511, 500, 150],
}

# Multi-layer: activation effect (hybrid, hypersphere + data biases)
ACTIVATIONS_ML = ["abs", "relu", "lif_rate", "softplus"]
HYBRID_ACT = {
    "MNIST": [97.2, 96.6, 92.3, 90.8],
    "Fashion-MNIST": [87.9, 87.2, 83.7, 82.4],
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
        ax.scatter(MULTI_TIME[ds], MULTI[ds], s=80, zorder=3)
        for j, strat in enumerate(STRATEGIES):
            ax.annotate(
                strat,
                (MULTI_TIME[ds][j], MULTI[ds][j]),
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
