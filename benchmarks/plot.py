"""Generate benchmark visualisation plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data (from benchmark runs)
# ---------------------------------------------------------------------------

NEURONS = [500, 1000, 2000, 5000]

# Single-layer: neuron scaling (ReLU + hypersphere)
SCALE = {
    "MNIST":         [88.4, 89.9, 92.1, 94.2],
    "Fashion-MNIST": [80.6, 82.0, 83.3, 84.5],
    "CIFAR-10":      [40.8, 42.8, 44.8, 46.7],
}

# Single-layer: encoder × activation (2000 neurons, test accuracy)
ENCODERS = ["hypersphere", "gaussian", "sparse"]
ACTIVATIONS_SL = ["relu", "softplus", "abs", "lif_rate"]
MNIST_SL = {
    "relu":     [91.8, 95.6, 95.6],
    "softplus": [87.7, 95.9, 95.7],
    "abs":      [92.8, 96.0, 95.8],
    "lif_rate": [91.3, 95.5, 95.3],
}
FASHION_SL = {
    "relu":     [83.2, 85.8, 85.6],
    "softplus": [81.4, 85.9, 85.6],
    "abs":      [84.0, 85.9, 86.0],
    "lif_rate": [83.0, 85.8, 85.5],
}

# Multi-layer: strategy comparison (gaussian, ReLU)
STRATEGIES = ["Linear", "NEFLayer", "Greedy", "Hybrid", "E2E", "MLP"]
MULTI = {
    "MNIST":         [85.4, 95.7, 95.4, 96.0, 98.0, 98.4],
    "Fashion-MNIST": [80.6, 85.9, 85.8, 86.0, 88.3, 89.6],
    "CIFAR-10":      [20.2, 46.6, 46.1, 48.0, 43.7, 53.4],
}
MULTI_TIME = {
    "MNIST":         [1, 2, 3, 64, 247, 87],
    "Fashion-MNIST": [2, 2, 3, 66, 250, 89],
    "CIFAR-10":      [16, 3, 3, 72, 331, 150],
}

# Multi-layer: activation effect (hybrid, gaussian)
ACTIVATIONS_ML = ["relu", "softplus", "abs", "lif_rate"]
HYBRID_ACT = {
    "MNIST":         [96.0, 96.2, 95.8, 93.6],
    "Fashion-MNIST": [86.0, 86.5, 86.2, 83.7],
}


def plot_neuron_scaling():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds, accs in SCALE.items():
        ax.plot(NEURONS, accs, "o-", label=ds, linewidth=2, markersize=6)
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Single-layer NEF: scaling with neuron count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(NEURONS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(OUT / "neuron_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'neuron_scaling.png'}")


def plot_encoder_activation_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, data) in zip(axes, [("MNIST", MNIST_SL),
                                         ("Fashion-MNIST", FASHION_SL)]):
        matrix = np.array([data[a] for a in ACTIVATIONS_SL])
        im = ax.imshow(matrix, cmap="YlGn", aspect="auto",
                        vmin=matrix.min() - 1, vmax=matrix.max() + 0.5)
        ax.set_xticks(range(len(ENCODERS)))
        ax.set_xticklabels(ENCODERS)
        ax.set_yticks(range(len(ACTIVATIONS_SL)))
        ax.set_yticklabels(ACTIVATIONS_SL)
        ax.set_title(title)
        for i in range(len(ACTIVATIONS_SL)):
            for j in range(len(ENCODERS)):
                ax.text(j, i, f"{matrix[i, j]:.1f}%",
                        ha="center", va="center", fontsize=10,
                        color="white" if matrix[i, j] > 93 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Single-layer: encoder × activation (2000 neurons, test %)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "encoder_activation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / 'encoder_activation.png'}")


def plot_strategy_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax = axes[0]
    x = np.arange(len(STRATEGIES))
    width = 0.25
    for i, (ds, accs) in enumerate(MULTI.items()):
        ax.bar(x + i * width, accs, width, label=ds)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Multi-layer: strategy comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(STRATEGIES, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Time vs accuracy (MNIST)
    ax = axes[1]
    for ds in ["MNIST", "Fashion-MNIST"]:
        ax.scatter(MULTI_TIME[ds], MULTI[ds], s=80, zorder=3)
        for j, strat in enumerate(STRATEGIES):
            ax.annotate(strat, (MULTI_TIME[ds][j], MULTI[ds][j]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8)
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
    ax.set_title("Multi-layer hybrid: activation effect (gaussian encoders)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ACTIVATIONS_ML)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(80, 98)
    fig.tight_layout()
    fig.savefig(OUT / "activation_multilayer.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'activation_multilayer.png'}")


if __name__ == "__main__":
    print("Generating plots...")
    plot_neuron_scaling()
    plot_encoder_activation_heatmap()
    plot_strategy_comparison()
    plot_activation_multilayer()
    print("Done.")
