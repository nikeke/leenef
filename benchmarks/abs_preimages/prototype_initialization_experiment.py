"""Single-run check: use linearized one-hot candidates as NEF centers.

The earlier preimage analysis produced one class prototype per digit by
anchoring the local one-hot solve at each class mean.  This script asks whether
those generated candidates are useful as `centers=` for a new layer.

To isolate the center effect, all compared layers share the same encoders and
gains; only the per-neuron center locations used to compute the data-driven
biases differ:

* `train_sample`: ordinary default-style centers sampled from training data.
* `class_mean`: balanced repeats of the ten class means.
* `onehot_preimage`: balanced repeats of the ten generated linearized one-hot
  candidates.

This is deliberately a single exploratory run, not a sweep.  CPU usage is kept
below 25% on the 12-core workstation by pinning PyTorch and BLAS libraries to
two threads.

Run:

    source ../leenef/venv/bin/activate
    LEENEF_DATA_ROOT=../leenef/data \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python benchmarks/abs_preimages/prototype_initialization_experiment.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from common import FIG_DIR, build_layer, load_mnist_subset  # noqa: E402

from leenef.layers import NEFLayer  # noqa: E402
from leenef.preimages import onehot_preimages_from_anchors  # noqa: E402

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "prototype_initialization_result.json"


def onehot_targets(y: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    targets = torch.zeros(len(y), n_classes)
    targets[torch.arange(len(y)), y] = 1.0
    return targets


def class_means(xtr: torch.Tensor, ytr: torch.Tensor) -> torch.Tensor:
    return torch.stack([xtr[ytr == c].mean(0) for c in range(10)])


def balanced_repeat(centers: torch.Tensor, n_neurons: int, *, seed: int) -> torch.Tensor:
    """Repeat ten class-level centers nearly evenly across neurons and shuffle."""
    reps = (n_neurons + len(centers) - 1) // len(centers)
    selected = centers.repeat((reps, 1))[:n_neurons].clone()
    gen = torch.Generator().manual_seed(seed)
    return selected[torch.randperm(n_neurons, generator=gen)]


def make_layer_with_fixed_centers(
    selected_centers: torch.Tensor,
    *,
    seed: int,
    n_neurons: int,
    d_in: int,
) -> NEFLayer:
    """Build a layer with deterministic encoders/gains and explicit centers.

    `NEFLayer(centers=...)` samples centers internally, which would make it hard
    to keep the compared layers perfectly matched.  Instead, instantiate the
    shared encoders/gains once via the normal constructor and overwrite only the
    data-driven bias implied by the selected centers.
    """
    rng = torch.Generator().manual_seed(seed)
    layer = NEFLayer(
        d_in=d_in,
        n_neurons=n_neurons,
        d_out=10,
        activation="abs",
        gain=(0.5, 2.0),
        rng=rng,
        centers=None,
    )
    bias = -layer.gain * (selected_centers.float() * layer.encoders).sum(dim=1)
    layer.bias.data.copy_(bias)
    return layer


@torch.no_grad()
def evaluate(layer: NEFLayer, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    logits = layer(x)
    pred = logits.argmax(1)
    target = logits[torch.arange(len(y)), y]
    other_logits = logits.clone()
    other_logits[torch.arange(len(y)), y] = float("-inf")
    margin = target - other_logits.max(1).values
    probs = torch.softmax(logits, dim=1)
    return {
        "accuracy": (pred == y).float().mean().item(),
        "mean_margin": margin.mean().item(),
        "mean_true_prob": probs[torch.arange(len(y)), y].mean().item(),
    }


def fit_and_eval(
    label: str,
    selected_centers: torch.Tensor,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    *,
    seed: int,
) -> dict[str, object]:
    layer = make_layer_with_fixed_centers(
        selected_centers, seed=seed, n_neurons=selected_centers.shape[0], d_in=xtr.shape[1]
    )
    layer.fit(xtr, onehot_targets(ytr))
    return {
        "label": label,
        "train": evaluate(layer, xtr, ytr),
        "test": evaluate(layer, xte, yte),
        "center_norm_mean": selected_centers.norm(dim=1).mean().item(),
        "center_pixel_min": selected_centers.min().item(),
        "center_pixel_max": selected_centers.max().item(),
    }


def plot_results(results: list[dict[str, object]]) -> None:
    labels = [str(r["label"]) for r in results]
    acc = [float(r["test"]["accuracy"]) for r in results]
    margins = [float(r["test"]["mean_margin"]) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(labels, acc)
    axes[0].set_ylim(min(acc) - 0.01, max(acc) + 0.01)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Accuracy")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[1].bar(labels, margins)
    axes[1].axhline(0.0, color="gray", lw=0.8, ls=":")
    axes[1].set_ylabel("test true-vs-best-other margin")
    axes[1].set_title("Margin")
    axes[1].tick_params(axis="x", labelrotation=20)
    fig.suptitle("Prototype-center initialization single-run check")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prototype_initialization_result.png", dpi=120)


def main() -> None:
    n_neurons = 1000
    xtr, ytr, xte, yte = load_mnist_subset(n_train=10000, seed=0)
    means = class_means(xtr, ytr)

    pilot, W, b = build_layer(n_neurons, xtr, ytr, seed=0)
    prototypes, prototype_stats = onehot_preimages_from_anchors(W, b, pilot.decoders, means)
    prototype_stats.update(
        {
            "prototype_min": prototypes.min().item(),
            "prototype_max": prototypes.max().item(),
            "prototype_mean": prototypes.mean().item(),
        }
    )

    center_seed = 123
    train_gen = torch.Generator().manual_seed(center_seed)
    train_centers = xtr[torch.randint(len(xtr), (n_neurons,), generator=train_gen)]
    mean_centers = balanced_repeat(means, n_neurons, seed=center_seed)
    prototype_centers = balanced_repeat(prototypes, n_neurons, seed=center_seed)

    final_seed = 1
    results = [
        fit_and_eval("train_sample", train_centers, xtr, ytr, xte, yte, seed=final_seed),
        fit_and_eval("class_mean", mean_centers, xtr, ytr, xte, yte, seed=final_seed),
        fit_and_eval("onehot_preimage", prototype_centers, xtr, ytr, xte, yte, seed=final_seed),
    ]

    payload = {
        "n_train": int(len(xtr)),
        "n_test": int(len(xte)),
        "n_neurons": n_neurons,
        "threads": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "prototype_stats": prototype_stats,
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    plot_results(results)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
