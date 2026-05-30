"""Shared helpers for the abs-activation preimage explorations.

The NEF encode path is affine-then-abs::

    z = W x + b,   a = |z|,   W = diag(gain) @ encoders   (n_neurons x d_in)

so a single NEF layer is piecewise-linear: inside a sign cell
``s = sign(W x + b)`` the map ``x -> a`` is the linear map ``x -> s ⊙ (W x + b)``.
These helpers build a fitted layer on MNIST and expose ``W`` / ``b`` directly so
the exploration scripts can reason about the geometry analytically.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "benchmarks" / "sl_clean"))
sys.path.insert(0, str(_REPO / "src"))

from datasets import load_mnist  # noqa: E402

from leenef.layers import NEFLayer  # noqa: E402

FIG_DIR = _HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_mnist_subset(
    n_train: int = 10000, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load MNIST and subsample the training split for speed."""
    bundle = load_mnist()
    xtr, ytr = bundle.x_train, bundle.y_train
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(xtr), generator=gen)[:n_train]
    return xtr[idx], ytr[idx], bundle.x_test, bundle.y_test


def build_layer(
    n_neurons: int, xtr: Tensor, ytr: Tensor, *, seed: int = 0
) -> tuple[NEFLayer, Tensor, Tensor]:
    """Fit a default-config NEF layer (abs, hypersphere, data-driven bias).

    Returns the layer plus the affine encoder pieces ``W = diag(gain) @ E``
    and ``b = bias`` so callers can analyse the piecewise-linear geometry.
    """
    rng = torch.Generator().manual_seed(seed)
    layer = NEFLayer(
        d_in=xtr.shape[1],
        n_neurons=n_neurons,
        d_out=10,
        activation="abs",
        gain=(0.5, 2.0),
        rng=rng,
        centers=xtr,
    )
    onehot = torch.zeros(len(ytr), 10)
    onehot[torch.arange(len(ytr)), ytr] = 1.0
    layer.fit(xtr, onehot)
    W = layer.gain[:, None] * layer.encoders  # (n_neurons, d_in)
    return layer, W, layer.bias


def sign_cell(W: Tensor, b: Tensor, x: Tensor) -> Tensor:
    """Sign pattern s = sign(W x + b) with zeros mapped to +1."""
    s = torch.sign(W @ x + b)
    s[s == 0] = 1.0
    return s


def decode(layer: NEFLayer, x: Tensor) -> tuple[Tensor, int]:
    """Softmax probabilities and argmax class for a single input."""
    logits = layer.encode(x.reshape(1, -1)).squeeze(0) @ layer.decoders
    return torch.softmax(logits, 0), int(logits.argmax())


def show(ax, img: Tensor, title: str, cmap: str = "gray") -> None:
    ax.imshow(img.reshape(28, 28).clamp(0, 1), cmap=cmap)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def quiet_proxychains() -> None:
    """No-op kept for symmetry; proxychains banners come from the shell."""
    os.environ.setdefault("MPLBACKEND", "Agg")
