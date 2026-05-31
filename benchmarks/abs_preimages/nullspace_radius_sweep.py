"""How far does the realizable preimage continuum extend? (part b)

Anchor at an exact one-hot preimage ``x_part`` of class ``c`` (a real digit nudged
onto the exact-one-hot solution within its sign cell ``s``).  Inside the cell the
decode is the affine map ``M x + const`` with ``M = D^T diag(s) W``; moving along
the (d_in - n_classes)-dimensional null space of ``M`` leaves the decode EXACTLY
one-hot.  We sweep the step radius ``r`` along

* null-space directions  (the realizable continuum), and
* arbitrary ambient directions  (control),

and measure, averaged over many random unit directions:

* max deviation of the decode from the exact one-hot,
* fraction of directions still predicting class ``c``,
* fraction of the ``n_neurons`` sign constraints that have flipped.

The null-space curve quantifies how far the continuum of confident preimages
reaches before the abs folds (sign flips at non-negligible neurons) finally tip
the decision.

Run::  python benchmarks/abs_preimages/nullspace_radius_sweep.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from common import (  # noqa: E402
    FIG_DIR,
    build_layer,
    image_stats,
    load_mnist_subset_with_metadata,
    sign_cell,
    write_result_json,
)

from leenef.preimages import nullspace_basis  # noqa: E402


def directions(basis_or_dim, n_dirs, gen):
    """Random unit directions; ``basis_or_dim`` is a (k, d_in) row basis or an int d_in."""
    if isinstance(basis_or_dim, int):
        v = torch.randn(n_dirs, basis_or_dim, generator=gen)
    else:
        coeff = torch.randn(n_dirs, basis_or_dim.shape[0], generator=gen)
        v = coeff @ basis_or_dim
    return v / v.norm(dim=1, keepdim=True)


def sweep(layer, W, b, D, x_part, s, dirs, radii, c):
    """Return (decode_dev, class_kept, sign_flipped) arrays over radii."""
    onehot = torch.eye(10)[c]
    dev, kept, flipped = [], [], []
    for r in radii:
        xs = x_part + r * dirs  # (n_dirs, d_in)
        a = (xs @ W.T + b).abs()
        logits = a @ D
        pred = logits.argmax(1)
        dev.append((logits - onehot).abs().max(dim=1).values.mean().item())
        kept.append((pred == c).float().mean().item())
        flips = (torch.sign(xs @ W.T + b) != s).float().mean(dim=1)
        flipped.append(flips.mean().item())
    return torch.tensor(dev), torch.tensor(kept), torch.tensor(flipped)


def main():
    n_train = 10000
    subset_seed = 0
    n_neurons = 2000
    xtr, ytr, _, _, data_metadata = load_mnist_subset_with_metadata(
        n_train=n_train, seed=subset_seed
    )
    layer, W, b = build_layer(n_neurons, xtr, ytr)
    D = layer.decoders
    c = 3
    x0 = xtr[ytr == c][0]
    s = sign_cell(W, b, x0)
    M = (D.T * s) @ W
    rhs = torch.eye(10)[c] - (D.T * s) @ b
    x_part = x0 + torch.linalg.pinv(M) @ (rhs - M @ x0)

    null = nullspace_basis(M)  # (d_in - 10, d_in)

    gen = torch.Generator().manual_seed(0)
    n_dirs = 40
    null_dirs = directions(null, n_dirs, gen)
    amb_dirs = directions(W.shape[1], n_dirs, gen)
    radii = torch.linspace(0.0, 12.0, 49)

    dev_n, kept_n, flip_n = sweep(layer, W, b, D, x_part, s, null_dirs, radii, c)
    dev_a, kept_a, flip_a = sweep(layer, W, b, D, x_part, s, amb_dirs, radii, c)

    def flip_radius(kept):
        below = (kept < 0.5).nonzero()
        return float(radii[below[0, 0]]) if len(below) else float("inf")

    print(f"null-space dim = {null.shape[0]}")
    print(
        f"class-flip radius (kept<0.5): null={flip_radius(kept_n):.2f}  ambient={flip_radius(kept_a):.2f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(radii, dev_n, label="null-space", lw=2)
    axes[0].plot(radii, dev_a, label="ambient", lw=2, ls="--")
    axes[0].set(
        xlabel="step radius |Δx|", ylabel="max |decode - one-hot|", title="decode deviation"
    )
    axes[0].legend()
    axes[1].plot(radii, kept_n, label="null-space", lw=2)
    axes[1].plot(radii, kept_a, label="ambient", lw=2, ls="--")
    axes[1].axhline(0.5, color="gray", lw=0.8, ls=":")
    axes[1].set(
        xlabel="step radius |Δx|", ylabel=f"fraction still class {c}", title="decision preserved"
    )
    axes[1].legend()
    axes[2].plot(radii, flip_n, label="null-space", lw=2)
    axes[2].plot(radii, flip_a, label="ambient", lw=2, ls="--")
    axes[2].set(
        xlabel="step radius |Δx|", ylabel="fraction of signs flipped", title="abs folds crossed"
    )
    axes[2].legend()
    fig.suptitle(
        f"Realizable preimage continuum extent (class {c}, n_neurons=2000): null-space "
        f"moves preserve the decision far longer than ambient moves",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nullspace_radius_sweep.png", dpi=110)
    print("saved nullspace_radius_sweep.png")
    write_result_json(
        "nullspace_radius_sweep.json",
        script="benchmarks/abs_preimages/nullspace_radius_sweep.py",
        config={
            "n_train": n_train,
            "subset_seed": subset_seed,
            "n_neurons": n_neurons,
            "target_class": c,
            "n_dirs": n_dirs,
            "direction_seed": 0,
            "radii": radii,
        },
        data_metadata=data_metadata,
        diagnostics={
            "rank_W": int(torch.linalg.matrix_rank(W).item()),
            "rank_M": int(torch.linalg.matrix_rank(M).item()),
            "nullspace_dim": int(null.shape[0]),
            "linear_residual_anchor": (M @ x_part - rhs).abs().max().item(),
            "class_flip_radius": {
                "null_space": flip_radius(kept_n),
                "ambient": flip_radius(kept_a),
            },
            "anchor_image_stats": image_stats(x_part.unsqueeze(0)),
        },
        plotted_arrays={
            "radii": radii,
            "null_space": {
                "decode_dev": dev_n,
                "class_kept": kept_n,
                "sign_flipped": flip_n,
            },
            "ambient": {
                "decode_dev": dev_a,
                "class_kept": kept_a,
                "sign_flipped": flip_a,
            },
        },
        artifact_paths=[FIG_DIR / "nullspace_radius_sweep.png"],
    )


if __name__ == "__main__":
    main()
