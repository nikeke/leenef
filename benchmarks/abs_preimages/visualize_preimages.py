"""Visualize abs-activation preimages in both regimes.

Produces three figures in ``docs/figures/abs_preimages/``:

* ``perfect_twins.png``       — underdetermined (n<d_in): distinct inputs with
                                identical activity and decode.
* ``overdet_preimages.png``   — realistic default (n>d_in): per-class
                                linearized candidate for the one-hot target
                                via the piecewise-linear cell, anchored at
                                the class mean.
* ``nullspace_continuum.png`` — the (d_in - n_classes)-dim null-space family of
                                local candidates for one class.

Run::  python benchmarks/abs_preimages/visualize_preimages.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from common import FIG_DIR, build_layer, decode, load_mnist_subset, show, sign_cell  # noqa: E402

from leenef.preimages import nullspace_basis  # noqa: E402


def perfect_twins(xtr, ytr, *, n_neurons=400, c=3, n_twins=5, seed=0):
    layer, W, b = build_layer(n_neurons, xtr, ytr, seed=seed)
    Wpinv = torch.linalg.pinv(W)
    x0 = xtr[ytr == c][0]
    a_star = (W @ x0 + b).abs()

    rng = torch.Generator().manual_seed(123)
    fig, axes = plt.subplots(1, n_twins + 1, figsize=(2 * (n_twins + 1), 2.4))
    p0, c0 = decode(layer, x0)
    show(axes[0], x0, f"real digit\npred {c0} p={p0[c0]:.2f}")
    acts = [a_star]
    for i in range(n_twins):
        s = torch.randint(0, 2, (n_neurons,), generator=rng).float() * 2 - 1
        x = Wpinv @ (s * a_star - b)
        acts.append(layer.encode(x.reshape(1, -1)).squeeze(0))
        p, cp = decode(layer, x)
        show(axes[i + 1], x, f"twin {i + 1}\npred {cp} p={p[cp]:.2f}")
    max_dev = (torch.stack(acts) - a_star).abs().max().item()
    fig.suptitle(
        f"A) Perfect twins (n_neurons={n_neurons} < d_in=784): identical activity "
        f"(max dev {max_dev:.1e}) and decode, distinct inputs",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "perfect_twins.png", dpi=110)
    print(f"[A] perfect_twins.png  max activity deviation = {max_dev:.2e}")


def overdet_preimages(xtr, ytr, *, n_neurons=2000, seed=0):
    layer, W, b = build_layer(n_neurons, xtr, ytr, seed=seed)
    D = layer.decoders

    # naive activity inverse of the one-hot: realizable?
    Dt_pinv = torch.linalg.pinv(D.T)
    frac_neg = torch.stack(
        [(Dt_pinv @ torch.eye(10)[c] < 0).float().mean() for c in range(10)]
    ).mean()
    print(
        f"[B] naive min-norm activity inverse of one-hot: "
        f"{frac_neg:.2%} negative entries on average (=> not realizable as |z|)"
    )

    means = torch.stack([xtr[ytr == c].mean(0) for c in range(10)])
    fig, axes = plt.subplots(2, 10, figsize=(20, 4.4))
    for c in range(10):
        anchor = means[c]
        s = sign_cell(W, b, anchor)
        M = (D.T * s) @ W  # (10, d_in): cell-linearized decode map
        rhs = torch.eye(10)[c] - (D.T * s) @ b
        x = anchor + torch.linalg.pinv(M) @ (rhs - M @ anchor)  # nearest local solve
        in_cell = bool((sign_cell(W, b, x) == s).all())
        p, cp = decode(layer, x)
        show(axes[0, c], means[c], f"class {c} mean")
        show(
            axes[1, c],
            x,
            f"candidate [{'in cell' if in_cell else 'approx'}]\npred {cp} p={p[cp]:.2f}",
        )
    fig.suptitle(
        f"B) Realistic default (n_neurons={n_neurons} > d_in=784): local candidate "
        "for each one-hot target, anchored at the class mean",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overdet_preimages.png", dpi=110)
    print("[B] overdet_preimages.png saved")
    return layer, W, b, D


def nullspace_continuum(layer, W, b, D, xtr, ytr, *, c=3, n_show=7):
    x0 = xtr[ytr == c][0]
    s = sign_cell(W, b, x0)
    M = (D.T * s) @ W
    rhs = torch.eye(10)[c] - (D.T * s) @ b
    x_part = x0 + torch.linalg.pinv(M) @ (rhs - M @ x0)
    null = nullspace_basis(M)  # (d_in - 10, d_in)

    rng = torch.Generator().manual_seed(7)
    mags = torch.linspace(0.3, 2.4, n_show)
    fig, axes = plt.subplots(1, n_show + 1, figsize=(2 * (n_show + 1), 2.4))
    p, cp = decode(layer, x_part)
    show(axes[0], x_part, f"anchor\npred {cp} p={p[cp]:.2f}")
    kept, max_dev = 0, 0.0
    for i in range(n_show):
        coeff = torch.randn(null.shape[0], generator=rng)
        coeff = coeff / coeff.norm() * float(mags[i])
        x = x_part + coeff @ null
        a = layer.encode(x.reshape(1, -1)).squeeze(0)
        dev = ((a @ D) - torch.eye(10)[c]).abs().max().item()
        max_dev = max(max_dev, dev)
        p, cp = decode(layer, x)
        kept += int(cp == c)
        show(axes[i + 1], x, f"|Δ|={mags[i]:.1f}\npred {cp} dev={dev:.1e}")
    fig.suptitle(
        f"B') Null-space continuum (class {c}): {null.shape[0]} free dims; "
        f"{kept}/{n_show} keep class {c}, max one-hot decode dev={max_dev:.1e}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nullspace_continuum.png", dpi=110)
    print(f"[B'] nullspace_continuum.png  null dim={null.shape[0]}  kept={kept}/{n_show}")


def main():
    xtr, ytr, _, _ = load_mnist_subset()
    perfect_twins(xtr, ytr)
    layer, W, b, D = overdet_preimages(xtr, ytr)
    nullspace_continuum(layer, W, b, D, xtr, ytr)


if __name__ == "__main__":
    main()
