"""Canonical class prototypes and the margin <-> realism tradeoff (part a).

The exact one-hot preimage only asks ``D^T a = e_c`` and decodes at the
unremarkable confidence ``p = e/(e+9) ~ 0.23``.  A *canonical* prototype instead
maximizes the decision margin

    m_c(x) = logit_c(x) - logsumexp_{j != c} logit_j(x)

over the pixel box ``x in [0, 1]`` (sigmoid reparametrization), optionally with a
manifold anchor ``lambda * ||x - mean_c||^2`` that pulls the prototype toward the
class mean.

The forward map is the closed-form piecewise-linear ``logits = |W x + b| @ D``, so
this is the NEF analogue of feature-visualization / activation-maximization.

Key finding: for a *random-encoder* abs layer, margin and realism are in direct
conflict.  Unconstrained margin-max produces high-margin adversarial noise far
off the data manifold; anchoring toward real digits collapses the margin to ~0.
The confident regions of the decoder live off the data manifold -- the random
features make the layer behave like a classic adversarial-prone linear model in
activity space.

Run::  python benchmarks/abs_preimages/canonical_prototypes.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from common import FIG_DIR, build_layer, load_mnist_subset, show  # noqa: E402

CLASSES = torch.arange(10)


def margin(logits):
    target = logits[CLASSES, CLASSES]
    others = logits.clone()
    others[CLASSES, CLASSES] = float("-inf")
    return target - torch.logsumexp(others, dim=1)


def maximize_margin(W, b, D, means, *, anchor_w=0.0, steps=400, lr=0.05):
    """Optimize one input per class to maximize margin, anchored to the mean."""
    eps = 1e-4
    theta = torch.logit(means.clamp(eps, 1 - eps)).clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x = torch.sigmoid(theta)
        logits = (x @ W.T + b).abs() @ D
        loss = -margin(logits).sum() + anchor_w * ((x - means) ** 2).sum()
        loss.backward()
        opt.step()
    x = torch.sigmoid(theta).detach()
    logits = (x @ W.T + b).abs() @ D
    m = margin(logits)
    dist = ((x - means) ** 2).sum(1).sqrt()
    acc = (logits.argmax(1) == CLASSES).float().mean().item()
    return x, m, dist, acc


def main():
    xtr, ytr, _, _ = load_mnist_subset()
    layer, W, b = build_layer(2000, xtr, ytr)
    D, W, b = layer.decoders.detach(), W.detach(), b.detach()
    means = torch.stack([xtr[ytr == c].mean(0) for c in range(10)])

    anchors = [0.0, 0.02, 0.05, 0.1, 0.2]
    protos, margins, dists = {}, {}, {}
    print("anchor   mean-margin   mean|x-mean|   argmax==class")
    base_m = margin((means @ W.T + b).abs() @ D)
    base_acc = (((means @ W.T + b).abs() @ D).argmax(1) == CLASSES).float().mean()
    print(f"  mean       {base_m.mean():+.2f}          0.00           {base_acc:.0%}")
    for aw in anchors:
        x, m, dist, acc = maximize_margin(W, b, D, means, anchor_w=aw)
        protos[aw], margins[aw], dists[aw] = x, m, dist
        print(f"  {aw:.2f}       {m.mean():+.2f}          {dist.mean():.2f}           {acc:.0%}")

    # --- grid: morph from adversarial noise to digit-like across anchors ---
    rows = 1 + len(anchors)
    fig, axes = plt.subplots(rows, 10, figsize=(20, 2.2 * rows))
    for c in range(10):
        show(axes[0, c], means[c], f"class {c} mean")
    for r, aw in enumerate(anchors, start=1):
        for c in range(10):
            show(axes[r, c], protos[aw][c], f"lam={aw}\nm={margins[aw][c]:+.1f}")
    fig.suptitle(
        "Canonical prototypes (n_neurons=2000): margin maximization with increasing "
        "manifold anchor lam*||x-mean||^2 -- high margin is adversarial noise "
        "(lam=0), realism returns as lam grows but margin collapses",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "canonical_prototypes.png", dpi=110)
    print("saved canonical_prototypes.png")

    # --- frontier curve: margin vs distance-from-mean ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [dists[aw].mean().item() for aw in anchors]
    ys = [margins[aw].mean().item() for aw in anchors]
    ax.plot(xs, ys, "-o")
    for aw, xv, yv in zip(anchors, xs, ys):
        ax.annotate(f"lam={aw}", (xv, yv), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set(
        xlabel="mean distance from class mean  ||x - mean||",
        ylabel="mean decision margin",
        title="Margin vs realism tradeoff (random-encoder abs NEF layer)",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "margin_realism_tradeoff.png", dpi=110)
    print("saved margin_realism_tradeoff.png")


if __name__ == "__main__":
    main()
