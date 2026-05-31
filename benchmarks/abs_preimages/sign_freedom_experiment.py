"""Regime comparison: how many sign-preimages of a target activity are realizable?

Given a target activity ``a*`` (nonnegative), every sign vector ``s in {±1}^n``
defines a candidate pre-activation ``z = s ⊙ a*``; recovering an input means
least-squares solving ``W x = s ⊙ a* - b``.  Whether the recovered ``x`` actually
re-encodes to ``a*`` depends on whether ``s ⊙ a* - b`` lies in ``col(W)``, a
``d_in``-dimensional subspace:

* ``n_neurons <= d_in``  -> W has full row rank, every ``t`` realizable
                           -> all 2**n sign flips are exact ("perfect twins").
* ``n_neurons >  d_in``  -> activities live on a thin d_in-dim manifold; only
                           signs near real data are realizable.

Run::  python benchmarks/abs_preimages/sign_freedom_experiment.py
"""

from __future__ import annotations

import torch
from common import build_layer, load_mnist_subset


def run_regime(n_neurons, xtr, ytr, xte, yte, *, c=3, n_flip_trials=200, seed=0):
    layer, W, b = build_layer(n_neurons, xtr, ytr, seed=seed)
    d_in = xtr.shape[1]
    acc = (layer(xte).argmax(1) == yte).float().mean().item()
    rank = int(torch.linalg.matrix_rank(W).item())

    D = layer.decoders
    a_star = layer.encode(xtr[ytr == c]).mean(0)  # class-mean activity target
    Wpinv = torch.linalg.pinv(W)

    # a single real sample gives an exactly-realizable "natural" sign baseline
    x0 = xtr[ytr == c][0]
    a_real = (W @ x0 + b).abs()
    s_nat = torch.sign(W @ x0 + b)
    s_nat[s_nat == 0] = 1.0

    def recon(target_act, s):
        x = Wpinv @ (s * target_act - b)
        a_hat = layer.encode(x.reshape(1, -1)).squeeze(0)
        rel_err = (a_hat - target_act).norm().item() / target_act.norm().item()
        return x, rel_err, int((a_hat @ D).argmax())

    _, err_nat, _ = recon(a_real, s_nat)

    rng = torch.Generator().manual_seed(seed + 1)
    errs, correct = [], 0
    for _ in range(n_flip_trials):
        s = torch.randint(0, 2, (n_neurons,), generator=rng).float() * 2 - 1
        _, err, cls = recon(a_star, s)
        errs.append(err)
        correct += int(cls == c)
    errs = torch.tensor(errs)

    print(f"\n=== n_neurons={n_neurons}  (d_in={d_in}) ===")
    print(f"  baseline test acc                   : {acc:.4f}")
    print(f"  rank(W)                             : {rank}/{min(n_neurons, d_in)}")
    print(f"  natural-sign recon err (real sample): {err_nat:.2e}")
    print(
        f"  random-flip recon err (class mean)  : median={errs.median():.2e}  max={errs.max():.2e}"
    )
    print(f"  random flips decoding to class {c}     : {correct}/{n_flip_trials}")


def main():
    xtr, ytr, xte, yte = load_mnist_subset()
    run_regime(400, xtr, ytr, xte, yte)  # underdetermined: all flips realizable
    run_regime(2000, xtr, ytr, xte, yte)  # overdetermined: only natural signs


if __name__ == "__main__":
    main()
