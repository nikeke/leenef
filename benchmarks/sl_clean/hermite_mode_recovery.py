"""Synthetic Hermite-mode recovery — does the activation's spectrum predict
what a fixed-feature ridge solve can learn?

This is the controlled experiment behind ``docs/activation_kernel_theory.md``.
With Gaussian encoders, zero bias and unit gain, the leenef pipeline is an
*exact* instance of random-feature ridge regression whose dual kernel is the
Mehler / Hermite series

    k_G(x, x') = sum_k (a_k^2 / k!) * rho^k ,   rho = <x, x'> / d ,

where ``a_k = E_{z ~ N(0,1)}[ G(z) He_k(z) ]`` are the Hermite coefficients of
the scalar activation ``G`` (``He_k`` = probabilists' Hermite polynomials).
The prediction is sharp: a fixed-feature solve can represent the pure mode
``y_k(x) = he_k(<e0, x>)`` (``he_k`` normalized so ``E[he_j he_k] = delta_jk``)
*only* in proportion to ``a_k^2``.  In particular:

    * linear  G = z       -> only a_1 != 0  -> learns k=1, fails every k>=2
      (this is the linear collapse of SSUP-I section 2);
    * abs / |z|           -> a_1 = a_3 = 0  -> learns even modes, fails odd;
    * relu                -> all a_k != 0   -> learns every mode (weakly at high k);
    * square  G = z^2      -> only a_0, a_2  -> learns k=2 alone.

We measure, per activation and per target mode k=1..4, the held-out coefficient
of determination ``R^2`` of a ridge fit on n random features, and compare the
ordering and zero-pattern of ``R^2_k`` against the analytic ``a_k^2``.

Run:  PYTHONPATH=src python -m benchmarks.sl_clean.hermite_mode_recovery
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from leenef.solvers import solve_from_normal_equations

from .sl_repro import RESULTS_DIR, code_identity, hardware_info, seed_policy

# Activations under test, as plain callables on a torch tensor.  Kept local
# (rather than importing the library registry) so the analytic Hermite
# coefficients below are unambiguously the same scalar functions.
_ACTS = {
    "linear": lambda z: z,
    "abs": lambda z: z.abs(),
    "relu": torch.relu,
    "leaky_relu": lambda z: torch.where(z >= 0, z, 0.1 * z),
    "bent_line": lambda z: torch.where(z >= 0, z, 0.5 * z),
    "square": lambda z: z * z,
    "double_fold": lambda z: (z.abs() - 1.0).abs(),
    "sqrt_abs": lambda z: z.abs().sqrt(),
    "cos": torch.cos,
    "gauss": lambda z: torch.exp(-0.5 * z * z),
    "cos_sin": lambda z: torch.cos(z) + torch.sin(z),
    "square_wave": lambda z: torch.sign(torch.cos(z)),
    "square_floor": lambda z: torch.sign(torch.cos(z)) + torch.floor(z),
    "box": lambda z: (z.abs() < 1.0).to(z.dtype),
    "staircase": lambda z: (torch.floor(z / 1.0) * 1.0).clamp(-3.0, 3.0),
    "gelu": lambda z: torch.nn.functional.gelu(z),
    "silu": lambda z: torch.nn.functional.silu(z),
    "tanh": torch.tanh,
    "softplus": lambda z: torch.nn.functional.softplus(z),
}

_K_MAX = 6  # report a_0 .. a_6


def _normalized_he(z: np.ndarray, k: int) -> np.ndarray:
    """Probabilists' He_k(z) / sqrt(k!) so that E[he_j he_k] = delta_jk."""
    coef = np.zeros(k + 1)
    coef[k] = 1.0
    he_k = np.polynomial.hermite_e.hermeval(z, coef)
    return he_k / math.sqrt(math.factorial(k))


def hermite_coefficients(act, k_max: int = _K_MAX, n_quad: int = 256) -> list[float]:
    """a_k = E_{z~N(0,1)}[ G(z) he_k(z) ] via Gauss-Hermite quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
    weights = weights / math.sqrt(2.0 * math.pi)  # normalize int e^{-x^2/2} dx
    gz = act(torch.from_numpy(nodes)).numpy()
    out = []
    for k in range(k_max + 1):
        hek = _normalized_he(nodes, k)
        out.append(float(np.sum(weights * gz * hek)))
    return out


def _make_features(
    x: torch.Tensor, w: torch.Tensor, act, gain: float, bias: torch.Tensor
) -> torch.Tensor:
    z = x @ w.T
    z = gain * z + bias
    return act(z)


def _ridge_r2(
    feat_tr: torch.Tensor,
    y_tr: torch.Tensor,
    feat_te: torch.Tensor,
    y_te: torch.Tensor,
    alpha: float,
) -> float:
    ata = feat_tr.T @ feat_tr
    aty = feat_tr.T @ y_tr
    d = solve_from_normal_equations(ata, aty, alpha=(alpha, "trace"))
    pred = feat_te @ d
    ss_res = float(((pred - y_te) ** 2).sum())
    ss_tot = float(((y_te - y_te.mean(dim=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def run(
    *,
    d_in: int,
    n_features: int,
    n_train: int,
    n_test: int,
    gain: float,
    bias_std: float,
    alpha: float,
    seeds: list[int],
    k_targets: tuple[int, ...],
) -> dict:
    results: dict = {"meta": {}, "coefficients": {}, "r2": {}}
    results["meta"] = {
        "d_in": d_in,
        "n_features": n_features,
        "n_train": n_train,
        "n_test": n_test,
        "gain": gain,
        "bias_std": bias_std,
        "alpha": alpha,
        "seeds": seeds,
        "k_targets": list(k_targets),
        "code_identity": code_identity(),
        "seed_policy": seed_policy(),
        "hardware": hardware_info(),
    }

    # Analytic Hermite spectra (independent of data / seeds).
    for name, act in _ACTS.items():
        results["coefficients"][name] = hermite_coefficients(act)

    # Per-(activation, mode) held-out R^2, averaged over seeds.
    for name, act in _ACTS.items():
        per_k: dict[str, list[float]] = {f"k{k}": [] for k in k_targets}
        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            x_tr = torch.randn(n_train, d_in, generator=g, dtype=torch.float64)
            x_te = torch.randn(n_test, d_in, generator=g, dtype=torch.float64)
            u = torch.randn(n_features, d_in, generator=g, dtype=torch.float64)
            w = u / u.norm(dim=1, keepdim=True)  # unit encoders -> z ~ N(0,1)
            bias = bias_std * torch.randn(n_features, generator=g, dtype=torch.float64)
            feat_tr = _make_features(x_tr, w, act, gain, bias)
            feat_te = _make_features(x_te, w, act, gain, bias)
            for k in k_targets:
                y_tr = torch.from_numpy(_normalized_he(x_tr[:, 0].numpy(), k)).reshape(-1, 1)
                y_te = torch.from_numpy(_normalized_he(x_te[:, 0].numpy(), k)).reshape(-1, 1)
                per_k[f"k{k}"].append(_ridge_r2(feat_tr, y_tr, feat_te, y_te, alpha))
        results["r2"][name] = {
            kk: {"mean": float(np.mean(v)), "se": float(np.std(v) / max(1, math.sqrt(len(v))))}
            for kk, v in per_k.items()
        }
    return results


def _print_table(results: dict) -> None:
    k_targets = results["meta"]["k_targets"]
    header = ["activation"] + [f"a{k}^2" for k in range(_K_MAX + 1)]
    header += [f"R2_k{k}" for k in k_targets]
    print("  ".join(f"{h:>11}" for h in header))
    for name in _ACTS:
        coefs = results["coefficients"][name]
        row = [f"{name:>11}"]
        row += [f"{c * c:>11.3f}" for c in coefs]
        for k in k_targets:
            m = results["r2"][name][f"k{k}"]["mean"]
            row.append(f"{m:>11.3f}")
        print("  ".join(row))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--d-in", type=int, default=20)
    ap.add_argument("--n-features", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--bias-std", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--k-targets", default="1,2,3,4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(t) for t in args.seeds.split(",") if t.strip()]
    k_targets = tuple(int(t) for t in args.k_targets.split(",") if t.strip())

    results = run(
        d_in=args.d_in,
        n_features=args.n_features,
        n_train=args.n_train,
        n_test=args.n_test,
        gain=args.gain,
        bias_std=args.bias_std,
        alpha=args.alpha,
        seeds=seeds,
        k_targets=k_targets,
    )
    _print_table(results)

    out_path = Path(args.out) if args.out else RESULTS_DIR / "hermite_mode_recovery.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\n[hermite] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
