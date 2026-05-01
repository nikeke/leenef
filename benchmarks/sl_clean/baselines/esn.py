"""Echo State Network (ESN) baseline for the SL clean comparison
(Paper I H4/H5).

A textbook leaky-integrator ESN: random fixed input weights ``W_in``
and a sparse random reservoir matrix ``W`` rescaled to a target
spectral radius.  The reservoir is unrolled across the sequence with

    u_t = W_in @ x_t + W @ h_{t-1}
    h_t = (1 - leak) * h_{t-1} + leak * tanh(u_t)

The classification head is a *ridge regression* on the final reservoir
state, solved analytically (no gradient descent) — mirroring the
streaming-NEF training discipline.

This is the cleanest "obvious thing" baseline for the streaming-NEF
claim: same fixed-feature + analytic-solve template, different feature
construction.  The interesting comparison is therefore *which random
feature construction* (reservoir dynamics vs delay-line + NEF
encoding) carries more class information at the readout.

All randomness is driven by explicit ``torch.Generator`` instances so
results are bit-exact reproducible, with no reliance on the global
RNG.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ESNConfig:
    n_reservoir: int
    spectral_radius: float
    leak_rate: float
    input_scale: float = 1.0
    sparsity: float = 0.1
    ridge_alpha: float = 1e-3
    chunk_size: int = 256
    use_bias_state: bool = True


def _make_input_weights(
    d_in: int, n: int, scale: float, generator: torch.Generator
) -> torch.Tensor:
    """W_in ~ U(-scale, scale)."""
    W = torch.empty(n, d_in)
    W.uniform_(-scale, scale, generator=generator)
    return W


def _make_reservoir(
    n: int, spectral_radius: float, sparsity: float, generator: torch.Generator
) -> torch.Tensor:
    """Sparse random reservoir matrix rescaled to ``spectral_radius``.

    Density is ``sparsity`` (fraction of nonzero entries).  Spectral
    radius is computed from eigenvalues for accuracy; for n <= 4000
    this is a few seconds and we only do it once per (cell, seed).
    """
    W = torch.zeros(n, n)
    n_nonzero = max(1, min(n * n, int(round(sparsity * n * n))))
    # Draw unique flat positions via randperm so duplicate scatter writes
    # (which are non-deterministic in PyTorch) cannot occur.  This costs
    # O(n^2) memory once per (cell, seed) but removes the reproducibility
    # hazard.
    flat = torch.randperm(n * n, generator=generator)[:n_nonzero]
    rows = flat // n
    cols = flat % n
    vals = torch.empty(n_nonzero).uniform_(-1.0, 1.0, generator=generator)
    W[rows, cols] = vals
    eigs = torch.linalg.eigvals(W)
    rho = eigs.abs().max().item()
    if rho > 0:
        W = W * (spectral_radius / rho)
    return W


def _collect_final_states(
    x: torch.Tensor,  # (N, T, d_in)
    W_in: torch.Tensor,  # (n, d_in)
    W: torch.Tensor,  # (n, n)
    leak: float,
    chunk_size: int,
    use_bias_state: bool,
) -> torch.Tensor:
    """Run the reservoir; return final hidden state per sequence."""
    n = W.shape[0]
    N, T, _ = x.shape
    device = x.device
    out_dim = n + (1 if use_bias_state else 0)
    states = torch.empty(N, out_dim, device=device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xb = x[start:end]
        bsz = xb.shape[0]
        h = torch.zeros(bsz, n, device=device)
        for t in range(T):
            u = xb[:, t, :] @ W_in.T + h @ W.T
            h = (1.0 - leak) * h + leak * torch.tanh(u)
        if use_bias_state:
            states[start:end] = torch.cat([h, torch.ones(bsz, 1, device=device)], dim=1)
        else:
            states[start:end] = h
    return states


def _ridge_solve(
    states: torch.Tensor,  # (N, n')
    targets: torch.Tensor,  # (N, k)  one-hot
    alpha: float,
) -> torch.Tensor:
    """Closed-form ridge regression in float64 for stability."""
    S = states.double()
    Y = targets.double()
    n = S.shape[1]
    A = S.T @ S + alpha * torch.eye(n, dtype=torch.float64, device=S.device)
    B = S.T @ Y
    W_out = torch.linalg.solve(A, B)
    return W_out


def train_esn(
    x_train: torch.Tensor,  # (N, T, d_in)
    y_train: torch.Tensor,  # (N,)
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: ESNConfig,
    *,
    generator_input: torch.Generator,
    generator_reservoir: torch.Generator,
    device: torch.device | str = "cpu",
) -> dict:
    device = torch.device(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    d_in = x_train.shape[-1]
    n_classes = int(y_train.max().item()) + 1

    t0 = time.perf_counter()
    # All random structure built on CPU then moved (eigvals stable on CPU).
    W_in_cpu = _make_input_weights(d_in, cfg.n_reservoir, cfg.input_scale, generator_input)
    W_cpu = _make_reservoir(
        cfg.n_reservoir, cfg.spectral_radius, cfg.sparsity, generator_reservoir
    )
    W_in = W_in_cpu.to(device)
    W = W_cpu.to(device)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    states_train = _collect_final_states(
        x_train, W_in, W, cfg.leak_rate, cfg.chunk_size, cfg.use_bias_state
    )
    states_test = _collect_final_states(
        x_test, W_in, W, cfg.leak_rate, cfg.chunk_size, cfg.use_bias_state
    )
    collect_time = time.perf_counter() - t0

    Y = torch.zeros(y_train.shape[0], n_classes, device=device)
    Y[torch.arange(y_train.shape[0]), y_train] = 1.0

    t0 = time.perf_counter()
    W_out = _ridge_solve(states_train, Y, cfg.ridge_alpha)
    solve_time = time.perf_counter() - t0

    pred_train = (states_train.double() @ W_out).argmax(dim=1)
    pred_test = (states_test.double() @ W_out).argmax(dim=1)
    train_acc = (pred_train == y_train).float().mean().item()
    test_acc = (pred_test == y_test).float().mean().item()

    n_params = int(W_in.numel() + W.numel() + W_out.numel())

    return {
        "test_accuracy": float(test_acc),
        "train_accuracy": float(train_acc),
        "build_time_sec": float(build_time),
        "collect_time_sec": float(collect_time),
        "solve_time_sec": float(solve_time),
        "fit_time_sec": float(build_time + collect_time + solve_time),
        "n_params": n_params,
    }
