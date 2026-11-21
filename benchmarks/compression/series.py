"""Raw synthetic trajectories adapted from the local solomonoff_leenef/series.py."""

import math
from collections.abc import Callable

import torch
from torch import Tensor


def _generator(n: int, warmup: int, noise_std: float, seed: int) -> torch.Generator:
    if type(n) is not int or n < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("n must be a positive integer and warmup a nonnegative integer")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return torch.Generator(device="cpu").manual_seed(seed)


def _observe(x: Tensor, noise_std: float, rng: torch.Generator) -> Tensor:
    values = x + noise_std * torch.randn(x.shape, generator=rng, dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("generator produced nonfinite observations")
    return values


def mackey_glass(
    n: int,
    *,
    seed: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    power: int = 10,
    dt: float = 1.0,
    warmup: int = 1000,
    noise_std: float = 0.01,
) -> Tensor:
    """Euler Mackey–Glass with the source's indexed delay and random history."""
    rng = _generator(n, warmup, noise_std, seed)
    if type(tau) is not int or tau < 1 or type(power) is not int or power < 1:
        raise ValueError("tau and power must be positive integers")
    if not all(map(math.isfinite, (beta, gamma, dt))) or dt <= 0:
        raise ValueError("beta and gamma must be finite and dt finite and positive")
    hist_len = tau + 1
    x = torch.empty(n + warmup + hist_len, dtype=torch.float64)
    x[:hist_len] = 1.2 + 0.2 * torch.rand(hist_len, generator=rng, dtype=torch.float64)
    for t in range(hist_len, x.numel()):
        x_tau = x[t - tau]
        dx = beta * x_tau / (1.0 + x_tau**power) - gamma * x[t - 1]
        x[t] = x[t - 1] + dt * dx
    return _observe(x[hist_len + warmup :], noise_std, rng)


def _rk4(
    n: int,
    state: Tensor,
    deriv: Callable[[Tensor], Tensor],
    dt: float,
    subsample: int,
    warmup: int,
) -> Tensor:
    if type(subsample) is not int or subsample < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("subsample must be a positive integer and dt finite and positive")
    xs = torch.empty(n * subsample + warmup, dtype=torch.float64)
    for t in range(xs.numel()):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs[t] = state[0]
    return xs[warmup::subsample]


def lorenz(
    n: int,
    *,
    seed: int,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    subsample: int = 3,
    warmup: int = 1000,
    noise_std: float = 0.01,
) -> Tensor:
    """Subsampled x-coordinate of the Lorenz flow (RK4)."""
    rng = _generator(n, warmup, noise_std, seed)
    if not all(map(math.isfinite, (sigma, rho, beta))):
        raise ValueError("Lorenz coefficients must be finite")

    def deriv(s: Tensor) -> Tensor:
        x, y, z = s
        return torch.stack((sigma * (y - x), x * (rho - z) - y, x * y - beta * z))

    state = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    return _observe(_rk4(n, state, deriv, dt, subsample, warmup), noise_std, rng)


def rossler(
    n: int,
    *,
    seed: int,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    dt: float = 0.05,
    subsample: int = 2,
    warmup: int = 2000,
    noise_std: float = 0.01,
) -> Tensor:
    """Subsampled x-coordinate of the Rössler flow (RK4)."""
    rng = _generator(n, warmup, noise_std, seed)
    if not all(map(math.isfinite, (a, b, c))):
        raise ValueError("Rössler coefficients must be finite")

    def deriv(s: Tensor) -> Tensor:
        x, y, z = s
        return torch.stack((-y - z, x + a * y, b + z * (x - c)))

    state = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
    return _observe(_rk4(n, state, deriv, dt, subsample, warmup), noise_std, rng)


def henon(
    n: int,
    *,
    seed: int,
    a: float = 1.4,
    b: float = 0.3,
    warmup: int = 200,
    noise_std: float = 0.005,
) -> Tensor:
    """x-coordinate of the Hénon map, starting at (0, 0)."""
    rng = _generator(n, warmup, noise_std, seed)
    if not all(map(math.isfinite, (a, b))):
        raise ValueError("Hénon coefficients must be finite")
    x = torch.empty(n + warmup, dtype=torch.float64)
    x[0], y = 0.0, 0.0
    for t in range(1, x.numel()):
        x[t] = 1.0 - a * x[t - 1] * x[t - 1] + y
        y = b * x[t - 1]
    return _observe(x[warmup:], noise_std, rng)
