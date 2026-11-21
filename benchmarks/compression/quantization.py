"""Fixed grids and Gaussian cell codelengths (adapted from solomonoff_leenef)."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

_INT_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
_LOG_2 = math.log(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _real_double(values: Tensor) -> Tensor:
    if not (values.is_floating_point() or values.dtype in _INT_DTYPES):
        raise ValueError("expected a real-valued tensor")
    return values.double()


@dataclass(frozen=True)
class UniformQuantizer:
    """Nearest-center grid with explicit, shared bounds and saturated end cells."""

    lo: float
    hi: float
    n_bits: int

    def __post_init__(self) -> None:
        if type(self.n_bits) is not int or not (1 <= self.n_bits <= 32):
            raise ValueError("n_bits must be an integer in [1, 32]")
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo < self.hi):
            raise ValueError("quantizer bounds must be finite and ascending")
        if not math.isfinite(self.delta) or self.delta < 4 * max(
            math.ulp(self.lo), math.ulp(self.hi)
        ):
            raise ValueError("quantizer spacing must be finite and resolvable in float64")

    @property
    def last_symbol(self) -> int:
        return (1 << self.n_bits) - 1

    @property
    def delta(self) -> float:
        return (self.hi - self.lo) / self.last_symbol

    def quantize(self, values: Tensor) -> Tensor:
        """Return int64 symbols, rounding halfway to even and clipping to the grid."""
        values = _real_double(values)
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")
        positions = (values.clamp(self.lo, self.hi) - self.lo) / self.delta
        return positions.round().clamp(0, self.last_symbol).long()

    def _symbol_indices(self, symbols: Tensor) -> Tensor:
        if symbols.dtype not in _INT_DTYPES:
            raise ValueError("symbols must be integers")
        indices = symbols.long()
        if ((indices < 0) | (indices > self.last_symbol)).any():
            raise ValueError("symbol outside the quantizer alphabet")
        return indices

    def reconstruct(self, symbols: Tensor) -> Tensor:
        """Return float64 bin centers, not the original continuous observations."""
        return self.lo + self._symbol_indices(symbols).double() * self.delta

    def gaussian_log_prob(self, symbols: Tensor, mean: Tensor, variance: Tensor) -> Tensor:
        """Natural log cell probabilities for broadcastable Gaussian moments."""
        indices, mean, variance = torch.broadcast_tensors(
            self._symbol_indices(symbols), _real_double(mean), _real_double(variance)
        )
        if not torch.isfinite(mean).all():
            raise ValueError("Gaussian means must be finite")
        if not (torch.isfinite(variance) & (variance > 0)).all():
            raise ValueError("Gaussian variances must be finite and positive")
        std = variance.sqrt()

        def standardized(value: Tensor) -> Tensor:
            difference = value - mean
            return torch.where(difference.isfinite(), difference / std, value / std - mean / std)

        # Adjacent cells share the same boundary expression. Reflect positive
        # tails so that neither CDF rounds to one before the log difference.
        s = indices.double()
        at_lo = indices == 0
        at_hi = indices == self.last_symbol
        lower = standardized(self.lo + (s - 0.5).clamp_min(0) * self.delta)
        upper = standardized(self.lo + (s + 0.5).clamp_max(self.last_symbol) * self.delta)
        log_cdf = torch.special.log_ndtr
        reflected = lower > 0
        log_big = log_cdf(torch.where(reflected, -lower, upper))
        log_small = log_cdf(torch.where(reflected, -upper, lower))
        interior = log_big + (-torch.expm1(log_small - log_big)).log()

        # When the density changes very little across a cell, CDF differences
        # can lose all digits. Integrate the relative density with three-node
        # Gauss–Legendre quadrature on this narrow interval instead.
        mid = standardized(self.lo + s * self.delta)
        half_width = (self.delta / 2) / std
        narrow = (half_width * (mid.abs() + half_width)) <= 0.01
        offset = math.sqrt(3 / 5) * half_width
        corrections = torch.stack(
            (
                math.log(4 / 9) + torch.zeros_like(mid),
                math.log(5 / 18) - offset * (mid + offset / 2),
                math.log(5 / 18) + offset * (mid - offset / 2),
            )
        )
        thin = (
            math.log(self.delta)
            - std.log()
            - _LOG_SQRT_2PI
            - (0.5 * mid) * mid
            + torch.logsumexp(corrections, dim=0)
        )
        log_prob = torch.where(narrow, thin, interior)
        log_prob = torch.where(at_lo, log_cdf(upper), log_prob)
        log_prob = torch.where(at_hi, log_cdf(-lower), log_prob)
        if not (log_prob.isfinite() & (log_prob <= 0)).all():
            raise FloatingPointError("Gaussian log cell probability is not representable")
        return log_prob

    def gaussian_bits(self, symbols: Tensor, mean: Tensor, variance: Tensor) -> Tensor:
        """Return per-symbol ideal bits, without flooring small probabilities."""
        bits = -self.gaussian_log_prob(symbols, mean, variance) / _LOG_2
        if not bits.isfinite().all():
            raise FloatingPointError("Gaussian cell codelength is not representable")
        return bits
