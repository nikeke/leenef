"""Causal categorical and linear controls for ideal compression experiments."""

import math
from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from leenef.solvers import inverse_from_normal_equations

from .prequential import PrequentialResult
from .quantization import UniformQuantizer


def _validate_stream(symbols: Tensor) -> None:
    if symbols.device.type != "cpu" or symbols.ndim != 1:
        raise ValueError("expected a one-dimensional CPU symbol stream")


def _positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _log_count_plus(count: int, log_prior: float) -> float:
    if count == 0:
        return log_prior
    log_count = math.log(count)
    return max(log_count, log_prior) + math.log1p(math.exp(-abs(log_count - log_prior)))


class AdaptiveCategorical:
    """Order-0 coder with a fixed positive pseudocount per grid symbol."""

    def __init__(self, quantizer: UniformQuantizer, *, pseudocount: float) -> None:
        _positive_finite(pseudocount, "pseudocount")
        self._quantizer = quantizer
        self._log_prior = math.log(pseudocount)
        self._log_other_prior = math.log(quantizer.last_symbol) + self._log_prior
        self._counts: dict[int, int] = {}
        self._seen = 0

    def _bits(self, count: int) -> float:
        if not self._seen:
            return float(self._quantizer.n_bits)
        # log(total / weight) = log(1 + other_weight / weight). Neither
        # large prior totals nor probabilities close to 0 or 1 need rounding.
        odds = _log_count_plus(self._seen - count, self._log_other_prior) - _log_count_plus(
            count, self._log_prior
        )
        return (max(0.0, odds) + math.log1p(math.exp(-abs(odds)))) / math.log(2.0)

    def bits_for(self, symbol: int) -> float:
        """Score a candidate without changing the current predictive distribution."""
        if type(symbol) is not int or not 0 <= symbol <= self._quantizer.last_symbol:
            raise ValueError("expected an integer symbol in the quantizer alphabet")
        return self._bits(self._counts.get(symbol, 0))

    def score_symbols(self, symbols: Tensor) -> Tensor:
        """Score a CPU vector before counting each symbol; retain state across calls."""
        _validate_stream(symbols)
        indices = self._quantizer._symbol_indices(symbols)
        bits = []
        for symbol in indices.tolist():
            count = self._counts.get(symbol, 0)
            bits.append(self._bits(count))
            self._counts[symbol] = count + 1
            self._seen += 1
        return torch.tensor(bits, dtype=torch.float64, device="cpu")


@dataclass(frozen=True)
class ARConfig:
    """Fixed positive ridge and observation variance in reconstructed-value units."""

    order: int
    warmup: int
    resolve_every: int
    fixed_alpha: float
    noise_variance: float

    def __post_init__(self) -> None:
        for name in ("order", "warmup", "resolve_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        _positive_finite(self.fixed_alpha, "fixed_alpha")
        _positive_finite(self.noise_variance, "noise_variance")


class PrequentialAR:
    """Ridge AR with a penalized intercept and C0's Gaussian/refit conventions."""

    def __init__(self, quantizer: UniformQuantizer, config: ARConfig) -> None:
        self._quantizer = quantizer
        self._config = config
        self._history: deque[float] = deque(maxlen=config.order)
        self._rows = 0
        width = config.order + 1
        self._gram = torch.zeros((width, width), dtype=torch.float64, device="cpu")
        self._cross = torch.zeros(width, dtype=torch.float64, device="cpu")
        self._fit: tuple[Tensor, Tensor] | None = None
        self._failed = False

    def _refit(self) -> None:
        if not (self._gram.isfinite().all() and self._cross.isfinite().all()):
            raise FloatingPointError("nonfinite AR sufficient statistics")
        inverse = inverse_from_normal_equations(
            self._gram, alpha=(self._config.fixed_alpha, "fixed")
        )
        coefficients = inverse @ self._cross
        if not (inverse.isfinite().all() and coefficients.isfinite().all()):
            raise FloatingPointError("nonfinite fitted AR model")
        self._fit = inverse, coefficients

    def score_symbols(self, symbols: Tensor) -> PrequentialResult:
        """Score and consume a chunk; moments cover only its nonuniform suffix.

        The first ``order + warmup`` symbols of the stream cost ``n_bits``
        each. Refit after scoring and consuming the warmup targets, then
        after each zero-based target-window index divisible by ``resolve_every``.
        A numerical failure makes this scorer unusable; input validation
        errors do not change its state.
        """
        if self._failed:
            raise RuntimeError(
                "AR scorer is unusable after a numerical failure; use a fresh scorer"
            )
        _validate_stream(symbols)
        recon = self._quantizer.reconstruct(symbols)
        try:
            return self._score_chunk(symbols, recon)
        except (FloatingPointError, torch.linalg.LinAlgError):
            self._failed = True
            raise

    def _score_chunk(self, symbols: Tensor, recon: Tensor) -> PrequentialResult:
        bits = torch.full(recon.shape, self._quantizer.n_bits, dtype=torch.float64, device="cpu")
        means, variances = [], []
        for t, center in enumerate(recon.tolist()):
            if len(self._history) == self._config.order:
                row = torch.tensor([1.0, *self._history], dtype=torch.float64, device="cpu")
                if self._fit is not None:
                    inverse, coefficients = self._fit
                    mean = row @ coefficients
                    leverage = row @ (inverse @ row)
                    if not math.isfinite(leverage) or leverage < 0:
                        raise FloatingPointError("invalid AR leverage; check conditioning")
                    variance = self._config.noise_variance * (1 + leverage)
                    if not (math.isfinite(mean) and math.isfinite(variance)):
                        raise FloatingPointError("nonfinite AR predictive moments")
                    bits[t] = self._quantizer.gaussian_bits(symbols[t], mean, variance)
                    means.append(mean)
                    variances.append(variance)

                self._gram.addr_(row, row)
                self._cross.add_(row, alpha=center)
                index = self._rows
                self._rows += 1
                if self._rows == self._config.warmup or (
                    self._rows > self._config.warmup and index % self._config.resolve_every == 0
                ):
                    self._refit()
            self._history.append(center)

        empty = torch.empty(0, dtype=torch.float64, device="cpu")
        return PrequentialResult(
            bits,
            torch.stack(means) if means else empty,
            torch.stack(variances) if variances else empty,
        )
