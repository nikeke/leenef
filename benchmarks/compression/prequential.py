"""Single-expert ideal codelength from a strictly causal fixed-feature predictor."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from leenef import NEFLayer
from leenef.activations import ACTIVATIONS
from leenef.solvers import (
    inverse_from_normal_equations,
    woodbury_decoders,
    woodbury_update_inverse_,
)

from .quantization import UniformQuantizer


@dataclass(frozen=True)
class PredictorConfig:
    """Fixed ridge strength and observation variance in reconstructed-value units."""

    window: int
    width: int
    warmup: int
    resolve_every: int
    fixed_alpha: float
    noise_variance: float
    seed: int
    activation: str = "abs"
    refresh_every: int = 16

    def __post_init__(self) -> None:
        for name in ("window", "width", "warmup", "resolve_every", "refresh_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fixed_alpha", "noise_variance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")


@dataclass(frozen=True)
class PrequentialResult:
    bits: Tensor
    mean: Tensor
    variance: Tensor

    @property
    def uniform_prefix(self) -> int:
        """Count of uniformly scored symbols before the first Gaussian prediction."""
        return self.bits.numel() - self.mean.numel()

    @property
    def total_bits(self) -> float:
        total = float(self.bits.sum())
        if not math.isfinite(total):
            raise FloatingPointError("total codelength is not representable")
        return total


def _layer(config: PredictorConfig, centers: Tensor) -> NEFLayer:
    # Match the source's order of RNG draws for feature initialization.
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    idx = torch.randperm(len(centers), generator=rng)[:4000]
    return NEFLayer(
        config.window,
        config.width,
        1,
        activation=config.activation,
        encoder_strategy="hypersphere",
        gain=(0.5, 2.0),
        rng=rng,
        centers=centers[idx],
    ).float()


@torch.no_grad()
def score_symbols(
    symbols: Tensor, quantizer: UniformQuantizer, config: PredictorConfig
) -> PrequentialResult:
    """Score a CPU symbol vector using only prior symbols at each prediction.

    The first ``window + warmup`` symbols are scored uniformly. ``mean`` and
    ``variance`` correspond to the remaining suffix, with one entry per symbol.
    """
    if symbols.device.type != "cpu" or symbols.ndim != 1:
        raise ValueError("expected a one-dimensional CPU symbol stream")
    recon = quantizer.reconstruct(symbols)
    n = recon.numel()
    start = min(config.window + config.warmup, n)
    bits = torch.full((n,), quantizer.n_bits, dtype=torch.float64)
    mean = torch.empty(n - start, dtype=torch.float64)
    variance = torch.empty_like(mean)
    if start == n:
        return PrequentialResult(bits, mean, variance)

    if max(abs(quantizer.lo), abs(quantizer.hi)) > torch.finfo(torch.float32).max:
        raise ValueError("quantizer bounds must be within the float32 feature range")
    windows = recon[:-1].float().unfold(0, config.window, 1)
    layer = _layer(config, windows[: config.warmup])
    gram = torch.zeros((config.width, config.width), dtype=torch.float64)
    cross = torch.zeros((config.width, 1), dtype=torch.float64)
    pending_phi = []
    pending_target = []
    inverse = decoders = None
    refits = 0
    for i, window in enumerate(windows):
        t = i + config.window
        phi = layer.encode(window[None, :]).double()
        if not phi.isfinite().all():
            raise FloatingPointError("nonfinite predictor activity")

        if inverse is not None:
            j = t - start
            mean[j] = (phi @ decoders).squeeze()
            leverage = ((phi @ inverse) * phi).sum()
            if not math.isfinite(leverage) or leverage < 0:
                raise FloatingPointError("invalid posterior leverage; check conditioning")
            variance[j] = config.noise_variance * (1 + leverage)
            bits[t] = quantizer.gaussian_bits(symbols[t], mean[j], variance[j])

        pending_phi.append(phi)
        pending_target.append(recon[t])
        seen = i + 1
        # Source refits after consuming the warmup target, then on zero-based
        # target-window multiples (not on multiples of `seen`).
        if seen >= config.warmup and (seen == config.warmup or i % config.resolve_every == 0):
            activities = torch.cat(pending_phi)
            targets = torch.stack(pending_target)[:, None]
            gram.addmm_(activities.T, activities)
            cross.addmm_(activities.T, targets)
            if (
                inverse is None
                or activities.shape[0] >= config.width
                or refits % config.refresh_every == 0
            ):
                inverse = inverse_from_normal_equations(gram, alpha=(config.fixed_alpha, "fixed"))
            else:
                woodbury_update_inverse_(inverse, activities)
            decoders = woodbury_decoders(inverse, cross, dtype=torch.float64)
            if not (inverse.isfinite().all() and decoders.isfinite().all()):
                raise FloatingPointError("nonfinite fitted posterior")
            refits += 1
            pending_phi.clear()
            pending_target.clear()

    return PrequentialResult(bits, mean, variance)
