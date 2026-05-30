"""Rate-based neuron activation functions for NEF layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LIFRate(nn.Module):
    """Leaky integrate-and-fire rate approximation.

    f(x) = 1 / (tau_ref - tau_rc * ln(1 - 1/max(x, 0)))
    for x > 1, else 0.  Operates on the pre-activation (gain * e·x + bias).
    """

    def __init__(self, tau_rc: float = 0.02, tau_ref: float = 0.002, amplitude: float = 1.0):
        super().__init__()
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amplitude = amplitude

    def forward(self, j: Tensor) -> Tensor:
        j = j.clamp(min=0)
        # Avoid log(0): where j <= 1 the neuron is below threshold
        safe = j.clamp(min=1 + 1e-7)
        rate = 1.0 / (self.tau_ref - self.tau_rc * torch.log1p(-1.0 / safe))
        return self.amplitude * torch.where(j > 1, rate, torch.zeros_like(rate))


class Abs(nn.Module):
    """Absolute-value activation — like ReLU but mirrors negatives."""

    def forward(self, x: Tensor) -> Tensor:
        return x.abs()


class Identity(nn.Module):
    """Linear / identity activation.

    Used as a sanity check (Paper I H8) — with linear activation, an
    NEFLayer's analytic decoder collapses to ordinary least-squares
    regression of `Y` on `X`, regardless of `n_neurons`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class Step(nn.Module):
    """Binary threshold activation at zero."""

    def forward(self, x: Tensor) -> Tensor:
        return (x > 0).to(dtype=x.dtype)


class Square(nn.Module):
    """Pointwise square activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x.square()


class BentLine(nn.Module):
    """Monotone single-kink activation — a "leaky" relative of abs/ReLU.

    ``f(x) = x`` for ``x >= 0`` and ``f(x) = slope * x`` for ``x < 0`` with
    ``0 < slope < 1``.  Both arms have positive slope, so unlike `abs` the
    map is monotone increasing; it merely *changes the angle at the origin*.
    Algebraically it is a fixed affine mix of identity and `abs`:

        f(x) = ((1 + slope) / 2) * x + ((1 - slope) / 2) * |x|.

    The default ``slope=0.5`` is chosen only because it is simple to write
    down; Paper I §2 predicts the exact slope is irrelevant (any genuine
    kink shares the same `abs` nonlinear component).
    """

    def __init__(self, slope: float = 0.5):
        super().__init__()
        self.slope = slope

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, self.slope * x)


class DoubleFold(nn.Module):
    """Two-kink folded activation ``f(x) = ||x| - c|``.

    A single neuron now contributes *two* fold lines instead of one, which
    enriches the even part of the dual activation kernel relative to `abs`.
    """

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        return (x.abs() - self.c).abs()


class SqrtAbs(nn.Module):
    """Concave magnitude activation ``f(x) = |x|^p`` (default ``p=0.5``).

    Same sign-folding as `abs` but a different even Hermite spectrum, so it
    selects a different (heavier low-frequency) dual kernel.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x.abs().pow(self.p)


class Cos(nn.Module):
    """Cosine activation ``f(x) = cos(x)`` — a random Fourier feature.

    With random encoders and per-neuron gain setting the bandwidth, a
    population of ``cos`` features approximates a stationary (RBF-like)
    kernel (Rahimi & Recht, 2007).  Strongly bandwidth- (gain-) sensitive.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(x)


class GaussBump(nn.Module):
    """Radial bump activation ``f(x) = exp(-x^2 / 2)``.

    With data-driven biases each neuron becomes a soft RBF receptive field
    centered on a training sample along its encoder direction.  Bandwidth is
    set by the per-neuron gain.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-0.5 * x.square())


class CosSin(nn.Module):
    """Parity-broken sinusoid ``f(x) = cos(x) + sin(x) = sqrt(2) cos(x - pi/4)``.

    A fixed-phase random Fourier feature.  Plain `cos` is an *even* function
    of its argument, so (at zero bias) its dual activation kernel keeps only
    even Hermite modes and discards every odd mode.  Adding the ``sin`` term
    breaks that symmetry: ``cos + sin`` has nonzero Hermite coefficients at
    *every* order, so the population realizes the full Gaussian-RBF kernel
    (even *and* odd modes) rather than its even-only projection.  This is the
    minimal "fix" for the even-only limitation of `cos`/`gauss` — it is a
    change of *phase* (symmetry), not of complexity.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(x) + torch.sin(x)


class SquareWave(nn.Module):
    """Digital cosine ``f(x) = sign(cos(x))`` — a piecewise-constant square wave.

    The Walsh/square-wave relative of `cos`.  Its Fourier series
    ``(4/pi)(cos x - cos 3x / 3 + cos 5x / 5 - ...)`` packs many frequencies
    into one neuron, giving a rich *even* dual kernel.  It is zero-gradient
    almost everywhere — inadmissible for gradient training, but perfectly
    usable here because decoders are solved analytically and the activation
    is never differentiated.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(torch.cos(x))


class SquareFloor(nn.Module):
    """Digital cosine plus monotone staircase ``f(x) = sign(cos(x)) + floor(x)``.

    This is a deliberately tiny fusion of two digital effects: the oscillatory
    `square_wave` contributes even Hermite modes, while ``floor`` contributes
    the strong constant/linear modes of a monotone quantizer.  Unlike
    `staircase`, it is unclamped; use it as a spectrum diagnostic rather than
    as a bounded default activation.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sign(torch.cos(x)) + torch.floor(x)


class Box(nn.Module):
    """Digital bump ``f(x) = 1[|x| < c]`` (default ``c=1``).

    The piecewise-constant (indicator) relative of the `gauss` bump.  With
    data-driven biases each neuron is a hard receptive field — it fires iff
    the projected input lands within ``c`` of the neuron's center.  Like the
    `gauss` bump it is an *even* function (even-only Hermite spectrum), and
    like `square_wave` it has zero gradient a.e., which is admissible only
    because the solve is analytic.
    """

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        return (x.abs() < self.c).to(dtype=x.dtype)


class Gelu(nn.Module):
    """Gaussian-error linear unit ``f(x) = x * Phi(x)``.

    A smooth, monotone-ish gate that is neither even nor odd, so — unlike
    `cos`/`gauss` — its dual activation kernel carries *both* even and odd
    Hermite modes while staying gain-robust (no bandwidth to tune).  A
    natural full-spectrum default candidate.
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class Silu(nn.Module):
    """Sigmoid linear unit / swish ``f(x) = x * sigmoid(x)``.

    Like `gelu`, a smooth full-spectrum gate (both parities) that needs no
    bandwidth tuning.  Included as a second gain-robust full-spectrum
    default candidate.
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)


class Softplus(nn.Module):
    """Smooth ReLU ``f(x) = log(1 + exp(beta * x)) / beta``.

    A smoothed relative of ReLU included as a kernel-equivalence control:
    smoothing the kink leaves the ``k >= 2`` Hermite spectrum essentially
    that of `abs`/ReLU.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x, beta=self.beta)


class Staircase(nn.Module):
    """Asymmetric quantizer / staircase ``f(x) = clamp(floor(x/s)*s, -L, L)``.

    A monotone, piecewise-constant *digital* activation (default step
    ``s=1``, saturation ``L=3``).  Using ``floor`` rather than ``round``
    makes it asymmetric (neither even nor odd).  It is zero-gradient almost
    everywhere (and undefined at the step edges), so it is inadmissible for
    gradient training but admissible here because the decoders are solved
    analytically and the activation is never differentiated.

    The clamp keeps the feature bounded and comparable to the other digital
    cells during wide-gain sweeps.  Spectrally it is a cautionary case rather
    than a strong default: a fine monotone quantizer of a unit-scale
    pre-activation tracks the identity closely, so its Hermite energy
    concentrates in the constant and linear ($k=0,1$) modes and its
    $k\\ge2$ coefficients are nearly zero (measured
    $a_2^2\\approx a_3^2\\approx0$).  It therefore sits close to the linear /
    collapse mode (cf. `identity`): being "digital" does not by itself buy
    discriminative ($k\\ge2$) power — the oscillatory `square_wave` and bump
    `box` are the digital activations that do carry a rich even spectrum.
    """

    def __init__(self, step: float = 1.0, limit: float = 3.0):
        super().__init__()
        self.step = step
        self.limit = limit

    def forward(self, x: Tensor) -> Tensor:
        q = torch.floor(x / self.step) * self.step
        return q.clamp_(min=-self.limit, max=self.limit)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
    "softplus": Softplus,
    "tanh": nn.Tanh,
    "lif_rate": LIFRate,
    "abs": Abs,
    "bent_line": BentLine,
    "double_fold": DoubleFold,
    "sqrt_abs": SqrtAbs,
    "cos": Cos,
    "gauss": GaussBump,
    "cos_sin": CosSin,
    "square_wave": SquareWave,
    "square_floor": SquareFloor,
    "box": Box,
    "staircase": Staircase,
    "gelu": Gelu,
    "silu": Silu,
    "linear": Identity,
    "identity": Identity,
    "step": Step,
    "square": Square,
}


def make_activation(name: str = "abs", **kwargs) -> nn.Module:
    """Create an activation by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name](**kwargs)
