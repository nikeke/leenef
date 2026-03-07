"""Rate-based neuron activation functions for NEF layers."""

import torch
import torch.nn as nn
from torch import Tensor


class LIFRate(nn.Module):
    """Leaky integrate-and-fire rate approximation.

    f(x) = 1 / (tau_ref - tau_rc * ln(1 - 1/max(x, 0)))
    for x > 1, else 0.  Operates on the pre-activation (gain * e·x + bias).
    """

    def __init__(self, tau_rc: float = 0.02, tau_ref: float = 0.002,
                 amplitude: float = 1.0):
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


ACTIVATIONS = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "lif_rate": LIFRate,
    "abs": Abs,
}


def make_activation(name: str = "relu", **kwargs) -> nn.Module:
    """Create an activation by name."""
    return ACTIVATIONS[name](**kwargs)
