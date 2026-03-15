"""NEFLayer — a single NEF layer as a PyTorch Module."""

import torch
import torch.nn as nn
from torch import Tensor

from .activations import make_activation
from .encoders import make_encoders
from .solvers import solve_decoders


def _make_gain(
    spec: float | tuple[float, float] | Tensor, n_neurons: int, rng: torch.Generator | None = None
) -> Tensor:
    """Build a per-neuron gain vector from a gain specification.

    Args:
        spec: ``float`` for uniform gain, ``(low, high)`` tuple for
              per-neuron uniform sampling, or a pre-built ``Tensor``.
    """
    if isinstance(spec, Tensor):
        if spec.shape != (n_neurons,):
            raise ValueError(
                f"gain tensor must have shape ({n_neurons},), got {tuple(spec.shape)}"
            )
        return spec.float()
    if isinstance(spec, tuple):
        lo, hi = spec
        return lo + (hi - lo) * torch.rand(n_neurons, generator=rng)
    return torch.full((n_neurons,), spec, dtype=torch.float32)


class NEFLayer(nn.Module):
    """Single NEF layer: encode → activate → decode.

    Encoders (input weights) are random and optionally trainable.
    Decoders (output weights) are computed analytically via ``fit()``
    or trained with backprop like a normal ``nn.Module``.

    Args:
        gain: neuron gain — ``float`` for uniform, ``(low, high)`` tuple
              for per-neuron uniform sampling, or a ``Tensor(n_neurons,)``.
    """

    def __init__(
        self,
        d_in: int,
        n_neurons: int,
        d_out: int,
        activation: str = "abs",
        encoder_strategy: str = "hypersphere",
        trainable_encoders: bool = False,
        gain: float | tuple[float, float] | Tensor = (0.5, 2.0),
        rng: torch.Generator | None = None,
        centers: Tensor | None = None,
        **act_kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_neurons = n_neurons
        self.d_out = d_out
        self._rng = rng

        # Per-neuron gain vector, stored as buffer for state_dict
        self.register_buffer("_gain", _make_gain(gain, n_neurons, rng))

        # Encoders
        enc = make_encoders(n_neurons, d_in, strategy=encoder_strategy, rng=rng)

        # Biases — data-driven or random
        if centers is not None:
            idx = torch.randint(len(centers), (n_neurons,), generator=rng)
            bias = -self._gain * (centers[idx].float() * enc).sum(dim=1)
        else:
            bias = torch.randn(n_neurons, generator=rng)

        if trainable_encoders:
            self.encoders = nn.Parameter(enc)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("encoders", enc)
            self.register_buffer("bias", bias)

        self.activation = make_activation(activation, **act_kwargs)

        # Decoders — always a parameter so it participates in state_dict
        self.decoders = nn.Parameter(torch.zeros(n_neurons, d_out), requires_grad=False)

    @property
    def gain(self) -> Tensor:
        """Per-neuron gain vector (n_neurons,)."""
        return self._gain

    def encode(self, x: Tensor) -> Tensor:
        """Compute neuron activities for input x (N, d_in) → (N, n_neurons)."""
        return self.activation(self._gain * (x @ self.encoders.T) + self.bias)

    @torch.no_grad()
    def set_centers(self, centers: Tensor) -> None:
        """Recompute biases from data-driven centers."""
        idx = torch.randint(len(centers), (self.n_neurons,), generator=self._rng)
        self.bias.data.copy_(-self._gain * (centers[idx].float() * self.encoders.data).sum(dim=1))

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass: encode → decode."""
        if x.dim() != 2 or x.shape[1] != self.d_in:
            raise ValueError(f"Expected input shape (N, {self.d_in}), got {tuple(x.shape)}")
        return self.encode(x) @ self.decoders

    @torch.no_grad()
    def fit(self, x: Tensor, targets: Tensor, solver: str = "tikhonov", **solver_kwargs) -> None:
        """Analytically solve for decoders given input-target pairs.

        Args:
            x:       (N, d_in) input data
            targets: (N, d_out) desired outputs
            solver:  solver name from ``leenef.solvers``
        """
        if x.shape[0] != targets.shape[0]:
            raise ValueError(
                f"x and targets must have same number of samples, "
                f"got {x.shape[0]} vs {targets.shape[0]}"
            )
        A = self.encode(x)
        D = solve_decoders(A, targets, method=solver, **solver_kwargs)
        self.decoders.data.copy_(D)
