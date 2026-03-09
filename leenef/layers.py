"""NEFLayer — a single NEF layer as a PyTorch Module."""

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import make_encoders
from .activations import make_activation
from .solvers import solve_decoders


class NEFLayer(nn.Module):
    """Single NEF layer: encode → activate → decode.

    Encoders (input weights) are random and optionally trainable.
    Decoders (output weights) are computed analytically via ``fit()``
    or trained with backprop like a normal ``nn.Module``.
    """

    def __init__(self, d_in: int, n_neurons: int, d_out: int,
                 activation: str = "abs",
                 encoder_strategy: str = "hypersphere",
                 trainable_encoders: bool = False,
                 gain: float = 1.0,
                 rng: torch.Generator | None = None,
                 centers: Tensor | None = None,
                 **act_kwargs):
        super().__init__()
        self.d_in = d_in
        self.n_neurons = n_neurons
        self.d_out = d_out
        self._rng = rng

        # Gain — stored as buffer for state_dict serialization
        self.register_buffer("_gain", torch.tensor(gain, dtype=torch.float32))

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
        self.decoders = nn.Parameter(torch.zeros(n_neurons, d_out),
                                     requires_grad=False)

    @property
    def gain(self) -> float:
        """Scalar gain value."""
        return self._gain.item()

    def encode(self, x: Tensor) -> Tensor:
        """Compute neuron activities for input x (N, d_in) → (N, n_neurons)."""
        return self.activation(self._gain * (x @ self.encoders.T) + self.bias)

    @torch.no_grad()
    def set_centers(self, centers: Tensor) -> None:
        """Recompute biases from data-driven centers."""
        idx = torch.randint(len(centers), (self.n_neurons,),
                            generator=self._rng)
        self.bias.data.copy_(
            -self._gain * (centers[idx].float() * self.encoders.data).sum(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass: encode → decode."""
        if x.dim() != 2 or x.shape[1] != self.d_in:
            raise ValueError(
                f"Expected input shape (N, {self.d_in}), got {tuple(x.shape)}")
        return self.encode(x) @ self.decoders

    @torch.no_grad()
    def fit(self, x: Tensor, targets: Tensor,
            solver: str = "tikhonov", **solver_kwargs) -> None:
        """Analytically solve for decoders given input-target pairs.

        Args:
            x:       (N, d_in) input data
            targets: (N, d_out) desired outputs
            solver:  solver name from ``leenef.solvers``
        """
        if x.shape[0] != targets.shape[0]:
            raise ValueError(
                f"x and targets must have same number of samples, "
                f"got {x.shape[0]} vs {targets.shape[0]}")
        A = self.encode(x)
        D = solve_decoders(A, targets, method=solver, **solver_kwargs)
        self.decoders.data.copy_(D)
