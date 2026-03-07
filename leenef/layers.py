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
                 activation: str = "relu",
                 encoder_strategy: str = "hypersphere",
                 trainable_encoders: bool = False,
                 gain: float = 1.0,
                 rng: torch.Generator | None = None,
                 encoder_kwargs: dict | None = None,
                 **act_kwargs):
        super().__init__()
        self.d_in = d_in
        self.n_neurons = n_neurons
        self.d_out = d_out

        # Encoders and bias
        enc_kw = encoder_kwargs or {}
        enc = make_encoders(n_neurons, d_in, strategy=encoder_strategy,
                            rng=rng, **enc_kw)
        bias = torch.randn(n_neurons, generator=rng)
        if trainable_encoders:
            self.encoders = nn.Parameter(enc)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("encoders", enc)
            self.register_buffer("bias", bias)

        self.gain = gain
        self.activation = make_activation(activation, **act_kwargs)

        # Decoders — always a parameter so it participates in state_dict
        self.decoders = nn.Parameter(torch.zeros(n_neurons, d_out),
                                     requires_grad=False)

    def encode(self, x: Tensor) -> Tensor:
        """Compute neuron activities for input x (N, d_in) → (N, n_neurons)."""
        return self.activation(self.gain * (x @ self.encoders.T) + self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass: encode → decode."""
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
        A = self.encode(x)
        D = solve_decoders(A, targets, method=solver, **solver_kwargs)
        self.decoders.data.copy_(D)
