"""Streaming NEF classifier for temporal/sequential data.

Encodes sequences with a delay-line reservoir approach: overlapping windows
of consecutive timesteps are projected through random NEF encoders, activities
are mean-pooled across time, and an output decoder maps pooled activities to
class labels.  Supports both batch and continuous (Woodbury) training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .activations import make_activation
from .encoders import make_encoders
from .layers import _make_gain
from .solvers import solve_decoders


class StreamingNEFClassifier(nn.Module):
    """Classify variable-length sequences via temporal NEF encoding.

    Architecture::

        x_seq  (N, T, d)
           │
           ▼  delay-line windows
        (N, T, K·d)
           │
           ▼  random NEF encoding per timestep
        activities  (N, T, n_neurons)
           │
           ▼  mean pooling over time
        pooled  (N, n_neurons)
           │
           ▼  linear decoder
        output  (N, d_out)

    The delay-line with window size *K* concatenates *K* consecutive timestep
    features, giving each neuron access to a short temporal context.  Mean
    pooling collapses the variable-length temporal dimension into a
    fixed-size representation.

    Args:
        d_timestep:        Feature dimension per timestep.
        n_neurons:         Number of random neurons.
        d_out:             Output dimension (number of classes).
        window_size:       Delay-line window length *K*.
        activation:        NEF activation function name.
        encoder_strategy:  Encoder strategy for the temporal encoders.
        gain:              Per-neuron gain specification.
        rng:               Optional ``torch.Generator`` for reproducibility.
        centers:           Optional (N, T, d) training sequences for
                           data-driven biases (first window is used).
        encoder_kwargs:    Extra keyword arguments for the encoder strategy.
    """

    def __init__(
        self,
        d_timestep: int,
        n_neurons: int,
        d_out: int,
        window_size: int = 5,
        activation: str = "abs",
        encoder_strategy: str = "hypersphere",
        gain: float | tuple[float, float] | Tensor = (0.5, 2.0),
        rng: torch.Generator | None = None,
        centers: Tensor | None = None,
        encoder_kwargs: dict | None = None,
        **act_kwargs,
    ):
        super().__init__()
        self.d_timestep = d_timestep
        self.n_neurons = n_neurons
        self.d_out = d_out
        self.window_size = window_size
        d_in = d_timestep * window_size

        self._rng = rng
        self.register_buffer("_gain", _make_gain(gain, n_neurons, rng))

        enc = make_encoders(
            n_neurons, d_in, strategy=encoder_strategy, rng=rng, **(encoder_kwargs or {})
        )
        self.register_buffer("encoders", enc)
        self.activation = make_activation(activation, **act_kwargs)

        # Data-driven biases from delay-line features of training sequences
        if centers is not None:
            delay = self._delay_features(centers)
            # Flatten all timesteps and subsample for centers
            flat = delay.reshape(-1, d_in)
            idx = torch.randint(len(flat), (n_neurons,), generator=rng)
            bias = -self._gain * (flat[idx].float() * enc).sum(dim=1)
        else:
            bias = torch.randn(n_neurons, generator=rng)
        self.register_buffer("bias", bias)

        self.decoders = nn.Parameter(torch.zeros(n_neurons, d_out), requires_grad=False)

        # Woodbury state — initialised lazily by continuous_fit

    def _delay_features(self, x_seq: Tensor) -> Tensor:
        """Convert (N, T, d) sequences to (N, T, K*d) delay-line features."""
        K = self.window_size
        # Pad beginning with zeros so the first timestep has a full window
        padded = F.pad(x_seq, (0, 0, K - 1, 0))  # (N, T+K-1, d)
        # Extract overlapping windows
        windows = padded.unfold(1, K, 1)  # (N, T, d, K)
        return windows.permute(0, 1, 3, 2).reshape(
            x_seq.shape[0], x_seq.shape[1], K * self.d_timestep
        )

    def _encode_flat(self, x_flat: Tensor) -> Tensor:
        """Encode flat delay-line features (M, K*d) → activities (M, n)."""
        return self.activation(self._gain * (x_flat @ self.encoders.T) + self.bias)

    def encode_sequence(self, x_seq: Tensor) -> Tensor:
        """Encode sequence to mean-pooled activities. (N, T, d) → (N, n)."""
        delay = self._delay_features(x_seq)  # (N, T, K*d)
        N, T, Kd = delay.shape
        acts = self._encode_flat(delay.reshape(N * T, Kd))  # (N*T, n)
        return acts.reshape(N, T, -1).mean(dim=1)  # (N, n)

    def forward(self, x_seq: Tensor) -> Tensor:
        """Forward pass: encode sequence → decode. (N, T, d) → (N, d_out)."""
        return self.encode_sequence(x_seq) @ self.decoders

    @torch.no_grad()
    def fit(self, x_seq: Tensor, targets: Tensor, solver: str = "tikhonov", **kwargs) -> None:
        """Batch-fit decoders from sequences and targets.

        Args:
            x_seq:   (N, T, d) input sequences.
            targets: (N, d_out) target values.
            solver:  Solver name from ``leenef.solvers``.
        """
        A = self.encode_sequence(x_seq)
        D = solve_decoders(A, targets, method=solver, **kwargs)
        self.decoders.data.copy_(D)

    @torch.no_grad()
    def continuous_fit(self, x_seq: Tensor, targets: Tensor, alpha: float = 1e-2) -> None:
        """Accumulate sequences and update decoders via Woodbury updates.

        Args:
            x_seq:   (N, T, d) input sequences.
            targets: (N, d_out) target values.
            alpha:   Fixed Tikhonov regularisation strength.
        """
        A = self.encode_sequence(x_seq)  # (N, n)
        ata = A.T @ A
        aty = A.T @ targets

        if not hasattr(self, "_ata") or self._ata is None:
            self.register_buffer("_ata", ata)
            self.register_buffer("_aty", aty)
        else:
            self._ata.add_(ata)
            self._aty.add_(aty)

        if not hasattr(self, "_M_inv") or self._M_inv is None:
            n = A.shape[1]
            self._M_inv = torch.eye(n, device=A.device, dtype=A.dtype) / alpha
            self._woodbury_alpha = alpha

        k, n = A.shape
        if k >= n:
            M = self._ata.clone()
            M.diagonal().add_(alpha)
            self._M_inv = torch.linalg.inv(M)
        else:
            V = A @ self._M_inv
            C = torch.eye(k, device=A.device, dtype=A.dtype)
            C.addmm_(A, V.T)
            C_inv_V = torch.linalg.solve(C, V)
            self._M_inv.sub_(V.T @ C_inv_V)

        self.decoders.data.copy_(self._M_inv @ self._aty)

    @torch.no_grad()
    def refresh_inverse(self, alpha: float | None = None) -> None:
        """Recompute the inverse exactly from accumulated statistics."""
        if not hasattr(self, "_ata") or self._ata is None:
            raise RuntimeError("No accumulated statistics — call continuous_fit() first")
        if alpha is None:
            alpha = getattr(self, "_woodbury_alpha", 1e-2)
        M = self._ata.clone()
        M.diagonal().add_(alpha)
        self._M_inv = torch.linalg.inv(M)
        self._woodbury_alpha = alpha
        self.decoders.data.copy_(self._M_inv @ self._aty)

    def reset_continuous(self) -> None:
        """Clear all continuous-learning state."""
        if hasattr(self, "_ata"):
            self._ata = None
        if hasattr(self, "_aty"):
            self._aty = None
        if hasattr(self, "_M_inv"):
            self._M_inv = None
        if hasattr(self, "_woodbury_alpha"):
            del self._woodbury_alpha
