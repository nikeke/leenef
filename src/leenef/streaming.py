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
from .solvers import (
    DEFAULT_ALPHA,
    DEFAULT_FIXED_ALPHA,
    AlphaPolicy,
    inverse_from_normal_equations,
    solve_decoders,
    solve_from_normal_equations,
    woodbury_decoders,
    woodbury_initial_inverse,
    woodbury_update_inverse_,
)


def _normal_equation_inputs(
    activities: Tensor,
    targets: Tensor,
    *,
    stats_dtype: str | torch.dtype | None,
) -> tuple[Tensor, Tensor]:
    if stats_dtype is None or stats_dtype == "preserve":
        A = (
            activities.float()
            if activities.dtype in (torch.float16, torch.bfloat16)
            else activities
        )
        T = targets.float() if targets.dtype in (torch.float16, torch.bfloat16) else targets
        return A, T
    if stats_dtype in ("float32", torch.float32):
        return activities.float(), targets.float()
    if stats_dtype in ("float64", torch.float64):
        return activities.double(), targets.double()
    raise ValueError(
        f"stats_dtype must be one of None, 'preserve', 'float32', or 'float64'; "
        f"got {stats_dtype!r}"
    )


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
        centers:           Optional (N, T, d) training sequences from which
                           delay-line windows are sampled for data-driven
                           biases.
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

        # Data-driven biases from sampled delay-line windows of the training
        # sequences. Sampling the flattened window indices directly avoids
        # materializing the full (N, T, K*d) tensor for large sequential tasks.
        if centers is not None:
            total_windows = centers.shape[0] * centers.shape[1]
            idx = torch.randint(total_windows, (n_neurons,), generator=rng)
            sampled = self._flat_delay_windows(centers, idx)
            bias = -self._gain * (sampled.float() * enc).sum(dim=1)
        else:
            bias = torch.randn(n_neurons, generator=rng)
        self.register_buffer("bias", bias)

        self.decoders = nn.Parameter(torch.zeros(n_neurons, d_out), requires_grad=False)

        # Woodbury state — initialized lazily by continuous_fit

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

    def _flat_delay_windows(self, x_seq: Tensor, flat_idx: Tensor) -> Tensor:
        """Gather selected delay-line windows by flattened ``(sample, timestep)`` index."""
        K = self.window_size
        _, n_steps, _ = x_seq.shape
        flat_idx = flat_idx.to(x_seq.device)
        sample_idx = torch.div(flat_idx, n_steps, rounding_mode="floor")
        step_idx = torch.remainder(flat_idx, n_steps)
        padded = F.pad(x_seq, (0, 0, K - 1, 0))
        offsets = torch.arange(K, device=x_seq.device)
        windows = padded[sample_idx.unsqueeze(1), step_idx.unsqueeze(1) + offsets]
        return windows.reshape(len(flat_idx), K * self.d_timestep)

    def _encode_flat(self, x_flat: Tensor) -> Tensor:
        """Encode flat delay-line features (M, K*d) → activities (M, n)."""
        return self.activation(self._gain * (x_flat @ self.encoders.T) + self.bias)

    def _encoded_sequence_chunks(self, x_seq: Tensor, max_tokens: int = 50_000):
        """Yield ``(start, end, pooled)`` chunks of pooled activities.

        Streaming variant of :meth:`encode_sequence` that avoids
        materializing the full ``(N, n_neurons)`` matrix at once.  Useful
        for accumulating ``AᵀA`` / ``AᵀY`` without large peak memory.
        """
        N, T, _ = x_seq.shape
        chunk = max(1, max_tokens // T)
        for i in range(0, N, chunk):
            d_chunk = self._delay_features(x_seq[i : i + chunk])
            Nc = d_chunk.shape[0]
            Kd = d_chunk.shape[2]
            a = self._encode_flat(d_chunk.reshape(Nc * T, Kd))
            yield i, i + Nc, a.reshape(Nc, T, -1).mean(dim=1)

    def encode_sequence(self, x_seq: Tensor, max_tokens: int = 50_000) -> Tensor:
        """Encode sequence to mean-pooled activities. (N, T, d) → (N, n).

        Processes in chunks when N*T exceeds *max_tokens* to limit peak
        memory (the intermediate activity tensor is (N*T, n_neurons)).
        """
        N, T, _ = x_seq.shape
        total = N * T
        if total <= max_tokens:
            delay = self._delay_features(x_seq)  # (N, T, K*d)
            _, _, Kd = delay.shape
            acts = self._encode_flat(delay.reshape(total, Kd))
            return acts.reshape(N, T, -1).mean(dim=1)
        return torch.cat(
            [pooled for _, _, pooled in self._encoded_sequence_chunks(x_seq, max_tokens)],
            dim=0,
        )

    def forward(self, x_seq: Tensor) -> Tensor:
        """Forward pass: encode sequence → decode. (N, T, d) → (N, d_out)."""
        return self.encode_sequence(x_seq) @ self.decoders

    @torch.no_grad()
    def fit(
        self,
        x_seq: Tensor,
        targets: Tensor,
        solver: str = "tikhonov",
        max_tokens: int = 50_000,
        **kwargs,
    ) -> None:
        """Batch-fit decoders from sequences and targets.

        Args:
            x_seq:   (N, T, d) input sequences.
            targets: (N, d_out) target values.
            solver:  Solver name from ``leenef.solvers``.
        """
        A = self.encode_sequence(x_seq, max_tokens=max_tokens)
        D = solve_decoders(A, targets, method=solver, **kwargs)
        self.decoders.data.copy_(D)

    @torch.no_grad()
    def continuous_fit(
        self,
        x_seq: Tensor,
        targets: Tensor,
        *,
        fixed_alpha: float = DEFAULT_FIXED_ALPHA,
        max_tokens: int = 50_000,
        stats_dtype: str | torch.dtype | None = None,
    ) -> None:
        """Accumulate sequences and update decoders via Woodbury updates.

        Args:
            x_seq:   (N, T, d) input sequences.
            targets: (N, d_out) target values.
            fixed_alpha: Fixed Tikhonov regularisation strength.
        """
        A = self.encode_sequence(x_seq, max_tokens=max_tokens)  # (N, n)
        # Match :meth:`accumulate`: float32 GEMM, float64 running stats.
        # Keeping the dtypes aligned across the two paths is what lets
        # H3 (woodbury vs accumulate vs batch_acc) be a real equivalence
        # check rather than a noise floor comparison.
        A_stats, T_stats = _normal_equation_inputs(A, targets, stats_dtype=stats_dtype)
        ata = (A_stats.T @ A_stats).double()
        aty = (A_stats.T @ T_stats).double()

        if not hasattr(self, "_ata") or self._ata is None:
            self.register_buffer("_ata", ata)
            self.register_buffer("_aty", aty)
        else:
            self._ata.add_(ata)
            self._aty.add_(aty)

        if not hasattr(self, "_M_inv") or self._M_inv is None:
            n = A.shape[1]
            self._M_inv = woodbury_initial_inverse(
                n, fixed_alpha=fixed_alpha, device=A.device, dtype=torch.float64
            )
            self._woodbury_alpha = fixed_alpha
        elif self._woodbury_alpha != fixed_alpha:
            raise ValueError(
                f"continuous_fit: fixed_alpha changed mid-stream "
                f"({self._woodbury_alpha} -> {fixed_alpha}); the cached inverse is "
                f"invalid for the new regularizer. Call refresh_inverse() or "
                f"reset_continuous() before changing fixed_alpha."
            )

        k, n = A.shape
        if k >= n:
            self._M_inv = inverse_from_normal_equations(self._ata, alpha=(fixed_alpha, "fixed"))
        else:
            woodbury_update_inverse_(self._M_inv, A)

        self.decoders.data.copy_(
            woodbury_decoders(self._M_inv, self._aty, dtype=self.decoders.dtype)
        )

    def _accumulate_encoded(
        self,
        A: Tensor,
        targets: Tensor,
        *,
        stats_dtype: str | torch.dtype | None = None,
    ) -> None:
        """Accumulate ``AᵀA`` / ``AᵀY`` from an encoded chunk.

        Float32 GEMMs, float64 running stats — see :meth:`accumulate`
        for the rationale.
        """
        A_stats, T_stats = _normal_equation_inputs(A, targets, stats_dtype=stats_dtype)
        ata = (A_stats.T @ A_stats).double()
        aty = (A_stats.T @ T_stats).double()
        if not hasattr(self, "_ata") or self._ata is None:
            self.register_buffer("_ata", ata)
            self.register_buffer("_aty", aty)
        else:
            self._ata.add_(ata)
            self._aty.add_(aty)

    @torch.no_grad()
    def accumulate(
        self,
        x_seq: Tensor,
        targets: Tensor,
        max_tokens: int = 50_000,
        chunk_stats: bool = False,
        stats_dtype: str | torch.dtype | None = None,
    ) -> None:
        """Accumulate normal-equation statistics without solving.

        GPU-friendly alternative to :meth:`continuous_fit`.  Computes
        AᵀA and AᵀY matmuls in float32 (fast on GPU), then promotes
        the n×n / n×d_out results to float64 for accumulation.  Avoids
        maintaining the expensive n×n Woodbury inverse.
        Call :meth:`solve` when you want updated decoders.

        Args:
            x_seq:        (N, T, d) input sequences.
            targets:      (N, d_out) target values.
            max_tokens:   Encoding chunk size in N*T tokens.
            chunk_stats:  If True, accumulate AᵀA / AᵀY chunk-by-chunk
                without ever materializing the full ``(N, n_neurons)``
                pooled matrix.  Slightly slower per call but caps peak
                memory at ``chunk × n_neurons`` instead of ``N × n_neurons``.
                Recommended for large n_neurons.
        """
        if chunk_stats:
            for s, e, A in self._encoded_sequence_chunks(x_seq, max_tokens):
                self._accumulate_encoded(A, targets[s:e], stats_dtype=stats_dtype)
            return
        A = self.encode_sequence(x_seq, max_tokens=max_tokens)  # (N, n) — float32
        self._accumulate_encoded(A, targets, stats_dtype=stats_dtype)

    @torch.no_grad()
    def solve(
        self,
        *,
        alpha: AlphaPolicy = DEFAULT_ALPHA,
    ) -> None:
        """Solve decoders from accumulated statistics.

        Defaults to the project trace-scaled ridge policy.  Pass
        ``alpha=(value, "fixed")`` only when matching a fixed-alpha Woodbury
        path or an already-computed trace-scaled effective alpha.
        The solve runs in the accumulator dtype (float64 when fed by
        :meth:`accumulate`).

        Must be called after one or more :meth:`accumulate` calls
        (or :meth:`continuous_fit` calls — they share the same AᵀA/AᵀY).

        Args:
            alpha: Explicit ``(value, mode)`` ridge policy.
        """
        if not hasattr(self, "_ata") or self._ata is None:
            raise RuntimeError("No accumulated statistics — call accumulate() first")
        D = solve_from_normal_equations(self._ata, self._aty, alpha=alpha)
        self.decoders.data.copy_(D.to(self.decoders.dtype))

    @torch.no_grad()
    def refresh_inverse(self, fixed_alpha: float | None = None) -> None:
        """Recompute the inverse exactly from accumulated statistics."""
        if not hasattr(self, "_ata") or self._ata is None:
            raise RuntimeError("No accumulated statistics — call continuous_fit() first")
        if fixed_alpha is None:
            if not hasattr(self, "_woodbury_alpha"):
                fixed_alpha = DEFAULT_FIXED_ALPHA
            else:
                fixed_alpha = self._woodbury_alpha
        self._M_inv = inverse_from_normal_equations(self._ata, alpha=(fixed_alpha, "fixed"))
        self._woodbury_alpha = fixed_alpha
        self.decoders.data.copy_(
            woodbury_decoders(self._M_inv, self._aty, dtype=self.decoders.dtype)
        )

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
