"""NEFLayer — a single NEF layer as a PyTorch Module."""

import torch
import torch.nn as nn
from torch import Tensor

from .activations import make_activation
from .encoders import make_encoders
from .solvers import solve_decoders, solve_from_normal_equations


def _make_gain(
    spec: float | tuple[float, float] | Tensor | str,
    n_neurons: int,
    rng: torch.Generator | None = None,
    *,
    selected_centers: Tensor | None = None,
    all_centers: Tensor | None = None,
    k_density: int = 10,
) -> Tensor:
    """Build a per-neuron gain vector from a gain specification.

    Args:
        spec: ``float`` for uniform gain, ``(low, high)`` tuple for
              per-neuron uniform sampling, ``Tensor(n_neurons,)`` for
              explicit gains, or ``"data_driven"`` for density-adaptive
              gains.
        selected_centers: ``(n_neurons, dim)`` — the center point assigned
              to each neuron.  Required when ``spec="data_driven"``.
        all_centers: ``(N, dim)`` — full training data for density
              estimation.  Required when ``spec="data_driven"``.
        k_density: number of neighbors for density estimation when
              ``spec="data_driven"``.
    """
    if isinstance(spec, Tensor):
        if spec.shape != (n_neurons,):
            raise ValueError(
                f"gain tensor must have shape ({n_neurons},), got {tuple(spec.shape)}"
            )
        return spec.float()
    if spec == "data_driven":
        if selected_centers is None or all_centers is None:
            raise ValueError(
                "gain='data_driven' requires both selected_centers and "
                "all_centers (pass centers=train_data to NEFLayer)"
            )
        return _density_adaptive_gain(selected_centers, all_centers, k_density)
    if isinstance(spec, tuple):
        lo, hi = spec
        return lo + (hi - lo) * torch.rand(n_neurons, generator=rng)
    return torch.full((n_neurons,), spec, dtype=torch.float32)


def _density_adaptive_gain(
    selected_centers: Tensor,
    all_centers: Tensor,
    k: int = 10,
    gain_range: tuple[float, float] = (0.5, 2.0),
) -> Tensor:
    """Per-neuron gains inversely proportional to local data density.

    Dense regions → lower gain (broad tuning curves); sparse regions →
    higher gain (sharp tuning curves).  Density is estimated by the mean
    distance to the ``k`` nearest neighbors in the training data.
    """
    n = selected_centers.shape[0]
    k_actual = min(k, all_centers.shape[0] - 1)

    # Process in chunks to bound memory
    chunk_size = 500
    mean_dists = torch.empty(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = torch.cdist(selected_centers[start:end].float(), all_centers.float())
        topk_dists, _ = dists.topk(k_actual + 1, dim=1, largest=False)
        mean_dists[start:end] = topk_dists[:, 1 : k_actual + 1].mean(dim=1)

    lo, hi = gain_range
    d_min, d_max = mean_dists.min(), mean_dists.max()
    if d_max > d_min:
        t = (mean_dists - d_min) / (d_max - d_min)
    else:
        t = torch.full_like(mean_dists, 0.5)
    return lo + (hi - lo) * t


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
        gain: float | tuple[float, float] | Tensor | str = (0.5, 2.0),
        rng: torch.Generator | None = None,
        centers: Tensor | None = None,
        encoder_kwargs: dict | None = None,
        **act_kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_neurons = n_neurons
        self.d_out = d_out
        self._rng = rng

        # 1. Create encoders — some strategies return (encoders, centers) tuples.
        #    Move to CPU; buffers are relocated to the target device via .to().
        result = make_encoders(
            n_neurons, d_in, strategy=encoder_strategy, rng=rng, **(encoder_kwargs or {})
        )
        if isinstance(result, tuple):
            enc, encoder_centers = result[0].cpu(), result[1].cpu()
        else:
            enc = result.cpu()
            encoder_centers = None

        # 2. Resolve per-neuron center points for bias and gain computation.
        #    Move to CPU so bias arithmetic matches encoder device; buffers
        #    are relocated to the target device later via .to(device).
        if encoder_centers is not None:
            selected_centers = encoder_centers.cpu()
        elif centers is not None:
            idx = torch.randint(len(centers), (n_neurons,), generator=rng)
            selected_centers = centers[idx].cpu()
        else:
            selected_centers = None

        # 3. Per-neuron gain — may depend on resolved centers
        self.register_buffer(
            "_gain",
            _make_gain(
                gain,
                n_neurons,
                rng,
                selected_centers=selected_centers,
                all_centers=centers.cpu() if centers is not None else None,
            ),
        )

        # 4. Biases from data-driven centers or random
        if selected_centers is not None:
            bias = -self._gain * (selected_centers.float() * enc).sum(dim=1)
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

    # ── Incremental / online fit ──────────────────────────────────────

    @torch.no_grad()
    def partial_fit(self, x: Tensor, targets: Tensor) -> None:
        """Accumulate normal-equation statistics for an incremental solve.

        Call repeatedly with data chunks, then call :meth:`solve_accumulated`
        once to set decoders.  This avoids materialising the full activity
        matrix when data arrives in batches or streams.

        Args:
            x:       (N, d_in) input data batch
            targets: (N, d_out) target batch
        """
        if x.shape[0] != targets.shape[0]:
            raise ValueError(
                f"x and targets must have same number of samples, "
                f"got {x.shape[0]} vs {targets.shape[0]}"
            )
        A = self.encode(x)
        ata = A.T @ A
        aty = A.T @ targets

        if not hasattr(self, "_ata") or self._ata is None:
            self.register_buffer("_ata", ata)
            self.register_buffer("_aty", aty)
        else:
            self._ata.add_(ata)
            self._aty.add_(aty)

    @torch.no_grad()
    def solve_accumulated(self, alpha: float = 1e-2) -> None:
        """Solve decoders from accumulated normal-equation statistics.

        Must be called after one or more :meth:`partial_fit` calls.

        Args:
            alpha: Tikhonov regularisation strength.
        """
        if not hasattr(self, "_ata") or self._ata is None:
            raise RuntimeError("No accumulated statistics — call partial_fit() first")
        D = solve_from_normal_equations(self._ata, self._aty, alpha=alpha)
        self.decoders.data.copy_(D)

    def reset_accumulators(self) -> None:
        """Clear accumulated normal-equation statistics."""
        if hasattr(self, "_ata"):
            self._ata = None
        if hasattr(self, "_aty"):
            self._aty = None

    # ── Continuous fit (Woodbury rank-k updates) ──────────────────────

    @torch.no_grad()
    def continuous_fit(self, x: Tensor, targets: Tensor, alpha: float = 1e-2) -> None:
        """Accumulate new data and update decoders via rank-k Woodbury update.

        On the first call (or after :meth:`reset_continuous`), performs an
        initial solve and caches the system inverse.  Subsequent calls apply
        the Sherman-Morrison-Woodbury identity for *O(n²k)* updates (where
        *k* is the batch size) instead of the *O(n³)* full re-solve.

        The method also maintains ``_ata`` / ``_aty`` accumulators so that
        :meth:`refresh_inverse` can periodically perform an exact re-solve
        to bound accumulated floating-point drift.

        Args:
            x:       (N, d_in) input data batch.
            targets: (N, d_out) target batch.
            alpha:   Fixed Tikhonov regularisation strength (not trace-scaled,
                     to keep the cached inverse consistent across updates).
        """
        if x.shape[0] != targets.shape[0]:
            raise ValueError(
                f"x and targets must have same number of samples, "
                f"got {x.shape[0]} vs {targets.shape[0]}"
            )
        A = self.encode(x)
        self.continuous_fit_encoded(A, targets, alpha=alpha)

    @torch.no_grad()
    def continuous_fit_encoded(
        self, activities: Tensor, targets: Tensor, alpha: float = 1e-2
    ) -> None:
        """Like :meth:`continuous_fit` but from pre-encoded activities.

        Useful when the activity matrix has already been computed externally
        (e.g. after temporal pooling in a streaming classifier).

        Args:
            activities: (N, n_neurons) pre-encoded neuron activities.
            targets:    (N, d_out) target batch.
            alpha:      Fixed Tikhonov regularisation strength.
        """
        if activities.shape[0] != targets.shape[0]:
            raise ValueError(
                f"activities and targets must have same number of samples, "
                f"got {activities.shape[0]} vs {targets.shape[0]}"
            )
        A = activities  # (k, n)
        ata = A.T @ A
        aty = A.T @ targets

        # Accumulate normal equations (for refresh_inverse / solve_accumulated)
        if not hasattr(self, "_ata") or self._ata is None:
            self.register_buffer("_ata", ata)
            self.register_buffer("_aty", aty)
        else:
            self._ata.add_(ata)
            self._aty.add_(aty)

        # Woodbury inverse is maintained in float64 to prevent drift
        # across many rank-k updates in float32.
        if not hasattr(self, "_M_inv") or self._M_inv is None:
            n = A.shape[1]
            self._M_inv = torch.eye(n, device=A.device, dtype=torch.float64) / alpha
            self._woodbury_alpha = alpha

        k, n = A.shape
        if k >= n:
            M = self._ata.double()
            M.diagonal().add_(alpha)
            self._M_inv = torch.linalg.inv(M)
        else:
            A_d = A.double()
            V = A_d @ self._M_inv  # (k, n) float64
            C = torch.eye(k, device=A.device, dtype=torch.float64)
            C.addmm_(A_d, V.T)  # I_k + A M⁻¹ Aᵀ  (k, k)
            C_inv_V = torch.linalg.solve(C, V)  # (k, n)
            self._M_inv.sub_(V.T @ C_inv_V)  # in-place (n, n)

        self.decoders.data.copy_((self._M_inv @ self._aty.double()).to(self.decoders.dtype))

    @torch.no_grad()
    def refresh_inverse(self, alpha: float | None = None) -> None:
        """Recompute the Woodbury inverse exactly from accumulated statistics.

        Call periodically during long streaming runs to bound numerical
        drift.  Requires at least one prior :meth:`partial_fit` or
        :meth:`continuous_fit` call.

        Args:
            alpha: Tikhonov α.  Defaults to the value used in the last
                   :meth:`continuous_fit` call.
        """
        if not hasattr(self, "_ata") or self._ata is None:
            raise RuntimeError("No accumulated statistics — call continuous_fit() first")
        if alpha is None:
            if hasattr(self, "_woodbury_alpha"):
                alpha = self._woodbury_alpha
            else:
                alpha = 1e-2
        M = self._ata.double()
        M.diagonal().add_(alpha)
        self._M_inv = torch.linalg.inv(M)
        self._woodbury_alpha = alpha
        self.decoders.data.copy_((self._M_inv @ self._aty.double()).to(self.decoders.dtype))

    def reset_continuous(self) -> None:
        """Clear Woodbury inverse, accumulators, and decoders."""
        self.reset_accumulators()
        if hasattr(self, "_M_inv"):
            self._M_inv = None
        if hasattr(self, "_woodbury_alpha"):
            del self._woodbury_alpha
