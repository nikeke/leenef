"""Decoder solvers and regularization policy for NEF layers.

Alpha handling is deliberately centralized here.  Regularized solves default
to trace-scaled ridge (``alpha * trace(A^T A) / n``), and fixed-alpha
Woodbury paths use ``fixed_alpha`` in their helper names/signatures so they
cannot be confused with trace-scaled ridge solves.
"""

from typing import Literal, TypeAlias

import torch
from torch import Tensor

AlphaMode = Literal["trace", "fixed"]
AlphaPolicy: TypeAlias = tuple[float, AlphaMode]
DEFAULT_ALPHA: AlphaPolicy = (1e-2, "trace")
DEFAULT_FIXED_ALPHA = 1e-2


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    return alpha


def _validate_policy_mode(mode: AlphaMode) -> AlphaMode:
    if mode not in ("trace", "fixed"):
        raise ValueError(f"alpha policy mode must be 'trace' or 'fixed', got {mode!r}")
    return mode


def alpha_policy(alpha: AlphaPolicy = DEFAULT_ALPHA) -> AlphaPolicy:
    """Validate and normalize an explicit ``(value, mode)`` alpha policy."""

    if not isinstance(alpha, tuple) or len(alpha) != 2:
        raise ValueError(f"alpha must be an explicit (value, mode) pair, got {alpha!r}")
    alpha_value, mode = alpha
    return _validate_alpha(alpha_value), _validate_policy_mode(mode)


def _as_scalar_like(value: float, like: Tensor) -> Tensor:
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def regularization_strength(
    normal_matrix: Tensor,
    *,
    alpha: AlphaPolicy = DEFAULT_ALPHA,
) -> Tensor:
    """Return the diagonal ridge regularizer for ``normal_matrix``.

    ``(value, 'trace')`` implements the project default
    ``value * trace(A^T A) / n``, floored at ``value``.  ``(value, 'fixed')``
    is reserved for algorithms whose state is explicitly parameterized by a
    fixed inverse regularizer (for example Woodbury/RLS updates).
    """

    alpha_value, mode = alpha_policy(alpha)
    alpha_t = _as_scalar_like(alpha_value, normal_matrix)
    if mode == "fixed":
        return alpha_t

    trace_scaled = alpha_t * torch.trace(normal_matrix) / normal_matrix.shape[0]
    return torch.maximum(trace_scaled, alpha_t)


def regularized_normal_matrix(
    normal_matrix: Tensor,
    *,
    alpha: AlphaPolicy = DEFAULT_ALPHA,
    copy: bool = True,
) -> Tensor:
    """Add the selected ridge regularizer to a normal matrix diagonal."""

    matrix = normal_matrix.clone() if copy else normal_matrix
    matrix.diagonal().add_(regularization_strength(matrix, alpha=alpha))
    return matrix


def solve_spd(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Solve a symmetric positive-definite system with a safe fallback."""

    rhs_was_vector = rhs.ndim == 1
    rhs_matrix = rhs.unsqueeze(1) if rhs_was_vector else rhs
    chol, info = torch.linalg.cholesky_ex(matrix)
    if int(info.max().item()) == 0:
        out = torch.cholesky_solve(rhs_matrix, chol)
    else:
        out = torch.linalg.solve(matrix, rhs_matrix)
    return out.squeeze(1) if rhs_was_vector else out


def inverse_spd(matrix: Tensor) -> Tensor:
    """Invert an SPD matrix through its Cholesky factor with LU fallback."""

    chol, info = torch.linalg.cholesky_ex(matrix)
    if int(info.max().item()) == 0:
        return torch.cholesky_inverse(chol)
    return torch.linalg.inv(matrix)


def solve_regularized_normal_equations(
    normal_matrix: Tensor,
    cross_matrix: Tensor,
    *,
    alpha: AlphaPolicy = DEFAULT_ALPHA,
) -> Tensor:
    """Solve ``(A^T A + reg I)D = A^T Y`` under an explicit alpha policy."""

    matrix = regularized_normal_matrix(normal_matrix, alpha=alpha, copy=True)
    return solve_spd(matrix, cross_matrix)


def inverse_from_normal_equations(
    normal_matrix: Tensor,
    *,
    alpha: AlphaPolicy,
) -> Tensor:
    """Return ``(A^T A + reg I)^-1`` under an explicit alpha policy."""

    matrix = regularized_normal_matrix(normal_matrix, alpha=alpha, copy=True)
    return inverse_spd(matrix)


def downdate_inverse(inverse: Tensor, keep: Tensor, drop: Tensor) -> Tensor:
    """Return the inverse of the kept principal submatrix using a Schur complement.

    ``inverse`` is the inverse of the full regularized system; ``keep`` and
    ``drop`` partition its indices. This is exact only at fixed regularization.
    """
    kept_inverse = inverse[keep[:, None], keep]
    if drop.numel() > 0:
        p_ks = inverse[keep[:, None], drop]
        p_ss = inverse[drop[:, None], drop]
        kept_inverse = kept_inverse - p_ks @ solve_spd(p_ss, p_ks.T)
    return 0.5 * (kept_inverse + kept_inverse.mT)


def woodbury_initial_inverse(
    n_neurons: int,
    *,
    fixed_alpha: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Initial ``(fixed_alpha I)^-1`` for fixed-alpha Woodbury updates."""

    fixed_alpha = _validate_alpha(fixed_alpha)
    return torch.eye(n_neurons, device=device, dtype=dtype) / fixed_alpha


def woodbury_update_inverse_(inverse: Tensor, activities: Tensor) -> Tensor:
    """In-place rank-k Woodbury inverse update.

    ``inverse`` must already be parameterized by the caller's chosen
    fixed/effective alpha.  This helper intentionally has no alpha argument.
    """

    A = activities.to(device=inverse.device, dtype=inverse.dtype)
    k = A.shape[0]
    V = A @ inverse
    C = torch.eye(k, device=A.device, dtype=A.dtype)
    C.addmm_(A, V.T)
    c_inv_v = solve_spd(C, V)
    inverse.sub_(V.T @ c_inv_v)
    return inverse


def woodbury_decoders(inverse: Tensor, cross_matrix: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Decode from a cached Woodbury inverse and accumulated ``A^T Y``."""

    return (inverse @ cross_matrix.to(inverse.dtype)).to(dtype)


def lstsq(activities: Tensor, targets: Tensor, **_kwargs) -> Tensor:
    """Solve D such that ``activities @ D ≈ targets`` without regularization."""

    result = torch.linalg.lstsq(activities, targets)
    return result.solution


def tikhonov(
    activities: Tensor,
    targets: Tensor,
    *,
    alpha: AlphaPolicy = DEFAULT_ALPHA,
) -> Tensor:
    """Tikhonov-regularized least squares with explicit alpha semantics."""

    A = activities
    return solve_regularized_normal_equations(
        A.T @ A,
        A.T @ targets,
        alpha=alpha,
    )


def normal_equations(
    activities: Tensor,
    targets: Tensor,
    *,
    alpha: AlphaPolicy = DEFAULT_ALPHA,
) -> Tensor:
    """Regularized normal equations from activities with explicit alpha semantics."""

    return tikhonov(activities, targets, alpha=alpha)


def solve_from_normal_equations(
    ATA: Tensor,
    ATY: Tensor,
    *,
    alpha: AlphaPolicy = DEFAULT_ALPHA,
) -> Tensor:
    """Solve for decoders from precomputed normal equations."""

    return solve_regularized_normal_equations(
        ATA,
        ATY,
        alpha=alpha,
    )


SOLVERS = {
    "lstsq": lstsq,
    "tikhonov": tikhonov,
    "cholesky": normal_equations,
}


def solve_decoders(
    activities: Tensor, targets: Tensor, method: str = "tikhonov", **kwargs
) -> Tensor:
    """Solve for decoders using a named method.

    Regularized methods default to trace-scaled ``alpha=(1e-2, "trace")``.
    Fixed regularization should normally be exposed as ``fixed_alpha`` at
    higher-level Woodbury/cached-inverse APIs.
    """

    if method not in SOLVERS:
        raise ValueError(f"Unknown solver {method!r}. Available: {sorted(SOLVERS)}")
    return SOLVERS[method](activities, targets, **kwargs)
