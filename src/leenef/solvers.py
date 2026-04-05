"""Decoder solvers for NEF layers — map activities → targets."""

import torch
from torch import Tensor


def lstsq(activities: Tensor, targets: Tensor, **_kwargs) -> Tensor:
    """Solve D such that activities @ D ≈ targets via torch.linalg.lstsq.

    Args:
        activities: (N, n_neurons)
        targets:    (N, d_out)
    Returns:
        decoders:   (n_neurons, d_out)
    """
    result = torch.linalg.lstsq(activities, targets)
    return result.solution


def tikhonov(activities: Tensor, targets: Tensor, alpha: float = 1e-2) -> Tensor:
    """Tikhonov-regularised least-squares (L2 penalty on decoder norms).

    Solves: (A^T A + reg * I) D = A^T targets

    Regularisation is scaled to the matrix norm for numerical stability:
    reg = alpha * trace(A^T A) / n_neurons, floored at alpha.
    """
    A = activities
    n = A.shape[1]
    ATA = A.T @ A
    reg = alpha * torch.trace(ATA) / n
    ATA.diagonal().add_(reg.clamp(min=alpha))
    return torch.linalg.solve(ATA, A.T @ targets)


def normal_equations(activities: Tensor, targets: Tensor, alpha: float = 1e-2) -> Tensor:
    """L2-regularised normal equations via Cholesky — fast for large N, moderate n_neurons."""
    A = activities
    ATA = A.T @ A
    # Scale regularisation to the matrix norm for numerical stability
    reg = alpha * torch.trace(ATA) / ATA.shape[0]
    ATA.diagonal().add_(reg.clamp(min=alpha))
    L = torch.linalg.cholesky(ATA)
    return torch.cholesky_solve(A.T @ targets, L)


def solve_from_normal_equations(ATA: Tensor, ATY: Tensor, alpha: float = 1e-2) -> Tensor:
    """Solve for decoders from precomputed normal equations.

    Useful when activities are accumulated incrementally (e.g. over
    timesteps in a recurrent layer) to avoid materialising the full
    activity matrix.

    Args:
        ATA:   (n_neurons, n_neurons) — accumulated A^T A.
        ATY:   (n_neurons, d_out) — accumulated A^T targets.
        alpha: Tikhonov regularisation strength.
    Returns:
        decoders: (n_neurons, d_out)
    """
    n = ATA.shape[0]
    ATA = ATA.clone()
    reg = alpha * torch.trace(ATA) / n
    ATA.diagonal().add_(reg.clamp(min=alpha))
    return torch.linalg.solve(ATA, ATY)


SOLVERS = {
    "lstsq": lstsq,
    "tikhonov": tikhonov,
    "cholesky": normal_equations,
}


def solve_decoders(
    activities: Tensor, targets: Tensor, method: str = "tikhonov", **kwargs
) -> Tensor:
    """Solve for decoders using a named method."""
    if method not in SOLVERS:
        raise ValueError(f"Unknown solver {method!r}. Available: {sorted(SOLVERS)}")
    return SOLVERS[method](activities, targets, **kwargs)
