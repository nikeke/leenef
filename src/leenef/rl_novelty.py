"""Normal-equation novelty utilities for NEF-RL."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def regularized_normal_matrix(ata: Tensor, alpha: float, n_neurons: int | None = None) -> Tensor:
    """Return ``A^T A`` with NEF-style trace-scaled diagonal regularization."""
    if ata.dim() != 2 or ata.shape[0] != ata.shape[1]:
        raise ValueError(f"ata must be square, got shape {tuple(ata.shape)}")
    if n_neurons is None:
        n_neurons = ata.shape[0]
    matrix = ata.double().clone()
    reg = alpha * torch.trace(matrix) / n_neurons
    matrix.diagonal().add_(reg.clamp(min=alpha))
    return matrix


def leverage_scores(features: Tensor, ata: Tensor, alpha: float) -> Tensor:
    """Compute ``phi^T (A^T A + alpha I)^-1 phi`` for each feature row."""
    if features.dim() == 1:
        features = features.unsqueeze(0)
    matrix = regularized_normal_matrix(ata, alpha, features.shape[1])
    phi = features.double()
    chol, info = torch.linalg.cholesky_ex(matrix)
    if int(info.item()) == 0:
        solved = torch.cholesky_solve(phi.T, chol).T
    else:
        solved = torch.linalg.solve(matrix, phi.T).T
    scores = (phi * solved).sum(dim=1)
    return scores.clamp_min(0).float()


def novelty_scores(features: Tensor, ata: Tensor, alpha: float, transform: str = "sqrt") -> Tensor:
    """Return a monotone novelty score derived from leverage."""
    leverage = leverage_scores(features, ata, alpha)
    if transform == "leverage":
        return leverage
    if transform == "sqrt":
        return leverage.sqrt()
    if transform == "log":
        return torch.log1p(leverage)
    raise ValueError(f"unknown novelty transform {transform!r}")


def nef_fqi_action_novelty(agent, obs: np.ndarray, transform: str = "sqrt") -> Tensor:
    """Compute per-action novelty for a fitted-Q NEF agent."""
    if getattr(agent, "forget_factor", None) is None:
        raise ValueError("NEFFQIAgent action novelty requires RLS sufficient statistics")
    with torch.no_grad():
        phi = agent.features.encode(torch.from_numpy(np.asarray(obs, dtype=np.float32)).float())
    scores = [
        novelty_scores(phi, agent._ata[action], float(agent.alpha), transform=transform).squeeze(0)
        for action in range(agent.n_actions)
    ]
    return torch.stack(scores)
