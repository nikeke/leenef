"""Random encoder generation strategies for NEF layers."""

import torch
import torch.nn as nn
from torch import Tensor


def uniform_hypersphere(n_neurons: int, dim: int, *, rng: torch.Generator | None = None) -> Tensor:
    """Sample encoder vectors uniformly from the unit hypersphere."""
    e = torch.randn(n_neurons, dim, generator=rng)
    return e / e.norm(dim=1, keepdim=True)


def gaussian(n_neurons: int, dim: int, *, rng: torch.Generator | None = None) -> Tensor:
    """Sample encoder vectors from a standard normal (not normalised)."""
    return torch.randn(n_neurons, dim, generator=rng)


def sparse(n_neurons: int, dim: int, *, sparsity: float = 0.9,
           rng: torch.Generator | None = None) -> Tensor:
    """Sparse random encoders — each neuron sees only (1-sparsity) of inputs."""
    e = torch.randn(n_neurons, dim, generator=rng)
    mask = torch.rand(n_neurons, dim, generator=rng) >= sparsity
    return e * mask


ENCODER_STRATEGIES = {
    "hypersphere": uniform_hypersphere,
    "gaussian": gaussian,
    "sparse": sparse,
}


def make_encoders(n_neurons: int, dim: int, strategy: str = "hypersphere",
                  **kwargs) -> Tensor:
    """Create encoders using a named strategy."""
    return ENCODER_STRATEGIES[strategy](n_neurons, dim, **kwargs)
