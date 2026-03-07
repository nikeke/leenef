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


def data_sample(n_neurons: int, dim: int, *, data: Tensor,
                noise: float = 0.1, center: bool = True,
                rng: torch.Generator | None = None) -> Tensor:
    """Encoder vectors sampled from training data.

    Selects *n_neurons* random training examples, optionally centres
    the data (subtracts the mean), adds Gaussian noise, and normalises
    to unit norm.  Aligns the encoder distribution with the true input
    distribution.
    """
    src = data.float()
    if center:
        src = src - src.mean(0)
    idx = torch.randint(len(src), (n_neurons,), generator=rng)
    e = src[idx].clone()
    if noise > 0:
        e = e + noise * torch.randn(e.shape, generator=rng)
    return e / (e.norm(dim=1, keepdim=True) + 1e-8)


def data_diff(n_neurons: int, dim: int, *, data: Tensor,
              noise: float = 0.0,
              rng: torch.Generator | None = None) -> Tensor:
    """Encoder vectors from differences between random data pairs.

    Each encoder is the normalised difference between two random
    training examples: ``(x_i - x_j) / ||x_i - x_j||``.
    This naturally captures the most informative input-space
    directions without requiring explicit centering.
    """
    src = data.float()
    idx_a = torch.randint(len(src), (n_neurons,), generator=rng)
    idx_b = torch.randint(len(src), (n_neurons,), generator=rng)
    e = src[idx_a] - src[idx_b]
    if noise > 0:
        e = e + noise * torch.randn(e.shape, generator=rng)
    return e / (e.norm(dim=1, keepdim=True) + 1e-8)


ENCODER_STRATEGIES = {
    "hypersphere": uniform_hypersphere,
    "gaussian": gaussian,
    "sparse": sparse,
    "data": data_sample,
    "data_diff": data_diff,
}


def make_encoders(n_neurons: int, dim: int, strategy: str = "hypersphere",
                  **kwargs) -> Tensor:
    """Create encoders using a named strategy."""
    return ENCODER_STRATEGIES[strategy](n_neurons, dim, **kwargs)
