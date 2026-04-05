"""Random encoder generation strategies for NEF layers."""

import torch
from torch import Tensor


def uniform_hypersphere(n_neurons: int, dim: int, *, rng: torch.Generator | None = None) -> Tensor:
    """Sample encoder vectors uniformly from the unit hypersphere."""
    e = torch.randn(n_neurons, dim, generator=rng)
    return e / e.norm(dim=1, keepdim=True)


def gaussian(n_neurons: int, dim: int, *, rng: torch.Generator | None = None) -> Tensor:
    """Sample encoder vectors from a standard normal (not normalised)."""
    return torch.randn(n_neurons, dim, generator=rng)


def sparse(
    n_neurons: int, dim: int, *, sparsity: float = 0.9, rng: torch.Generator | None = None
) -> Tensor:
    """Sparse random encoders — each neuron sees only (1-sparsity) of inputs."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    e = torch.randn(n_neurons, dim, generator=rng)
    mask = torch.rand(n_neurons, dim, generator=rng) >= sparsity
    return e * mask


def receptive_field(
    n_neurons: int,
    dim: int,
    *,
    patch_size: int = 5,
    image_shape: tuple[int, int] = (28, 28),
    rng: torch.Generator | None = None,
) -> Tensor:
    """Sparse encoders with local receptive fields for image data.

    Each neuron connects to a random ``patch_size × patch_size`` patch of the
    image.  Weights within the patch are drawn from a unit hypersphere
    (normalised to unit norm), and all other entries are zero.  This injects
    spatial locality — like a randomised convolution — while keeping the
    analytic solve unchanged.

    Following McDonnell et al. (2015), local receptive fields dramatically
    improve single-layer accuracy on image tasks (e.g. ~95% → ~99% on MNIST
    when combined with ensembling).

    Args:
        n_neurons: number of encoder vectors to generate.
        dim: flattened input dimensionality (must equal H × W or H × W × C).
        patch_size: side length of the square receptive field patch.
        image_shape: ``(H, W)`` spatial dimensions of the input image.
        rng: optional ``torch.Generator`` for reproducibility.
    """
    H, W = image_shape
    n_spatial = H * W
    if dim % n_spatial != 0:
        raise ValueError(
            f"dim={dim} must be divisible by H*W={n_spatial} (image_shape={image_shape})"
        )
    n_channels = dim // n_spatial

    if patch_size > H or patch_size > W:
        raise ValueError(f"patch_size={patch_size} exceeds image dimensions {image_shape}")

    # Sample random patch top-left corners for all neurons at once
    max_row = H - patch_size
    max_col = W - patch_size
    rows = torch.randint(0, max_row + 1, (n_neurons,), generator=rng)
    cols = torch.randint(0, max_col + 1, (n_neurons,), generator=rng)

    # Build flat index offsets for a patch_size × patch_size patch
    pr = torch.arange(patch_size)
    pc = torch.arange(patch_size)
    # (patch_size, patch_size) grid of row/col offsets
    grid_r, grid_c = torch.meshgrid(pr, pc, indexing="ij")
    grid_r = grid_r.reshape(-1)  # (patch_size²,)
    grid_c = grid_c.reshape(-1)

    # Spatial indices per neuron: (n_neurons, patch_size²)
    spatial_idx = (rows.unsqueeze(1) + grid_r) * W + (cols.unsqueeze(1) + grid_c)

    # Expand to all channels: (n_neurons, patch_size² × n_channels)
    if n_channels > 1:
        ch_offsets = torch.arange(n_channels) * n_spatial  # (C,)
        # (n_neurons, patch_size², C) → (n_neurons, patch_size² * C)
        spatial_idx = (spatial_idx.unsqueeze(2) + ch_offsets).reshape(n_neurons, -1)

    # Random weights within each patch, normalised to unit norm
    patch_dim = patch_size * patch_size * n_channels
    weights = torch.randn(n_neurons, patch_dim, generator=rng)
    weights = weights / weights.norm(dim=1, keepdim=True)

    # Scatter into full encoder matrix
    e = torch.zeros(n_neurons, dim)
    e.scatter_(1, spatial_idx, weights)
    return e


ENCODER_STRATEGIES = {
    "hypersphere": uniform_hypersphere,
    "gaussian": gaussian,
    "sparse": sparse,
    "receptive_field": receptive_field,
}


def make_encoders(n_neurons: int, dim: int, strategy: str = "hypersphere", **kwargs) -> Tensor:
    """Create encoders using a named strategy."""
    if strategy not in ENCODER_STRATEGIES:
        raise ValueError(
            f"Unknown encoder strategy {strategy!r}. Available: {sorted(ENCODER_STRATEGIES)}"
        )
    return ENCODER_STRATEGIES[strategy](n_neurons, dim, **kwargs)
