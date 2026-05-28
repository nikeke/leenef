"""Random encoder generation strategies for NEF layers."""

import torch
from torch import Tensor


def uniform_hypersphere(n_neurons: int, dim: int, *, rng: torch.Generator | None = None) -> Tensor:
    """Sample encoder vectors uniformly from the unit hypersphere."""
    e = torch.randn(n_neurons, dim, generator=rng)
    return e / e.norm(dim=1, keepdim=True)


def gaussian(n_neurons: int, dim: int, *, rng: torch.Generator | None = None) -> Tensor:
    """Sample encoder vectors from a standard normal (not normalized)."""
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
    (normalized to unit norm), and all other entries are zero.  This injects
    spatial locality — like a randomized convolution — while keeping the
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

    # Random weights within each patch, normalized to unit norm
    patch_dim = patch_size * patch_size * n_channels
    weights = torch.randn(n_neurons, patch_dim, generator=rng)
    weights = weights / weights.norm(dim=1, keepdim=True)

    # Scatter into full encoder matrix
    e = torch.zeros(n_neurons, dim)
    e.scatter_(1, spatial_idx, weights)
    return e


def whitened(
    n_neurons: int,
    dim: int,
    *,
    train_data: Tensor,
    variance_ratio: float = 0.95,
    rng: torch.Generator | None = None,
) -> Tensor:
    """Random encoders projected into the principal subspace of the data.

    Computes PCA of the training data and retains the top eigenvectors that
    explain at least ``variance_ratio`` of total variance.  Random unit
    encoders are generated in the reduced-dimension subspace and projected
    back to the original space.  This concentrates encoder diversity in the
    informative dimensions and avoids wasting capacity on dead or
    near-constant features (e.g. border pixels in MNIST).

    Args:
        n_neurons: number of encoder vectors to generate.
        dim: input dimensionality.
        train_data: ``(N, dim)`` training data used to estimate the covariance.
        variance_ratio: fraction of variance to retain (default 0.95).
        rng: optional ``torch.Generator`` for reproducibility.
    """
    X = train_data.float().cpu()
    X_centered = X - X.mean(dim=0)
    cov = (X_centered.T @ X_centered) / X.shape[0]

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # eigh returns ascending order; flip to descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    # Retain top-k eigenvectors explaining >= variance_ratio of variance
    cumvar = eigenvalues.cumsum(0) / eigenvalues.sum()
    k = int((cumvar < variance_ratio).sum().item()) + 1
    k = max(k, 1)
    V_k = eigenvectors[:, :k]  # (dim, k)

    # Random encoders in the k-dimensional subspace, projected to full space
    e_sub = uniform_hypersphere(n_neurons, k, rng=rng)
    e_proj = e_sub @ V_k.T  # (n_neurons, dim)
    # V_k columns are orthonormal, so e_proj already has unit norm
    return e_proj


def class_contrast(
    n_neurons: int,
    dim: int,
    *,
    train_data: Tensor,
    train_labels: Tensor,
    n_candidates: int = 5000,
    rng: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Encoders pointing from each sample toward nearest different-class sample.

    Each encoder direction is maximally discriminative by construction: it
    points from a training sample directly toward the nearest sample of a
    different class.  The source sample serves as the natural center for
    data-driven bias computation.

    Returns a ``(encoders, centers)`` tuple.  The caller should use the
    returned centers for bias computation instead of randomly selecting from
    the training set.

    Args:
        n_neurons: number of encoder vectors to generate.
        dim: input dimensionality.
        train_data: ``(N, dim)`` training data.
        train_labels: ``(N,)`` integer class labels.
        n_candidates: number of candidate samples for nearest-neighbor lookup
            (subsampled from training data for efficiency).
        rng: optional ``torch.Generator`` for reproducibility.
    """
    X = train_data.float().cpu()
    y = train_labels.long().cpu()
    N = X.shape[0]

    # Select random anchor samples
    anchor_idx = torch.randint(N, (n_neurons,), generator=rng)
    anchors = X[anchor_idx]
    anchor_labels = y[anchor_idx]

    # Subsample candidates for neighbor lookup
    n_cand = min(N, n_candidates)
    if N > n_cand:
        cand_idx = torch.randperm(N, generator=rng)[:n_cand]
        X_cand = X[cand_idx]
        y_cand = y[cand_idx]
    else:
        X_cand = X
        y_cand = y

    # Process in chunks to bound memory: chunk_size × n_cand distances
    chunk_size = 1000
    encoders = torch.empty(n_neurons, dim)
    for start in range(0, n_neurons, chunk_size):
        end = min(start + chunk_size, n_neurons)
        batch_anchors = anchors[start:end]
        batch_labels = anchor_labels[start:end]

        # Squared distances (avoid sqrt for argmin)
        dists = torch.cdist(batch_anchors, X_cand)
        # Mask same-class candidates with inf
        same_class = batch_labels.unsqueeze(1) == y_cand.unsqueeze(0)
        dists[same_class] = float("inf")

        nn_idx = dists.argmin(dim=1)
        nn_points = X_cand[nn_idx]

        direction = nn_points - batch_anchors
        norms = direction.norm(dim=1, keepdim=True).clamp(min=1e-8)
        encoders[start:end] = direction / norms

    return encoders, anchors


def local_pca(
    n_neurons: int,
    dim: int,
    *,
    train_data: Tensor,
    k_neighbors: int = 50,
    rng: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Encoders aligned with the top local PCA direction near each center.

    For each neuron, a random training sample is chosen as the center.  The
    ``k_neighbors`` nearest training samples to that center define a local
    neighborhood.  The encoder direction is the top eigenvector (axis of
    maximum variance) of that neighborhood.

    Returns a ``(encoders, centers)`` tuple so the chosen centers are reused
    for data-driven bias computation.

    Works on arbitrarily high-dimensional data by using batched SVD on the
    ``(k, dim)`` centered neighbor matrices (avoids forming ``dim × dim``
    covariance matrices).  Neurons are processed in chunks to bound memory.

    Args:
        n_neurons: number of encoder vectors to generate.
        dim: input dimensionality.
        train_data: ``(N, dim)`` training data.
        k_neighbors: number of nearest neighbors for local PCA.
        rng: optional ``torch.Generator`` for reproducibility.
    """
    X = train_data.float().cpu()
    N = X.shape[0]
    k = min(k_neighbors, N - 1)

    # Select random center samples
    center_idx = torch.randint(N, (n_neurons,), generator=rng)
    centers = X[center_idx]

    # Process in chunks to bound memory: chunk_neurons × N distances
    # + chunk_neurons × k × dim neighbor gather
    chunk_size = min(200, n_neurons)
    encoders = torch.empty(n_neurons, dim)

    for start in range(0, n_neurons, chunk_size):
        end = min(start + chunk_size, n_neurons)
        batch_centers = centers[start:end]  # (chunk, dim)
        chunk_n = end - start

        # Find k nearest neighbors for each center
        dists = torch.cdist(batch_centers, X)  # (chunk, N)
        _, nn_indices = dists.topk(k + 1, dim=1, largest=False)
        # Exclude the center itself (distance 0) — take indices 1..k
        nn_indices = nn_indices[:, 1 : k + 1]  # (chunk, k)

        # Gather neighbor data: (chunk, k, dim)
        neighbors = X[nn_indices.reshape(-1)].reshape(chunk_n, k, dim)

        # Local PCA via SVD on centered neighbors
        centered = neighbors - neighbors.mean(dim=1, keepdim=True)
        # SVD: centered is (chunk, k, dim), k < dim typically
        # U: (chunk, k, k), S: (chunk, k), Vh: (chunk, k, dim)
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Top singular vector = first row of Vh
        top_dir = Vh[:, 0, :]  # (chunk, dim)

        # Normalize to unit norm
        norms = top_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
        encoders[start:end] = top_dir / norms

    return encoders, centers


ENCODER_STRATEGIES = {
    "hypersphere": uniform_hypersphere,
    "gaussian": gaussian,
    "sparse": sparse,
    "receptive_field": receptive_field,
    "whitened": whitened,
    "class_contrast": class_contrast,
    "local_pca": local_pca,
}


def make_encoders(n_neurons: int, dim: int, strategy: str = "hypersphere", **kwargs) -> Tensor:
    """Create encoders using a named strategy."""
    if strategy not in ENCODER_STRATEGIES:
        raise ValueError(
            f"Unknown encoder strategy {strategy!r}. Available: {sorted(ENCODER_STRATEGIES)}"
        )
    return ENCODER_STRATEGIES[strategy](n_neurons, dim, **kwargs)
