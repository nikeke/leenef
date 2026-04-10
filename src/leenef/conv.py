"""Gradient-free convolutional feature extraction via PCA filters.

Implements PCANet-style convolutional stages where filters are learned
from data via PCA of observed patches, without any gradient descent.
Combined with a NEF classification head, this creates a fully
gradient-free hierarchical feature learning pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activations import make_activation
from .layers import NEFLayer


class ConvNEFStage(nn.Module):
    """PCA-based convolutional feature extraction stage.

    Extracts overlapping patches from input images or feature maps,
    computes PCA of the patch population, and uses the top eigenvectors
    as fixed convolutional filters.  The result is passed through an
    activation function and spatial average pooling.

    Filters are stored as buffers (not parameters) — they are
    data-derived but not gradient-trained.

    Args:
        n_filters: number of PCA filters (eigenvectors) to retain.
        patch_size: spatial size of square patches (must be odd).
        pool_size: spatial average pooling factor (default 2).
        activation: activation function name (default ``'abs'``).
        max_patches: maximum patches to subsample for PCA
            (default 100_000).
    """

    def __init__(
        self,
        n_filters: int,
        patch_size: int,
        pool_size: int = 2,
        activation: str = "abs",
        max_patches: int = 100_000,
    ):
        super().__init__()
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.pool_size = pool_size
        self.act = make_activation(activation)
        self.max_patches = max_patches
        # Populated by fit()
        self.register_buffer("filters", torch.empty(0))
        self.register_buffer("conv_bias", torch.empty(0))

    def fit(self, images: Tensor) -> None:
        """Learn PCA filters from training images.

        Extracts all overlapping patches, subsamples if necessary,
        centers, and computes PCA via eigendecomposition.  The top
        ``n_filters`` eigenvectors become convolutional filters.

        Args:
            images: ``(N, C, H, W)`` training images or feature maps.
        """
        N, C, H, W = images.shape
        p = self.patch_size
        patch_dim = C * p * p

        # Extract patches in chunks to limit memory
        chunk_size = max(1, self.max_patches // ((H - p + 1) * (W - p + 1)))
        chunk_size = min(chunk_size, N)
        all_patches = []
        collected = 0
        for i in range(0, N, chunk_size):
            batch = images[i : i + chunk_size]
            # (chunk, patch_dim, n_spatial)
            patches = F.unfold(batch, kernel_size=p, stride=1, padding=p // 2)
            # (chunk * n_spatial, patch_dim)
            patches = patches.permute(0, 2, 1).reshape(-1, patch_dim).float()
            all_patches.append(patches)
            collected += patches.shape[0]
            if collected >= self.max_patches:
                break

        patches = torch.cat(all_patches, dim=0)
        if patches.shape[0] > self.max_patches:
            idx = torch.randperm(patches.shape[0], device=patches.device)[: self.max_patches]
            patches = patches[idx]

        # Center
        mean = patches.mean(dim=0)
        patches_centered = patches - mean

        # Covariance and PCA
        cov = (patches_centered.T @ patches_centered) / patches_centered.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # Top n_filters eigenvectors (largest eigenvalues at end of eigh output)
        k = min(self.n_filters, eigenvectors.shape[1])
        top = eigenvectors[:, -k:].flip(1).T  # (k, patch_dim)

        self.filters = top.reshape(k, C, p, p)
        # Centering bias: applying filters to centered patches is equivalent
        # to conv2d(x, filters, bias) where bias_f = -mean · filter_f
        self.conv_bias = -(mean @ top.T)  # (k,)

    def forward(self, x: Tensor) -> Tensor:
        """Apply PCA filters, activation, and pooling.

        Args:
            x: ``(N, C, H, W)`` input images or feature maps.

        Returns:
            ``(N, n_filters, H_out, W_out)`` output feature maps.
        """
        pad = self.patch_size // 2
        out = F.conv2d(x, self.filters, bias=self.conv_bias, padding=pad)
        out = self.act(out)
        if self.pool_size > 1:
            out = F.avg_pool2d(out, self.pool_size, ceil_mode=True)
        return out


class ConvNEFPipeline(nn.Module):
    """Multi-stage PCA convolutional pipeline with NEF classification head.

    Stacks one or more :class:`ConvNEFStage` modules for hierarchical
    feature extraction, followed by a :class:`NEFLayer` for classification.
    The entire pipeline is gradient-free: PCA filters are learned from
    data patches, and decoders are solved analytically.

    When ``pool_levels`` is set, the final feature maps are processed
    through spatial pyramid pooling (adaptive average pool at each level)
    before being passed to the NEF head.  This provides translation
    invariance at multiple spatial scales.

    Args:
        stages: list of dicts, each passed as kwargs to
            :class:`ConvNEFStage`.
        n_neurons: number of neurons in the NEF classification head.
        pool_levels: optional list of spatial pool sizes for pyramid
            pooling (e.g. ``[1, 2, 4]``).  ``None`` flattens directly.
        pool_order: ``1`` for mean-only pooling (default), ``2`` to
            append per-channel variance in each spatial block.
        nef_kwargs: additional keyword arguments for :class:`NEFLayer`
            (e.g. ``encoder_strategy``, ``activation``, ``gain``).
    """

    def __init__(
        self,
        stages: list[dict],
        n_neurons: int,
        pool_levels: list[int] | None = None,
        pool_order: int = 1,
        **nef_kwargs,
    ):
        super().__init__()
        self.stages = nn.ModuleList(ConvNEFStage(**cfg) for cfg in stages)
        self.n_neurons = n_neurons
        self.pool_levels = pool_levels
        self.pool_order = pool_order
        self.nef_kwargs = nef_kwargs
        self.head: NEFLayer | None = None

    def _pool_features(self, x: Tensor) -> Tensor:
        """Convert feature maps to flat feature vectors.

        If ``pool_levels`` is set, applies spatial pyramid pooling
        (adaptive average pool at each level, concatenated).
        When ``pool_order`` is 2, appends per-channel variance in each
        spatial block for second-order statistics.
        Otherwise, simply flattens.
        """
        if self.pool_levels is None:
            return x.reshape(x.shape[0], -1)
        parts = []
        for level in self.pool_levels:
            pooled = F.adaptive_avg_pool2d(x, level)
            parts.append(pooled.reshape(x.shape[0], -1))
            if self.pool_order >= 2:
                # Per-channel variance in each spatial block
                x_sq = x * x
                mean_sq = F.adaptive_avg_pool2d(x_sq, level)
                var = mean_sq - pooled * pooled
                parts.append(var.reshape(x.shape[0], -1))
        return torch.cat(parts, dim=1)

    def fit(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        alpha: float = 1e-2,
        fit_subsample: int = 10_000,
        batch_size: int = 1000,
        seed: int = 0,
    ) -> None:
        """Fit all stages and the classification head.

        Each stage is fit on a subsample of its input (PCA of patches).
        The full dataset is then processed in chunks through all stages
        using :meth:`~NEFLayer.partial_fit` to accumulate normal-equation
        statistics without storing all features in memory.

        Args:
            images: ``(N, C, H, W)`` training images.
            targets: ``(N, n_classes)`` one-hot target matrix.
            alpha: Tikhonov regularisation for the decoder solve.
            fit_subsample: max samples for stage fitting and center
                selection.
            batch_size: chunk size for incremental feature extraction.
            seed: random seed for subsampling reproducibility.
        """
        N = images.shape[0]
        g = torch.Generator(device=images.device).manual_seed(seed)

        # Subsample indices for stage fitting and center selection
        sub_n = min(fit_subsample, N)
        sub_idx = torch.randperm(N, generator=g, device=images.device)[:sub_n]
        sub_images = images[sub_idx]

        # Fit stages sequentially on subsample
        x_sub = sub_images
        for stage in self.stages:
            stage.fit(x_sub)
            x_sub = stage(x_sub)

        # Pool features and use as centers
        centers = self._pool_features(x_sub)
        feat_dim = centers.shape[1]

        # Create NEF classification head
        self.head = NEFLayer(
            feat_dim,
            self.n_neurons,
            targets.shape[1],
            centers=centers,
            **self.nef_kwargs,
        )

        # Accumulate normal equations over full dataset in chunks
        self.head.reset_accumulators()
        for i in range(0, N, batch_size):
            x_batch = images[i : i + batch_size]
            for stage in self.stages:
                x_batch = stage(x_batch)
            features = self._pool_features(x_batch)
            self.head.partial_fit(features, targets[i : i + batch_size])

        self.head.solve_accumulated(alpha=alpha)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all stages and classification head."""
        for stage in self.stages:
            x = stage(x)
        features = self._pool_features(x)
        return self.head(features)

    def predict(self, images: Tensor, batch_size: int = 1000) -> Tensor:
        """Batched prediction to limit peak memory.

        Args:
            images: ``(N, C, H, W)`` input images.
            batch_size: chunk size for processing.

        Returns:
            ``(N, n_classes)`` prediction matrix.
        """
        chunks = []
        for i in range(0, images.shape[0], batch_size):
            chunks.append(self.forward(images[i : i + batch_size]))
        return torch.cat(chunks, dim=0)
