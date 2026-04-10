"""Gradient-free convolutional feature extraction.

Implements convolutional stages where filters are learned from data
patches (PCA or k-means), without any gradient descent.  Combined
with a NEF classification head, this creates a fully gradient-free
hierarchical feature learning pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activations import make_activation
from .layers import NEFLayer


def local_contrast_normalize(
    x: Tensor, kernel_size: int = 5, eps: float = 1e-6
) -> Tensor:
    """Apply local contrast normalization to images.

    Subtracts the local mean and divides by the local standard deviation
    using a uniform box filter.  Operates per-channel.

    Args:
        x: ``(N, C, H, W)`` input tensor.
        kernel_size: spatial size of the local neighbourhood (default 5).
        eps: stability constant added to the local std.

    Returns:
        Normalized tensor of the same shape.
    """
    C = x.shape[1]
    pad = kernel_size // 2
    ones = torch.ones(C, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    area = kernel_size * kernel_size

    local_mean = F.conv2d(x, ones / area, padding=pad, groups=C)
    centered = x - local_mean
    local_sq = F.conv2d(centered * centered, ones / area, padding=pad, groups=C)
    local_std = local_sq.clamp(min=0).sqrt() + eps
    return centered / local_std


def global_contrast_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply global contrast normalization to images.

    For each image, subtracts the global mean and divides by the
    global standard deviation.  This normalises brightness and
    contrast across the dataset.

    Args:
        x: ``(N, C, H, W)`` or ``(N, D)`` input tensor.
        eps: stability constant added to the standard deviation.

    Returns:
        Normalized tensor of the same shape.
    """
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=eps)
    return ((flat - mean) / std).reshape(x.shape)


class ConvNEFStage(nn.Module):
    """Convolutional feature extraction stage with PCA or k-means filters.

    Extracts overlapping patches from input images or feature maps,
    learns filters from the patch population, and applies them as
    fixed convolutional filters.  The result is passed through an
    activation function and spatial average pooling.

    Two filter strategies are supported:

    - ``'pca'``: top eigenvectors of the patch covariance (PCANet-style).
    - ``'kmeans'``: k-means centroids of the patch population, capturing
      the most representative local patterns.  Follows Coates & Ng (2012).

    Filters are stored as buffers (not parameters) — they are
    data-derived but not gradient-trained.

    Args:
        n_filters: number of filters to learn.
        patch_size: spatial size of square patches (must be odd).
        pool_size: spatial average pooling factor (default 2).
        activation: activation function name (default ``'abs'``).
        max_patches: maximum patches to subsample for filter learning
            (default 100_000).
        filter_strategy: ``'pca'`` or ``'kmeans'`` (default ``'pca'``).
        kmeans_iter: number of k-means iterations (default 20).
        whiten_patches: if ``True``, ZCA-whiten patches before k-means.
            Only used when ``filter_strategy='kmeans'``.
        normalize_patches: if ``True``, apply per-patch contrast
            normalization: subtract each patch's mean and divide by its
            L2 norm.  This removes brightness variation and focuses
            PCA/k-means on texture/edge patterns (Coates & Ng 2012).
            Applied both during filter learning and at inference time.
    """

    def __init__(
        self,
        n_filters: int,
        patch_size: int,
        pool_size: int = 2,
        activation: str = "abs",
        max_patches: int = 100_000,
        filter_strategy: str = "pca",
        kmeans_iter: int = 20,
        whiten_patches: bool = False,
        normalize_patches: bool = False,
    ):
        super().__init__()
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        self.n_filters = n_filters
        self.patch_size = patch_size
        self.pool_size = pool_size
        self.act = make_activation(activation)
        self.max_patches = max_patches
        self.filter_strategy = filter_strategy
        self.kmeans_iter = kmeans_iter
        self.whiten_patches = whiten_patches
        self.normalize_patches = normalize_patches
        # Populated by fit()
        self.register_buffer("filters", torch.empty(0))
        self.register_buffer("conv_bias", torch.empty(0))
        self.register_buffer("_patch_mean", torch.empty(0))
        # For k-means: ||c_k||² after centering, for distance computation
        self.register_buffer("_centroid_sq", torch.empty(0))

    def _extract_patches(self, images: Tensor) -> Tensor:
        """Extract and subsample patches from images.

        Returns:
            ``(n_patches, patch_dim)`` mean-centered patches.
        """
        N, C, H, W = images.shape
        p = self.patch_size
        patch_dim = C * p * p

        chunk_size = max(1, self.max_patches // ((H - p + 1) * (W - p + 1)))
        chunk_size = min(chunk_size, N)
        all_patches = []
        collected = 0
        for i in range(0, N, chunk_size):
            batch = images[i : i + chunk_size]
            patches = F.unfold(batch, kernel_size=p, stride=1, padding=p // 2)
            patches = patches.permute(0, 2, 1).reshape(-1, patch_dim).float()
            all_patches.append(patches)
            collected += patches.shape[0]
            if collected >= self.max_patches:
                break

        patches = torch.cat(all_patches, dim=0)
        if patches.shape[0] > self.max_patches:
            idx = torch.randperm(patches.shape[0], device=patches.device)[: self.max_patches]
            patches = patches[idx]
        return patches

    @staticmethod
    def _normalize_patches_inplace(patches: Tensor, eps: float = 1e-8) -> Tensor:
        """Per-patch contrast normalization: zero mean, unit L2 norm.

        Removes brightness variation so that PCA/k-means focus on
        texture and edge patterns rather than overall intensity.

        Args:
            patches: ``(n, d)`` patch matrix, modified in-place.
            eps: stability constant for near-zero-norm patches.

        Returns:
            The normalised patches tensor (same object as input).
        """
        patches -= patches.mean(dim=1, keepdim=True)
        norms = patches.norm(dim=1, keepdim=True).clamp(min=eps)
        patches /= norms
        return patches

    @staticmethod
    def _kmeans(patches: Tensor, k: int, n_iter: int = 20, seed: int = 0) -> Tensor:
        """Pure-PyTorch k-means clustering.

        Args:
            patches: ``(n, d)`` data points.
            k: number of clusters.
            n_iter: number of Lloyd iterations.
            seed: random seed for centroid initialisation.

        Returns:
            ``(k, d)`` cluster centroids.
        """
        n, d = patches.shape
        g = torch.Generator(device=patches.device).manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=patches.device)[:k]
        centroids = patches[idx].clone()

        chunk = 10_000
        for _ in range(n_iter):
            c_sq = centroids.pow(2).sum(1)  # (k,)
            labels = torch.empty(n, dtype=torch.long, device=patches.device)
            for i in range(0, n, chunk):
                batch = patches[i : i + chunk]
                x_sq = batch.pow(2).sum(1)  # (batch,)
                xc = batch @ centroids.T  # (batch, k)
                dist = x_sq.unsqueeze(1) - 2 * xc + c_sq.unsqueeze(0)
                labels[i : i + batch.shape[0]] = dist.argmin(dim=1)

            sums = torch.zeros(k, d, device=patches.device, dtype=patches.dtype)
            counts = torch.zeros(k, device=patches.device, dtype=patches.dtype)
            sums.scatter_add_(0, labels.unsqueeze(1).expand(-1, d), patches)
            counts.scatter_add_(
                0, labels, torch.ones(n, device=patches.device, dtype=patches.dtype)
            )
            valid = counts > 0
            centroids[valid] = sums[valid] / counts[valid].unsqueeze(1)

        return centroids

    def fit(self, images: Tensor) -> None:
        """Learn filters from training images.

        For ``filter_strategy='pca'``, computes PCA of patches and uses
        the top eigenvectors.  For ``'kmeans'``, runs k-means on patches
        and uses the cluster centroids.

        Args:
            images: ``(N, C, H, W)`` training images or feature maps.
        """
        C = images.shape[1]
        p = self.patch_size

        patches = self._extract_patches(images)
        if self.normalize_patches:
            self._normalize_patches_inplace(patches)
        mean = patches.mean(dim=0)
        patches_centered = patches - mean

        if self.filter_strategy == "pca":
            cov = (patches_centered.T @ patches_centered) / patches_centered.shape[0]
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            k = min(self.n_filters, eigenvectors.shape[1])
            top = eigenvectors[:, -k:].flip(1).T  # (k, patch_dim)
        elif self.filter_strategy == "kmeans":
            if self.whiten_patches:
                # ZCA whitening: safe for low-dim patches
                cov = (patches_centered.T @ patches_centered) / patches_centered.shape[0]
                eigvals, eigvecs = torch.linalg.eigh(cov)
                floor = eigvals.max().item() * 0.01
                eigvals_safe = eigvals.clamp(min=floor)
                W = eigvecs @ torch.diag(eigvals_safe.rsqrt()) @ eigvecs.T
                patches_w = patches_centered @ W
            else:
                patches_w = patches_centered
            centroids = self._kmeans(patches_w, self.n_filters, self.kmeans_iter)
            if self.whiten_patches:
                # Map centroids back to original space
                W_inv = eigvecs @ torch.diag(eigvals_safe.sqrt()) @ eigvecs.T
                top = centroids @ W_inv  # (k, patch_dim)
            else:
                top = centroids
        else:
            raise ValueError(
                f"Unknown filter_strategy '{self.filter_strategy}', expected 'pca' or 'kmeans'"
            )

        k = top.shape[0]
        self.filters = top.reshape(k, C, p, p)
        self.conv_bias = -(mean @ top.T)  # (k,)
        self._patch_mean = mean
        if self.filter_strategy == "kmeans":
            # Pre-compute ||c_k - mean||² for distance-based activation
            self._centroid_sq = top.pow(2).sum(1)  # (k,)
        else:
            self._centroid_sq = torch.empty(0)

    def forward(self, x: Tensor) -> Tensor:
        """Apply learned filters, activation, and pooling.

        For PCA filters, uses inner product + configurable activation.
        For k-means filters, uses triangle activation (Coates & Ng 2012):
        ``max(0, mean_dist - dist_k)`` where ``dist_k`` is the distance
        from each spatial patch to centroid k.

        When ``normalize_patches=True``, patches are extracted, normalised
        per-patch (zero mean, unit L2 norm), and then inner-producted with
        filters.  This matches the preprocessing used during filter
        learning and removes brightness variation at each spatial position.

        Args:
            x: ``(N, C, H, W)`` input images or feature maps.

        Returns:
            ``(N, n_filters, H_out, W_out)`` output feature maps.
        """
        if self.normalize_patches:
            return self._forward_normalized(x)

        pad = self.patch_size // 2
        if self.filter_strategy == "kmeans":
            # Triangle activation: max(0, mean_dist - dist_k)
            # ||x-c||² = ||x||² - 2(x·c) + ||c||²
            xc = F.conv2d(x, self.filters, bias=self.conv_bias, padding=pad)
            # ||x||² at each spatial position (sum of squares in each patch)
            ones_filter = torch.ones(
                1,
                x.shape[1],
                self.patch_size,
                self.patch_size,
                device=x.device,
                dtype=x.dtype,
            )
            x_sq = F.conv2d(x * x, ones_filter, padding=pad)  # (N, 1, H, W)
            # ||x_centered||² = ||x||² - 2*mean·x_patch + ||mean||²
            mean_filter = self._patch_mean.reshape(1, x.shape[1], self.patch_size, self.patch_size)
            mean_x = F.conv2d(x, mean_filter, padding=pad)  # (N, 1, H, W)
            mean_sq = self._patch_mean.pow(2).sum()
            x_centered_sq = x_sq - 2 * mean_x + mean_sq
            # dist² = x_centered_sq - 2*xc + ||c||²
            dist_sq = x_centered_sq - 2 * xc + self._centroid_sq.reshape(1, -1, 1, 1)
            dist = dist_sq.clamp(min=0).sqrt()
            mean_dist = dist.mean(dim=1, keepdim=True)
            out = (mean_dist - dist).clamp(min=0)
        else:
            out = F.conv2d(x, self.filters, bias=self.conv_bias, padding=pad)
            out = self.act(out)
        if self.pool_size > 1:
            out = F.avg_pool2d(out, self.pool_size, ceil_mode=True)
        return out

    def _forward_normalized(self, x: Tensor) -> Tensor:
        """Forward pass with per-patch contrast normalization.

        Unfolds input into patches, normalises each patch to zero mean
        and unit L2 norm, subtracts the population mean, then computes
        inner products with the learned filters.
        """
        N, C, H, W = x.shape
        p = self.patch_size
        pad = p // 2
        k = self.filters.shape[0]
        filters_flat = self.filters.reshape(k, -1)  # (k, patch_dim)

        # Unfold: (N, patch_dim, H*W)
        patches = F.unfold(x, kernel_size=p, stride=1, padding=pad)
        # (N, H*W, patch_dim)
        patches = patches.permute(0, 2, 1).contiguous()

        # Per-patch contrast normalization
        patches -= patches.mean(dim=2, keepdim=True)
        norms = patches.norm(dim=2, keepdim=True).clamp(min=1e-8)
        patches /= norms

        # Subtract population mean and compute inner products with filters
        patches = patches - self._patch_mean.unsqueeze(0).unsqueeze(0)
        # (N, L, k) = (N, L, d) @ (d, k)
        out = patches @ filters_flat.T
        # Reshape to spatial: (N, k, H, W)
        out = out.permute(0, 2, 1).reshape(N, k, H, W)

        if self.filter_strategy == "kmeans":
            # Triangle activation on normalised patches
            dist_sq = (
                patches.pow(2).sum(2, keepdim=True)  # (N, L, 1)
                - 2 * (patches @ filters_flat.T)  # (N, L, k)
                + filters_flat.pow(2).sum(1).unsqueeze(0).unsqueeze(0)  # (1, 1, k)
            )
            dist = dist_sq.clamp(min=0).sqrt()
            mean_dist = dist.mean(dim=2, keepdim=True)
            out = (mean_dist - dist).clamp(min=0)
            out = out.permute(0, 2, 1).reshape(N, k, H, W)
        else:
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
            :class:`ConvNEFStage`.  When ``parallel=True``, all stages
            process the original input independently and their feature
            maps are concatenated along the channel dimension before
            pooling.  When ``parallel=False`` (default), stages are
            applied sequentially (hierarchical).
        n_neurons: number of neurons in the NEF classification head.
        pool_levels: optional list of spatial pool sizes for pyramid
            pooling (e.g. ``[1, 2, 4]``).  ``None`` flattens directly.
        pool_order: ``1`` for mean-only pooling (default), ``2`` to
            append per-channel variance in each spatial block.
        standardize: if ``True``, standardize pooled features to zero
            mean and unit variance before the NEF head (default ``False``).
        lcn_kernel: if set, apply local contrast normalization with this
            kernel size as a preprocessing step on the input images
            (default ``None`` = disabled).
        gcn: if ``True``, apply global contrast normalization (per-image
            mean/std normalisation) as a preprocessing step.  Applied
            before LCN when both are enabled (default ``False``).
        parallel: if ``True``, stages run in parallel on the same input
            and their outputs are concatenated.  Useful for multi-scale
            feature extraction with different patch sizes.  Feature maps
            are resized to a common spatial size before concatenation
            (default ``False``).
        nef_kwargs: additional keyword arguments for :class:`NEFLayer`
            (e.g. ``encoder_strategy``, ``activation``, ``gain``).
    """

    def __init__(
        self,
        stages: list[dict],
        n_neurons: int,
        pool_levels: list[int] | None = None,
        pool_order: int = 1,
        standardize: bool = False,
        lcn_kernel: int | None = None,
        gcn: bool = False,
        parallel: bool = False,
        **nef_kwargs,
    ):
        super().__init__()
        self.stages = nn.ModuleList(ConvNEFStage(**cfg) for cfg in stages)
        self.n_neurons = n_neurons
        self.pool_levels = pool_levels
        self.pool_order = pool_order
        self.standardize = standardize
        self.lcn_kernel = lcn_kernel
        self.gcn = gcn
        self.parallel = parallel
        self.nef_kwargs = nef_kwargs
        self.head: NEFLayer | None = None
        self._feat_mean: Tensor | None = None
        self._feat_std: Tensor | None = None

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

    def _standardize_features(self, features: Tensor, *, fit: bool = False) -> Tensor:
        """Optionally standardize features to zero mean, unit variance.

        When ``fit=True``, computes and stores the mean/std from the
        provided features.  Otherwise, uses the stored statistics.
        """
        if not self.standardize:
            return features
        if fit:
            self._feat_mean = features.mean(dim=0)
            self._feat_std = features.std(dim=0).clamp(min=1e-6)
        assert self._feat_mean is not None
        return (features - self._feat_mean) / self._feat_std

    def _preprocess(self, x: Tensor) -> Tensor:
        """Apply optional preprocessing (GCN, LCN) to input images."""
        if self.gcn:
            x = global_contrast_normalize(x)
        if self.lcn_kernel is not None:
            x = local_contrast_normalize(x, kernel_size=self.lcn_kernel)
        return x

    def _apply_stages(self, x: Tensor) -> Tensor:
        """Run conv stages on the input.

        In sequential mode (default), stages are chained: each takes
        the output of the previous stage.  In parallel mode, all stages
        independently process the same input and their feature maps
        are concatenated along the channel dimension.  When parallel
        stages produce different spatial sizes, they are resized to
        the minimum spatial dimensions using adaptive average pooling.
        """
        if self.parallel and len(self.stages) > 1:
            outs = [stage(x) for stage in self.stages]
            min_h = min(o.shape[2] for o in outs)
            min_w = min(o.shape[3] for o in outs)
            aligned = []
            for o in outs:
                if o.shape[2] != min_h or o.shape[3] != min_w:
                    o = F.adaptive_avg_pool2d(o, (min_h, min_w))
                aligned.append(o)
            return torch.cat(aligned, dim=1)
        else:
            for stage in self.stages:
                x = stage(x)
            return x

    def fit(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        alpha: float = 1e-2,
        fit_subsample: int = 10_000,
        batch_size: int = 1000,
        seed: int = 0,
        augment_fn: "Callable[[Tensor], Tensor] | None" = None,
    ) -> None:
        """Fit all stages and the classification head.

        Each stage is fit on a subsample of its input (PCA of patches).
        The full dataset is then processed in chunks through all stages
        using :meth:`~NEFLayer.partial_fit` to accumulate normal-equation
        statistics without storing all features in memory.

        When ``augment_fn`` is provided, augmented copies of each batch
        are also accumulated during the decoder solve (but not used for
        stage fitting or center selection).  This effectively trains on
        a larger dataset without gradient descent.

        Args:
            images: ``(N, C, H, W)`` training images.
            targets: ``(N, n_classes)`` one-hot target matrix.
            alpha: Tikhonov regularisation for the decoder solve.
            fit_subsample: max samples for stage fitting and center
                selection.
            batch_size: chunk size for incremental feature extraction.
            seed: random seed for subsampling reproducibility.
            augment_fn: optional function ``(batch) -> augmented_batch``
                producing augmented copies.  Called once per batch during
                decoder accumulation.  The augmented batch inherits the
                same targets.
        """
        N = images.shape[0]
        g = torch.Generator(device=images.device).manual_seed(seed)

        # Subsample indices for stage fitting and center selection
        sub_n = min(fit_subsample, N)
        sub_idx = torch.randperm(N, generator=g, device=images.device)[:sub_n]
        sub_images = self._preprocess(images[sub_idx])

        # Fit stages on subsample
        if self.parallel:
            for stage in self.stages:
                stage.fit(sub_images)
        else:
            x_sub = sub_images
            for stage in self.stages:
                stage.fit(x_sub)
                x_sub = stage(x_sub)
        x_sub = self._apply_stages(sub_images)

        # Pool features and standardize / use as centers
        centers = self._pool_features(x_sub)
        centers = self._standardize_features(centers, fit=True)
        feat_dim = centers.shape[1]

        # Create NEF classification head.
        # When the encoder strategy needs training data (whitened,
        # class_contrast, local_pca), inject the subsample features.
        head_kwargs = dict(self.nef_kwargs)
        strategy = head_kwargs.get("encoder_strategy", "hypersphere")
        data_strategies = {"whitened", "class_contrast", "local_pca"}
        if strategy in data_strategies:
            ek = dict(head_kwargs.get("encoder_kwargs", {}) or {})
            if "train_data" not in ek:
                ek["train_data"] = centers
            head_kwargs["encoder_kwargs"] = ek

        self.head = NEFLayer(
            feat_dim,
            self.n_neurons,
            targets.shape[1],
            centers=centers,
            **head_kwargs,
        )

        # Accumulate normal equations over full dataset in chunks
        self.head.reset_accumulators()
        for i in range(0, N, batch_size):
            x_batch = self._preprocess(images[i : i + batch_size])
            t_batch = targets[i : i + batch_size]
            x_conv = self._apply_stages(x_batch)
            features = self._standardize_features(self._pool_features(x_conv))
            self.head.partial_fit(features, t_batch)

            if augment_fn is not None:
                x_aug = self._preprocess(augment_fn(images[i : i + batch_size]))
                x_conv_aug = self._apply_stages(x_aug)
                features_aug = self._standardize_features(
                    self._pool_features(x_conv_aug)
                )
                self.head.partial_fit(features_aug, t_batch)

        self.head.solve_accumulated(alpha=alpha)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all stages and classification head."""
        x = self._preprocess(x)
        x = self._apply_stages(x)
        features = self._standardize_features(self._pool_features(x))
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


class ConvNEFEnsemble(nn.Module):
    """Ensemble of :class:`ConvNEFPipeline` members with different seeds.

    Each member independently learns PCA (or k-means) convolutional
    filters and NEF decoders.  Predictions are combined by probability
    averaging (softmax then mean).  Exploits the fast analytical solve:
    10 members × 20s = 200s total, still gradient-free.

    For diverse ensembles, pass ``member_stages`` — a list of stage
    configs per member — instead of ``stages``.  This allows each
    member to use different patch sizes, filter counts, or strategies,
    increasing ensemble diversity.

    Args:
        n_members: number of ensemble members.
        stages: list of dicts passed to each member's stages.  Ignored
            when ``member_stages`` is provided.
        n_neurons: neurons per member's NEF head.
        pool_levels: spatial pyramid pool sizes (passed to each member).
        pool_order: pooling order (passed to each member).
        combine: ``'mean'`` for probability averaging (default),
            ``'vote'`` for majority voting.
        member_stages: optional list of ``n_members`` stage config lists.
            When provided, ``member_stages[i]`` is used as the stages
            for member ``i``, enabling diverse configurations.
        nef_kwargs: additional keyword arguments for :class:`NEFLayer`.
    """

    def __init__(
        self,
        n_members: int,
        stages: list[dict],
        n_neurons: int,
        pool_levels: list[int] | None = None,
        pool_order: int = 1,
        combine: str = "mean",
        standardize: bool = False,
        lcn_kernel: int | None = None,
        gcn: bool = False,
        parallel: bool = False,
        member_stages: list[list[dict]] | None = None,
        **nef_kwargs,
    ):
        super().__init__()
        self.n_members = n_members
        self.combine = combine
        if member_stages is not None:
            if len(member_stages) != n_members:
                raise ValueError(
                    f"member_stages has {len(member_stages)} entries but "
                    f"n_members={n_members}"
                )
            per_member = member_stages
        else:
            per_member = [[{**s} for s in stages] for _ in range(n_members)]

        self.members = nn.ModuleList(
            ConvNEFPipeline(
                stages=ms,
                n_neurons=n_neurons,
                pool_levels=pool_levels,
                pool_order=pool_order,
                standardize=standardize,
                lcn_kernel=lcn_kernel,
                gcn=gcn,
                parallel=parallel,
                **nef_kwargs,
            )
            for ms in per_member
        )

    def fit(
        self,
        images: Tensor,
        targets: Tensor,
        *,
        alpha: float = 1e-2,
        fit_subsample: int = 10_000,
        batch_size: int = 1000,
        base_seed: int = 0,
        augment_fn: "Callable[[Tensor], Tensor] | None" = None,
    ) -> None:
        """Fit all ensemble members with different random seeds.

        Args:
            images: ``(N, C, H, W)`` training images.
            targets: ``(N, n_classes)`` one-hot target matrix.
            alpha: Tikhonov regularisation for decoder solves.
            fit_subsample: max samples for stage fitting per member.
            batch_size: chunk size for incremental feature extraction.
            base_seed: seed for the first member; subsequent members
                use ``base_seed + i``.
            augment_fn: optional augmentation function passed to each
                member's :meth:`ConvNEFPipeline.fit`.
        """
        for i, member in enumerate(self.members):
            member.fit(
                images,
                targets,
                alpha=alpha,
                fit_subsample=fit_subsample,
                batch_size=batch_size,
                seed=base_seed + i,
                augment_fn=augment_fn,
            )

    def predict(self, images: Tensor, batch_size: int = 1000) -> Tensor:
        """Ensemble prediction via probability averaging or voting.

        Args:
            images: ``(N, C, H, W)`` input images.
            batch_size: chunk size for processing.

        Returns:
            ``(N, n_classes)`` combined predictions.
        """
        if self.combine == "vote":
            votes = torch.zeros(
                images.shape[0],
                self.members[0].head.d_out,
                device=images.device,
            )
            for member in self.members:
                pred = member.predict(images, batch_size=batch_size)
                votes.scatter_add_(
                    1,
                    pred.argmax(1, keepdim=True),
                    torch.ones(images.shape[0], 1, device=images.device),
                )
            return votes

        # Probability averaging (default)
        total = None
        for member in self.members:
            pred = member.predict(images, batch_size=batch_size)
            probs = torch.softmax(pred, dim=1)
            if total is None:
                total = probs
            else:
                total = total + probs
        return total / self.n_members
