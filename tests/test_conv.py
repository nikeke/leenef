"""Tests for the gradient-free convolutional NEF pipeline."""

import pytest
import torch
import torch.nn.functional as F

from leenef.conv import (
    ConvNEFEnsemble,
    ConvNEFPipeline,
    ConvNEFStage,
    global_contrast_normalize,
    local_contrast_normalize,
)

# ── ConvNEFStage tests ─────────────────────────────────────────────────


class TestConvNEFStage:
    @pytest.fixture
    def images(self):
        g = torch.Generator().manual_seed(42)
        return torch.randn(100, 3, 16, 16, generator=g)

    def test_odd_patch_required(self):
        with pytest.raises(ValueError, match="odd"):
            ConvNEFStage(8, patch_size=4)

    def test_fit_sets_filters(self, images):
        stage = ConvNEFStage(8, patch_size=5)
        stage.fit(images)
        assert stage.filters.shape == (8, 3, 5, 5)
        assert stage.conv_bias.shape == (8,)

    def test_forward_shape(self, images):
        stage = ConvNEFStage(8, patch_size=5, pool_size=2)
        stage.fit(images)
        out = stage(images)
        # 16 // 2 = 8
        assert out.shape == (100, 8, 8, 8)

    def test_no_pool(self, images):
        stage = ConvNEFStage(4, patch_size=3, pool_size=1)
        stage.fit(images)
        out = stage(images)
        # No pooling: spatial dims unchanged
        assert out.shape == (100, 4, 16, 16)

    def test_output_nonnegative_abs(self, images):
        """abs activation produces non-negative outputs."""
        stage = ConvNEFStage(8, patch_size=5, activation="abs")
        stage.fit(images)
        out = stage(images)
        assert (out >= 0).all()

    def test_filters_ordered_by_variance(self, images):
        """First filter captures more variance than later ones."""
        stage = ConvNEFStage(8, patch_size=5, pool_size=1)
        stage.fit(images)
        out = stage(images)
        # Variance of each filter's output map
        var = out.var(dim=(0, 2, 3))
        # First filter should have highest variance (top eigenvector)
        assert var[0] >= var[-1]

    def test_single_channel_input(self):
        g = torch.Generator().manual_seed(7)
        images = torch.randn(50, 1, 12, 12, generator=g)
        stage = ConvNEFStage(4, patch_size=3)
        stage.fit(images)
        assert stage.filters.shape == (4, 1, 3, 3)
        out = stage(images)
        assert out.shape == (50, 4, 6, 6)

    def test_multi_channel_input(self):
        """Stage handles arbitrary channel counts (for stacking stages)."""
        g = torch.Generator().manual_seed(8)
        feature_maps = torch.randn(50, 16, 8, 8, generator=g)
        stage = ConvNEFStage(12, patch_size=3)
        stage.fit(feature_maps)
        assert stage.filters.shape == (12, 16, 3, 3)

    def test_max_patches_subsample(self):
        """Large images trigger patch subsampling."""
        g = torch.Generator().manual_seed(9)
        images = torch.randn(200, 1, 32, 32, generator=g)
        stage = ConvNEFStage(4, patch_size=5, max_patches=500)
        stage.fit(images)
        assert stage.filters.shape == (4, 1, 5, 5)

    def test_deterministic(self, images):
        """Two stages fit on the same data produce identical filters."""
        # max_patches high enough to avoid random subsampling
        s1 = ConvNEFStage(8, patch_size=5, max_patches=200_000)
        s2 = ConvNEFStage(8, patch_size=5, max_patches=200_000)
        s1.fit(images)
        s2.fit(images)
        # Eigenvectors may differ in sign; compare absolute values
        assert torch.allclose(s1.filters.abs(), s2.filters.abs(), atol=1e-5)

    def test_n_filters_exceeds_patch_dim(self):
        """Requesting more filters than patch dimensions is clamped."""
        g = torch.Generator().manual_seed(10)
        images = torch.randn(50, 1, 8, 8, generator=g)
        # patch_size=3 -> patch_dim=9, but requesting 20 filters
        stage = ConvNEFStage(20, patch_size=3)
        stage.fit(images)
        # Should clamp to min(20, 9) = 9
        assert stage.filters.shape[0] == 9

    def test_kmeans_fit_sets_filters(self, images):
        """K-means strategy produces filters of the correct shape."""
        stage = ConvNEFStage(8, patch_size=5, filter_strategy="kmeans", kmeans_iter=5)
        stage.fit(images)
        assert stage.filters.shape == (8, 3, 5, 5)
        assert stage.conv_bias.shape == (8,)

    def test_kmeans_forward_shape(self, images):
        """K-means stage produces correct output shape."""
        stage = ConvNEFStage(8, patch_size=5, pool_size=2, filter_strategy="kmeans")
        stage.fit(images)
        out = stage(images)
        assert out.shape == (100, 8, 8, 8)

    def test_kmeans_whiten(self, images):
        """K-means with ZCA-whitened patches produces valid filters."""
        stage = ConvNEFStage(
            8,
            patch_size=5,
            filter_strategy="kmeans",
            kmeans_iter=5,
            whiten_patches=True,
        )
        stage.fit(images)
        assert stage.filters.shape == (8, 3, 5, 5)
        assert torch.isfinite(stage.filters).all()

    def test_invalid_filter_strategy(self):
        """Unknown filter_strategy raises ValueError."""
        stage = ConvNEFStage(8, patch_size=5, filter_strategy="magic")
        with pytest.raises(ValueError, match="magic"):
            stage.fit(torch.randn(10, 1, 8, 8))


# ── ConvNEFPipeline tests ──────────────────────────────────────────────


class TestConvNEFPipeline:
    @pytest.fixture
    def cifar_like(self):
        g = torch.Generator().manual_seed(42)
        images = torch.randn(200, 3, 16, 16, generator=g)
        # 5-class one-hot targets
        labels = torch.randint(0, 5, (200,), generator=g)
        targets = F.one_hot(labels, 5).float()
        return images, targets

    def test_single_stage_shapes(self, cifar_like):
        images, targets = cifar_like
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 8, "patch_size": 5}],
            n_neurons=100,
        )
        pipe.fit(images, targets, fit_subsample=100, batch_size=50)
        out = pipe(images[:10])
        assert out.shape == (10, 5)

    def test_two_stage_shapes(self, cifar_like):
        images, targets = cifar_like
        pipe = ConvNEFPipeline(
            stages=[
                {"n_filters": 8, "patch_size": 5},
                {"n_filters": 8, "patch_size": 3},
            ],
            n_neurons=100,
        )
        pipe.fit(images, targets, fit_subsample=100, batch_size=50)
        out = pipe(images[:10])
        assert out.shape == (10, 5)

    def test_predict_matches_forward(self, cifar_like):
        images, targets = cifar_like
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3}],
            n_neurons=50,
        )
        pipe.fit(images, targets, fit_subsample=100, batch_size=50)
        out_fwd = pipe(images[:20])
        out_pred = pipe.predict(images[:20], batch_size=5)
        assert torch.allclose(out_fwd, out_pred, atol=1e-5)

    def test_above_chance_accuracy(self, cifar_like):
        """Pipeline achieves above-chance accuracy on easy synthetic data."""
        g = torch.Generator().manual_seed(99)
        # Structured data: each class has distinct spatial pattern
        images = torch.zeros(500, 1, 8, 8)
        labels = torch.zeros(500, dtype=torch.long)
        for c in range(5):
            start = c * 100
            images[start : start + 100, 0, c : c + 4, c : c + 4] = 1.0
            images[start : start + 100] += 0.1 * torch.randn(100, 1, 8, 8, generator=g)
            labels[start : start + 100] = c
        targets = F.one_hot(labels, 5).float()

        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 8, "patch_size": 3}],
            n_neurons=200,
        )
        pipe.fit(images, targets, fit_subsample=500, batch_size=100)
        pred = pipe.predict(images, batch_size=100)
        acc = (pred.argmax(dim=1) == labels).float().mean().item()
        # 5 classes -> chance is 20%, should be well above
        assert acc > 0.5, f"accuracy {acc:.2%} not above chance"

    def test_nef_kwargs_passed(self, cifar_like):
        """NEF head receives kwargs like gain and activation."""
        images, targets = cifar_like
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3}],
            n_neurons=50,
            gain=(1.0, 3.0),
            activation="relu",
        )
        pipe.fit(images, targets, fit_subsample=100, batch_size=50)
        assert pipe.head is not None
        out = pipe(images[:5])
        assert out.shape == (5, 5)

    def test_state_dict_save_load(self, cifar_like):
        """Pipeline can be saved and loaded via state_dict."""
        images, targets = cifar_like
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3}],
            n_neurons=50,
        )
        pipe.fit(images, targets, fit_subsample=100, batch_size=50)

        state = pipe.state_dict()
        pipe2 = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3}],
            n_neurons=50,
        )
        # Need to fit pipe2 first to create head, then load state
        pipe2.fit(images, targets, fit_subsample=100, batch_size=50)
        pipe2.load_state_dict(state)

        out1 = pipe(images[:10])
        out2 = pipe2(images[:10])
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_spatial_pyramid_pooling(self, cifar_like):
        """Pool levels produce correct feature dimensions."""
        images, targets = cifar_like
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 8, "patch_size": 3, "pool_size": 1}],
            n_neurons=50,
            pool_levels=[1, 2, 4],
        )
        pipe.fit(images, targets, fit_subsample=100, batch_size=50)
        out = pipe(images[:5])
        assert out.shape == (5, 5)
        # Feature dim: 8*(1 + 4 + 16) = 168
        assert pipe.head.d_in == 8 * (1 + 4 + 16)

    def test_spp_above_chance(self):
        """SPP pipeline achieves above-chance on structured data."""
        g = torch.Generator().manual_seed(99)
        images = torch.zeros(500, 1, 8, 8)
        labels = torch.zeros(500, dtype=torch.long)
        for c in range(5):
            start = c * 100
            images[start : start + 100, 0, c : c + 4, c : c + 4] = 1.0
            images[start : start + 100] += 0.1 * torch.randn(100, 1, 8, 8, generator=g)
            labels[start : start + 100] = c
        targets = F.one_hot(labels, 5).float()

        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 8, "patch_size": 3, "pool_size": 1}],
            n_neurons=200,
            pool_levels=[1, 2, 4],
        )
        pipe.fit(images, targets, fit_subsample=500, batch_size=100)
        pred = pipe.predict(images, batch_size=100)
        acc = (pred.argmax(dim=1) == labels).float().mean().item()
        assert acc > 0.5, f"accuracy {acc:.2%} not above chance"

    def test_second_order_pooling_shape(self):
        """pool_order=2 doubles the SPP feature dimension."""
        images = torch.randn(20, 1, 8, 8)
        targets = F.one_hot(torch.randint(0, 3, (20,)), 3).float()
        pipe1 = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=50,
            pool_levels=[1, 2],
            pool_order=1,
        )
        pipe1.fit(images, targets, fit_subsample=20, batch_size=20)
        pipe2 = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=50,
            pool_levels=[1, 2],
            pool_order=2,
        )
        pipe2.fit(images, targets, fit_subsample=20, batch_size=20)
        # Order-1: 4*(1+4)=20; Order-2: 4*(1+4)*2=40
        assert pipe1.head.d_in == 20
        assert pipe2.head.d_in == 40

    def test_augment_fn(self):
        """augment_fn doubles effective training data."""
        g = torch.Generator().manual_seed(77)
        images = torch.randn(50, 1, 8, 8, generator=g)
        targets = F.one_hot(torch.randint(0, 3, (50,), generator=g), 3).float()
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=30,
        )
        pipe.fit(
            images,
            targets,
            fit_subsample=50,
            batch_size=25,
            augment_fn=lambda x: x.flip(-1),
        )
        pred = pipe(images[:5])
        assert pred.shape == (5, 3)

    def test_standardize(self):
        """standardize=True normalizes features before NEF head."""
        g = torch.Generator().manual_seed(88)
        images = torch.randn(50, 1, 8, 8, generator=g)
        targets = F.one_hot(torch.randint(0, 3, (50,), generator=g), 3).float()
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=30,
            pool_levels=[1, 2],
            standardize=True,
        )
        pipe.fit(images, targets, fit_subsample=50, batch_size=25)
        assert pipe._feat_mean is not None
        assert pipe._feat_std is not None
        pred = pipe(images[:5])
        assert pred.shape == (5, 3)
        # Standardisation stats should have correct dimension
        assert pipe._feat_mean.shape[0] == pipe.head.d_in


class TestConvNEFEnsemble:
    def test_fit_and_predict(self):
        """Ensemble fits all members and produces correct output shape."""
        g = torch.Generator().manual_seed(42)
        images = torch.randn(100, 1, 8, 8, generator=g)
        labels = torch.randint(0, 3, (100,), generator=g)
        targets = F.one_hot(labels, 3).float()
        ens = ConvNEFEnsemble(
            n_members=3,
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=50,
            pool_levels=[1, 2],
        )
        ens.fit(images, targets, fit_subsample=100, batch_size=50)
        pred = ens.predict(images, batch_size=50)
        assert pred.shape == (100, 3)

    def test_members_differ(self):
        """Different seeds produce different decoders."""
        g = torch.Generator().manual_seed(7)
        images = torch.randn(80, 1, 8, 8, generator=g)
        targets = F.one_hot(torch.randint(0, 2, (80,), generator=g), 2).float()
        ens = ConvNEFEnsemble(
            n_members=2,
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=30,
        )
        ens.fit(images, targets, fit_subsample=80, batch_size=40)
        d0 = ens.members[0].head.decoders.data
        d1 = ens.members[1].head.decoders.data
        assert not torch.allclose(d0, d1)

    def test_vote_combine(self):
        """Voting combination produces valid output."""
        g = torch.Generator().manual_seed(99)
        images = torch.randn(40, 1, 8, 8, generator=g)
        targets = F.one_hot(torch.randint(0, 4, (40,), generator=g), 4).float()
        ens = ConvNEFEnsemble(
            n_members=2,
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=30,
            combine="vote",
        )
        ens.fit(images, targets, fit_subsample=40, batch_size=20)
        pred = ens.predict(images, batch_size=20)
        assert pred.shape == (40, 4)
        # Vote counts should sum to n_members for each sample
        assert (pred.sum(dim=1) == 2).all()


# ── Local contrast normalization tests ─────────────────────────────────


class TestLocalContrastNormalize:
    def test_output_shape(self):
        x = torch.randn(4, 3, 16, 16)
        out = local_contrast_normalize(x, kernel_size=5)
        assert out.shape == x.shape

    def test_approximately_zero_mean(self):
        """After LCN, local mean should be approximately zero."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(2, 1, 32, 32, generator=g) * 5 + 3
        out = local_contrast_normalize(x, kernel_size=5)
        # The global mean should be near zero (not exact due to edges)
        assert out.mean().abs() < 0.5

    def test_pipeline_with_lcn(self):
        """Pipeline with lcn_kernel preprocesses correctly."""
        g = torch.Generator().manual_seed(42)
        images = torch.randn(50, 3, 16, 16, generator=g)
        targets = F.one_hot(torch.randint(0, 3, (50,), generator=g), 3).float()
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=30,
            pool_levels=[1, 2],
            lcn_kernel=5,
        )
        pipe.fit(images, targets, fit_subsample=50, batch_size=25)
        pred = pipe(images[:5])
        assert pred.shape == (5, 3)


class TestGlobalContrastNormalize:
    def test_output_shape(self):
        x = torch.randn(4, 3, 16, 16)
        out = global_contrast_normalize(x)
        assert out.shape == x.shape

    def test_unit_stats(self):
        """After GCN, each image should have ~zero mean, ~unit std."""
        g = torch.Generator().manual_seed(42)
        x = torch.randn(8, 3, 32, 32, generator=g) * 5 + 3
        out = global_contrast_normalize(x)
        flat = out.reshape(out.shape[0], -1)
        means = flat.mean(dim=1)
        stds = flat.std(dim=1)
        assert means.abs().max() < 1e-5
        assert (stds - 1.0).abs().max() < 1e-4

    def test_pipeline_with_gcn(self):
        """Pipeline with gcn=True preprocesses correctly."""
        g = torch.Generator().manual_seed(42)
        images = torch.randn(50, 3, 16, 16, generator=g)
        targets = F.one_hot(torch.randint(0, 3, (50,), generator=g), 3).float()
        pipe = ConvNEFPipeline(
            stages=[{"n_filters": 4, "patch_size": 3, "pool_size": 1}],
            n_neurons=30,
            pool_levels=[1, 2],
            gcn=True,
        )
        pipe.fit(images, targets, fit_subsample=50, batch_size=25)
        pred = pipe(images[:5])
        assert pred.shape == (5, 3)
