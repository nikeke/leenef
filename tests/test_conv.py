"""Tests for the gradient-free convolutional NEF pipeline."""

import pytest
import torch
import torch.nn.functional as F

from leenef.conv import ConvNEFPipeline, ConvNEFStage

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
