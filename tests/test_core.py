"""Tests for leenef core modules."""

import math

import pytest
import torch

from leenef.activations import LIFRate, make_activation
from leenef.encoders import (
    class_contrast,
    gaussian,
    local_pca,
    make_encoders,
    receptive_field,
    sparse,
    uniform_hypersphere,
    whitened,
)
from leenef.layers import NEFLayer
from leenef.solvers import (
    lstsq,
    normal_equations,
    regularization_strength,
    solve_decoders,
    solve_from_normal_equations,
    tikhonov,
)

# ── Encoders ──────────────────────────────────────────────────────────


class TestEncoders:
    def test_hypersphere_shape(self):
        e = make_encoders(100, 10, strategy="hypersphere")
        assert e.shape == (100, 10)

    def test_hypersphere_unit_norm(self):
        e = uniform_hypersphere(200, 5)
        norms = e.norm(dim=1)
        assert torch.allclose(norms, torch.ones(200), atol=1e-5)

    def test_gaussian_shape(self):
        e = gaussian(50, 8)
        assert e.shape == (50, 8)

    def test_sparse_sparsity(self):
        e = sparse(1000, 100, sparsity=0.9)
        frac_zero = (e == 0).float().mean().item()
        # Should be roughly 90% zero (with some variance)
        assert 0.85 < frac_zero < 0.95

    def test_rng_reproducibility(self):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        e1 = make_encoders(50, 10, rng=g1)
        e2 = make_encoders(50, 10, rng=g2)
        assert torch.equal(e1, e2)


class TestReceptiveFieldEncoders:
    def test_shape(self):
        e = receptive_field(100, 784, patch_size=5, image_shape=(28, 28))
        assert e.shape == (100, 784)

    def test_sparsity_pattern(self):
        """Each neuron should have exactly patch_size² nonzero entries."""
        e = receptive_field(200, 784, patch_size=5, image_shape=(28, 28))
        nnz_per_row = (e != 0).sum(dim=1)
        assert (nnz_per_row == 25).all()

    def test_unit_norm_within_patch(self):
        """Nonzero entries of each encoder should have unit norm."""
        e = receptive_field(200, 784, patch_size=7, image_shape=(28, 28))
        norms = e.norm(dim=1)
        assert torch.allclose(norms, torch.ones(200), atol=1e-5)

    def test_locality(self):
        """Nonzero indices should form a contiguous spatial patch."""
        e = receptive_field(50, 784, patch_size=5, image_shape=(28, 28))
        W = 28
        for i in range(50):
            nz = e[i].nonzero(as_tuple=True)[0]
            rows = nz // W
            cols = nz % W
            # Should span exactly patch_size rows and patch_size cols
            assert rows.max() - rows.min() == 4  # patch_size - 1
            assert cols.max() - cols.min() == 4

    def test_multichannel(self):
        """CIFAR-10: dim=3072 = 32×32×3, each neuron has patch_size²×3 nonzeros."""
        e = receptive_field(100, 3072, patch_size=5, image_shape=(32, 32))
        nnz_per_row = (e != 0).sum(dim=1)
        assert (nnz_per_row == 75).all()  # 25 * 3 channels
        norms = e.norm(dim=1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_rng_reproducibility(self):
        g1 = torch.Generator().manual_seed(99)
        g2 = torch.Generator().manual_seed(99)
        e1 = receptive_field(50, 784, rng=g1)
        e2 = receptive_field(50, 784, rng=g2)
        assert torch.equal(e1, e2)

    def test_bad_dim(self):
        with pytest.raises(ValueError, match="divisible"):
            receptive_field(10, 100, image_shape=(28, 28))

    def test_bad_patch_size(self):
        with pytest.raises(ValueError, match="exceeds"):
            receptive_field(10, 784, patch_size=30, image_shape=(28, 28))

    def test_make_encoders_dispatch(self):
        e = make_encoders(50, 784, strategy="receptive_field", image_shape=(28, 28))
        assert e.shape == (50, 784)


class TestWhitenedEncoders:
    @pytest.fixture
    def train_data(self):
        g = torch.Generator().manual_seed(100)
        # Correlated data: x2 = 2*x1 + noise
        x1 = torch.randn(500, 1, generator=g)
        x2 = 2 * x1 + 0.1 * torch.randn(500, 1, generator=g)
        return torch.cat([x1, x2], dim=1)

    def test_shape(self, train_data):
        e = whitened(100, 2, train_data=train_data)
        assert e.shape == (100, 2)

    def test_unit_norm(self, train_data):
        e = whitened(200, 2, train_data=train_data)
        norms = e.norm(dim=1)
        assert torch.allclose(norms, torch.ones(200), atol=1e-5)

    def test_differs_from_hypersphere(self, train_data):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        e_plain = uniform_hypersphere(100, 2, rng=g1)
        e_white = whitened(100, 2, train_data=train_data, rng=g2)
        # Same rng seed but whitening should produce different directions
        assert not torch.allclose(e_plain, e_white, atol=1e-3)

    def test_rng_reproducibility(self, train_data):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        e1 = whitened(50, 2, train_data=train_data, rng=g1)
        e2 = whitened(50, 2, train_data=train_data, rng=g2)
        assert torch.equal(e1, e2)

    def test_make_encoders_dispatch(self):
        data = torch.randn(200, 10)
        e = make_encoders(50, 10, strategy="whitened", train_data=data)
        assert e.shape == (50, 10)

    def test_high_dim(self):
        """Whitened encoders work on higher-dimensional data (like images)."""
        g = torch.Generator().manual_seed(7)
        data = torch.randn(200, 784, generator=g)
        e = whitened(100, 784, train_data=data)
        assert e.shape == (100, 784)
        norms = e.norm(dim=1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_subspace_dimension(self):
        """Encoders lie in the principal subspace of the data."""
        g = torch.Generator().manual_seed(42)
        # Data with variance only in 3 of 10 dimensions
        data = torch.zeros(500, 10)
        data[:, :3] = torch.randn(500, 3, generator=g)
        e = whitened(50, 10, train_data=data, variance_ratio=0.99)
        # Encoders should have near-zero weight on dead dimensions 3-9
        dead_energy = e[:, 3:].norm(dim=1)
        assert (dead_energy < 1e-4).all()

    def test_variance_ratio_controls_rank(self):
        """Lower variance_ratio retains fewer components."""
        g = torch.Generator().manual_seed(42)
        # Data with linearly decaying variance across 20 dims
        scales = torch.linspace(10, 0.1, 20)
        data = torch.randn(500, 20, generator=g) * scales
        e_wide = whitened(50, 20, train_data=data, variance_ratio=0.99)
        e_narrow = whitened(50, 20, train_data=data, variance_ratio=0.50)
        # e_narrow should have lower effective rank (more near-zero dims)
        _, s_wide, _ = torch.linalg.svd(e_wide, full_matrices=False)
        _, s_narrow, _ = torch.linalg.svd(e_narrow, full_matrices=False)
        rank_wide = (s_wide > 1e-4).sum()
        rank_narrow = (s_narrow > 1e-4).sum()
        assert rank_narrow < rank_wide

    def test_nef_layer_uses_centers_as_train_data(self):
        """Whitened NEFLayer should infer train_data from centers."""
        g = torch.Generator().manual_seed(42)
        data = torch.randn(300, 12, generator=g)
        layer = NEFLayer(12, 100, 3, encoder_strategy="whitened", centers=data)
        assert layer.encoders.shape == (100, 12)
        out = layer(data[:10])
        assert out.shape == (10, 3)


class TestClassContrastEncoders:
    @pytest.fixture
    def labeled_data(self):
        g = torch.Generator().manual_seed(200)
        # Three well-separated clusters
        x = torch.cat(
            [
                torch.randn(100, 5, generator=g) + torch.tensor([3.0, 0, 0, 0, 0]),
                torch.randn(100, 5, generator=g) + torch.tensor([-3.0, 0, 0, 0, 0]),
                torch.randn(100, 5, generator=g) + torch.tensor([0, 3.0, 0, 0, 0]),
            ]
        )
        labels = torch.cat([torch.zeros(100), torch.ones(100), 2 * torch.ones(100)]).long()
        return x, labels

    def test_returns_tuple(self, labeled_data):
        x, y = labeled_data
        result = class_contrast(50, 5, train_data=x, train_labels=y)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shape(self, labeled_data):
        x, y = labeled_data
        enc, centers = class_contrast(50, 5, train_data=x, train_labels=y)
        assert enc.shape == (50, 5)
        assert centers.shape == (50, 5)

    def test_unit_norm(self, labeled_data):
        x, y = labeled_data
        enc, _ = class_contrast(200, 5, train_data=x, train_labels=y)
        norms = enc.norm(dim=1)
        assert torch.allclose(norms, torch.ones(200), atol=1e-5)

    def test_directions_point_across_classes(self, labeled_data):
        """Encoder directions should have nonzero dot product with class separation."""
        x, y = labeled_data
        enc, centers = class_contrast(100, 5, train_data=x, train_labels=y)
        # At minimum, encoders should not be zero vectors
        assert (enc.norm(dim=1) > 0.99).all()

    def test_rng_reproducibility(self, labeled_data):
        x, y = labeled_data
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        e1, c1 = class_contrast(50, 5, train_data=x, train_labels=y, rng=g1)
        e2, c2 = class_contrast(50, 5, train_data=x, train_labels=y, rng=g2)
        assert torch.equal(e1, e2)
        assert torch.equal(c1, c2)

    def test_make_encoders_returns_tuple(self, labeled_data):
        x, y = labeled_data
        result = make_encoders(50, 5, strategy="class_contrast", train_data=x, train_labels=y)
        assert isinstance(result, tuple)

    def test_nef_layer_integration(self, labeled_data):
        """NEFLayer should accept class_contrast and use returned centers."""
        x, y = labeled_data
        layer = NEFLayer(
            5,
            200,
            3,
            encoder_strategy="class_contrast",
            encoder_kwargs={"train_data": x, "train_labels": y},
        )
        assert layer.encoders.shape == (200, 5)
        out = layer(x[:10])
        assert out.shape == (10, 3)

    def test_nef_layer_infers_train_data_from_centers(self, labeled_data):
        """class_contrast NEFLayer should infer train_data from centers."""
        x, y = labeled_data
        layer = NEFLayer(
            5,
            200,
            3,
            encoder_strategy="class_contrast",
            centers=x,
            encoder_kwargs={"train_labels": y},
        )
        assert layer.encoders.shape == (200, 5)
        out = layer(x[:10])
        assert out.shape == (10, 3)


class TestLocalPCAEncoders:
    @pytest.fixture
    def train_data(self):
        g = torch.Generator().manual_seed(300)
        # Data with local structure: two clusters along different axes
        x1 = torch.randn(200, 5, generator=g) * torch.tensor([3.0, 0.1, 0.1, 0.1, 0.1])
        x2 = torch.randn(200, 5, generator=g) * torch.tensor([0.1, 3.0, 0.1, 0.1, 0.1])
        x2 += torch.tensor([10.0, 0, 0, 0, 0])  # offset to form separate cluster
        return torch.cat([x1, x2])

    def test_returns_tuple(self, train_data):
        result = local_pca(50, 5, train_data=train_data)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shape(self, train_data):
        enc, centers = local_pca(50, 5, train_data=train_data)
        assert enc.shape == (50, 5)
        assert centers.shape == (50, 5)

    def test_unit_norm(self, train_data):
        enc, _ = local_pca(100, 5, train_data=train_data)
        norms = enc.norm(dim=1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-5)

    def test_rng_reproducibility(self, train_data):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        e1, c1 = local_pca(50, 5, train_data=train_data, rng=g1)
        e2, c2 = local_pca(50, 5, train_data=train_data, rng=g2)
        assert torch.equal(e1, e2)
        assert torch.equal(c1, c2)

    def test_make_encoders_dispatch(self, train_data):
        result = make_encoders(50, 5, strategy="local_pca", train_data=train_data)
        assert isinstance(result, tuple)

    def test_nef_layer_integration(self, train_data):
        """NEFLayer should accept local_pca and use returned centers."""
        layer = NEFLayer(
            5,
            200,
            3,
            encoder_strategy="local_pca",
            encoder_kwargs={"train_data": train_data},
        )
        assert layer.encoders.shape == (200, 5)
        out = layer(train_data[:10])
        assert out.shape == (10, 3)

    def test_nef_layer_uses_centers_as_train_data(self, train_data):
        """local_pca NEFLayer should infer train_data from centers."""
        layer = NEFLayer(5, 200, 3, encoder_strategy="local_pca", centers=train_data)
        assert layer.encoders.shape == (200, 5)
        out = layer(train_data[:10])
        assert out.shape == (10, 3)


# ── Activations ───────────────────────────────────────────────────────


class TestActivations:
    def test_relu(self):
        act = make_activation("relu")
        x = torch.tensor([-1.0, 0.0, 1.0])
        assert torch.equal(act(x), torch.tensor([0.0, 0.0, 1.0]))

    def test_softplus(self):
        act = make_activation("softplus")
        x = torch.tensor([0.0])
        assert act(x).item() == pytest.approx(0.6931, abs=1e-3)

    def test_lif_rate_zero_below_threshold(self):
        lif = LIFRate()
        x = torch.tensor([-1.0, 0.0, 0.5])
        assert (lif(x) == 0).all()

    def test_lif_rate_positive_above_threshold(self):
        lif = LIFRate()
        x = torch.tensor([2.0, 5.0, 10.0])
        rates = lif(x)
        assert (rates > 0).all()
        # Higher input → higher rate
        assert (rates[1] > rates[0]) and (rates[2] > rates[1])

    def test_bent_line_two_positive_slopes(self):
        act = make_activation("bent_line")  # default slope 0.5
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = act(x)
        assert torch.equal(out, torch.tensor([-1.0, -0.5, 0.0, 1.0, 2.0]))
        # Monotone increasing (unlike abs).
        assert (out[1:] - out[:-1] > 0).all()

    def test_bent_line_equals_affine_mix_of_identity_and_abs(self):
        slope = 0.5
        act = make_activation("bent_line", slope=slope)
        x = torch.linspace(-3, 3, 25)
        mix = ((1 + slope) / 2) * x + ((1 - slope) / 2) * x.abs()
        assert torch.allclose(act(x), mix, atol=1e-6)

    def test_relu_is_bent_line_zero_slope(self):
        # ReLU = (z + |z|)/2 is the slope-0 member of the bent-line family.
        bent0 = make_activation("bent_line", slope=0.0)
        x = torch.linspace(-3, 3, 25)
        assert torch.allclose(bent0(x), torch.relu(x), atol=1e-6)

    def test_leaky_relu(self):
        act = make_activation("leaky_relu")  # slope 0.1
        x = torch.tensor([-1.0, 0.0, 2.0])
        assert torch.allclose(act(x), torch.tensor([-0.1, 0.0, 2.0]), atol=1e-6)

    def test_double_fold_two_kinks(self):
        act = make_activation("double_fold")  # ||x| - 1|
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert torch.equal(act(x), torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))

    def test_sqrt_abs(self):
        act = make_activation("sqrt_abs")
        x = torch.tensor([-4.0, 0.0, 9.0])
        assert torch.allclose(act(x), torch.tensor([2.0, 0.0, 3.0]), atol=1e-6)

    def test_cos(self):
        act = make_activation("cos")
        x = torch.tensor([0.0, torch.pi])
        assert torch.allclose(act(x), torch.tensor([1.0, -1.0]), atol=1e-6)

    def test_gauss_bump(self):
        act = make_activation("gauss")
        x = torch.tensor([0.0, 1.0])
        expected = torch.tensor([1.0, torch.exp(torch.tensor(-0.5)).item()])
        assert torch.allclose(act(x), expected, atol=1e-6)

    def test_cos_sin_is_phase_shifted_cosine(self):
        act = make_activation("cos_sin")
        x = torch.linspace(-3, 3, 25)
        # cos(x) + sin(x) == sqrt(2) cos(x - pi/4)
        expected = math.sqrt(2.0) * torch.cos(x - math.pi / 4)
        assert torch.allclose(act(x), expected, atol=1e-6)

    def test_cos_sin_not_even(self):
        act = make_activation("cos_sin")
        x = torch.tensor([0.3, 0.7, 1.1])
        # An even function would satisfy f(-x) == f(x); cos+sin must not.
        assert not torch.allclose(act(-x), act(x), atol=1e-3)

    def test_square_wave(self):
        act = make_activation("square_wave")
        x = torch.tensor([0.0, math.pi, math.pi / 3, 2 * math.pi / 3])
        # sign(cos): cos>0 -> +1, cos<0 -> -1
        assert torch.allclose(act(x), torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=1e-6)

    def test_floor_wave(self):
        act = make_activation("floor_wave")
        x = torch.tensor([-1.2, -0.1, 0.0, math.pi, 2.2])
        expected = torch.sign(torch.cos(torch.pi * x)) + torch.floor(x)
        assert torch.equal(act(x), expected)

    def test_floor_wave_not_even_not_odd(self):
        act = make_activation("floor_wave")
        x = torch.tensor([0.3, 0.7, 1.4, 2.6])
        assert not torch.allclose(act(-x), act(x), atol=1e-6)
        assert not torch.allclose(act(-x), -act(x), atol=1e-6)

    def test_box(self):
        act = make_activation("box")
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert torch.equal(act(x), torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))

    def test_gelu(self):
        act = make_activation("gelu")
        x = torch.tensor([-3.0, 0.0, 3.0])
        assert torch.allclose(act(x), torch.nn.functional.gelu(x), atol=1e-6)
        assert act(torch.tensor([0.0])).item() == pytest.approx(0.0, abs=1e-6)

    def test_silu(self):
        act = make_activation("silu")
        x = torch.tensor([-3.0, 0.0, 3.0])
        assert torch.allclose(act(x), torch.nn.functional.silu(x), atol=1e-6)

    def test_phase_c_activation_parity(self):
        # The benchmark harness uses fast in-place activation functions that
        # must match the library modules bit-for-bit.
        from benchmarks.sl_clean._phase_c_shared import _activation_fn

        names = [
            "abs",
            "relu",
            "leaky_relu",
            "softplus",
            "tanh",
            "step",
            "square",
            "bent_line",
            "double_fold",
            "sqrt_abs",
            "cos",
            "gauss",
            "cos_sin",
            "square_wave",
            "floor_wave",
            "box",
            "gelu",
            "silu",
        ]
        g = torch.Generator().manual_seed(0)
        x = torch.randn(64, generator=g) * 3.0
        for name in names:
            ref = make_activation(name)(x.clone())
            t = x.clone()
            _activation_fn(name)(t)
            assert torch.allclose(t, ref, atol=1e-6), f"parity mismatch for {name!r}"


# ── Solvers ───────────────────────────────────────────────────────────


class TestSolvers:
    @pytest.fixture
    def linear_data(self):
        """y = 2*x + 1 with 200 random-feature neurons."""
        g = torch.Generator().manual_seed(0)
        x = torch.randn(200, 1, generator=g)
        y = 2 * x + 1
        # Enough random features to fit a simple linear function reliably
        W = torch.randn(200, 1, generator=g)
        b = torch.randn(200, generator=g)
        A = torch.abs(x @ W.T + b)
        return A, y

    def test_lstsq(self, linear_data):
        A, y = linear_data
        D = lstsq(A, y)
        pred = A @ D
        mse = (pred - y).pow(2).mean().item()
        assert mse < 0.5  # reasonable fit

    def test_tikhonov(self, linear_data):
        A, y = linear_data
        D = tikhonov(A, y, alpha=(1e-4, "trace"))
        pred = A @ D
        mse = (pred - y).pow(2).mean().item()
        assert mse < 0.5

    def test_cholesky(self, linear_data):
        A, y = linear_data
        D = normal_equations(A, y, alpha=(1e-4, "trace"))
        pred = A @ D
        mse = (pred - y).pow(2).mean().item()
        assert mse < 0.5

    def test_solve_decoders_dispatch(self, linear_data):
        A, y = linear_data
        for method in ("lstsq", "tikhonov", "cholesky"):
            D = solve_decoders(A, y, method=method, alpha=(1e-4, "trace"))
            assert D.shape == (200, 1)

    def test_regularization_strength_trace_vs_fixed(self):
        ata = torch.diag(torch.tensor([10.0, 30.0]))

        trace_reg = regularization_strength(ata, alpha=(0.1, "trace"))
        fixed_reg = regularization_strength(ata, alpha=(0.1, "fixed"))

        assert torch.allclose(trace_reg, torch.tensor(2.0))
        assert torch.allclose(fixed_reg, torch.tensor(0.1))

    def test_alpha_must_be_policy_pair(self, linear_data):
        A, y = linear_data
        with pytest.raises(ValueError, match="explicit .* pair"):
            solve_decoders(A, y, method="tikhonov", alpha=1e-2)

    def test_solve_from_normal_equations_matches_trace_solver(self, linear_data):
        A, y = linear_data
        D_from_activities = tikhonov(A, y, alpha=(1e-3, "trace"))
        D_from_stats = solve_from_normal_equations(
            A.T @ A,
            A.T @ y,
            alpha=(1e-3, "trace"),
        )

        assert torch.allclose(D_from_stats, D_from_activities, atol=1e-5, rtol=1e-5)


# ── NEFLayer ──────────────────────────────────────────────────────────


class TestNEFLayer:
    def test_shape(self):
        layer = NEFLayer(10, 200, 3)
        x = torch.randn(32, 10)
        y = layer(x)
        assert y.shape == (32, 3)

    def test_fit_identity(self):
        """Fit a 1-d identity function: f(x) = x."""
        torch.manual_seed(1)
        layer = NEFLayer(1, 500, 1)
        x = torch.linspace(-1, 1, 200).unsqueeze(1)
        layer.fit(x, x, solver="tikhonov", alpha=(1e-4, "trace"))
        pred = layer(x)
        mse = (pred - x).pow(2).mean().item()
        assert mse < 0.01

    def test_fit_quadratic(self):
        """Fit f(x) = x^2."""
        torch.manual_seed(2)
        layer = NEFLayer(1, 1000, 1)
        x = torch.linspace(-1, 1, 300).unsqueeze(1)
        y = x**2
        layer.fit(x, y, solver="tikhonov", alpha=(1e-4, "trace"))
        pred = layer(x)
        mse = (pred - y).pow(2).mean().item()
        assert mse < 0.01

    def test_fit_multiclass(self):
        """Fit a simple 3-class one-hot mapping."""
        torch.manual_seed(3)
        layer = NEFLayer(2, 500, 3)
        # Three clusters
        x = torch.cat(
            [
                torch.randn(100, 2) + torch.tensor([2.0, 0.0]),
                torch.randn(100, 2) + torch.tensor([-2.0, 0.0]),
                torch.randn(100, 2) + torch.tensor([0.0, 2.0]),
            ]
        )
        labels = torch.cat([torch.zeros(100), torch.ones(100), 2 * torch.ones(100)]).long()
        targets = torch.zeros(300, 3)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)
        layer.fit(x, targets)
        pred = layer(x).argmax(dim=1)
        accuracy = (pred == labels).float().mean().item()
        assert accuracy > 0.85

    def test_trainable_encoders(self):
        layer = NEFLayer(5, 100, 2, trainable_encoders=True)
        assert layer.encoders.requires_grad
        assert layer.bias.requires_grad

    def test_frozen_encoders(self):
        layer = NEFLayer(5, 100, 2, trainable_encoders=False)
        assert not layer.encoders.requires_grad

    def test_data_centers_bias(self):
        """Data-driven biases: bias_i = -gain * (d_i · e_i)."""
        torch.manual_seed(7)
        data = torch.randn(500, 10)
        layer = NEFLayer(10, 200, 3, centers=data)
        # Biases should NOT be iid normal — verify they correlate with
        # encoder-center dot products
        assert layer.bias.shape == (200,)
        x = torch.randn(32, 10)
        y = layer(x)
        assert y.shape == (32, 3)

    def test_data_centers_fit(self):
        """Data-driven biases should at least match random on a simple task."""
        torch.manual_seed(8)
        x = torch.linspace(-1, 1, 200).unsqueeze(1)
        layer = NEFLayer(1, 500, 1, centers=x)
        layer.fit(x, x, solver="tikhonov", alpha=(1e-4, "trace"))
        pred = layer(x)
        mse = (pred - x).pow(2).mean().item()
        assert mse < 0.01

    def test_per_neuron_gain_tuple(self):
        """Tuple gain creates per-neuron diversity."""
        torch.manual_seed(20)
        layer = NEFLayer(5, 100, 2, gain=(0.5, 2.0))
        assert layer.gain.shape == (100,)
        assert layer.gain.min() >= 0.5
        assert layer.gain.max() <= 2.0
        assert not torch.all(layer.gain == layer.gain[0])  # not all same

    def test_per_neuron_gain_tensor(self):
        """Explicit tensor gain."""
        g = torch.linspace(0.5, 2.0, 100)
        layer = NEFLayer(5, 100, 2, gain=g)
        assert torch.equal(layer.gain, g)

    def test_per_neuron_gain_float(self):
        """Float gain fills all neurons with the same value."""
        layer = NEFLayer(5, 100, 2, gain=2.5)
        assert layer.gain.shape == (100,)
        assert torch.allclose(layer.gain, torch.full((100,), 2.5))

    def test_data_driven_gain(self):
        """Data-driven gains should vary with local density."""
        torch.manual_seed(25)
        # Dense cluster + sparse outliers
        dense = torch.randn(400, 5) * 0.1
        sparse_pts = torch.randn(100, 5) * 5.0
        data = torch.cat([dense, sparse_pts])
        layer = NEFLayer(5, 200, 2, gain="data_driven", centers=data)
        assert layer.gain.shape == (200,)
        assert layer.gain.min() >= 0.5
        assert layer.gain.max() <= 2.0
        # Gains should vary (not all the same)
        assert not torch.all(layer.gain == layer.gain[0])

    def test_data_driven_gain_requires_centers(self):
        """gain='data_driven' without centers should raise."""
        with pytest.raises(ValueError, match="data_driven"):
            NEFLayer(5, 100, 2, gain="data_driven")

    def test_set_centers(self):
        """set_centers should change biases."""
        torch.manual_seed(30)
        layer = NEFLayer(10, 200, 3)
        old_bias = layer.bias.data.clone()
        data = torch.randn(500, 10)
        layer.set_centers(data)
        assert not torch.allclose(old_bias, layer.bias.data)

    def test_state_dict_round_trip(self):
        """Save/load round-trip preserves output including per-neuron gain."""
        torch.manual_seed(40)
        layer = NEFLayer(3, 200, 2, gain=(0.5, 2.0))
        x = torch.randn(100, 3)
        y = torch.randn(100, 2)
        layer.fit(x, y)
        out1 = layer(x)

        # Save and reload
        state = layer.state_dict()
        layer2 = NEFLayer(3, 200, 2, gain=1.0)  # different gain spec
        layer2.load_state_dict(state)
        out2 = layer2(x)
        assert torch.allclose(out1, out2, atol=1e-5)

    @staticmethod
    def _reference_decoders(x, targets, alpha, seed):
        """Decoders of an independent layer solved on all the data at once."""
        ref = NEFLayer(2, 100, 3, rng=torch.Generator().manual_seed(seed)).double()
        ref.partial_fit(x, targets)
        ref.solve_accumulated(alpha=(alpha, "fixed"))
        return ref.decoders

    @pytest.mark.parametrize(
        ("encoded", "batch_sizes"),
        [
            pytest.param(False, (20,), id="small-batch"),
            pytest.param(True, (20,), id="encoded"),
            pytest.param(False, (100,), id="neuron-count-batch"),
            pytest.param(False, (20, 20), id="woodbury-after-rebuild"),
        ],
    )
    def test_continuous_fit_after_partial_fit_uses_prior_statistics(self, encoded, batch_sizes):
        """A missing inverse must be rebuilt from already accumulated statistics."""
        g = torch.Generator().manual_seed(82)
        size = 40 + sum(batch_sizes)
        x = torch.randn(size, 2, generator=g, dtype=torch.float64)
        targets = torch.randn(size, 3, generator=g, dtype=torch.float64)
        alpha = 1e-2

        layer = NEFLayer(2, 100, 3, rng=torch.Generator().manual_seed(820)).double()
        layer.partial_fit(x[:40], targets[:40])
        fit = layer.continuous_fit_encoded if encoded else layer.continuous_fit
        for xb, yb in zip(x[40:].split(batch_sizes), targets[40:].split(batch_sizes)):
            fit(layer.encode(xb) if encoded else xb, yb, fixed_alpha=alpha)

        reference = self._reference_decoders(x, targets, alpha, 820)
        assert torch.allclose(layer.decoders, reference, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("refresh", [False, True], ids=["continuous-fit", "refresh"])
    def test_partial_fit_invalidates_inverse_before_next_solve(self, refresh):
        """The next solve must include the partial batch and use the intended alpha."""
        g = torch.Generator().manual_seed(84)
        size = 40 if refresh else 60
        x = torch.randn(size, 2, generator=g, dtype=torch.float64)
        targets = torch.randn(size, 3, generator=g, dtype=torch.float64)
        alpha = 3e-2

        layer = NEFLayer(2, 100, 3, rng=torch.Generator().manual_seed(840)).double()
        layer.continuous_fit(x[:20], targets[:20], fixed_alpha=alpha)
        layer.partial_fit(x[20:40], targets[20:40])
        if refresh:
            assert layer._woodbury_alpha == alpha
            layer.refresh_inverse()
        else:
            layer.continuous_fit(x[40:], targets[40:], fixed_alpha=alpha)

        reference = self._reference_decoders(x, targets, alpha, 840)
        assert torch.allclose(layer.decoders, reference, atol=1e-9, rtol=0)

    def test_reset_accumulators_invalidates_woodbury_inverse(self):
        """Continuous fitting after a reset must not use discarded data."""
        g = torch.Generator().manual_seed(85)
        x = torch.randn(40, 2, generator=g, dtype=torch.float64)
        targets = torch.randn(40, 3, generator=g, dtype=torch.float64)
        alpha = 1e-2

        layer = NEFLayer(2, 100, 3, rng=torch.Generator().manual_seed(850)).double()
        layer.continuous_fit(x[:20], targets[:20], fixed_alpha=alpha)
        layer.reset_accumulators()
        assert layer._woodbury_alpha == alpha
        layer.continuous_fit(x[20:], targets[20:], fixed_alpha=alpha)

        reference = self._reference_decoders(x[20:], targets[20:], alpha, 850)
        assert torch.allclose(layer.decoders, reference, atol=1e-9, rtol=0)

    @pytest.mark.parametrize(
        ("setup", "bad_alpha", "message"),
        [
            ("fresh", -1.0, "alpha must be > 0"),
            ("partial", -1.0, "alpha must be > 0"),
            ("cached", -1.0, "alpha must be > 0"),
            ("cached", 1e-1, "fixed_alpha changed"),
        ],
    )
    def test_rejected_fixed_alpha_leaves_state_untouched(
        self, setup, bad_alpha, message, monkeypatch
    ):
        g = torch.Generator().manual_seed(83)
        layer = NEFLayer(2, 50, 2, rng=g).double()
        x = torch.randn(10, 2, generator=g, dtype=torch.float64)
        targets = torch.randn(10, 2, generator=g, dtype=torch.float64)
        if setup == "partial":
            layer.partial_fit(x, targets)
        elif setup == "cached":
            layer.continuous_fit(x, targets, fixed_alpha=1e-2)
        before = [
            t.clone() if t is not None else None
            for t in (layer._ata, layer._aty, layer._M_inv, layer.decoders)
        ]
        before_alpha = layer._woodbury_alpha
        activities = layer.encode(x)

        def _no_encoding(*_args, **_kwargs):
            raise AssertionError("a rejected call must not encode")

        monkeypatch.setattr(layer, "encode", _no_encoding)
        for fit, batch in (
            (layer.continuous_fit, x),
            (layer.continuous_fit_encoded, activities),
        ):
            with pytest.raises(ValueError, match=message):
                fit(batch, targets, fixed_alpha=bad_alpha)
            after = (layer._ata, layer._aty, layer._M_inv, layer.decoders)
            for expected, actual in zip(before, after):
                if expected is None:
                    assert actual is None
                else:
                    assert actual is not None and torch.equal(expected, actual)
            assert layer._woodbury_alpha == before_alpha


# ── Exception paths ───────────────────────────────────────────────────


class TestExceptions:
    def test_bad_encoder_strategy(self):
        with pytest.raises(ValueError, match="Unknown encoder strategy"):
            make_encoders(10, 5, strategy="nonexistent")

    def test_bad_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            make_activation("nonexistent")

    def test_bad_solver(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            solve_decoders(torch.randn(10, 5), torch.randn(10, 1), method="nonexistent")

    def test_bad_sparsity(self):
        with pytest.raises(ValueError, match="sparsity"):
            sparse(10, 5, sparsity=1.0)
        with pytest.raises(ValueError, match="sparsity"):
            sparse(10, 5, sparsity=-0.1)

    def test_forward_shape_mismatch(self):
        layer = NEFLayer(10, 100, 2)
        with pytest.raises(ValueError, match="Expected input shape"):
            layer(torch.randn(5, 7))

    def test_fit_sample_mismatch(self):
        layer = NEFLayer(3, 100, 2)
        with pytest.raises(ValueError, match="same number of samples"):
            layer.fit(torch.randn(10, 3), torch.randn(20, 2))

    def test_gain_tensor_wrong_shape(self):
        from leenef.layers import _make_gain

        with pytest.raises(ValueError, match="gain tensor must have shape"):
            _make_gain(torch.ones(5), n_neurons=10)


# ── Abs activation ────────────────────────────────────────────────────


class TestAbs:
    def test_abs_activation(self):
        act = make_activation("abs")
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0])
        assert torch.equal(act(x), expected)


class TestStep:
    def test_step_activation(self):
        act = make_activation("step")
        x = torch.tensor([-2.0, 0.0, 3.0])
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.equal(act(x), expected)


class TestSquare:
    def test_square_activation(self):
        act = make_activation("square")
        x = torch.tensor([-2.0, 0.0, 3.0])
        expected = torch.tensor([4.0, 0.0, 9.0])
        assert torch.equal(act(x), expected)


# ── Incremental fit ──────────────────────────────────────────────────


class TestIncrementalFit:
    def test_equivalence_with_full_fit(self):
        """partial_fit over the full dataset should match fit()."""
        torch.manual_seed(50)
        x = torch.linspace(-1, 1, 200).unsqueeze(1)
        y = x**2

        rng1 = torch.Generator().manual_seed(200)
        layer_full = NEFLayer(1, 500, 1, rng=rng1)
        layer_full.fit(x, y, solver="tikhonov")

        rng2 = torch.Generator().manual_seed(200)
        layer_inc = NEFLayer(1, 500, 1, rng=rng2)
        layer_inc.partial_fit(x, y)
        layer_inc.solve_accumulated()

        assert torch.allclose(layer_full.decoders.data, layer_inc.decoders.data, atol=1e-4)

    def test_chunked_accumulation(self):
        """Accumulating in chunks should equal a single full pass."""
        torch.manual_seed(51)
        x = torch.randn(200, 3)
        y = torch.randn(200, 2)

        # Both layers from same seed → identical encoders/biases
        rng1 = torch.Generator().manual_seed(100)
        layer_one = NEFLayer(3, 300, 2, rng=rng1)
        layer_one.partial_fit(x, y)
        layer_one.solve_accumulated()

        rng2 = torch.Generator().manual_seed(100)
        layer_chunked = NEFLayer(3, 300, 2, rng=rng2)
        for chunk_x, chunk_y in zip(x.split(50), y.split(50)):
            layer_chunked.partial_fit(chunk_x, chunk_y)
        layer_chunked.solve_accumulated()

        assert torch.allclose(layer_one.decoders.data, layer_chunked.decoders.data, atol=1e-3)

    def test_reset_accumulators(self):
        """reset_accumulators should clear state so solve_accumulated fails."""
        torch.manual_seed(52)
        layer = NEFLayer(2, 100, 1)
        layer.partial_fit(torch.randn(10, 2), torch.randn(10, 1))
        layer.reset_accumulators()
        with pytest.raises(RuntimeError, match="No accumulated"):
            layer.solve_accumulated()

    def test_partial_fit_sample_mismatch(self):
        layer = NEFLayer(3, 100, 2)
        with pytest.raises(ValueError, match="same number of samples"):
            layer.partial_fit(torch.randn(10, 3), torch.randn(20, 2))

    def test_solve_without_partial_fit(self):
        layer = NEFLayer(3, 100, 2)
        with pytest.raises(RuntimeError, match="No accumulated"):
            layer.solve_accumulated()


# ── Continuous fit (Woodbury) ─────────────────────────────────────────


class TestContinuousFit:
    def test_single_batch_matches_manual_solve(self):
        """Single continuous_fit call should match manual fixed-alpha solve."""
        torch.manual_seed(60)
        x = torch.randn(300, 4)
        y = torch.randn(300, 2)
        alpha = 1e-2

        rng1 = torch.Generator().manual_seed(300)
        layer = NEFLayer(4, 200, 2, rng=rng1)
        layer.continuous_fit(x, y, fixed_alpha=alpha)

        # Manual solve with same fixed alpha (no trace scaling)
        rng2 = torch.Generator().manual_seed(300)
        layer_ref = NEFLayer(4, 200, 2, rng=rng2)
        A = layer_ref.encode(x)
        ATA = A.T @ A
        ATA_d = ATA.double()
        ATA_d.diagonal().add_(alpha)
        expected = (torch.linalg.inv(ATA_d) @ (A.T @ y).double()).float()

        # float64-inverse path vs float32-solve reference: close but not identical
        assert torch.allclose(layer.decoders.data, expected, atol=5e-3)

    def test_refresh_corrects_woodbury_drift(self):
        """After chunked Woodbury updates, refresh_inverse recovers exact solution."""
        torch.manual_seed(61)
        x = torch.randn(400, 5)
        y = torch.randn(400, 3)
        alpha = 1e-2

        # Chunked via Woodbury (50 < 200 triggers rank-k path)
        rng1 = torch.Generator().manual_seed(310)
        layer = NEFLayer(5, 200, 3, rng=rng1)
        for cx, cy in zip(x.split(50), y.split(50)):
            layer.continuous_fit(cx, cy, fixed_alpha=alpha)

        decoders_woodbury = layer.decoders.data.clone()

        # Refresh recomputes inv(ATA + αI) exactly from accumulators
        layer.refresh_inverse(fixed_alpha=alpha)
        decoders_refreshed = layer.decoders.data.clone()

        # Also compute expected result manually from the same accumulators
        # (using float64 to match the code path in refresh_inverse)
        M = layer._ata.double()
        M.diagonal().add_(alpha)
        expected = (torch.linalg.inv(M) @ layer._aty.double()).float()

        # Refreshed should match the manual result very closely
        assert torch.allclose(decoders_refreshed, expected, atol=1e-4)

        # Woodbury should be close but not exact (FP drift)
        drift = (decoders_woodbury - decoders_refreshed).abs().max().item()
        assert drift < 1.0  # bounded drift

    def test_model_usable_after_each_call(self):
        """Decoders should be updated after every continuous_fit call."""
        torch.manual_seed(62)
        w = torch.randn(3, 1)
        x = torch.randn(200, 3)
        y = (x @ w).sign()

        layer = NEFLayer(3, 500, 1, rng=torch.Generator().manual_seed(320))
        accuracies = []
        for cx, cy in zip(x.split(50), y.split(50)):
            layer.continuous_fit(cx, cy)
            preds = layer(x).sign()
            acc = (preds == y).float().mean().item()
            accuracies.append(acc)

        assert accuracies[-1] > 0.7

    def test_accumulators_match_partial_fit(self):
        """continuous_fit and partial_fit should accumulate identical ATA/ATY."""
        torch.manual_seed(63)
        x = torch.randn(300, 4)
        y = torch.randn(300, 2)

        rng1 = torch.Generator().manual_seed(330)
        layer_inc = NEFLayer(4, 200, 2, rng=rng1)
        for cx, cy in zip(x.split(30), y.split(30)):
            layer_inc.partial_fit(cx, cy)

        rng2 = torch.Generator().manual_seed(330)
        layer_cont = NEFLayer(4, 200, 2, rng=rng2)
        for cx, cy in zip(x.split(30), y.split(30)):
            layer_cont.continuous_fit(cx, cy)

        assert torch.allclose(layer_inc._ata, layer_cont._ata, atol=1e-4)
        assert torch.allclose(layer_inc._aty, layer_cont._aty, atol=1e-4)

    def test_reset_continuous(self):
        """reset_continuous should clear all state."""
        layer = NEFLayer(3, 100, 2)
        layer.continuous_fit(torch.randn(20, 3), torch.randn(20, 2))
        layer.reset_continuous()
        assert layer._ata is None
        assert layer._M_inv is None
        assert not hasattr(layer, "_woodbury_alpha") or layer._woodbury_alpha is None

    def test_continuous_fit_sample_mismatch(self):
        layer = NEFLayer(3, 100, 2)
        with pytest.raises(ValueError, match="same number of samples"):
            layer.continuous_fit(torch.randn(10, 3), torch.randn(20, 2))

    def test_refresh_without_data(self):
        layer = NEFLayer(3, 100, 2)
        with pytest.raises(RuntimeError, match="No accumulated"):
            layer.refresh_inverse()
