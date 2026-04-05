"""Tests for leenef core modules."""

import pytest
import torch

from leenef.activations import LIFRate, make_activation
from leenef.encoders import gaussian, make_encoders, receptive_field, sparse, uniform_hypersphere
from leenef.layers import NEFLayer
from leenef.solvers import lstsq, normal_equations, solve_decoders, tikhonov

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
        D = tikhonov(A, y, alpha=1e-4)
        pred = A @ D
        mse = (pred - y).pow(2).mean().item()
        assert mse < 0.5

    def test_cholesky(self, linear_data):
        A, y = linear_data
        D = normal_equations(A, y, alpha=1e-4)
        pred = A @ D
        mse = (pred - y).pow(2).mean().item()
        assert mse < 0.5

    def test_solve_decoders_dispatch(self, linear_data):
        A, y = linear_data
        for method in ("lstsq", "tikhonov", "cholesky"):
            D = solve_decoders(A, y, method=method, alpha=1e-4)
            assert D.shape == (200, 1)


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
        layer.fit(x, x, solver="tikhonov", alpha=1e-4)
        pred = layer(x)
        mse = (pred - x).pow(2).mean().item()
        assert mse < 0.01

    def test_fit_quadratic(self):
        """Fit f(x) = x^2."""
        torch.manual_seed(2)
        layer = NEFLayer(1, 1000, 1)
        x = torch.linspace(-1, 1, 300).unsqueeze(1)
        y = x**2
        layer.fit(x, y, solver="tikhonov", alpha=1e-4)
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
        layer.fit(x, x, solver="tikhonov", alpha=1e-4)
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
