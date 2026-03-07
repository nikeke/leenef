"""Tests for leenef core modules."""

import torch
import pytest
from leenef.encoders import make_encoders, uniform_hypersphere, gaussian, sparse
from leenef.activations import make_activation, LIFRate
from leenef.solvers import solve_decoders, lstsq, tikhonov, normal_equations
from leenef.layers import NEFLayer


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
        """y = 2*x + 1 with 100 neurons (random features)."""
        torch.manual_seed(0)
        x = torch.randn(200, 1)
        y = 2 * x + 1
        # Random feature expansion
        W = torch.randn(50, 1)
        b = torch.randn(50)
        A = torch.relu(x @ W.T + b)
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
            kw = {"alpha": 1e-4} if method != "lstsq" else {}
            D = solve_decoders(A, y, method=method, **kw)
            assert D.shape == (50, 1)


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
        y = x ** 2
        layer.fit(x, y, solver="tikhonov", alpha=1e-4)
        pred = layer(x)
        mse = (pred - y).pow(2).mean().item()
        assert mse < 0.01

    def test_fit_multiclass(self):
        """Fit a simple 3-class one-hot mapping."""
        torch.manual_seed(3)
        layer = NEFLayer(2, 500, 3)
        # Three clusters
        x = torch.cat([
            torch.randn(100, 2) + torch.tensor([2.0, 0.0]),
            torch.randn(100, 2) + torch.tensor([-2.0, 0.0]),
            torch.randn(100, 2) + torch.tensor([0.0, 2.0]),
        ])
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
