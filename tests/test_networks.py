"""Tests for multi-layer NEFNetwork."""

import torch
import pytest

from leenef.networks import NEFNetwork


def _make_net(d_in=4, d_out=2, hidden=None, output_n=200, **kw):
    hidden = hidden or [100]
    return NEFNetwork(d_in, d_out, hidden_neurons=hidden,
                      output_neurons=output_n,
                      rng=torch.Generator().manual_seed(42), **kw)


# ---- Architecture ----

class TestArchitecture:
    def test_forward_shape(self):
        net = _make_net()
        x = torch.randn(16, 4)
        y = net(x)
        assert y.shape == (16, 2)

    def test_multi_hidden(self):
        net = _make_net(hidden=[100, 80])
        assert len(net.hidden) == 2
        x = torch.randn(8, 4)
        assert net(x).shape == (8, 2)

    def test_no_hidden(self):
        net = _make_net(hidden=[])
        x = torch.randn(8, 4)
        assert net(x).shape == (8, 2)

    def test_encode_hidden_shape(self):
        net = _make_net(hidden=[100, 80])
        x = torch.randn(8, 4)
        h = net._encode_hidden(x)
        assert h.shape == (8, 80)  # last hidden layer's n_neurons


# ---- Strategy A: greedy ----

class TestGreedy:
    def test_fit_identity(self):
        torch.manual_seed(0)
        net = _make_net(d_in=3, d_out=3, hidden=[200], output_n=500)
        x = torch.randn(500, 3)
        net.fit_greedy(x, x)
        with torch.no_grad():
            err = (net(x) - x).pow(2).mean().item()
        assert err < 0.1, f"identity MSE too high: {err}"

    def test_fit_quadratic(self):
        torch.manual_seed(1)
        x = torch.rand(600, 1) * 2 - 1
        y = x ** 2
        net = _make_net(d_in=1, d_out=1, hidden=[300], output_n=500)
        net.fit_greedy(x, y)
        with torch.no_grad():
            err = (net(x) - y).pow(2).mean().item()
        assert err < 0.05, f"quadratic MSE too high: {err}"

    def test_multiclass(self):
        torch.manual_seed(2)
        x = torch.randn(400, 4)
        labels = (x[:, 0] > 0).long()
        targets = torch.zeros(400, 2).scatter_(1, labels.unsqueeze(1), 1.0)
        net = _make_net(d_in=4, d_out=2, hidden=[200], output_n=400)
        net.fit_greedy(x, targets)
        with torch.no_grad():
            preds = net(x).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        assert acc > 0.90, f"accuracy too low: {acc}"


# ---- Strategy B: hybrid ----

class TestHybrid:
    def test_improves_over_greedy(self):
        torch.manual_seed(3)
        x = torch.rand(300, 2) * 2 - 1
        y = torch.sin(x[:, :1] * 3) + torch.cos(x[:, 1:] * 2)

        greedy = _make_net(d_in=2, d_out=1, hidden=[200], output_n=300)
        greedy.fit_greedy(x, y)
        with torch.no_grad():
            greedy_err = (greedy(x) - y).pow(2).mean().item()

        hybrid = _make_net(d_in=2, d_out=1, hidden=[200], output_n=300)
        hybrid.fit_hybrid(x, y, n_iters=20, lr=5e-4)
        with torch.no_grad():
            hybrid_err = (hybrid(x) - y).pow(2).mean().item()

        assert hybrid_err < greedy_err, \
            f"hybrid ({hybrid_err:.4f}) should beat greedy ({greedy_err:.4f})"

    def test_multiclass(self):
        torch.manual_seed(4)
        x = torch.randn(400, 4)
        labels = (x[:, 0] + x[:, 1] > 0).long()
        targets = torch.zeros(400, 2).scatter_(1, labels.unsqueeze(1), 1.0)
        net = _make_net(d_in=4, d_out=2, hidden=[200], output_n=400)
        net.fit_hybrid(x, targets, n_iters=10, lr=1e-3)
        with torch.no_grad():
            acc = (net(x).argmax(1) == labels).float().mean().item()
        assert acc > 0.90, f"hybrid classification accuracy too low: {acc}"


# ---- Strategy C: end-to-end ----

class TestEndToEnd:
    def test_improves_over_greedy(self):
        torch.manual_seed(5)
        x = torch.rand(300, 2) * 2 - 1
        y = torch.sin(x[:, :1] * 3) + torch.cos(x[:, 1:] * 2)

        greedy = _make_net(d_in=2, d_out=1, hidden=[200], output_n=300)
        greedy.fit_greedy(x, y)
        with torch.no_grad():
            greedy_err = (greedy(x) - y).pow(2).mean().item()

        e2e = _make_net(d_in=2, d_out=1, hidden=[200], output_n=300)
        e2e.fit_end_to_end(x, y, n_epochs=30, lr=1e-3, batch_size=64)
        with torch.no_grad():
            e2e_err = (e2e(x) - y).pow(2).mean().item()

        assert e2e_err < greedy_err, \
            f"end-to-end ({e2e_err:.4f}) should beat greedy ({greedy_err:.4f})"

    def test_decoders_frozen_after(self):
        """Decoders should have requires_grad=False after fit_end_to_end."""
        net = _make_net()
        x = torch.randn(32, 4)
        y = torch.randn(32, 2)
        net.fit_end_to_end(x, y, n_epochs=1, batch_size=32)
        assert not net.output.decoders.requires_grad
