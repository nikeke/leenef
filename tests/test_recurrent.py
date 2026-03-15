"""Tests for RecurrentNEFLayer."""

import pytest
import torch

from leenef.recurrent import RecurrentNEFLayer, _temporal_projection_matrix


class TestRecurrentShapes:
    """Output shape tests for RecurrentNEFLayer."""

    def test_forward_shape(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3)
        seq = torch.randn(4, 5, 8)
        y = layer(seq)
        assert y.shape == (4, 3)

    def test_encode_step_shape(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3, d_state=4)
        u = torch.randn(4, 8)
        s = torch.zeros(4, 4)
        a = layer.encode_step(u, s)
        assert a.shape == (4, 50)

    def test_custom_d_state(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3, d_state=16)
        assert layer.d_state == 16
        assert layer.state_decoders.shape == (50, 16)
        assert layer.encoders.shape == (50, 8 + 16)

    def test_default_d_state_equals_d_in(self):
        layer = RecurrentNEFLayer(d_in=12, n_neurons=50, d_out=3)
        assert layer.d_state == 12


class TestRecurrentExceptions:
    """Validation and exception-path tests."""

    def test_invalid_d_state(self):
        with pytest.raises(ValueError, match="d_state must be positive"):
            RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2, d_state=0)

    def test_forward_rejects_bad_sequence_shape(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2)
        with pytest.raises(ValueError, match="Expected seq shape"):
            layer(torch.randn(8, 4))

    def test_encode_step_rejects_bad_state_shape(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2, d_state=3)
        with pytest.raises(ValueError, match="Expected s_prev shape"):
            layer.encode_step(torch.randn(8, 4), torch.randn(8, 4))

    def test_fit_greedy_rejects_target_shape(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2)
        seq = torch.randn(8, 3, 4)
        with pytest.raises(ValueError, match="Expected targets shape"):
            layer.fit_greedy(seq, torch.randn(8, 3))

    def test_tp_rejects_unsupported_solver(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2)
        seq = torch.randn(8, 3, 4)
        targets = torch.randn(8, 2)
        with pytest.raises(ValueError, match="supports only"):
            layer.fit_target_prop(seq, targets, n_iters=1, solver="lstsq")


class TestRecurrentGreedy:
    """Greedy training tests."""

    def test_greedy_solves_decoders(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=100, d_out=3)
        seq = torch.randn(32, 5, 8)
        targets = torch.randn(32, 3)
        layer.fit_greedy(seq, targets, n_iters=3)
        assert (layer.decoders != 0).any()
        assert (layer.state_decoders != 0).any()

    def test_greedy_reduces_error(self):
        """Greedy training should reduce reconstruction error over iterations."""
        torch.manual_seed(42)
        layer = RecurrentNEFLayer(d_in=4, n_neurons=200, d_out=2)
        seq = torch.randn(64, 3, 4)
        targets = torch.randn(64, 2)

        # After 1 iteration
        layer.fit_greedy(seq, targets, n_iters=1)
        err1 = (layer(seq) - targets).pow(2).mean().item()

        # After 5 more iterations
        layer.fit_greedy(seq, targets, n_iters=5)
        err5 = (layer(seq) - targets).pow(2).mean().item()

        assert err5 <= err1, f"Error should decrease: {err5} > {err1}"

    def test_greedy_d_state_smaller(self):
        """Greedy works when d_state < d_in."""
        layer = RecurrentNEFLayer(d_in=8, n_neurons=100, d_out=3, d_state=4)
        seq = torch.randn(32, 5, 8)
        targets = torch.randn(32, 3)
        layer.fit_greedy(seq, targets, n_iters=3)
        assert layer.state_decoders.shape == (100, 4)


class TestRecurrentHybrid:
    """Hybrid training tests."""

    def test_hybrid_runs(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3)
        seq = torch.randn(16, 5, 8)
        targets = torch.randn(16, 3)
        layer.fit_hybrid(seq, targets, n_iters=2, lr=1e-3)
        assert (layer.decoders != 0).any()

    def test_hybrid_ce_loss(self):
        """Hybrid with cross-entropy loss on one-hot targets."""
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=5)
        seq = torch.randn(16, 5, 8)
        targets = torch.zeros(16, 5)
        targets[torch.arange(16), torch.randint(5, (16,))] = 1.0
        layer.fit_hybrid(seq, targets, n_iters=2, loss="ce")

    def test_hybrid_with_batches(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3)
        seq = torch.randn(32, 5, 8)
        targets = torch.randn(32, 3)
        layer.fit_hybrid(seq, targets, n_iters=2, batch_size=16, grad_steps=2)

    def test_state_decoders_grad_restored(self):
        """state_decoders.requires_grad should be False after hybrid."""
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2)
        seq = torch.randn(8, 3, 4)
        targets = torch.randn(8, 2)
        layer.fit_hybrid(seq, targets, n_iters=1)
        assert not layer.state_decoders.requires_grad


class TestRecurrentE2E:
    """End-to-end training tests."""

    def test_e2e_runs(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3)
        seq = torch.randn(16, 5, 8)
        targets = torch.randn(16, 3)
        layer.fit_end_to_end(seq, targets, n_epochs=2, greedy_iters=2, batch_size=8)
        assert (layer.decoders != 0).any()

    def test_e2e_grad_restored(self):
        """Both decoders should have requires_grad=False after E2E."""
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2)
        seq = torch.randn(8, 3, 4)
        targets = torch.randn(8, 2)
        layer.fit_end_to_end(seq, targets, n_epochs=1, greedy_iters=1, batch_size=4)
        assert not layer.decoders.requires_grad
        assert not layer.state_decoders.requires_grad


class TestRecurrentHybridE2E:
    """Hybrid -> E2E training tests."""

    def test_hybrid_e2e_runs(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3)
        seq = torch.randn(16, 5, 8)
        targets = torch.randn(16, 3)
        layer.fit_hybrid_e2e(seq, targets, n_iters=2, n_epochs=2, batch_size=8)
        assert (layer.decoders != 0).any()

    def test_hybrid_e2e_grad_restored(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2)
        seq = torch.randn(8, 3, 4)
        targets = torch.randn(8, 2)
        layer.fit_hybrid_e2e(seq, targets, n_iters=1, n_epochs=1, batch_size=4, loss="mse")
        assert not layer.decoders.requires_grad
        assert not layer.state_decoders.requires_grad


class TestRecurrentLongSequences:
    """Longer unroll smoke tests with gradient clipping."""

    def test_hybrid_long_sequence_stays_finite(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=40, d_out=2)
        seq = torch.randn(8, 32, 4)
        targets = torch.randn(8, 2)
        layer.fit_hybrid(seq, targets, n_iters=1, batch_size=4, grad_clip=0.5)
        assert torch.isfinite(layer(seq)).all()

    def test_e2e_long_sequence_stays_finite(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=40, d_out=2)
        seq = torch.randn(8, 32, 4)
        targets = torch.randn(8, 2)
        layer.fit_end_to_end(
            seq,
            targets,
            n_epochs=1,
            greedy_iters=1,
            batch_size=4,
            grad_clip=0.5,
        )
        assert torch.isfinite(layer(seq)).all()


class TestRecurrentDeterminism:
    """Reproducibility tests."""

    def test_same_seed_same_output(self):
        def make_layer():
            rng = torch.Generator().manual_seed(123)
            return RecurrentNEFLayer(d_in=4, n_neurons=50, d_out=2, rng=rng)

        seq = torch.randn(8, 3, 4)
        y1 = make_layer()(seq)
        y2 = make_layer()(seq)
        assert torch.allclose(y1, y2), "Same seed should produce identical output"


class TestRecurrentTargetProp:
    """Target propagation through time tests."""

    def test_temporal_projection_uses_only_state_channels(self):
        state_decoders = torch.randn(6, 3)
        encoders = torch.randn(6, 7)
        encoders_alt = encoders.clone()
        encoders_alt[:, :4] = torch.randn_like(encoders_alt[:, :4])
        assert torch.allclose(
            _temporal_projection_matrix(state_decoders, encoders, d_in=4),
            _temporal_projection_matrix(state_decoders, encoders_alt, d_in=4),
        )

    def test_tp_runs(self):
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3)
        seq = torch.randn(16, 5, 8)
        targets = torch.randn(16, 3)
        layer.fit_target_prop(seq, targets, n_iters=2, lr=1e-3, eta=0.1)
        assert (layer.decoders != 0).any()

    def test_tp_reduces_error(self):
        """TP should reduce error from the untrained baseline."""
        torch.manual_seed(42)
        seq = torch.randn(64, 3, 4)
        targets = torch.randn(64, 2)

        layer = RecurrentNEFLayer(
            d_in=4, n_neurons=100, d_out=2, rng=torch.Generator().manual_seed(42)
        )
        err_init = (layer(seq) - targets).pow(2).mean().item()

        layer.fit_target_prop(seq, targets, n_iters=10, lr=1e-3, eta=0.1)
        err_tp = (layer(seq) - targets).pow(2).mean().item()
        assert err_tp < err_init

    def test_tp_with_schedule(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2)
        seq = torch.randn(16, 3, 4)
        targets = torch.randn(16, 2)
        layer.fit_target_prop(seq, targets, n_iters=3, schedule=True)
        assert torch.isfinite(layer(seq)).all()

    def test_tp_finite_output(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=50, d_out=2)
        seq = torch.randn(16, 5, 4)
        targets = torch.randn(16, 2)
        layer.fit_target_prop(seq, targets, n_iters=5, eta=0.05)
        assert torch.isfinite(layer(seq)).all()


class TestRecurrentCenters:
    """Data-driven biases for recurrent layer."""

    def test_centers_from_sequences(self):
        """Passing sequences as centers should produce data-driven biases."""
        seq = torch.randn(32, 5, 4)
        layer = RecurrentNEFLayer(
            d_in=4, n_neurons=50, d_out=2, centers=seq, rng=torch.Generator().manual_seed(0)
        )
        # Bias should not be all-random; it's deterministic from centers
        layer2 = RecurrentNEFLayer(
            d_in=4, n_neurons=50, d_out=2, centers=seq, rng=torch.Generator().manual_seed(0)
        )
        assert torch.allclose(layer.bias, layer2.bias)

    def test_centers_from_flat(self):
        """Flat 2D tensor also works as centers."""
        flat = torch.randn(32, 4)
        layer = RecurrentNEFLayer(
            d_in=4, n_neurons=50, d_out=2, centers=flat, rng=torch.Generator().manual_seed(0)
        )
        assert torch.isfinite(layer.bias).all()


class TestRecurrentRoundTrip:
    """Save/load round-trip tests."""

    def test_state_dict_round_trip(self, tmp_path):
        rng = torch.Generator().manual_seed(42)
        layer = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3, d_state=6, rng=rng)
        seq = torch.randn(16, 5, 8)
        targets = torch.randn(16, 3)
        layer.fit_greedy(seq, targets, n_iters=2)

        path = tmp_path / "recurrent.pt"
        torch.save(layer.state_dict(), path)

        layer2 = RecurrentNEFLayer(d_in=8, n_neurons=50, d_out=3, d_state=6)
        layer2.load_state_dict(torch.load(path, weights_only=True))

        y1 = layer(seq)
        y2 = layer2(seq)
        assert torch.allclose(y1, y2), "Round-trip should preserve outputs"

    def test_state_decoders_in_state_dict(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=30, d_out=2)
        sd = layer.state_dict()
        assert "state_decoders" in sd
        assert "decoders" in sd
        assert "_gain" in sd


class TestRecurrentStepUnrollEquivalence:
    """Verify that manual step-by-step matches forward()."""

    def test_step_vs_forward(self):
        layer = RecurrentNEFLayer(d_in=4, n_neurons=50, d_out=2, d_state=4)
        # Give it nonzero state decoders
        layer.state_decoders.data.normal_()
        layer.decoders.data.normal_()

        seq = torch.randn(8, 5, 4)
        B, T, _ = seq.shape

        # Manual unroll
        s = torch.zeros(B, layer.d_state)
        for t in range(T):
            a = layer.encode_step(seq[:, t], s)
            s = a @ layer.state_decoders
        y_manual = a @ layer.decoders

        # forward()
        y_forward = layer(seq)

        assert torch.allclose(y_manual, y_forward, atol=1e-6)
