"""Tests for multi-layer NEFNetwork."""

import pytest
import torch

from leenef.networks import (
    NEFNetwork,
    _bounded_target_update,
    _ce_targets,
    _difference_target,
    _project_activity_target,
    _scheduled_step_scale,
)


def _make_net(d_in=4, d_out=2, hidden=None, output_n=200, **kw):
    hidden = hidden or [100]
    return NEFNetwork(
        d_in,
        d_out,
        hidden_neurons=hidden,
        output_neurons=output_n,
        rng=torch.Generator().manual_seed(42),
        **kw,
    )


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
        y = x**2
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

    @pytest.mark.parametrize("encoder_strategy", ["whitened", "class_contrast", "local_pca"])
    def test_data_driven_encoders_work_through_output_layer(self, encoder_strategy):
        """Layer-data encoders should initialize hidden and output layers."""
        torch.manual_seed(23)
        x = torch.randn(160, 4)
        labels = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
        targets = torch.zeros(160, 2).scatter_(1, labels.unsqueeze(1), 1.0)
        encoder_kwargs = {"train_data": x}
        if encoder_strategy == "class_contrast":
            encoder_kwargs["train_labels"] = labels

        net = _make_net(
            d_in=4,
            d_out=2,
            hidden=[60],
            output_n=120,
            encoder_strategy=encoder_strategy,
            encoder_kwargs=encoder_kwargs,
            centers=x,
        )
        net.fit_greedy(x, targets)

        out = net(x[:16])
        assert out.shape == (16, 2)


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

        assert hybrid_err < greedy_err, (
            f"hybrid ({hybrid_err:.4f}) should beat greedy ({greedy_err:.4f})"
        )

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

    def test_ce_loss(self):
        torch.manual_seed(10)
        x = torch.randn(400, 4)
        labels = (x[:, 0] + x[:, 1] > 0).long()
        targets = torch.zeros(400, 2).scatter_(1, labels.unsqueeze(1), 1.0)
        net = _make_net(d_in=4, d_out=2, hidden=[200], output_n=400)
        net.fit_hybrid(x, targets, n_iters=10, lr=1e-3, loss="ce")
        with torch.no_grad():
            acc = (net(x).argmax(1) == labels).float().mean().item()
        assert acc > 0.90, f"hybrid CE accuracy too low: {acc}"

    def test_cosine_schedule(self):
        torch.manual_seed(11)
        x = torch.rand(300, 2) * 2 - 1
        y = torch.sin(x[:, :1] * 3)
        net = _make_net(d_in=2, d_out=1, hidden=[200], output_n=300)
        net.fit_hybrid(x, y, n_iters=15, lr=5e-4, schedule=True)
        with torch.no_grad():
            err = (net(x) - y).pow(2).mean().item()
        assert err < 0.2, f"hybrid with schedule MSE too high: {err}"

    def test_incremental_init(self):
        torch.manual_seed(12)
        x = torch.randn(400, 4)
        labels = (x[:, 0] + x[:, 1] > 0).long()
        targets = torch.zeros(400, 2).scatter_(1, labels.unsqueeze(1), 1.0)
        net = _make_net(d_in=4, d_out=2, hidden=[200], output_n=400)
        net.fit_hybrid(x, targets, n_iters=10, lr=1e-3, init="incremental", centers=x)
        with torch.no_grad():
            acc = (net(x).argmax(1) == labels).float().mean().item()
        assert acc > 0.90, f"hybrid incremental accuracy too low: {acc}"

    def test_mini_batch(self):
        torch.manual_seed(13)
        x = torch.randn(400, 4)
        labels = (x[:, 0] + x[:, 1] > 0).long()
        targets = torch.zeros(400, 2).scatter_(1, labels.unsqueeze(1), 1.0)
        net = _make_net(d_in=4, d_out=2, hidden=[200], output_n=400)
        net.fit_hybrid(x, targets, n_iters=10, lr=1e-3, batch_size=64, grad_steps=3)
        with torch.no_grad():
            acc = (net(x).argmax(1) == labels).float().mean().item()
        assert acc > 0.85, f"hybrid mini-batch accuracy too low: {acc}"


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

        assert e2e_err < greedy_err, (
            f"end-to-end ({e2e_err:.4f}) should beat greedy ({greedy_err:.4f})"
        )

    def test_decoders_frozen_after(self):
        """Decoders should have requires_grad=False after fit_end_to_end."""
        net = _make_net()
        x = torch.randn(32, 4)
        y = torch.randn(32, 2)
        net.fit_end_to_end(x, y, n_epochs=1, batch_size=32)
        assert not net.output.decoders.requires_grad


# ---- Strategy D: hybrid → E2E ----


class TestHybridE2E:
    def test_runs_and_improves(self):
        torch.manual_seed(6)
        x = torch.randn(400, 4)
        labels = (x[:, 0] + x[:, 1] > 0).long()
        targets = torch.zeros(400, 2).scatter_(1, labels.unsqueeze(1), 1.0)
        net = _make_net(d_in=4, d_out=2, hidden=[200], output_n=400)
        net.fit_hybrid_e2e(
            x,
            targets,
            n_iters=5,
            hybrid_lr=1e-3,
            n_epochs=5,
            e2e_lr=1e-3,
            batch_size=64,
            loss="ce",
        )
        with torch.no_grad():
            acc = (net(x).argmax(1) == labels).float().mean().item()
        assert acc > 0.90, f"hybrid_e2e accuracy too low: {acc}"

    def test_decoders_frozen_after(self):
        net = _make_net()
        x = torch.randn(32, 4)
        y = torch.randn(32, 2)
        net.fit_hybrid_e2e(x, y, n_iters=2, n_epochs=2, batch_size=32)
        assert not net.output.decoders.requires_grad


# ---- Strategy E: target-prop -> E2E ----


class TestTargetPropE2E:
    def test_runs_and_improves(self):
        torch.manual_seed(24)
        x = torch.randn(400, 4)
        labels = (x[:, 0] + x[:, 1] > 0).long()
        targets = torch.zeros(400, 2).scatter_(1, labels.unsqueeze(1), 1.0)
        net = _make_net(d_in=4, d_out=2, hidden=[200], output_n=400)
        net.fit_target_prop_e2e(
            x,
            targets,
            n_iters=5,
            tp_lr=1e-3,
            eta=0.1,
            n_epochs=5,
            e2e_lr=1e-3,
            batch_size=64,
            loss="ce",
        )
        with torch.no_grad():
            acc = (net(x).argmax(1) == labels).float().mean().item()
        assert acc > 0.90, f"target_prop_e2e accuracy too low: {acc}"

    def test_decoders_frozen_after(self):
        net = _make_net()
        x = torch.randn(32, 4)
        y = torch.randn(32, 2)
        net.fit_target_prop_e2e(x, y, n_iters=2, n_epochs=2, batch_size=32, loss="mse")
        assert not net.output.decoders.requires_grad


# ---- Propagated centers ----


class TestPropagateCenters:
    def test_output_bias_changes(self):
        """propagate_centers should update output layer biases."""
        torch.manual_seed(20)
        x = torch.randn(200, 4)
        net = _make_net(d_in=4, d_out=2, hidden=[100], centers=x)
        old_bias = net.output.bias.data.clone()
        # Re-propagate with different data
        net.propagate_centers(torch.randn(200, 4))
        assert not torch.allclose(old_bias, net.output.bias.data)

    def test_centers_improve_greedy(self):
        """Data-driven biases on all layers should help greedy."""
        torch.manual_seed(21)
        x = torch.randn(400, 4)
        labels = (x[:, 0] + x[:, 1] > 0).long()
        targets = torch.zeros(400, 2).scatter_(1, labels.unsqueeze(1), 1.0)

        # Without centers
        net_no = NEFNetwork(4, 2, [200], output_neurons=400, rng=torch.Generator().manual_seed(42))
        net_no.fit_greedy(x, targets)
        with torch.no_grad():
            acc_no = (net_no(x).argmax(1) == labels).float().mean().item()

        # With centers (propagated to all layers)
        net_yes = NEFNetwork(
            4, 2, [200], output_neurons=400, rng=torch.Generator().manual_seed(42), centers=x
        )
        net_yes.fit_greedy(x, targets)
        with torch.no_grad():
            acc_yes = (net_yes(x).argmax(1) == labels).float().mean().item()

        assert acc_yes >= acc_no - 0.02, (
            f"centers ({acc_yes:.3f}) should not hurt vs no centers ({acc_no:.3f})"
        )


# ---- Determinism ----


class TestDeterminism:
    def test_greedy_deterministic(self):
        """Same RNG seed → identical greedy results."""
        x = torch.randn(200, 4)
        y = torch.randn(200, 2)
        results = []
        for _ in range(2):
            net = NEFNetwork(
                4, 2, [100], output_neurons=200, rng=torch.Generator().manual_seed(99)
            )
            net.fit_greedy(x, y)
            with torch.no_grad():
                results.append(net(x))
        assert torch.allclose(results[0], results[1], atol=1e-5)


# ---- Save/load round-trip ----


class TestSaveLoad:
    def test_network_state_dict_round_trip(self):
        """Save/load preserves network output exactly."""
        torch.manual_seed(50)
        x = torch.randn(100, 4)
        y = torch.randn(100, 2)
        net = _make_net(d_in=4, d_out=2, hidden=[100], output_n=200)
        net.fit_greedy(x, y)
        out1 = net(x)

        state = net.state_dict()
        net2 = _make_net(d_in=4, d_out=2, hidden=[100], output_n=200)
        net2.load_state_dict(state)
        out2 = net2(x)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_per_neuron_gain_survives_save_load(self):
        """Per-neuron gain in state_dict is restored correctly."""
        torch.manual_seed(51)
        net = NEFNetwork(
            4, 2, [100], output_neurons=200, gain=(0.5, 2.0), rng=torch.Generator().manual_seed(42)
        )
        x = torch.randn(50, 4)
        y = torch.randn(50, 2)
        net.fit_greedy(x, y)
        out1 = net(x)

        state = net.state_dict()
        net2 = NEFNetwork(
            4, 2, [100], output_neurons=200, gain=1.0, rng=torch.Generator().manual_seed(0)
        )
        net2.load_state_dict(state)
        out2 = net2(x)
        assert torch.allclose(out1, out2, atol=1e-5)


# ---- Exception paths ----


class TestNetworkExceptions:
    def test_ce_targets_requires_2d(self):
        with pytest.raises(ValueError, match="2-D one-hot"):
            _ce_targets(torch.tensor([0, 1, 2]))


# ---- Target Propagation ----


class TestTargetPropHelpers:
    def test_project_activity_target_clamps_nonnegative_activations(self):
        target = torch.tensor([[-1.0, 0.5], [2.0, -3.0]])
        expected = torch.tensor([[0.0, 0.5], [2.0, 0.0]])
        assert torch.equal(_project_activity_target(target, "relu"), expected)
        assert torch.equal(_project_activity_target(target, "abs"), expected)
        assert torch.equal(_project_activity_target(target, "softplus"), target)

    def test_difference_target_is_noop_when_upper_target_matches_activity(self):
        activity = torch.rand(8, 6)
        next_activity = torch.rand(8, 10)
        repr_decoder = torch.randn(10, 6)
        target = _difference_target(
            activity, next_activity, next_activity.clone(), repr_decoder, "relu"
        )
        assert torch.allclose(target, activity)

    def test_bounded_target_update_backoff_limits_infeasible_fraction(self):
        activity = torch.full((4, 3), 0.2)
        update = -torch.ones_like(activity)
        target, applied_scale, pre_fraction, post_fraction = _bounded_target_update(
            activity,
            update,
            step_scale=1.0,
            activation="relu",
            max_infeasible_fraction=0.1,
            project=True,
        )
        assert applied_scale < 1.0
        assert pre_fraction <= 0.1
        assert post_fraction == 0.0
        assert torch.all(target >= 0)

    def test_bounded_target_update_can_back_off_without_projection(self):
        activity = torch.full((4, 3), 0.2)
        update = -torch.ones_like(activity)
        target, applied_scale, pre_fraction, post_fraction = _bounded_target_update(
            activity,
            update,
            step_scale=1.0,
            activation="relu",
            max_infeasible_fraction=0.1,
            project=False,
        )
        assert applied_scale < 1.0
        assert pre_fraction <= 0.1
        assert post_fraction == pytest.approx(pre_fraction)
        assert torch.allclose(target, activity + applied_scale * update)

    def test_bounded_target_update_noops_when_projection_not_needed(self):
        activity = torch.full((4, 3), 0.2)
        update = torch.full((4, 3), 0.05)
        target, applied_scale, pre_fraction, post_fraction = _bounded_target_update(
            activity,
            update,
            step_scale=0.5,
            activation="abs",
            max_infeasible_fraction=0.01,
        )
        assert applied_scale == pytest.approx(0.5)
        assert pre_fraction == 0.0
        assert post_fraction == 0.0
        assert torch.allclose(target, activity + 0.5 * update)

    def test_scheduled_step_scale_constant_keeps_eta(self):
        assert _scheduled_step_scale(0.03, 2, 5, schedule="constant", final_fraction=0.25) == 0.03

    def test_scheduled_step_scale_cosine_decay_hits_final_fraction(self):
        assert _scheduled_step_scale(
            0.04, 0, 5, schedule="cosine_decay", final_fraction=0.25
        ) == pytest.approx(0.04)
        assert _scheduled_step_scale(
            0.04, 4, 5, schedule="cosine_decay", final_fraction=0.25
        ) == pytest.approx(0.01)


class TestTargetProp:
    def test_forward_shape_after_tp(self):
        """Network forward still works after TP training."""
        net = _make_net(d_in=4, d_out=2, hidden=[50], output_n=100)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(30, 4, generator=g)
        targets = torch.randn(30, 2, generator=g)
        net.fit_target_prop(x, targets, n_iters=2, lr=1e-3, eta=0.1)
        pred = net(x)
        assert pred.shape == (30, 2)

    def test_tp_improves_over_greedy(self):
        """TP should beat greedy on a learnable task."""
        g = torch.Generator().manual_seed(7)
        x = torch.randn(200, 4, generator=g)
        w = torch.randn(4, 2, generator=g)
        targets = x @ w

        net_g = _make_net(d_in=4, d_out=2, hidden=[100], output_n=200, centers=x)
        net_g.fit_greedy(x, targets)
        mse_greedy = (net_g(x) - targets).pow(2).mean().item()

        net_tp = _make_net(d_in=4, d_out=2, hidden=[100], output_n=200, centers=x)
        net_tp.fit_target_prop(x, targets, n_iters=10, lr=1e-3, eta=0.1)
        mse_tp = (net_tp(x) - targets).pow(2).mean().item()

        assert mse_tp < mse_greedy

    def test_tp_with_relu(self):
        """TP works with relu activation."""
        g = torch.Generator().manual_seed(3)
        x = torch.randn(100, 4, generator=g)
        targets = torch.randn(100, 2, generator=g)
        net = _make_net(d_in=4, d_out=2, hidden=[50], output_n=100, activation="relu")
        net.fit_target_prop(x, targets, n_iters=3, lr=1e-3, eta=0.1)
        pred = net(x)
        assert pred.shape == (100, 2)
        assert torch.isfinite(pred).all()

    def test_tp_with_schedule(self):
        """TP runs with cosine LR schedule without error."""
        g = torch.Generator().manual_seed(5)
        x = torch.randn(50, 4, generator=g)
        targets = torch.randn(50, 2, generator=g)
        net = _make_net(d_in=4, d_out=2, hidden=[50], output_n=100)
        net.fit_target_prop(x, targets, n_iters=5, lr=1e-3, eta=0.1, schedule=True)
        assert torch.isfinite(net(x)).all()

    def test_tp_with_eta_decay_schedule(self):
        g = torch.Generator().manual_seed(6)
        x = torch.randn(64, 4, generator=g)
        targets = torch.randn(64, 2, generator=g)
        net = _make_net(d_in=4, d_out=2, hidden=[50], output_n=100)
        net.fit_target_prop(
            x,
            targets,
            n_iters=4,
            lr=1e-3,
            eta=0.1,
            eta_schedule="cosine_decay",
            eta_final_fraction=0.25,
            collect_diagnostics=True,
        )
        diag = net.last_target_prop_diagnostics
        assert diag is not None
        assert diag[0]["requested_output_step_scale"] == pytest.approx(0.1)
        assert diag[-1]["requested_output_step_scale"] == pytest.approx(0.025)

    def test_tp_multi_hidden(self):
        """TP works with multiple hidden layers."""
        g = torch.Generator().manual_seed(9)
        x = torch.randn(80, 4, generator=g)
        targets = torch.randn(80, 2, generator=g)
        net = _make_net(d_in=4, d_out=2, hidden=[50, 30], output_n=100)
        net.fit_target_prop(x, targets, n_iters=3, lr=1e-3, eta=0.1)
        pred = net(x)
        assert pred.shape == (80, 2)
        assert torch.isfinite(pred).all()

    def test_tp_collects_diagnostics(self):
        g = torch.Generator().manual_seed(21)
        x = torch.randn(64, 4, generator=g)
        targets = torch.randn(64, 2, generator=g)
        net = _make_net(d_in=4, d_out=2, hidden=[50], output_n=100, activation="relu")
        net.fit_target_prop(
            x,
            targets,
            n_iters=3,
            lr=1e-3,
            eta=0.1,
            collect_diagnostics=True,
            project_targets=True,
        )
        diagnostics = net.last_target_prop_diagnostics
        assert diagnostics is not None
        assert len(diagnostics) == 3
        assert len(diagnostics[-1]["layers"]) == 2
        for layer in diagnostics[-1]["layers"]:
            assert layer["target_drift"] >= 0
            assert layer["infeasible_fraction_after"] == 0.0
            assert layer["requested_step_scale"] >= layer["applied_step_scale"] >= 0.0
        assert (
            diagnostics[-1]["requested_output_step_scale"]
            >= diagnostics[-1]["applied_output_step_scale"]
            >= 0.0
        )

    def test_tp_defaults_to_unprojected_targets(self):
        g = torch.Generator().manual_seed(29)
        x = torch.randn(64, 4, generator=g)
        targets = torch.randn(64, 2, generator=g)
        net = _make_net(d_in=4, d_out=2, hidden=[50], output_n=100, activation="relu")
        net.fit_target_prop(x, targets, n_iters=2, lr=1e-3, eta=0.1, collect_diagnostics=True)
        diagnostics = net.last_target_prop_diagnostics
        assert diagnostics is not None
        assert any(layer["infeasible_fraction_after"] > 0.0 for layer in diagnostics[0]["layers"])

    def test_tp_with_raw_step(self):
        g = torch.Generator().manual_seed(22)
        x = torch.randn(64, 4, generator=g)
        targets = torch.randn(64, 2, generator=g)
        net = _make_net(d_in=4, d_out=2, hidden=[50], output_n=100)
        net.fit_target_prop(
            x,
            targets,
            n_iters=2,
            lr=1e-3,
            eta=0.1,
            normalize_step=False,
            collect_diagnostics=True,
        )
        diag = net.last_target_prop_diagnostics
        assert diag is not None
        assert diag[0]["applied_grad_norm"] == pytest.approx(diag[0]["raw_grad_norm"])

    def test_tp_with_robust_repr_decoders(self):
        g = torch.Generator().manual_seed(23)
        x = torch.randn(64, 8, generator=g)
        targets = torch.randn(64, 2, generator=g)
        net = _make_net(d_in=8, d_out=2, hidden=[60], output_n=80, centers=x)
        net.fit_target_prop(
            x,
            targets,
            n_iters=2,
            lr=1e-3,
            eta=0.1,
            repr_noise_std=0.1,
            repr_noise_repeats=2,
            collect_diagnostics=True,
        )
        diagnostics = net.last_target_prop_diagnostics
        assert diagnostics is not None
        for layer in diagnostics[-1]["layers"][1:]:
            assert layer["repr_mse"] is not None
            assert layer["repr_perturbed_mse"] is not None
            assert torch.isfinite(torch.tensor(layer["repr_perturbed_mse"]))

    def test_tp_hidden_feasibility_backoff_rescales_hidden_targets(self):
        g = torch.Generator().manual_seed(24)
        x = torch.randn(64, 4, generator=g)
        targets = torch.randn(64, 2, generator=g) * 10
        net = _make_net(d_in=4, d_out=2, hidden=[8], output_n=10, activation="relu")
        net.fit_target_prop(
            x,
            targets,
            n_iters=1,
            lr=1e-3,
            eta=1.0,
            normalize_step=False,
            hidden_max_infeasible_fraction=0.0,
            collect_diagnostics=True,
        )
        diag = net.last_target_prop_diagnostics
        assert diag is not None
        assert any(
            layer["applied_step_scale"] < layer["requested_step_scale"]
            for layer in diag[0]["layers"][:-1]
        )

    def test_repr_decoder_quality(self):
        """Representational decoders should reconstruct inputs reasonably."""
        from leenef.solvers import solve_decoders

        g = torch.Generator().manual_seed(11)
        x = torch.randn(200, 8, generator=g)
        net = _make_net(d_in=8, d_out=2, hidden=[200], output_n=200, centers=x)
        a = net.hidden[0].encode(x)
        D_repr = solve_decoders(a, x, method="tikhonov")
        recon = a @ D_repr
        mse = (recon - x).pow(2).mean().item()
        # With 200 neurons and 8-dim input, reconstruction should be decent
        assert mse < 1.0
