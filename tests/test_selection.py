"""Tests for neuron pruning, growth, and directed proposals."""

import pytest
import torch

import leenef.selection as selection
from leenef import (
    NEFLayer,
    agop,
    obs_saliency,
    phd_bases,
    prune_from_stats,
    psd_power,
    residual_moments,
    sample_directions_from_bases,
    sample_shaped_directions,
)
from leenef.solvers import (
    downdate_inverse,
    inverse_from_normal_equations,
    solve_from_normal_equations,
)


def _rng(seed: int = 19) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_prune_slices_exact_statistics_and_activities():
    rng = _rng()
    x = torch.randn(53, 4, generator=rng, dtype=torch.float64)
    y = torch.randn(53, 2, generator=rng, dtype=torch.float64)
    layer = NEFLayer(4, 9, 2, rng=rng).double()
    keep = torch.tensor([7, 1, 3], dtype=torch.int32)
    kept_activities = layer.encode(x)[:, keep]

    reference = NEFLayer(4, len(keep), 2, gain=layer.gain[keep], rng=rng).double()
    reference.encoders.copy_(layer.encoders[keep])
    reference.bias.copy_(layer.bias[keep])
    reference.partial_fit(x, y)

    layer.partial_fit(x, y)
    layer.prune(keep)
    torch.testing.assert_close(layer._ata, reference._ata, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer._aty, reference._aty, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer.encode(x), kept_activities, rtol=1e-12, atol=1e-12)


def test_prune_preserves_parameters_and_round_trips_accumulated_state():
    rng = _rng()
    x = torch.randn(32, 4, generator=rng, dtype=torch.float64)
    y = torch.randn(32, 2, generator=rng, dtype=torch.float64)
    layer = NEFLayer(4, 8, 2, trainable_encoders=True, rng=rng).double()
    layer.bias.requires_grad_(False)
    layer.continuous_fit(x, y)
    keep = torch.tensor([0, 3, 6])
    decoded = layer.decoders[keep]
    layer.prune(keep)
    assert layer.n_neurons == 3
    assert isinstance(layer.encoders, torch.nn.Parameter) and layer.encoders.requires_grad
    assert isinstance(layer.bias, torch.nn.Parameter) and not layer.bias.requires_grad
    assert layer._M_inv is None and layer._woodbury_alpha is None
    torch.testing.assert_close(layer.decoders, decoded)

    restored = NEFLayer(4, 3, 2, trainable_encoders=True, rng=rng).double()
    restored.load_state_dict(layer.state_dict())
    torch.testing.assert_close(restored(x), layer(x))
    torch.testing.assert_close(restored._ata, layer._ata)
    torch.testing.assert_close(restored._aty, layer._aty)

    # A rank-one continuous update after pruning must still use the retained statistics.
    layer.continuous_fit(x[:1], y[:1], fixed_alpha=0.2)
    expected = solve_from_normal_equations(layer._ata, layer._aty, alpha=(0.2, "fixed"))
    torch.testing.assert_close(layer.decoders, expected, rtol=1e-12, atol=1e-12)


def test_load_without_statistics_clears_stale_buffers_but_preserves_alpha():
    rng = _rng()
    x = torch.randn(6, 2, generator=rng)
    y = torch.randn(6, 1, generator=rng)
    layer = NEFLayer(2, 4, 1, rng=rng)
    fixed_alpha = 0.7
    layer.continuous_fit(x, y, fixed_alpha=fixed_alpha)
    state = NEFLayer(2, 4, 1, rng=rng).state_dict()

    result = layer.load_state_dict(state, strict=False)
    assert set(result.missing_keys) == {"_ata", "_aty"}
    assert layer._ata is layer._aty is layer._M_inv is None
    layer.partial_fit(x, y)
    layer.refresh_inverse()
    expected = solve_from_normal_equations(layer._ata, layer._aty, alpha=(fixed_alpha, "fixed"))
    torch.testing.assert_close(layer.decoders, expected.float())

    with pytest.raises(RuntimeError, match="Missing key.*_ata.*_aty"):
        layer.load_state_dict(state)


def test_load_with_malformed_statistics_leaves_no_partial_accumulators():
    rng = _rng()
    x = torch.randn(6, 2, generator=rng)
    source = NEFLayer(2, 4, 1, rng=rng)
    source.partial_fit(x, x[:, :1])
    state = source.state_dict()
    state["_ata"] = state["_ata"][:3, :3]

    layer = NEFLayer(2, 4, 1, rng=rng)
    with pytest.raises(RuntimeError, match="size mismatch for _ata"):
        layer.load_state_dict(state)
    assert layer._ata is layer._aty is None


def test_rejected_continuous_fit_does_not_accumulate_batch():
    rng = _rng()
    layer = NEFLayer(2, 4, 1, rng=rng)
    activities = torch.randn(3, 4, generator=rng)
    targets = torch.randn(3, 1, generator=rng)
    layer.continuous_fit_encoded(activities, targets, fixed_alpha=0.1)
    ata, aty = layer._ata.clone(), layer._aty.clone()
    with pytest.raises(ValueError, match="fixed_alpha changed"):
        layer.continuous_fit_encoded(activities, targets, fixed_alpha=0.2)
    assert torch.equal(layer._ata, ata)
    assert torch.equal(layer._aty, aty)


def test_prune_rejects_invalid_indices_and_neuron_specific_activations():
    layer = NEFLayer(2, 4, 1, rng=_rng())
    for keep in (
        torch.tensor([[1]]),
        torch.tensor([1.0]),
        torch.tensor([True]),
        torch.empty(0, dtype=torch.long),
        torch.tensor([0, 0]),
        torch.tensor([-1, 4]),
    ):
        with pytest.raises(ValueError, match="keep"):
            layer.prune(keep)

    layer.activation = torch.nn.PReLU(num_parameters=layer.n_neurons)
    with pytest.raises(NotImplementedError, match="per-neuron"):
        layer.prune(torch.tensor([0, 1]))


def test_obs_saliency_is_exact_ridge_objective_increase():
    rng = _rng()
    a = torch.randn(36, 12, generator=rng, dtype=torch.float64)
    y = torch.randn(36, 3, generator=rng, dtype=torch.float64)
    g, r = a.T @ a, a.T @ y
    alpha = (0.4, "fixed")
    full_explained = (r * solve_from_normal_equations(g, r, alpha=alpha)).sum()
    losses = []
    indices = torch.arange(a.shape[1])
    for i in indices:
        keep = indices[indices != i]
        g_sub, r_sub = g[keep[:, None], keep], r[keep]
        explained = (r_sub * solve_from_normal_equations(g_sub, r_sub, alpha=alpha)).sum()
        losses.append(full_explained - explained)
    torch.testing.assert_close(
        obs_saliency(g, r, alpha=alpha), torch.stack(losses), rtol=1e-8, atol=1e-10
    )


def test_obs_saliency_rejects_non_positive_inverse_diagonal():
    g = torch.diag(torch.tensor([1.0, -2.0]))
    with pytest.raises(RuntimeError, match="positive inverse diagonal"):
        obs_saliency(g, torch.ones(2, 1), alpha=(0.1, "fixed"))


def test_prune_from_stats_recovers_planted_neurons_with_both_methods():
    rng = _rng()
    x = torch.randn(128, 6, generator=rng, dtype=torch.float64)
    layer = NEFLayer(6, 12, 2, rng=rng).double()
    a = layer.encode(x)
    planted = torch.tensor([2, 5, 10])
    y = a[:, planted] @ torch.tensor([[1.0, 3.0], [-2.0, 1.0], [3.0, 1.0]], dtype=a.dtype)
    for method in ("refit", "downdate"):
        keep = prune_from_stats(
            a.T @ a, a.T @ y, len(planted), alpha=(0.01, "fixed"), method=method
        )
        assert torch.equal(keep, planted)


def test_overcomplete_downdates_stay_symmetric_and_match_refits(monkeypatch):
    rng = _rng()
    x = torch.randn(60, 8, generator=rng, dtype=torch.float64)
    y = torch.randn(60, 2, generator=rng, dtype=torch.float64)
    layer = NEFLayer(8, 120, 2, rng=rng).double()
    layer.partial_fit(x, y)

    symmetry = []

    def record_symmetry(*args):
        inverse = downdate_inverse(*args)
        symmetry.append(torch.equal(inverse, inverse.mT))
        return inverse

    monkeypatch.setattr(selection, "downdate_inverse", record_symmetry)
    args = (layer._ata, layer._aty, 10)
    alpha = (1e-4, "fixed")
    refit = prune_from_stats(*args, alpha=alpha, method="refit")
    downdate = prune_from_stats(*args, alpha=alpha, method="downdate")
    assert torch.equal(downdate, refit)
    assert symmetry and all(symmetry), "downdated inverse lost symmetry across pruning rounds"


def test_prune_from_stats_trace_policy_and_stable_ties():
    g = torch.diag(torch.tensor([3000.0, 100.0, 1.0]))
    r = torch.tensor([[0.0], [3.0], [1.0]])
    assert prune_from_stats(g, r, 1, alpha=(0.1, "trace"), method="refit").item() == 2
    assert prune_from_stats(g, r, 1, alpha=(0.1, "trace"), method="downdate").item() == 1
    keep = prune_from_stats(g, torch.zeros_like(r), 2)
    torch.testing.assert_close(keep, torch.tensor([1, 2]))

    with pytest.raises(ValueError, match="n_keep"):
        prune_from_stats(g, r, 4)
    with pytest.raises(ValueError, match="drop_fraction"):
        prune_from_stats(g, r, 1, drop_fraction=1)
    with pytest.raises(ValueError, match="method"):
        prune_from_stats(g, r, 1, method="unknown")


def test_invalid_downdate_is_rebuilt_at_frozen_regularizer(monkeypatch):
    g = torch.diag(torch.tensor([3000.0, 100.0, 1.0]))
    r = torch.tensor([[0.0], [3.0], [1.0]])
    downdated = torch.diag(torch.tensor([float("nan"), -1.0], dtype=torch.float64))
    monkeypatch.setattr(selection, "downdate_inverse", lambda *_: downdated)

    keep = prune_from_stats(g, r, 1, alpha=(0.1, "trace"), method="downdate")
    assert keep.item() == 1


def test_downdate_inverse_matches_direct_inverse():
    a = torch.randn(20, 9, generator=_rng(), dtype=torch.float64)
    matrix = a.T @ a
    keep = torch.tensor([7, 1, 3, 6, 5, 8])
    drop = torch.tensor([2, 0, 4])
    alpha = (0.25, "fixed")
    inverse = inverse_from_normal_equations(matrix, alpha=alpha)
    expected = inverse_from_normal_equations(matrix[keep[:, None], keep], alpha=alpha)
    torch.testing.assert_close(
        downdate_inverse(inverse, keep, drop), expected, rtol=1e-12, atol=1e-12
    )


def test_grow_initializes_new_neurons_and_invalidates_statistics():
    rng = _rng()
    layer = NEFLayer(5, 7, 2, trainable_encoders=True, rng=rng).double()
    centers = torch.randn(15, 5, generator=rng, dtype=torch.float64)
    y = torch.randn(15, 2, generator=rng, dtype=torch.float64)
    layer.continuous_fit(centers, y)
    old_output = layer(centers)
    new_encoders = torch.randn(3, 5, generator=rng)
    chosen = torch.randint(len(centers), (3,), generator=_rng(9))

    layer.grow(new_encoders, centers=centers, rng=_rng(9))
    assert layer.n_neurons == 10
    assert layer.encoders.shape == (10, 5) and layer.encoders.requires_grad
    assert layer.bias.shape == layer.gain.shape == (10,) and layer.bias.requires_grad
    assert layer.decoders.shape == (10, 2)
    assert layer._ata is layer._aty is layer._M_inv is layer._woodbury_alpha is None
    torch.testing.assert_close(layer.encoders[-3:], new_encoders.double())
    torch.testing.assert_close(layer.decoders[-3:], torch.zeros_like(layer.decoders[-3:]))
    activities = layer.encode(centers[chosen])[:, -3:].diagonal()
    torch.testing.assert_close(activities, torch.zeros_like(activities), rtol=0, atol=1e-12)
    torch.testing.assert_close(layer(centers), old_output)
    with pytest.raises(RuntimeError, match="No accumulated"):
        layer.solve_accumulated()


def test_grow_validates_arguments_and_accepts_explicit_bias_gain():
    layer = NEFLayer(2, 3, 1, rng=_rng())
    enc = torch.ones(1, 2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        layer.grow(enc, bias=torch.ones(1), centers=enc)
    with pytest.raises(ValueError, match="gain"):
        layer.grow(enc, gain="data_driven")
    with pytest.raises(ValueError, match="encoders"):
        layer.grow(torch.ones(1, 3))
    with pytest.raises(ValueError, match="bias"):
        layer.grow(enc, bias=torch.ones(2))
    with pytest.raises(ValueError, match="centers"):
        layer.grow(enc, centers=torch.empty(0, 2))

    layer.grow(enc, bias=torch.tensor([-0.5]), gain=torch.tensor([2.0]))
    torch.testing.assert_close(layer.encode(enc)[0, -1], torch.tensor(3.5))
    assert "encoders" in dict(layer.named_buffers())


def test_grow_requires_rng_only_for_random_draws():
    layer = NEFLayer(2, 3, 1, rng=_rng())
    layer._rng = None  # Use a deterministic construction, then test with no generator.
    enc = torch.ones(1, 2)
    bias = torch.tensor([-0.5])
    for kwargs in ({"bias": bias}, {"centers": enc, "gain": 1.0}, {"gain": 1.0}):
        with pytest.raises(ValueError, match="Pass rng="):
            layer.grow(enc, **kwargs)
    assert layer.n_neurons == 3

    layer.grow(enc, bias=bias, gain=2.0)
    layer.grow(enc, bias=bias, gain=torch.tensor([3.0]))
    torch.testing.assert_close(layer.encode(enc)[0, -2:], torch.tensor([3.5, 5.5]))


def test_residual_moments_are_additive_and_phd_recovers_curvature():
    rng = _rng()
    centered = torch.randn(4096, 8, generator=rng, dtype=torch.float64)
    mean = torch.randn(8, generator=rng, dtype=torch.float64)
    x = centered + mean
    w = torch.randn(8, generator=rng, dtype=torch.float64)
    w = w / w.norm()
    r = (centered @ w).square() - 1
    residual = torch.stack((r, -2 * r), dim=1)
    moments = residual_moments(x, residual, mean=mean)
    accumulated = residual_moments(x[:2048], residual[:2048], mean=mean, batch_size=500)
    assert residual_moments(x[2048:], residual[2048:], mean=mean, out=accumulated) is accumulated
    torch.testing.assert_close(accumulated, moments, rtol=1e-12, atol=1e-11)
    torch.testing.assert_close(residual_moments(centered, residual), moments)
    bases = phd_bases(moments, 1)
    leading = bases[:, :, 0]
    torch.testing.assert_close(
        leading.norm(dim=1)[1] / leading.norm(dim=1)[0], torch.tensor(2.0).sqrt()
    )
    cosine = (leading @ w.float()).abs() / leading.norm(dim=1)
    assert (cosine > 0.95).all()

    dirs = sample_directions_from_bases(31, bases, rng=_rng())
    torch.testing.assert_close(dirs.norm(dim=1), torch.ones(31))
    assert torch.equal(dirs, sample_directions_from_bases(31, bases, rng=_rng()))
    assert ((dirs @ w.float()).abs() > 0.95).all()


def test_shaped_sampling_and_zero_basis_fallback():
    shaping = psd_power(torch.diag(torch.tensor([-0.01, 1.0, 4.0])), 0.5)
    torch.testing.assert_close(shaping, torch.diag(torch.tensor([0.0, 0.5, 1.0])))
    dirs = sample_shaped_directions(64, shaping, rng=_rng())
    torch.testing.assert_close(dirs.norm(dim=1), torch.ones(64))
    assert torch.count_nonzero(dirs[:, 0]) == 0

    for fallback in (
        sample_shaped_directions(64, torch.zeros_like(shaping), rng=_rng()),
        sample_directions_from_bases(64, torch.zeros(2, 3, 2), rng=_rng()),
    ):
        torch.testing.assert_close(fallback.norm(dim=1), torch.ones(64))
        assert (fallback.square().mean(dim=0) > 0.1).all()


def test_agop_matches_autograd_and_recovers_target_direction():
    rng = _rng()
    x = torch.randn(180, 5, generator=rng, dtype=torch.float64)
    w = torch.randn(5, generator=rng, dtype=torch.float64)
    w = w / w.norm()
    y = (3 * (x @ w)).tanh().unsqueeze(1) * torch.tensor([1.0, -2.0])
    layer = NEFLayer(5, 15, 2, activation="tanh", rng=rng).double()
    layer.encoders[0] = w
    layer.gain[0] = 3.0
    layer.bias[0] = 0.0
    layer.fit(x, y, alpha=(1e-4, "trace"))

    _, vectors = torch.linalg.eigh(agop(layer, x, batch_size=17))
    assert (vectors[:, -1] @ w).abs() > 0.9
    points = x[:5]
    jac = torch.autograd.functional.jacobian(layer.forward, points)
    indices = torch.arange(len(points))
    jac = jac[indices, :, indices, :].flatten(0, 1)
    expected = (jac.T @ jac) / len(points)
    torch.testing.assert_close(agop(layer, points, batch_size=2), expected, rtol=1e-11, atol=1e-11)
