"""Tests for RL utilities."""

import torch

from leenef.rl import NEFFeatures


def test_apply_broadness_uses_alive_mean_for_cumulative_stats():
    features = NEFFeatures(d_in=2, n_neurons=4, track_activations=True)
    features._act_sum = torch.tensor([1.0, 3.0, 10.0, 20.0])
    features._act_count = 5.0

    features.apply_broadness(0.25, torch.tensor([0, 1]))

    expected = torch.tensor([15.0, 15.0, 10.0, 20.0]) * 0.75
    assert torch.allclose(features._act_sum, expected)
    assert features._act_count == 3.75


def test_apply_broadness_decays_even_without_recentered_neurons():
    features = NEFFeatures(d_in=2, n_neurons=3, track_activations=True)
    features._act_sum = torch.tensor([2.0, 4.0, 6.0])
    features._act_count = 8.0

    features.apply_broadness(0.5, torch.tensor([], dtype=torch.long))

    assert torch.allclose(features._act_sum, torch.tensor([1.0, 2.0, 3.0]))
    assert features._act_count == 4.0


def test_apply_broadness_forgets_everything_at_one():
    features = NEFFeatures(d_in=2, n_neurons=3, track_activations=True)
    features._act_sum = torch.tensor([2.0, 4.0, 6.0])
    features._act_count = 8.0

    features.apply_broadness(1.0, torch.tensor([1]))

    assert torch.equal(features._act_sum, torch.zeros(3))
    assert features._act_count == 0.0


def test_apply_broadness_uses_alive_mean_for_ema_stats():
    features = NEFFeatures(d_in=2, n_neurons=4, track_activations=True, activity_decay=0.99)
    features._ema_activity = torch.tensor([1.0, 3.0, 10.0, 20.0])

    features.apply_broadness(0.25, torch.tensor([0, 1]))

    expected = torch.tensor([15.0, 15.0, 10.0, 20.0]) * 0.75
    assert torch.allclose(features._ema_activity, expected)
