"""Tests for RL utilities."""

import numpy as np
import torch

from leenef.rl import NEFFeatures, NEFFQIAgent


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


def test_deferred_rls_accumulation_matches_per_episode_recurrence():
    agent = NEFFQIAgent(
        d_obs=2,
        n_actions=3,
        n_neurons=8,
        target_mode="nstep",
        forget_factor=0.9,
        seed=0,
    )
    episodes = [
        [
            (np.array([0.0, 1.0]), 0, 1.0),
            (np.array([1.0, 0.5]), 2, 0.5),
            (np.array([0.5, -0.5]), 0, -0.25),
        ],
        [
            (np.array([1.5, -1.0]), 1, 2.0),
            (np.array([-0.5, 0.25]), 2, -1.0),
        ],
        [
            (np.array([0.25, 0.75]), 2, 0.75),
            (np.array([-1.0, 0.0]), 1, 0.25),
        ],
    ]

    expected_ata = torch.zeros_like(agent._ata)
    expected_aty = torch.zeros_like(agent._aty)
    expected_counts = torch.zeros_like(agent._n_accumulated)
    beta = agent.forget_factor

    for episode in episodes:
        states = torch.tensor(np.array([t[0] for t in episode]), dtype=torch.float32)
        actions = np.array([t[1] for t in episode])
        targets = torch.tensor(np.array([t[2] for t in episode]), dtype=torch.float64)
        with torch.no_grad():
            phi = agent.features.encode(states).double()

        for action in range(agent.n_actions):
            expected_ata[action].mul_(beta)
            expected_aty[action].mul_(beta)
            mask = actions == action
            if mask.sum() == 0:
                continue
            phi_a = phi[mask]
            y_a = targets[mask]
            expected_ata[action].addmm_(phi_a.T, phi_a)
            expected_aty[action].addmv_(phi_a.T, y_a)
            expected_counts[action] += int(mask.sum())

        agent._accumulate_episode(episode)

    assert len(agent._pending_stats) == len(episodes)

    agent._flush_pending_stats()

    torch.testing.assert_close(agent._ata, expected_ata)
    torch.testing.assert_close(agent._aty, expected_aty)
    assert torch.equal(agent._n_accumulated, expected_counts)
    assert agent._pending_stats == []


def test_solve_from_stats_flushes_deferred_rls_episodes():
    agent = NEFFQIAgent(
        d_obs=2,
        n_actions=2,
        n_neurons=6,
        target_mode="nstep",
        forget_factor=0.95,
        seed=1,
    )
    episode = [
        (np.array([0.0, 0.0]), 0, 1.0),
        (np.array([1.0, 0.0]), 0, 0.5),
        (np.array([0.0, 1.0]), 1, -0.25),
    ]

    agent._accumulate_episode(episode)

    assert len(agent._pending_stats) == 1

    agent._solve_from_stats()

    assert agent._pending_stats == []
    assert int(agent._n_accumulated.sum().item()) == len(episode)
