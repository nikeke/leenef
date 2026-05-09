"""Tests for the live RL primitives in :mod:`leenef.rl`."""

from __future__ import annotations

import numpy as np
import torch

from leenef.rl import NEFFeatures, NEFFQIAgent, NEFFQIEnsemble


def test_nef_features_are_seed_deterministic_and_preserve_global_rng():
    torch.manual_seed(999)
    before = torch.get_rng_state()

    a = NEFFeatures(d_in=3, n_neurons=16, seed=7)
    after = torch.get_rng_state()
    b = NEFFeatures(d_in=3, n_neurons=16, seed=7)

    assert torch.equal(before, after)
    assert torch.equal(a.encoders, b.encoders)
    assert torch.equal(a._gain, b._gain)
    assert torch.equal(a.bias, b.bias)


def test_apply_broadness_uses_alive_mean_for_cumulative_stats():
    features = NEFFeatures(d_in=2, n_neurons=4, track_activations=True)
    features._act_sum = torch.tensor([1.0, 3.0, 10.0, 20.0])
    features._act_count = 5.0

    features.apply_broadness(0.25, torch.tensor([0, 1]))

    expected = torch.tensor([15.0, 15.0, 10.0, 20.0]) * 0.75
    assert torch.allclose(features._act_sum, expected)
    assert features._act_count == 3.75


def test_apply_broadness_uses_alive_mean_for_ema_stats():
    features = NEFFeatures(d_in=2, n_neurons=4, track_activations=True, activity_decay=0.99)
    features._ema_activity = torch.tensor([1.0, 3.0, 10.0, 20.0])

    features.apply_broadness(0.25, torch.tensor([0, 1]))

    expected = torch.tensor([15.0, 15.0, 10.0, 20.0]) * 0.75
    assert torch.allclose(features._ema_activity, expected)


def test_nstep_returns_use_truncated_discounted_horizon():
    agent = NEFFQIAgent(d_obs=2, n_actions=2, n_neurons=8, gamma=0.5, n_step=2)
    transitions = [
        (np.array([0.0, 0.0]), 0, 1.0),
        (np.array([1.0, 0.0]), 1, 2.0),
        (np.array([2.0, 0.0]), 0, 4.0),
    ]

    out = agent._compute_nstep_returns(transitions)

    assert [round(row[2], 6) for row in out] == [2.0, 4.0, 4.0]


def test_lambda_returns_blend_bootstrap_and_mc_continuation():
    agent = NEFFQIAgent(
        d_obs=2,
        n_actions=2,
        n_neurons=8,
        gamma=0.5,
        td_lambda=0.5,
        target_mode="mc",
    )
    agent.W.zero_()
    transitions = [
        (np.array([0.0, 0.0]), 0, 1.0, np.array([1.0, 0.0])),
        (np.array([1.0, 0.0]), 1, 2.0, np.array([2.0, 0.0])),
        (np.array([2.0, 0.0]), 0, 4.0, np.array([3.0, 0.0])),
    ]

    out = agent._compute_lambda_returns(transitions)

    assert [round(row[2], 6) for row in out] == [1.75, 3.0, 4.0]


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
    beta = float(agent.forget_factor)

    encoded = []
    for episode in episodes:
        states = torch.tensor(np.array([t[0] for t in episode]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in episode], dtype=torch.long)
        targets = torch.tensor(np.array([t[2] for t in episode]), dtype=torch.float64)
        with torch.no_grad():
            phi = agent.features.encode(states).double()
        encoded.append((phi, actions, targets))
        agent._accumulate_episode(episode)

    n_pending = len(encoded)
    for action in range(agent.n_actions):
        phi_chunks = []
        target_chunks = []
        for i, (phi, actions, targets) in enumerate(encoded):
            mask = actions == action
            if not bool(mask.any()):
                continue
            sqrt_weight = beta ** ((n_pending - 1 - i) / 2)
            phi_chunks.append(phi[mask] * sqrt_weight)
            target_chunks.append(targets[mask] * sqrt_weight)
            expected_counts[action] += int(mask.sum())
        if phi_chunks:
            phi_a = torch.cat(phi_chunks)
            y_a = torch.cat(target_chunks)
            expected_ata[action].addmm_(phi_a.T, phi_a)
            expected_aty[action].addmv_(phi_a.T, y_a)

    agent._flush_pending_stats()

    torch.testing.assert_close(agent._ata, expected_ata)
    torch.testing.assert_close(agent._aty, expected_aty)
    assert torch.equal(agent._n_accumulated, expected_counts)
    assert agent._pending_stats == []


class _TinyActionSpace:
    def __init__(self, n: int):
        self.n = n
        self.seed_values: list[int] = []

    def seed(self, seed: int) -> None:
        self.seed_values.append(seed)


class _TinyObservationSpace:
    shape = (2,)


class _TinyEnv:
    observation_space = _TinyObservationSpace()

    def __init__(self):
        self.action_space = _TinyActionSpace(2)
        self.reset_seeds: list[int | None] = []
        self._seed = 0
        self._step = 0

    def reset(self, seed=None):
        self._seed = 0 if seed is None else int(seed)
        self._step = 0
        self.reset_seeds.append(seed)
        return np.array([self._seed % 3, 0.0], dtype=np.float32), {}

    def step(self, action: int):
        self._step += 1
        obs = np.array([self._seed % 5, action + self._step], dtype=np.float32)
        reward = float(self._step - action)
        terminated = self._step >= 2
        return obs, reward, terminated, False, {}


def test_ensemble_warmup_broadcasts_one_environment_stream_to_all_members():
    env = _TinyEnv()
    ensemble = NEFFQIEnsemble(
        n_members=2,
        d_obs=2,
        n_actions=2,
        n_neurons=12,
        target_mode="nstep",
        forget_factor=0.95,
        solve_every=1,
        n_step=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        base_seed=4,
    )

    ensemble.warmup(env=env, n_episodes=3, seed=10)

    assert env.reset_seeds == [10, 11, 12]
    assert env.action_space.seed_values == [10, 11, 12]
    assert all(member._episode == 3 for member in ensemble.members)
    assert all(int(member._n_accumulated.sum()) == 6 for member in ensemble.members)
