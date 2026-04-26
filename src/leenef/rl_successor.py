"""Successor-feature NEF agents for sparse-reward RL."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .rl import NEFFeatures
from .rl_novelty import novelty_scores


def truncated_successor_targets(features: Tensor, gamma: float, horizon: int) -> Tensor:
    """Return truncated Monte Carlo successor features for one trajectory.

    For features ``phi_t`` this computes
    ``sum_{k=0}^{min(horizon, T-t)-1} gamma^k phi_{t+k}``.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be 2D, got shape {tuple(features.shape)}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")

    targets = torch.zeros_like(features)
    for t in range(features.shape[0]):
        acc = torch.zeros(features.shape[1], dtype=features.dtype, device=features.device)
        for k in reversed(range(min(horizon, features.shape[0] - t))):
            acc = features[t + k] + gamma * acc
        targets[t] = acc
    return targets


class NEFSuccessorFeatureAgent:
    """NEF successor-feature agent with analytical solves.

    The encoder receives an augmented ``[observation, one_hot(action)]`` vector.
    Completed trajectories produce two supervised targets:

    - truncated successor features, solved as a multi-output decoder
    - immediate reward, solved as a scalar decoder

    Greedy control composes them as ``Q(s, a) = psi(s, a) @ reward_weight``.
    """

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        n_neurons: int = 2000,
        gamma: float = 0.99,
        alpha: float = 1e-2,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        solve_every: int = 10,
        successor_horizon: int = 50,
        forget_factor: float | None = 1.0,
        novelty_bonus: float = 0.0,
        novelty_transform: str = "sqrt",
        activation: str = "abs",
        encoder_strategy: str = "hypersphere",
        gain: float | tuple[float, float] = (0.5, 2.0),
        features_kwargs: dict | None = None,
        seed: int = 0,
    ):
        if successor_horizon <= 0:
            raise ValueError(f"successor_horizon must be positive, got {successor_horizon!r}")
        if forget_factor is not None and not 0 < forget_factor <= 1:
            raise ValueError(f"forget_factor must be in (0, 1], got {forget_factor!r}")

        self.d_obs = d_obs
        self.n_actions = n_actions
        self.n_neurons = n_neurons
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.solve_every = solve_every
        self.successor_horizon = successor_horizon
        self.forget_factor = forget_factor
        self.novelty_bonus = novelty_bonus
        self.novelty_transform = novelty_transform
        self.rng = np.random.default_rng(seed)

        feat_kw = dict(features_kwargs or {})
        self.features = NEFFeatures(
            d_obs + n_actions,
            n_neurons,
            activation=activation,
            encoder_strategy=encoder_strategy,
            gain=gain,
            seed=seed,
            **feat_kw,
        )

        self.successor_decoder = torch.zeros(n_neurons, n_neurons)
        self.reward_weight = torch.zeros(n_neurons)

        self._ata = torch.zeros(n_neurons, n_neurons, dtype=torch.float64)
        self._aty_successor = torch.zeros(n_neurons, n_neurons, dtype=torch.float64)
        self._aty_reward = torch.zeros(n_neurons, dtype=torch.float64)
        self._n_accumulated = 0

        self._episode_transitions: list[tuple[np.ndarray, int, float]] = []
        self._episode = 0

    def _epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self._episode / self.epsilon_decay
        )

    def _augment(self, obs: Tensor, actions: Tensor) -> Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        actions = actions.to(dtype=torch.long).reshape(-1)
        if obs.shape[0] != actions.shape[0]:
            raise ValueError(
                f"obs batch and actions must align, got {obs.shape[0]} and {actions.shape[0]}"
            )
        one_hot = torch.nn.functional.one_hot(actions, num_classes=self.n_actions).float()
        return torch.cat([obs.float(), one_hot], dim=1)

    def _candidate_features(self, obs: np.ndarray) -> Tensor:
        obs_t = torch.tensor(
            np.repeat(np.asarray(obs, dtype=np.float32)[None, :], self.n_actions, 0)
        )
        actions = torch.arange(self.n_actions)
        return self.features.encode(self._augment(obs_t, actions))

    def select_action(self, obs: np.ndarray) -> int:
        if self.rng.random() < self._epsilon():
            return int(self.rng.integers(self.n_actions))
        q = self.q_values(obs)
        if self.novelty_bonus != 0.0:
            q = q + self.novelty_bonus * self.action_novelty(obs)
        return int(q.argmax().item())

    def q_values(self, obs: np.ndarray) -> Tensor:
        with torch.no_grad():
            phi = self._candidate_features(obs)
            successor = phi @ self.successor_decoder
            return successor @ self.reward_weight

    def action_novelty(self, obs: np.ndarray) -> Tensor:
        """Return per-action leverage novelty from the shared successor Gram matrix."""
        with torch.no_grad():
            phi = self._candidate_features(obs)
            return novelty_scores(phi, self._ata, self.alpha, transform=self.novelty_transform)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        del next_obs
        augmented = self._augment(
            torch.from_numpy(np.asarray(obs, dtype=np.float32)),
            torch.tensor([action]),
        )
        self.features.update_stats(augmented)
        self._episode_transitions.append((obs.copy(), int(action), float(reward)))
        if done:
            self._flush_episode()

    def _flush_episode(self) -> None:
        transitions = self._episode_transitions
        self._episode_transitions = []
        if not transitions:
            return

        obs = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in transitions], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float64)
        with torch.no_grad():
            phi = self.features.encode(self._augment(obs, actions)).double()
        successor_targets = truncated_successor_targets(
            phi.float(),
            self.gamma,
            self.successor_horizon,
        ).double()

        if self.forget_factor is not None:
            beta = float(self.forget_factor)
            self._ata.mul_(beta)
            self._aty_successor.mul_(beta)
            self._aty_reward.mul_(beta)

        self._ata.addmm_(phi.T, phi)
        self._aty_successor.addmm_(phi.T, successor_targets)
        self._aty_reward.addmv_(phi.T, rewards)
        self._n_accumulated += len(transitions)

    @staticmethod
    def _solve_spd(matrix: Tensor, rhs: Tensor) -> Tensor:
        chol, info = torch.linalg.cholesky_ex(matrix)
        if int(info.item()) == 0:
            rhs_2d = rhs.unsqueeze(1) if rhs.dim() == 1 else rhs
            solved = torch.cholesky_solve(rhs_2d, chol)
            return solved.squeeze(1) if rhs.dim() == 1 else solved
        return torch.linalg.solve(matrix, rhs)

    def solve(self) -> None:
        if self._episode_transitions:
            self._flush_episode()
        if self._n_accumulated == 0:
            return

        ata = self._ata.clone()
        reg = self.alpha * torch.trace(ata) / self.n_neurons
        ata.diagonal().add_(reg.clamp(min=self.alpha))
        try:
            successor = self._solve_spd(ata, self._aty_successor)
            reward = self._solve_spd(ata, self._aty_reward)
        except torch.linalg.LinAlgError:
            successor = torch.linalg.lstsq(ata, self._aty_successor).solution
            reward = torch.linalg.lstsq(ata, self._aty_reward.unsqueeze(1)).solution.squeeze(1)

        self.successor_decoder = successor.float()
        self.reward_weight = reward.float()

    def end_episode(self) -> None:
        self._episode += 1
        if self._episode % self.solve_every == 0:
            self.solve()

    def warmup(
        self,
        env_step: Callable[[int], tuple[np.ndarray, float, bool, bool, dict]] | None = None,
        *,
        env=None,
        n_episodes: int = 20,
        seed: int | None = None,
    ) -> None:
        if env is None and env_step is None:
            raise ValueError("Provide either env or env_step")
        if env is None:
            raise ValueError("Callable env_step warmup still needs reset-capable env context")

        for i in range(n_episodes):
            if seed is not None:
                action_space = getattr(env, "action_space", None)
                if action_space is not None and hasattr(action_space, "seed"):
                    action_space.seed(seed + i)
                obs, _ = env.reset(seed=seed + i)
            else:
                obs, _ = env.reset()
            done = False
            while not done:
                action = int(self.rng.integers(self.n_actions))
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
            self.end_episode()
        self.solve()
