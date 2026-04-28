"""NEF Successor Features (D1) — analytical-solve value backbone.

Two analytical decoders share the existing :class:`NEFFeatures` encoder:

- a **successor decoder** ``Ψ_a`` per discrete action, mapping
  ``φ(s) → ψ(s, a) = φ(s) @ Ψ_a`` where
  ``ψ(s, a) ≈ Σ_k γ^k E[φ(s_{t+k}) | s_t = s, a_t = a]`` under the
  data-collection policy (truncated MC on features over a horizon of
  ``n_step``).  This is the "n-step return on φ" — same move that
  resolved the FQI bootstrap dilemma in ``rl_notes.md`` §6, applied
  to features instead of rewards.

- a **reward decoder** ``w_a`` per action mapping
  ``φ(s) → r̂(s, a) ≈ φ(s) · w_a``.

Combined: ``Q(s, a) = ψ(s, a) · w_a = (φ(s) @ Ψ_a) · w_a``.

Both decoders are obtained from the *same* sufficient-statistic
machinery as :class:`NEFFQIAgent`: per-action ``AᵀA`` accumulators with
optional RLS forgetting.  The new pieces are a per-action ``AᵀΦ``
accumulator (right-hand side for ``Ψ_a``) and a per-action ``Aᵀr``
accumulator (right-hand side for ``w_a``).  No bootstrapping; targets
are completed-trajectory truncated-MC sums.

This module exposes :class:`NEFSuccessor`, the encoder + solver core,
and :class:`NEFSFAgent`, a thin training-loop wrapper around it that
mirrors :class:`NEFFQIAgent`'s public surface (``select_action`` /
``update`` / ``q_values``).  The agent also implements the D2 leverage
novelty bonus: an intrinsic reward
``r_int(s) = β · sqrt(φ(s)ᵀ (AᵀA + αI)⁻¹ φ(s))`` derived from the
cached sufficient-statistic inverse — zero new state, one scalar
hyperparameter, and the exact analytical-solve analog of RND.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .rl import NEFFeatures


class NEFSuccessor:
    """Successor-feature analytical solve over a fixed NEF encoder.

    Encode once, solve twice: a multi-output regression for the
    per-action successor decoder ``Ψ_a`` and a scalar regression for
    the per-action reward weight ``w_a``.  Both decoders share the same
    per-action ``AᵀA``, so a single Cholesky factorization resolves
    them simultaneously.

    Targets are truncated-MC successor returns over ``n_step`` future
    timesteps, computed from completed-episode buffers.  Episode ends
    truncate the sum naturally (no bootstrap across termination).

    Args:
        d_obs: observation dimensionality.
        n_actions: number of discrete actions.
        n_neurons: number of encoding neurons.
        gamma: discount factor.
        alpha: Tikhonov regularization (trace-scaled, mirroring
            :class:`NEFFQIAgent`).
        n_step: truncated successor horizon.  ``None`` means full MC
            over the episode.
        forget_factor: optional RLS β for exponentially-weighted
            sufficient statistics.  ``None`` keeps additive
            statistics; with this skeleton we do not yet implement a
            replay-buffer fallback.
        features_kwargs: forwarded to :class:`NEFFeatures`.
        seed: random seed for the encoder.
    """

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        n_neurons: int = 2000,
        gamma: float = 0.99,
        alpha: float = 1e-2,
        n_step: int | None = 50,
        forget_factor: float | None = None,
        features_kwargs: dict | None = None,
        seed: int = 0,
    ):
        if forget_factor is not None and not 0.0 < forget_factor <= 1.0:
            raise ValueError(f"forget_factor must be in (0, 1], got {forget_factor!r}")
        if n_step is not None and n_step <= 0:
            raise ValueError(f"n_step must be positive or None, got {n_step!r}")

        self.d_obs = d_obs
        self.n_actions = int(n_actions)
        self.n_neurons = int(n_neurons)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.n_step = n_step
        self.forget_factor = forget_factor

        self.features = NEFFeatures(d_obs, n_neurons, seed=seed, **(features_kwargs or {}))

        # Per-action sufficient statistics, all float64.
        self._ata = torch.zeros(self.n_actions, n_neurons, n_neurons, dtype=torch.float64)
        self._aty_phi = torch.zeros(
            self.n_actions, n_neurons, n_neurons, dtype=torch.float64
        )  # right-hand side for Ψ_a:  Σ φ(s_t) ⊗ ψ̂(s_t,a)
        self._aty_r = torch.zeros(
            self.n_actions, n_neurons, dtype=torch.float64
        )  # right-hand side for w_a:    Σ φ(s_t) · r_t
        self._n_accumulated = torch.zeros(self.n_actions, dtype=torch.long)

        # Decoders (zero-initialized; populated by solve).
        self.Psi = torch.zeros(self.n_actions, n_neurons, n_neurons, dtype=torch.float32)
        self.W_r = torch.zeros(self.n_actions, n_neurons, dtype=torch.float32)

        # Episode buffer of (obs, action, reward) triples; flushed on done.
        self._episode_transitions: list[tuple[np.ndarray, int, float]] = []

        self._episode = 0
        self._total_steps = 0
        self._solve_count = 0

    # -- ingestion ------------------------------------------------------------

    def update_stats(self, batch: Tensor) -> None:
        """Forward to the underlying :class:`NEFFeatures` Welford stats."""
        self.features.update_stats(batch)

    def add_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Record one transition and flush sufficient statistics on episode end."""
        self.features.update_stats(torch.from_numpy(np.stack([obs, next_obs])).float())
        self._episode_transitions.append((obs, int(action), float(reward)))
        self._total_steps += 1
        if done:
            self._flush_episode()
            self._episode += 1

    def _flush_episode(self) -> None:
        transitions = self._episode_transitions
        self._episode_transitions = []
        if not transitions:
            return

        states = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32)
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions], dtype=np.float64)

        with torch.no_grad():
            phi = self.features.encode(states).double()  # (T, n)

        psi_targets = self._compute_successor_targets(phi, rewards.size)

        beta = self.forget_factor
        if beta is not None:
            self._ata.mul_(beta)
            self._aty_phi.mul_(beta)
            self._aty_r.mul_(beta)

        for a in range(self.n_actions):
            mask = actions == a
            if not mask.any():
                continue
            phi_a = phi[mask]  # (T_a, n)
            psi_a = psi_targets[mask]  # (T_a, n)
            r_a = torch.from_numpy(rewards[mask])  # (T_a,)

            self._ata[a].addmm_(phi_a.T, phi_a)
            self._aty_phi[a].addmm_(phi_a.T, psi_a)
            self._aty_r[a].addmv_(phi_a.T, r_a)
            self._n_accumulated[a] += int(mask.sum())

    def _compute_successor_targets(self, phi: Tensor, T: int) -> Tensor:
        """Truncated-MC successor targets ψ̂_t = Σ_{k=0}^{H_t-1} γ^k φ(s_{t+k}).

        ``phi`` is ``(T, n_neurons)`` for a single completed episode and
        ``H_t = min(n_step, T - t)``.  Episode end truncates the sum (no
        bootstrap).  Pure-MC case (``n_step is None`` or ``>= T``) uses
        an O(T) backward recurrence ``G ← φ_t + γ G``; truncated case
        uses a sliding-window backward recurrence that subtracts the
        term falling out of the horizon, also O(T).
        """
        psi = torch.zeros_like(phi)
        if self.n_step is None or self.n_step >= T:
            G = torch.zeros_like(phi[0])
            for t in reversed(range(T)):
                G = phi[t] + self.gamma * G
                psi[t] = G
            return psi

        H = int(self.n_step)
        gamma_H = self.gamma**H
        G = torch.zeros_like(phi[0])
        for t in reversed(range(T)):
            G = phi[t] + self.gamma * G
            back_idx = t + H
            if back_idx < T:
                # ψ_{t+1} held terms φ_{t+1}..φ_{t+H}; the new ψ_t window
                # is φ_t..φ_{t+H-1}, so the φ_{t+H} contribution leaves.
                G = G - gamma_H * phi[back_idx]
            psi[t] = G
        return psi

    def _compute_reward_targets(self, rewards: np.ndarray) -> np.ndarray:
        """Reward target is the immediate reward for that (s, a) transition."""
        return rewards

    # -- solving --------------------------------------------------------------

    @staticmethod
    def _solve_spd(matrix: Tensor, rhs: Tensor) -> Tensor:
        """Cholesky-backed multi-RHS solve with a generic fallback."""
        chol, info = torch.linalg.cholesky_ex(matrix)
        if int(info.item()) == 0:
            if rhs.dim() == 1:
                return torch.cholesky_solve(rhs.unsqueeze(1), chol).squeeze(1)
            return torch.cholesky_solve(rhs, chol)
        return torch.linalg.solve(matrix, rhs)

    def solve(self) -> None:
        """Solve for all per-action ``Ψ_a`` and ``w_a`` simultaneously.

        For each action ``a`` with at least 10 accumulated samples,
        a single Cholesky of ``(AᵀA + αI)`` resolves both right-hand
        sides at once.  Underused actions retain their previous
        decoders so the policy remains evaluable.
        """
        for a in range(self.n_actions):
            if self._n_accumulated[a] < 10:
                continue

            ATA = self._ata[a].clone()
            reg = self.alpha * torch.trace(ATA) / self.n_neurons
            ATA.diagonal().add_(reg.clamp(min=self.alpha))

            try:
                Psi_a = self._solve_spd(ATA, self._aty_phi[a])  # (n, n)
                w_a = self._solve_spd(ATA, self._aty_r[a])  # (n,)
            except torch.linalg.LinAlgError:
                Psi_a = torch.linalg.lstsq(ATA, self._aty_phi[a]).solution
                w_a = torch.linalg.lstsq(ATA, self._aty_r[a].unsqueeze(1)).solution.squeeze(1)

            self.Psi[a] = Psi_a.float()
            self.W_r[a] = w_a.float()

        self._solve_count += 1

    # -- inference ------------------------------------------------------------

    def successor(self, obs: np.ndarray | Tensor) -> Tensor:
        """Per-action successor representation ``ψ(s, a)`` of shape ``(n_actions, n_neurons)``."""
        with torch.no_grad():
            s = obs if isinstance(obs, Tensor) else torch.from_numpy(obs).float()
            phi = self.features.encode(s)  # (n,) or (B, n)
            if phi.dim() == 1:
                # Ψ stored as (n_actions, n, n); ψ_a = φ @ Ψ_a -> (n_actions, n)
                return torch.einsum("n,anm->am", phi, self.Psi)
            return torch.einsum("bn,anm->bam", phi, self.Psi)

    def q_values(self, obs: np.ndarray | Tensor) -> Tensor:
        """Q(s, a) = ψ(s, a) · w_a, returning ``(n_actions,)`` or ``(B, n_actions)``."""
        with torch.no_grad():
            s = obs if isinstance(obs, Tensor) else torch.from_numpy(obs).float()
            phi = self.features.encode(s)
            if phi.dim() == 1:
                psi = torch.einsum("n,anm->am", phi, self.Psi)  # (a, n)
                return (psi * self.W_r).sum(dim=1)  # (a,)
            psi = torch.einsum("bn,anm->bam", phi, self.Psi)  # (b, a, n)
            return (psi * self.W_r.unsqueeze(0)).sum(dim=2)  # (b, a)

    def reward_estimate(self, obs: np.ndarray | Tensor) -> Tensor:
        """One-step reward estimate ``r̂(s, a) = φ(s) · w_a`` for each action."""
        with torch.no_grad():
            s = obs if isinstance(obs, Tensor) else torch.from_numpy(obs).float()
            phi = self.features.encode(s)
            if phi.dim() == 1:
                return phi @ self.W_r.T  # (a,)
            return phi @ self.W_r.T  # (b, a)

    # -- diagnostics passthrough ---------------------------------------------

    def gram_matrix(self, action: int | None = None) -> Tensor:
        """Return ``AᵀA`` for a specific action or summed across actions."""
        if action is not None:
            return self._ata[action].clone()
        return self._ata.sum(dim=0)


class NEFSFAgent:
    """Thin training-loop wrapper around :class:`NEFSuccessor`.

    Mirrors :class:`NEFFQIAgent`'s public surface (``select_action`` /
    ``update`` / ``q_values``) so it slots into the existing
    :mod:`benchmarks.run_rl` / :mod:`benchmarks.rl_repro` harnesses,
    but the value backbone is the per-action successor solve described
    in :class:`NEFSuccessor` rather than per-action FQI.

    Optional D2 leverage novelty bonus: when ``novelty_beta > 0``, an
    intrinsic reward
    ``r_int(s) = β · sqrt(φ(s)ᵀ (AᵀA + αI)⁻¹ φ(s))`` is added to the
    extrinsic reward before episode targets are computed.  This is
    free — the inverse is cached after each solve from the same
    sufficient statistics already maintained by the successor core.
    Tracked separately for diagnostics so the reported reward stays
    pure-extrinsic.

    The agent solves once per ``solve_every`` *episodes* (not steps):
    successor targets are completed-trajectory MC sums, so per-step
    solving does not add information.

    Args:
        d_obs: observation dimensionality.
        n_actions: discrete action count.
        n_neurons: encoder size.
        gamma: discount factor.
        alpha: trace-scaled Tikhonov regularization.
        n_step: truncated successor horizon.
        forget_factor: optional RLS β.
        epsilon_start / epsilon_end / epsilon_decay: ε-greedy schedule
            (same form as :class:`NEFFQIAgent`).
        solve_every: episodes between successor solves.  ``solve()``
            also refreshes the leverage-novelty inverse.
        novelty_beta: D2 intrinsic-reward coefficient (``0`` disables).
        novelty_eps: numerical floor for the leverage square root.
        warmup_episodes: episodes during which actions are drawn from
            ``epsilon_start`` (no greedy action) regardless of decay.
        features_kwargs: forwarded to :class:`NEFFeatures`.
        seed: random seed.
    """

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        n_neurons: int = 2000,
        gamma: float = 0.99,
        alpha: float = 1e-2,
        n_step: int | None = 50,
        forget_factor: float | None = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,
        solve_every: int = 10,
        novelty_beta: float = 0.0,
        novelty_eps: float = 1e-8,
        warmup_episodes: int = 0,
        features_kwargs: dict | None = None,
        seed: int = 0,
    ):
        if novelty_beta < 0:
            raise ValueError(f"novelty_beta must be >= 0, got {novelty_beta!r}")
        if warmup_episodes < 0:
            raise ValueError(f"warmup_episodes must be >= 0, got {warmup_episodes!r}")

        self.successor = NEFSuccessor(
            d_obs=d_obs,
            n_actions=n_actions,
            n_neurons=n_neurons,
            gamma=gamma,
            alpha=alpha,
            n_step=n_step,
            forget_factor=forget_factor,
            features_kwargs=features_kwargs,
            seed=seed,
        )
        self.n_actions = int(n_actions)
        self.n_neurons = int(n_neurons)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(epsilon_decay)
        self.solve_every = int(solve_every)
        self.novelty_beta = float(novelty_beta)
        self.novelty_eps = float(novelty_eps)
        self.warmup_episodes = int(warmup_episodes)

        self.rng = np.random.default_rng(seed)

        # Cached (AᵀA + αI)⁻¹ summed across actions for novelty leverage.
        self._novelty_inv: Tensor | None = None
        # Per-episode intrinsic-reward bookkeeping (diagnostics only).
        self._intrinsic_episode = 0.0
        self._extrinsic_episode = 0.0
        self.last_extrinsic_reward = 0.0
        self.last_intrinsic_reward = 0.0

        self._episode = 0
        self._total_steps = 0
        self._solve_count = 0

    # -- exposed convenience --------------------------------------------------

    @property
    def features(self) -> NEFFeatures:
        return self.successor.features

    @property
    def Psi(self) -> Tensor:  # noqa: N802 (mirrors NEFSuccessor)
        return self.successor.Psi

    @property
    def W_r(self) -> Tensor:  # noqa: N802
        return self.successor.W_r

    # -- exploration schedule -------------------------------------------------

    def _epsilon(self) -> float:
        if self._episode < self.warmup_episodes:
            return self.epsilon_start
        decayed = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -(self._episode - self.warmup_episodes) / max(1, self.epsilon_decay)
        )
        return float(decayed)

    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy with respect to the current Q estimate."""
        if self.rng.random() < self._epsilon() or self._solve_count == 0:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.successor.q_values(obs)
            return int(q.argmax().item())

    def q_values(self, obs: np.ndarray) -> Tensor:
        return self.successor.q_values(obs)

    # -- novelty --------------------------------------------------------------

    def _refresh_novelty_inverse(self) -> None:
        if self.novelty_beta == 0.0:
            self._novelty_inv = None
            return
        ata = self.successor.gram_matrix(action=None).clone()
        reg = self.alpha * torch.trace(ata) / self.n_neurons
        ata.diagonal().add_(reg.clamp(min=self.alpha))
        try:
            chol, info = torch.linalg.cholesky_ex(ata)
            if int(info.item()) == 0:
                inv = torch.cholesky_inverse(chol)
            else:
                inv = torch.linalg.pinv(ata)
        except torch.linalg.LinAlgError:
            inv = torch.linalg.pinv(ata)
        self._novelty_inv = inv  # float64

    def novelty(self, obs: np.ndarray) -> float:
        """``β · sqrt(φ(s)ᵀ (AᵀA + αI)⁻¹ φ(s))`` (0 before any solve)."""
        if self.novelty_beta == 0.0 or self._novelty_inv is None:
            return 0.0
        with torch.no_grad():
            s = torch.from_numpy(obs).float()
            phi = self.successor.features.encode(s).double()
            quad = float((phi @ self._novelty_inv @ phi).item())
        return self.novelty_beta * float(np.sqrt(max(quad, 0.0) + self.novelty_eps))

    # -- ingestion ------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Buffer one transition; flush + maybe solve at episode end.

        The reward stored for the successor solve is ``reward + r_int``
        where ``r_int`` is the leverage novelty bonus.  The bonus is
        also tracked separately on ``last_intrinsic_reward`` and the
        per-episode accumulators so harnesses can report extrinsic-
        only return.
        """
        intrinsic = self.novelty(obs) if self.novelty_beta > 0.0 else 0.0
        self.last_extrinsic_reward = float(reward)
        self.last_intrinsic_reward = float(intrinsic)
        self._extrinsic_episode += float(reward)
        self._intrinsic_episode += float(intrinsic)

        augmented = float(reward) + intrinsic
        self.successor.add_transition(obs, int(action), augmented, next_obs, bool(done))
        self._total_steps += 1

        if done:
            self._episode += 1
            if self.solve_every > 0 and self._episode % self.solve_every == 0:
                self.solve()
            self._intrinsic_episode = 0.0
            self._extrinsic_episode = 0.0

    def solve(self) -> None:
        """Run the per-action successor solve and refresh the novelty inverse."""
        self.successor.solve()
        self._solve_count += 1
        self._refresh_novelty_inverse()

    def warmup(self, *, env, n_episodes: int = 20, seed: int | None = None) -> None:
        """Random-policy warmup to seed Welford stats and prime sufficient stats.

        Mirrors :meth:`NEFFQIAgent.warmup`.  After collecting random
        transitions, performs a single solve so the policy is
        immediately evaluable (no zero-Q random-tie window).
        """
        for i in range(n_episodes):
            obs, _ = env.reset() if seed is None else env.reset(seed=seed + i)
            done = False
            while not done:
                action = int(self.rng.integers(self.n_actions))
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
        self.solve()


class NEFHindsightSFAgent(NEFSFAgent):
    """SF agent with hindsight goal-conditioning (D4).

    Exploits the SF decomposition ``Q(s, a) = ψ(s, a) · w_a`` to
    *redefine* the reward weight on the fly without retraining:

    - Pick any goal observation ``g`` (sampled from a ring buffer of
      recently-visited states).
    - Define ``r_g(s) = φ(s) · u(g)`` with ``u(g) = φ(g) / ‖φ(g)‖``.
      This is a unit-norm "feature-similarity" reward.
    - Then the goal-conditioned Q is

      ``Q(s, a; g) = ψ(s, a) · u(g) = (φ(s) @ Ψ_a) · φ(g) / ‖φ(g)‖``

      with **no extra solve** — only the cached ``Ψ_a`` is used.

    Action selection during training uses ``Q(s, a; g)`` for a goal
    held fixed for ``goal_horizon`` consecutive steps, then resamples.
    The goal-conditioned greedy action commits to a directed motor
    sequence — exactly the failure mode (sustained back-and-forth in
    MC) that ε-greedy random walks cannot bridge.

    The successor solve and reward decoder are unchanged: ``Ψ_a`` is
    learned from real trajectories (via truncated-MC successor
    targets) and ``w_a`` from real rewards.  Hindsight is pure
    behavior — the *value* the agent acts on at training time is
    re-derived from the same ψ via a fresh weight vector ``u(g)``.

    Evaluation always uses the extrinsic Q (``ψ · w_a``); goal
    conditioning is only active during data collection.

    Args:
        goal_buffer_size: ring-buffer capacity of past observations
            used as candidate goals.
        goal_horizon: number of consecutive training steps a goal is
            held before resampling.
        goal_warmup_solves: number of solves required before goal-
            conditioned action selection kicks in.  Until then the
            agent is pure ε-greedy on the extrinsic Q (since ``Ψ_a``
            is meaningless before the first solve).
        goal_eps: numerical floor for ``‖φ(g)‖`` to keep ``u(g)``
            well-defined.
        goal_mode: how goal directions are sampled.

            ``"buffer"`` (default) draws a visited observation
            uniformly from the ring buffer and uses
            ``u(g) = φ(g)/‖φ(g)‖``.  Best for tasks where the buffer
            already spans the relevant region (e.g. dense-reward
            chains, Acrobot once swings start sampling state space).

            ``"random_direction"`` draws ``u`` directly as a
            uniformly-random unit vector in feature space (no buffer
            lookup, no encoder pass).  Each goal direction defines a
            fresh "task" ``r(s) = φ(s)·u``.  Targets unexplored
            feature-space directions whose Ψ has been built only from
            *trajectories that crossed them*: when the agent has been
            stuck in a small neighbourhood, random directions give it
            sustained outward pushes that the buffer-mode goals (all
            inside that neighbourhood) cannot.  Recommended for the
            MountainCar-style "agent never leaves the valley" failure
            mode.

            ``"mixed"`` flips a coin (``p = goal_random_prob``) per
            resample between buffer and random_direction modes — the
            buffer side exploits known reachable structure, the
            random_direction side keeps pushing on the frontier.
        goal_random_prob: probability of drawing a random_direction
            goal in ``"mixed"`` mode (ignored otherwise).
        Otherwise identical to :class:`NEFSFAgent`.
    """

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        *,
        goal_buffer_size: int = 1024,
        goal_horizon: int = 50,
        goal_warmup_solves: int = 1,
        goal_eps: float = 1e-6,
        goal_mode: str = "buffer",
        goal_random_prob: float = 0.5,
        goal_action_mode: str = "epsilon",
        **kwargs,
    ):
        super().__init__(d_obs=d_obs, n_actions=n_actions, **kwargs)
        if goal_buffer_size <= 0:
            raise ValueError(f"goal_buffer_size must be > 0, got {goal_buffer_size!r}")
        if goal_horizon <= 0:
            raise ValueError(f"goal_horizon must be > 0, got {goal_horizon!r}")
        if goal_warmup_solves < 0:
            raise ValueError(f"goal_warmup_solves must be >= 0, got {goal_warmup_solves!r}")
        if goal_mode not in ("buffer", "random_direction", "mixed"):
            raise ValueError(
                f"goal_mode must be one of 'buffer', 'random_direction', "
                f"'mixed'; got {goal_mode!r}"
            )
        if not (0.0 <= goal_random_prob <= 1.0):
            raise ValueError(f"goal_random_prob must be in [0,1], got {goal_random_prob!r}")
        if goal_action_mode not in ("epsilon", "commit"):
            raise ValueError(
                f"goal_action_mode must be 'epsilon' or 'commit'; got {goal_action_mode!r}"
            )

        self.goal_buffer_size = int(goal_buffer_size)
        self.goal_horizon = int(goal_horizon)
        self.goal_warmup_solves = int(goal_warmup_solves)
        self.goal_eps = float(goal_eps)
        self.goal_mode = goal_mode
        self.goal_random_prob = float(goal_random_prob)
        self.goal_action_mode = goal_action_mode

        self._goal_buffer: list[np.ndarray] = []
        self._goal_buffer_pos = 0
        self._current_goal: np.ndarray | None = None
        self._current_goal_phi: Tensor | None = None
        self._steps_on_goal = 0
        self._goal_resample_count = 0
        self._goal_random_count = 0
        self._goal_buffer_count = 0
        self._goal_action_count = 0

    def _push_goal_candidate(self, obs: np.ndarray) -> None:
        if len(self._goal_buffer) < self.goal_buffer_size:
            self._goal_buffer.append(np.asarray(obs, dtype=np.float32).copy())
        else:
            self._goal_buffer[self._goal_buffer_pos] = np.asarray(obs, dtype=np.float32).copy()
            self._goal_buffer_pos = (self._goal_buffer_pos + 1) % self.goal_buffer_size

    def _draw_random_direction(self) -> Tensor:
        """Sample a unit vector uniformly on the n-sphere in feature space."""
        n = self.successor.features.n_neurons
        v = torch.from_numpy(self.rng.standard_normal(n).astype(np.float64))
        norm = float(torch.linalg.vector_norm(v).item())
        if norm < self.goal_eps:
            v = torch.zeros(n, dtype=torch.float64)
            v[0] = 1.0
            return v
        return v / norm

    def _resample_goal(self) -> None:
        if self.goal_mode == "random_direction":
            use_random = True
        elif self.goal_mode == "mixed":
            use_random = (not self._goal_buffer) or (self.rng.random() < self.goal_random_prob)
        else:  # "buffer"
            use_random = False

        if use_random:
            self._current_goal = None
            self._current_goal_phi = self._draw_random_direction()
            self._steps_on_goal = 0
            self._goal_resample_count += 1
            self._goal_random_count += 1
            return

        if not self._goal_buffer:
            self._current_goal = None
            self._current_goal_phi = None
            return
        idx = int(self.rng.integers(len(self._goal_buffer)))
        g = self._goal_buffer[idx]
        with torch.no_grad():
            phi_g = self.successor.features.encode(torch.from_numpy(g).float()).double()
            norm = float(torch.linalg.vector_norm(phi_g).item())
        if norm < self.goal_eps:
            self._current_goal = None
            self._current_goal_phi = None
            return
        self._current_goal = g
        self._current_goal_phi = phi_g / norm
        self._steps_on_goal = 0
        self._goal_resample_count += 1
        self._goal_buffer_count += 1

    def goal_q_values(self, obs: np.ndarray) -> Tensor | None:
        """``Q(s, a; g) = ψ(s, a) · u(g)`` for the current goal.

        Returns ``None`` when no goal is active (no buffer, pre-warmup,
        or degenerate ``φ(g)``).
        """
        if self._current_goal_phi is None:
            return None
        with torch.no_grad():
            s = torch.from_numpy(obs).float()
            phi = self.successor.features.encode(s)  # (n,) float32
            psi = torch.einsum("n,anm->am", phi, self.Psi)  # (a, n) float32
            return psi @ self._current_goal_phi.to(psi.dtype)  # (a,)

    def select_action(self, obs: np.ndarray) -> int:
        """Action selection with hindsight goal conditioning during training.

        Two modes:

        - ``goal_action_mode="epsilon"`` (default): standard ε-greedy.
          On greedy steps after warmup, action is argmax of
          ``Q(s, a; g)`` (goal-conditioned).  ε-random steps are
          uniform random.  Faithful to plain ε-greedy semantics, but
          on tasks where ε remains high (e.g. early MountainCar) the
          goal-greedy fraction is too small to sustain a directed
          motor sequence.

        - ``goal_action_mode="commit"``: after warmup, *every*
          training step takes ``argmax_a Q(s, a; g)``.  ε is
          **ignored** entirely — goals rotate every ``goal_horizon``
          steps and exploration arises from rotating *directions* in
          feature space rather than from random actions.  This is the
          SF-native answer to the "ε-greedy random walks cannot reach
          the goal" failure mode: the agent always commits to *some*
          directed sequence, and the diversity of sampled goal
          directions is what keeps the policy non-degenerate.

        Evaluation always uses the extrinsic Q.
        """
        self._push_goal_candidate(obs)

        goals_available = (self.goal_mode != "buffer") or bool(self._goal_buffer)
        warm = self._solve_count >= self.goal_warmup_solves and goals_available

        if self.goal_action_mode == "commit" and warm:
            if self._current_goal_phi is None or self._steps_on_goal >= self.goal_horizon:
                self._resample_goal()
            self._steps_on_goal += 1

            if self._solve_count == 0 or self._current_goal_phi is None:
                return int(self.rng.integers(self.n_actions))
            gq = self.goal_q_values(obs)
            if gq is None:
                with torch.no_grad():
                    q = self.successor.q_values(obs)
                    return int(q.argmax().item())
            self._goal_action_count += 1
            return int(gq.argmax().item())

        if warm:
            if self._current_goal_phi is None or self._steps_on_goal >= self.goal_horizon:
                self._resample_goal()
            self._steps_on_goal += 1

        if self.rng.random() < self._epsilon() or self._solve_count == 0:
            return int(self.rng.integers(self.n_actions))

        gq = self.goal_q_values(obs) if warm else None
        if gq is None:
            with torch.no_grad():
                q = self.successor.q_values(obs)
                return int(q.argmax().item())
        self._goal_action_count += 1
        return int(gq.argmax().item())

    def goal_diagnostics(self) -> dict:
        return {
            "goal_buffer_size": len(self._goal_buffer),
            "goal_resample_count": int(self._goal_resample_count),
            "goal_random_count": int(self._goal_random_count),
            "goal_buffer_count": int(self._goal_buffer_count),
            "goal_action_count": int(self._goal_action_count),
            "current_goal_active": self._current_goal_phi is not None,
            "goal_mode": self.goal_mode,
            "goal_action_mode": self.goal_action_mode,
        }
