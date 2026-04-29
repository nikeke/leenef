"""Tests for RL utilities."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import (
    analyze_rl_sweep,
    rl_repro,
    run_config_sweep,
    run_ensemble_sweep,
    run_hp_sweep,
    run_sparse_rl_diagnostics,
)
from benchmarks.rl_repro import can_reuse_result, env_tag, hp_run_context
from benchmarks.rl_sweep_data import load_sweep
from leenef.rl import NEFFeatures, NEFFQIAgent, NEFFQIEnsemble
from leenef.rl_hindsight import (
    acrobot_end_effector_height,
    append_goal,
    goal_reached,
    hindsight_event_targets,
    mountaincar_achieved_goal,
    terminal_hindsight_goal,
)
from leenef.rl_novelty import leverage_scores, nef_fqi_action_novelty, novelty_scores
from leenef.rl_successor import NEFSuccessorFeatureAgent, truncated_successor_targets


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


def test_event_trace_returns_spread_terminal_event_over_window():
    agent = NEFFQIAgent(
        d_obs=2,
        n_actions=2,
        n_neurons=6,
        gamma=0.5,
        target_mode="event_trace",
        n_step=3,
        event_trace_value=1.0,
        event_reward_min=0.0,
        event_terminal_only=True,
        seed=0,
    )
    transitions = [
        (np.array([0.0, 0.0]), 0, -1.0, False),
        (np.array([0.1, 0.0]), 1, -1.0, False),
        (np.array([0.2, 0.0]), 1, -1.0, False),
        (np.array([0.3, 0.0]), 0, 0.0, True),
    ]

    returns = agent._compute_event_trace_returns(transitions)

    assert [round(t[2], 3) for t in returns] == [0.0, 0.25, 0.5, 1.0]


def test_event_trace_ignores_nonterminal_event_when_terminal_only():
    agent = NEFFQIAgent(
        d_obs=2,
        n_actions=2,
        n_neurons=6,
        gamma=0.5,
        target_mode="event_trace",
        n_step=2,
        event_trace_value=1.0,
        event_reward_min=0.0,
        event_terminal_only=True,
        seed=0,
    )
    transitions = [
        (np.array([0.0, 0.0]), 0, 0.0, False),
        (np.array([0.1, 0.0]), 1, -1.0, True),
    ]

    returns = agent._compute_event_trace_returns(transitions)

    assert [t[2] for t in returns] == [0.0, 0.0]


def test_event_trace_agent_accumulates_event_targets():
    agent = NEFFQIAgent(
        d_obs=2,
        n_actions=2,
        n_neurons=6,
        gamma=0.9,
        target_mode="event_trace",
        n_step=2,
        event_trace_value=1.0,
        event_reward_min=0.0,
        event_terminal_only=True,
        forget_factor=0.99,
        solve_every=1,
        seed=0,
    )
    episode = [
        (np.array([0.0, 0.0]), 0, -1.0, False),
        (np.array([0.2, 0.0]), 1, 0.0, True),
    ]

    for i, (obs, action, reward, done) in enumerate(episode):
        next_obs = episode[min(i + 1, len(episode) - 1)][0]
        agent.update(obs, action, reward, next_obs, done)
    agent.end_episode()

    assert int(agent._n_accumulated.sum().item()) == len(episode)
    assert torch.count_nonzero(agent._aty).item() > 0


class _OneStepEnv:
    def __init__(self):
        self.reset_seeds: list[int | None] = []
        self.action_space = _SeedableActionSpace(2)
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        self.reset_seeds.append(seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        del action
        self.steps += 1
        terminated = self.steps >= 10
        return np.ones(2, dtype=np.float32), 1.0, terminated, False, {}


class _SeedableActionSpace:
    def __init__(self, n):
        self.n = n
        self.seed_values: list[int] = []

    def seed(self, seed):
        self.seed_values.append(seed)


class _TinyObsSpace:
    shape = (2,)


class _TinyEnv:
    def __init__(self):
        self.observation_space = _TinyObsSpace()
        self.action_space = _SeedableActionSpace(2)
        self.reset_seeds: list[int | None] = []
        self._episode_seed = 0
        self._step = 0

    def reset(self, seed=None):
        self._episode_seed = 0 if seed is None else int(seed)
        self._step = 0
        self.reset_seeds.append(seed)
        obs = np.array(
            [self._episode_seed % 7, (self._episode_seed // 7) % 7],
            dtype=np.float32,
        )
        return obs, {}

    def step(self, action):
        self._step += 1
        next_obs = np.array(
            [self._episode_seed % 5 + self._step, action + self._step],
            dtype=np.float32,
        )
        reward = float((self._episode_seed % 11) - action + self._step)
        terminated = self._step >= 2
        return next_obs, reward, terminated, False, {}

    def close(self):
        return None


def _tiny_hp_config(**overrides):
    config = {
        "n_neurons": 8,
        "gamma": 0.9,
        "alpha": 1e-2,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay": 1,
        "solve_every": 1,
        "target_mode": "nstep",
        "n_step": 1,
        "forget_factor": 0.95,
        "fqi_iters": 1,
        "tau": 0.3,
        "broadness": 0.0,
        "recenter_interval": None,
    }
    config.update(overrides)
    return config


def _stub_hp_result(label: str, seed: int, agent_kwargs: dict) -> dict:
    config = {k: v for k, v in agent_kwargs.items() if not k.startswith("_")}
    run_context = {
        "env_name": "LunarLander-v3",
        "n_episodes": 1,
        "eval_every": 50,
        "eval_episodes": 20,
    }
    return {
        **run_hp_sweep.result_metadata(config, seed, run_context=run_context),
        "label": label,
        "seed": seed,
        "knob": agent_kwargs.get("_knob"),
        "value": agent_kwargs.get("_value"),
        "best_50": 1.0,
        "final_50": 1.0,
        "best_eval": 1.0,
        "final_eval": 1.0,
        "robust_eval": 1.0,
        "time_s": 0.0,
        "warmup_seed": run_hp_sweep.warmup_seed(seed),
        "evaluation_seed_base": run_hp_sweep.EVAL_SEED_BASE,
        "eval_history": [{"episode": 1, "eval": 1.0}],
        "rewards": [],
    }


def test_warmup_uses_seed_sequence_when_provided():
    agent = NEFFQIAgent(
        d_obs=2,
        n_actions=2,
        n_neurons=6,
        target_mode="nstep",
        forget_factor=0.95,
        seed=2,
    )
    env = _OneStepEnv()

    agent.warmup(env=env, n_episodes=3, seed=100)

    assert env.reset_seeds == [100, 101, 102]
    assert env.action_space.seed_values == [100, 101, 102]


def test_ensemble_warmup_uses_seed_sequence_when_provided():
    env = _OneStepEnv()
    ensemble = NEFFQIEnsemble(
        n_members=2,
        d_obs=2,
        n_actions=2,
        n_neurons=6,
        target_mode="nstep",
        forget_factor=0.95,
        base_seed=3,
    )

    ensemble.warmup(env=env, n_episodes=3, seed=200)

    assert env.reset_seeds == [200, 201, 202]
    assert env.action_space.seed_values == [200, 201, 202]


def test_run_hp_sweep_is_deterministic_for_same_config_and_seed(monkeypatch):
    monkeypatch.setattr(run_hp_sweep.gym, "make", lambda _: _TinyEnv())
    kwargs = _tiny_hp_config(_knob="n_step", _value=1)

    first = run_hp_sweep.run_experiment(
        "n_step=1_s7",
        "TinyEnv-v0",
        n_episodes=4,
        seed=7,
        agent_kwargs=kwargs,
        eval_every=2,
    )
    second = run_hp_sweep.run_experiment(
        "n_step=1_s7",
        "TinyEnv-v0",
        n_episodes=4,
        seed=7,
        agent_kwargs=kwargs,
        eval_every=2,
    )

    for key in (
        "rewards",
        "eval_history",
        "robust_eval",
        "best_eval",
        "final_eval",
        "best_50",
        "final_50",
    ):
        assert second[key] == first[key]


def test_run_hp_sweep_metrics_ignore_knob_label_for_same_config(monkeypatch):
    monkeypatch.setattr(run_hp_sweep.gym, "make", lambda _: _TinyEnv())
    config = _tiny_hp_config()

    first = run_hp_sweep.run_experiment(
        "alpha=0.01_s4",
        "TinyEnv-v0",
        n_episodes=4,
        seed=4,
        agent_kwargs={**config, "_knob": "alpha", "_value": 0.01},
        eval_every=2,
    )
    second = run_hp_sweep.run_experiment(
        "tau=0.3_s4",
        "TinyEnv-v0",
        n_episodes=4,
        seed=4,
        agent_kwargs={**config, "_knob": "tau", "_value": 0.3},
        eval_every=2,
    )

    for key in (
        "rewards",
        "eval_history",
        "robust_eval",
        "best_eval",
        "final_eval",
        "best_50",
        "final_50",
    ):
        assert second[key] == first[key]


def test_run_hp_sweep_records_run_context_extra(monkeypatch):
    monkeypatch.setattr(run_hp_sweep.gym, "make", lambda _: _TinyEnv())

    result = run_hp_sweep.run_experiment(
        "h1_pilot_n50_k4_s0",
        "TinyEnv-v0",
        n_episodes=4,
        seed=0,
        agent_kwargs={
            **_tiny_hp_config(),
            "_knob": "h1_pilot",
            "_value": "n50_k4",
            "_run_context_extra": {"sweep_type": "config_grid", "phase": "h1_pilot"},
        },
        eval_every=2,
    )

    assert result["run_context"]["sweep_type"] == "config_grid"
    assert result["run_context"]["phase"] == "h1_pilot"
    assert "_run_context_extra" not in result["config"]


def test_run_hp_sweep_reruns_stale_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_hp_sweep, "_GYMNASIUM_IMPORT_ERROR", None)
    base_path = tmp_path / "base_config.json"
    out_path = tmp_path / "n_step.json"
    config = _tiny_hp_config()
    base_path.write_text(json.dumps(config))
    label = "n_step=1_s0"
    out_path.write_text(json.dumps({label: {"label": label, "seed": 0, "config": config}}))

    seen: list[str] = []

    def fake_worker(args):
        label, _, _, seed, agent_kwargs = args
        seen.append(label)
        return _stub_hp_result(label, seed, agent_kwargs)

    monkeypatch.setattr(run_hp_sweep, "_worker", fake_worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hp_sweep.py",
            "--knob",
            "n_step",
            "--values",
            "1",
            "--seeds",
            "0",
            "--episodes",
            "1",
            "--base-config",
            str(base_path),
            "--output",
            str(out_path),
        ],
    )

    run_hp_sweep.main()

    assert seen == [label]
    assert "Re-running n_step=1_s0" in capsys.readouterr().out


def test_run_hp_sweep_skips_current_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_hp_sweep, "_GYMNASIUM_IMPORT_ERROR", None)
    base_path = tmp_path / "base_config.json"
    out_path = tmp_path / "n_step.json"
    config = _tiny_hp_config()
    base_path.write_text(json.dumps(config))
    label = "n_step=1_s0"
    out_path.write_text(
        json.dumps({label: _stub_hp_result(label, 0, {**config, "_knob": "n_step", "_value": 1})})
    )

    seen: list[str] = []
    monkeypatch.setattr(run_hp_sweep, "_worker", lambda args: seen.append(args[0]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hp_sweep.py",
            "--knob",
            "n_step",
            "--values",
            "1",
            "--seeds",
            "0",
            "--episodes",
            "1",
            "--base-config",
            str(base_path),
            "--output",
            str(out_path),
        ],
    )

    run_hp_sweep.main()

    assert seen == []
    assert "Skipping n_step=1_s0" in capsys.readouterr().out


def test_run_hp_sweep_reruns_when_run_context_changes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_hp_sweep, "_GYMNASIUM_IMPORT_ERROR", None)
    base_path = tmp_path / "base_config.json"
    out_path = tmp_path / "n_step.json"
    config = _tiny_hp_config()
    base_path.write_text(json.dumps(config))
    label = "n_step=1_s0"
    stored = _stub_hp_result(label, 0, {**config, "_knob": "n_step", "_value": 1})
    stored["run_context"]["n_episodes"] = 2
    out_path.write_text(json.dumps({label: stored}))

    seen: list[str] = []

    def fake_worker(args):
        label, _, _, seed, agent_kwargs = args
        seen.append(label)
        return _stub_hp_result(label, seed, agent_kwargs)

    monkeypatch.setattr(run_hp_sweep, "_worker", fake_worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hp_sweep.py",
            "--knob",
            "n_step",
            "--values",
            "1",
            "--seeds",
            "0",
            "--episodes",
            "1",
            "--base-config",
            str(base_path),
            "--output",
            str(out_path),
        ],
    )

    run_hp_sweep.main()

    assert seen == [label]
    assert "Re-running n_step=1_s0" in capsys.readouterr().out


def test_run_hp_sweep_fails_fast_without_gymnasium(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_hp_sweep,
        "_GYMNASIUM_IMPORT_ERROR",
        ModuleNotFoundError(run_hp_sweep._MISSING_GYMNASIUM_MESSAGE),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hp_sweep.py",
            "--knob",
            "n_step",
            "--values",
            "1",
            "--seeds",
            "0",
            "--episodes",
            "1",
            "--base-config",
            str(tmp_path / "base_config.json"),
            "--output",
            str(tmp_path / "n_step.json"),
        ],
    )

    with pytest.raises(SystemExit, match="gymnasium is required to run RL sweeps"):
        run_hp_sweep.main()


def test_hp_analysis_and_plot_filter_stale_rows(tmp_path):
    knob = "n_step"
    base_config = _tiny_hp_config()
    expected_run_context = hp_run_context("LunarLander-v3", 1)
    current = _stub_hp_result("n_step=1_s0", 0, {**base_config, "_knob": knob, "_value": 1})
    stale = _stub_hp_result("n_step=1_s1", 1, {**base_config, "_knob": knob, "_value": 1})
    stale["config"]["gamma"] = 0.8
    sweep = {current["label"]: current, stale["label"]: stale}

    stats = analyze_rl_sweep.aggregate(
        sweep,
        knob,
        base_config=base_config,
        expected_run_context=expected_run_context,
    )

    assert stats["1"]["n"] == 1
    sweep_path = tmp_path / "n_step_lunarlander.json"
    sweep_path.write_text(json.dumps(sweep))
    plotted = load_sweep(
        str(sweep_path),
        knob,
        base_config=base_config,
        expected_run_context=expected_run_context,
    )
    series = next(iter(plotted.values()))
    assert series["robust"] == [1.0]
    assert series["best"] == [1.0]


def test_can_reuse_result_requires_matching_code_version():
    config = _tiny_hp_config()
    result = _stub_hp_result("n_step=1_s0", 0, {**config, "_knob": "n_step", "_value": 1})

    assert can_reuse_result(
        result,
        config,
        expected_seed=0,
        expected_run_context=result["run_context"],
    )

    result["code_version"] = "stale-version"
    assert not can_reuse_result(
        result,
        config,
        expected_seed=0,
        expected_run_context=result["run_context"],
    )


def test_code_version_ignores_generated_rl_clean_outputs(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(cmd)
        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[:3] == ["git", "diff", "--no-ext-diff"]:
            assert "results/rl_clean" not in cmd
            assert "docs/figures" not in cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["git", "ls-files", "--others"]:
            assert "results/rl_clean" not in cmd
            assert "docs/figures" not in cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(cmd)

    rl_repro.code_version.cache_clear()
    monkeypatch.setattr(rl_repro.subprocess, "run", fake_run)

    assert rl_repro.code_version() == "abc123"
    assert any(cmd[:3] == ["git", "diff", "--no-ext-diff"] for cmd in calls)
    rl_repro.code_version.cache_clear()


def test_env_tag_preserves_environment_version():
    assert env_tag("LunarLander-v3") == "lunarlander_v3"
    assert run_hp_sweep._default_output_path("LunarLander-v3", "n_step").endswith(
        "n_step_lunarlander_v3.json"
    )


def test_config_sweep_reuse_requires_grid_run_context():
    config = _tiny_hp_config(n_step=2, n_neurons=16)
    context = run_config_sweep.config_grid_run_context("LunarLander-v3", 1000, phase="h1_pilot")
    result = {
        **run_hp_sweep.result_metadata(config, 0, run_context=context),
        "label": "h1_pilot_n2_k16_s0",
        "seed": 0,
        "knob": "h1_pilot",
        "value": "n2_k16",
        "config_name": "n2_k16",
        "robust_eval": 1.0,
    }

    assert can_reuse_result(result, config, expected_seed=0, expected_run_context=context)
    hp_context = hp_run_context("LunarLander-v3", 1000)
    assert not can_reuse_result(result, config, expected_seed=0, expected_run_context=hp_context)


def test_ensemble_config_records_member_metadata_without_agent_keys():
    base = _tiny_hp_config()

    config = run_ensemble_sweep.ensemble_config(
        base,
        n_members=3,
        n_neurons=16,
        train_policy="thompson",
        eval_policies=["mean", "voting"],
    )

    assert config["n_neurons"] == 16
    assert config["n_members"] == 3
    assert config["train_policy"] == "thompson"
    assert config["eval_policies"] == ["mean", "voting"]
    assert "n_members" not in run_ensemble_sweep.agent_config(config)
    assert "train_policy" not in run_ensemble_sweep.agent_config(config)
    assert "eval_policies" not in run_ensemble_sweep.agent_config(config)


def test_ensemble_run_context_is_distinct_from_hp_context():
    context = run_ensemble_sweep.ensemble_run_context(
        "LunarLander-v3",
        1000,
        phase="p8_ensemble_pilot",
    )

    assert context["sweep_type"] == "ensemble"
    assert context["phase"] == "p8_ensemble_pilot"
    assert context != hp_run_context("LunarLander-v3", 1000)


def test_sparse_success_event_detects_classic_control_goals():
    assert run_sparse_rl_diagnostics.sparse_success_event(
        "MountainCar-v0",
        np.array([0.51, 0.0], dtype=np.float32),
        -1.0,
        terminated=True,
        truncated=False,
    )
    assert not run_sparse_rl_diagnostics.sparse_success_event(
        "MountainCar-v0",
        np.array([0.49, 0.0], dtype=np.float32),
        -1.0,
        terminated=True,
        truncated=False,
    )
    assert run_sparse_rl_diagnostics.sparse_success_event(
        "Acrobot-v1",
        np.zeros(6, dtype=np.float32),
        0.0,
        terminated=True,
        truncated=False,
    )
    assert not run_sparse_rl_diagnostics.sparse_success_event(
        "Acrobot-v1",
        np.zeros(6, dtype=np.float32),
        -1.0,
        terminated=False,
        truncated=True,
    )


def test_sparse_episode_diagnostics_report_target_diversity_and_success():
    transitions = [
        {
            "action": 0,
            "reward": -1.0,
            "terminated": False,
            "truncated": False,
            "success_event": False,
        },
        {
            "action": 1,
            "reward": -1.0,
            "terminated": False,
            "truncated": False,
            "success_event": False,
        },
        {
            "action": 1,
            "reward": 0.0,
            "terminated": True,
            "truncated": False,
            "success_event": True,
        },
    ]

    diag = run_sparse_rl_diagnostics.diagnose_episode(
        transitions,
        env_name="MountainCar-v0",
        episode=1,
        gamma=0.9,
        target_mode="nstep",
        n_step=3,
        n_actions=3,
    )

    assert diag["success"]
    assert diag["first_success_step"] == 2
    assert diag["action_counts"] == [1, 2, 0]
    assert diag["target_stats"]["unique_rounded"] == 3
    assert "1_5" in diag["target_by_distance_to_event"]


def test_sparse_diagnostic_run_is_deterministic_for_same_seed(monkeypatch):
    monkeypatch.setattr(run_sparse_rl_diagnostics.gym, "make", lambda _: _TinyEnv())
    config = _tiny_hp_config(n_neurons=8, n_step=1, solve_every=1)

    first = run_sparse_rl_diagnostics.run_diagnostic_experiment(
        "tiny_s3",
        "TinyEnv-v0",
        n_episodes=3,
        seed=3,
        agent_kwargs=config,
        eval_every=1,
        eval_episodes=2,
        warmup_episodes=1,
    )
    second = run_sparse_rl_diagnostics.run_diagnostic_experiment(
        "tiny_s3",
        "TinyEnv-v0",
        n_episodes=3,
        seed=3,
        agent_kwargs=config,
        eval_every=1,
        eval_episodes=2,
        warmup_episodes=1,
    )

    assert second["diagnostic_summary"] == first["diagnostic_summary"]
    assert second["eval_history"] == first["eval_history"]
    assert second["episode_diagnostics"] == first["episode_diagnostics"]


def test_truncated_successor_targets_use_finite_horizon():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )

    targets = truncated_successor_targets(features, gamma=0.5, horizon=2)

    expected = torch.tensor(
        [
            [1.0, 0.5],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    torch.testing.assert_close(targets, expected)


def test_successor_feature_agent_accumulates_and_solves():
    agent = NEFSuccessorFeatureAgent(
        d_obs=2,
        n_actions=2,
        n_neurons=8,
        gamma=0.9,
        alpha=1e-2,
        solve_every=1,
        successor_horizon=3,
        seed=0,
    )
    episode = [
        (np.array([0.0, 0.0], dtype=np.float32), 0, -1.0, False),
        (np.array([0.2, 0.0], dtype=np.float32), 1, -1.0, False),
        (np.array([0.4, 0.1], dtype=np.float32), 1, 0.0, True),
    ]

    for i, (obs, action, reward, done) in enumerate(episode):
        next_obs = episode[min(i + 1, len(episode) - 1)][0]
        agent.update(obs, action, reward, next_obs, done)
    agent.end_episode()

    assert agent._n_accumulated == len(episode)
    assert agent.successor_decoder.shape == (8, 8)
    assert agent.reward_weight.shape == (8,)
    q = agent.q_values(np.array([0.1, 0.0], dtype=np.float32))
    assert q.shape == (2,)
    assert torch.isfinite(q).all()


def test_successor_feature_agent_is_deterministic_on_tiny_env():
    def run_once():
        env = _TinyEnv()
        agent = NEFSuccessorFeatureAgent(
            d_obs=2,
            n_actions=2,
            n_neurons=8,
            gamma=0.9,
            alpha=1e-2,
            epsilon_decay=1,
            solve_every=1,
            successor_horizon=2,
            seed=5,
        )
        agent.warmup(env=env, n_episodes=1, seed=100)
        rewards = []
        for ep in range(3):
            obs, _ = env.reset(seed=ep)
            done = False
            total = 0.0
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total += reward
            agent.end_episode()
            rewards.append(total)
        return rewards, agent.q_values(np.array([1.0, 0.0], dtype=np.float32))

    rewards_a, q_a = run_once()
    rewards_b, q_b = run_once()

    assert rewards_b == rewards_a
    torch.testing.assert_close(q_b, q_a)


def test_leverage_novelty_decreases_with_feature_coverage():
    features = torch.tensor([[1.0, 0.0]])
    sparse_ata = torch.zeros(2, 2)
    covered_ata = torch.diag(torch.tensor([10.0, 0.0]))

    sparse = leverage_scores(features, sparse_ata, alpha=0.1)
    covered = leverage_scores(features, covered_ata, alpha=0.1)

    assert covered.item() < sparse.item()
    torch.testing.assert_close(novelty_scores(features, sparse_ata, 0.1), sparse.sqrt())


def test_nef_fqi_action_novelty_uses_per_action_statistics():
    agent = NEFFQIAgent(
        d_obs=2,
        n_actions=2,
        n_neurons=6,
        target_mode="nstep",
        forget_factor=0.99,
        seed=0,
    )
    obs = np.array([0.5, -0.25], dtype=np.float32)
    phi = agent.features.encode(torch.from_numpy(obs).float()).double()
    agent._ata[0].add_(20.0 * torch.outer(phi, phi))

    novelty = nef_fqi_action_novelty(agent, obs)

    assert novelty.shape == (2,)
    assert novelty[0] < novelty[1]


def test_successor_agent_novelty_bonus_changes_greedy_selection():
    agent = NEFSuccessorFeatureAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=2,
        epsilon_start=0.0,
        epsilon_end=0.0,
        novelty_bonus=1.0,
        seed=1,
    )
    obs = np.array([0.0], dtype=np.float32)
    agent._candidate_features = lambda _: torch.eye(2)
    agent._ata.add_(torch.diag(torch.tensor([20.0, 0.0], dtype=torch.float64)))

    novelty = agent.action_novelty(obs)

    assert novelty[1] > novelty[0]
    assert agent.select_action(obs) == 1


def test_hindsight_goal_helpers_for_classic_control():
    mountain_obs = np.array([-0.2, 0.03], dtype=np.float32)
    assert np.array_equal(
        mountaincar_achieved_goal(mountain_obs), np.array([-0.2], dtype=np.float32)
    )
    augmented = append_goal(mountain_obs, np.array([0.5], dtype=np.float32))
    np.testing.assert_allclose(augmented, np.array([-0.2, 0.03, 0.5], dtype=np.float32))

    upright_acrobot = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(acrobot_end_effector_height(upright_acrobot), np.array([-0.0]))

    assert goal_reached(np.array([0.49]), np.array([0.5]), tolerance=0.02)
    assert not goal_reached(np.array([0.45]), np.array([0.5]), tolerance=0.02)


def test_hindsight_event_targets_use_terminal_goal():
    achieved = [
        np.array([0.0], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
        np.array([0.4], dtype=np.float32),
    ]
    goal = terminal_hindsight_goal(achieved)

    targets = hindsight_event_targets(
        achieved,
        goal,
        gamma=0.5,
        horizon=2,
        tolerance=1e-6,
        success_value=1.0,
    )

    np.testing.assert_allclose(goal, np.array([0.4], dtype=np.float32))
    assert targets == [0.0, 0.5, 1.0]
