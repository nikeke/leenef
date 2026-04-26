"""Tests for RL utilities."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import analyze_rl_sweep, rl_repro, run_config_sweep, run_hp_sweep
from benchmarks.rl_repro import can_reuse_result, env_tag, hp_run_context
from benchmarks.rl_sweep_data import load_sweep
from leenef.rl import NEFFeatures, NEFFQIAgent, NEFFQIEnsemble


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
