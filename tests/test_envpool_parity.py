"""Parity tests for the envpool-batched HP sweep runner.

The full statistical parity check (4 values × 8 seeds, ~hours on CPU)
is tracked in ``docs/envpool_port_handoff.md``'s addendum.  This file
holds the cheap structural checks that can run in CI plus a focused
test of the batch-invariant seeding model that was the main fix of
the brutal review.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# Make the worktree root importable independently of how pytest was invoked.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

envpool = pytest.importorskip("envpool")


def _gym_path_keys() -> set:
    """Reference set of keys produced by the gym path's ``run_experiment``."""
    return {
        "label",
        "seed",
        "knob",
        "value",
        "best_50",
        "final_50",
        "best_eval",
        "final_eval",
        "time_s",
        "config",
        "eval_history",
        "rewards",
    }


def test_envpool_run_dict_has_full_schema():
    from benchmarks.run_hp_sweep_envpool import _build_run_dict

    d = _build_run_dict(
        label="x=1_s0",
        seed=0,
        knob="x",
        value=1,
        rewards=[0.0],
        eval_history=[],
        elapsed=0.0,
        config={},
    )
    assert set(d.keys()) == _gym_path_keys()


def test_envpool_existing_results_schema_parity():
    """If a real gym result file is present, ensure schema parity."""
    gym_path = os.path.join(_REPO_ROOT, "results", "rl", "n_step_lunarlander.json")
    if not os.path.exists(gym_path):
        pytest.skip("no reference gym result file")
    with open(gym_path) as f:
        gym_data = json.load(f)
    if not gym_data:
        pytest.skip("empty reference file")
    ref_keys = set(next(iter(gym_data.values())).keys())
    assert ref_keys == _gym_path_keys()


def test_envpool_seed_stream_independent_of_pool_size():
    """Sub-env idx ``s`` at ``master_seed=M`` must produce the same trajectory
    regardless of ``num_envs``.  Pinning label ``s`` to sub-env ``s`` and
    using a constant master seed only gives label-stable training if this
    holds — so we guard it explicitly.
    """
    import envpool as ep

    def traj(num_envs, idx, n_steps=15):
        p = ep.make(
            "LunarLander-v3",
            env_type="gymnasium",
            num_envs=num_envs,
            seed=0,
            max_episode_steps=1000,
        )
        obs, _ = p.reset()
        seq = [obs[idx].copy()]
        rews = []
        for t in range(n_steps):
            actions = np.zeros(num_envs, dtype=np.int32)
            actions[idx] = t % 4
            obs, r, _term, _trunc, _info = p.step(actions)
            seq.append(obs[idx].copy())
            rews.append(float(r[idx]))
        p.close()
        return np.stack(seq), np.array(rews)

    o4, r4 = traj(4, 2)
    o8, r8 = traj(8, 2)
    assert np.allclose(o4, o8), "sub-env trajectory drifts with num_envs"
    assert np.allclose(r4, r8), "sub-env rewards drift with num_envs"


@pytest.mark.slow
def test_envpool_batch_runs_end_to_end():
    """Tiny batched run to catch wiring/import regressions."""
    from benchmarks.run_hp_sweep import DEFAULT_BASE
    from benchmarks.run_hp_sweep_envpool import run_envpool_batch

    base = dict(DEFAULT_BASE)
    base["n_neurons"] = 500
    base["_knob"] = "smoke"
    base["_value"] = None

    results = list(
        run_envpool_batch(
            env_name="LunarLander-v3",
            n_episodes=3,
            labels=["smoke_s0", "smoke_s1"],
            seeds=[0, 1],
            agent_kwargs_list=[dict(base), dict(base)],
            eval_every=1000,  # too high to trigger eval — keeps the test fast
        )
    )
    assert len(results) == 2
    assert {r["label"] for r in results} == {"smoke_s0", "smoke_s1"}
    for r in results:
        assert set(r.keys()) == _gym_path_keys()
        assert len(r["rewards"]) == 3


@pytest.mark.slow
def test_envpool_label_invariant_across_batch_composition():
    """Label ``s=2``'s reward sequence must be identical whether it runs
    alone or alongside other labels — the property the brutal review
    flagged as broken in the v1 implementation.
    """
    from benchmarks.run_hp_sweep import DEFAULT_BASE
    from benchmarks.run_hp_sweep_envpool import run_envpool_batch

    base = dict(DEFAULT_BASE)
    base["n_neurons"] = 300
    base["_knob"] = "rep"
    base["_value"] = None

    # Run label s=2 alone (sub-env idx 2 of pool of size 3).
    [solo] = list(
        run_envpool_batch(
            env_name="LunarLander-v3",
            n_episodes=2,
            labels=["rep_s2"],
            seeds=[2],
            agent_kwargs_list=[dict(base)],
            eval_every=1000,
        )
    )

    # Run s=2 in a batch with s=0 and s=1.
    multi = list(
        run_envpool_batch(
            env_name="LunarLander-v3",
            n_episodes=2,
            labels=["rep_s0", "rep_s1", "rep_s2"],
            seeds=[0, 1, 2],
            agent_kwargs_list=[dict(base), dict(base), dict(base)],
            eval_every=1000,
        )
    )
    s2 = next(r for r in multi if r["label"] == "rep_s2")

    assert solo["rewards"] == s2["rewards"], (
        "label s=2 trajectory depends on batch composition: "
        f"solo={solo['rewards']} vs in-batch={s2['rewards']}"
    )


@pytest.mark.slow
def test_envpool_eval_cadence_populates_history():
    """Sanity-check that ``eval_history`` actually gets populated when
    ``eval_every`` divides the episode count.
    """
    from benchmarks.run_hp_sweep import DEFAULT_BASE
    from benchmarks.run_hp_sweep_envpool import run_envpool_batch

    base = dict(DEFAULT_BASE)
    base["n_neurons"] = 300
    base["_knob"] = "evalcad"
    base["_value"] = None

    [r] = list(
        run_envpool_batch(
            env_name="LunarLander-v3",
            n_episodes=2,
            labels=["evalcad_s0"],
            seeds=[0],
            agent_kwargs_list=[dict(base)],
            eval_every=1,
        )
    )
    assert len(r["eval_history"]) == 2, (
        f"expected 2 eval entries (one per episode), got {r['eval_history']}"
    )
    assert [e["episode"] for e in r["eval_history"]] == [1, 2]
    assert r["best_eval"] is not None
    assert r["final_eval"] == r["eval_history"][-1]["eval"]


@pytest.mark.slow
def test_envpool_partial_completion_yields_eagerly():
    """``run_envpool_batch`` must yield each agent's run dict the moment
    that agent finishes, so a later crash cannot lose completed work.
    """
    from benchmarks.run_hp_sweep import DEFAULT_BASE
    from benchmarks.run_hp_sweep_envpool import run_envpool_batch

    base = dict(DEFAULT_BASE)
    base["n_neurons"] = 300
    base["_knob"] = "partial"
    base["_value"] = None

    saved: list[str] = []
    n_yielded = 0
    for r in run_envpool_batch(
        env_name="LunarLander-v3",
        n_episodes=2,
        labels=["partial_s0", "partial_s1"],
        seeds=[0, 1],
        agent_kwargs_list=[dict(base), dict(base)],
        eval_every=1000,
        on_complete=lambda r: saved.append(r["label"]),
    ):
        n_yielded += 1
        assert r["label"] in saved, "on_complete must fire before/with yield"

    assert n_yielded == 2
    assert set(saved) == {"partial_s0", "partial_s1"}
