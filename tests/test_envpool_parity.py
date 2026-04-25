"""Parity tests for the envpool-batched HP sweep runner.

The full statistical parity check (4 values × 8 seeds, ~2-3 h on CPU) is
tracked in ``docs/envpool_port_handoff.md``'s addendum.  This file holds
the cheap structural checks that can run in CI: schema parity, ability
to instantiate an envpool batch, and a tiny end-to-end batch (2 agents
× 5 episodes) against ``LunarLander-v3``.
"""

from __future__ import annotations

import json
import os
import sys

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


def test_envpool_existing_results_schema_parity(tmp_path):
    """If a real gym result file is present, ensure schema parity."""
    gym_path = os.path.join(_REPO_ROOT, "results", "rl", "n_step_lunarlander.json")
    if not os.path.exists(gym_path):
        pytest.skip("no reference gym result file")
    with open(gym_path) as f:
        gym = json.load(f)
    if not gym:
        pytest.skip("empty reference file")
    ref_keys = set(next(iter(gym.values())).keys())
    assert ref_keys == _gym_path_keys()


@pytest.mark.slow
def test_envpool_batch_runs_end_to_end(tmp_path):
    """Tiny batched run to catch wiring/import regressions (~30s).

    Marked ``slow`` because it instantiates two NEFFQIAgents with the
    default 4000 neurons.  Run with ``pytest -m slow``.
    """
    from benchmarks.run_hp_sweep import DEFAULT_BASE
    from benchmarks.run_hp_sweep_envpool import run_envpool_batch

    base = dict(DEFAULT_BASE)
    base["n_neurons"] = 500  # keep it small so this test is cheap
    base["_knob"] = "smoke"
    base["_value"] = None

    results = run_envpool_batch(
        env_name="LunarLander-v3",
        n_episodes=3,
        labels=["smoke_s0", "smoke_s1"],
        seeds=[0, 1],
        agent_kwargs_list=[dict(base), dict(base)],
        eval_every=1000,  # too high to trigger eval — keeps the test fast
    )
    assert len(results) == 2
    for r in results:
        assert set(r.keys()) == _gym_path_keys()
        assert len(r["rewards"]) == 3
