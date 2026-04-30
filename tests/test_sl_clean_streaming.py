"""Sanity tests for the B2 streaming sMNIST experiments + analyzer wiring.

Lightweight: ``n_neurons=200``, ``window_size=4``, batch solve.  The
suite validates wiring, RNG isolation, and end-to-end run on a tiny
config.  It does not validate H3/H4 — those are the scientific claims
that the full sweep produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmarks.sl_clean._streaming_exp import (
    STREAMING_PERMUTED_BASE_CONFIG,
    STREAMING_ROW_BASE_CONFIG,
    STREAMING_SOLVE_MODES,
    streaming_make_rng,
)
from benchmarks.sl_clean.experiments import get
from benchmarks.sl_clean.sl_phase_runner import run_phase

DATA_PRESENT = (
    Path(__file__).resolve().parents[1] / "data" / "MNIST" / "raw" / "train-images-idx3-ubyte"
).exists()


def test_streaming_experiments_registered():
    for name in (
        "streaming_smnist_row",
        "streaming_smnist_permuted",
        "streaming_solve_modes",
    ):
        spec = get(name)
        assert spec.primary_metric == "test_accuracy"
        assert spec.higher_is_better is True


def test_streaming_solve_modes_complete():
    assert set(STREAMING_SOLVE_MODES) == {"batch_acc", "accumulate", "woodbury"}


def test_streaming_make_rng_deterministic_per_seed():
    a = torch.randn(100, generator=streaming_make_rng(0))
    b = torch.randn(100, generator=streaming_make_rng(0))
    c = torch.randn(100, generator=streaming_make_rng(1))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_streaming_base_configs_have_required_keys():
    required = {
        "n_neurons",
        "window_size",
        "encoder_strategy",
        "activation",
        "gain_low",
        "gain_high",
        "alpha",
        "solve_mode",
        "batch_size",
        "data_driven_biases",
    }
    assert required <= set(STREAMING_ROW_BASE_CONFIG)
    assert required <= set(STREAMING_PERMUTED_BASE_CONFIG)


@pytest.mark.skipif(not DATA_PRESENT, reason="MNIST raw IDX files not available")
def test_streaming_row_smoketest():
    """End-to-end on the row variant at tiny sizes for all three solve modes."""
    spec = get("streaming_smnist_row")
    base = {**STREAMING_ROW_BASE_CONFIG, "n_neurons": 100, "window_size": 3, "batch_size": 5000}
    accs: dict[str, float] = {}
    for mode in STREAMING_SOLVE_MODES:
        config = {**base, "solve_mode": mode}
        out = spec.run_fn(config, seed=0, run_context=spec.default_run_context())
        accs[mode] = out["test_accuracy"]
        assert 0.5 < out["test_accuracy"] < 1.0
        assert out["factors"]["solve_mode"] == mode
        assert out["factors"]["dataset"] == "sMNIST-row"

    # batch_acc and accumulate share the same fixed-alpha solve and the
    # same pooled activities — they should land within tight numerical
    # precision of each other.
    assert abs(accs["batch_acc"] - accs["accumulate"]) < 5e-3


@pytest.mark.skipif(not DATA_PRESENT, reason="MNIST raw IDX files not available")
def test_streaming_permuted_smoketest():
    spec = get("streaming_smnist_permuted")
    config = {**STREAMING_PERMUTED_BASE_CONFIG, "n_neurons": 100, "window_size": 3}
    out = spec.run_fn(config, seed=0, run_context=spec.default_run_context())
    assert 0.2 < out["test_accuracy"] < 1.0
    assert out["factors"]["dataset"] == "sMNIST-permuted"
    assert out["dataset_metadata"]["sequence_shape"] == [784, 1]


@pytest.mark.skipif(not DATA_PRESENT, reason="MNIST raw IDX files not available")
def test_streaming_runs_through_phase_runner(tmp_path: Path):
    """Drive the per-cell runner with two n_neurons values × one seed."""
    out_path = tmp_path / "streaming_row_smoke.json"
    base_config = {
        **STREAMING_ROW_BASE_CONFIG,
        "n_neurons": 100,
        "window_size": 3,
    }
    spec = get("streaming_smnist_row")
    results = run_phase(
        experiment="streaming_smnist_row",
        phase="t_smoke",
        knob="n_neurons",
        values=[100, 150],
        seeds=[0],
        base_config=base_config,
        out_path=out_path,
        run_context=spec.default_run_context(),
    )
    assert len(results) == 2
    for row in results.values():
        assert "error" not in row
        assert row["metrics"]["test_accuracy"] > 0.5

    # Re-run: should be cache-hit on both cells (idempotent).
    results2 = run_phase(
        experiment="streaming_smnist_row",
        phase="t_smoke",
        knob="n_neurons",
        values=[100, 150],
        seeds=[0],
        base_config=base_config,
        out_path=out_path,
        run_context=spec.default_run_context(),
    )
    assert results2 == results
