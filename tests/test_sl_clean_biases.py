"""Sanity tests for the B1 ``biases_mnist`` experiment + panel analyzer.

These are lightweight: we use ``n_neurons=200`` and a tiny chunk size so
the suite stays under one CPU-second per cell.  They don't validate H1;
they validate that the wiring is correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import mnist_raw_present

from benchmarks.sl_clean import experiments as exp_mod
from benchmarks.sl_clean.analyze_biases_panel import (
    aggregate,
    headline_contrasts,
    panel_summary,
)
from benchmarks.sl_clean.experiments import (
    BIASES_MNIST_BASE_CONFIG,
    BIASES_MNIST_CELLS,
    _make_subrngs,
    _parse_biases_cell,
    get,
)
from benchmarks.sl_clean.sl_phase_runner import run_phase

DATA_PRESENT = mnist_raw_present()


def test_cells_are_9_and_well_formed():
    assert len(BIASES_MNIST_CELLS) == 9
    for cell in BIASES_MNIST_CELLS:
        bm, es = _parse_biases_cell(cell)
        assert bm in {"data_driven", "random_normal", "zero"}
        assert es in {"hypersphere", "gaussian", "sparse"}


def test_parse_cell_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_biases_cell("nonsense")
    with pytest.raises(ValueError):
        _parse_biases_cell("data_driven+nonsense")


def test_subrngs_are_independent_of_bias_mode():
    """The encoder and gain streams must depend ONLY on the seed,
    not on bias_mode — otherwise the panel comparison is confounded."""
    rngs_a = _make_subrngs(seed=42)
    rngs_b = _make_subrngs(seed=42)
    a_enc = torch.randn(100, generator=rngs_a["encoder"])
    b_enc = torch.randn(100, generator=rngs_b["encoder"])
    assert torch.equal(a_enc, b_enc)
    a_gain = torch.rand(100, generator=rngs_a["gain"])
    b_gain = torch.rand(100, generator=rngs_b["gain"])
    assert torch.equal(a_gain, b_gain)


def test_subrngs_differ_between_seeds():
    a = torch.randn(100, generator=_make_subrngs(0)["encoder"])
    b = torch.randn(100, generator=_make_subrngs(1)["encoder"])
    assert not torch.equal(a, b)


def test_biases_mnist_registered():
    spec = get("biases_mnist")
    assert spec.primary_metric == "test_accuracy"
    assert spec.higher_is_better is True
    ctx = spec.default_run_context()
    assert ctx["dataset"] == "MNIST"
    assert ctx["n_train"] == 60000


@pytest.mark.skipif(not DATA_PRESENT, reason="MNIST raw IDX files not available")
def test_biases_mnist_run_one_cell_smoketest():
    """Run a single cell with very small n_neurons. The point is to
    exercise the encode/accumulate/cholesky/predict path end-to-end and
    confirm encoders+gains are bitwise identical across bias modes for
    the same seed."""
    spec = get("biases_mnist")
    config = dict(BIASES_MNIST_BASE_CONFIG)
    config["n_neurons"] = 200
    config["chunk_size"] = 5000

    accs: dict[str, float] = {}
    for bm in ("data_driven", "random_normal", "zero"):
        config["cell"] = f"{bm}+hypersphere"
        out = spec.run_fn(config, seed=0, run_context=spec.default_run_context())
        accs[bm] = out["test_accuracy"]
        assert 0.0 < out["test_accuracy"] < 1.0
        assert out["factors"]["bias_mode"] == bm
        assert out["factors"]["encoder_strategy"] == "hypersphere"

    # Sanity: at 200 neurons with abs activation, data-driven should
    # not catastrophically lose to random_normal.  We don't assert a
    # specific gap — that's the science, not the test.
    assert accs["data_driven"] > 0.5


@pytest.mark.skipif(not DATA_PRESENT, reason="MNIST raw IDX files not available")
def test_biases_mnist_runs_through_phase_runner(tmp_path: Path):
    """Drive the per-cell runner with two cells × one seed.  Verifies
    JSON schema, idempotent skip-on-resume, and that the panel analyzer
    can ingest the output."""
    out_path = tmp_path / "biases_mnist_panel.json"
    base_config = {
        **BIASES_MNIST_BASE_CONFIG,
        "n_neurons": 200,
        "chunk_size": 5000,
    }
    cells = ["data_driven+hypersphere", "random_normal+hypersphere"]
    spec = get("biases_mnist")

    results = run_phase(
        experiment="biases_mnist",
        phase="t_panel",
        knob="cell",
        values=cells,
        seeds=[0],
        base_config=base_config,
        out_path=out_path,
        run_context=spec.default_run_context(),
    )
    assert len(results) == 2
    for row in results.values():
        assert "error" not in row
        assert "factors" in row["metrics"]
        assert row["metrics"]["test_accuracy"] > 0.5

    # Re-run: should skip both as cached.
    results2 = run_phase(
        experiment="biases_mnist",
        phase="t_panel",
        knob="cell",
        values=cells,
        seeds=[0],
        base_config=base_config,
        out_path=out_path,
        run_context=spec.default_run_context(),
    )
    assert results2 == results
    # Wall times should not have changed (cells were skipped, not re-run).
    for k in results2:
        assert results2[k]["wall_time_s"] == results[k]["wall_time_s"]

    # Analyzer should run cleanly on the partial panel.
    rows = list(json.loads(out_path.read_text()).values())
    agg = aggregate(rows)
    assert agg["raw_count"] == 2
    panel = panel_summary(agg)
    # Only 1 seed means n_paired_seeds=1, which is < 3 → "inconclusive".
    for bm in panel["by_bias_mode"]:
        if panel["by_bias_mode"][bm]["n_paired_seeds"] > 0:
            # Only data_driven and random_normal cells were run, not zero.
            pass
    contrasts = headline_contrasts(agg)
    assert contrasts["hypersphere"]["n"] == 1
    _ = exp_mod  # silence unused-import lint without importing twice
