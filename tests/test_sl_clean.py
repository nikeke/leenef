"""Tests for the ``benchmarks.sl_clean`` infrastructure package."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.sl_clean import experiments, sl_repro
from benchmarks.sl_clean.analyze_sl_sweep import aggregate
from benchmarks.sl_clean.sl_phase_runner import run_phase

# ---------------------------------------------------------------------------
# sl_repro.code_identity
# ---------------------------------------------------------------------------


class TestCodeIdentity:
    def test_returns_a_string_or_none(self):
        ident = sl_repro.code_identity()
        assert ident is None or isinstance(ident, str)

    def test_clean_tree_is_a_bare_sha_dirty_tree_is_suffixed(self):
        ident = sl_repro.code_identity()
        if ident is None:
            pytest.skip("not a git checkout")
        assert re.fullmatch(r"[0-9a-f]{40}(-dirty-[0-9a-f]{12})?", ident)

    def test_cache_can_be_reset(self):
        first = sl_repro.code_identity()
        sl_repro.reset_code_identity_cache()
        second = sl_repro.code_identity()
        assert first == second  # same tree → same identity even after reset


# ---------------------------------------------------------------------------
# sl_repro.result_metadata / can_reuse_result
# ---------------------------------------------------------------------------


class TestMetadata:
    def _meta(self, **overrides):
        cfg = {"n_neurons": 200, "alpha": 1e-3}
        ctx = {"dataset": "synthetic", "n_train": 256}
        meta = sl_repro.result_metadata(
            config=cfg,
            seed=0,
            experiment="smoke",
            run_context=ctx,
            primary_metric="test_accuracy",
            higher_is_better=True,
        )
        meta.update(overrides)
        return meta, cfg, ctx

    def test_required_fields_present(self):
        meta, _, _ = self._meta()
        for key in (
            "result_namespace",
            "runner_version",
            "experiment",
            "config",
            "seed",
            "seed_policy",
            "code_identity",
            "run_context",
            "primary_metric",
            "higher_is_better",
            "hardware",
        ):
            assert key in meta, key
        assert meta["result_namespace"] == "sl_clean"
        assert meta["runner_version"] == "sl-clean-v1"

    def test_can_reuse_round_trips(self):
        meta, cfg, ctx = self._meta()
        assert sl_repro.can_reuse_result(
            meta,
            experiment="smoke",
            expected_config=cfg,
            expected_seed=0,
            expected_run_context=ctx,
        )

    def test_can_reuse_rejects_seed_mismatch(self):
        meta, cfg, ctx = self._meta()
        assert not sl_repro.can_reuse_result(
            meta,
            experiment="smoke",
            expected_config=cfg,
            expected_seed=1,
            expected_run_context=ctx,
        )

    def test_can_reuse_rejects_config_mismatch(self):
        meta, cfg, ctx = self._meta()
        bad_cfg = dict(cfg, n_neurons=999)
        assert not sl_repro.can_reuse_result(
            meta,
            experiment="smoke",
            expected_config=bad_cfg,
            expected_seed=0,
            expected_run_context=ctx,
        )

    def test_can_reuse_rejects_code_identity_mismatch(self):
        meta, cfg, ctx = self._meta(code_identity="deadbeef" * 5)
        assert not sl_repro.can_reuse_result(
            meta,
            experiment="smoke",
            expected_config=cfg,
            expected_seed=0,
            expected_run_context=ctx,
        )

    def test_can_reuse_rejects_error_rows(self):
        meta, cfg, ctx = self._meta(error="boom")
        assert not sl_repro.can_reuse_result(
            meta,
            experiment="smoke",
            expected_config=cfg,
            expected_seed=0,
            expected_run_context=ctx,
        )


# ---------------------------------------------------------------------------
# expected_config_with_knob
# ---------------------------------------------------------------------------


def test_expected_config_with_knob_validates_knob_membership():
    base = {"n_neurons": 200, "alpha": 1e-3}
    assert sl_repro.expected_config_with_knob(base, "n_neurons", 400) == {
        "n_neurons": 400,
        "alpha": 1e-3,
    }
    with pytest.raises(KeyError):
        sl_repro.expected_config_with_knob(base, "nope", 1)


# ---------------------------------------------------------------------------
# experiments registry
# ---------------------------------------------------------------------------


def test_smoke_experiment_registered():
    spec = experiments.get("smoke")
    assert spec.primary_metric == "test_accuracy"
    assert spec.higher_is_better is True
    ctx = spec.default_run_context()
    assert ctx["experiment_kind"] == "synthetic_binary_classification"


def test_smoke_run_fn_produces_metrics():
    spec = experiments.get("smoke")
    metrics = spec.run_fn(
        experiments.SMOKE_BASE_CONFIG, seed=0, run_context=spec.default_run_context()
    )
    assert {"train_accuracy", "test_accuracy", "fit_time_s"} <= metrics.keys()
    assert 0.0 <= metrics["test_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Runner end-to-end: idempotency and skip-on-resume.
# ---------------------------------------------------------------------------


class TestRunner:
    @pytest.fixture
    def base_config(self):
        return dict(experiments.SMOKE_BASE_CONFIG)

    def test_runner_writes_one_row_per_cell(self, tmp_path: Path, base_config: dict):
        out = tmp_path / "smoke_n_neurons.json"
        results = run_phase(
            experiment="smoke",
            phase="smoke_n_neurons",
            knob="n_neurons",
            values=[100, 200],
            seeds=[0, 1],
            base_config=base_config,
            out_path=out,
        )
        assert len(results) == 4
        assert out.exists()
        on_disk = json.loads(out.read_text())
        assert on_disk == results
        for key in ("smoke_n_neurons_100_s0", "smoke_n_neurons_200_s1"):
            assert key in results
            assert "metrics" in results[key]
            assert results[key]["knob"] == "n_neurons"

    def test_runner_is_idempotent_on_resume(self, tmp_path: Path, base_config: dict):
        out = tmp_path / "smoke_n_neurons.json"
        first = run_phase(
            experiment="smoke",
            phase="smoke_n_neurons",
            knob="n_neurons",
            values=[100],
            seeds=[0, 1],
            base_config=base_config,
            out_path=out,
        )
        # Capture identity that includes the wall_time, which would be
        # different on a re-execution.
        first_times = {k: v["wall_time_s"] for k, v in first.items()}
        second = run_phase(
            experiment="smoke",
            phase="smoke_n_neurons",
            knob="n_neurons",
            values=[100],
            seeds=[0, 1],
            base_config=base_config,
            out_path=out,
        )
        second_times = {k: v["wall_time_s"] for k, v in second.items()}
        # Cells were reused, so wall_time fields are byte-identical.
        assert first_times == second_times

    def test_runner_recomputes_when_config_changes(self, tmp_path: Path, base_config: dict):
        out = tmp_path / "smoke_alpha.json"
        run_phase(
            experiment="smoke",
            phase="smoke_alpha",
            knob="alpha",
            values=[1e-3],
            seeds=[0],
            base_config=base_config,
            out_path=out,
        )
        first = json.loads(out.read_text())
        # Mutate stored config so it no longer matches the base.
        for row in first.values():
            row["config"]["n_neurons"] = 999
        out.write_text(json.dumps(first))
        run_phase(
            experiment="smoke",
            phase="smoke_alpha",
            knob="alpha",
            values=[1e-3],
            seeds=[0],
            base_config=base_config,
            out_path=out,
        )
        second = json.loads(out.read_text())
        # The mutated row must have been overwritten with a fresh one
        # whose config matches the current base.
        for row in second.values():
            assert row["config"]["n_neurons"] == base_config["n_neurons"]


# ---------------------------------------------------------------------------
# Analyzer aggregation.
# ---------------------------------------------------------------------------


def test_aggregate_filters_off_namespace_rows(tmp_path: Path):
    base_config = dict(experiments.SMOKE_BASE_CONFIG)
    out = tmp_path / "phase.json"
    run_phase(
        experiment="smoke",
        phase="agg",
        knob="n_neurons",
        values=[100, 200],
        seeds=[0, 1, 2],
        base_config=base_config,
        out_path=out,
    )
    sweep = json.loads(out.read_text())
    sweep["junk_row"] = {"result_namespace": "other", "knob": "n_neurons", "value": 100}
    spec = experiments.get("smoke")
    stats = aggregate(
        sweep,
        experiment="smoke",
        knob="n_neurons",
        base_config=base_config,
        expected_run_context=spec.default_run_context(),
        metric="test_accuracy",
    )
    assert set(stats.keys()) == {"100", "200"}
    for tag, s in stats.items():
        assert s["n"] == 3
        assert 0.0 <= s["mean"] <= 1.0
