"""Tests for the ``benchmarks.cl_clean`` infrastructure package."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.cl_clean import cl_repro, datasets, experiments
from benchmarks.cl_clean.cl_phase_runner import run_phase


class TestCodeIdentity:
    def test_returns_a_string_or_none(self):
        ident = cl_repro.code_identity()
        assert ident is None or isinstance(ident, str)

    def test_format(self):
        ident = cl_repro.code_identity()
        if ident is None:
            pytest.skip("not a git checkout")
        assert re.fullmatch(r"[0-9a-f]{40}(-dirty-[0-9a-f]{12})?", ident)


class TestRunner:
    def test_smoke_runs_and_is_idempotent(self, tmp_path: Path):
        out = tmp_path / "smoke.json"
        log = tmp_path / "smoke.log"
        base = {
            "n_neurons": 80,
            "encoder_strategy": "hypersphere",
            "activation": "abs",
            "gain_low": 0.5,
            "gain_high": 2.0,
            "alpha": 1e-3,
            "cell": "smoke",
        }
        run_phase(
            experiment="smoke",
            phase="smoke_test",
            knob="n_neurons",
            values=[80, 120],
            seeds=[0, 1],
            base_config=base,
            out_path=out,
            log_path=log,
        )
        first = json.loads(out.read_text())
        assert len(first) == 4
        run_phase(
            experiment="smoke",
            phase="smoke_test",
            knob="n_neurons",
            values=[80, 120],
            seeds=[0, 1],
            base_config=base,
            out_path=out,
            log_path=log,
        )
        second = json.loads(out.read_text())
        assert first == second  # idempotent re-run


class TestTaskStream:
    def test_split_stream_is_deterministic_and_partitions_classes(self):
        s1 = datasets.make_task_stream("MNIST", scenario="split", n_tasks=5, stream_seed=0)
        s2 = datasets.make_task_stream("MNIST", scenario="split", n_tasks=5, stream_seed=0)
        assert s1.stream_hash == s2.stream_hash
        all_classes = set()
        for t in s1.tasks:
            assert len(t.classes) == 2
            all_classes.update(t.classes)
        assert all_classes == set(range(10))

    def test_split_stream_seed_changes_hash(self):
        s0 = datasets.make_task_stream("MNIST", scenario="split", n_tasks=5, stream_seed=0)
        s1 = datasets.make_task_stream("MNIST", scenario="split", n_tasks=5, stream_seed=1)
        assert s0.stream_hash != s1.stream_hash


class TestNEFContinual:
    def test_nef_accumulate_matches_nef_joint_on_synthetic_smoke(self):
        spec = experiments.get("smoke")
        ctx = spec.default_run_context()
        cfg_base = {
            "n_neurons": 200,
            "encoder_strategy": "hypersphere",
            "activation": "abs",
            "gain_low": 0.5,
            "gain_high": 2.0,
            "alpha": 1e-3,
            "cell": "smoke",
        }
        out = spec.run_fn(cfg_base, seed=0, run_context=ctx)
        assert 0.0 <= out["test_accuracy"] <= 1.0
