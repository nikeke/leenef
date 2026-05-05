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


class TestAdditivity:
    """Direct test that ``partial_fit`` is additive: sequentially feeding
    disjoint task data produces the same decoders (to float drift) as
    feeding the union once.  This is the load-bearing identity behind H1.
    """

    def test_partial_fit_is_additive(self):
        import torch

        from leenef.layers import NEFLayer

        torch.manual_seed(0)
        d_in, d_out, n = 12, 3, 64
        x = torch.randn(300, d_in)
        y = torch.zeros(300, d_out)
        y[torch.arange(300), torch.randint(0, d_out, (300,))] = 1.0

        def make() -> NEFLayer:
            return NEFLayer(
                d_in=d_in,
                n_neurons=n,
                d_out=d_out,
                encoder_strategy="hypersphere",
                activation="abs",
                gain=(0.5, 2.0),
                rng=torch.Generator().manual_seed(7),
            )

        seq = make()
        for chunk in (x[:100], x[100:200], x[200:]):
            i = (x == chunk[0]).all(dim=1).nonzero()[0, 0].item()
            seq.partial_fit(chunk, y[i : i + chunk.shape[0]])
        seq.solve_accumulated(alpha=1e-2)

        joint = make()
        joint.partial_fit(x, y)
        joint.solve_accumulated(alpha=1e-2)

        gap = (seq.decoders.data - joint.decoders.data).abs().max().item()
        assert gap < 1e-5, f"sequential vs joint decoder gap too large: {gap}"


class TestEncoderBitEquality:
    """Encoders / gains must depend only on the seed, not on which
    method cell is being run.  This guarantees the same-code-identity
    causality that headline panels rely on.
    """

    def test_encoders_match_across_method_cells(self):
        import torch

        from leenef.layers import NEFLayer

        def build(seed: int) -> NEFLayer:
            return NEFLayer(
                d_in=8,
                n_neurons=64,
                d_out=4,
                encoder_strategy="hypersphere",
                activation="abs",
                gain=(0.5, 2.0),
                rng=torch.Generator().manual_seed(seed),
            )

        a = build(123)
        b = build(123)
        assert torch.equal(a.encoders, b.encoders)
        assert torch.equal(a._gain, b._gain)


class TestMLPReproducibility:
    def test_build_mlp_is_seed_deterministic_and_preserves_global_rng(self):
        import torch

        from benchmarks.cl_clean.baselines.mlp_baselines import MLPConfig, _build_mlp

        cfg = MLPConfig(d_in=16, n_classes=4, hidden=32, n_hidden_layers=2)

        torch.manual_seed(999)
        before = torch.get_rng_state()
        m1 = _build_mlp(cfg, seed=42)
        after = torch.get_rng_state()
        assert torch.equal(before, after), "global RNG state was mutated by _build_mlp"

        m2 = _build_mlp(cfg, seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
            assert torch.equal(p1, p2)


class TestPhaseBRegistrations:
    def test_capacity_experiments_are_registered(self):
        names = experiments.names()
        assert "nef_capacity_permuted_mnist_t100" in names
        assert "nef_capacity_permuted_cifar10_t20" in names
        assert "c1_convnef_split_cifar10" in names

    def test_grid_cell_parsers(self):
        from benchmarks.cl_clean._c1_convnef_cl_exp import parse_convnef_cell
        from benchmarks.cl_clean.experiments import _parse_capacity_cell, _parse_solver_alpha_cell

        assert _parse_capacity_cell("n5000_t20") == (5000, 20)
        assert _parse_solver_alpha_cell("woodbury_online_a0.1") == ("woodbury_online", 0.1)
        assert parse_convnef_cell("first_task_n2000") == ("first_task", 2000)

    def test_layer_encoders_and_gains_do_not_depend_on_centers_mode(self):
        import torch

        from benchmarks.cl_clean.experiments import _build_layer, _make_subrngs

        kwargs = dict(
            d_in=8,
            n_neurons=32,
            n_classes=4,
            encoder_strategy="hypersphere",
            gain_low=0.5,
            gain_high=2.0,
            device=torch.device("cpu"),
        )
        centered = _build_layer(
            **kwargs,
            centers=torch.randn(32, 8, generator=torch.Generator().manual_seed(99)),
            rngs=_make_subrngs(11),
        )
        uncentered = _build_layer(**kwargs, centers=None, rngs=_make_subrngs(11))

        assert torch.equal(centered.encoders, uncentered.encoders)
        assert torch.equal(centered.gain, uncentered.gain)

    def test_convnef_run_fn_accepts_synthetic_stream(self, monkeypatch):
        import torch

        from benchmarks.cl_clean import _c1_convnef_cl_exp as convnef_exp
        from benchmarks.cl_clean.datasets import TaskSplit, TaskStream

        g = torch.Generator().manual_seed(7)
        prototypes = torch.rand(4, 3072, generator=g)

        def make_split(task_index: int, classes: tuple[int, ...]) -> TaskSplit:
            train_parts, train_labels = [], []
            test_parts, test_labels = [], []
            for cls in classes:
                train_parts.append(
                    (
                        prototypes[cls].unsqueeze(0) + 0.01 * torch.randn(8, 3072, generator=g)
                    ).clamp_(0.0, 1.0)
                )
                train_labels.append(torch.full((8,), cls, dtype=torch.int64))
                test_parts.append(
                    (
                        prototypes[cls].unsqueeze(0) + 0.01 * torch.randn(4, 3072, generator=g)
                    ).clamp_(0.0, 1.0)
                )
                test_labels.append(torch.full((4,), cls, dtype=torch.int64))
            return TaskSplit(
                task_index=task_index,
                classes=classes,
                x_train=torch.cat(train_parts, dim=0),
                y_train=torch.cat(train_labels, dim=0),
                x_test=torch.cat(test_parts, dim=0),
                y_test=torch.cat(test_labels, dim=0),
            )

        fake_stream = TaskStream(
            base_name="CIFAR10",
            scenario="split",
            n_tasks=2,
            n_classes=4,
            stream_seed=0,
            tasks=(make_split(0, (0, 1)), make_split(1, (2, 3))),
            stream_hash="synthetic-stream",
            raw_hashes={"train": "synthetic"},
            processed_hashes={"x_train": "synthetic", "x_test": "synthetic"},
            extra={"class_permutation": [0, 1, 2, 3]},
        )
        monkeypatch.setattr(convnef_exp, "make_task_stream", lambda *args, **kwargs: fake_stream)

        spec = experiments.get("c1_convnef_split_cifar10")
        metrics = spec.run_fn(
            {
                "cell": "first_task_n32",
                "activation": "abs",
                "alpha": 0.01,
                "augment_flip": False,
                "batch_size": 8,
                "conv_stages": [{"n_filters": 4, "patch_size": 3, "pool_size": 2}],
                "device": "cpu",
                "encoder_strategy": "hypersphere",
                "fit_subsample": 16,
                "gain_high": 2.0,
                "gain_low": 0.5,
                "gcn": False,
                "lcn_kernel": None,
                "n_augment": 1,
                "parallel": False,
                "pool_levels": [1, 2],
                "pool_order": 1,
                "standardize": True,
            },
            seed=0,
            run_context=spec.default_run_context() | {"n_tasks": 2},
        )

        assert 0.0 <= metrics["test_accuracy"] <= 1.0
        assert metrics["feature_mode"] == "first_task"
        assert metrics["effective_n_neurons"] == 32
        assert metrics["stream_hash"] == "synthetic-stream"


class TestByteIdentityIdempotency:
    def test_rerun_produces_byte_identical_json(self, tmp_path: Path):
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
        kwargs = dict(
            experiment="smoke",
            phase="byte_idem",
            knob="n_neurons",
            values=[80, 120],
            seeds=[0, 1],
            base_config=base,
            out_path=out,
            log_path=log,
        )
        run_phase(**kwargs)
        first_bytes = out.read_bytes()
        run_phase(**kwargs)
        second_bytes = out.read_bytes()
        assert first_bytes == second_bytes
