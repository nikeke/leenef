"""Fast synthetic tests for the neuron-selection experiment protocol."""

import json
import os
import time
from dataclasses import replace

import pytest
import torch

from benchmarks.neuron_selection import analyze, common, growth_panel, prune_panel
from benchmarks.neuron_selection.common import (
    LOOP_ALPHA,
    accumulate,
    atomic_write,
    keyed_rng,
    make_layer,
    make_split,
    read_results,
    row_key,
    seed_shards,
    select_alpha,
    stats,
)
from benchmarks.sl_clean.datasets import DatasetBundle
from leenef import NEFLayer, make_encoders, solve_from_normal_equations


@pytest.fixture(scope="module")
def bundle():
    rng = torch.Generator().manual_seed(11)
    # Four nonlinear classes concentrated in four directions in a 64-D space.
    x = torch.randn(1500, 64, generator=rng)
    w = torch.linalg.qr(torch.randn(64, 4, generator=rng)).Q
    y = (x @ w).abs().argmax(dim=1)
    return DatasetBundle(
        name="Synthetic",
        x_train=x[:1200],
        y_train=y[:1200],
        x_test=x[1200:],
        y_test=y[1200:],
        n_classes=4,
        preprocessing="div255_flat",
        raw_hashes={"source": "synthetic-seed-11"},
        processed_hashes={"data": "synthetic-seed-11"},
    )


@pytest.fixture(scope="module", autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _metrics(result):
    return {
        key: value
        for key, value in result.items()
        if key.startswith("alpha") or "accuracy" in key or "mse" in key
    }


def _assert_row(row):
    assert {
        "key",
        "seed",
        "wall_time_s",
        "code_identity",
        "dataset_metadata",
        "hardware",
        "torch_threads",
        "peak_rss_mb",
        "factors",
    } <= row.keys()
    assert row["dataset_metadata"]["raw_hashes"] and row["dataset_metadata"]["processed_hashes"]
    assert row["peak_rss_mb"] > 0 and row["peak_rss_source"] in {"VmHWM", "ru_maxrss"}
    assert row["alpha_loop"] == LOOP_ALPHA[0]
    for key in ("val_accuracy", "test_accuracy", "fit_accuracy", "val_mse", "test_mse"):
        assert key in row and f"{key}_loop" in row


def test_prune_rows_use_one_random_reference_per_width_and_resume(bundle, tmp_path, monkeypatch):
    out = tmp_path / "prune.json"
    opts = dict(
        seeds=[0],
        out=out,
        strategies=("hypersphere",),
        pool_max=160,
        pools=(80, 160),
        ks=(20, 40),
        random_widths=(20, 40, 60, 120, 160),
        image_shape=(8, 8),
        val_size=200,
    )
    rows = prune_panel.run_panel(bundle, **opts)["rows"]
    assert len(rows) == 23
    assert len({r["key"] for r in rows}) == len(rows)
    for row in rows:
        _assert_row(row)
    random = [r for r in rows if r["criterion"] == "random"]
    assert sorted(r["k"] for r in random) == [20, 40, 60, 120, 160]
    assert all(r["P"] is None for r in random)
    keys = {(r["criterion"], r["P"], r["k"]): r for r in rows}
    assert keys["obs", 160, 40]["test_accuracy"] >= keys["activity", 160, 40]["test_accuracy"]

    stale = {**rows[0], "code_identity": "stale", "test_accuracy": -123.0}
    payload = read_results(out)
    payload["rows"][0] = stale
    atomic_write(out, payload)
    resumed = prune_panel.run_panel(bundle, **opts)["rows"]
    assert len(resumed) == len(rows) + 1
    assert resumed[0] == stale and resumed[-1]["key"] == stale["key"]
    monkeypatch.setattr(prune_panel, "make_split", lambda *_a, **_kw: pytest.fail("not resumed"))
    prune_panel.run_panel(bundle, **opts)
    assert analyze.main([str(out), "--identity", rows[0]["code_identity"]]) == 0
    report = read_results(out.with_name("prune_summary.json"))
    random20 = next(r for r in report["prune"] if r["criterion"] == "random" and r["k"] == 20)
    assert random20["test"] == {"mean": rows[0]["test_accuracy"], "se": None, "n": 1}


def test_growth_baselines_and_matched_rng_draws(bundle, tmp_path, monkeypatch):
    observed_centers, observed_classes = [], []
    original_grow = NEFLayer.grow
    original_sample = growth_panel.sample_directions_from_bases

    def capture_grow(layer, encoders, *, centers, rng):
        copy = torch.Generator().set_state(rng.get_state())
        expected_centers = torch.randint(len(centers), (len(encoders),), generator=copy)
        original_grow(layer, encoders, centers=centers, rng=rng)
        gains = layer.gain[-len(encoders) :].clone()
        expected_bias = -gains * (centers[expected_centers] * encoders).sum(dim=1)
        torch.testing.assert_close(layer.bias[-len(encoders) :], expected_bias)
        observed_centers.append((expected_centers, gains))

    def capture_sample(k, bases, *, rng):
        copy = torch.Generator().set_state(rng.get_state())
        observed_classes.append(torch.randint(len(bases), (k,), generator=copy))
        return original_sample(k, bases, rng=rng)

    with monkeypatch.context() as context:
        context.setattr(NEFLayer, "grow", capture_grow)
        context.setattr(growth_panel, "sample_directions_from_bases", capture_sample)
        growth_panel.run_panel(
            bundle,
            seeds=[0],
            out=tmp_path / "matched.json",
            families=("global",),
            arms=("phd", "phd_classshuffle"),
            ks=(20,),
            generations=1,
            val_size=200,
            image_shape=(8, 8),
        )
    assert len(observed_centers) == len(observed_classes) == 2
    for phd, shuffle in zip(*observed_centers, strict=True):
        assert torch.equal(phd, shuffle)
    assert torch.equal(*observed_classes)

    rows = growth_panel.run_panel(
        bundle,
        seeds=[0],
        out=tmp_path / "growth.json",
        ks=(30,),
        generations=4,
        val_size=200,
        image_shape=(8, 8),
    )["rows"]
    assert len(rows) == 10
    for family in growth_panel.FAMILIES:
        group = [r for r in rows if r["family"] == family]
        initial = _metrics(group[0]["generations"][0])
        assert all(_metrics(r["generations"][0]) == initial for r in group)
    for row in rows:
        _assert_row(row)
        assert row["final"] == row["generations"][-1]
        assert row["val_selected"] == max(row["generations"], key=lambda g: g["val_accuracy"])
    phd = next(r for r in rows if r["arm"] == "phd")
    assert phd["final"]["val_accuracy"] > phd["generations"][0]["val_accuracy"]
    assert phd["generations"][0]["eig_converged"] is None
    assert all(
        g["eig_converged"] and g["eig_iterations"] >= 5 and g["eig_worst_change"] < 1e-7
        for g in phd["generations"][1:]
    )


def test_randkeep_uses_exact_budget_and_run_is_deterministic(bundle, monkeypatch):
    keeps = []
    original = NEFLayer.prune

    def record(layer, keep):
        keeps.append(keep.clone())
        original(layer, keep)

    monkeypatch.setattr(NEFLayer, "prune", record)
    split = make_split(bundle, 3, val_size=200)
    options = dict(
        family="global",
        arm="phd_randkeep",
        k=20,
        seed=3,
        generations=2,
        image_shape=(8, 8),
        chunk_size=250,
    )
    first, second = [
        growth_panel.run_trial(split, split.x_fit.mean(0), **options) for _ in range(2)
    ]
    for g in (1, 2):
        expected = torch.randperm(40, generator=keyed_rng(3, "retain", g))[:20].sort().values
        assert torch.equal(keeps[g - 1], expected) and torch.equal(keeps[g + 1], expected)
        assert first["generations"][g]["n_survivors"] == int((expected < 20).sum())
    assert [_metrics(g) for g in first["generations"]] == [
        _metrics(g) for g in second["generations"]
    ]


def test_validation_tuning_leaves_the_loop_decoder_in_the_layer(bundle):
    split = make_split(bundle, 1, val_size=20)
    layer = make_layer(split, 10, "hypersphere", (8, 8), keyed_rng(1, "pool"))
    accumulate(layer, split, 250)
    g, r = stats(layer)
    r.zero_()  # All candidate decoders tie on validation.
    reported = select_alpha(g, r, split.evaluation(), chunk_size=250, layer=layer)
    assert reported["alpha"] > reported["alpha_loop"]

    accumulate(layer, split, 250)
    g, r = stats(layer)
    select_alpha(g, r, split.evaluation(), chunk_size=250, layer=layer)
    expected = solve_from_normal_equations(g, r, alpha=LOOP_ALPHA).float()
    assert torch.equal(layer.decoders, expected)


def test_receptive_field_support_sizes_match_dataset_layout():
    for dim, shape, width in [(784, (28, 28), 25), (3072, (32, 32), 75)]:
        enc = make_encoders(
            1000,
            dim,
            strategy="receptive_field",
            image_shape=shape,
            patch_size=5,
            rng=keyed_rng(0, "proposal", 1),
        )
        assert ((enc != 0).sum(dim=1) == width).all()


def test_analyzer_partial_growth_pairs_final_values_and_corrects_holm(
    tmp_path, monkeypatch, capsys
):
    identity = "test-identity"
    growth = []
    prune = []
    for seed, control in enumerate([0.70, 0.72, 0.74]):
        for arm, score in [
            ("phd", 0.83 + 0.01 * seed),
            ("random", control),
            ("phd_classshuffle", control + 0.11),
            ("phd_randkeep", control + 0.05),
        ]:
            growth.append(
                {
                    "dataset": "MNIST",
                    "family": "global",
                    "k": 1000,
                    "arm": arm,
                    "seed": seed,
                    "code_identity": identity,
                    "final": {"test_accuracy": score, "test_accuracy_loop": control},
                    "val_selected": {"test_accuracy": 0.9},
                    "generations": [
                        {
                            "gen": 1,
                            "val_mse_loop": 0.1 if arm == "phd" else 0.2,
                            "eig_converged": False if arm == "phd" and seed == 2 else None,
                        }
                    ],
                }
            )
        prune.append(
            {
                "dataset": "MNIST",
                "encoder_strategy": "hypersphere",
                "k": 3000,
                "P": None,
                "seed": seed,
                "criterion": "random",
                "code_identity": identity,
                "test_accuracy": control + 0.08,
                "test_accuracy_loop": control + 0.08,
            }
        )
    for row in growth:
        row["key"] = row_key({"dataset": row["dataset"], "arm": row["arm"]}, row["seed"])
    gp, gp2, pp = tmp_path / "growth_s0.json", tmp_path / "growth_s1.json", tmp_path / "prune.json"
    atomic_write(gp, {"rows": growth[:8]})
    duplicate = {**growth[0], "final": {"test_accuracy": 0.01}}
    stale = {**growth[8], "code_identity": "stale", "final": {"test_accuracy": 0.01}}
    atomic_write(gp2, {"rows": [duplicate, stale, *growth[8:]]})
    atomic_write(pp, {"rows": prune})
    monkeypatch.setattr(analyze, "student_t", None)
    assert analyze.main([str(gp), str(gp2), "--prune-json", str(pp), "--identity", identity]) == 0
    summary = json.loads((tmp_path / "growth_s0_summary.json").read_text())
    assert summary == json.loads((tmp_path / "growth_s1_summary.json").read_text())
    contrasts = {r["contrast"]: r for r in summary["contrasts"]}
    assert contrasts["width3x"]["mean"] == pytest.approx(0.04)
    assert contrasts["width3x"]["n"] == 3
    assert contrasts["width3x"]["p_one_sided"] == pytest.approx(0.01010205144, rel=1e-6)
    assert contrasts["width3x"]["lower_95"] == pytest.approx(0.02314145539, rel=1e-6)
    assert contrasts["width3x"]["decision"] == "pass"
    assert contrasts["mechanism"]["decision"] == "fail"
    assert contrasts["proposal"]["p_holm"] >= contrasts["proposal"]["p_one_sided"]
    assert "p_holm" not in contrasts["selection"]
    assert "decision" not in contrasts["selection"]
    assert next(r for r in summary["contrasts_loop"] if r["contrast"] == "proposal")["mean"] == 0
    assert summary["eig_failures"] == [
        dict(dataset="MNIST", family="global", arm="phd", k=1000, seed=2, generations=[1])
    ]
    output = capsys.readouterr().out
    assert "1 growth row(s) have non-converged generations" in output
    assert "Secondary paired contrasts: `test_accuracy_loop`" in output
    assert "| Holm p | Decision |" in output
    assert summary["diagnostics"][0]["mean"] == pytest.approx(0.1)


@pytest.mark.parametrize("panel", [prune_panel, growth_panel])
def test_panel_cli_rejects_wrong_pinned_identity(panel, monkeypatch):
    monkeypatch.setenv("NS_IDENTITY", "other-checkout")
    with pytest.raises(RuntimeError, match="code identity mismatch: NS_IDENTITY=other-checkout"):
        panel.main(["--dataset", "MNIST", "--smoke"])


def test_panel_source_guard_names_foreign_and_expected_paths(monkeypatch):
    monkeypatch.setattr(common.leenef, "__file__", "other/src/leenef/__init__.py")
    with pytest.raises(RuntimeError, match=r"other/src/leenef/__init__.py; expected .*?/src"):
        common.verify_panel_source()


def test_smoke_manifest_verifies_actual_rows_and_rejects_stale_sentinel(
    bundle, monkeypatch, tmp_path, capsys
):
    dataset = replace(
        bundle,
        name="MNIST",
        x_train=torch.nn.functional.pad(bundle.x_train, (0, 720)),
        x_test=torch.nn.functional.pad(bundle.x_test, (0, 720)),
    )
    monkeypatch.setattr(common, "load_dataset", lambda *_args, **_kwargs: dataset)
    for panel in (prune_panel, growth_panel):
        monkeypatch.setattr(
            panel, "make_split", lambda data, seed, **kw: make_split(data, seed, val_size=200)
        )
    identity = common.code_identity()
    monkeypatch.setenv("NS_IDENTITY", identity)
    logs = tmp_path / "logs"
    logs.mkdir()
    manifest = logs / "run_manifest.txt"
    assert (
        common.main(
            [
                "manifest",
                str(manifest),
                "--identity",
                identity,
                "--seeds",
                "0",
                "--threads",
                "2",
                "--shards",
                "2",
                "--smoke",
                "MNIST=random",
            ]
        )
        == 0
    )
    start = json.loads(manifest.read_text())
    expected = common.expected_result_keys(start)
    base = ["--dataset", "MNIST", "--seeds", "0", "--threads", "2", "--smoke"]
    done = tmp_path / "panels_done"
    done.touch()
    os.utime(done, ns=(start["started_ns"] - 1, start["started_ns"] - 1))
    assert common.main(["status", str(manifest)]) == 2
    assert prune_panel.main([*base, "--out", str(tmp_path / "prune_MNIST.json")]) == 0
    assert (
        growth_panel.main(
            [*base, "--arms", "random", "--out", str(tmp_path / "growth_MNIST_s0.json")]
        )
        == 0
    )
    for filename, keys in expected.items():
        payload = read_results(tmp_path / filename)
        assert {row["key"] for row in payload["rows"]} == keys
    fresh = max(time.time_ns(), start["started_ns"])
    os.utime(done, ns=(fresh, fresh))
    assert common.main(["status", str(manifest)]) == 0
    status = capsys.readouterr().out.splitlines()[-1]
    assert all(
        f"{filename} {len(keys)}/{len(keys)}" in status for filename, keys in expected.items()
    )
    (tmp_path / "growth_MNIST_s0.json").unlink()
    assert common.main(["status", str(manifest)]) == 1


def test_seed_shards_preserve_seed_order_without_empty_workers():
    seeds = [9, 1, 8, 6, 3]
    assert seed_shards(seeds, 2) == [[9, 1], [8, 6, 3]]
    assert seed_shards(seeds, 6) == [[seed] for seed in seeds]
    with pytest.raises(ValueError, match="GROWTH_SHARDS"):
        seed_shards(seeds, 0)
