import hashlib
import json
import struct
from dataclasses import replace

import pytest
import torch

from benchmarks.compression import run_henon_pilot as runner
from benchmarks.compression.baselines import AdaptiveCategorical, ARConfig, PrequentialAR
from benchmarks.compression.prequential import PredictorConfig, score_symbols


@pytest.fixture
def toy(monkeypatch):
    pilot = runner._Pilot(
        n=6,
        trajectory_seeds=(0, 1),
        c0=PredictorConfig(
            window=1,
            width=2,
            warmup=1,
            resolve_every=2,
            fixed_alpha=0.1,
            noise_variance=0.1,
            seed=7,
        ),
        ar=ARConfig(order=1, warmup=2, resolve_every=2, fixed_alpha=0.1, noise_variance=0.1),
    )
    raw = torch.tensor([-3, -2, -0.033, 0.785, 2, 3], dtype=torch.float64)
    monkeypatch.setattr(
        runner._Pilot, "observe", lambda self, seed: raw if seed == 0 else raw.flip(0)
    )
    return pilot, raw


def test_pilot_artifact_hashes_and_method_accounting(tmp_path, toy):
    pilot, raw = toy
    before = torch.random.get_rng_state()
    runner._run(tmp_path, pilot, None)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not (tmp_path / "pilot.json").exists()
    manifest = json.loads((tmp_path / "smoke.json").read_bytes())
    assert manifest["run"]["kind"] == "synthetic_smoke"
    assert manifest["run"]["common_suffix_start"] == 3
    assert manifest["run"]["all_stream_symbols"] == 6
    assert manifest["run"]["common_suffix_symbols"] == 3
    row = manifest["rows"][0]
    assert row["kind"] == "synthetic_smoke"
    assert row["code_identity"] is None
    assert row["saturation"] == {"below_lo": 1, "above_hi": 1}
    symbols = pilot.grid.quantize(raw)
    expected = {
        "henon_seed0_raw.f64le": struct.pack("<6d", *raw.tolist()),
        "henon_seed0_reconstructed.f32le": struct.pack(
            "<6f", *(pilot.grid.lo + symbol * pilot.grid.delta for symbol in symbols.tolist())
        ),
        "henon_seed0_symbols.u8": bytes(symbols.tolist()),
    }
    hashes = row["dataset_metadata"]["raw_hashes"] | row["dataset_metadata"]["processed_hashes"]
    assert hashes.keys() == expected.keys()
    for name, contents in expected.items():
        assert (tmp_path / name).read_bytes() == contents
        assert hashes[name] == hashlib.sha256(contents).hexdigest()

    reference = {
        "c0_fixed_feature": score_symbols(symbols, pilot.grid, pilot.c0).bits,
        "categorical_order0": AdaptiveCategorical(
            pilot.grid, pseudocount=pilot.pseudocount
        ).score_symbols(symbols),
        "ridge_ar": PrequentialAR(pilot.grid, pilot.ar).score_symbols(symbols).bits,
    }
    assert row["methods"] == {
        name: {
            "all_stream_ideal_bits": float(bits.sum()),
            "common_suffix_ideal_bits": float(bits[3:].sum()),
        }
        for name, bits in reference.items()
    }


def test_pilot_resumes_after_interruption_and_rejects_mismatch(tmp_path, toy, monkeypatch):
    pilot, _ = toy
    score = runner._score
    calls = 0

    def interrupt_second(config, symbols):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("synthetic interruption")
        return score(config, symbols)

    monkeypatch.setattr(runner, "_score", interrupt_second)
    with pytest.raises(InterruptedError, match="synthetic"):
        runner._run(tmp_path, pilot, None)
    manifest = tmp_path / "smoke.json"
    checkpoint = json.loads(manifest.read_bytes())
    assert [row["trajectory_seed"] for row in checkpoint["rows"]] == [0]
    assert (tmp_path / "henon_seed1_symbols.u8").is_file()

    with pytest.raises(ValueError, match="config, code identity or environment"):
        runner._run(tmp_path, replace(pilot, pseudocount=1.0), None)
    real_head = runner.subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=runner._ROOT, text=True
    ).strip()
    manifest.write_text(
        json.dumps(checkpoint | {"run": checkpoint["run"] | {"code_identity": real_head}})
    )
    with pytest.raises(ValueError, match="config, code identity or environment"):
        runner._run(tmp_path, pilot, None)

    manifest.write_text(json.dumps(checkpoint))
    stale = tmp_path / ".smoke.json.partial"
    stale.write_bytes(b"incomplete JSON")
    runner._run(tmp_path, pilot, None)
    assert calls == 3
    assert not stale.exists()
    final = json.loads(manifest.read_bytes())
    assert final["rows"][0] == checkpoint["rows"][0]
    assert [row["trajectory_seed"] for row in final["rows"]] == [0, 1]
    runner._run(tmp_path, pilot, None)
    assert calls == 3


def test_pilot_refuses_missing_or_corrupted_completed_artifacts(tmp_path, toy):
    pilot, _ = toy
    runner._run(tmp_path, pilot, None)
    raw = tmp_path / "henon_seed0_raw.f64le"
    original = raw.read_bytes()
    raw.write_bytes(struct.pack("<d", -4.0) + original[8:])
    with pytest.raises(ValueError, match="artifact hashes do not match"):
        runner._run(tmp_path, pilot, None)
    raw.write_bytes(original)
    path = tmp_path / "henon_seed0_symbols.u8"
    path.write_bytes(b"\x00" * pilot.n)
    with pytest.raises(ValueError, match="artifact contents do not match"):
        runner._run(tmp_path, pilot, None)
    path.unlink()
    with pytest.raises(FileNotFoundError, match=path.name):
        runner._run(tmp_path, pilot, None)
    (tmp_path / "smoke.json").unlink()
    with pytest.raises(ValueError, match="empty output directory"):
        runner._run(tmp_path, pilot, None)


@pytest.mark.parametrize(
    ("status", "ignored", "error"),
    [
        (" M benchmarks/compression/baselines.py\n", "", "clean checkout"),
        ("?? benchmarks/compression/extra.py\n", "", "clean checkout"),
        ("", "benchmarks/compression/ignored.py\0", "ignored implementation"),
        ("", "src/leenef/__pycache__/layers.cpython-312.pyc\0", None),
    ],
)
def test_pilot_clean_identity_gate(monkeypatch, status, ignored, error):
    check_output = runner.subprocess.check_output
    real_head = check_output(("git", "rev-parse", "HEAD"), cwd=runner._ROOT, text=True).strip()

    def read_git(args, **kwargs):
        if args[1] == "status":
            assert "--untracked-files=all" in args
            return status
        if args[1] == "ls-files":
            return ignored
        return check_output(args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "check_output", read_git)
    if error is None:
        assert runner._clean_identity(runner._ROOT) == real_head
    else:
        with pytest.raises(RuntimeError, match=error):
            runner._clean_identity(runner._ROOT)
