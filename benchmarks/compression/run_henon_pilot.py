"""Run the fixed exploratory C1 Hénon candidate; no tuning or measured bytes."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import leenef

from . import series
from .baselines import AdaptiveCategorical, ARConfig, PrequentialAR
from .prequential import PredictorConfig, score_symbols
from .quantization import UniformQuantizer

_ROOT = Path(__file__).resolve().parents[2]
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_METHODS = ("c0_fixed_feature", "categorical_order0", "ridge_ar")


@dataclass(frozen=True)
class _Pilot:
    n: int = 512
    trajectory_seeds: tuple[int, ...] = (0, 1, 2)
    a: float = 1.4
    b: float = 0.3
    burn_in: int = 200
    noise_std: float = 0.005
    grid: UniformQuantizer = UniformQuantizer(-2.0, 2.0, 8)
    c0: PredictorConfig = PredictorConfig(
        window=2,
        width=16,
        warmup=32,
        resolve_every=8,
        fixed_alpha=0.1,
        noise_variance=0.01,
        seed=7,
        activation="abs",
        refresh_every=16,
    )
    ar: ARConfig = ARConfig(
        order=2, warmup=32, resolve_every=8, fixed_alpha=0.1, noise_variance=0.01
    )
    pseudocount: float = 0.5

    @property
    def suffix_start(self) -> int:
        return max(self.c0.window + self.c0.warmup, self.ar.order + self.ar.warmup)

    def observe(self, seed: int) -> Tensor:
        return series.henon(
            self.n, seed=seed, a=self.a, b=self.b, warmup=self.burn_in, noise_std=self.noise_std
        )


def _clean_identity(root: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=root, text=True)

    if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != root:
        raise RuntimeError("expected the repository root of this runner")
    if git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        raise RuntimeError("pilot requires a clean checkout with no untracked files")
    ignored = git(
        "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", "benchmarks", "src"
    )
    if any(Path(name).suffix in {".py", ".so", ".pyd"} for name in ignored.split("\0")):
        raise RuntimeError("pilot refuses ignored implementation files under benchmarks/ or src/")
    return git("rev-parse", "--verify", "HEAD^{commit}").strip()


def _write_atomic(path: Path, contents: bytes, directory_fd: int) -> None:
    pending = path.with_name(f".{path.name}.partial")
    pending.unlink(missing_ok=True)
    with pending.open("xb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    os.fsync(directory_fd)


def _bytes(values: Tensor, dtype: str) -> bytes:
    return values.numpy().astype(dtype, copy=False).tobytes(order="C")


def _artifact_names(seed: int) -> tuple[str, str, str]:
    prefix = f"henon_seed{seed}"
    return f"{prefix}_raw.f64le", f"{prefix}_reconstructed.f32le", f"{prefix}_symbols.u8"


def _read_artifact(path: Path, size: int) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        raise ValueError(f"expected a regular artifact of {size} bytes: {path}")
    return path.read_bytes()


def _artifacts(pilot: _Pilot, seed: int, raw: Tensor) -> tuple[Tensor, dict[str, bytes], dict]:
    if (
        raw.dtype != torch.float64
        or raw.device.type != "cpu"
        or raw.shape != (pilot.n,)
        or not raw.isfinite().all()
    ):
        raise ValueError("expected a finite CPU float64 trajectory of the configured length")
    raw_name, recon_name, symbol_name = _artifact_names(seed)
    symbols = pilot.grid.quantize(raw)
    contents = {
        raw_name: _bytes(raw, "<f8"),
        recon_name: _bytes(pilot.grid.reconstruct(symbols), "<f4"),
        symbol_name: _bytes(symbols, "u1"),
    }
    saturation = {
        "below_lo": int((raw < pilot.grid.lo).sum()),
        "above_hi": int((raw > pilot.grid.hi).sum()),
    }
    return symbols, contents, saturation


def _verify_artifacts(
    output: Path, contents: dict[str, bytes], *, directory_fd: int | None = None
) -> dict:
    hashes = {}
    for name, expected in contents.items():
        path = output / name
        if path.is_symlink():
            raise ValueError(f"artifact must not be a symlink: {path}")
        if directory_fd is not None and not path.exists():
            _write_atomic(path, expected, directory_fd)
        actual = _read_artifact(path, len(expected))
        if actual != expected:
            raise ValueError(f"artifact contents do not match: {path}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    raw_name, *processed = contents
    return {
        "raw_hashes": {raw_name: hashes[raw_name]},
        "processed_hashes": {name: hashes[name] for name in processed},
    }


def _score(pilot: _Pilot, symbols: Tensor) -> dict:
    scores = (
        score_symbols(symbols, pilot.grid, pilot.c0).bits,
        AdaptiveCategorical(pilot.grid, pseudocount=pilot.pseudocount).score_symbols(symbols),
        PrequentialAR(pilot.grid, pilot.ar).score_symbols(symbols).bits,
    )
    if any(
        bits.shape != symbols.shape or not (bits.isfinite() & (bits >= 0)).all() for bits in scores
    ):
        raise ValueError("expected finite nonnegative ideal bits for every symbol")
    return {
        name: {
            "all_stream_ideal_bits": float(bits.sum()),
            "common_suffix_ideal_bits": float(bits[pilot.suffix_start :].sum()),
        }
        for name, bits in zip(_METHODS, scores, strict=True)
    }


def _valid_scores(scores: dict) -> bool:
    if not isinstance(scores, dict) or set(scores) != set(_METHODS):
        return False
    for method in scores.values():
        if not isinstance(method, dict) or set(method) != {
            "all_stream_ideal_bits",
            "common_suffix_ideal_bits",
        }:
            return False
        total, suffix = method["all_stream_ideal_bits"], method["common_suffix_ideal_bits"]
        if not all(
            type(value) in (int, float) and math.isfinite(value) for value in (total, suffix)
        ):
            return False
        if not 0 <= suffix <= total:
            return False
    return True


def _run(output: Path, pilot: _Pilot, code_identity: str | None) -> None:
    """Execute incrementally; a null identity is reserved for synthetic unit tests."""
    if pilot.grid.n_bits != 8 or pilot.n <= pilot.suffix_start:
        raise ValueError("pilot artifacts require an 8-bit grid and a nonempty common suffix")
    header = {
        "protocol": "c1-henon-ideal-v1",
        "kind": "synthetic_smoke" if code_identity is None else "exploratory",
        "code_identity": code_identity,
        "config": asdict(pilot) | {"trajectory_seeds": list(pilot.trajectory_seeds)},
        "source": "synthetic_fixture"
        if code_identity is None
        else "benchmarks.compression.series.henon",
        "calibration": "none; grid fixed before observing data",
        "ridge_policy": "fixed",
        "common_suffix_start": pilot.suffix_start,
        "all_stream_symbols": pilot.n,
        "common_suffix_symbols": pilot.n - pilot.suffix_start,
        "libraries": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "leenef_source": tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"][
                "version"
            ],
        },
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "overheads_included": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _resume(output, fd, pilot, header)
    finally:
        os.close(fd)


def _resume(output: Path, fd: int, pilot: _Pilot, header: dict) -> None:
    manifest = output / ("smoke.json" if header["kind"] == "synthetic_smoke" else "pilot.json")

    def checkpoint() -> None:
        content = json.dumps(
            {"run": header, "rows": rows}, indent=2, sort_keys=True, allow_nan=False
        )
        _write_atomic(manifest, (content + "\n").encode(), fd)

    if manifest.is_symlink():
        raise ValueError("manifest must not be a symlink")
    if manifest.exists():
        result = json.loads(manifest.read_bytes())
        if result["run"] != header:
            raise ValueError(
                "pilot manifest does not match current config, code identity or environment"
            )
        rows = result["rows"]
        if not isinstance(rows, list) or len(rows) > len(pilot.trajectory_seeds):
            raise ValueError("invalid pilot checkpoint rows")
    else:
        if any(path.name != f".{manifest.name}.partial" for path in output.iterdir()):
            raise ValueError("use an empty output directory or one with a matching manifest")
        rows = []
        checkpoint()

    for seed, row in zip(pilot.trajectory_seeds[: len(rows)], rows, strict=True):
        if (
            row["trajectory_seed"] != seed
            or row["code_identity"] != header["code_identity"]
            or row["kind"] != header["kind"]
        ):
            raise ValueError("pilot checkpoint has unexpected seed, kind or code identity")
        raw_path = output / _artifact_names(seed)[0]
        raw_bytes = _read_artifact(raw_path, pilot.n * 8)
        raw = torch.from_numpy(np.frombuffer(raw_bytes, dtype="<f8").astype("f8"))
        _, contents, saturation = _artifacts(pilot, seed, raw)
        if _verify_artifacts(output, contents) != row["dataset_metadata"]:
            raise ValueError(f"artifact hashes do not match checkpoint for seed {seed}")
        if row["saturation"] != saturation or not _valid_scores(row["methods"]):
            raise ValueError(f"invalid pilot checkpoint accounting for seed {seed}")

    for seed in pilot.trajectory_seeds[len(rows) :]:
        symbols, contents, saturation = _artifacts(pilot, seed, pilot.observe(seed))
        dataset_metadata = _verify_artifacts(output, contents, directory_fd=fd)
        methods = _score(pilot, symbols)
        if not _valid_scores(methods):
            raise ValueError(f"invalid ideal bits for seed {seed}")
        rows.append(
            {
                "trajectory_seed": seed,
                "kind": header["kind"],
                "code_identity": header["code_identity"],
                "dataset_metadata": dataset_metadata,
                "saturation": saturation,
                "methods": methods,
            }
        )
        checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="dedicated output directory")
    args = parser.parse_args()
    if Path(leenef.__file__).resolve() != _ROOT / "src/leenef/__init__.py":
        raise RuntimeError('use PYTHONPATH="$PWD/src:$PWD" from the compression checkout')
    identity = _clean_identity(_ROOT)
    if any(os.environ.get(name) != "1" for name in _THREAD_VARS):
        raise RuntimeError("set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1")
    output = args.output.resolve()
    if output.is_relative_to(_ROOT):
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--", f"{output.relative_to(_ROOT)}/"),
            cwd=_ROOT,
            check=False,
        )
        if result.returncode == 1:
            raise ValueError("choose an output directory outside the checkout or ignored by Git")
        result.check_returncode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _run(output, _Pilot(), identity)
    print(f"Exploratory ideal-bit manifest: {output / 'pilot.json'}")


if __name__ == "__main__":
    main()
