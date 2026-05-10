"""GPU suite for the Paper-II streaming continual-learning follow-up.

Runs the H8-style StreamingNEFClassifier panel on a 10-task permuted
sequential-MNIST stream.  The suite uses the standard cl_clean phase
runner and bundles result JSON/log/config artifacts for download.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import statistics
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "cl_clean"
LOGS_DIR = RESULTS_DIR / "logs"
GPU_CONFIG_DIR = RESULTS_DIR / "colab_gpu_configs"
BUNDLE_PATH = RESULTS_DIR / "cl_clean_streaming_smnist_gpu_bundle.zip"

MNIST_URLS = {
    "train-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _verify_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this suite. In Colab, choose Runtime -> "
            "Change runtime type -> Hardware accelerator -> GPU (G4 preferred, L4 fallback)."
        )
    props = torch.cuda.get_device_properties(0)
    print(
        f"[streaming-suite] CUDA: {props.name}, "
        f"{props.total_memory / 1024**3:.1f} GiB, capability {props.major}.{props.minor}",
        flush=True,
    )


def _ensure_mnist_raw() -> None:
    raw_dir = DATA_DIR / "MNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        out = raw_dir / name
        if out.exists():
            continue
        gz_path = raw_dir / f"{name}.gz"
        if not gz_path.exists():
            print(f"[streaming-suite] downloading {name}.gz", flush=True)
            urllib.request.urlretrieve(url, gz_path)
        print(f"[streaming-suite] extracting {name}.gz", flush=True)
        with gzip.open(gz_path, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def _write_gpu_config(*, alpha: float, alpha_scale: str, stats_dtype: str) -> Path:
    src = RESULTS_DIR / "base_config_h8_streaming_permuted_smnist.json"
    if not src.exists():
        raise FileNotFoundError(f"base config not found: {src}")
    cfg = json.loads(src.read_text())
    cfg["alpha"] = alpha
    cfg["alpha_scale"] = alpha_scale
    cfg["stats_dtype"] = stats_dtype
    cfg["device"] = "cuda"
    GPU_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (
        "gpu_h8_streaming_permuted_smnist"
        if alpha_scale == "fixed"
        else f"gpu_h8_streaming_trace_a{_alpha_label(alpha)}_{stats_dtype}"
    )
    dst = GPU_CONFIG_DIR / f"base_config_{suffix}.json"
    dst.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return dst


def _run_phase(*, phase: str, values: str, seeds: str, base_config: Path) -> Path:
    out = RESULTS_DIR / f"{phase}.json"
    log = LOGS_DIR / f"{phase}.log"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "benchmarks.cl_clean.cl_phase_runner",
        "--experiment",
        "h8_streaming_permuted_smnist",
        "--phase",
        phase,
        "--knob",
        "cell",
        "--values",
        values,
        "--seeds",
        seeds,
        "--base-config",
        str(base_config),
        "--out",
        str(out),
        "--log",
        str(log),
    ]
    print(f"[streaming-suite] running {phase}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out


def _summarize(path: Path) -> None:
    if not path.exists():
        print(f"[streaming-suite] missing result file: {path}")
        return
    rows = json.loads(path.read_text())
    grouped: dict[str, list[tuple[float, float | None]]] = {}
    for row in rows.values():
        if "error" in row:
            print(f"  ERROR {row.get('label', '?')}: {row['error']}")
            continue
        metrics = row.get("metrics", {})
        acc = metrics.get("test_accuracy")
        if acc is None:
            continue
        grouped.setdefault(str(row.get("value")), []).append((float(acc), row.get("wall_time_s")))

    print(f"\n[streaming-suite] {path.name}: {len(rows)} rows")
    for value, vals in sorted(grouped.items()):
        accs = [v[0] for v in vals]
        walls = [v[1] for v in vals if v[1] is not None]
        mean = statistics.mean(accs) * 100
        se = statistics.stdev(accs) / (len(accs) ** 0.5) * 100 if len(accs) > 1 else 0.0
        wall = statistics.mean(walls) if walls else float("nan")
        print(f"  {value:32s} acc={mean:.2f}±{se:.2f}% n={len(accs)} wall={wall:.1f}s")


def _bundle_artifacts(outputs: list[Path], configs: list[Path]) -> Path:
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in outputs + configs:
            if path.exists():
                zf.write(path, arcname=path.relative_to(REPO_ROOT))
        for out in outputs:
            log = LOGS_DIR / f"{out.stem}.log"
            if log.exists():
                zf.write(log, arcname=log.relative_to(REPO_ROOT))
    return BUNDLE_PATH


def _values(*, quick: bool, extended: bool) -> str:
    if quick:
        return "accumulate_final_n500"
    values = [
        "accumulate_final_n4000",
        "accumulate_task_solve_n4000",
        "woodbury_online_n4000",
        "joint_n4000",
        "reset_n4000",
        "accumulate_final_n8000",
        "woodbury_online_n8000",
        "joint_n8000",
        "reset_n8000",
    ]
    if extended:
        values.extend(["accumulate_final_n16000", "joint_n16000", "woodbury_online_n16000"])
    return ",".join(values)


def run_suite(
    seeds: str,
    *,
    quick: bool,
    extended: bool,
    alpha: float,
    alpha_scale: str,
    stats_dtype: str,
) -> tuple[list[Path], Path]:
    from benchmarks.cl_clean.cl_repro import code_identity
    from benchmarks.cl_clean.experiments import names

    if "h8_streaming_permuted_smnist" not in set(names()):
        raise SystemExit("This checkout does not contain h8_streaming_permuted_smnist.")

    print(f"[streaming-suite] code_identity={code_identity()}", flush=True)
    cfg = _write_gpu_config(alpha=alpha, alpha_scale=alpha_scale, stats_dtype=stats_dtype)
    if quick:
        phase = (
            "gpu_h8_streaming_trace_quick"
            if alpha_scale == "trace"
            else "gpu_h8_streaming_permuted_smnist_quick"
        )
    elif alpha_scale == "trace":
        phase = f"gpu_h8_streaming_trace_a{_alpha_label(alpha)}_{stats_dtype}_panel"
    else:
        phase = "gpu_h8_streaming_permuted_smnist_panel"
    outputs = [
        _run_phase(
            phase=phase,
            values=_values(quick=quick, extended=extended),
            seeds=seeds,
            base_config=cfg,
        )
    ]
    bundle = _bundle_artifacts(outputs, [cfg])
    return outputs, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-4", help="Seed list/range passed to cl_phase_runner")
    parser.add_argument("--quick", action="store_true", help="Run a one-cell validation")
    parser.add_argument("--extended", action="store_true", help="Also run 16k-neuron cells")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Nominal alpha. With --alpha-scale trace this is multiplied by trace(A^T A)/n.",
    )
    parser.add_argument(
        "--alpha-scale",
        choices=("fixed", "trace"),
        default="trace",
        help="Use fixed alpha or Paper-I-style trace-scaled alpha.",
    )
    parser.add_argument(
        "--stats-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Dtype for normal-equation GEMMs before float64 accumulation.",
    )
    args = parser.parse_args(argv)

    _verify_cuda()
    _ensure_mnist_raw()
    outputs, bundle = run_suite(
        args.seeds,
        quick=args.quick,
        extended=args.extended,
        alpha=args.alpha,
        alpha_scale=args.alpha_scale,
        stats_dtype=args.stats_dtype,
    )
    for path in outputs:
        _summarize(path)
    print("\n[streaming-suite] result files:")
    for path in outputs:
        print(f"  {path}")
    print(f"[streaming-suite] bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
