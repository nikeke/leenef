"""Diagnostic GPU suite for the Paper-II H8 streaming continual boundary.

Runs targeted cells that explain why exact accumulate/joint solves stay
identical while online Woodbury lands much higher on 10-task permuted
sequential MNIST.  The suite uses the standard cl_clean phase runner and
bundles result JSON/log/config artifacts for download.
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
BUNDLE_PATH = RESULTS_DIR / "cl_clean_streaming_smnist_diagnostics_bundle.zip"

MNIST_URLS = {
    "train-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

DEFAULT_CELLS = [
    # Direct mechanism check: does exact inverse refresh collapse Woodbury?
    "refresh_n4000_a1e-3_t10",
    "refresh_n8000_a1e-3_t10",
    # Small alpha grid at the two widths that already showed the split.
    "exact_n4000_a1e-4_t10",
    "woodbury_n4000_a1e-4_t10",
    "exact_n4000_a1e-3_t10",
    "woodbury_n4000_a1e-3_t10",
    "exact_n4000_a1e-2_t10",
    "woodbury_n4000_a1e-2_t10",
    "exact_n8000_a1e-4_t10",
    "woodbury_n8000_a1e-4_t10",
    "exact_n8000_a1e-3_t10",
    "woodbury_n8000_a1e-3_t10",
    "exact_n8000_a1e-2_t10",
    "woodbury_n8000_a1e-2_t10",
    # Task-count decomposition: streaming feature capacity vs permutation load.
    "exact_n4000_a1e-3_t1",
    "woodbury_n4000_a1e-3_t1",
    "exact_n4000_a1e-3_t2",
    "woodbury_n4000_a1e-3_t2",
    "exact_n4000_a1e-3_t5",
    "woodbury_n4000_a1e-3_t5",
    # Center/bias causality for the exact solve.
    "exact_center_first_n4000_a1e-3_t10",
    "exact_center_all_n4000_a1e-3_t10",
    "exact_center_zero_n4000_a1e-3_t10",
    # Path-dependence probes for the online inverse path.
    "woodbury_bs128_n4000_a1e-3_t10",
    "woodbury_bs512_n4000_a1e-3_t10",
    "woodbury_reverse_n4000_a1e-3_t10",
]

EXTENDED_CELLS = [
    "refresh_n16000_a1e-3_t10",
    "exact_n16000_a1e-4_t10",
    "woodbury_n16000_a1e-4_t10",
    "exact_n16000_a1e-3_t10",
    "woodbury_n16000_a1e-3_t10",
    "exact_n16000_a1e-2_t10",
    "woodbury_n16000_a1e-2_t10",
]


def _verify_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this suite. In Colab, choose Runtime -> "
            "Change runtime type -> Hardware accelerator -> GPU (G4 preferred, L4 fallback)."
        )
    props = torch.cuda.get_device_properties(0)
    print(
        f"[h8-diagnostics] CUDA: {props.name}, "
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
            print(f"[h8-diagnostics] downloading {name}.gz", flush=True)
            urllib.request.urlretrieve(url, gz_path)
        print(f"[h8-diagnostics] extracting {name}.gz", flush=True)
        with gzip.open(gz_path, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _write_gpu_config() -> Path:
    src = RESULTS_DIR / "base_config_h8_streaming_diagnostics.json"
    if not src.exists():
        raise FileNotFoundError(f"base config not found: {src}")
    cfg = json.loads(src.read_text())
    cfg["device"] = "cuda"
    GPU_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    dst = GPU_CONFIG_DIR / "base_config_gpu_h8_streaming_diagnostics.json"
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
        "h8_streaming_permuted_smnist_diag",
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
    print(f"[h8-diagnostics] running {phase}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out


def _summarize(path: Path) -> None:
    rows = json.loads(path.read_text())
    grouped: dict[str, list[dict]] = {}
    for row in rows.values():
        if "error" in row:
            print(f"  ERROR {row.get('label', '?')}: {row['error']}")
            continue
        grouped.setdefault(str(row.get("value")), []).append(row["metrics"])

    print(f"\n[h8-diagnostics] {path.name}: {len(rows)} rows")
    for value, vals in sorted(grouped.items()):
        accs = [float(v["test_accuracy"]) for v in vals]
        mean = statistics.mean(accs) * 100
        se = statistics.stdev(accs) / (len(accs) ** 0.5) * 100 if len(accs) > 1 else 0.0
        msg = f"  {value:34s} acc={mean:.2f}±{se:.2f}% n={len(accs)}"
        if "refresh_test_accuracy" in vals[0]:
            refresh = statistics.mean(float(v["refresh_test_accuracy"]) for v in vals) * 100
            gap = statistics.mean(float(v["refresh_gap_pp"]) for v in vals)
            msg += f" refresh={refresh:.2f}% gap={gap:+.2f}pp"
        if "train_subset_accuracy" in vals[0]:
            train = statistics.mean(float(v["train_subset_accuracy"]) for v in vals) * 100
            msg += f" train_subset={train:.2f}%"
        print(msg)


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
        return "refresh_n500_a1e-3_t2"
    values = list(DEFAULT_CELLS)
    if extended:
        values.extend(EXTENDED_CELLS)
    return ",".join(values)


def run_suite(seeds: str, *, quick: bool, extended: bool) -> tuple[list[Path], Path]:
    from benchmarks.cl_clean.cl_repro import code_identity
    from benchmarks.cl_clean.experiments import names

    if "h8_streaming_permuted_smnist_diag" not in set(names()):
        raise SystemExit("This checkout does not contain h8_streaming_permuted_smnist_diag.")

    print(f"[h8-diagnostics] code_identity={code_identity()}", flush=True)
    cfg = _write_gpu_config()
    phase = "gpu_h8_streaming_diagnostics_quick" if quick else "gpu_h8_streaming_diagnostics"
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
    parser.add_argument(
        "--seeds",
        default="0-1",
        help="Seed list/range passed to cl_phase_runner; default is diagnostic n=2",
    )
    parser.add_argument("--quick", action="store_true", help="Run a one-cell validation")
    parser.add_argument("--extended", action="store_true", help="Also run 16k-neuron cells")
    args = parser.parse_args(argv)

    _verify_cuda()
    _ensure_mnist_raw()
    outputs, bundle = run_suite(args.seeds, quick=args.quick, extended=args.extended)
    for path in outputs:
        _summarize(path)
    print("\n[h8-diagnostics] result files:")
    for path in outputs:
        print(f"  {path}")
    print(f"[h8-diagnostics] bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
