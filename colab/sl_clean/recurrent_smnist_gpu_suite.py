"""GPU suite for the recurrent sMNIST comparison panel.

Run from a CUDA Colab runtime on `main`. The suite intentionally uses
committed benchmark code and the standard `sl_phase_runner`; no runtime
monkey-patching is involved.

Default panels:

* C5 recurrent NEF (`end_to_end`, `target_prop`) on row sMNIST.
* Stream-NEF row sMNIST at `n={4000,8000}`.
* Best-accuracy CPU-tuned LSTM row-sMNIST cell (`h256_l2_lr0.003`)
  and fast near-best LSTM cell (`h64_l2_lr0.003`).
* Best CPU-tuned ESN row-sMNIST cell (`n2000_sr0.95_leak0.3`).

All four panels write separate JSON files with `"device": "cuda"` in
their configs, so GPU rows cannot be confused with the CPU panels.
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
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "sl_clean"
LOGS_DIR = RESULTS_DIR / "logs"
GPU_CONFIG_DIR = RESULTS_DIR / "colab_gpu_configs"

MNIST_URLS = {
    "train-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _verify_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this suite. In Colab, choose "
            "Runtime -> Change runtime type -> Hardware accelerator -> GPU "
            "(L4 recommended)."
        )
    props = torch.cuda.get_device_properties(0)
    print(
        f"[gpu-suite] CUDA: {props.name}, "
        f"{props.total_memory / 1024**3:.1f} GiB, "
        f"capability {props.major}.{props.minor}",
        flush=True,
    )


def _ensure_mnist_raw() -> None:
    """Populate data/MNIST/raw with the uncompressed IDX files our loader hashes."""
    raw_dir = DATA_DIR / "MNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in MNIST_URLS if not (raw_dir / name).exists()]
    if not missing:
        print(f"[gpu-suite] MNIST raw files already present at {raw_dir}", flush=True)
        return

    print(f"[gpu-suite] downloading MNIST raw IDX files to {raw_dir}", flush=True)
    for name in missing:
        url = MNIST_URLS[name]
        gz_path = raw_dir / f"{name}.gz"
        print(f"[gpu-suite]   {name}", flush=True)
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as src, open(raw_dir / name, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _write_gpu_config(src_name: str, dst_name: str) -> Path:
    src = RESULTS_DIR / src_name
    if not src.exists():
        raise FileNotFoundError(f"base config not found: {src}")
    cfg = json.loads(src.read_text())
    cfg["device"] = "cuda"
    GPU_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    dst = GPU_CONFIG_DIR / dst_name
    dst.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return dst


def _run_phase(
    *,
    experiment: str,
    phase: str,
    knob: str,
    values: str,
    seeds: str,
    base_config: Path,
) -> Path:
    out = RESULTS_DIR / f"{phase}.json"
    log = LOGS_DIR / f"{phase}.log"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "benchmarks.sl_clean.sl_phase_runner",
        "--experiment",
        experiment,
        "--phase",
        phase,
        "--knob",
        knob,
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
    print(f"[gpu-suite] running {phase}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out


def _summarize(path: Path) -> None:
    if not path.exists():
        print(f"[gpu-suite] missing result file: {path}")
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
        wall = row.get("wall_time_s")
        grouped.setdefault(str(row.get("value")), []).append((float(acc), wall))

    print(f"\n[gpu-suite] {path.name}: {len(rows)} rows")
    for value, vals in sorted(grouped.items()):
        accs = [v[0] for v in vals]
        walls = [v[1] for v in vals if v[1] is not None]
        mean = statistics.mean(accs) * 100
        se = statistics.stdev(accs) / (len(accs) ** 0.5) * 100 if len(accs) > 1 else 0.0
        wall = statistics.mean(walls) if walls else float("nan")
        print(f"  {value:24s} acc={mean:.2f}±{se:.2f}% n={len(accs)} wall={wall:.1f}s")


def run_suite(seeds: str) -> list[Path]:
    from benchmarks.sl_clean.experiments import names
    from benchmarks.sl_clean.sl_repro import code_identity

    required = {
        "c5_recurrent_smnist",
        "streaming_smnist_row",
        "b5_lstm_smnist_row",
        "b6_esn_smnist_row",
    }
    missing = required - set(names())
    if missing:
        raise SystemExit(
            "This checkout does not contain the recurrent GPU suite experiments: "
            f"{sorted(missing)}. Update this checkout to a main revision that "
            "includes the GPU-suite merge."
        )

    print(f"[gpu-suite] code_identity={code_identity()}", flush=True)
    outputs: list[Path] = []

    c5_cfg = _write_gpu_config(
        "base_config_c5_recurrent_smnist.json",
        "base_config_gpu_c5_recurrent_smnist.json",
    )
    outputs.append(
        _run_phase(
            experiment="c5_recurrent_smnist",
            phase="gpu_c5_recurrent_smnist_panel",
            knob="cell",
            values="end_to_end,target_prop",
            seeds=seeds,
            base_config=c5_cfg,
        )
    )

    streaming_cfg = _write_gpu_config(
        "base_config_streaming_smnist_row.json",
        "base_config_gpu_streaming_smnist_row.json",
    )
    outputs.append(
        _run_phase(
            experiment="streaming_smnist_row",
            phase="gpu_streaming_smnist_row_panel",
            knob="n_neurons",
            values="4000,8000",
            seeds=seeds,
            base_config=streaming_cfg,
        )
    )

    lstm_cfg = _write_gpu_config(
        "base_config_b5_lstm_smnist_row.json",
        "base_config_gpu_b5_lstm_smnist_row.json",
    )
    outputs.append(
        _run_phase(
            experiment="b5_lstm_smnist_row",
            phase="gpu_b5_lstm_smnist_row_pareto_panel",
            knob="cell",
            values="h256_l2_lr0.003,h64_l2_lr0.003",
            seeds=seeds,
            base_config=lstm_cfg,
        )
    )

    esn_cfg = _write_gpu_config(
        "base_config_b6_esn_smnist_row.json",
        "base_config_gpu_b6_esn_smnist_row.json",
    )
    outputs.append(
        _run_phase(
            experiment="b6_esn_smnist_row",
            phase="gpu_b6_esn_smnist_row_best_panel",
            knob="cell",
            values="n2000_sr0.95_leak0.3",
            seeds=seeds,
            base_config=esn_cfg,
        )
    )

    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-4", help="Seed list/range passed to sl_phase_runner")
    args = parser.parse_args(argv)

    _verify_cuda()
    _ensure_mnist_raw()
    outputs = run_suite(args.seeds)
    for path in outputs:
        _summarize(path)
    print("\n[gpu-suite] result files:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
