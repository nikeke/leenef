"""GPU suite for the Paper-II ConvNEF continual-learning panel.

Run from a CUDA Colab runtime on ``paper-ii/cl-scaffold`` (or a pushed
ref that contains the same committed files).  The suite uses committed
``benchmarks.cl_clean`` experiments and the standard phase runner; the
Colab wrapper only rewrites configs, launches phases, and bundles the
artifacts for download.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "cl_clean"
LOGS_DIR = RESULTS_DIR / "logs"
GPU_CONFIG_DIR = RESULTS_DIR / "colab_gpu_configs"
BUNDLE_PATH = RESULTS_DIR / "cl_clean_phase_c_gpu_bundle.zip"

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _verify_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this suite. In Colab, choose "
            "Runtime -> Change runtime type -> Hardware accelerator -> GPU "
            "(G4 preferred, L4 fallback)."
        )
    props = torch.cuda.get_device_properties(0)
    print(
        f"[gpu-suite] CUDA: {props.name}, "
        f"{props.total_memory / 1024**3:.1f} GiB, "
        f"capability {props.major}.{props.minor}",
        flush=True,
    )


def _ensure_cifar10_raw() -> None:
    raw_dir = DATA_DIR / "cifar-10-batches-py"
    if raw_dir.exists():
        print(f"[gpu-suite] CIFAR-10 raw files already present at {raw_dir}", flush=True)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "cifar-10-python.tar.gz"
    if not archive.exists():
        print(f"[gpu-suite] downloading CIFAR-10 archive to {archive}", flush=True)
        urllib.request.urlretrieve(CIFAR10_URL, archive)

    print(f"[gpu-suite] extracting {archive.name}", flush=True)
    with tarfile.open(archive, "r:gz") as tf:
        try:
            tf.extractall(DATA_DIR, filter="data")
        except TypeError:
            tf.extractall(DATA_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError(f"expected extracted CIFAR-10 directory at {raw_dir}")


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
        "benchmarks.cl_clean.cl_phase_runner",
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


def _bundle_artifacts(outputs: list[Path], config_paths: list[Path]) -> Path:
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in outputs + config_paths:
            if path.exists():
                zf.write(path, arcname=path.relative_to(REPO_ROOT))
        for out in outputs:
            log = LOGS_DIR / f"{out.stem}.log"
            if log.exists():
                zf.write(log, arcname=log.relative_to(REPO_ROOT))
    return BUNDLE_PATH


def run_suite(seeds: str, *, quick: bool) -> tuple[list[Path], Path]:
    from benchmarks.cl_clean.cl_repro import code_identity
    from benchmarks.cl_clean.experiments import names

    required = {"c1_convnef_split_cifar10", "nef_continual_split_cifar10"}
    missing = required - set(names())
    if missing:
        raise SystemExit(
            "This checkout does not contain the Phase-C CL experiments: "
            f"{sorted(missing)}. Update to a revision that includes the "
            "ConvNEF continual-learning scaffold."
        )

    print(f"[gpu-suite] code_identity={code_identity()}", flush=True)
    outputs: list[Path] = []
    configs: list[Path] = []

    c1_cfg = _write_gpu_config(
        "base_config_c1_convnef_split_cifar10.json",
        "base_config_gpu_c1_convnef_split_cifar10.json",
    )
    configs.append(c1_cfg)

    if quick:
        outputs.append(
            _run_phase(
                experiment="c1_convnef_split_cifar10",
                phase="gpu_c1_convnef_split_cifar10_quick",
                knob="cell",
                values="all_data_n2000",
                seeds=seeds,
                base_config=c1_cfg,
            )
        )
        bundle = _bundle_artifacts(outputs, configs)
        return outputs, bundle

    outputs.append(
        _run_phase(
            experiment="c1_convnef_split_cifar10",
            phase="gpu_c1_convnef_split_cifar10_panel",
            knob="cell",
            values="all_data_n10000,first_task_n10000,joint_n10000,"
            "all_data_n20000,first_task_n20000,joint_n20000",
            seeds=seeds,
            base_config=c1_cfg,
        )
    )

    c2_cfg = _write_gpu_config(
        "base_config_nef_continual_split_cifar10.json",
        "base_config_gpu_c2_flat_nef_split_cifar10.json",
    )
    configs.append(c2_cfg)
    outputs.append(
        _run_phase(
            experiment="nef_continual_split_cifar10",
            phase="gpu_c2_flat_nef_split_cifar10_panel",
            knob="cell",
            values="nef_accumulate",
            seeds=seeds,
            base_config=c2_cfg,
        )
    )

    bundle = _bundle_artifacts(outputs, configs)
    return outputs, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-9", help="Seed list/range passed to cl_phase_runner")
    parser.add_argument("--quick", action="store_true", help="Run a small one-cell validation")
    args = parser.parse_args(argv)

    _verify_cuda()
    _ensure_cifar10_raw()
    outputs, bundle = run_suite(args.seeds, quick=args.quick)
    for path in outputs:
        _summarize(path)
    print("\n[gpu-suite] result files:")
    for path in outputs:
        print(f"  {path}")
    print(f"[gpu-suite] bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
