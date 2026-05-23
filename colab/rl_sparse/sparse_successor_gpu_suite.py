"""Colab GPU suite for Paper III sparse-reward successor-feature panels."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "rl_sparse" / "gpu"
LOGS_DIR = RESULTS_DIR / "logs"
BUNDLE_PATH = RESULTS_DIR / "rl_sparse_gpu_bundle.zip"


@dataclass(frozen=True)
class SparseCell:
    name: str
    env: str
    eval_goal: str
    warmup_episodes: int
    goal_horizon: int
    forget_factor: float
    episodes: int
    eval_every: int = 25
    eval_episodes: int = 5


def _verify_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this suite. In Colab, choose Runtime -> "
            "Change runtime type -> Hardware accelerator -> GPU (G4 preferred, L4 fallback)."
        )
    props = torch.cuda.get_device_properties(0)
    print(
        f"[rl-sparse-suite] CUDA: {props.name}, "
        f"{props.total_memory / 1024**3:.1f} GiB, capability {props.major}.{props.minor}",
        flush=True,
    )


def _seed_args(seeds: str) -> list[str]:
    values: list[int] = []
    for part in seeds.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("at least one seed is required")
    return [str(v) for v in values]


def _cells(
    *, quick: bool, width: int, episodes: int | None, include_scan: bool
) -> list[SparseCell]:
    if quick:
        return [
            SparseCell(
                name=f"quick_mountaincar_directional_n{width}",
                env="MountainCar-v0",
                eval_goal="0.5,0.0",
                warmup_episodes=2,
                goal_horizon=10,
                forget_factor=0.999,
                episodes=episodes or 5,
                eval_every=1,
                eval_episodes=1,
            )
        ]

    cells = [
        SparseCell(
            name=f"mountaincar_directional_n{width}",
            env="MountainCar-v0",
            eval_goal="0.5,0.0",
            warmup_episodes=10,
            goal_horizon=25,
            forget_factor=0.999,
            episodes=episodes or 300,
        ),
        SparseCell(
            name=f"acrobot_directional_ff9999_n{width}",
            env="Acrobot-v1",
            eval_goal="-1,0,1,0,0,0",
            warmup_episodes=20,
            goal_horizon=50,
            forget_factor=0.9999,
            episodes=episodes or 300,
        ),
    ]
    if include_scan:
        for ff in (0.999, 0.9995, 0.9999):
            label = str(ff).replace(".", "p")
            cells.append(
                SparseCell(
                    name=f"acrobot_directional_ff{label}_scan_n{width}",
                    env="Acrobot-v1",
                    eval_goal="-1,0,1,0,0,0",
                    warmup_episodes=20,
                    goal_horizon=50,
                    forget_factor=ff,
                    episodes=episodes or 300,
                )
            )
    return cells


def _run_cell(cell: SparseCell, *, seeds: str, width: int) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{cell.name}.json"
    log = LOGS_DIR / f"{cell.name}.log"
    cmd = [
        sys.executable,
        "-u",
        "benchmarks/run_rl_sparse.py",
        "--env",
        cell.env,
        "--episodes",
        str(cell.episodes),
        "--eval-every",
        str(cell.eval_every),
        "--eval-episodes",
        str(cell.eval_episodes),
        "--seeds",
        *_seed_args(seeds),
        "--variants",
        "nef_sf_hindsight_directional",
        "--warmup-episodes",
        str(cell.warmup_episodes),
        "--goal-horizon",
        str(cell.goal_horizon),
        "--goal-mode",
        "eval_goal",
        "--goal-action-mode",
        "commit",
        "--eval-goal",
        cell.eval_goal,
        "--config-override",
        f"n_neurons={width}",
        "--config-override",
        f"forget_factor={cell.forget_factor}",
        "--device",
        "cuda",
        "--skip-agent-diagnostics",
        "--output",
        str(out),
    ]
    print(f"[rl-sparse-suite] running {cell.name}: {' '.join(cmd)}", flush=True)
    with log.open("w") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return out


def _summarize(path: Path) -> None:
    if not path.exists():
        print(f"[rl-sparse-suite] missing result file: {path}")
        return
    data = json.loads(path.read_text())
    runs = data.get("runs", []) if isinstance(data, dict) else []
    vals = [float(r["robust_eval"]) for r in runs if r.get("robust_eval") is not None]
    finals = [float(r["final_eval"]) for r in runs if r.get("final_eval") is not None]
    errors = [r for r in runs if "error" in r]
    print(f"\n[rl-sparse-suite] {path.name}: rows={len(runs)} errors={len(errors)}")
    if vals:
        se = statistics.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
        fse = statistics.stdev(finals) / (len(finals) ** 0.5) if len(finals) > 1 else 0.0
        print(
            f"  robust={statistics.mean(vals):.1f} +/- {se:.1f} "
            f"final={statistics.mean(finals):.1f} +/- {fse:.1f} n={len(vals)}"
        )
    for row in errors:
        print(f"  ERROR seed={row.get('seed')}: {row.get('error')}")


def _bundle(outputs: list[Path]) -> Path:
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in outputs:
            if path.exists():
                zf.write(path, arcname=path.relative_to(REPO_ROOT))
            log = LOGS_DIR / f"{path.stem}.log"
            if log.exists():
                zf.write(log, arcname=log.relative_to(REPO_ROOT))
    return BUNDLE_PATH


def run_suite(
    *,
    seeds: str,
    width: int,
    episodes: int | None,
    quick: bool,
    include_scan: bool,
) -> tuple[list[Path], Path]:
    from benchmarks.rl_repro import code_version

    if not (REPO_ROOT / "benchmarks" / "run_rl_sparse.py").exists():
        raise SystemExit("This checkout does not contain benchmarks/run_rl_sparse.py")
    print(f"[rl-sparse-suite] code_version={code_version()}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        _run_cell(cell, seeds=seeds, width=width)
        for cell in _cells(quick=quick, width=width, episodes=episodes, include_scan=include_scan)
    ]
    bundle = _bundle(outputs)
    return outputs, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-9", help="Seed list/range, e.g. 0-9 or 0,1,2")
    parser.add_argument("--width", type=int, default=6000, help="NEF neuron count")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count")
    parser.add_argument("--quick", action="store_true", help="Run a short validation cell")
    parser.add_argument(
        "--include-scan", action="store_true", help="Add Acrobot forget-factor scan"
    )
    parser.add_argument("--clean", action="store_true", help="Remove previous GPU outputs first")
    args = parser.parse_args(argv)

    if args.clean and RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    _verify_cuda()
    outputs, bundle = run_suite(
        seeds=args.seeds,
        width=args.width,
        episodes=args.episodes,
        quick=args.quick,
        include_scan=args.include_scan,
    )
    for path in outputs:
        _summarize(path)
    print("\n[rl-sparse-suite] result files:")
    for path in outputs:
        print(f"  {path}")
    print(f"[rl-sparse-suite] bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
