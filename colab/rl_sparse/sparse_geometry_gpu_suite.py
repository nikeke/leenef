"""Colab GPU suite for successor-readout geometry follow-up rows."""

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
RESULTS_DIR = REPO_ROOT / "results" / "rl_sparse" / "geometry_gpu"
LOGS_DIR = RESULTS_DIR / "logs"
BUNDLE_PATH = RESULTS_DIR / "rl_sparse_geometry_gpu_bundle.zip"
READOUT_MODES = ("dot", "cosine", "future_cosine")


@dataclass(frozen=True)
class GeometryCell:
    mode: str
    width: int
    episodes: int
    eval_every: int = 25
    eval_episodes: int = 5
    warmup_episodes: int = 10
    goal_horizon: int = 25
    forget_factor: float = 0.999

    @property
    def name(self) -> str:
        return f"mountaincar_{self.mode}_n{self.width}"


def _verify_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this suite. In Colab, choose Runtime -> "
            "Change runtime type -> Hardware accelerator -> GPU (G4 preferred, L4 fallback)."
        )
    props = torch.cuda.get_device_properties(0)
    print(
        f"[rl-sparse-geometry] CUDA: {props.name}, "
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
    *, quick: bool, width: int, episodes: int | None, modes: list[str]
) -> list[GeometryCell]:
    if quick:
        return [
            GeometryCell(
                mode=modes[0], width=512, episodes=episodes or 5, eval_every=1, eval_episodes=1
            )
        ]
    return [GeometryCell(mode=mode, width=width, episodes=episodes or 300) for mode in modes]


def _run_cell(cell: GeometryCell, *, seeds: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{cell.name}.json"
    log = LOGS_DIR / f"{cell.name}.log"
    cmd = [
        sys.executable,
        "-u",
        "benchmarks/run_rl_sparse.py",
        "--env",
        "MountainCar-v0",
        "--episodes",
        str(cell.episodes),
        "--eval-every",
        str(cell.eval_every),
        "--eval-episodes",
        str(cell.eval_episodes),
        "--seeds",
        *_seed_args(seeds),
        "--variants",
        "nef_sf_hindsight",
        "--warmup-episodes",
        str(cell.warmup_episodes),
        "--goal-horizon",
        str(cell.goal_horizon),
        "--goal-mode",
        "random_direction",
        "--goal-action-mode",
        "commit",
        "--eval-goal=0.5,0.0",
        "--goal-readout-mode",
        cell.mode,
        "--config-override",
        f"n_neurons={cell.width}",
        "--config-override",
        f"forget_factor={cell.forget_factor}",
        "--device",
        "cuda",
        "--skip-agent-diagnostics",
        "--output",
        str(out),
    ]
    print(f"[rl-sparse-geometry] running {cell.name}: {' '.join(cmd)}", flush=True)
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
        print(f"[rl-sparse-geometry] missing result file: {path}")
        return
    data = json.loads(path.read_text())
    runs = data.get("runs", []) if isinstance(data, dict) else []
    vals = [float(r["robust_eval"]) for r in runs if r.get("robust_eval") is not None]
    finals = [float(r["final_eval"]) for r in runs if r.get("final_eval") is not None]
    times = [float(r["time_s"]) for r in runs if r.get("time_s") is not None]
    errors = [r for r in runs if "error" in r]
    print(f"\n[rl-sparse-geometry] {path.name}: rows={len(runs)} errors={len(errors)}")
    if vals:
        se = statistics.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
        fse = statistics.stdev(finals) / (len(finals) ** 0.5) if len(finals) > 1 else 0.0
        mean_time = statistics.mean(times) if times else float("nan")
        print(
            f"  robust={statistics.mean(vals):.1f} +/- {se:.1f} "
            f"final={statistics.mean(finals):.1f} +/- {fse:.1f} "
            f"mean_row_s={mean_time:.1f} n={len(vals)}"
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
    modes: list[str],
) -> tuple[list[Path], Path]:
    from benchmarks.rl_repro import code_version

    if not (REPO_ROOT / "benchmarks" / "run_rl_sparse.py").exists():
        raise SystemExit("This checkout does not contain benchmarks/run_rl_sparse.py")
    print(f"[rl-sparse-geometry] code_version={code_version()}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        _run_cell(cell, seeds=seeds)
        for cell in _cells(quick=quick, width=width, episodes=episodes, modes=modes)
    ]
    bundle = _bundle(outputs)
    return outputs, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-9", help="Seed list/range, e.g. 0-9 or 0,1,2")
    parser.add_argument("--width", type=int, default=2000, help="NEF neuron count")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count")
    parser.add_argument("--quick", action="store_true", help="Run a short validation cell")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=READOUT_MODES,
        default=list(READOUT_MODES),
        help="Readout geometry modes to run.",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove previous geometry GPU outputs first"
    )
    args = parser.parse_args(argv)

    if args.clean and RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    _verify_cuda()
    outputs, bundle = run_suite(
        seeds=args.seeds,
        width=args.width,
        episodes=args.episodes,
        quick=args.quick,
        modes=list(args.modes),
    )
    for path in outputs:
        _summarize(path)
    print("\n[rl-sparse-geometry] result files:")
    for path in outputs:
        print(f"  {path}")
    print(f"[rl-sparse-geometry] bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
