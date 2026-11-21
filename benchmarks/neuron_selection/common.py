"""Shared protocol, evaluation, and result recording for neuron selection."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import leenef
from benchmarks.sl_clean.datasets import DatasetBundle, load_dataset
from benchmarks.sl_clean.sl_repro import REPO_ROOT, code_identity, result_metadata, set_sl_seed


def verify_panel_source() -> None:
    filename = leenef.__file__
    imported = Path(filename).resolve() if filename else None
    expected = (REPO_ROOT / "src").resolve()
    if imported is None or not imported.is_relative_to(expected):
        raise RuntimeError(f"leenef imported from {imported}; expected a path under {expected}")


verify_panel_source()

ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2)
LOOP_ALPHA = (1e-3, "trace")
IMAGE_SHAPES = {"MNIST": (28, 28), "FashionMNIST": (28, 28), "CIFAR10": (32, 32)}
RESULTS = REPO_ROOT / "results" / "neuron_selection"
PATCH_SIZE = 5

# Role IDs are permanent; keyed streams are independent across roles and generations.
STREAMS = {
    "split": "validation/fit permutation",
    "pool": "initial encoders, gains, and centers",
    "center": "new neurons' gains and centers",
    "proposal": "candidate encoders/directions",
    "subsample": "fixed AGOP fit subset",
    "shuffle": "fresh residual row permutation each generation",
    "eval_subset": "fixed pruning fit-accuracy subset",
    "retain": "uniform random survivor subset",
    "eig": "initial search basis for global pHd subspace iteration",
}


def keyed_rng(seed: int, role: str, generation: int = 0) -> torch.Generator:
    """Key an independent stream by seed, role ID and generation (no arm input)."""
    if not 0 <= seed < 2**31 or not 0 <= generation < 2**20:
        raise ValueError("seed must be in [0, 2**31) and generation in [0, 2**20)")
    role_id = tuple(STREAMS).index(role)
    # Odd multipliers preserve distinct role/generation pairs in the CPU RNG's 32 bits.
    stream = (generation << 4) | role_id
    return torch.Generator().manual_seed(
        (seed * 2_000_003 + stream * 2_654_435_761 + 1009) % 2**32
    )


@dataclass(frozen=True)
class Split:
    x_fit: Tensor
    y_fit: Tensor
    y_targets: Tensor
    x_val: Tensor
    y_val: Tensor
    x_test: Tensor
    y_test: Tensor

    def evaluation(self, fit_subset: Tensor | None = None) -> tuple[tuple[Tensor, Tensor], ...]:
        fit = (self.x_fit, self.y_fit)
        if fit_subset is not None:
            fit = (self.x_fit[fit_subset], self.y_fit[fit_subset])
        return ((self.x_val, self.y_val), (self.x_test, self.y_test), fit)


def make_split(
    bundle: DatasetBundle, seed: int, *, val_size: int = 10000, fit_limit: int | None = None
) -> Split:
    if not 0 < val_size < len(bundle.x_train) or (fit_limit is not None and fit_limit < 1):
        raise ValueError("validation and fit splits must both be nonempty")
    set_sl_seed(seed)
    order = torch.randperm(len(bundle.x_train), generator=keyed_rng(seed, "split"))
    val, fit = order[:val_size], order[val_size:]
    if fit_limit is not None:
        fit = fit[:fit_limit]
    y_fit = bundle.y_train[fit]
    return Split(
        bundle.x_train[fit],
        y_fit,
        bundle.one_hot(y_fit),
        bundle.x_train[val],
        bundle.y_train[val],
        bundle.x_test,
        bundle.y_test,
    )


def subset(n_rows: int, rng: torch.Generator, limit: int = 10000) -> Tensor:
    return torch.randperm(n_rows, generator=rng)[:limit]


def encoder_kwargs(strategy: str, image_shape: tuple[int, int]) -> dict:
    return (
        {"image_shape": image_shape, "patch_size": PATCH_SIZE}
        if strategy == "receptive_field"
        else {}
    )


def make_layer(
    split: Split,
    width: int,
    strategy: str,
    image_shape: tuple[int, int],
    rng: torch.Generator,
) -> leenef.NEFLayer:
    return leenef.NEFLayer(
        split.x_fit.shape[1],
        width,
        split.y_targets.shape[1],
        activation="abs",
        gain=(0.5, 2.0),
        centers=split.x_fit,
        encoder_strategy=strategy,
        encoder_kwargs=encoder_kwargs(strategy, image_shape),
        rng=rng,
    )


def accumulate(layer: leenef.NEFLayer, split: Split, chunk_size: int) -> None:
    for x, y in zip(split.x_fit.split(chunk_size), split.y_targets.split(chunk_size), strict=True):
        layer.partial_fit(x, y)


@torch.no_grad()
def cache_activities(layer: leenef.NEFLayer, x: Tensor, chunk_size: int) -> Tensor:
    cache = x.new_empty((len(x), layer.n_neurons))
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        cache[start:stop] = layer.encode(x[start:stop])
    return cache


@torch.no_grad()
def select_alpha(
    normal: Tensor,
    cross: Tensor,
    evaluation: tuple[tuple[Tensor, Tensor], ...],
    *,
    chunk_size: int,
    layer: leenef.NEFLayer | None = None,
    columns: slice | Tensor = slice(None),
) -> dict[str, float]:
    """Tune on validation, then report the selected and loop-alpha decoders.

    ``evaluation`` holds validation/test/fit data and labels in that order.
    Inputs are cached activities if ``layer`` is omitted, and raw inputs
    otherwise. Each validation batch is encoded at most once. A supplied
    layer is left fitted at the fixed loop alpha, not the validation choice.
    """
    decoders = torch.stack(
        [
            leenef.solve_from_normal_equations(normal, cross, alpha=(alpha, "trace"))
            for alpha in ALPHAS
        ],
        dim=1,
    ).float()

    def evaluate(x: Tensor, y: Tensor, weights: Tensor, *, mse: bool) -> tuple[Tensor, Tensor]:
        n_alphas, n_classes = weights.shape[1:]
        flat = weights.flatten(1)
        counts = torch.zeros(n_alphas, dtype=torch.long)
        errors = torch.zeros(n_alphas, dtype=torch.float64)
        for x_batch, y_batch in zip(x.split(chunk_size), y.split(chunk_size), strict=True):
            a = layer.encode(x_batch) if layer is not None else x_batch[:, columns]
            pred = (a @ flat).reshape(len(a), n_alphas, n_classes)
            counts += (pred.argmax(dim=-1) == y_batch[:, None]).sum(dim=0)
            if mse:
                target = torch.nn.functional.one_hot(y_batch, n_classes)
                errors += (pred - target[:, None, :]).square().sum(dim=(0, 2), dtype=torch.float64)
        return counts, errors / (len(x) * n_classes)

    val, test, fit = evaluation
    val_correct, val_mse = evaluate(*val, decoders, mse=True)
    best = len(ALPHAS) - 1 - int(val_correct.flip(0).argmax())
    loop = ALPHAS.index(LOOP_ALPHA[0])
    selected = decoders[:, [best, loop]]
    test_correct, test_mse = evaluate(*test, selected, mse=True)
    fit_correct, _ = evaluate(*fit, selected, mse=False)
    if layer is not None:
        layer.solve_accumulated(alpha=LOOP_ALPHA)
    return {
        "alpha": ALPHAS[best],
        "val_accuracy": int(val_correct[best]) / len(val[1]),
        "test_accuracy": int(test_correct[0]) / len(test[1]),
        "fit_accuracy": int(fit_correct[0]) / len(fit[1]),
        "val_mse": float(val_mse[best]),
        "test_mse": float(test_mse[0]),
        "alpha_loop": LOOP_ALPHA[0],
        "val_accuracy_loop": int(val_correct[loop]) / len(val[1]),
        "test_accuracy_loop": int(test_correct[1]) / len(test[1]),
        "fit_accuracy_loop": int(fit_correct[1]) / len(fit[1]),
        "val_mse_loop": float(val_mse[loop]),
        "test_mse_loop": float(test_mse[1]),
    }


def stats(layer: leenef.NEFLayer) -> tuple[Tensor, Tensor]:
    assert layer._ata is not None and layer._aty is not None
    return layer._ata, layer._aty


def row_key(factors: dict[str, Any], seed: int) -> str:
    return ":".join([*(f"{k}={v}" for k, v in sorted(factors.items())), f"seed={seed}"])


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reset_peak_rss() -> bool:
    try:
        Path("/proc/self/clear_refs").write_text("5\n")
        return True
    except OSError:
        return False


def peak_rss(reset: bool) -> tuple[float, str]:
    if reset:
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024, "VmHWM"
        except (OSError, ValueError, IndexError):
            pass
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, "ru_maxrss"


def read_results(path: Path) -> dict:
    if not path.exists():
        return {"meta": {}, "rows": []}
    return json.loads(path.read_text())


class Results:
    def __init__(self, path: Path, panel: str, bundle: DatasetBundle):
        self.path = path
        self.panel = panel
        self.dataset = bundle.metadata()
        self.identity = panel_identity()
        if self.identity is None:
            raise RuntimeError("benchmark results require a Git code identity")
        self.payload = read_results(path)
        self.payload["meta"].update(
            {
                "panel": panel,
                "code_identity": self.identity,
                "alphas": ALPHAS,
                "loop_alpha": LOOP_ALPHA,
                "rng_streams": STREAMS,
                "rng_seed": "(seed * 2000003 + ((generation << 4) | role_id) * 2654435761 + 1009) % 2**32",
            }
        )
        self.completed = {
            row["key"] for row in current_rows(self.payload, self.identity) if "key" in row
        }

    def add(self, factors: dict, seed: int, details: dict, *, rss_reset: bool) -> None:
        rss, rss_source = peak_rss(rss_reset)
        key = row_key(factors, seed)
        row = {
            **result_metadata(
                config=factors,
                seed=seed,
                experiment=f"neuron_selection_{self.panel}",
                primary_metric="test_accuracy",
            ),
            **factors,
            **details,
            "code_identity": self.identity,
            "panel": self.panel,
            "result_namespace": "neuron_selection",
            "key": key,
            "factors": factors,
            "dataset_metadata": self.dataset,
            "torch_threads": torch.get_num_threads(),
            "peak_rss_mb": rss,
            "peak_rss_source": rss_source,
        }
        self.payload["rows"].append(row)
        atomic_write(self.path, self.payload)
        self.completed.add(key)
        print(f"{key}: test={row['test_accuracy']:.4f} ({row['wall_time_s']:.2f}s)", flush=True)


def current_rows(payload: dict, identity: str) -> list[dict]:
    return [row for row in payload.get("rows", []) if row.get("code_identity") == identity]


def parse_seeds(value: str) -> list[int]:
    try:
        if "-" in value and "," not in value:
            start, end = map(int, value.split("-", 1))
            numbers = list(range(start, end + 1))
        else:
            numbers = [int(part) for part in value.split(",")]
        if not numbers or min(numbers) < 0:
            raise ValueError
        return list(dict.fromkeys(numbers))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected nonnegative seeds as A-B or A,B,...") from exc


def seed_shards(seeds: list[int], count: int) -> list[list[int]]:
    """Partition seeds into balanced, nonempty, contiguous slices in input order."""
    if count < 1 or not seeds:
        raise ValueError("GROWTH_SHARDS must be positive and the seed list must be nonempty")
    count = min(count, len(seeds))
    return [
        seeds[(i * len(seeds)) // count : ((i + 1) * len(seeds)) // count] for i in range(count)
    ]


def parse_widths(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(sorted({int(item) for item in value.split(",")}))
        if widths[0] < 1:
            raise ValueError
        return widths
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive widths") from exc


def comma_list(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.split(",")))


def panel_parser(doc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--dataset", choices=IMAGE_SHAPES, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default="0-9")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    return parser


def load_cli_data(args: argparse.Namespace) -> DatasetBundle:
    panel_identity()
    torch.set_num_threads(args.threads)
    return load_dataset(args.dataset, preprocessing="div255_flat")


def panel_identity() -> str | None:
    identity = code_identity()
    pinned = os.environ.get("NS_IDENTITY")
    if pinned is not None and pinned != identity:
        raise RuntimeError(f"code identity mismatch: NS_IDENTITY={pinned}, checkout={identity}")
    return identity


def expected_result_keys(manifest: dict) -> dict[str, set[str]]:
    """Enumerate the exact result keys for the driver's current launch."""
    from . import growth_panel, prune_panel

    expected = {}
    for dataset in manifest["datasets"]:
        flags = ["--dataset", dataset, *(["--smoke"] if manifest["smoke"] else [])]
        args = prune_panel.parse_args(flags)
        prune = prune_panel.panel_configs(dataset, **prune_panel.cli_options(args))
        expected[f"prune_{dataset}.json"] = {
            row_key(cfg, seed)
            for cells in prune.values()
            for cfg in cells
            for seed in manifest["seeds"]
        }
        args = growth_panel.parse_args([*flags, "--arms", ",".join(manifest["arms"][dataset])])
        growth = growth_panel.configs(dataset, **growth_panel.cli_options(args))
        for i, shard in enumerate(manifest["shards"]):
            expected[f"growth_{dataset}_s{i}.json"] = {
                row_key(cfg, seed) for cfg in growth for seed in shard
            }
    return expected


def _write_manifest(args: argparse.Namespace) -> int:
    from .growth_panel import FAMILIES

    datasets = dict(item.split("=", 1) for item in args.dataset_arms)
    all_arms = tuple(dict.fromkeys(arm for _, arms in FAMILIES.values() for arm in arms))
    shards = seed_shards(args.seeds, args.shards)
    start = time.time_ns()
    event = {
        "event": "start",
        "identity": args.identity,
        "started_at": datetime.fromtimestamp(start / 1e9, timezone.utc).isoformat(),
        "started_ns": start,
        "seeds": args.seeds,
        "threads": args.threads,
        "shards": shards,
        "datasets": list(datasets),
        "arms": {
            dataset: comma_list(arms) if arms else all_arms for dataset, arms in datasets.items()
        },
        "smoke": args.smoke,
    }
    with args.manifest.open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    for shard in shards:
        print(",".join(map(str, shard)))
    return 0


def _status(manifest_path: Path) -> int:
    """Print each file's status; return 2 while running, 0/1 after completion."""
    if not manifest_path.is_file():
        print("waiting for run manifest")
        return 2
    launches = [
        json.loads(line)
        for line in manifest_path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]
    manifest = next((event for event in reversed(launches) if event["event"] == "start"), None)
    if manifest is None:
        print("waiting for run manifest")
        return 2
    out = manifest_path.parent.parent
    done = out / "panels_done"
    try:
        fresh_done = done.stat().st_mtime_ns >= manifest["started_ns"]
    except FileNotFoundError:
        fresh_done = False
    try:
        expected = expected_result_keys(manifest)
    except ValueError as exc:
        print(f"invalid manifest configuration: {exc}")
        return 1 if fresh_done else 2
    valid = True
    progress = []
    for name, keys in expected.items():
        path = out / name
        try:
            rows = current_rows(read_results(path), manifest["identity"])
        except (OSError, ValueError) as exc:
            progress.append(f"{name} unreadable/{len(keys)} ({exc})")
            valid = False
            continue
        matches = (
            path.is_file() and len(rows) == len(keys) and {r.get("key") for r in rows} == keys
        )
        valid &= matches
        detail = " (wrong keys)" if rows and len(rows) == len(keys) and not matches else ""
        progress.append(f"{name} {len(rows)}/{len(keys)}{detail}")
    print("; ".join(progress))
    return int(not valid) if fresh_done else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("manifest")
    start.add_argument("manifest", type=Path)
    start.add_argument("--identity", required=True)
    start.add_argument("--seeds", type=parse_seeds, required=True)
    start.add_argument("--threads", type=int, required=True)
    start.add_argument("--shards", type=int, required=True)
    start.add_argument("--smoke", action="store_true")
    start.add_argument("dataset_arms", nargs="+")
    status = subparsers.add_parser("status")
    status.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    return _write_manifest(args) if args.command == "manifest" else _status(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
