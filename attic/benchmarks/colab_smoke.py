"""Smoke test for running leenef benchmarks in Google Colab.

This script is intentionally small and fast:

1. Resolve a target device (CPU / CUDA / auto).
2. Run a tiny MNIST NEFLayer fit on real data.
3. Run a tiny row-wise sMNIST StreamingNEFClassifier fit on real data.
4. Persist a compact JSON report for later inspection.

The goal is not to produce publishable numbers, but to validate that:

- the repository installs correctly in Colab,
- torchvision dataset downloads work,
- the NEF modules execute on the selected device,
- and benchmark outputs can be saved to disk.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmarks.run import (  # noqa: E402
    classification_accuracy,
    load_vision_dataset,
    one_hot,
    set_benchmark_seed,
)
from benchmarks.run_recurrent import load_sequential_mnist  # noqa: E402
from leenef.layers import NEFLayer  # noqa: E402
from leenef.streaming import StreamingNEFClassifier  # noqa: E402


def resolve_device(device: str) -> torch.device:
    """Resolve the requested device string to a concrete torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(device)


def sync_if_needed(device: torch.device) -> None:
    """Synchronize CUDA before/after timings."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_mnist_smoke(
    *,
    device: torch.device,
    data_root: str,
    train_samples: int,
    test_samples: int,
    n_neurons: int,
    alpha: float,
) -> dict:
    """Run a tiny MNIST NEFLayer fit on real data."""
    (x_train, y_train), (x_test, y_test) = load_vision_dataset("mnist", root=data_root)
    x_train = x_train[:train_samples]
    y_train = y_train[:train_samples]
    x_test = x_test[:test_samples]
    y_test = y_test[:test_samples]
    centers = x_train.clone()

    layer = NEFLayer(
        x_train.shape[1],
        n_neurons,
        10,
        activation="abs",
        encoder_strategy="hypersphere",
        gain=(0.5, 2.0),
        centers=centers,
    ).to(device)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    targets = one_hot(y_train.cpu(), 10).to(device)

    sync_if_needed(device)
    t0 = time.perf_counter()
    layer.fit(x_train, targets, solver="tikhonov", alpha=alpha)
    sync_if_needed(device)
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        train_acc = classification_accuracy(layer(x_train), y_train)
        test_acc = classification_accuracy(layer(x_test), y_test)

    if test_acc < 0.60:
        raise RuntimeError(f"MNIST smoke accuracy too low: {test_acc:.2%}")

    return {
        "name": "mnist_nef_smoke",
        "dataset": "mnist",
        "device": str(device),
        "n_neurons": n_neurons,
        "alpha": alpha,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "fit_time": fit_time,
    }


def run_streaming_smoke(
    *,
    device: torch.device,
    data_root: str,
    train_samples: int,
    test_samples: int,
    n_neurons: int,
    window_size: int,
    batch_size: int,
    alpha: float,
) -> dict:
    """Run a tiny row-wise sMNIST streaming fit on real data."""
    (x_train_seq, _, y_train), (x_test_seq, _, y_test) = load_sequential_mnist(
        mode="row",
        root=data_root,
        seed=0,
    )
    x_train_seq = x_train_seq[:train_samples]
    y_train = y_train[:train_samples]
    x_test_seq = x_test_seq[:test_samples]
    y_test = y_test[:test_samples]
    centers = x_train_seq.clone()

    clf = StreamingNEFClassifier(
        d_timestep=x_train_seq.shape[2],
        n_neurons=n_neurons,
        d_out=10,
        window_size=window_size,
        activation="abs",
        encoder_strategy="hypersphere",
        gain=(0.5, 2.0),
        centers=centers,
    ).to(device)

    x_train_seq = x_train_seq.to(device)
    y_train = y_train.to(device)
    x_test_seq = x_test_seq.to(device)
    y_test = y_test.to(device)
    targets = one_hot(y_train.cpu(), 10).to(device)

    sync_if_needed(device)
    t0 = time.perf_counter()
    for i in range(0, len(x_train_seq), batch_size):
        clf.continuous_fit(
            x_train_seq[i : i + batch_size], targets[i : i + batch_size], alpha=alpha
        )
    clf.refresh_inverse(alpha=alpha)
    sync_if_needed(device)
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        train_acc = classification_accuracy(clf(x_train_seq), y_train)
        test_acc = classification_accuracy(clf(x_test_seq), y_test)

    if test_acc < 0.35:
        raise RuntimeError(f"Streaming smoke accuracy too low: {test_acc:.2%}")

    return {
        "name": "smnist_streaming_smoke",
        "dataset": "smnist_row",
        "device": str(device),
        "n_neurons": n_neurons,
        "window_size": window_size,
        "batch_size": batch_size,
        "alpha": alpha,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "fit_time": fit_time,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Smoke test leenef in Google Colab")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mnist-train", type=int, default=512)
    parser.add_argument("--mnist-test", type=int, default=256)
    parser.add_argument("--mnist-neurons", type=int, default=256)
    parser.add_argument("--mnist-alpha", type=float, default=1e-2)
    parser.add_argument("--stream-train", type=int, default=512)
    parser.add_argument("--stream-test", type=int, default=256)
    parser.add_argument("--stream-neurons", type=int, default=256)
    parser.add_argument("--stream-window", type=int, default=5)
    parser.add_argument("--stream-batch", type=int, default=64)
    parser.add_argument("--stream-alpha", type=float, default=1e-2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    set_benchmark_seed(args.seed)
    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.time()
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
        "results": [
            run_mnist_smoke(
                device=device,
                data_root=args.data_root,
                train_samples=args.mnist_train,
                test_samples=args.mnist_test,
                n_neurons=args.mnist_neurons,
                alpha=args.mnist_alpha,
            ),
            run_streaming_smoke(
                device=device,
                data_root=args.data_root,
                train_samples=args.stream_train,
                test_samples=args.stream_test,
                n_neurons=args.stream_neurons,
                window_size=args.stream_window,
                batch_size=args.stream_batch,
                alpha=args.stream_alpha,
            ),
        ],
        "finished_at_unix": time.time(),
        "wall_time": time.time() - started,
    }
    if device.type == "cuda":
        payload["environment"]["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
