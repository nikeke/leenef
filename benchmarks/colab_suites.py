"""Predefined Colab experiment suites for GPU-friendly benchmarks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmarks.run import (  # noqa: E402
    BenchmarkResult,
    format_results,
    save_results_csv,
    save_results_json,
    set_benchmark_seed,
)
from benchmarks.run_recurrent import run_lstm_baseline, run_streaming_nef  # noqa: E402


def _run_labeled(label: str, fn, /, **kwargs):
    """Run one suite item with explicit progress messages."""
    print(f"Running {label}...", flush=True)
    result = fn(**kwargs)
    print(
        f"Finished {label}: test={result.test_metric:.2%}, fit_time={result.fit_time:.2f}s",
        flush=True,
    )
    return result


def run_row_focus_suite(args: argparse.Namespace) -> list:
    """Run the row-wise sMNIST suite."""
    if args.quick:
        return [
            _run_labeled(
                "StreamNEF row quick",
                run_streaming_nef,
                mode="row",
                n_neurons=512,
                window_size=5,
                alpha=1e-2,
                batch_size=128,
                data_root=args.data_root,
                seed=args.seed,
                solve_mode="woodbury",
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=True,
            ),
            _run_labeled(
                "LSTM row quick",
                run_lstm_baseline,
                mode="row",
                hidden_size=64,
                n_epochs=1,
                batch_size=128,
                data_root=args.data_root,
                seed=args.seed,
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=True,
            ),
        ]

    return [
        _run_labeled(
            "StreamNEF row 2k (woodbury)",
            run_streaming_nef,
            mode="row",
            n_neurons=2000,
            window_size=10,
            alpha=1e-2,
            batch_size=500,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="woodbury",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF row 2k (accumulate)",
            run_streaming_nef,
            mode="row",
            n_neurons=2000,
            window_size=10,
            alpha=1e-2,
            batch_size=500,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF row 8k (woodbury)",
            run_streaming_nef,
            mode="row",
            n_neurons=8000,
            window_size=10,
            alpha=5e-3,
            batch_size=500,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="woodbury",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF row 8k (accumulate)",
            run_streaming_nef,
            mode="row",
            n_neurons=8000,
            window_size=10,
            alpha=5e-3,
            batch_size=500,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "LSTM row 128",
            run_lstm_baseline,
            mode="row",
            hidden_size=128,
            n_epochs=args.lstm_epochs,
            batch_size=args.lstm_batch,
            data_root=args.data_root,
            seed=args.seed,
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
    ]


def run_sequential_hard_suite(args: argparse.Namespace) -> list:
    """Run the longer-sequence sMNIST pixel suites."""
    if args.quick:
        return [
            _run_labeled(
                "StreamNEF pixel quick",
                run_streaming_nef,
                mode="pixel",
                n_neurons=512,
                window_size=28,
                alpha=1e-2,
                batch_size=128,
                data_root=args.data_root,
                seed=args.seed,
                solve_mode="woodbury",
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=True,
            ),
            _run_labeled(
                "LSTM pixel quick",
                run_lstm_baseline,
                mode="pixel",
                hidden_size=64,
                n_epochs=1,
                batch_size=128,
                data_root=args.data_root,
                seed=args.seed,
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=True,
            ),
        ]

    return [
        _run_labeled(
            "StreamNEF pixel w28 (accumulate)",
            run_streaming_nef,
            mode="pixel",
            n_neurons=4000,
            window_size=28,
            alpha=1e-2,
            batch_size=250,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF pixel w56 (accumulate)",
            run_streaming_nef,
            mode="pixel",
            n_neurons=4000,
            window_size=56,
            alpha=5e-3,
            batch_size=250,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF permuted w56 (accumulate)",
            run_streaming_nef,
            mode="pixel_permuted",
            n_neurons=4000,
            window_size=56,
            alpha=5e-3,
            batch_size=250,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "LSTM pixel 128",
            run_lstm_baseline,
            mode="pixel",
            hidden_size=128,
            n_epochs=args.lstm_epochs,
            batch_size=args.lstm_batch,
            data_root=args.data_root,
            seed=args.seed,
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "LSTM permuted 128",
            run_lstm_baseline,
            mode="pixel_permuted",
            hidden_size=128,
            n_epochs=args.lstm_epochs,
            batch_size=args.lstm_batch,
            data_root=args.data_root,
            seed=args.seed,
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
    ]


def _run_conv_config(
    label: str,
    *,
    x_train,
    y_train,
    x_test,
    y_test,
    targets_train,
    stages: list[dict],
    n_neurons: int,
    pool_levels: list[int] | None = None,
    pool_order: int = 1,
    alpha: float = 1e-2,
    fit_subsample: int = 10_000,
    batch_size: int = 2000,
    seed: int = 0,
    n_members: int = 1,
    augment_flip: bool = False,
    standardize: bool = False,
    lcn_kernel: int | None = None,
    gcn: bool = False,
    parallel: bool = False,
    member_stages: list[list[dict]] | None = None,
    **nef_kwargs,
) -> BenchmarkResult:
    """Run one ConvNEF configuration and return a BenchmarkResult."""
    import torch

    from leenef.conv import ConvNEFEnsemble, ConvNEFPipeline

    aug_fn = (lambda x: x.flip(-1)) if augment_flip else None

    print(f"Running {label}...", flush=True)
    t0 = time.time()

    if not stages:
        # Flat pixel baseline: NEFLayer on flattened input
        from leenef.layers import NEFLayer

        flat_train = x_train.reshape(x_train.shape[0], -1)
        flat_test = x_test.reshape(x_test.shape[0], -1)
        d_in = flat_train.shape[1]

        # Subsample centers
        g = torch.Generator(device=x_train.device).manual_seed(seed)
        sub_n = min(fit_subsample, flat_train.shape[0])
        sub_idx = torch.randperm(flat_train.shape[0], generator=g, device=x_train.device)[
            :sub_n
        ]
        centers = flat_train[sub_idx]

        head = NEFLayer(d_in, n_neurons, 10, centers=centers, **nef_kwargs)
        head.reset_accumulators()
        for i in range(0, flat_train.shape[0], batch_size):
            head.partial_fit(
                flat_train[i : i + batch_size],
                targets_train[i : i + batch_size],
            )
        head.solve_accumulated(alpha=alpha)
        fit_time = time.time() - t0

        with torch.no_grad():
            preds = []
            for i in range(0, flat_test.shape[0], batch_size):
                preds.append(head(flat_test[i : i + batch_size]))
            pred = torch.cat(preds)
            test_acc = (pred.argmax(1) == y_test).float().mean().item()

            preds_train = []
            for i in range(0, flat_train.shape[0], batch_size):
                preds_train.append(head(flat_train[i : i + batch_size]))
            pred_train = torch.cat(preds_train)
            train_acc = (pred_train.argmax(1) == y_train).float().mean().item()
    elif n_members > 1:
        model = ConvNEFEnsemble(
            n_members=n_members,
            stages=stages,
            n_neurons=n_neurons,
            pool_levels=pool_levels,
            pool_order=pool_order,
            standardize=standardize,
            lcn_kernel=lcn_kernel,
            gcn=gcn,
            parallel=parallel,
            member_stages=member_stages,
            **nef_kwargs,
        )
        model.fit(
            x_train,
            targets_train,
            alpha=alpha,
            fit_subsample=fit_subsample,
            batch_size=batch_size,
            base_seed=seed,
            augment_fn=aug_fn,
        )

        fit_time = time.time() - t0
        pred = model.predict(x_test, batch_size=batch_size)
        test_acc = (pred.argmax(1) == y_test).float().mean().item()
        pred_train = model.predict(x_train, batch_size=batch_size)
        train_acc = (pred_train.argmax(1) == y_train).float().mean().item()
    else:
        model = ConvNEFPipeline(
            stages=stages,
            n_neurons=n_neurons,
            pool_levels=pool_levels,
            pool_order=pool_order,
            standardize=standardize,
            lcn_kernel=lcn_kernel,
            gcn=gcn,
            parallel=parallel,
            **nef_kwargs,
        )
        model.fit(
            x_train,
            targets_train,
            alpha=alpha,
            fit_subsample=fit_subsample,
            batch_size=batch_size,
            seed=seed,
            augment_fn=aug_fn,
        )

        fit_time = time.time() - t0
        pred = model.predict(x_test, batch_size=batch_size)
        test_acc = (pred.argmax(1) == y_test).float().mean().item()
        pred_train = model.predict(x_train, batch_size=batch_size)
        train_acc = (pred_train.argmax(1) == y_train).float().mean().item()

    print(
        f"Finished {label}: test={test_acc:.2%}, train={train_acc:.2%}, "
        f"fit_time={fit_time:.2f}s",
        flush=True,
    )

    return BenchmarkResult(
        name=label,
        dataset="cifar10",
        n_neurons=n_neurons,
        activation="abs",
        encoder_strategy="hypersphere",
        solver="ridge",
        solver_kwargs={"alpha": alpha},
        metric_name="accuracy",
        train_metric=train_acc,
        test_metric=test_acc,
        fit_time=fit_time,
    )


def _load_cifar10(data_root: str, device: str):
    """Load CIFAR-10 onto the target device."""
    import torch
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=transform
    )
    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds]).to(device)
    x_test = torch.stack([img for img, _ in test_ds]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_ds]).to(device)
    targets_train = F.one_hot(y_train, 10).float()
    return x_train, y_train, x_test, y_test, targets_train


def run_conv_cifar_suite(args: argparse.Namespace) -> list:
    """Run convolutional NEF benchmark on CIFAR-10."""
    import torch

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_benchmark_seed(args.seed)

    x_train, y_train, x_test, y_test, targets_train = _load_cifar10(
        args.data_root, dev
    )
    common = dict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        targets_train=targets_train,
        seed=args.seed,
    )

    if args.quick:
        return [
            _run_conv_config(
                "ConvNEF quick",
                stages=[{"n_filters": 16, "patch_size": 5, "pool_size": 1}],
                n_neurons=500,
                pool_levels=[1, 2],
                alpha=1e-2,
                fit_subsample=1000,
                batch_size=500,
                **common,
            ),
        ]

    return [
        # ── Baseline: flat pixel NEF ──────────────────────────────────
        _run_conv_config(
            "Flat pixel 5k",
            stages=[],
            n_neurons=5000,
            pool_levels=None,
            alpha=1e-2,
            fit_subsample=10_000,
            **common,
        ),
        # ── PCA filters ───────────────────────────────────────────────
        _run_conv_config(
            "PCA 32f p5 spp124 5k",
            stages=[{"n_filters": 32, "patch_size": 5, "pool_size": 1}],
            n_neurons=5000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 5k",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=5000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            **common,
        ),
        _run_conv_config(
            "PCA 128f p5 spp124 10k",
            stages=[{"n_filters": 128, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            **common,
        ),
        _run_conv_config(
            "PCA 256f p5 spp124 10k",
            stages=[{"n_filters": 256, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=5e-3,
            fit_subsample=10_000,
            **common,
        ),
        # ── Different patch sizes ─────────────────────────────────────
        _run_conv_config(
            "PCA 64f p3 spp124 10k",
            stages=[{"n_filters": 64, "patch_size": 3, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p7 spp124 10k",
            stages=[{"n_filters": 64, "patch_size": 7, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            **common,
        ),
        # ── With augmentation ─────────────────────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 10k +hflip",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            augment_flip=True,
            **common,
        ),
        # ── Ensemble ──────────────────────────────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 5k ×5",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=5000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=5,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k ×5",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=5,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k ×10",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=10,
            **common,
        ),
        # ── Ensemble + augmentation ───────────────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 10k ×5 +hflip",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=5,
            augment_flip=True,
            **common,
        ),
        # ── Feature standardization ───────────────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 10k +std",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            standardize=True,
            **common,
        ),
        # ── Best combos ──────────────────────────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 10k +std +hflip",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            standardize=True,
            augment_flip=True,
            **common,
        ),
        _run_conv_config(
            "PCA 128f p5 spp124 10k +std",
            stages=[{"n_filters": 128, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            standardize=True,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k ×5 +std +hflip",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=5,
            standardize=True,
            augment_flip=True,
            **common,
        ),
        # ── Local contrast normalization ──────────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 10k +lcn",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            lcn_kernel=5,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k +lcn +std",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            lcn_kernel=5,
            standardize=True,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k ×5 +lcn +std +hflip",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=5,
            lcn_kernel=5,
            standardize=True,
            augment_flip=True,
            **common,
        ),
        # ── Global contrast normalization ─────────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 10k +gcn",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            gcn=True,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k +gcn +lcn +std",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            gcn=True,
            lcn_kernel=5,
            standardize=True,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k ×5 +gcn +lcn +std +hflip",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=5,
            gcn=True,
            lcn_kernel=5,
            standardize=True,
            augment_flip=True,
            **common,
        ),
        # ── Multi-scale parallel stages ───────────────────────────────
        _run_conv_config(
            "PCA multi(32p3+32p5+32p7) spp124 10k",
            stages=[
                {"n_filters": 32, "patch_size": 3, "pool_size": 1},
                {"n_filters": 32, "patch_size": 5, "pool_size": 1},
                {"n_filters": 32, "patch_size": 7, "pool_size": 1},
            ],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            parallel=True,
            **common,
        ),
        _run_conv_config(
            "PCA multi(32p3+32p5+32p7) spp124 10k +gcn +std",
            stages=[
                {"n_filters": 32, "patch_size": 3, "pool_size": 1},
                {"n_filters": 32, "patch_size": 5, "pool_size": 1},
                {"n_filters": 32, "patch_size": 7, "pool_size": 1},
            ],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            parallel=True,
            gcn=True,
            standardize=True,
            **common,
        ),
        _run_conv_config(
            "PCA multi(64p3+64p5+64p7) spp124 10k +gcn +std",
            stages=[
                {"n_filters": 64, "patch_size": 3, "pool_size": 1},
                {"n_filters": 64, "patch_size": 5, "pool_size": 1},
                {"n_filters": 64, "patch_size": 7, "pool_size": 1},
            ],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            parallel=True,
            gcn=True,
            standardize=True,
            **common,
        ),
        # ── Per-patch contrast normalization ──────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 10k +patchnorm",
            stages=[
                {"n_filters": 64, "patch_size": 5, "pool_size": 1, "normalize_patches": True}
            ],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k +patchnorm +gcn +std",
            stages=[
                {"n_filters": 64, "patch_size": 5, "pool_size": 1, "normalize_patches": True}
            ],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            gcn=True,
            standardize=True,
            **common,
        ),
        _run_conv_config(
            "PCA 128f p5 spp124 10k +patchnorm +gcn +std",
            stages=[
                {"n_filters": 128, "patch_size": 5, "pool_size": 1, "normalize_patches": True}
            ],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            gcn=True,
            standardize=True,
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k ×5 +patchnorm +gcn +std +hflip",
            stages=[
                {"n_filters": 64, "patch_size": 5, "pool_size": 1, "normalize_patches": True}
            ],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=5,
            gcn=True,
            standardize=True,
            augment_flip=True,
            **common,
        ),
        # ── Whitened head encoders ────────────────────────────────────
        _run_conv_config(
            "PCA 64f p5 spp124 10k +whitened",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            encoder_strategy="whitened",
            **common,
        ),
        _run_conv_config(
            "PCA 64f p5 spp124 10k +patchnorm +gcn +std +whitened",
            stages=[
                {"n_filters": 64, "patch_size": 5, "pool_size": 1, "normalize_patches": True}
            ],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            gcn=True,
            standardize=True,
            encoder_strategy="whitened",
            **common,
        ),
        # ── Diverse ensemble (different patch sizes per member) ───────
        _run_conv_config(
            "PCA diverse(p3/p5/p7) spp124 10k ×6 +gcn +std",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=6,
            gcn=True,
            standardize=True,
            member_stages=[
                [{"n_filters": 64, "patch_size": 3, "pool_size": 1}],
                [{"n_filters": 64, "patch_size": 3, "pool_size": 1}],
                [{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
                [{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
                [{"n_filters": 64, "patch_size": 7, "pool_size": 1}],
                [{"n_filters": 64, "patch_size": 7, "pool_size": 1}],
            ],
            **common,
        ),
        _run_conv_config(
            "PCA diverse(p3/p5/p7) spp124 10k ×6 +patchnorm +gcn +std",
            stages=[{"n_filters": 64, "patch_size": 5, "pool_size": 1}],
            n_neurons=10_000,
            pool_levels=[1, 2, 4],
            alpha=1e-2,
            fit_subsample=10_000,
            n_members=6,
            gcn=True,
            standardize=True,
            member_stages=[
                [{"n_filters": 64, "patch_size": 3, "pool_size": 1, "normalize_patches": True}],
                [{"n_filters": 64, "patch_size": 3, "pool_size": 1, "normalize_patches": True}],
                [{"n_filters": 64, "patch_size": 5, "pool_size": 1, "normalize_patches": True}],
                [{"n_filters": 64, "patch_size": 5, "pool_size": 1, "normalize_patches": True}],
                [{"n_filters": 64, "patch_size": 7, "pool_size": 1, "normalize_patches": True}],
                [{"n_filters": 64, "patch_size": 7, "pool_size": 1, "normalize_patches": True}],
            ],
            **common,
        ),
    ]


SUITES = {
    "row_focus": run_row_focus_suite,
    "sequential_hard": run_sequential_hard_suite,
    "conv_cifar": run_conv_cifar_suite,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run predefined Colab benchmark suites")
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./results/colab"))
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=2048)
    parser.add_argument("--lstm-epochs", type=int, default=20)
    parser.add_argument("--lstm-batch", type=int, default=256)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a much smaller suite for validation instead of the full Colab workload",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    results = SUITES[args.suite](args)

    stem = args.suite if not args.quick else f"{args.suite}-quick"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    save_results_json(results, json_path)
    save_results_csv(results, csv_path)

    print()
    print(format_results(results))
    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
