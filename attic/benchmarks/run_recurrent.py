"""Benchmark harness for recurrent NEF on temporal classification tasks."""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# Ensure the checkout root and src layout are importable when run as a script.
_project_root = Path(__file__).resolve().parent.parent
for _path in (str(_project_root), str(_project_root / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmarks.run import (  # noqa: E402
    BenchmarkResult,
    batched_classification_accuracy,
    format_results,
    one_hot,
    resolve_benchmark_device,
    save_results_csv,
    save_results_json,
    set_benchmark_seed,
    synchronize_if_cuda,
)
from leenef.recurrent import RecurrentNEFLayer  # noqa: E402
from leenef.streaming import StreamingNEFClassifier  # noqa: E402


def load_sequential_mnist(
    mode: str = "row",
    root: str = "./data",
    seed: int = 0,
) -> tuple[
    tuple[Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
]:
    """Load MNIST as sequences for temporal classification.

    Args:
        mode: ``"row"`` — 28 rows of 28 pixels (T=28, d=28).
              ``"pixel"`` — 784 individual pixels (T=784, d=1).
              ``"pixel_permuted"`` — permuted pixel order (T=784, d=1).
    Returns:
        ((x_train_seq, x_train_flat, y_train),
         (x_test_seq, x_test_flat, y_test))
        where x_*_seq has shape (N, T, d) and x_*_flat is (N, features).
    """
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    train_ds = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)

    def to_tensors(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        x, y = next(iter(loader))
        return x, y

    x_train_img, y_train = to_tensors(train_ds)  # (N, 1, 28, 28)
    x_test_img, y_test = to_tensors(test_ds)

    if mode == "row":
        # (N, 1, 28, 28) → (N, 28, 28): T=28, d=28
        x_train_seq = x_train_img.squeeze(1).float()
        x_test_seq = x_test_img.squeeze(1).float()
    elif mode in ("pixel", "pixel_permuted"):
        # (N, 1, 28, 28) → (N, 784, 1): T=784, d=1
        x_train_seq = x_train_img.reshape(-1, 784, 1).float()
        x_test_seq = x_test_img.reshape(-1, 784, 1).float()
        if mode == "pixel_permuted":
            perm = torch.randperm(784, generator=torch.Generator().manual_seed(seed))
            x_train_seq = x_train_seq[:, perm]
            x_test_seq = x_test_seq[:, perm]
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected 'row', 'pixel', or 'pixel_permuted'")

    x_train_flat = x_train_img.reshape(-1, 784).float()
    x_test_flat = x_test_img.reshape(-1, 784).float()

    return (x_train_seq, x_train_flat, y_train), (x_test_seq, x_test_flat, y_test)


# ------------------------------------------------------------------
# Recurrent NEF benchmark
# ------------------------------------------------------------------


def run_recurrent_nef(
    mode: str = "row",
    strategy: str = "greedy",
    n_neurons: int = 2000,
    d_state: int | None = None,
    activation: str = "relu",
    encoder_strategy: str = "hypersphere",
    solver: str = "tikhonov",
    solver_kwargs: dict | None = None,
    greedy_iters: int = 5,
    hybrid_iters: int = 10,
    hybrid_lr: float = 1e-3,
    hybrid_e2e_epochs: int = 20,
    hybrid_e2e_lr: float = 1e-3,
    e2e_epochs: int = 50,
    e2e_lr: float = 1e-3,
    e2e_batch: int = 256,
    tp_iters: int = 50,
    tp_lr: float = 1e-3,
    tp_eta: float = 0.1,
    tp_normalize: bool = True,
    tp_schedule: bool = False,
    loss: str = "mse",
    gain: float | tuple[float, float] = (0.5, 2.0),
    use_centers: bool = True,
    data_root: str = "./data",
    grad_clip: float | None = 1.0,
    seed: int | None = 0,
    state_target: str = "reconstruction",
    auxiliary_weight: float = 0.0,
    tp_project_targets: bool = False,
    device: str = "cpu",
    eval_batch_size: int = 2048,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a recurrent NEF benchmark on Sequential MNIST.

    Args:
        mode: ``"row"``, ``"pixel"``, or ``"pixel_permuted"``.
        strategy:
            ``"greedy"``, ``"hybrid"``, ``"target_prop"``, ``"e2e"``, or
            ``"hybrid_e2e"``.
    """
    solver_kwargs = solver_kwargs or {"alpha": 1e-2}
    set_benchmark_seed(seed)
    runtime_device = resolve_benchmark_device(device)
    data_seed = 0 if seed is None else seed

    (x_train_seq, _, y_train), (x_test_seq, _, y_test) = load_sequential_mnist(
        mode=mode, root=data_root, seed=data_seed
    )
    n_classes = 10
    targets = one_hot(y_train, n_classes)

    centers = x_train_seq if use_centers else None
    _, _, d_in = x_train_seq.shape
    layer = RecurrentNEFLayer(
        d_in=d_in,
        n_neurons=n_neurons,
        d_out=n_classes,
        d_state=d_state,
        activation=activation,
        encoder_strategy=encoder_strategy,
        gain=gain,
        centers=centers,
    ).to(runtime_device)

    x_train_seq = x_train_seq.to(runtime_device)
    x_test_seq = x_test_seq.to(runtime_device)
    y_train = y_train.to(runtime_device)
    y_test = y_test.to(runtime_device)
    targets = targets.to(runtime_device)

    if verbose:
        print(
            f"  RecNEF-{strategy}: mode={mode}, neurons={n_neurons}, "
            f"state_target={state_target}, device={runtime_device}",
            flush=True,
        )

    synchronize_if_cuda(runtime_device)
    t0 = time.perf_counter()
    if strategy == "greedy":
        layer.fit_greedy(
            x_train_seq,
            targets,
            n_iters=greedy_iters,
            solver=solver,
            state_target=state_target,
            auxiliary_weight=auxiliary_weight,
            **solver_kwargs,
        )
    elif strategy == "hybrid":
        layer.fit_hybrid(
            x_train_seq,
            targets,
            n_iters=hybrid_iters,
            lr=hybrid_lr,
            solver=solver,
            loss=loss,
            grad_clip=grad_clip,
            auxiliary_weight=auxiliary_weight,
            **solver_kwargs,
        )
    elif strategy == "hybrid_e2e":
        layer.fit_hybrid_e2e(
            x_train_seq,
            targets,
            n_iters=hybrid_iters,
            hybrid_lr=hybrid_lr,
            solver=solver,
            n_epochs=hybrid_e2e_epochs,
            e2e_lr=hybrid_e2e_lr,
            batch_size=e2e_batch,
            loss=loss,
            grad_clip=grad_clip,
            auxiliary_weight=auxiliary_weight,
            **solver_kwargs,
        )
    elif strategy == "e2e":
        layer.fit_end_to_end(
            x_train_seq,
            targets,
            n_epochs=e2e_epochs,
            lr=e2e_lr,
            batch_size=e2e_batch,
            loss=loss,
            grad_clip=grad_clip,
            greedy_iters=greedy_iters,
            auxiliary_weight=auxiliary_weight,
        )
    elif strategy == "target_prop":
        layer.fit_target_prop(
            x_train_seq,
            targets,
            n_iters=tp_iters,
            lr=tp_lr,
            eta=tp_eta,
            solver=solver,
            normalize_step=tp_normalize,
            schedule=tp_schedule,
            state_target=state_target,
            auxiliary_weight=auxiliary_weight,
            project_targets=tp_project_targets,
            **solver_kwargs,
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")
    synchronize_if_cuda(runtime_device)
    fit_time = time.perf_counter() - t0

    if verbose:
        print("    evaluating", flush=True)

    train_acc = batched_classification_accuracy(
        layer, x_train_seq, y_train, batch_size=eval_batch_size
    )
    test_acc = batched_classification_accuracy(
        layer, x_test_seq, y_test, batch_size=eval_batch_size
    )

    return BenchmarkResult(
        name=f"RecNEF-{strategy}",
        dataset=f"sMNIST-{mode}",
        n_neurons=n_neurons,
        activation=activation,
        encoder_strategy=encoder_strategy,
        solver=solver,
        solver_kwargs=solver_kwargs,
        metric_name="accuracy",
        train_metric=train_acc,
        test_metric=test_acc,
        fit_time=fit_time,
    )


# ------------------------------------------------------------------
# Streaming NEF benchmark (delay-line reservoir)
# ------------------------------------------------------------------


def run_streaming_nef(
    mode: str = "row",
    n_neurons: int = 2000,
    window_size: int = 5,
    activation: str = "abs",
    encoder_strategy: str = "hypersphere",
    gain: float | tuple[float, float] = (0.5, 2.0),
    use_centers: bool = True,
    alpha: float = 1e-2,
    batch_size: int = 1000,
    data_root: str = "./data",
    seed: int | None = 0,
    solve_mode: str = "woodbury",
    device: str = "cpu",
    eval_batch_size: int = 2048,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a streaming NEF benchmark on Sequential MNIST.

    Uses a delay-line reservoir: overlapping windows of consecutive timesteps
    are encoded through random NEF neurons, mean-pooled over time, and decoded
    to class labels.

    Args:
        mode:            ``"row"``, ``"pixel"``, or ``"pixel_permuted"``.
        solve_mode:      Training strategy:
                         ``"woodbury"`` — streaming with Woodbury rank-k
                         updates (float64 inverse, true online learning).
                         ``"accumulate"`` — streaming accumulate + final
                         solve (float32-safe, GPU-friendly).
                         ``"batch"`` — full-dataset fit in one shot.
    """
    if solve_mode not in ("woodbury", "accumulate", "batch"):
        raise ValueError(f"Unknown solve_mode {solve_mode!r}")
    set_benchmark_seed(seed)
    runtime_device = resolve_benchmark_device(device)
    data_seed = 0 if seed is None else seed

    (x_train_seq, _, y_train), (x_test_seq, _, y_test) = load_sequential_mnist(
        mode=mode, root=data_root, seed=data_seed
    )
    n_classes = 10
    targets = one_hot(y_train, n_classes)

    _, _, d_in = x_train_seq.shape
    rng = torch.Generator().manual_seed(data_seed)
    clf = StreamingNEFClassifier(
        d_timestep=d_in,
        n_neurons=n_neurons,
        d_out=n_classes,
        window_size=window_size,
        activation=activation,
        encoder_strategy=encoder_strategy,
        gain=gain,
        rng=rng,
        centers=x_train_seq if use_centers else None,
    ).to(runtime_device)

    x_train_seq = x_train_seq.to(runtime_device)
    x_test_seq = x_test_seq.to(runtime_device)
    y_train = y_train.to(runtime_device)
    y_test = y_test.to(runtime_device)
    targets = targets.to(runtime_device)

    if verbose:
        print(
            f"  StreamNEF-{solve_mode}: mode={mode}, neurons={n_neurons}, "
            f"window={window_size}, alpha={alpha}, device={runtime_device}",
            flush=True,
        )

    synchronize_if_cuda(runtime_device)
    t0 = time.perf_counter()
    if solve_mode == "woodbury":
        n_batches = (len(x_train_seq) + batch_size - 1) // batch_size
        log_every = max(1, n_batches // 10)
        for batch_idx, i in enumerate(range(0, len(x_train_seq), batch_size), start=1):
            clf.continuous_fit(
                x_train_seq[i : i + batch_size],
                targets[i : i + batch_size],
                alpha=alpha,
            )
            if verbose and (
                batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0
            ):
                print(f"    batch {batch_idx}/{n_batches}", flush=True)
        if verbose:
            print("    refreshing inverse", flush=True)
        clf.refresh_inverse(alpha=alpha)
    elif solve_mode == "accumulate":
        n_batches = (len(x_train_seq) + batch_size - 1) // batch_size
        log_every = max(1, n_batches // 10)
        for batch_idx, i in enumerate(range(0, len(x_train_seq), batch_size), start=1):
            clf.accumulate(
                x_train_seq[i : i + batch_size],
                targets[i : i + batch_size],
            )
            if verbose and (
                batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0
            ):
                print(f"    batch {batch_idx}/{n_batches}", flush=True)
        if verbose:
            print("    solving decoders (float32)", flush=True)
        clf.solve(alpha=alpha)
    else:
        if verbose:
            print("    solving batch decoder", flush=True)
        clf.fit(x_train_seq, targets, alpha=alpha)
    synchronize_if_cuda(runtime_device)
    fit_time = time.perf_counter() - t0

    if verbose:
        print("    evaluating", flush=True)
    train_acc = batched_classification_accuracy(
        clf, x_train_seq, y_train, batch_size=eval_batch_size
    )
    test_acc = batched_classification_accuracy(clf, x_test_seq, y_test, batch_size=eval_batch_size)

    return BenchmarkResult(
        name=f"StreamNEF-{solve_mode}",
        dataset=f"sMNIST-{mode}",
        n_neurons=n_neurons,
        activation=activation,
        encoder_strategy=encoder_strategy,
        solver=f"{solve_mode}-α{alpha}",
        solver_kwargs={"alpha": alpha, "window_size": window_size, "batch_size": batch_size},
        metric_name="accuracy",
        train_metric=train_acc,
        test_metric=test_acc,
        fit_time=fit_time,
    )


# ------------------------------------------------------------------
# LSTM baseline
# ------------------------------------------------------------------


class _LSTMClassifier(nn.Module):
    """Minimal LSTM baseline for sequential classification."""

    def __init__(self, d_in: int, hidden_size: int, n_classes: int, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def run_lstm_baseline(
    mode: str = "row",
    hidden_size: int = 128,
    n_layers: int = 1,
    n_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    data_root: str = "./data",
    seed: int | None = 0,
    device: str = "cpu",
    eval_batch_size: int = 2048,
    verbose: bool = False,
) -> BenchmarkResult:
    """Train an LSTM baseline on Sequential MNIST."""
    set_benchmark_seed(seed)
    runtime_device = resolve_benchmark_device(device)
    data_seed = 0 if seed is None else seed
    (x_train_seq, _, y_train), (x_test_seq, _, y_test) = load_sequential_mnist(
        mode=mode, root=data_root, seed=data_seed
    )
    _, _, d_in = x_train_seq.shape

    model = _LSTMClassifier(d_in, hidden_size, n_classes=10, n_layers=n_layers).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(x_train_seq, y_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(data_seed),
    )

    if verbose:
        print(
            f"  LSTM-{hidden_size}: mode={mode}, epochs={n_epochs}, batch={batch_size}, "
            f"device={runtime_device}",
            flush=True,
        )

    synchronize_if_cuda(runtime_device)
    t0 = time.perf_counter()
    log_every = max(1, n_epochs // 10)
    for epoch in range(1, n_epochs + 1):
        for xb, yb in loader:
            xb = xb.to(runtime_device)
            yb = yb.to(runtime_device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and (epoch == 1 or epoch == n_epochs or epoch % log_every == 0):
            print(f"    epoch {epoch}/{n_epochs}", flush=True)
    synchronize_if_cuda(runtime_device)
    fit_time = time.perf_counter() - t0

    if verbose:
        print("    evaluating", flush=True)

    # LSTM memory scales with sequence length — reduce eval batch for long
    # sequences to avoid GPU OOM (e.g. pixel mode: T=784).
    T = x_train_seq.shape[1]
    lstm_eval_batch = max(64, eval_batch_size * 28 // max(T, 1))

    train_acc = batched_classification_accuracy(
        model,
        x_train_seq,
        y_train,
        batch_size=lstm_eval_batch,
        move_to_device=runtime_device,
    )
    test_acc = batched_classification_accuracy(
        model,
        x_test_seq,
        y_test,
        batch_size=lstm_eval_batch,
        move_to_device=runtime_device,
    )

    return BenchmarkResult(
        name=f"LSTM-{hidden_size}",
        dataset=f"sMNIST-{mode}",
        n_neurons=hidden_size * n_layers,
        activation="tanh/sigmoid",
        encoder_strategy="learned",
        solver="adam",
        solver_kwargs={"lr": lr, "epochs": n_epochs},
        metric_name="accuracy",
        train_metric=train_acc,
        test_metric=test_acc,
        fit_time=fit_time,
    )


# ------------------------------------------------------------------
# Speech Commands v2 dataset
# ------------------------------------------------------------------

_SPEECH_COMMANDS_LABELS = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]


def load_speech_commands(
    root: str = "./data",
    n_mfcc: int = 40,
    seed: int = 0,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], int]:
    """Load Google Speech Commands v2 as MFCC feature sequences.

    Extracts MFCC features from 1-second audio clips (padded/trimmed to
    16kHz).  Merges the training and validation splits for training; uses
    the standard test split for evaluation.  MFCCs are normalized to
    zero-mean unit-variance per coefficient (statistics from training set).

    Args:
        root:   Data download directory.
        n_mfcc: Number of MFCC coefficients per frame.
        seed:   Random seed for shuffling training data.

    Returns:
        ((x_train, y_train), (x_test, y_test), n_classes)
        where x shapes are (N, T, n_mfcc) with T ≈ 101 frames.
    """
    import torchaudio

    label_to_idx = {label: i for i, label in enumerate(_SPEECH_COMMANDS_LABELS)}
    n_classes = len(_SPEECH_COMMANDS_LABELS)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
    )
    target_len = 16000  # 1 second at 16kHz

    def _process_split(subset: str) -> tuple[Tensor, Tensor]:
        ds = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset=subset)
        features, labels = [], []
        for i, (waveform, sr, label, *_) in enumerate(ds):
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            if waveform.shape[1] < target_len:
                waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_len]
            feat = mfcc_transform(waveform).squeeze(0)  # (n_mfcc, T)
            features.append(feat)
            labels.append(label_to_idx[label])
            if (i + 1) % 20000 == 0:
                print(f"    {subset}: {i + 1} samples", flush=True)
        x = torch.stack(features).permute(0, 2, 1).float()  # (N, T, n_mfcc)
        y = torch.tensor(labels, dtype=torch.long)
        print(f"    {subset}: {len(y)} samples total", flush=True)
        return x, y

    print("  Loading Speech Commands v2 (MFCC extraction)...", flush=True)
    x_train, y_train = _process_split("training")
    x_val, y_val = _process_split("validation")
    x_test, y_test = _process_split("testing")

    # Merge train + validation for training
    x_train = torch.cat([x_train, x_val], dim=0)
    y_train = torch.cat([y_train, y_val], dim=0)

    # Normalize per-coefficient to zero-mean unit-variance (training stats)
    mean = x_train.mean(dim=(0, 1), keepdim=True)
    std = x_train.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Shuffle training data
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x_train), generator=rng)
    x_train, y_train = x_train[perm], y_train[perm]

    return (x_train, y_train), (x_test, y_test), n_classes


# ------------------------------------------------------------------
# Sequential CIFAR-10 dataset
# ------------------------------------------------------------------


def load_sequential_cifar10(
    mode: str = "row",
    root: str = "./data",
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """Load CIFAR-10 as sequential data for temporal classification.

    Args:
        mode: ``"row"`` — 32 rows of 96 features (T=32, d=96, channels
              interleaved per row).
              ``"pixel"`` — 1024 pixels with 3 channel features (T=1024, d=3).

    Returns:
        ((x_train_seq, y_train), (x_test_seq, y_test))
    """
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)

    def _to_tensors(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        x, y = next(iter(loader))
        return x, y

    x_train_img, y_train = _to_tensors(train_ds)  # (N, 3, 32, 32)
    x_test_img, y_test = _to_tensors(test_ds)

    if mode == "row":
        # (N, 3, 32, 32) → (N, 32, 96): T=32, d=96
        x_train_seq = x_train_img.permute(0, 2, 3, 1).reshape(-1, 32, 96).float()
        x_test_seq = x_test_img.permute(0, 2, 3, 1).reshape(-1, 32, 96).float()
    elif mode == "pixel":
        # (N, 3, 32, 32) → (N, 1024, 3): T=1024, d=3
        x_train_seq = x_train_img.permute(0, 2, 3, 1).reshape(-1, 1024, 3).float()
        x_test_seq = x_test_img.permute(0, 2, 3, 1).reshape(-1, 1024, 3).float()
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected 'row' or 'pixel'")

    return (x_train_seq, y_train), (x_test_seq, y_test)


# ------------------------------------------------------------------
# Helpers for data-adapted encoder strategies in StreamNEF runners
# ------------------------------------------------------------------


def _prepare_encoder_kwargs(
    encoder_strategy: str,
    encoder_kwargs: dict | None,
    x_train_seq: Tensor,
    window_size: int,
    d_in: int,
    seed: int,
    max_seqs: int = 2000,
    max_windows: int = 20000,
) -> dict | None:
    """Build encoder_kwargs for data-adapted strategies (e.g. whitened).

    For ``whitened`` encoders, computes a subsample of delay-line windowed
    features to estimate PCA.  Uses a dedicated RNG so the main model seed
    is unaffected.
    """
    ekw = dict(encoder_kwargs or {})
    if encoder_strategy == "whitened" and "train_data" not in ekw:
        rng_sub = torch.Generator().manual_seed(seed + 9999)
        n_sub = min(max_seqs, len(x_train_seq))
        sub_idx = torch.randperm(len(x_train_seq), generator=rng_sub)[:n_sub]
        sub_seq = x_train_seq[sub_idx]
        K = window_size
        padded = F.pad(sub_seq, (0, 0, K - 1, 0))
        windows = padded.unfold(1, K, 1).permute(0, 1, 3, 2)
        delay_flat = windows.reshape(-1, K * d_in)
        n_win = min(max_windows, delay_flat.shape[0])
        win_idx = torch.randperm(delay_flat.shape[0], generator=rng_sub)[:n_win]
        ekw["train_data"] = delay_flat[win_idx]
    return ekw if ekw else None


# ------------------------------------------------------------------
# StreamNEF and LSTM on Speech Commands
# ------------------------------------------------------------------


def run_streaming_nef_speech(
    n_neurons: int = 4000,
    window_size: int = 10,
    activation: str = "abs",
    encoder_strategy: str = "hypersphere",
    encoder_kwargs: dict | None = None,
    gain: float | tuple[float, float] = (0.5, 2.0),
    use_centers: bool = True,
    alpha: float = 1e-2,
    batch_size: int = 500,
    n_mfcc: int = 40,
    data_root: str = "./data",
    seed: int | None = 0,
    solve_mode: str = "accumulate",
    device: str = "cpu",
    eval_batch_size: int = 2048,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run StreamNEF on Google Speech Commands v2 (MFCC features)."""
    if solve_mode not in ("woodbury", "accumulate", "batch"):
        raise ValueError(f"Unknown solve_mode {solve_mode!r}")
    set_benchmark_seed(seed)
    runtime_device = resolve_benchmark_device(device)
    data_seed = 0 if seed is None else seed

    (x_train_seq, y_train), (x_test_seq, y_test), n_classes = load_speech_commands(
        root=data_root, n_mfcc=n_mfcc, seed=data_seed
    )
    targets = one_hot(y_train, n_classes)

    _, _, d_in = x_train_seq.shape
    ekw = _prepare_encoder_kwargs(
        encoder_strategy, encoder_kwargs, x_train_seq, window_size, d_in, data_seed
    )
    rng = torch.Generator().manual_seed(data_seed)
    clf = StreamingNEFClassifier(
        d_timestep=d_in,
        n_neurons=n_neurons,
        d_out=n_classes,
        window_size=window_size,
        activation=activation,
        encoder_strategy=encoder_strategy,
        gain=gain,
        rng=rng,
        centers=x_train_seq if use_centers else None,
        encoder_kwargs=ekw,
    ).to(runtime_device)

    x_train_seq = x_train_seq.to(runtime_device)
    x_test_seq = x_test_seq.to(runtime_device)
    y_train = y_train.to(runtime_device)
    y_test = y_test.to(runtime_device)
    targets = targets.to(runtime_device)

    if verbose:
        print(
            f"  StreamNEF-{solve_mode}: dataset=SpeechCmds, neurons={n_neurons}, "
            f"window={window_size}, mfcc={n_mfcc}, alpha={alpha}, device={runtime_device}",
            flush=True,
        )

    synchronize_if_cuda(runtime_device)
    t0 = time.perf_counter()
    if solve_mode == "woodbury":
        n_batches = (len(x_train_seq) + batch_size - 1) // batch_size
        log_every = max(1, n_batches // 10)
        for batch_idx, i in enumerate(range(0, len(x_train_seq), batch_size), start=1):
            clf.continuous_fit(
                x_train_seq[i : i + batch_size],
                targets[i : i + batch_size],
                alpha=alpha,
            )
            if verbose and (
                batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0
            ):
                print(f"    batch {batch_idx}/{n_batches}", flush=True)
        if verbose:
            print("    refreshing inverse", flush=True)
        clf.refresh_inverse(alpha=alpha)
    elif solve_mode == "accumulate":
        n_batches = (len(x_train_seq) + batch_size - 1) // batch_size
        log_every = max(1, n_batches // 10)
        for batch_idx, i in enumerate(range(0, len(x_train_seq), batch_size), start=1):
            clf.accumulate(
                x_train_seq[i : i + batch_size],
                targets[i : i + batch_size],
            )
            if verbose and (
                batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0
            ):
                print(f"    batch {batch_idx}/{n_batches}", flush=True)
        if verbose:
            print("    solving decoders", flush=True)
        clf.solve(alpha=alpha)
    else:
        if verbose:
            print("    solving batch decoder", flush=True)
        clf.fit(x_train_seq, targets, alpha=alpha)
    synchronize_if_cuda(runtime_device)
    fit_time = time.perf_counter() - t0

    if verbose:
        print("    evaluating", flush=True)
    train_acc = batched_classification_accuracy(
        clf, x_train_seq, y_train, batch_size=eval_batch_size
    )
    test_acc = batched_classification_accuracy(clf, x_test_seq, y_test, batch_size=eval_batch_size)

    return BenchmarkResult(
        name=f"StreamNEF-{solve_mode}",
        dataset=f"SpeechCmds-mfcc{n_mfcc}",
        n_neurons=n_neurons,
        activation=activation,
        encoder_strategy=encoder_strategy,
        solver=f"{solve_mode}-α{alpha}",
        solver_kwargs={"alpha": alpha, "window_size": window_size, "batch_size": batch_size},
        metric_name="accuracy",
        train_metric=train_acc,
        test_metric=test_acc,
        fit_time=fit_time,
    )


def run_lstm_speech(
    hidden_size: int = 128,
    n_layers: int = 1,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    n_mfcc: int = 40,
    data_root: str = "./data",
    seed: int | None = 0,
    device: str = "cpu",
    eval_batch_size: int = 2048,
    verbose: bool = False,
) -> BenchmarkResult:
    """Train an LSTM on Google Speech Commands v2 (MFCC features)."""
    set_benchmark_seed(seed)
    runtime_device = resolve_benchmark_device(device)
    data_seed = 0 if seed is None else seed

    (x_train_seq, y_train), (x_test_seq, y_test), n_classes = load_speech_commands(
        root=data_root, n_mfcc=n_mfcc, seed=data_seed
    )
    _, _, d_in = x_train_seq.shape

    model = _LSTMClassifier(d_in, hidden_size, n_classes=n_classes, n_layers=n_layers).to(
        runtime_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(x_train_seq, y_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(data_seed),
    )

    if verbose:
        print(
            f"  LSTM-{hidden_size}: dataset=SpeechCmds, mfcc={n_mfcc}, "
            f"epochs={n_epochs}, batch={batch_size}, device={runtime_device}",
            flush=True,
        )

    synchronize_if_cuda(runtime_device)
    t0 = time.perf_counter()
    log_every = max(1, n_epochs // 10)
    for epoch in range(1, n_epochs + 1):
        for xb, yb in loader:
            xb = xb.to(runtime_device)
            yb = yb.to(runtime_device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and (epoch == 1 or epoch == n_epochs or epoch % log_every == 0):
            print(f"    epoch {epoch}/{n_epochs}", flush=True)
    synchronize_if_cuda(runtime_device)
    fit_time = time.perf_counter() - t0

    if verbose:
        print("    evaluating", flush=True)

    T = x_train_seq.shape[1]
    lstm_eval_batch = max(64, eval_batch_size * 28 // max(T, 1))

    train_acc = batched_classification_accuracy(
        model, x_train_seq, y_train, batch_size=lstm_eval_batch, move_to_device=runtime_device
    )
    test_acc = batched_classification_accuracy(
        model, x_test_seq, y_test, batch_size=lstm_eval_batch, move_to_device=runtime_device
    )

    return BenchmarkResult(
        name=f"LSTM-{hidden_size}",
        dataset=f"SpeechCmds-mfcc{n_mfcc}",
        n_neurons=hidden_size * n_layers,
        activation="tanh/sigmoid",
        encoder_strategy="learned",
        solver="adam",
        solver_kwargs={"lr": lr, "epochs": n_epochs},
        metric_name="accuracy",
        train_metric=train_acc,
        test_metric=test_acc,
        fit_time=fit_time,
    )


# ------------------------------------------------------------------
# StreamNEF and LSTM on Sequential CIFAR-10
# ------------------------------------------------------------------


def run_streaming_nef_scifar(
    mode: str = "row",
    n_neurons: int = 4000,
    window_size: int = 4,
    activation: str = "abs",
    encoder_strategy: str = "hypersphere",
    encoder_kwargs: dict | None = None,
    gain: float | tuple[float, float] = (0.5, 2.0),
    use_centers: bool = True,
    alpha: float = 1e-2,
    batch_size: int = 500,
    data_root: str = "./data",
    seed: int | None = 0,
    solve_mode: str = "accumulate",
    device: str = "cpu",
    eval_batch_size: int = 2048,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run StreamNEF on Sequential CIFAR-10."""
    if solve_mode not in ("woodbury", "accumulate", "batch"):
        raise ValueError(f"Unknown solve_mode {solve_mode!r}")
    set_benchmark_seed(seed)
    runtime_device = resolve_benchmark_device(device)
    data_seed = 0 if seed is None else seed

    (x_train_seq, y_train), (x_test_seq, y_test) = load_sequential_cifar10(
        mode=mode, root=data_root
    )
    n_classes = 10
    targets = one_hot(y_train, n_classes)

    _, _, d_in = x_train_seq.shape
    ekw = _prepare_encoder_kwargs(
        encoder_strategy, encoder_kwargs, x_train_seq, window_size, d_in, data_seed
    )
    rng = torch.Generator().manual_seed(data_seed)
    clf = StreamingNEFClassifier(
        d_timestep=d_in,
        n_neurons=n_neurons,
        d_out=n_classes,
        window_size=window_size,
        activation=activation,
        encoder_strategy=encoder_strategy,
        gain=gain,
        rng=rng,
        centers=x_train_seq if use_centers else None,
        encoder_kwargs=ekw,
    ).to(runtime_device)

    x_train_seq = x_train_seq.to(runtime_device)
    x_test_seq = x_test_seq.to(runtime_device)
    y_train = y_train.to(runtime_device)
    y_test = y_test.to(runtime_device)
    targets = targets.to(runtime_device)

    if verbose:
        print(
            f"  StreamNEF-{solve_mode}: dataset=sCIFAR-{mode}, neurons={n_neurons}, "
            f"window={window_size}, alpha={alpha}, device={runtime_device}",
            flush=True,
        )

    synchronize_if_cuda(runtime_device)
    t0 = time.perf_counter()
    if solve_mode == "woodbury":
        n_batches = (len(x_train_seq) + batch_size - 1) // batch_size
        log_every = max(1, n_batches // 10)
        for batch_idx, i in enumerate(range(0, len(x_train_seq), batch_size), start=1):
            clf.continuous_fit(
                x_train_seq[i : i + batch_size],
                targets[i : i + batch_size],
                alpha=alpha,
            )
            if verbose and (
                batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0
            ):
                print(f"    batch {batch_idx}/{n_batches}", flush=True)
        if verbose:
            print("    refreshing inverse", flush=True)
        clf.refresh_inverse(alpha=alpha)
    elif solve_mode == "accumulate":
        n_batches = (len(x_train_seq) + batch_size - 1) // batch_size
        log_every = max(1, n_batches // 10)
        for batch_idx, i in enumerate(range(0, len(x_train_seq), batch_size), start=1):
            clf.accumulate(
                x_train_seq[i : i + batch_size],
                targets[i : i + batch_size],
            )
            if verbose and (
                batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0
            ):
                print(f"    batch {batch_idx}/{n_batches}", flush=True)
        if verbose:
            print("    solving decoders", flush=True)
        clf.solve(alpha=alpha)
    else:
        if verbose:
            print("    solving batch decoder", flush=True)
        clf.fit(x_train_seq, targets, alpha=alpha)
    synchronize_if_cuda(runtime_device)
    fit_time = time.perf_counter() - t0

    if verbose:
        print("    evaluating", flush=True)
    train_acc = batched_classification_accuracy(
        clf, x_train_seq, y_train, batch_size=eval_batch_size
    )
    test_acc = batched_classification_accuracy(clf, x_test_seq, y_test, batch_size=eval_batch_size)

    return BenchmarkResult(
        name=f"StreamNEF-{solve_mode}",
        dataset=f"sCIFAR10-{mode}",
        n_neurons=n_neurons,
        activation=activation,
        encoder_strategy=encoder_strategy,
        solver=f"{solve_mode}-α{alpha}",
        solver_kwargs={"alpha": alpha, "window_size": window_size, "batch_size": batch_size},
        metric_name="accuracy",
        train_metric=train_acc,
        test_metric=test_acc,
        fit_time=fit_time,
    )


def run_lstm_scifar(
    mode: str = "row",
    hidden_size: int = 128,
    n_layers: int = 1,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    data_root: str = "./data",
    seed: int | None = 0,
    device: str = "cpu",
    eval_batch_size: int = 2048,
    verbose: bool = False,
) -> BenchmarkResult:
    """Train an LSTM on Sequential CIFAR-10."""
    set_benchmark_seed(seed)
    runtime_device = resolve_benchmark_device(device)
    data_seed = 0 if seed is None else seed

    (x_train_seq, y_train), (x_test_seq, y_test) = load_sequential_cifar10(
        mode=mode, root=data_root
    )
    n_classes = 10
    _, _, d_in = x_train_seq.shape

    model = _LSTMClassifier(d_in, hidden_size, n_classes=n_classes, n_layers=n_layers).to(
        runtime_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(x_train_seq, y_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(data_seed),
    )

    if verbose:
        print(
            f"  LSTM-{hidden_size}: dataset=sCIFAR-{mode}, "
            f"epochs={n_epochs}, batch={batch_size}, device={runtime_device}",
            flush=True,
        )

    synchronize_if_cuda(runtime_device)
    t0 = time.perf_counter()
    log_every = max(1, n_epochs // 10)
    for epoch in range(1, n_epochs + 1):
        for xb, yb in loader:
            xb = xb.to(runtime_device)
            yb = yb.to(runtime_device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and (epoch == 1 or epoch == n_epochs or epoch % log_every == 0):
            print(f"    epoch {epoch}/{n_epochs}", flush=True)
    synchronize_if_cuda(runtime_device)
    fit_time = time.perf_counter() - t0

    if verbose:
        print("    evaluating", flush=True)

    T = x_train_seq.shape[1]
    lstm_eval_batch = max(64, eval_batch_size * 28 // max(T, 1))

    train_acc = batched_classification_accuracy(
        model, x_train_seq, y_train, batch_size=lstm_eval_batch, move_to_device=runtime_device
    )
    test_acc = batched_classification_accuracy(
        model, x_test_seq, y_test, batch_size=lstm_eval_batch, move_to_device=runtime_device
    )

    return BenchmarkResult(
        name=f"LSTM-{hidden_size}",
        dataset=f"sCIFAR10-{mode}",
        n_neurons=hidden_size * n_layers,
        activation="tanh/sigmoid",
        encoder_strategy="learned",
        solver="adam",
        solver_kwargs={"lr": lr, "epochs": n_epochs},
        metric_name="accuracy",
        train_metric=train_acc,
        test_metric=test_acc,
        fit_time=fit_time,
    )


def build_recurrent_benchmark_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for recurrent benchmarks."""
    parser = argparse.ArgumentParser(description="Recurrent NEF benchmarks")
    parser.add_argument("--mode", default="row", choices=["row", "pixel", "pixel_permuted"])
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["greedy", "hybrid", "target_prop", "e2e", "hybrid_e2e"],
        choices=["greedy", "hybrid", "target_prop", "e2e", "hybrid_e2e"],
        help="Recurrent NEF strategies to benchmark",
    )
    parser.add_argument("--neurons", type=int, default=2000)
    parser.add_argument("--d-state", type=int, default=None)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--encoder", default="hypersphere")
    parser.add_argument("--solver", default="tikhonov")
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--greedy-iters", type=int, default=5)
    parser.add_argument("--hybrid-iters", type=int, default=10)
    parser.add_argument("--hybrid-lr", type=float, default=1e-3)
    parser.add_argument("--hybrid-e2e-epochs", type=int, default=20)
    parser.add_argument("--hybrid-e2e-lr", type=float, default=1e-3)
    parser.add_argument("--e2e-epochs", type=int, default=50)
    parser.add_argument("--e2e-lr", type=float, default=1e-3)
    parser.add_argument("--e2e-batch", type=int, default=256)
    parser.add_argument("--tp-iters", type=int, default=50)
    parser.add_argument("--tp-lr", type=float, default=1e-3)
    parser.add_argument("--tp-eta", type=float, default=0.1)
    parser.add_argument("--tp-no-normalize", action="store_true")
    parser.add_argument("--tp-schedule", action="store_true")
    parser.add_argument("--tp-project-targets", action="store_true")
    parser.add_argument(
        "--state-target",
        default="reconstruction",
        choices=["reconstruction", "predictive"],
        help="Decoded recurrent state target for greedy/target_prop (default: reconstruction)",
    )
    parser.add_argument(
        "--auxiliary-weight",
        type=float,
        default=0.0,
        help="Total auxiliary label-supervision weight assigned to pre-final timesteps",
    )
    parser.add_argument("--loss", default="mse", choices=["mse", "ce"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-centers", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible benchmark runs (default: 0)",
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--verbose", action="store_true", help="Print per-benchmark progress")
    parser.add_argument(
        "--eval-batch",
        type=int,
        default=2048,
        help="Batch size for chunked evaluation (default: 2048)",
    )
    parser.add_argument("--lstm", action="store_true", help="Also run the LSTM baseline")
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-epochs", type=int, default=20)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-batch", type=int, default=256)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Also run the streaming NEF (delay-line reservoir) benchmark",
    )
    parser.add_argument(
        "--streaming-solve-mode",
        nargs="+",
        default=["woodbury"],
        choices=["woodbury", "accumulate", "batch"],
        help="Solve mode(s) for streaming benchmark (default: woodbury)",
    )
    parser.add_argument("--streaming-window", type=int, default=5, help="Delay-line window size")
    parser.add_argument(
        "--streaming-neurons",
        type=int,
        default=None,
        help="Neurons for streaming (default: same as --neurons)",
    )
    parser.add_argument(
        "--streaming-batch", type=int, default=1000, help="Batch size for streaming continuous_fit"
    )
    parser.add_argument(
        "--save-json", type=Path, default=None, help="Write results to a JSON file"
    )
    parser.add_argument("--save-csv", type=Path, default=None, help="Write results to a CSV file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for recurrent benchmarks."""
    parser = build_recurrent_benchmark_parser()
    args = parser.parse_args(argv)

    results = []
    use_centers = not args.no_centers
    for strategy in args.strategies:
        print(f"Running RecNEF-{strategy} on sMNIST-{args.mode}...")
        results.append(
            run_recurrent_nef(
                mode=args.mode,
                strategy=strategy,
                n_neurons=args.neurons,
                d_state=args.d_state,
                activation=args.activation,
                encoder_strategy=args.encoder,
                solver=args.solver,
                solver_kwargs={"alpha": args.alpha},
                greedy_iters=args.greedy_iters,
                hybrid_iters=args.hybrid_iters,
                hybrid_lr=args.hybrid_lr,
                hybrid_e2e_epochs=args.hybrid_e2e_epochs,
                hybrid_e2e_lr=args.hybrid_e2e_lr,
                e2e_epochs=args.e2e_epochs,
                e2e_lr=args.e2e_lr,
                e2e_batch=args.e2e_batch,
                tp_iters=args.tp_iters,
                tp_lr=args.tp_lr,
                tp_eta=args.tp_eta,
                tp_normalize=not args.tp_no_normalize,
                tp_schedule=args.tp_schedule,
                tp_project_targets=args.tp_project_targets,
                state_target=args.state_target,
                auxiliary_weight=args.auxiliary_weight,
                loss=args.loss,
                use_centers=use_centers,
                data_root=args.data_root,
                grad_clip=args.grad_clip,
                seed=args.seed,
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=args.verbose,
            )
        )

    if args.lstm:
        print(f"Running LSTM baseline on sMNIST-{args.mode}...")
        results.append(
            run_lstm_baseline(
                mode=args.mode,
                hidden_size=args.lstm_hidden_size,
                n_layers=args.lstm_layers,
                n_epochs=args.lstm_epochs,
                lr=args.lstm_lr,
                batch_size=args.lstm_batch,
                data_root=args.data_root,
                seed=args.seed,
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=args.verbose,
            )
        )

    if args.streaming:
        stream_neurons = args.streaming_neurons or args.neurons
        for sm in args.streaming_solve_mode:
            print(f"Running StreamNEF-{sm} on sMNIST-{args.mode}...")
            results.append(
                run_streaming_nef(
                    mode=args.mode,
                    n_neurons=stream_neurons,
                    window_size=args.streaming_window,
                    alpha=args.alpha,
                    batch_size=args.streaming_batch,
                    use_centers=use_centers,
                    data_root=args.data_root,
                    seed=args.seed,
                    solve_mode=sm,
                    device=args.device,
                    eval_batch_size=args.eval_batch,
                    verbose=args.verbose,
                )
            )

    print()
    print(format_results(results))
    if args.save_json is not None:
        save_results_json(results, args.save_json)
    if args.save_csv is not None:
        save_results_csv(results, args.save_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
