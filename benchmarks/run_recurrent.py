"""Benchmark harness for recurrent NEF on temporal classification tasks."""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# Ensure project root is on path when run as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from benchmarks.run import BenchmarkResult, classification_accuracy, one_hot  # noqa: E402
from leenef.recurrent import RecurrentNEFLayer  # noqa: E402


def load_sequential_mnist(
    mode: str = "row",
    root: str = "./data",
    seed: int = 42,
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
    activation: str = "abs",
    encoder_strategy: str = "hypersphere",
    solver: str = "tikhonov",
    solver_kwargs: dict | None = None,
    greedy_iters: int = 5,
    hybrid_iters: int = 10,
    hybrid_lr: float = 1e-3,
    e2e_epochs: int = 50,
    e2e_lr: float = 1e-3,
    e2e_batch: int = 256,
    loss: str = "mse",
    gain: float | tuple[float, float] = (0.5, 2.0),
    data_root: str = "./data",
) -> BenchmarkResult:
    """Run a recurrent NEF benchmark on Sequential MNIST.

    Args:
        mode: ``"row"``, ``"pixel"``, or ``"pixel_permuted"``.
        strategy: ``"greedy"``, ``"hybrid"``, or ``"e2e"``.
    """
    solver_kwargs = solver_kwargs or {"alpha": 1e-2}

    (x_train_seq, _, y_train), (x_test_seq, _, y_test) = load_sequential_mnist(
        mode=mode, root=data_root
    )
    n_classes = 10
    targets = one_hot(y_train, n_classes)

    _, _, d_in = x_train_seq.shape
    layer = RecurrentNEFLayer(
        d_in=d_in,
        n_neurons=n_neurons,
        d_out=n_classes,
        d_state=d_state,
        activation=activation,
        encoder_strategy=encoder_strategy,
        gain=gain,
    )

    t0 = time.perf_counter()
    if strategy == "greedy":
        layer.fit_greedy(
            x_train_seq, targets, n_iters=greedy_iters, solver=solver, **solver_kwargs
        )
    elif strategy == "hybrid":
        layer.fit_hybrid(
            x_train_seq,
            targets,
            n_iters=hybrid_iters,
            lr=hybrid_lr,
            solver=solver,
            loss=loss,
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
            greedy_iters=greedy_iters,
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        train_acc = classification_accuracy(layer(x_train_seq), y_train)
        test_acc = classification_accuracy(layer(x_test_seq), y_test)

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
) -> BenchmarkResult:
    """Train an LSTM baseline on Sequential MNIST."""
    (x_train_seq, _, y_train), (x_test_seq, _, y_test) = load_sequential_mnist(
        mode=mode, root=data_root
    )
    _, _, d_in = x_train_seq.shape

    model = _LSTMClassifier(d_in, hidden_size, n_classes=10, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(x_train_seq, y_train)
    g = torch.Generator()
    g.manual_seed(0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    t0 = time.perf_counter()
    for _ in range(n_epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        train_acc = classification_accuracy(model(x_train_seq), y_train)
        test_acc = classification_accuracy(model(x_test_seq), y_test)

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
