"""Benchmark harness for recurrent NEF on temporal classification tasks."""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# Ensure the checkout root and src layout are importable when run as a script.
_project_root = Path(__file__).resolve().parent.parent
for _path in (str(_project_root), str(_project_root / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmarks.run import (  # noqa: E402
    BenchmarkResult,
    classification_accuracy,
    format_results,
    one_hot,
    save_results_csv,
    save_results_json,
    set_benchmark_seed,
)
from leenef.recurrent import RecurrentNEFLayer  # noqa: E402


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
    )

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
    seed: int | None = 0,
) -> BenchmarkResult:
    """Train an LSTM baseline on Sequential MNIST."""
    set_benchmark_seed(seed)
    data_seed = 0 if seed is None else seed
    (x_train_seq, _, y_train), (x_test_seq, _, y_test) = load_sequential_mnist(
        mode=mode, root=data_root, seed=data_seed
    )
    _, _, d_in = x_train_seq.shape

    model = _LSTMClassifier(d_in, hidden_size, n_classes=10, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(x_train_seq, y_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(data_seed),
    )

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
    parser.add_argument("--lstm", action="store_true", help="Also run the LSTM baseline")
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-epochs", type=int, default=20)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-batch", type=int, default=256)
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
