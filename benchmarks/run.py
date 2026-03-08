"""Benchmark harness for NEF single-layer and multi-layer experiments."""

import time
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

from leenef.layers import NEFLayer
from leenef.networks import NEFNetwork


@dataclass
class BenchmarkResult:
    name: str
    dataset: str
    n_neurons: int
    activation: str
    encoder_strategy: str
    solver: str
    solver_kwargs: dict
    metric_name: str          # "accuracy" or "mse"
    train_metric: float
    test_metric: float
    fit_time: float           # seconds


def load_vision_dataset(name: str, root: str = "./data"):
    """Load a torchvision dataset, returning flat float tensors and labels."""
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    ds_cls = {
        "mnist": torchvision.datasets.MNIST,
        "fashion_mnist": torchvision.datasets.FashionMNIST,
        "cifar10": torchvision.datasets.CIFAR10,
    }[name]

    train = ds_cls(root, train=True, download=True, transform=transform)
    test = ds_cls(root, train=False, download=True, transform=transform)

    def to_tensors(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        x, y = next(iter(loader))
        return x.reshape(x.shape[0], -1).float(), y
    return to_tensors(train), to_tensors(test)


def load_regression_dataset(name: str = "california"):
    """Load a sklearn regression dataset as tensors."""
    if name == "california":
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        data = fetch_california_housing()
        x, y = torch.tensor(data.data, dtype=torch.float32), \
               torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)
        # Normalise features
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)
        idx = torch.randperm(len(x))
        split = int(0.8 * len(x))
        train_idx, test_idx = idx[:split], idx[split:]
        return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])
    raise ValueError(f"Unknown regression dataset: {name}")


def one_hot(labels: Tensor, n_classes: int) -> Tensor:
    """Convert integer labels to one-hot float targets."""
    return torch.zeros(len(labels), n_classes).scatter_(
        1, labels.unsqueeze(1), 1.0)


def classification_accuracy(pred: Tensor, labels: Tensor) -> float:
    return (pred.argmax(dim=1) == labels).float().mean().item()


def mse(pred: Tensor, targets: Tensor) -> float:
    return (pred - targets).pow(2).mean().item()


def run_nef_classification(
    dataset_name: str,
    n_neurons: int = 2000,
    activation: str = "abs",
    encoder_strategy: str = "hypersphere",
    solver: str = "tikhonov",
    solver_kwargs: dict | None = None,
    data_root: str = "./data",
    use_centers: bool = True,
) -> BenchmarkResult:
    """Run a single NEF classification benchmark."""
    solver_kwargs = solver_kwargs or {"alpha": 1e-2}

    (x_train, y_train), (x_test, y_test) = load_vision_dataset(
        dataset_name, root=data_root)
    n_classes = int(y_train.max().item()) + 1
    targets = one_hot(y_train, n_classes)

    centers = x_train if use_centers else None
    layer = NEFLayer(x_train.shape[1], n_neurons, n_classes,
                     activation=activation, encoder_strategy=encoder_strategy,
                     centers=centers)

    t0 = time.perf_counter()
    layer.fit(x_train, targets, solver=solver, **solver_kwargs)
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        train_acc = classification_accuracy(layer(x_train), y_train)
        test_acc = classification_accuracy(layer(x_test), y_test)

    return BenchmarkResult(
        name="NEFLayer", dataset=dataset_name, n_neurons=n_neurons,
        activation=activation, encoder_strategy=encoder_strategy,
        solver=solver, solver_kwargs=solver_kwargs,
        metric_name="accuracy",
        train_metric=train_acc, test_metric=test_acc,
        fit_time=fit_time,
    )


def run_nef_regression(
    dataset_name: str = "california",
    n_neurons: int = 2000,
    activation: str = "abs",
    encoder_strategy: str = "hypersphere",
    solver: str = "tikhonov",
    solver_kwargs: dict | None = None,
    use_centers: bool = True,
) -> BenchmarkResult:
    """Run a single NEF regression benchmark."""
    solver_kwargs = solver_kwargs or {"alpha": 1e-2}

    (x_train, y_train), (x_test, y_test) = load_regression_dataset(dataset_name)

    centers = x_train if use_centers else None
    layer = NEFLayer(x_train.shape[1], n_neurons, y_train.shape[1],
                     activation=activation, encoder_strategy=encoder_strategy,
                     centers=centers)

    t0 = time.perf_counter()
    layer.fit(x_train, y_train, solver=solver, **solver_kwargs)
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        train_mse = mse(layer(x_train), y_train)
        test_mse = mse(layer(x_test), y_test)

    return BenchmarkResult(
        name="NEFLayer", dataset=dataset_name, n_neurons=n_neurons,
        activation=activation, encoder_strategy=encoder_strategy,
        solver=solver, solver_kwargs=solver_kwargs,
        metric_name="mse",
        train_metric=train_mse, test_metric=test_mse,
        fit_time=fit_time,
    )


def run_linear_baseline(
    dataset_name: str,
    data_root: str = "./data",
) -> BenchmarkResult:
    """Run a simple linear model (no hidden layer) as baseline."""
    (x_train, y_train), (x_test, y_test) = load_vision_dataset(
        dataset_name, root=data_root)
    n_classes = int(y_train.max().item()) + 1
    targets = one_hot(y_train, n_classes)

    t0 = time.perf_counter()
    D = torch.linalg.pinv(x_train) @ targets
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        train_acc = classification_accuracy(x_train @ D, y_train)
        test_acc = classification_accuracy(x_test @ D, y_test)

    return BenchmarkResult(
        name="LinearBaseline", dataset=dataset_name, n_neurons=0,
        activation="none", encoder_strategy="none",
        solver="lstsq", solver_kwargs={},
        metric_name="accuracy",
        train_metric=train_acc, test_metric=test_acc,
        fit_time=fit_time,
    )


def run_nef_multi(
    dataset_name: str,
    strategy: str = "hybrid",
    hidden_neurons: list[int] | None = None,
    output_neurons: int = 2000,
    activation: str = "abs",
    encoder_strategy: str = "hypersphere",
    solver: str = "tikhonov",
    solver_kwargs: dict | None = None,
    hybrid_iters: int = 50,
    hybrid_lr: float = 1e-3,
    hybrid_alpha: float | None = None,
    hybrid_loss: str = "mse",
    hybrid_schedule: bool = False,
    hybrid_init: str = "random",
    hybrid_batch: int | None = None,
    hybrid_grad_steps: int = 1,
    hybrid_e2e_epochs: int = 20,
    hybrid_e2e_lr: float = 1e-3,
    e2e_epochs: int = 50,
    e2e_lr: float = 3e-3,
    e2e_batch: int = 256,
    data_root: str = "./data",
    use_centers: bool = True,
) -> BenchmarkResult:
    """Run a multi-layer NEFNetwork benchmark."""
    hidden_neurons = hidden_neurons or [1000]
    solver_kwargs = solver_kwargs or {"alpha": 1e-2}
    hybrid_kw = dict(solver_kwargs)
    if hybrid_alpha is not None:
        hybrid_kw["alpha"] = hybrid_alpha

    (x_train, y_train), (x_test, y_test) = load_vision_dataset(
        dataset_name, root=data_root)
    n_classes = int(y_train.max().item()) + 1
    targets = one_hot(y_train, n_classes)

    centers = x_train if use_centers else None
    net = NEFNetwork(x_train.shape[1], n_classes,
                     hidden_neurons=hidden_neurons,
                     output_neurons=output_neurons,
                     activation=activation,
                     encoder_strategy=encoder_strategy,
                     centers=centers)

    t0 = time.perf_counter()
    if strategy == "greedy":
        net.fit_greedy(x_train, targets, solver=solver, **solver_kwargs)
    elif strategy == "hybrid":
        net.fit_hybrid(x_train, targets, n_iters=hybrid_iters,
                       lr=hybrid_lr, solver=solver, loss=hybrid_loss,
                       schedule=hybrid_schedule, init=hybrid_init,
                       batch_size=hybrid_batch, grad_steps=hybrid_grad_steps,
                       centers=centers, **hybrid_kw)
    elif strategy == "hybrid_e2e":
        net.fit_hybrid_e2e(x_train, targets, n_iters=hybrid_iters,
                           hybrid_lr=hybrid_lr, solver=solver,
                           n_epochs=hybrid_e2e_epochs, e2e_lr=hybrid_e2e_lr,
                           batch_size=e2e_batch, loss="ce",
                           centers=centers, **hybrid_kw)
    elif strategy == "e2e":
        net.fit_end_to_end(x_train, targets, n_epochs=e2e_epochs,
                           lr=e2e_lr, batch_size=e2e_batch, loss="ce")
    fit_time = time.perf_counter() - t0

    with torch.no_grad():
        train_acc = classification_accuracy(net(x_train), y_train)
        test_acc = classification_accuracy(net(x_test), y_test)

    n_total = sum(hidden_neurons) + output_neurons
    return BenchmarkResult(
        name=f"NEFNet-{strategy}", dataset=dataset_name,
        n_neurons=n_total,
        activation=activation, encoder_strategy=encoder_strategy,
        solver=solver, solver_kwargs=solver_kwargs,
        metric_name="accuracy",
        train_metric=train_acc, test_metric=test_acc,
        fit_time=fit_time,
    )


def run_mlp_baseline(
    dataset_name: str,
    hidden_sizes: list[int] | None = None,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    data_root: str = "./data",
) -> BenchmarkResult:
    """Train a standard MLP via SGD as a comparison baseline."""
    hidden_sizes = hidden_sizes or [1000, 1000]

    (x_train, y_train), (x_test, y_test) = load_vision_dataset(
        dataset_name, root=data_root)
    n_classes = int(y_train.max().item()) + 1

    layers = []
    prev = x_train.shape[1]
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    model = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

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
        train_acc = classification_accuracy(model(x_train), y_train)
        test_acc = classification_accuracy(model(x_test), y_test)

    return BenchmarkResult(
        name="MLP", dataset=dataset_name,
        n_neurons=sum(hidden_sizes),
        activation="relu", encoder_strategy="learned",
        solver="adam", solver_kwargs={"lr": lr, "epochs": n_epochs},
        metric_name="accuracy",
        train_metric=train_acc, test_metric=test_acc,
        fit_time=fit_time,
    )


def format_results(results: list[BenchmarkResult]) -> str:
    """Format results as an aligned text table."""
    lines = []
    header = f"{'Name':<18} {'Dataset':<15} {'Neurons':>7} {'Act':<10} " \
             f"{'Encoder':<14} {'Train':>8} {'Test':>8} {'Time':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        fmt = ".4f" if r.metric_name == "mse" else ".2%"
        lines.append(
            f"{r.name:<18} {r.dataset:<15} {r.n_neurons:>7} "
            f"{r.activation:<10} {r.encoder_strategy:<14} "
            f"{r.train_metric:>8{fmt}} {r.test_metric:>8{fmt}} "
            f"{r.fit_time:>7.2f}s"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NEF benchmarks")
    parser.add_argument("--datasets", nargs="+",
                        default=["mnist", "fashion_mnist", "cifar10"],
                        help="Classification datasets to benchmark")
    parser.add_argument("--neurons", type=int, nargs="+", default=[2000],
                        help="Number of neurons (single-layer)")
    parser.add_argument("--activation", default="abs")
    parser.add_argument("--encoder", default="hypersphere")
    parser.add_argument("--solver", default="tikhonov")
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--hybrid-alpha", type=float, default=1e-3,
                        help="Solver alpha for hybrid training (default: 1e-3)")
    parser.add_argument("--hybrid-loss", default="mse",
                        choices=["mse", "ce"],
                        help="Loss for hybrid encoder gradients")
    parser.add_argument("--hybrid-schedule", action="store_true",
                        help="Use cosine LR schedule for hybrid")
    parser.add_argument("--hybrid-init", default="random",
                        choices=["random", "incremental"],
                        help="Hidden layer init strategy for hybrid")
    parser.add_argument("--hybrid-batch", type=int, default=None,
                        help="Mini-batch size for hybrid gradient steps")
    parser.add_argument("--hybrid-grad-steps", type=int, default=1,
                        help="Gradient steps per decoder solve in hybrid")
    parser.add_argument("--no-centers", action="store_true",
                        help="Disable data-driven biases (use random biases)")
    parser.add_argument("--regression", action="store_true",
                        help="Also run California Housing regression")
    parser.add_argument("--multi", action="store_true",
                        help="Run multi-layer benchmarks")
    parser.add_argument("--mlp", action="store_true",
                        help="Run MLP baseline")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    results = []
    use_centers = not args.no_centers
    for ds in args.datasets:
        # Linear baseline
        print(f"Running linear baseline on {ds}...")
        results.append(run_linear_baseline(ds, data_root=args.data_root))
        # NEF single-layer with each neuron count
        for n in args.neurons:
            print(f"Running NEF ({n} neurons) on {ds}...")
            results.append(run_nef_classification(
                ds, n_neurons=n, activation=args.activation,
                encoder_strategy=args.encoder, solver=args.solver,
                solver_kwargs={"alpha": args.alpha},
                data_root=args.data_root,
                use_centers=use_centers,
            ))

        if args.multi:
            for strat in ["greedy", "hybrid", "e2e"]:
                print(f"Running NEFNet-{strat} on {ds}...")
                results.append(run_nef_multi(
                    ds, strategy=strat,
                    hidden_neurons=[1000],
                    output_neurons=2000,
                    activation=args.activation,
                    encoder_strategy=args.encoder,
                    solver=args.solver,
                    solver_kwargs={"alpha": args.alpha},
                    hybrid_alpha=args.hybrid_alpha,
                    hybrid_loss=args.hybrid_loss,
                    hybrid_schedule=args.hybrid_schedule,
                    hybrid_init=args.hybrid_init,
                    hybrid_batch=args.hybrid_batch,
                    hybrid_grad_steps=args.hybrid_grad_steps,
                    data_root=args.data_root,
                    use_centers=use_centers,
                ))

        if args.mlp:
            print(f"Running MLP baseline on {ds}...")
            results.append(run_mlp_baseline(ds, data_root=args.data_root))

    if args.regression:
        print("Running NEF regression on California Housing...")
        for n in args.neurons:
            results.append(run_nef_regression(
                "california", n_neurons=n, activation=args.activation,
                encoder_strategy=args.encoder, solver=args.solver,
                solver_kwargs={"alpha": args.alpha},
                use_centers=use_centers,
            ))

    print()
    print(format_results(results))
