"""Continual learning benchmarks for NEF.

Demonstrates the zero-forgetting property of NEF's analytical solve
by comparing against MLP fine-tuning and EWC baselines on standard
continual learning scenarios.

Datasets:
  mnist    -- MNIST (28×28, 10 classes, 784-dim)
  cifar10  -- CIFAR-10 (32×32×3, 10 classes, 3072-dim)
  cifar100 -- CIFAR-100 (32×32×3, 100 classes, 3072-dim)

Scenarios:
  split    -- Class-incremental: tasks with disjoint class subsets
  permuted -- Domain-incremental: tasks with random pixel permutations

Usage:
  python benchmarks/run_continual.py --scenario both --seed 0
  python benchmarks/run_continual.py --dataset cifar10 --scenario split --n-neurons 5000 --seed 0
  python benchmarks/run_continual.py --dataset cifar100 --scenario split --n-neurons 5000 --seed 0
  python benchmarks/run_continual.py --scenario permuted --n-tasks-permuted 10 --seed 0
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from leenef.activations import make_activation  # noqa: E402
from leenef.encoders import make_encoders  # noqa: E402
from leenef.layers import NEFLayer  # noqa: E402
from leenef.solvers import solve_from_normal_equations  # noqa: E402

# ── Utilities ─────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot(labels: Tensor, n_classes: int) -> Tensor:
    return torch.zeros(len(labels), n_classes).scatter_(1, labels.unsqueeze(1), 1.0)


def load_mnist(root: str = "./data"):
    """Load MNIST as flat float tensors and integer labels."""
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    train = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)

    def to_tensors(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        x, y = next(iter(loader))
        return x.reshape(x.shape[0], -1).float(), y

    return to_tensors(train), to_tensors(test)


def load_cifar10(root: str = "./data"):
    """Load CIFAR-10 as flat float tensors and integer labels."""
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)

    def to_tensors(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        x, y = next(iter(loader))
        return x.reshape(x.shape[0], -1).float(), y

    return to_tensors(train), to_tensors(test)


def load_cifar100(root: str = "./data"):
    """Load CIFAR-100 as flat float tensors and integer labels."""
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    train = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform)

    def to_tensors(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        x, y = next(iter(loader))
        return x.reshape(x.shape[0], -1).float(), y

    return to_tensors(train), to_tensors(test)


DATASET_LOADERS = {
    "mnist": load_mnist,
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
}

DATASET_N_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
}

DATASET_DEFAULT_NEURONS = {
    "mnist": 2000,
    "cifar10": 5000,
    "cifar100": 5000,
}


# ── Task construction ─────────────────────────────────────────────────


def split_class_tasks(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    n_classes: int,
    classes_per_task: int = 2,
) -> list[dict]:
    """Split a dataset into class-incremental tasks with disjoint class subsets."""
    n_tasks = n_classes // classes_per_task
    tasks = []
    for t in range(n_tasks):
        lo = t * classes_per_task
        hi = (t + 1) * classes_per_task
        task_classes = list(range(lo, hi))
        train_mask = torch.zeros(len(y_train), dtype=torch.bool)
        test_mask = torch.zeros(len(y_test), dtype=torch.bool)
        for c in task_classes:
            train_mask |= y_train == c
            test_mask |= y_test == c
        tasks.append(
            {
                "name": f"classes {task_classes}",
                "classes": task_classes,
                "x_train": x_train[train_mask],
                "y_train": y_train[train_mask],
                "x_test": x_test[test_mask],
                "y_test": y_test[test_mask],
            }
        )
    return tasks


def permuted_tasks(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    n_tasks: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Create permuted-pixel tasks (same classes, permuted input features)."""
    n_classes = int(y_train.max().item()) + 1
    tasks = [
        {
            "name": "original",
            "classes": list(range(n_classes)),
            "x_train": x_train.clone(),
            "y_train": y_train.clone(),
            "x_test": x_test.clone(),
            "y_test": y_test.clone(),
        }
    ]
    rng = torch.Generator().manual_seed(seed + 1000)
    for i in range(1, n_tasks):
        perm = torch.randperm(x_train.shape[1], generator=rng)
        tasks.append(
            {
                "name": f"perm-{i}",
                "classes": list(range(n_classes)),
                "x_train": x_train[:, perm],
                "y_train": y_train.clone(),
                "x_test": x_test[:, perm],
                "y_test": y_test.clone(),
            }
        )
    return tasks


# ── Evaluation ────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_on_tasks(model: nn.Module, tasks: list[dict]) -> list[float]:
    """Return per-task test accuracy."""
    was_training = model.training
    model.eval()
    accs = []
    for task in tasks:
        pred = model(task["x_test"])
        acc = (pred.argmax(dim=1) == task["y_test"]).float().mean().item()
        accs.append(acc)
    if was_training:
        model.train()
    return accs


# ── Result container ──────────────────────────────────────────────────


@dataclass
class ContinualResult:
    method: str
    scenario: str
    n_tasks: int
    accuracy_matrix: list[list[float]]
    fit_times: list[float]
    n_neurons: int = 0
    config: dict = field(default_factory=dict)

    @property
    def final_average_accuracy(self) -> float:
        """Mean accuracy across all tasks after the last training step."""
        row = self.accuracy_matrix[-1]
        return sum(row) / len(row)

    @property
    def forgetting(self) -> float:
        """Mean forgetting: peak accuracy minus final accuracy per task."""
        n_rows = len(self.accuracy_matrix)
        if n_rows <= 1:
            return 0.0
        n = len(self.accuracy_matrix[-1])
        total = 0.0
        for j in range(n):
            peak = max(self.accuracy_matrix[i][j] for i in range(n_rows))
            final = self.accuracy_matrix[-1][j]
            total += peak - final
        return total / n

    @property
    def backward_transfer(self) -> float:
        """Mean change in old-task accuracy after learning new tasks."""
        n_rows = len(self.accuracy_matrix)
        n = len(self.accuracy_matrix[-1])
        if n_rows <= 1 or n <= 1:
            return 0.0
        total = 0.0
        count = 0
        for j in range(min(n - 1, n_rows - 1)):
            total += self.accuracy_matrix[-1][j] - self.accuracy_matrix[j][j]
            count += 1
        return total / count if count > 0 else 0.0

    @property
    def total_time(self) -> float:
        return sum(self.fit_times)


# ── NEF methods ───────────────────────────────────────────────────────


def run_nef_accumulate(
    tasks: list[dict],
    scenario: str,
    *,
    n_neurons: int = 2000,
    alpha: float = 1e-2,
    use_centers: str = "none",
    seed: int = 0,
) -> ContinualResult:
    """NEF with accumulative partial_fit — our main method.

    Args:
        use_centers: 'none' | 'first_task' | 'all_tasks'
    """
    set_seed(seed)
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1
    d_in = tasks[0]["x_train"].shape[1]

    if use_centers == "first_task":
        centers = tasks[0]["x_train"]
    elif use_centers == "all_tasks":
        centers = torch.cat([t["x_train"] for t in tasks])
    else:
        centers = None

    layer = NEFLayer(
        d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0), centers=centers
    )

    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []

    for task in tasks:
        targets = one_hot(task["y_train"], n_classes)
        t0 = time.perf_counter()
        layer.partial_fit(task["x_train"], targets)
        layer.solve_accumulated(alpha=alpha)
        fit_times.append(time.perf_counter() - t0)
        accuracy_matrix.append(evaluate_on_tasks(layer, tasks))

    return ContinualResult(
        method=f"NEF-accumulate (centers={use_centers})",
        scenario=scenario,
        n_tasks=len(tasks),
        accuracy_matrix=accuracy_matrix,
        fit_times=fit_times,
        n_neurons=n_neurons,
        config={"alpha": alpha, "use_centers": use_centers},
    )


def run_nef_reset(
    tasks: list[dict],
    scenario: str,
    *,
    n_neurons: int = 2000,
    alpha: float = 1e-2,
    seed: int = 0,
) -> ContinualResult:
    """NEF with accumulators reset between tasks — forgetting control."""
    set_seed(seed)
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1
    d_in = tasks[0]["x_train"].shape[1]

    layer = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))

    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []

    for task in tasks:
        targets = one_hot(task["y_train"], n_classes)
        t0 = time.perf_counter()
        layer.reset_accumulators()
        layer.partial_fit(task["x_train"], targets)
        layer.solve_accumulated(alpha=alpha)
        fit_times.append(time.perf_counter() - t0)
        accuracy_matrix.append(evaluate_on_tasks(layer, tasks))

    return ContinualResult(
        method="NEF-reset (forgetting control)",
        scenario=scenario,
        n_tasks=len(tasks),
        accuracy_matrix=accuracy_matrix,
        fit_times=fit_times,
        n_neurons=n_neurons,
        config={"alpha": alpha},
    )


def run_nef_joint(
    tasks: list[dict],
    scenario: str,
    *,
    n_neurons: int = 2000,
    alpha: float = 1e-2,
    seed: int = 0,
) -> ContinualResult:
    """NEF trained on all tasks jointly — upper bound."""
    set_seed(seed)
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1
    d_in = tasks[0]["x_train"].shape[1]

    x_all = torch.cat([t["x_train"] for t in tasks])
    y_all = torch.cat([t["y_train"] for t in tasks])
    targets_all = one_hot(y_all, n_classes)

    layer = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))

    t0 = time.perf_counter()
    layer.fit(x_all, targets_all, alpha=alpha)
    fit_time = time.perf_counter() - t0

    accs = evaluate_on_tasks(layer, tasks)

    return ContinualResult(
        method="NEF-joint (upper bound)",
        scenario=scenario,
        n_tasks=len(tasks),
        accuracy_matrix=[accs],
        fit_times=[fit_time],
        n_neurons=n_neurons,
        config={"alpha": alpha},
    )


def run_nef_growing(
    tasks: list[dict],
    scenario: str,
    *,
    n_neurons: int = 2000,
    alpha: float = 1e-2,
    seed: int = 0,
) -> ContinualResult:
    """NEF with growing neuron pool — add neurons with per-task centers.

    Distributes the total neuron budget evenly across tasks.  Each task's
    neuron group uses that task's training data as centers.  Old AᵀA blocks
    are preserved exactly; cross-terms between old and new neurons only
    include data from the current task (no replay).
    """
    set_seed(seed)
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1
    d_in = tasks[0]["x_train"].shape[1]
    n_tasks = len(tasks)

    # Distribute neurons across tasks
    base = n_neurons // n_tasks
    remainder = n_neurons % n_tasks
    neurons_per_task = [base + (1 if i < remainder else 0) for i in range(n_tasks)]

    activation_fn = make_activation("abs")

    # Growing encoder parameters and accumulators
    encoders = torch.empty(0, d_in)
    bias = torch.empty(0)
    gain = torch.empty(0)
    ata = None  # (current_n, current_n)
    aty = None  # (current_n, n_classes)

    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []

    for task_idx, task in enumerate(tasks):
        t0 = time.perf_counter()
        n_new = neurons_per_task[task_idx]

        # Create new neurons with this task's centers
        rng_t = torch.Generator().manual_seed(seed + task_idx * 1000)
        new_enc = make_encoders(n_new, d_in, strategy="hypersphere", rng=rng_t)
        new_gain = 0.5 + 1.5 * torch.rand(n_new, generator=rng_t)

        # Data-driven biases from this task's centers
        centers = task["x_train"]
        idx = torch.randint(len(centers), (n_new,), generator=rng_t)
        selected = centers[idx].float()
        new_bias = -new_gain * (selected * new_enc).sum(dim=1)

        # Append to growing parameter tensors
        old_n = encoders.shape[0]
        encoders = torch.cat([encoders, new_enc], dim=0)
        bias = torch.cat([bias, new_bias])
        gain = torch.cat([gain, new_gain])

        current_n = encoders.shape[0]

        # Encode current task through ALL neurons (old + new)
        x = task["x_train"]
        A = activation_fn(gain * (x @ encoders.T) + bias)
        targets = one_hot(task["y_train"], n_classes)

        task_ata = A.T @ A
        task_aty = A.T @ targets

        # Expand existing accumulators and add this task's contribution
        if ata is None:
            ata = task_ata
            aty = task_aty
        else:
            new_ata = torch.zeros(current_n, current_n)
            new_ata[:old_n, :old_n] = ata
            new_ata += task_ata
            ata = new_ata

            new_aty = torch.zeros(current_n, n_classes)
            new_aty[:old_n] = aty
            new_aty += task_aty
            aty = new_aty

        # Solve decoders
        decoders = solve_from_normal_equations(ata, aty, alpha=alpha)

        fit_times.append(time.perf_counter() - t0)

        # Evaluate on all tasks
        row = []
        for eval_task in tasks:
            A_eval = activation_fn(gain * (eval_task["x_test"] @ encoders.T) + bias)
            pred = (A_eval @ decoders).argmax(dim=1)
            acc = (pred == eval_task["y_test"]).float().mean().item()
            row.append(acc)
        accuracy_matrix.append(row)

    return ContinualResult(
        method="NEF-growing (per-task centers)",
        scenario=scenario,
        n_tasks=len(tasks),
        accuracy_matrix=accuracy_matrix,
        fit_times=fit_times,
        n_neurons=n_neurons,
        config={"alpha": alpha, "strategy": "growing"},
    )


# ── MLP baselines ────────────────────────────────────────────────────


class SimpleMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _train_mlp(
    model: SimpleMLP,
    x: Tensor,
    y: Tensor,
    *,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 256,
    extra_loss_fn=None,
):
    """Train MLP with SGD + optional extra loss (e.g. EWC penalty)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = ce_loss(model(x[idx]), y[idx])
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn()
            loss.backward()
            optimizer.step()
    model.eval()


def run_mlp_finetune(
    tasks: list[dict],
    scenario: str,
    *,
    d_hidden: int = 2000,
    epochs: int = 10,
    lr: float = 0.01,
    seed: int = 0,
) -> ContinualResult:
    """MLP fine-tuned sequentially — shows catastrophic forgetting."""
    set_seed(seed)
    d_in = tasks[0]["x_train"].shape[1]
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1

    model = SimpleMLP(d_in, d_hidden, n_classes)

    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []

    for task in tasks:
        t0 = time.perf_counter()
        _train_mlp(model, task["x_train"], task["y_train"], epochs=epochs, lr=lr)
        fit_times.append(time.perf_counter() - t0)
        accuracy_matrix.append(evaluate_on_tasks(model, tasks))

    return ContinualResult(
        method="MLP-finetune",
        scenario=scenario,
        n_tasks=len(tasks),
        accuracy_matrix=accuracy_matrix,
        fit_times=fit_times,
        n_neurons=d_hidden,
        config={"d_hidden": d_hidden, "epochs": epochs, "lr": lr},
    )


def run_mlp_joint(
    tasks: list[dict],
    scenario: str,
    *,
    d_hidden: int = 2000,
    epochs: int = 10,
    lr: float = 0.01,
    seed: int = 0,
) -> ContinualResult:
    """MLP trained on all tasks jointly — upper bound."""
    set_seed(seed)
    d_in = tasks[0]["x_train"].shape[1]
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1

    model = SimpleMLP(d_in, d_hidden, n_classes)

    x_all = torch.cat([t["x_train"] for t in tasks])
    y_all = torch.cat([t["y_train"] for t in tasks])

    t0 = time.perf_counter()
    _train_mlp(model, x_all, y_all, epochs=epochs, lr=lr)
    fit_time = time.perf_counter() - t0

    accs = evaluate_on_tasks(model, tasks)

    return ContinualResult(
        method="MLP-joint (upper bound)",
        scenario=scenario,
        n_tasks=len(tasks),
        accuracy_matrix=[accs],
        fit_times=[fit_time],
        n_neurons=d_hidden,
        config={"d_hidden": d_hidden, "epochs": epochs, "lr": lr},
    )


# ── EWC baseline ─────────────────────────────────────────────────────


class _EWCPenalty:
    """Diagonal Fisher penalty for one completed task."""

    def __init__(
        self, model: nn.Module, x: Tensor, y: Tensor, lambda_ewc: float, n_samples: int = 1000
    ):
        self.lambda_ewc = lambda_ewc
        self.saved = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._estimate_fisher(model, x, y, n_samples)

    @staticmethod
    def _estimate_fisher(
        model: nn.Module, x: Tensor, y: Tensor, n_samples: int
    ) -> dict[str, Tensor]:
        fisher: dict[str, Tensor] = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        idx = torch.randperm(len(x))[:n_samples]
        ce_loss = nn.CrossEntropyLoss()
        for i in idx:
            model.zero_grad()
            loss = ce_loss(model(x[i : i + 1]), y[i : i + 1])
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] /= len(idx)
        return fisher

    def penalty(self, model: nn.Module) -> Tensor:
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.saved[n]).pow(2)).sum()
        return 0.5 * self.lambda_ewc * loss


def run_mlp_ewc(
    tasks: list[dict],
    scenario: str,
    *,
    d_hidden: int = 2000,
    epochs: int = 10,
    lr: float = 0.01,
    lambda_ewc: float = 1000.0,
    seed: int = 0,
) -> ContinualResult:
    """MLP with Elastic Weight Consolidation."""
    set_seed(seed)
    d_in = tasks[0]["x_train"].shape[1]
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1

    model = SimpleMLP(d_in, d_hidden, n_classes)
    ewc_penalties: list[_EWCPenalty] = []

    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []

    for task in tasks:
        # Capture current penalties for the closure
        current_penalties = list(ewc_penalties)

        def extra_loss(m=model, ps=current_penalties):
            return sum(p.penalty(m) for p in ps) if ps else torch.tensor(0.0)

        t0 = time.perf_counter()
        _train_mlp(
            model, task["x_train"], task["y_train"], epochs=epochs, lr=lr, extra_loss_fn=extra_loss
        )
        ewc_penalties.append(_EWCPenalty(model, task["x_train"], task["y_train"], lambda_ewc))
        fit_times.append(time.perf_counter() - t0)
        accuracy_matrix.append(evaluate_on_tasks(model, tasks))

    return ContinualResult(
        method=f"MLP-EWC (lambda={lambda_ewc:.0f})",
        scenario=scenario,
        n_tasks=len(tasks),
        accuracy_matrix=accuracy_matrix,
        fit_times=fit_times,
        n_neurons=d_hidden,
        config={"d_hidden": d_hidden, "epochs": epochs, "lr": lr, "lambda_ewc": lambda_ewc},
    )


# ── Experience Replay baseline ────────────────────────────────────────


class _ReplayBuffer:
    """Fixed-size ring buffer for experience replay."""

    def __init__(self, capacity: int, d_in: int, device: str = "cpu"):
        self.capacity = capacity
        self.x = torch.empty(capacity, d_in, device=device)
        self.y = torch.empty(capacity, dtype=torch.long, device=device)
        self.count = 0

    def add(self, x: Tensor, y: Tensor) -> None:
        """Reservoir sampling: each sample has equal probability of being stored."""
        for i in range(len(x)):
            total = self.count + i
            if total < self.capacity:
                idx = total
            else:
                idx = torch.randint(0, total + 1, (1,)).item()
                if idx >= self.capacity:
                    continue
            self.x[idx] = x[i]
            self.y[idx] = y[i]
        self.count += len(x)

    def sample(self, n: int) -> tuple[Tensor, Tensor]:
        """Return up to n samples from the buffer."""
        size = min(n, min(self.count, self.capacity))
        idx = torch.randperm(min(self.count, self.capacity))[:size]
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return min(self.count, self.capacity)


def run_mlp_er(
    tasks: list[dict],
    scenario: str,
    *,
    d_hidden: int = 2000,
    epochs: int = 10,
    lr: float = 0.01,
    buffer_size: int = 200,
    replay_batch: int = 64,
    seed: int = 0,
) -> ContinualResult:
    """MLP with Experience Replay (ER).

    After each task, stores samples in a fixed-size buffer via reservoir
    sampling.  During training, each mini-batch is augmented with replayed
    samples from the buffer.

    ``buffer_size`` is the total buffer capacity (across all tasks).
    """
    set_seed(seed)
    d_in = tasks[0]["x_train"].shape[1]
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1

    model = SimpleMLP(d_in, d_hidden, n_classes)
    buffer = _ReplayBuffer(buffer_size, d_in)

    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []

    for task in tasks:
        t0 = time.perf_counter()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        ce_loss = nn.CrossEntropyLoss()
        model.train()

        x_task, y_task = task["x_train"], task["y_train"]
        batch_size = 256

        for _ in range(epochs):
            perm = torch.randperm(len(x_task))
            for i in range(0, len(x_task), batch_size):
                idx = perm[i : i + batch_size]
                x_batch = x_task[idx]
                y_batch = y_task[idx]

                # Mix in replay samples if buffer is non-empty
                if len(buffer) > 0:
                    x_rep, y_rep = buffer.sample(replay_batch)
                    x_batch = torch.cat([x_batch, x_rep])
                    y_batch = torch.cat([y_batch, y_rep])

                optimizer.zero_grad()
                loss = ce_loss(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

        model.eval()

        # Add current task to buffer
        buffer.add(x_task, y_task)

        fit_times.append(time.perf_counter() - t0)
        accuracy_matrix.append(evaluate_on_tasks(model, tasks))

    return ContinualResult(
        method=f"MLP-ER (buf={buffer_size})",
        scenario=scenario,
        n_tasks=len(tasks),
        accuracy_matrix=accuracy_matrix,
        fit_times=fit_times,
        n_neurons=d_hidden,
        config={
            "d_hidden": d_hidden,
            "epochs": epochs,
            "lr": lr,
            "buffer_size": buffer_size,
            "replay_batch": replay_batch,
        },
    )


def print_result(result: ContinualResult) -> None:
    """Print a formatted accuracy matrix and summary metrics."""
    n_tasks = len(result.accuracy_matrix[-1])
    print(f"\n{'=' * 72}")
    print(f"  {result.method}")
    print(f"  Scenario: {result.scenario}, {result.n_tasks} tasks, {result.n_neurons} neurons")
    print(f"{'=' * 72}")

    # Header
    hdr = "After task |"
    for j in range(n_tasks):
        hdr += f" T{j:d}     |"
    hdr += "  Avg"
    print(hdr)
    print("-" * len(hdr))

    for i, accs in enumerate(result.accuracy_matrix):
        row = f"    {i:d}      |"
        avg = sum(accs) / len(accs)
        for acc in accs:
            row += f" {acc * 100:5.1f}% |"
        row += f" {avg * 100:5.1f}%"
        print(row)

    print()
    print(f"  Average accuracy (final): {result.final_average_accuracy * 100:.1f}%")
    print(f"  Forgetting:               {result.forgetting * 100:.1f}%")
    print(f"  Backward transfer:        {result.backward_transfer * 100:+.1f}%")
    print(f"  Total fit time:           {result.total_time:.2f}s")


def save_results(results: list[ContinualResult], path: str | Path) -> None:
    """Save results as JSON, flushing after write."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in results:
        d = asdict(r)
        d["final_average_accuracy"] = r.final_average_accuracy
        d["forgetting"] = r.forgetting
        d["backward_transfer"] = r.backward_transfer
        d["total_time"] = r.total_time
        payload.append(d)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nResults saved to {out}")


# ── Main ──────────────────────────────────────────────────────────────


def run_scenario(
    scenario_name: str,
    tasks: list[dict],
    *,
    n_neurons: int,
    alpha: float,
    mlp_epochs: int,
    mlp_lr: float,
    ewc_lambda: float,
    er_buffer: int,
    seed: int,
    skip_mlp: bool,
    skip_ewc: bool,
    skip_er: bool,
    skip_growing: bool = False,
    centers_strategies: list[str],
) -> list[ContinualResult]:
    """Run all methods for one scenario and return results."""
    results: list[ContinualResult] = []

    # NEF-accumulate with different center strategies
    for cs in centers_strategies:
        print(f"\n>>> NEF-accumulate (centers={cs}) ...")
        r = run_nef_accumulate(
            tasks, scenario_name, n_neurons=n_neurons, alpha=alpha, use_centers=cs, seed=seed
        )
        print_result(r)
        results.append(r)

    # NEF-growing (per-task centers, no replay)
    if not skip_growing:
        print("\n>>> NEF-growing (per-task centers) ...")
        r = run_nef_growing(tasks, scenario_name, n_neurons=n_neurons, alpha=alpha, seed=seed)
        print_result(r)
        results.append(r)

    # NEF-reset
    print("\n>>> NEF-reset ...")
    r = run_nef_reset(tasks, scenario_name, n_neurons=n_neurons, alpha=alpha, seed=seed)
    print_result(r)
    results.append(r)

    # NEF-joint
    print("\n>>> NEF-joint ...")
    r = run_nef_joint(tasks, scenario_name, n_neurons=n_neurons, alpha=alpha, seed=seed)
    print_result(r)
    results.append(r)

    if not skip_mlp:
        # MLP-finetune
        print("\n>>> MLP-finetune ...")
        r = run_mlp_finetune(
            tasks, scenario_name, d_hidden=n_neurons, epochs=mlp_epochs, lr=mlp_lr, seed=seed
        )
        print_result(r)
        results.append(r)

        # MLP-joint
        print("\n>>> MLP-joint ...")
        r = run_mlp_joint(
            tasks, scenario_name, d_hidden=n_neurons, epochs=mlp_epochs, lr=mlp_lr, seed=seed
        )
        print_result(r)
        results.append(r)

    if not skip_ewc:
        print("\n>>> MLP-EWC ...")
        r = run_mlp_ewc(
            tasks,
            scenario_name,
            d_hidden=n_neurons,
            epochs=mlp_epochs,
            lr=mlp_lr,
            lambda_ewc=ewc_lambda,
            seed=seed,
        )
        print_result(r)
        results.append(r)

    if not skip_er:
        print(f"\n>>> MLP-ER (buffer={er_buffer}) ...")
        r = run_mlp_er(
            tasks,
            scenario_name,
            d_hidden=n_neurons,
            epochs=mlp_epochs,
            lr=mlp_lr,
            buffer_size=er_buffer,
            seed=seed,
        )
        print_result(r)
        results.append(r)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual learning benchmarks for NEF")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_LOADERS),
        default="mnist",
        help="Dataset (default: mnist)",
    )
    parser.add_argument(
        "--scenario", choices=["split", "permuted", "both"], default="both", help="CL scenario"
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=None,
        help="NEF neurons / MLP hidden dim (default: dataset-dependent)",
    )
    parser.add_argument(
        "--classes-per-task",
        type=int,
        default=2,
        help="Classes per task for split scenario (default: 2)",
    )
    parser.add_argument(
        "--n-tasks-permuted", type=int, default=5, help="tasks for permuted scenario"
    )
    parser.add_argument("--alpha", type=float, default=1e-2, help="Tikhonov alpha")
    parser.add_argument("--mlp-epochs", type=int, default=10, help="SGD epochs per task")
    parser.add_argument("--mlp-lr", type=float, default=0.01, help="SGD learning rate")
    parser.add_argument("--ewc-lambda", type=float, default=1000.0, help="EWC penalty strength")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--output-dir", type=str, default="results/continual")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--skip-mlp", action="store_true", help="skip MLP baselines")
    parser.add_argument("--skip-ewc", action="store_true", help="skip EWC baseline")
    parser.add_argument("--skip-er", action="store_true", help="skip ER baseline")
    parser.add_argument("--er-buffer", type=int, default=500, help="ER replay buffer capacity")
    parser.add_argument("--skip-growing", action="store_true", help="skip NEF-growing method")
    args = parser.parse_args()

    dataset = args.dataset
    n_classes = DATASET_N_CLASSES[dataset]
    n_neurons = args.n_neurons or DATASET_DEFAULT_NEURONS[dataset]
    cpt = args.classes_per_task

    print(f"Loading {dataset.upper()} ...")
    (x_train, y_train), (x_test, y_test) = DATASET_LOADERS[dataset](args.data_root)
    print(f"  train: {x_train.shape}, test: {x_test.shape}, {n_classes} classes")

    all_results: list[ContinualResult] = []

    if args.scenario in ("split", "both"):
        n_split_tasks = n_classes // cpt
        label = (
            f"SPLIT-{dataset.upper()}  (class-incremental, {n_split_tasks} tasks × {cpt} classes)"
        )
        print("\n" + "=" * 72)
        print(f"  {label}")
        print("=" * 72)
        tasks = split_class_tasks(x_train, y_train, x_test, y_test, n_classes, cpt)
        centers_strategies = (
            ["none", "first_task", "all_tasks"] if dataset != "cifar100" else ["none"]
        )
        results = run_scenario(
            f"split-{dataset}",
            tasks,
            n_neurons=n_neurons,
            alpha=args.alpha,
            mlp_epochs=args.mlp_epochs,
            mlp_lr=args.mlp_lr,
            ewc_lambda=args.ewc_lambda,
            er_buffer=args.er_buffer,
            seed=args.seed,
            skip_mlp=args.skip_mlp,
            skip_ewc=args.skip_ewc,
            skip_er=args.skip_er,
            skip_growing=args.skip_growing,
            centers_strategies=centers_strategies,
        )
        all_results.extend(results)

    if args.scenario in ("permuted", "both"):
        print("\n" + "=" * 72)
        print(f"  PERMUTED-{dataset.upper()}  (domain-incremental, {args.n_tasks_permuted} tasks)")
        print("=" * 72)
        tasks = permuted_tasks(
            x_train, y_train, x_test, y_test, n_tasks=args.n_tasks_permuted, seed=args.seed
        )
        results = run_scenario(
            f"permuted-{dataset}",
            tasks,
            n_neurons=n_neurons,
            alpha=args.alpha,
            mlp_epochs=args.mlp_epochs,
            mlp_lr=args.mlp_lr,
            ewc_lambda=args.ewc_lambda,
            er_buffer=args.er_buffer,
            seed=args.seed,
            skip_mlp=args.skip_mlp,
            skip_ewc=args.skip_ewc,
            skip_er=args.skip_er,
            skip_growing=args.skip_growing,
            centers_strategies=["none", "first_task", "all_tasks"],
        )
        all_results.extend(results)

    # ── Summary table ──

    print("\n\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(
        f"{'Method':<40s} {'Scenario':<20s} {'Avg Acc':>7s} {'Forget':>7s} {'BWT':>7s} {'Time':>7s}"
    )
    print("-" * 88)
    for r in all_results:
        print(
            f"{r.method:<40s} {r.scenario:<20s} "
            f"{r.final_average_accuracy * 100:6.1f}% "
            f"{r.forgetting * 100:6.1f}% "
            f"{r.backward_transfer * 100:+6.1f}% "
            f"{r.total_time:6.1f}s"
        )

    out_name = f"continual_{dataset}_results.json"
    save_results(all_results, Path(args.output_dir) / out_name)


if __name__ == "__main__":
    main()
