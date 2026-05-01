"""MLP baseline for the SL clean comparison.

A plain feed-forward network with configurable depth, width, activation
and weight decay. All stochasticity is parametrised by explicit
``torch.Generator`` instances so the cell is reproducible regardless of
the global RNG state (we have been bitten by default-generator drift
in the past).

Training protocol:
- Adam optimiser (the standard reasonable choice for MLP baselines).
- Cosine LR schedule from ``lr`` down to ``lr * 0.01`` over ``epochs``.
- Mini-batches of ``batch_size`` drawn via a seeded ``DataLoader``.
- Early stopping on a held-out 10% validation split with patience.
- Returns final test accuracy at the *best validation epoch*, plus
  diagnostics (epochs run, total wall time, train acc, val acc).

The MLP is *not* meant to be state-of-the-art. It is a *fairly tuned*
baseline used to contextualise the leenef numbers in Paper I (H2).
We report whatever it produces.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for an MLP baseline run."""

    depth: int
    width: int
    activation: str  # "relu", "gelu", "tanh"
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    val_fraction: float = 0.1
    patience: int = 5
    dropout: float = 0.0


def _activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


def _init_linear_(module: nn.Linear, generator: torch.Generator) -> None:
    """Kaiming-uniform init driven by an explicit generator."""
    fan_in = module.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    with torch.no_grad():
        module.weight.uniform_(-bound, bound, generator=generator)
        if module.bias is not None:
            module.bias.uniform_(-bound, bound, generator=generator)


def build_mlp(
    d_in: int,
    n_classes: int,
    cfg: MLPConfig,
    generator: torch.Generator,
) -> nn.Sequential:
    """Build an MLP with reproducible weight init."""
    layers: list[nn.Module] = []
    in_dim = d_in
    for _ in range(cfg.depth):
        lin = nn.Linear(in_dim, cfg.width)
        _init_linear_(lin, generator)
        layers.append(lin)
        layers.append(_activation_module(cfg.activation))
        if cfg.dropout > 0.0:
            layers.append(nn.Dropout(cfg.dropout))
        in_dim = cfg.width
    out = nn.Linear(in_dim, n_classes)
    _init_linear_(out, generator)
    layers.append(out)
    return nn.Sequential(*layers)


def _split_train_val(
    x: torch.Tensor,
    y: torch.Tensor,
    val_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    n_val = max(1, int(round(val_fraction * n)))
    perm = torch.randperm(n, generator=generator)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 1024) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = x[i : i + batch]
            yb = y[i : i + batch]
            logits = model(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.shape[0]
    return correct / max(1, total)


def train_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: MLPConfig,
    *,
    generator_init: torch.Generator,
    generator_split: torch.Generator,
    generator_shuffle: torch.Generator,
    device: torch.device | str = "cpu",
) -> dict:
    """Train one MLP and return diagnostics dict.

    Three independent generators are required:

    - ``generator_init``: weight initialisation
    - ``generator_split``: train/val split
    - ``generator_shuffle``: per-epoch batch shuffling
    """
    device = torch.device(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    x_tr, y_tr, x_val, y_val = _split_train_val(
        x_train, y_train, cfg.val_fraction, generator_split
    )

    d_in = x_train.shape[1]
    n_classes = int(y_train.max().item()) + 1
    model = build_mlp(d_in, n_classes, cfg, generator_init).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    n_train = x_tr.shape[0]
    train_acc_at_best = 0.0

    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=generator_shuffle, device="cpu").to(device)
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb = x_tr[idx]
            yb = y_tr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        val_acc = _accuracy(model, x_val, y_val)
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_test_acc = _accuracy(model, x_test, y_test)
            train_acc_at_best = _accuracy(model, x_tr, y_tr)
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break
    elapsed = time.perf_counter() - t0

    return {
        "test_accuracy": float(best_test_acc),
        "val_accuracy": float(best_val_acc),
        "train_accuracy": float(train_acc_at_best),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch + 1),
        "fit_time_sec": float(elapsed),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
