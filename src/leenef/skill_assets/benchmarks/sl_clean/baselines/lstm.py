"""LSTM baseline for the SL clean comparison (Paper I H4/H5).

Standard PyTorch ``nn.LSTM`` with reproducible weight init and DataLoader
shuffling.  Used to contextualise the StreamNEF claim on temporal
benchmarks (sMNIST-row, permuted sMNIST, later Speech Commands and
sCIFAR-10).

We match the streaming-NEF protocol: the LSTM consumes the sequence
once, takes the *last hidden state* as the classification feature, and
trains end-to-end with cross-entropy on the final-step prediction.

Training protocol:
- Adam optimiser, cosine LR.
- Gradient clipping at ``grad_clip`` to stabilise BPTT.
- Mini-batches of ``batch_size`` shuffled with an explicit Generator.
- Held-out 10% validation with patience-based early stopping.
- Returns best-val test accuracy plus diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LSTMConfig:
    hidden_size: int
    num_layers: int
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    val_fraction: float = 0.1
    patience: int = 5
    grad_clip: float = 1.0
    dropout: float = 0.0


class LSTMClassifier(nn.Module):
    """LSTM with a linear head on the last hidden state."""

    def __init__(self, d_in: int, hidden: int, layers: int, n_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, T, d)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def _init_lstm_(model: LSTMClassifier, generator: torch.Generator) -> None:
    """Reproducible weight init driven by an explicit generator.

    nn.LSTM initialises weights with ``torch.uniform_`` against a
    ``1/sqrt(hidden)`` bound by default but using the global RNG.  We
    overwrite every parameter with that same distribution but driven
    by ``generator``.
    """
    hidden = model.lstm.hidden_size
    bound = 1.0 / math.sqrt(hidden) if hidden > 0 else 0.0
    with torch.no_grad():
        for name, p in model.lstm.named_parameters():
            if "weight" in name or "bias" in name:
                p.uniform_(-bound, bound, generator=generator)
        # head: fan-in init.
        fan_in = model.head.weight.shape[1]
        b = 1.0 / math.sqrt(fan_in)
        model.head.weight.uniform_(-b, b, generator=generator)
        if model.head.bias is not None:
            model.head.bias.uniform_(-b, b, generator=generator)


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


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 256) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            logits = model(x[i : i + batch])
            correct += (logits.argmax(dim=1) == y[i : i + batch]).sum().item()
            total += min(batch, x.shape[0] - i)
    return correct / max(1, total)


def train_lstm(
    x_train: torch.Tensor,  # (N, T, d)
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: LSTMConfig,
    *,
    generator_init: torch.Generator,
    generator_split: torch.Generator,
    generator_shuffle: torch.Generator,
    device: torch.device | str = "cpu",
) -> dict:
    device = torch.device(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    x_tr, y_tr, x_val, y_val = _split_train_val(
        x_train, y_train, cfg.val_fraction, generator_split
    )

    d_in = x_train.shape[-1]
    n_classes = int(y_train.max().item()) + 1
    model = LSTMClassifier(d_in, cfg.hidden_size, cfg.num_layers, n_classes, cfg.dropout)
    _init_lstm_(model, generator_init)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_epoch = 0
    train_acc_at_best = 0.0
    epochs_no_improve = 0
    n_train = x_tr.shape[0]

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
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
