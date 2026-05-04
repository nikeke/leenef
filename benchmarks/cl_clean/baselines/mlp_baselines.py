"""MLP-family baselines for cl_clean.

These are the *baseline* implementations used by the cross-method panel
(Paper II H5).  All four share a small MLP architecture and the same
training discipline; they differ only in how they handle the task
sequence:

* :func:`finetune_run` — train sequentially, no protection against
  forgetting (lower bound).
* :func:`joint_run` — train once on the union of all tasks (upper
  bound).
* :func:`ewc_run` — Elastic Weight Consolidation with a per-task Fisher
  diagonal estimated on the task's training subset.
* :func:`er_run` — Experience Replay with a fixed-size class-balanced
  reservoir.

Reproducibility discipline (must match cl_clean hazards):

* All randomness uses an explicit ``torch.Generator`` seeded from the
  experiment seed.  No global-RNG draws.
* The MLP construction order is fixed; no implicit dependence on the
  order of ``nn.Linear`` parameter init.
* The dataset stream comes from :func:`cl_clean.datasets.make_task_stream`
  so the task ordering is reproducible across runs.

All baselines return the same metric dict shape as the NEF runs in
:mod:`cl_clean.experiments` so a single panel analyzer can compare
them: ``test_accuracy`` (final average over all tasks),
``per_task_accuracy``, ``fit_time_s``, ``stream_hash``,
``stream_metadata``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..cl_repro import set_cl_seed
from ..datasets import TaskStream, make_task_stream

__all__ = ["finetune_run", "joint_run", "ewc_run", "er_run"]


# ---------------------------------------------------------------------------
# Shared MLP and training utilities.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLPConfig:
    d_in: int
    n_classes: int
    hidden: int
    n_hidden_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs_per_task: int = 5
    batch_size: int = 128


def _build_mlp(cfg: MLPConfig, seed: int) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    layers: list[nn.Module] = []
    in_dim = cfg.d_in
    for _ in range(cfg.n_hidden_layers):
        lin = nn.Linear(in_dim, cfg.hidden)
        with torch.no_grad():
            bound = (1.0 / max(in_dim, 1)) ** 0.5
            lin.weight.copy_((torch.rand(lin.weight.shape, generator=g) * 2 - 1) * bound)
            lin.bias.copy_((torch.rand(lin.bias.shape, generator=g) * 2 - 1) * bound)
        layers.append(lin)
        layers.append(nn.ReLU())
        in_dim = cfg.hidden
    head = nn.Linear(in_dim, cfg.n_classes)
    with torch.no_grad():
        bound = (1.0 / max(in_dim, 1)) ** 0.5
        head.weight.copy_((torch.rand(head.weight.shape, generator=g) * 2 - 1) * bound)
        head.bias.copy_((torch.rand(head.bias.shape, generator=g) * 2 - 1) * bound)
    layers.append(head)
    return nn.Sequential(*layers)


def _train_epoch(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    cfg: MLPConfig,
    optim: torch.optim.Optimizer,
    rng: torch.Generator,
    extra_loss_fn=None,
) -> None:
    n = x.shape[0]
    perm = torch.randperm(n, generator=rng)
    for s in range(0, n, cfg.batch_size):
        idx = perm[s : s + cfg.batch_size]
        xb, yb = x[idx], y[idx]
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()


def _evaluate_per_task(
    model: nn.Module, stream: TaskStream, *, chunk: int = 5000
) -> tuple[list[float], float]:
    per_task: list[float] = []
    n_correct_total = 0
    n_total = 0
    model.eval()
    with torch.no_grad():
        for task in stream.tasks:
            x, y = task.x_test, task.y_test
            n_correct = 0
            for s in range(0, x.shape[0], chunk):
                e = min(s + chunk, x.shape[0])
                pred = model(x[s:e]).argmax(dim=1)
                n_correct += int((pred == y[s:e]).sum())
            per_task.append(round(n_correct / x.shape[0], 6))
            n_correct_total += n_correct
            n_total += x.shape[0]
    model.train()
    return per_task, round(n_correct_total / n_total, 6)


def _make_stream_and_cfg(
    config: dict, seed: int, run_context: dict
) -> tuple[TaskStream, MLPConfig, torch.Generator]:
    set_cl_seed(seed)
    stream = make_task_stream(
        str(run_context["base_dataset"]),
        scenario=str(run_context["scenario"]),
        n_tasks=int(run_context["n_tasks"]),
        stream_seed=seed,
    )
    cfg = MLPConfig(
        d_in=int(stream.tasks[0].x_train.shape[1]),
        n_classes=stream.n_classes,
        hidden=int(config.get("hidden", 256)),
        n_hidden_layers=int(config.get("n_hidden_layers", 2)),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        epochs_per_task=int(config.get("epochs_per_task", 5)),
        batch_size=int(config.get("batch_size", 128)),
    )
    rng = torch.Generator().manual_seed(seed * 1_000_003 + 31)
    return stream, cfg, rng


def _result_envelope(stream: TaskStream, per_task: list[float], avg: float, fit_s: float) -> dict:
    return {
        "test_accuracy": avg,
        "per_task_accuracy": per_task,
        "fit_time_s": round(fit_s, 4),
        "stream_hash": stream.stream_hash,
        "stream_metadata": stream.metadata(),
    }


# ---------------------------------------------------------------------------
# finetune (lower bound).
# ---------------------------------------------------------------------------


def finetune_run(config: dict, seed: int, run_context: dict) -> dict:
    stream, cfg, rng = _make_stream_and_cfg(config, seed, run_context)
    model = _build_mlp(cfg, seed=seed * 1_000_003 + 1)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    t0 = time.perf_counter()
    for task in stream.tasks:
        for _ in range(cfg.epochs_per_task):
            _train_epoch(model, task.x_train, task.y_train, cfg=cfg, optim=optim, rng=rng)
    fit_s = time.perf_counter() - t0
    per_task, avg = _evaluate_per_task(model, stream)
    return _result_envelope(stream, per_task, avg, fit_s)


# ---------------------------------------------------------------------------
# joint (upper bound).
# ---------------------------------------------------------------------------


def joint_run(config: dict, seed: int, run_context: dict) -> dict:
    stream, cfg, rng = _make_stream_and_cfg(config, seed, run_context)
    model = _build_mlp(cfg, seed=seed * 1_000_003 + 1)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    all_x = torch.cat([t.x_train for t in stream.tasks], dim=0)
    all_y = torch.cat([t.y_train for t in stream.tasks], dim=0)
    total_epochs = cfg.epochs_per_task * stream.n_tasks
    t0 = time.perf_counter()
    for _ in range(total_epochs):
        _train_epoch(model, all_x, all_y, cfg=cfg, optim=optim, rng=rng)
    fit_s = time.perf_counter() - t0
    per_task, avg = _evaluate_per_task(model, stream)
    return _result_envelope(stream, per_task, avg, fit_s)


# ---------------------------------------------------------------------------
# EWC (Kirkpatrick et al. 2017).  Per-task diagonal Fisher.
# ---------------------------------------------------------------------------


def _diag_fisher(
    model: nn.Module, x: Tensor, y: Tensor, *, n_samples: int, rng: torch.Generator
) -> dict[str, Tensor]:
    n = x.shape[0]
    sample_idx = torch.randint(n, (min(n_samples, n),), generator=rng)
    fisher: dict[str, Tensor] = {
        name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad
    }
    model.eval()
    for i in sample_idx.tolist():
        model.zero_grad(set_to_none=True)
        log_probs = nn.functional.log_softmax(model(x[i : i + 1]), dim=1)
        loss = -log_probs[0, int(y[i])]
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
    model.train()
    n_eff = max(1, len(sample_idx))
    return {k: v / n_eff for k, v in fisher.items()}


def ewc_run(config: dict, seed: int, run_context: dict) -> dict:
    stream, cfg, rng = _make_stream_and_cfg(config, seed, run_context)
    model = _build_mlp(cfg, seed=seed * 1_000_003 + 1)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ewc_lambda = float(config.get("ewc_lambda", 1e3))
    fisher_samples = int(config.get("fisher_samples", 200))

    accumulated: list[tuple[dict[str, Tensor], dict[str, Tensor]]] = []

    def ewc_penalty(m: nn.Module) -> Tensor:
        if not accumulated:
            return torch.zeros((), dtype=torch.float32)
        loss = torch.zeros((), dtype=torch.float32)
        params = dict(m.named_parameters())
        for fisher, anchor in accumulated:
            for name, fdiag in fisher.items():
                loss = loss + (fdiag * (params[name] - anchor[name]) ** 2).sum()
        return 0.5 * ewc_lambda * loss

    t0 = time.perf_counter()
    for task in stream.tasks:
        for _ in range(cfg.epochs_per_task):
            _train_epoch(
                model,
                task.x_train,
                task.y_train,
                cfg=cfg,
                optim=optim,
                rng=rng,
                extra_loss_fn=ewc_penalty,
            )
        f = _diag_fisher(model, task.x_train, task.y_train, n_samples=fisher_samples, rng=rng)
        a = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        accumulated.append((f, a))
    fit_s = time.perf_counter() - t0
    per_task, avg = _evaluate_per_task(model, stream)
    return _result_envelope(stream, per_task, avg, fit_s)


# ---------------------------------------------------------------------------
# Experience Replay (reservoir-based).
# ---------------------------------------------------------------------------


def er_run(config: dict, seed: int, run_context: dict) -> dict:
    stream, cfg, rng = _make_stream_and_cfg(config, seed, run_context)
    model = _build_mlp(cfg, seed=seed * 1_000_003 + 1)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer_size = int(config.get("buffer_size", 500))
    replay_batch = int(config.get("replay_batch", 64))

    buf_x = torch.empty(0, cfg.d_in, dtype=torch.float32)
    buf_y = torch.empty(0, dtype=torch.int64)
    seen = 0

    def reservoir_add(x_new: Tensor, y_new: Tensor) -> tuple[Tensor, Tensor]:
        nonlocal seen, buf_x, buf_y
        for i in range(x_new.shape[0]):
            if buf_x.shape[0] < buffer_size:
                buf_x = torch.cat([buf_x, x_new[i : i + 1]], dim=0)
                buf_y = torch.cat([buf_y, y_new[i : i + 1]], dim=0)
            else:
                j = int(torch.randint(0, seen + 1, (1,), generator=rng).item())
                if j < buffer_size:
                    buf_x[j] = x_new[i]
                    buf_y[j] = y_new[i]
            seen += 1
        return buf_x, buf_y

    def er_train_epoch(task_x: Tensor, task_y: Tensor) -> None:
        n = task_x.shape[0]
        perm = torch.randperm(n, generator=rng)
        for s in range(0, n, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            xb, yb = task_x[idx], task_y[idx]
            if buf_x.shape[0] > 0:
                k = min(replay_batch, buf_x.shape[0])
                ridx = torch.randperm(buf_x.shape[0], generator=rng)[:k]
                xb = torch.cat([xb, buf_x[ridx]], dim=0)
                yb = torch.cat([yb, buf_y[ridx]], dim=0)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

    t0 = time.perf_counter()
    for task in stream.tasks:
        for _ in range(cfg.epochs_per_task):
            er_train_epoch(task.x_train, task.y_train)
        reservoir_add(task.x_train, task.y_train)
    fit_s = time.perf_counter() - t0
    per_task, avg = _evaluate_per_task(model, stream)
    return _result_envelope(stream, per_task, avg, fit_s)
