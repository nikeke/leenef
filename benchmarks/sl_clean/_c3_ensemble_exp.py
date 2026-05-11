"""C3 — ensemble + receptive-field experiment.

Knob is ``cell`` = ``"m{n_members}+{encoder_strategy}"``, e.g.
``"m10+hypersphere"`` or ``"m20+receptive_field"``.  8-cell grid:
  n_members ∈ {1, 5, 10, 20}
  encoder   ∈ {hypersphere, receptive_field}

Run on CIFAR-10 only (that is where the encoder-prior interaction is
strongest — TR §5.4).  ``n_neurons=5000`` per member, ``alpha=1e-3``.
5 seeds (each cell trains ``n_members`` independent layers, so the
seed-level cost is M × per-member solve).

Expected outcome: receptive_field beats hypersphere at every M;
ensembling closes a fraction of the receptive_field-vs-hypersphere
gap for hypersphere but receptive_field stays ahead at any M.
"""

from __future__ import annotations

import time

import torch

from leenef.encoders import make_encoders

from ._phase_c_shared import (
    _make_subrngs_phase_c,
    chunked_accumulate_solve,
    load_dataset_for_phase_c,
    set_sl_seed,
)

C3_CELLS: tuple[str, ...] = tuple(
    f"m{m}+{enc}" for m in (1, 5, 10, 20) for enc in ("hypersphere", "receptive_field")
)


def _parse_cell(cell: str) -> tuple[int, str]:
    if "+" not in cell:
        raise ValueError(f"C3 cell {cell!r}: expected 'm{{n_members}}+{{encoder}}'")
    m_token, enc = cell.split("+", 1)
    if not m_token.startswith("m"):
        raise ValueError(f"C3 cell {cell!r}: m-token must look like 'm{{n}}'")
    return int(m_token[1:]), enc


def _build_member(
    *,
    encoder_strategy: str,
    n_neurons: int,
    d_in: int,
    rng_encoder: torch.Generator,
    rng_gain: torch.Generator,
    rng_center: torch.Generator,
    x_train: torch.Tensor,
    image_shape: tuple[int, int],
    patch_size: int,
    gain_low: float,
    gain_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs: dict[str, object] = {"strategy": encoder_strategy, "rng": rng_encoder}
    if encoder_strategy == "receptive_field":
        kwargs["image_shape"] = image_shape
        kwargs["patch_size"] = patch_size
    out = make_encoders(n_neurons, d_in, **kwargs)
    enc = out[0] if isinstance(out, tuple) else out
    idx = torch.randint(len(x_train), (n_neurons,), generator=rng_center)
    centers = x_train[idx]
    gain = gain_low + (gain_high - gain_low) * torch.rand(n_neurons, generator=rng_gain)
    bias = -gain * (centers * enc).sum(dim=1)
    return enc, gain, bias


def _chunked_logits(
    *,
    enc: torch.Tensor,
    gain: torch.Tensor,
    bias: torch.Tensor,
    D: torch.Tensor,
    x: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.zeros(x.shape[0], D.shape[1], dtype=torch.float32)
    for s in range(0, x.shape[0], chunk_size):
        e = min(s + chunk_size, x.shape[0])
        z = x[s:e] @ enc.T
        z.mul_(gain)
        z.add_(bias)
        z.abs_()
        out[s:e] = z @ D
    return out


def _c3_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    n_members, encoder_strategy = _parse_cell(str(config["cell"]))
    n_neurons = int(config["n_neurons"])
    alpha = float(config["alpha"])
    gain_low = float(config["gain_low"])
    gain_high = float(config["gain_high"])
    chunk_size = int(config.get("chunk_size", 2500))
    patch_size = int(config.get("patch_size", 5))

    bundle = load_dataset_for_phase_c(run_context)
    x_train, y_train = bundle.x_train, bundle.y_train
    x_test, y_test = bundle.x_test, bundle.y_test
    d_in = int(x_train.shape[1])
    image_shape = tuple(run_context["image_shape"])
    Y_oh = bundle.one_hot(y_train).float()

    rngs = _make_subrngs_phase_c(seed)
    base_member_seed = int(rngs["ensemble_member"].initial_seed())

    test_logits = torch.zeros(x_test.shape[0], bundle.n_classes, dtype=torch.float32)
    train_logits = torch.zeros(x_train.shape[0], bundle.n_classes, dtype=torch.float32)
    encode_fit_total = 0.0
    solve_total = 0.0

    t0_total = time.perf_counter()
    for m in range(n_members):
        member_seed = base_member_seed + 9176 * m + 1
        m_enc_rng = torch.Generator().manual_seed(member_seed)
        m_gain_rng = torch.Generator().manual_seed(member_seed + 1)
        m_ctr_rng = torch.Generator().manual_seed(member_seed + 2)
        enc, gain, bias = _build_member(
            encoder_strategy=encoder_strategy,
            n_neurons=n_neurons,
            d_in=d_in,
            rng_encoder=m_enc_rng,
            rng_gain=m_gain_rng,
            rng_center=m_ctr_rng,
            x_train=x_train,
            image_shape=image_shape,  # type: ignore[arg-type]
            patch_size=patch_size,
            gain_low=gain_low,
            gain_high=gain_high,
        )
        D, timings = chunked_accumulate_solve(
            enc=enc,
            gain=gain,
            bias=bias,
            activation="abs",
            x_train=x_train,
            y_train_targets=Y_oh,
            chunk_size=chunk_size,
            alpha=(alpha, "trace"),
        )
        encode_fit_total += float(timings["encode_fit_time_s"])
        solve_total += float(timings["solve_time_s"])
        train_logits += _chunked_logits(
            enc=enc,
            gain=gain,
            bias=bias,
            D=D,
            x=x_train,
            chunk_size=chunk_size,
        )
        test_logits += _chunked_logits(
            enc=enc,
            gain=gain,
            bias=bias,
            D=D,
            x=x_test,
            chunk_size=chunk_size,
        )
    wall = time.perf_counter() - t0_total

    train_logits.div_(n_members)
    test_logits.div_(n_members)

    train_acc = float((train_logits.argmax(dim=1) == y_train).float().mean())
    test_acc = float((test_logits.argmax(dim=1) == y_test).float().mean())

    return {
        "test_accuracy": round(test_acc, 6),
        "train_accuracy": round(train_acc, 6),
        "encode_fit_time_s": round(encode_fit_total, 4),
        "solve_time_s": round(solve_total, 4),
        "wall_sec": round(wall, 4),
        "factors": {
            "n_members": n_members,
            "encoder_strategy": encoder_strategy,
        },
        "diagnostics": {
            "n_members": n_members,
            "n_neurons_per_member": n_neurons,
            "total_neurons": n_neurons * n_members,
        },
        "dataset_metadata": bundle.metadata(),
    }


def _c3_cifar10_context() -> dict:
    return {
        "experiment_kind": "cifar10_singlelayer_c3_ensemble_panel",
        "dataset": "CIFAR10",
        "preprocessing": "div255_flat",
        "n_train": 50000,
        "n_test": 10000,
        "d_in": 3072,
        "n_classes": 10,
        "image_shape": [32, 32],
    }


C3_BASE_CONFIG: dict[str, object] = {
    "cell": "m1+hypersphere",
    "n_neurons": 5000,
    "alpha": 1e-3,
    "gain_low": 0.5,
    "gain_high": 2.0,
    "chunk_size": 2500,
    "patch_size": 5,
}


C3_DESCRIPTION = (
    "C3 ensemble × encoder ablation on CIFAR-10. Knob 'cell' = "
    "'m{n_members}+{encoder_strategy}'.  8-cell grid: "
    "n_members ∈ {1,5,10,20} × encoder ∈ {hypersphere, receptive_field}.  "
    "n_neurons=5000 per member, alpha=1e-3, abs activation, data_driven biases."
)
