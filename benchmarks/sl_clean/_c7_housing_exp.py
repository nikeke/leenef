"""C7 — California Housing regression (light touch).

Knob is ``cell`` = ``"n{n_neurons}"`` — same encoding as C1, but on a
regression target (median house value).  3 seeds.  Confirms the
universality demo from TR §5.6: the same primitive solves regression
with no architectural change beyond setting ``d_out=1``.

Dataset is fetched via ``sklearn.datasets.fetch_california_housing``
on first use and cached by sklearn under
``~/scikit_learn_data/cal_housing/``.  Determinism is guaranteed
because we explicitly seed the train/test split with a sub-RNG.
"""

from __future__ import annotations

import hashlib
import time

import torch

from leenef.encoders import make_encoders

from ._phase_c_shared import (
    _activation_fn,
    _make_subrngs_phase_c,
    set_sl_seed,
)

C7_CELLS: tuple[str, ...] = ("n500", "n1000", "n2000", "n5000", "n10000")


def _parse_cell(cell: str) -> int:
    if not cell.startswith("n"):
        raise ValueError(f"C7 cell {cell!r}: expected 'n{{n_neurons}}'")
    return int(cell[1:])


def _load_california_housing(seed: int, test_frac: float = 0.2):
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    x = torch.from_numpy(data.data).float()
    y = torch.from_numpy(data.target).float().unsqueeze(1)

    # standardize features (zero mean, unit variance) — independent of seed
    x = (x - x.mean(dim=0)) / x.std(dim=0).clamp(min=1e-8)

    g = torch.Generator().manual_seed(seed * 31337 + 11)
    perm = torch.randperm(len(x), generator=g)
    x, y = x[perm], y[perm]
    n_test = int(test_frac * len(x))
    x_train, x_test = x[n_test:], x[:n_test]
    y_train, y_test = y[n_test:], y[:n_test]

    raw_hash = hashlib.sha256(data.data.tobytes()).hexdigest()
    target_hash = hashlib.sha256(data.target.tobytes()).hexdigest()
    return (
        x_train,
        y_train,
        x_test,
        y_test,
        {
            "dataset": "CaliforniaHousing",
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "d_in": int(x_train.shape[1]),
            "preprocessing": "standardized_features",
            "raw_hashes": {"data_npz": raw_hash, "target_npz": target_hash},
            "split_seed_derivation": "seed * 31337 + 11",
            "test_frac": test_frac,
        },
    )


def _c7_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    n_neurons = _parse_cell(str(config["cell"]))
    alpha = float(config["alpha"])
    gain_low = float(config["gain_low"])
    gain_high = float(config["gain_high"])
    chunk_size = int(config.get("chunk_size", 4000))
    activation = str(config.get("activation", "abs"))

    x_train, y_train, x_test, y_test, ds_meta = _load_california_housing(seed)
    d_in = int(x_train.shape[1])

    rngs = _make_subrngs_phase_c(seed)
    enc = make_encoders(n_neurons, d_in, strategy="hypersphere", rng=rngs["encoder"])
    if isinstance(enc, tuple):
        enc = enc[0]
    idx = torch.randint(len(x_train), (n_neurons,), generator=rngs["center_index"])
    centers = x_train[idx]
    gain = gain_low + (gain_high - gain_low) * torch.rand(n_neurons, generator=rngs["gain"])
    bias = -gain * (centers * enc).sum(dim=1)

    act = _activation_fn(activation)
    ATA = torch.zeros(n_neurons, n_neurons, dtype=torch.float32)
    ATY = torch.zeros(n_neurons, 1, dtype=torch.float32)

    t0 = time.perf_counter()
    for s in range(0, x_train.shape[0], chunk_size):
        e = min(s + chunk_size, x_train.shape[0])
        z = x_train[s:e] @ enc.T
        z.mul_(gain)
        z.add_(bias)
        act(z)
        ATA.addmm_(z.T, z)
        ATY.addmm_(z.T, y_train[s:e])
    encode_fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    reg = alpha * torch.trace(ATA) / n_neurons
    ATA.diagonal().add_(torch.clamp(reg, min=alpha))
    L = torch.linalg.cholesky(ATA)
    D = torch.cholesky_solve(ATY, L)
    solve_s = time.perf_counter() - t0

    def _mse(x, y):
        sse = 0.0
        n = 0
        for s in range(0, x.shape[0], chunk_size):
            e = min(s + chunk_size, x.shape[0])
            z = x[s:e] @ enc.T
            z.mul_(gain)
            z.add_(bias)
            act(z)
            preds = z @ D
            diff = preds - y[s:e]
            sse += float((diff * diff).sum())
            n += int(diff.numel())
        return sse / max(n, 1)

    train_mse = _mse(x_train, y_train)
    test_mse = _mse(x_test, y_test)
    test_var = float(((y_test - y_test.mean()) ** 2).mean())
    test_r2 = 1.0 - test_mse / max(test_var, 1e-12)

    return {
        "test_mse": round(test_mse, 6),
        "test_r2": round(test_r2, 6),
        "train_mse": round(train_mse, 6),
        "encode_fit_time_s": round(encode_fit_s, 4),
        "solve_time_s": round(solve_s, 4),
        "factors": {"n_neurons": n_neurons},
        "diagnostics": {"n_neurons": n_neurons, "d_in": d_in},
        "dataset_metadata": ds_meta,
    }


def _c7_context() -> dict:
    return {
        "experiment_kind": "california_housing_c7_panel",
        "dataset": "CaliforniaHousing",
        "preprocessing": "standardized_features",
    }


C7_BASE_CONFIG: dict[str, object] = {
    "cell": "n2000",
    "alpha": 1e-3,
    "gain_low": 0.5,
    "gain_high": 2.0,
    "chunk_size": 4000,
    "activation": "abs",
}


C7_DESCRIPTION = (
    "C7 California Housing regression (3 seeds). Knob 'cell' = "
    "'n{n_neurons}'.  Confirms TR §5.6 universality demo: same "
    "primitive solves regression by setting d_out=1.  Reports "
    "test_mse and test_r2 (1 - test_mse/var(y_test))."
)
