"""Compare directed growth under a fixed budget with batched greedy OBS."""

from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

from benchmarks.neuron_selection.common import (
    IMAGE_SHAPES,
    LOOP_ALPHA,
    RESULTS,
    Results,
    Split,
    accumulate,
    comma_list,
    encoder_kwargs,
    keyed_rng,
    load_cli_data,
    make_layer,
    make_split,
    panel_parser,
    parse_widths,
    peak_rss,
    reset_peak_rss,
    row_key,
    select_alpha,
    stats,
    subset,
)
from benchmarks.sl_clean.datasets import DatasetBundle
from leenef import (
    NEFLayer,
    agop,
    make_encoders,
    phd_bases,
    prune_from_stats,
    psd_power,
    residual_moments,
    sample_directions_from_bases,
    sample_patch_directions_from_moments,
    sample_shaped_directions,
)

FAMILIES = {
    "global": (
        "hypersphere",
        (
            "random",
            "phd",
            "agop",
            "phd_classshuffle",
            "phd_frozen",
            "pca",
            "first_order",
            "phd_randkeep",
        ),
    ),
    "patch": ("receptive_field", ("random", "phd_patch")),
}
DEFAULT_KS = (250, 1000)
DEFAULT_GENERATIONS = 12
GLOBAL_PHD_ARMS = {"phd", "phd_classshuffle", "phd_frozen", "phd_randkeep"}


@contextmanager
def _eigen_threads():
    """Avoid measured CPU LAPACK contention for the full PCA and AGOP eigensolves."""
    threads = torch.get_num_threads()
    if threads > 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if threads > 1:
            torch.set_num_threads(threads)


def _patch_proposals(moments: Tensor, encoders: Tensor, rng: torch.Generator) -> Tensor:
    mask = encoders != 0
    sizes = mask.sum(dim=1)
    groups = sizes.unique(sorted=True)
    if len(groups) == 1:
        supports = mask.nonzero()[:, 1].reshape(len(encoders), int(groups[0]))
        return sample_patch_directions_from_moments(moments, supports, 8, rng=rng)

    directions = torch.empty_like(encoders)
    for size in groups.tolist():
        rows = (sizes == size).nonzero(as_tuple=True)[0]
        supports = mask[rows].nonzero()[:, 1].reshape(len(rows), size)
        directions[rows] = sample_patch_directions_from_moments(moments, supports, 8, rng=rng)
    return directions


def _residual(layer: NEFLayer, split: Split, chunk_size: int) -> Tensor:
    prediction = torch.cat([layer(batch) for batch in split.x_fit.split(chunk_size)])
    return split.y_targets - prediction


def _classshuffle(
    residual: Tensor, labels: Tensor, n_classes: int, rng: torch.Generator
) -> Tensor:
    shuffled = torch.empty_like(residual)
    for c in range(n_classes):
        indices = (labels == c).nonzero(as_tuple=True)[0]
        order = indices[torch.randperm(len(indices), generator=rng)]
        # A randomized cycle avoids self-matches except in singleton classes.
        shuffled[order] = residual[order.roll(1)]
    return shuffled


def _pca_bases(split: Split, mean: Tensor, chunk_size: int) -> Tensor:
    cov = split.x_fit.new_zeros((mean.numel(), mean.numel()))
    for batch in split.x_fit.split(chunk_size):
        centered = batch - mean
        cov.addmm_(centered.T, centered)
    cov.div_(len(split.x_fit))
    # This eigendecomposition measures input covariance; it is not a ridge solve.
    with _eigen_threads():
        values, vectors = torch.linalg.eigh(cov)
    return (vectors[:, -40:] * values[-40:].clamp_min_(0).sqrt()).flip(1).unsqueeze(0)


def _eig_metrics(info: dict) -> dict:
    """No solve is recorded at generation zero or when frozen bases are reused."""
    return {
        "eig_iterations": info.get("iterations"),
        "eig_worst_change": info.get("worst_relative_change"),
        "eig_converged": info.get("converged"),
    }


def _propose(
    layer: NEFLayer,
    split: Split,
    mean: Tensor,
    agop_inputs: Tensor | None,
    strategy: str,
    arm: str,
    seed: int,
    gen: int,
    image_shape: tuple[int, int],
    chunk_size: int,
    frozen_bases: Tensor | None,
    eig_info: dict,
) -> tuple[Tensor, Tensor | None]:
    k = layer.n_neurons
    rng = keyed_rng(seed, "proposal", gen)
    if arm == "random":
        return make_encoders(
            k, layer.d_in, strategy=strategy, rng=rng, **encoder_kwargs(strategy, image_shape)
        ), None
    if arm == "agop":
        assert agop_inputs is not None
        gradient_moment = agop(layer, agop_inputs, batch_size=chunk_size)
        with _eigen_threads():
            shaping = psd_power(gradient_moment, 0.5)
        return sample_shaped_directions(k, shaping, rng=rng), None
    if arm == "pca":
        bases = _pca_bases(split, mean, chunk_size) if frozen_bases is None else frozen_bases
        return sample_directions_from_bases(k, bases, rng=rng), bases
    if frozen_bases is not None:
        return sample_directions_from_bases(k, frozen_bases, rng=rng), frozen_bases

    residual = _residual(layer, split, chunk_size)
    if arm == "first_order":
        means = residual.T @ split.x_fit - residual.sum(dim=0)[:, None] * mean
        return sample_directions_from_bases(k, means[..., None], rng=rng), None
    if arm == "phd_classshuffle":
        residual = _classshuffle(
            residual, split.y_fit, layer.d_out, keyed_rng(seed, "shuffle", gen)
        )
    moments = residual_moments(split.x_fit, residual, mean=mean, batch_size=chunk_size)
    if arm == "phd_patch":
        rf = make_encoders(
            k, layer.d_in, strategy=strategy, rng=rng, **encoder_kwargs(strategy, image_shape)
        )
        return _patch_proposals(moments, rf, rng), None
    bases = phd_bases(
        moments, 40, method="subspace", rng=keyed_rng(seed, "eig", gen), info=eig_info
    )
    if not eig_info["converged"]:
        print(
            f"pHd did not converge: arm={arm} seed={seed} generation={gen} {eig_info}", flush=True
        )
    del moments
    return sample_directions_from_bases(k, bases, rng=rng), bases if arm == "phd_frozen" else None


@torch.no_grad()
def run_trial(
    split: Split,
    mean: Tensor,
    *,
    family: str,
    arm: str,
    k: int,
    seed: int,
    generations: int,
    image_shape: tuple[int, int],
    chunk_size: int,
    rss_reset: bool = False,
) -> dict:
    """Fit and run one complete growth row; all arms restart the same streams."""
    t0 = start = perf_counter()
    strategy, arms = FAMILIES[family]
    if arm not in arms:
        raise ValueError(f"unsupported arm {arm!r} for {family!r}")
    layer = make_layer(split, k, strategy, image_shape, keyed_rng(seed, "pool"))
    evaluation = split.evaluation()
    agop_inputs = (
        split.x_fit[subset(len(split.x_fit), keyed_rng(seed, "subsample"))]
        if arm == "agop"
        else None
    )
    accumulate(layer, split, chunk_size)
    after_fit = perf_counter()
    metrics = select_alpha(*stats(layer), evaluation, chunk_size=chunk_size, layer=layer)
    trajectory = [
        {
            "gen": 0,
            **metrics,
            **(_eig_metrics({}) if arm in GLOBAL_PHD_ARMS else {}),
            "n_survivors": None,
            "accumulate_time_s": after_fit - start,
            "evaluation_time_s": perf_counter() - after_fit,
            "wall_time_s": perf_counter() - t0,
        }
    ]
    frozen_bases = None
    for gen in range(1, generations + 1):
        start = perf_counter()
        eig_info = {}
        directions, frozen_bases = _propose(
            layer,
            split,
            mean,
            agop_inputs,
            strategy,
            arm,
            seed,
            gen,
            image_shape,
            chunk_size,
            frozen_bases,
            eig_info,
        )
        layer.grow(directions, centers=split.x_fit, rng=keyed_rng(seed, "center", gen))
        after_proposal = perf_counter()
        accumulate(layer, split, chunk_size)
        after_fit = perf_counter()
        if arm == "phd_randkeep":
            keep = (
                torch.randperm(2 * k, generator=keyed_rng(seed, "retain", gen))[:k].sort().values
            )
        else:
            keep = prune_from_stats(
                *stats(layer),
                k,
                alpha=LOOP_ALPHA,
                method="downdate",
                drop_fraction=0.1,
            )
        n_survivors = int((keep < k).sum())
        layer.prune(keep)
        after_prune = perf_counter()
        metrics = select_alpha(*stats(layer), evaluation, chunk_size=chunk_size, layer=layer)
        end = perf_counter()
        trajectory.append(
            {
                "gen": gen,
                **metrics,
                **(_eig_metrics(eig_info) if arm in GLOBAL_PHD_ARMS else {}),
                "n_survivors": n_survivors,
                "proposal_time_s": after_proposal - start,
                "accumulate_time_s": after_fit - after_proposal,
                "prune_time_s": after_prune - after_fit,
                "evaluation_time_s": end - after_prune,
                "wall_time_s": end - start,
                "peak_rss_mb": peak_rss(rss_reset)[0],
            }
        )
    return {
        **metrics,
        "generations": trajectory,
        "final": trajectory[-1],
        "val_selected": max(trajectory, key=lambda row: row["val_accuracy"]),
        "wall_time_s": perf_counter() - t0,
    }


def configs(
    dataset: str,
    *,
    families: tuple[str, ...] = tuple(FAMILIES),
    arms: tuple[str, ...] | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
    generations: int = DEFAULT_GENERATIONS,
    image_shape: tuple[int, int] | None = None,
    val_size: int = 10000,
    fit_limit: int | None = None,
    chunk_size: int | None = None,
) -> list[dict]:
    if not families or set(families) - FAMILIES.keys():
        raise ValueError(f"families must be chosen from {tuple(FAMILIES)}")
    if set(arms or ()) - {arm for _, names in FAMILIES.values() for arm in names}:
        raise ValueError("unknown growth arm")
    if generations < 0 or not ks or min(ks) < 1:
        raise ValueError("generations must be nonnegative and widths must be positive")
    cells = [
        (family, arm, k)
        for family in families
        for arm in FAMILIES[family][1]
        if arms is None or arm in arms
        for k in ks
    ]
    if not cells:
        raise ValueError("no requested arm belongs to a requested family")
    chunk_size = chunk_size or (2500 if dataset == "CIFAR10" else 5000)
    image_shape = image_shape or IMAGE_SHAPES[dataset]
    return [
        {
            "dataset": dataset,
            "family": family,
            "encoder_strategy": FAMILIES[family][0],
            "arm": arm,
            "k": k,
            "n_generations": generations,
            "val_size": val_size,
            "fit_limit": fit_limit,
            "chunk_size": chunk_size,
            "image_shape": list(image_shape),
            "phd_method": "subspace"
            if arm in GLOBAL_PHD_ARMS
            else "eigh"
            if arm == "phd_patch"
            else None,
            "eigh_threads": 1 if arm in ("agop", "pca") else None,
        }
        for family, arm, k in cells
    ]


def run_panel(bundle: DatasetBundle, *, seeds: list[int], out: Path, **options) -> dict:
    cases = configs(bundle.name, **options)
    store = Results(out, "growth", bundle)
    for seed in seeds:
        pending = [cfg for cfg in cases if row_key(cfg, seed) not in store.completed]
        if not pending:
            continue
        split = make_split(
            bundle, seed, val_size=cases[0]["val_size"], fit_limit=cases[0]["fit_limit"]
        )
        mean = split.x_fit.mean(dim=0)
        for cfg in pending:
            rss_reset = reset_peak_rss()
            row = run_trial(
                split,
                mean,
                family=cfg["family"],
                arm=cfg["arm"],
                k=cfg["k"],
                seed=seed,
                generations=cfg["n_generations"],
                image_shape=tuple(cfg["image_shape"]),
                chunk_size=cfg["chunk_size"],
                rss_reset=rss_reset,
            )
            store.add(cfg, seed, row, rss_reset=rss_reset)
    return store.payload


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = panel_parser(__doc__)
    parser.add_argument("--families", type=comma_list, default=tuple(FAMILIES))
    parser.add_argument("--arms", type=comma_list)
    parser.add_argument("--ks", type=parse_widths)
    parser.add_argument("--generations", type=int)
    return parser.parse_args(argv)


def cli_options(args: Namespace) -> dict:
    return {
        "families": args.families,
        "arms": args.arms,
        "ks": args.ks or ((50, 100) if args.smoke else DEFAULT_KS),
        "generations": args.generations
        if args.generations is not None
        else (2 if args.smoke else DEFAULT_GENERATIONS),
        "fit_limit": 2000 if args.smoke else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_panel(
        load_cli_data(args),
        seeds=args.seeds,
        out=args.out or RESULTS / f"growth_{args.dataset}.json",
        **cli_options(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
