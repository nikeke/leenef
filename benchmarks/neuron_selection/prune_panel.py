"""Compare rankings and batched greedy OBS on nested random feature pools."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

from benchmarks.neuron_selection.common import (
    IMAGE_SHAPES,
    LOOP_ALPHA,
    RESULTS,
    Results,
    accumulate,
    cache_activities,
    comma_list,
    keyed_rng,
    load_cli_data,
    make_layer,
    make_split,
    panel_parser,
    parse_widths,
    reset_peak_rss,
    row_key,
    select_alpha,
    stats,
    subset,
)
from benchmarks.sl_clean.datasets import DatasetBundle
from leenef import obs_saliency, prune_from_stats, solve_from_normal_equations

STRATEGIES = ("hypersphere", "receptive_field")
CRITERIA = ("activity", "magnitude", "obs_oneshot", "obs")
POOL_MAX = 8000
POOLS = (2000, 4000, 8000)
KS = (250, 500, 1000, 2000)
RANDOM_WIDTHS = (250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000)


def _topk(score: Tensor, k: int) -> Tensor:
    return score.argsort(descending=True, stable=True)[:k].sort().values


def configs(
    dataset: str,
    strategy: str,
    pools: tuple[int, ...],
    ks: tuple[int, ...],
    random_widths: tuple[int, ...],
    **protocol: object,
) -> list[dict]:
    base = {"dataset": dataset, "encoder_strategy": strategy, **protocol}
    rows = [{**base, "P": None, "k": k, "criterion": "random"} for k in random_widths]
    for pool in pools:
        rows.append({**base, "P": pool, "k": pool, "criterion": "pool"})
        path = [k for k in sorted(ks, reverse=True) if k < pool]
        for i, k in enumerate(path):
            for criterion in CRITERIA:
                row = {**base, "P": pool, "k": k, "criterion": criterion}
                if criterion == "obs":
                    row["obs_path"] = path[: i + 1]
                rows.append(row)
    return rows


def panel_configs(
    dataset: str,
    *,
    strategies: tuple[str, ...] = STRATEGIES,
    pool_max: int = POOL_MAX,
    pools: tuple[int, ...] = POOLS,
    ks: tuple[int, ...] = KS,
    random_widths: tuple[int, ...] = RANDOM_WIDTHS,
    image_shape: tuple[int, int] | None = None,
    val_size: int = 10000,
    fit_limit: int | None = None,
    chunk_size: int | None = None,
) -> dict[str, list[dict]]:
    if not strategies or set(strategies) - set(STRATEGIES):
        raise ValueError(f"strategies must be chosen from {STRATEGIES}")
    if not pools or min(pools) <= 0 or max(pools) > pool_max or not ks or min(ks) <= 0:
        raise ValueError("widths must be positive and pools must not exceed pool_max")
    if random_widths and (min(random_widths) < 1 or max(random_widths) > pool_max):
        raise ValueError("random widths must be positive and not exceed pool_max")
    chunk_size = chunk_size or (2500 if dataset == "CIFAR10" else 5000)
    image_shape = image_shape or IMAGE_SHAPES[dataset]
    setup = {
        "pool_max": pool_max,
        "val_size": val_size,
        "fit_limit": fit_limit,
        "chunk_size": chunk_size,
        "image_shape": list(image_shape),
    }
    return {
        strategy: configs(dataset, strategy, pools, ks, random_widths, **setup)
        for strategy in strategies
    }


def run_panel(bundle: DatasetBundle, *, seeds: list[int], out: Path, **options) -> dict:
    all_configs = panel_configs(bundle.name, **options)
    store = Results(out, "prune", bundle)
    protocol = next(iter(all_configs.values()))[0]
    pool_max, chunk_size = protocol["pool_max"], protocol["chunk_size"]
    image_shape = tuple(protocol["image_shape"])
    for seed in seeds:
        pending = {
            strategy: cells
            for strategy, cases in all_configs.items()
            if (cells := [cfg for cfg in cases if row_key(cfg, seed) not in store.completed])
        }
        if not pending:
            continue
        split = make_split(
            bundle, seed, val_size=protocol["val_size"], fit_limit=protocol["fit_limit"]
        )
        fit_subset = subset(len(split.x_fit), keyed_rng(seed, "eval_subset"))
        evaluation = split.evaluation(fit_subset)
        for strategy, cells in pending.items():
            start = perf_counter()
            layer = make_layer(split, pool_max, strategy, image_shape, keyed_rng(seed, "pool"))
            accumulate(layer, split, chunk_size)
            after_fit = perf_counter()
            normal, cross = stats(layer)
            cached = tuple((cache_activities(layer, x, chunk_size), y) for x, y in evaluation)
            cache_time = perf_counter() - after_fit
            for cfg in cells:
                if cfg["P"] is not None:
                    continue
                rss_reset = reset_peak_rss()
                t0 = perf_counter()
                k = cfg["k"]
                metrics = select_alpha(
                    normal[:k, :k], cross[:k], cached, chunk_size=chunk_size, columns=slice(k)
                )
                store.add(
                    cfg,
                    seed,
                    {
                        **metrics,
                        "pool_accumulate_time_s": after_fit - start,
                        "cache_time_s": cache_time,
                        "wall_time_s": perf_counter() - t0,
                    },
                    rss_reset=rss_reset,
                )
            for pool in sorted({cfg["P"] for cfg in cells if cfg["P"] is not None}):
                _run_pool(
                    normal[:pool, :pool],
                    cross[:pool],
                    cached,
                    pool,
                    [cfg for cfg in cells if cfg["P"] == pool],
                    store,
                    seed,
                    chunk_size,
                    after_fit - start,
                    cache_time,
                )
    return store.payload


def _run_pool(
    normal: Tensor,
    cross: Tensor,
    cached: tuple[tuple[Tensor, Tensor], ...],
    pool: int,
    pending: list[dict],
    store: Results,
    seed: int,
    chunk_size: int,
    pool_accumulate_time: float,
    cache_time: float,
) -> None:
    by_cell = {(cfg["criterion"], cfg["k"]): cfg for cfg in pending}
    cache = tuple((a[:, :pool], y) for a, y in cached)
    if cfg := by_cell.get(("pool", pool)):
        rss_reset = reset_peak_rss()
        t0 = perf_counter()
        metrics = select_alpha(normal, cross, cache, chunk_size=chunk_size)
        store.add(
            cfg,
            seed,
            {
                **metrics,
                "pool_accumulate_time_s": pool_accumulate_time,
                "cache_time_s": cache_time,
                "wall_time_s": perf_counter() - t0,
            },
            rss_reset=rss_reset,
        )
    rms = normal.diagonal().clamp_min(0).sqrt()
    obs_widths = [k for crit, k in by_cell if crit == "obs"]

    def evaluate(
        criterion: str, k: int, keep: Tensor, started: float, rss_reset: bool, **timings: float
    ) -> None:
        cfg = by_cell[criterion, k]
        g = normal[keep[:, None], keep]
        scores = select_alpha(g, cross[keep], cache, chunk_size=chunk_size, columns=keep)
        store.add(
            cfg,
            seed,
            {**scores, **timings, "wall_time_s": perf_counter() - started},
            rss_reset=rss_reset,
        )

    for criterion in CRITERIA[:-1]:
        widths = [k for crit, k in by_cell if crit == criterion]
        if not widths:
            continue
        rss_reset = reset_peak_rss()
        start = perf_counter()
        if criterion == "activity":
            score = rms
        elif criterion == "magnitude":
            score = solve_from_normal_equations(normal, cross, alpha=LOOP_ALPHA).norm(dim=1) * rms
        else:
            score = obs_saliency(normal, cross, alpha=LOOP_ALPHA)
        for i, k in enumerate(widths):
            if i:
                rss_reset = reset_peak_rss()
            evaluate(criterion, k, _topk(score, k), start, rss_reset)
            start = perf_counter()

    if obs_widths:
        keep = torch.arange(pool)
        elapsed = 0.0
        # Resume may still need intermediate keep sets for a pending smaller width.
        for k in by_cell["obs", min(obs_widths)]["obs_path"]:
            rss_reset = reset_peak_rss() if k in obs_widths else False
            t0 = perf_counter()
            if keep.numel() == pool:
                g, r = normal, cross
            else:
                g, r = normal[keep[:, None], keep], cross[keep]
            keep = keep[
                prune_from_stats(g, r, k, method="downdate", drop_fraction=0.05, alpha=LOOP_ALPHA)
            ]
            elimination = perf_counter() - t0
            elapsed += elimination
            if k in obs_widths:
                evaluate(
                    "obs",
                    k,
                    keep,
                    t0,
                    rss_reset,
                    elimination_time_s=elimination,
                    cumulative_elimination_time_s=elapsed,
                )


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = panel_parser(__doc__)
    parser.add_argument("--strategies", type=comma_list, default=STRATEGIES)
    parser.add_argument("--pool-max", type=int)
    parser.add_argument("--pools", type=parse_widths)
    parser.add_argument("--ks", type=parse_widths)
    parser.add_argument("--random-widths", type=parse_widths)
    return parser.parse_args(argv)


def cli_options(args: Namespace) -> dict:
    pool_max = args.pool_max if args.pool_max is not None else (400 if args.smoke else POOL_MAX)
    default_pools = (100, 200, 400) if args.smoke else POOLS
    default_random = (50, 100, 150, 300, 400) if args.smoke else RANDOM_WIDTHS
    return {
        "strategies": args.strategies,
        "pool_max": pool_max,
        "pools": args.pools or tuple(p for p in default_pools if p <= pool_max) or (pool_max,),
        "ks": args.ks or ((50, 100) if args.smoke else KS),
        "random_widths": args.random_widths or tuple(k for k in default_random if k <= pool_max),
        "fit_limit": 2000 if args.smoke else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_panel(
        load_cli_data(args),
        seeds=args.seeds,
        out=args.out or RESULTS / f"prune_{args.dataset}.json",
        **cli_options(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
