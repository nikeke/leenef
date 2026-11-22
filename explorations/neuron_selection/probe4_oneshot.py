"""Exploratory probe (NOT confirmatory): is iterative grow-and-prune needed?

Compares, on validation accuracy only, one-shot pools pruned once by OBS:
  rand      : k random + (P-k) random
  phd       : k random, fit, residual pHd bases, (P-k) pHd proposals
  label     : model-free pHd from centered one-hot labels (class covariance contrasts)
against the archived NS-1 iterative result (12 generations, 13k candidates seen).
"""

import argparse
import json
import sys
from time import perf_counter

import torch

from benchmarks.neuron_selection.common import (
    IMAGE_SHAPES,
    LOOP_ALPHA,
    accumulate,
    keyed_rng,
    make_layer,
    make_split,
    select_alpha,
    stats,
)
from benchmarks.sl_clean.datasets import load_dataset
from leenef.encoders import make_encoders
from leenef.selection import (
    phd_bases,
    prune_from_stats,
    residual_moments,
    sample_directions_from_bases,
)

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--k", type=int, default=1000)
p.add_argument("--pools", default="2,4,8,13")
p.add_argument("--arms", default="rand,phd,label")
p.add_argument("--rounds", type=int, default=1)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--out", required=True)
args = p.parse_args()
torch.set_num_threads(args.threads)
CH = 4096
bundle = load_dataset(args.dataset, preprocessing="div255_flat")
shape = IMAGE_SHAPES[args.dataset]
k = args.k
rows = []


def val(layer, split):
    m = select_alpha(*stats(layer), split.evaluation(), chunk_size=CH, layer=layer)
    return m["val_accuracy"]  # test metrics deliberately discarded


@torch.no_grad()
def trial(split, mean, arm, mult, seed):
    t0 = perf_counter()
    layer = make_layer(split, k, "hypersphere", shape, keyed_rng(seed, "pool"))
    accumulate(layer, split, CH)
    layer.solve_accumulated(alpha=LOOP_ALPHA)
    out = []
    for rnd in range(1, args.rounds + 1):
        n_new = mult * k - k
        rng = keyed_rng(seed, "proposal", rnd)
        if arm == "rand":
            d = make_encoders(n_new, layer.d_in, strategy="hypersphere", rng=rng)
        else:
            if arm == "label":
                res = split.y_targets - split.y_targets.mean(dim=0)
            else:
                pred = torch.cat([layer(b) for b in split.x_fit.split(CH)])
                res = split.y_targets - pred
            mom = residual_moments(split.x_fit, res, mean=mean, batch_size=CH)
            info = {}
            bases = phd_bases(mom, 40, method="subspace", rng=keyed_rng(seed, "eig", rnd), info=info)
            del mom
            d = sample_directions_from_bases(n_new, bases, rng=rng)
        layer.grow(d, centers=split.x_fit, rng=keyed_rng(seed, "center", rnd))
        accumulate(layer, split, CH)
        keep = prune_from_stats(
            *stats(layer), k, alpha=LOOP_ALPHA, method="downdate", drop_fraction=0.1
        )
        surv = int((keep < k).sum())
        layer.prune(keep)
        out.append((val(layer, split), surv))
    return out, perf_counter() - t0


for seed in [int(s) for s in args.seeds.split(",")]:
    split = make_split(bundle, seed)
    mean = split.x_fit.mean(dim=0)
    for arm in args.arms.split(","):
        for mult in [int(m) for m in args.pools.split(",")]:
            out, dt = trial(split, mean, arm, mult, seed)
            row = dict(dataset=args.dataset, seed=seed, arm=arm, k=k, pool=mult * k,
                       val=[o[0] for o in out], survivors=[o[1] for o in out], time_s=dt)
            rows.append(row)
            print(json.dumps(row), flush=True)
            json.dump(rows, open(args.out, "w"))
sys.exit(0)
