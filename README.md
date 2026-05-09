# leenef

Machine learning with a rate-based specialization of the Neural
Engineering Framework (NEF) in PyTorch.

> **Repository status — SSUP trilogy rebuild.** The stable core lives in
> `src/leenef/` and `tests/`.  Active supervised-learning experiments and
> preserved artifacts live under `benchmarks/sl_clean/`,
> `results/sl_clean/`, and `docs/SSUP-I.md`; older pre-restart
> infrastructure remains in `attic/` for reference.

## Overview

leenef keeps the NEF encoder/decoder split but uses rate neurons and an
analytic decoder solve for machine learning:

```text
a = G(g ⊙ (E x) + b)
D = (AᵀA + α_eff I)⁻¹ AᵀY
y = Dᵀa
```

The practical consequence is that the expensive part of training is not
backpropagation through all weights, but accumulation of the sufficient
statistics `AᵀA` and `AᵀY`.  The same primitive supports:

- single-layer supervised learning with `NEFLayer`
- ensembles with `NEFEnsemble`
- online / continuous updates via `partial_fit`, `solve_accumulated`,
  and Woodbury-style inverse updates
- temporal classification with `StreamingNEFClassifier`
- multi-layer and recurrent variants for controlled comparisons
- reinforcement-learning agents built on the same analytic solve

The current supervised-learning paper is
[`docs/SSUP-I.md`](docs/SSUP-I.md).  It is the authoritative source for
the repository's present claims, numbers, and caveats.

Paper I frames the current supervised-learning story like this:

> NEF's regularized least-squares solve over fixed-encoder neural
> activities — `D = (AᵀA + α_eff I)⁻¹ AᵀY` — gives a compact
> sufficient-statistics primitive for supervised learning.  In a
> rate-based specialization we call **leenef**, we interpret the
> activities as magnitudes of frame coefficients and study how
> encoder choice, bias placement, activation, and regularization
> shape the resulting fixed-feature learner.  The resulting primitive
> composes into ensembles, online updates (Woodbury),
> GPU-friendly accumulate-and-solve, and, non-trivially, a streaming
> temporal classifier.  Under multi-seed, code-identity-tagged
> evaluation, the analytic streaming classifier reaches
> **98.45 ± 0.02 %** on row sMNIST: **0.6 pp below** a tuned near-best
> LSTM while using **2.4× less wall-clock time**, and **~5 pp above**
> the completed ESN row panel while using **roughly 5× less wall-clock
> time**.  Static single-layer leenef models with generic encoders end
> **1–4 pp below** gradient-trained MLPs on image features while using
> **12–50× less wall-clock time**; on CIFAR-10, adding a
> receptive-field encoder reverses that gap (**+1.37 pp**) with
> **~11× less wall-clock time**, and a gradient-free convolutional
> feature stack (ConvNEF, §5.5) lifts the same head by a further
> **+10.78 pp** to **65.89 ± 0.12 %**.  The same evidence also marks clear
> limits: generic fixed encoders are not a substitute for spatial
> priors, purely analytic depth does not buy accuracy on its own, and
> mean-pooled delay-line features lose the long-range order
> information needed for harder sequence tasks.

## Quick start

```python
from leenef.layers import NEFLayer
from leenef.ensemble import NEFEnsemble
from leenef.streaming import StreamingNEFClassifier

# Single layer: fixed encoders, analytic decoder solve
layer = NEFLayer(d_in=784, n_neurons=5000, d_out=10, centers=x_train)
layer.fit(x_train, y_train)
predictions = layer(x_test)

# Incremental / online accumulation
online = NEFLayer(d_in=784, n_neurons=5000, d_out=10, centers=x_train)
for batch_x, batch_y in data_stream:
    online.partial_fit(batch_x, batch_y)
online.solve_accumulated(alpha=1e-3)

# Ensemble of independent feature banks
ens = NEFEnsemble(
    d_in=784,
    n_neurons=3000,
    d_out=10,
    n_members=10,
    encoder_strategy="receptive_field",
    encoder_kwargs={"image_shape": (28, 28)},
    centers=x_train,
)
ens.fit(x_train, y_train)

# Streaming classifier on sequence data
stream = StreamingNEFClassifier(
    d_timestep=1,
    n_neurons=8000,
    d_out=10,
    window_size=10,
    centers=x_train_seq,
)
stream.fit(x_train_seq, y_train)
```

## Setup

Requires Python 3.12+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

## Tests

```bash
source venv/bin/activate
pytest
pytest -k test_fit_identity -q
ruff check src/leenef/ tests/ benchmarks/
ruff format --check src/leenef/ tests/ benchmarks/
```

## Benchmarks and paper artifacts

The active supervised-learning harness is the `sl_clean` stack used for
Paper I.  Example launches:

```bash
source venv/bin/activate

PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_mnist
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_fashion
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_cifar10

# Follow-up bias-placement panels
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_cifar10_centered
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_mnist_decentered

# Regenerate the paper figures
python3 docs/generate_paper_i_figures.py
```

All paper-grade result rows carry a code identity, dataset hashes, full
config dictionaries, per-seed metrics, and aggregate summaries.  The
preserved JSON/log artifacts used by Paper I live in `results/sl_clean/`.

## Components

| Path | Purpose |
|---|---|
| `src/leenef/layers.py` | `NEFLayer`: encode -> activate -> decode; batch, accumulated, and continuous analytic solves |
| `src/leenef/ensemble.py` | `NEFEnsemble`: independent NEF layers combined by averaging or voting |
| `src/leenef/networks.py` | `NEFNetwork`: multi-layer comparison model with greedy, hybrid, target-prop, end-to-end, hybrid->E2E, and target-prop->E2E training |
| `src/leenef/recurrent.py` | `RecurrentNEFLayer`: recurrent decode-then-re-encode sequence model |
| `src/leenef/streaming.py` | `StreamingNEFClassifier`: delay-line temporal classifier with batch, Woodbury, and accumulate+solve training |
| `src/leenef/conv.py` | `ConvNEFStage`, `ConvNEFPipeline`, `ConvNEFEnsemble`: gradient-free convolutional front ends and ensembles |
| `src/leenef/rl.py` | `NEFFeatures`, `NEFFQIAgent`, `NEFFQIEnsemble`: RL agents and ensembles built on analytic value solves |
| `src/leenef/encoders.py` | Encoder strategies: `hypersphere`, `gaussian`, `sparse`, `receptive_field`, `whitened`, `class_contrast`, `local_pca` |
| `src/leenef/activations.py` | Activations: `abs`, `relu`, `softplus`, `lif_rate` |
| `src/leenef/solvers.py` | Decoder solvers and normal-equation utilities |
| `benchmarks/sl_clean/` | Active reproducible supervised-learning sweep harness for Paper I |
| `results/sl_clean/` | Preserved multi-seed result rows, summaries, and logs used by Paper I |
| `attic/` | Archived pre-restart infrastructure, reports, and notes kept for historical reference |
