# leenef

Machine learning with a rate-based specialization of the Neural
Engineering Framework (NEF) in PyTorch.

> **Repository status — SSUP paper series rebuild.** The stable core lives in
> `src/leenef/` and `tests/`.  Active supervised-learning,
> continual-learning, and reinforcement-learning experiments and preserved
> artifacts live under `benchmarks/`, `results/`, and
> `docs/SSUP-{I,II,III}.md`; older pre-restart infrastructure remains in
> `attic/` for reference.

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

The current supervised-learning, continual-learning, and
reinforcement-learning papers are [`docs/SSUP-I.md`](docs/SSUP-I.md),
[`docs/SSUP-II.md`](docs/SSUP-II.md), and
[`docs/SSUP-III.md`](docs/SSUP-III.md).  They are the authoritative
source for the repository's present claims, numbers, and caveats.

Paper I abstract:

> NEF's regularized least-squares solve over fixed-encoder neural
> activities — $D = (A^\top A + \alpha_{\mathrm{eff}} I)^{-1} A^\top Y$
> — is a compact sufficient-statistics primitive for supervised learning.
> In a rate-based specialization we call **leenef**, fixed encoders
> produce frame-coefficient-like activities and only the decoder is fit.
> This primitive supports fast static classifiers, ensembles, online
> Woodbury updates, GPU-friendly accumulate-and-solve, and a streaming
> temporal classifier.  Under multi-seed, code-identity-tagged evaluation,
> Stream-NEF reaches **98.45 ± 0.02 %** on row sMNIST: **0.6 pp below** a
> tuned near-best LSTM while using **2.4× less wall-clock time**, and
> **~5 pp above** the completed ESN row panel.  Static image classifiers
> are fast but feature-limited: generic encoders trail tuned MLPs by
> **1–4 pp**, receptive-field encoders reverse the CIFAR-10 MLP gap, and a
> gradient-free ConvNEF front end reaches **65.89 ± 0.12 %** on CIFAR-10.
> The limits are equally clear: spatial priors matter, purely analytic
> depth does not buy accuracy on its own, and mean-pooled delay-line
> features lose long-range order information.

Paper II abstract:

> NEF's regularized least-squares solve over fixed-encoder neural
> activities — $D = (A^\top A + \alpha_{\mathrm{eff}} I)^{-1} A^\top Y$
> — is a compact sufficient-statistics primitive for continual learning.
> Because $A^\top A$ and $A^\top Y$ add over data partitions, sequential
> `partial_fit` over a task stream produces the **same** decoder as joint
> training over the union of tasks: the forgetting axis has length zero by
> construction relative to that joint fixed-feature model.  Across
> Split-MNIST, Permuted-MNIST, Split-CIFAR-10, and Split-CIFAR-100,
> `nef_accumulate` matches `nef_joint` to 0.000000 pp over 10 seeds.  This
> removes forgetting from the analytic readout and collapses the
> class-incremental failure modes of the tested EWC/finetune baselines
> (Split-MNIST: **97.0 %** vs ~20.0 %), while remaining above ER at
> 86.5 % and exposing the tradeoff directly: the linear head no longer
> forgets, but absolute accuracy is bounded by the fixed feature map.
> Capacity scales smoothly with neuron budget
> per task, a gradient-free ConvNEF front end lifts Split-CIFAR-10 from
> 49.5 % to **74.7 $\pm$ 0.1 %** without breaking the identity, and a
> streaming-continual panel carries the same exact/joint/Woodbury
> agreement through 10-task permuted sMNIST while revealing a lower
> mean-pooled-delay-line ceiling of **58.3 $\pm$ 0.2 %** at 16000 neurons.

Paper III abstract:

> NEF's regularized least-squares solve over fixed-encoder neural
> activities,
> $D = (A^\top A + \alpha_{\mathrm{eff}} I)^{-1} A^\top Y$, can be used as
> a compact sufficient-statistics primitive for reinforcement learning when
> the target matrix $Y$ is constructed to match the control problem.  In
> dense control, a non-bootstrapped $100$-step NEF action-value regression
> (NEF-AVR) configuration reaches $201.3 \pm 9.8$ robust return on
> LunarLander-v3 over 10 seeds, crossing the informal 200-return reference
> line under the evaluation protocol used here; the same analytic control
> path reaches near-perfect CartPole-v1 best
> evaluation (`best_eval` $=499.4 \pm 0.6$).  A baseline DQN implementation
> evaluated under the exact same protocol reaches $129.3 \pm 21.7$ on
> LunarLander-v3.  In sparse
> control, target construction is the result: a full-successor
> random-direction mechanism breaks the flat MountainCar-v0 floor but does
> not solve the environment, reaching
> $-158.8 \pm 1.4$ at 2000 neurons, and a 6000-neuron directional
> known-goal readout gives a stable Acrobot-v1 signal
> ($-194.9 \pm 10.4$ robust, $-132.1 \pm 6.1$ best) when successor
> statistics are retained slowly enough.  The boundary rows explain the
> conditions rather than replacing the headline: one-step bootstrapped TD
> collapses on LunarLander-v3 ($-123.8 \pm 13.1$), the cheaper directional
> MountainCar readout stays flat ($-197.0 \pm 1.7$), and widening the full
> MountainCar successor model to 6000 neurons costs $5.7\times$ more
> without improving robust return.  The paper's claim is therefore not that
> analytic fixed-feature RL is a universal control algorithm.  It is that
> the same sufficient-statistics object can support successful control when
> $Y$ is stable, informative, and matched to the task.

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

From PyPI or a built wheel:

```bash
pip install leenef
```

For development from a checkout:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

See [`docs/USAGE.md`](docs/USAGE.md) for the usage manual and
[`SKILL.md`](SKILL.md) for method-selection guidance that coding agents can
install as a standard skill.

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

# Regenerate the Paper-I figures
python3 docs/generate_paper_i_figures.py

# Regenerate the Paper-III figures
python3 docs/generate_paper_iii_figures.py

# Build Paper I / Paper II / Paper III PDFs
bash docs/build_ssup_pdf.sh docs/SSUP-I.md docs/SSUP-I.pdf
bash docs/build_ssup_pdf.sh docs/SSUP-II.md docs/SSUP-II.pdf
bash docs/build_ssup_pdf.sh docs/SSUP-III.md docs/SSUP-III.pdf
```

All paper-grade result rows carry a code identity, dataset hashes, full
config dictionaries, per-seed metrics, and aggregate summaries.  The
preserved JSON/log artifacts used by Papers I, II, and III live under
`results/sl_clean/`, `results/cl_clean/`, `results/rl_paper/`, and
`results/rl_sparse/`.

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
| `benchmarks/cl_clean/` | Active reproducible continual-learning sweep harness for Paper II |
| `results/sl_clean/` | Preserved multi-seed result rows, summaries, and logs used by Paper I |
| `results/cl_clean/` | Preserved multi-seed result rows, summaries, and logs used by Paper II |
| `attic/` | Archived pre-restart infrastructure, reports, and notes kept for historical reference |
