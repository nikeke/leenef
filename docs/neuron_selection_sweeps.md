# Neuron selection — sweep log

Chronological execution log for `research/neuron-selection`.  Synthesis
lives in [`neuron_selection_notes.md`](neuron_selection_notes.md).

## Phase NS-1 — pruning and directed-growth panels

### Pre-registration (written before any confirmatory result exists)

Recorded 2026-11-21, before the harness was run on any dataset other
than the MNIST feasibility probes.  The protocol was reviewed by an
independent model (GPT-6.2 Astra); its must-have changes are
incorporated below.

**Status of the datasets.**

* **MNIST — exploratory.**  Its test set informed the design through the
  probes in `explorations/neuron_selection/`.  MNIST rows are tabulated
  but do not count toward the gate.
* **Fashion-MNIST and CIFAR-10 — confirmatory.**  No test number from
  either dataset has been seen for any method in this line of work.

**Data protocol.**  Flat `div255` vectors.  Per seed, a seeded
permutation of the training set: 10000 validation samples, the rest is
the fit split.  The test set is used for reporting only.  Seeds 0–9.
Every random stream is keyed by (seed, role, generation) and never by
the contrasted factor, so arms are paired by seed and matched arms share
their class draws, centers, and gains.

**Model.**  `NEFLayer`, `abs`, gain $U(0.5, 2)$, data-driven biases from
fit-split samples, one-hot targets.

**Regularization.**  Inside every loop (residuals, rankings, pruning)
the regularizer is the fixed, training-only policy
$\alpha = 10^{-3}$ trace-scaled.  Validation selects $\alpha$ from
$\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ only for the reported decoder
of a finished model and never feeds back.  Both the fixed-$\alpha$ and
the validation-selected metrics are recorded.

**Selection procedure.**  *Batched greedy OBS*: the unit saliency
$s_i = \lVert d_i \rVert^2 / [M^{-1}]_{ii}$ is the exact objective
increase for one deletion at fixed $\lambda$; dropping 5–10 % of the
units per round between recomputations is a greedy approximation, and
nothing here finds the best $k$-subset.

**Pruning panel.**  One 8000-neuron random pool per seed and encoder
family (`hypersphere`, `receptive_field`); nested pools
$P \in \{2000, 4000, 8000\}$; $k \in \{250, 500, 1000, 2000\}$, $k < P$;
criteria `activity`, `magnitude`, `obs_oneshot`, `obs`; plus the
random-width reference curve (leading-$k$ neurons) at widths 250–8000.

**Growth panel.**  Budgets $k \in \{250, 1000\}$, 12 generations, $k$
proposals per generation, prune $2k \to k$.

| Family | Arm | Purpose |
|---|---|---|
| global | `random` | proposal-quality baseline at matched candidates and schedule |
| global | `phd` | the method |
| global | `phd_classshuffle` | residual rows permuted within true class: keeps class-level weights, breaks which image was misfit |
| global | `phd_randkeep` | pHd proposals, uniform random retention from the same $2k$ pool: isolates selection |
| global | `phd_frozen` | pHd bases from generation 0 only: is updating the residual needed? |
| global | `pca` | covariance-shaped proposals: is it just principal-subspace concentration? |
| global | `first_order` | residual-weighted mean directions: tests the "second order is needed" rationale |
| global | `agop` | gradient-outer-product shaping |
| patch | `random` | receptive-field proposals |
| patch | `phd_patch` | pHd on each proposal's patch |

CIFAR-10 runs the first four global arms and both patch arms.

**Primary contrasts** (final generation, family `global`, $k = 1000$,
paired by seed, one-sided paired $t$-tests, Holm-adjusted across the
two confirmatory datasets × three contrasts, family-wise
$\alpha = 0.05$):

1. `width3x`: `phd` at $k = 1000$ minus an unpruned random layer of
   width 3000.  Non-inferiority margin 0: the one-sided 95 % lower
   confidence bound of the paired difference must be $\ge 0$.
2. `proposal`: `phd` minus `random` proposals, $> 0$.
3. `mechanism`: `phd` minus `phd_classshuffle`, $> 0$.

**Gate.**

* *Pass:* contrast 1 and contrast 2 hold on both confirmatory datasets.
  The claim supported is then the narrow one: "at 1000 units, pHd
  proposals with batched greedy OBS selection are at least as accurate
  as a random layer three times wider".  Proceed with the streaming,
  continual, replication, and compression uses of the loop.
* *Partial:* holds on Fashion-MNIST only.  Proceed with the patch
  variant as the primary form and re-plan before the dependent tracks.
* *Fail:* contrast 2 fails on both.  Stop and re-plan.
* Contrast 3 does not gate; it decides how the mechanism may be
  described.  If it fails, the gain must not be attributed to
  per-sample residual information.

**Secondary, descriptive only:** `selection` contrast (`phd` −
`phd_randkeep`); pruning contrast `obs` − `random` at $P = 4000$,
$k = 1000$; $k = 250$ results; patch family; validation-selected
generation; equivalent random width; generation-1 validation-loss
difference between `phd_classshuffle` and `phd`; wall time and peak
memory.

**Not controlled in this phase** (recorded so they are not forgotten):
activity-scale effects under ridge (columns are not RMS-normalized);
center choice for new neurons (random samples, not high-residual ones);
parameter and operation counts alongside unit counts; comparison with a
gradient-trained MLP of equal width.

### Execution

_To be filled at launch: code identity, command, log paths, sentinel._

### Results

_To be filled from the analyzer output._
