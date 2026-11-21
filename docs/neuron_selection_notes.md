# Neuron selection — exact forgetting and directed growth (working notes)

Branch: `research/neuron-selection`.  Status: exploratory.  Nothing in
this file is paper evidence until the corresponding panel in
[`neuron_selection_sweeps.md`](neuron_selection_sweeps.md) has passed its
gate under the multi-seed protocol.

Sibling branches `research/replication` and `research/compression` start
from the `selection-core` tag on this branch and reuse its primitives.

## 1. Why the earlier attempts failed

Two earlier lines of work touched this area and both ended negative:

* **RL recentering** (`NEFFeatures.dead_neuron_indices`,
  `NEFAVRAgent._recenter_dead`): the bottom activity percentile is
  re-biased to recent observations and, under `forget_factor`, the
  corresponding Gram rows and columns are zeroed.
* **CL `growing`** (attic `run_continual.py`, SSUP-II §5.3): neurons are
  added per task; cross-terms with earlier tasks are simply missing.

They fail for three separable reasons.

1. **Wrong utility.**  Mean activity and decoder magnitude are not the
   cost of removing a neuron.  In the pruning probe below they are
   *worse than random* as keep-criteria.  The ridge objective is
   quadratic, so the true cost is available in closed form (§2).
2. **Deletion and addition are not symmetric.**  Deleting a neuron is
   exact and needs no data: the statistics of the survivors are a
   principal submatrix.  Adding a neuron is not: its cross-moments with
   everything already accumulated do not exist.  Zeroing or omitting
   them (both earlier attempts) solves a system that describes no
   dataset.
3. **Undirected proposals.**  Recentering only moved the bias; the
   direction stayed random.  Random candidates are a weak proposal
   distribution; the residual says where capacity is missing (§3).

## 2. Exact deletion

Let $G = A^\top A$, $R = A^\top Y$, $M = G + \lambda I$, $D = M^{-1} R$.

* **Closure.**  For a kept index set $K$, the sufficient statistics of
  the sub-population are $G_{KK}$ and $R_K$.  No data pass.
* **Exact saliency.**  Removing neuron $i$ and refitting the others at
  the same $\lambda$ increases the minimum of
  $J(D) = \lVert Y - AD \rVert_F^2 + \lambda \lVert D \rVert_F^2$ by

  $$s_i = \frac{\lVert d_i \rVert^2}{[M^{-1}]_{ii}} .$$

  This is Optimal Brain Surgeon, exact here because $J$ is quadratic.
  $[M^{-1}]_{ii}$ is large when neuron $i$ is redundant given the
  others, which is why raw magnitude misleads.
* **Exact downdate.**  With $P = M^{-1}$ and drop set $S$,
  $(M_{KK})^{-1} = P_{KK} - P_{KS} P_{SS}^{-1} P_{SK}$ at fixed
  $\lambda$, so a backward-elimination sweep costs one inversion plus
  rank-$|S|$ downdates.
* **Caveat.**  Under the trace-scaled policy $\lambda$ depends on the
  submatrix; the identity is exact for the $\lambda$ in force at each
  step.  `prune_from_stats(method="refit")` re-evaluates the policy
  every step; `method="downdate"` freezes it for the sweep.

## 3. Directed growth

**Propose → accumulate → select.**  One generation at budget $k$:

1. fit, compute residual $r = Y - AD$ on training data;
2. propose $m$ candidate neurons from the residual;
3. accumulate the $(k+m) \times (k+m)$ statistics over the data (this
   is the price of addition — one pass);
4. prune back to $k$ with the exact saliency.

**Proposal distribution (pHd).**  For output $c$,

$$H_c = \sum_x r_c(x)\,(x-\mu)(x-\mu)^\top .$$

Candidate directions are random Gaussian combinations of the
top-$|\lambda|$ eigenvectors of one randomly chosen $H_c$, scaled by
$\sqrt{|\lambda|}$, then normalized.  Rationale: `abs` is even, so it
has no first-order Hermite component and a residual–input correlation
$\sum r\,x$ carries no information about where to put a hinge; the
second-order moment is the leading directional statistic (Li's
principal Hessian directions).  $H_c$ is additive over samples, so it
is a sufficient statistic in the same sense as $G$ and $R$ and can be
accumulated in a stream.

**What the probes say about each ingredient** (§4): selection alone
gives roughly $2\times$ neuron efficiency; an AGOP-reshaped frame alone
gives $+0.3$ to $+1.0$ pp and adds nothing once selection is on; pHd
proposals with selection give $5$–$7\times$.

## 4. Probe evidence (NOT paper evidence)

Throwaway scripts: `explorations/neuron_selection/probe{1,2,3}.py`.
MNIST, 50k train / 10k validation / 10k test, **one seed**, `abs`,
hypersphere encoders, gain $U(0.5, 2)$, data-driven biases,
trace-scaled $\alpha \in \{10^{-5},\dots,10^{-2}\}$ picked on
validation, test accuracy reported.  Test curves were inspected across
generations, so treat every number as optimistic.

| Setting | Test accuracy (%) |
|---|---|
| Random layer, width 250 / 500 / 1000 / 2000 / 4000 | 89.1 / 92.2 / 94.3 / 95.9 / 96.8 |
| Prune 4000 → 500: exact OBS / random / high-activity / magnitude | **94.4** / 92.5 / 90.8 / 90.6 |
| Prune 4000 → 250: exact OBS / random / high-activity / magnitude | **92.6** / 89.5 / 87.3 / 87.3 |
| Prune 8000 → 1000 / 500 (one shot) | 95.9 / 94.9 |
| Generational, random proposals, 12 generations, $k = 500$ / $1000$ | 94.8 / 96.2 |
| Generational, AGOP proposals, 12 generations, $k = 500$ / $1000$ | 94.7 / 96.3 |
| AGOP-reshaped frame, no selection, width 1000 / 4000 (best of 3 iterations) | 95.3 / 97.3 |
| **Generational, pHd proposals, 10 generations, $k = 250$ / $1000$** | **95.4 / 97.45** |
| Generational, half pHd + half random, $k = 250$ / $1000$ | 95.0 / 97.1 |
| Sparse ternary encoders, unit gain, width 4000, nnz 784 / 64 / 24 / 8 / 4 | 96.9 / 96.8 / 97.0 / 96.6 / 96.4 |
| Student regressing CNN-teacher logits (teacher 98.7), width 4000: one-hot / logits / logits + 12 shifted copies | 96.8 / 96.6 / 96.4 |

Readings:

* A 250-neuron pHd-selected layer matches a random layer of
  ~1500–2000; a 1000-neuron one exceeds a random 4000.
* pHd-only beats the pHd/random mix: random candidates are not needed
  as an exploration floor at this scale.
* Sparse ternary encoders with unit gain are free.  With `abs` and a
  center-derived bias, $|g\,z| = g\,|z|$: gain is a pure column scaling
  and only acts through the ridge metric.  Inference then needs integer
  adds and one absolute value per neuron.
* Regressing teacher logits does not help a fixed-feature student, and
  teacher-labelled augmentation hurts: the student is feature-limited,
  not target-limited.  Replication therefore depends on directed
  proposals, not on better targets.

## 5. Open questions the panels must answer

1. Does the pHd gain survive 10 seeds, Fashion-MNIST, and CIFAR-10?
   Global second-moment eigenvectors are dense; on CIFAR-10 the
   receptive-field prior is worth $+5.4$ pp, so a local variant
   (pHd on patches, or sparse-pHd) is the likely necessary form.
2. Overfitting: proposals are fitted to training residuals.  Track the
   train–validation gap per generation and include a label-shuffle
   control.
3. Is it the proposal frame or the selection?  Arms: pHd frame without
   selection; selection with random proposals; both.
4. Where does the curve plateau, and how does the plateau scale with
   $k$?  Compare with the tuned MLP grid at equal width and at equal
   inference operation count.
5. Does the loop work with an accumulated $H_c$ in a stream (EMA), with
   newborn neurons kept in a shadow cohort until their statistics are
   mature?  This is the plasticity mechanism.
6. Continual learning: deletion preserves the joint-fit identity
   exactly.  Growth needs past cross-moments; candidate fix is "moment
   replay" from additive per-class input moments $(n_c, \mu_c,
   \Sigma_c)$.  Approximate — forgetting must be measured.

## 6. Issues noticed along the way

* `rl.py::_recenter_dead`: activity criterion, bias-only change, zeroed
  Gram rows without age correction.  A likely cause of the null
  `min_relative` sweep.  Revisit only after the streaming panel.
* SSUP-II §5.3 and its negative-results appendix attribute the failure
  of `growing` to sensitivity to relative neuron counts and conclude
  that the encoder must never change mid-stream.  The mechanism is
  missing cross-moments; deletion is exact.  Reword once the continual
  panel has run.
* The gain-as-column-scaling observation above may deserve a sentence
  in SSUP-I's bias section if it is not already there.

## 7. Prior art to position against

To be filled from the literature pass: Optimal Brain Surgeon and its
layer-wise exact forms; OP-ELM, I-ELM, EM-ELM; orthogonal least squares
for RBF centers; cascade-correlation; projection pursuit; SIR and pHd;
recursive feature machines (AGOP); one-gradient-step feature learning;
ReLU model extraction; ternary random projections.
