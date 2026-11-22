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
* **Saliency for one deletion.**  Removing neuron $i$ and refitting the
  others at the same $\lambda$ increases the minimum of
  $J(D) = \lVert Y - AD \rVert_F^2 + \lambda \lVert D \rVert_F^2$ by

  $$s_i = \frac{\lVert d_i \rVert^2}{[M^{-1}]_{ii}} .$$

  This is unit-level Optimal Brain Surgeon, exact for a single deletion
  because $J$ is quadratic.  $[M^{-1}]_{ii}$ is large when neuron $i$ is
  redundant given the others, which is why raw magnitude misleads.
  Dropping several units per round between recomputations is a greedy
  approximation (*batched greedy OBS*); it does not find the best
  $k$-subset.
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
$\sqrt{|\lambda|}$, then normalized.  Working rationale: the `abs`
feature has no first-order Hermite component, so the second-order
residual moment is the natural leading directional statistic (Li's
principal Hessian directions; information exponent 2).  This rationale
is not established: a neuron $|e \cdot (x - c_i)|$ is even about its own
center, not about the data mean, so a residual-weighted mean can still
identify a useful axis, and an unwhitened residual second moment is not
a Hessian in general.  The `first_order` arm of the growth panel tests
the rationale instead of assuming it.  $H_c$ is additive over samples,
so it is a sufficient statistic in the same sense as $G$ and $R$ and can
be accumulated in a stream.

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

From a delegated literature pass (2026-11-21).  The references were
checked online by the search agent but have not been independently
re-checked; verify each one before it is cited in a manuscript.

**What is already known.**

* *Unit-level OBS.*  Hassibi & Stork (1993) give OBS; Stahlberger &
  Riedmiller (NeurIPS 1996, "Unit-OBS") remove whole hidden units by
  grouping their outgoing weights.  Our saliency is that criterion,
  exact for a ridge readout, and equivalent to one step of backward
  stepwise regression.  Layer-wise OBS (Dong et al., 2017) and
  OBC / SparseGPT (Frantar & Alistarh, 2022–2023) prune weights, not
  fixed-feature units.  **The selection step is not a novelty claim.**
* *Grow/prune ELMs.*  OP-ELM (Miche et al., 2010) ranks a random pool
  by LARS/MRSR with leave-one-out sizing; CS-ELM (Lan et al., 2010)
  forward-selects from a random reservoir; DCI-ELM (Sun et al., 2017)
  combines incremental addition with removal of low-output-weight
  units; OLS for RBF centers (Chen, Cowan & Grant, 1991).  All draw
  candidates at random or from data points; none proposes spectral
  directions from the residual.
* *Residual-directed constructive units.*  Cascade-correlation
  (Fahlman & Lebiere, 1990) and projection pursuit regression
  (Friedman & Stuetzle, 1981) fit candidate units to the residual by
  iterative optimization; DA-ELM (Feng et al., 2015) adjusts
  insignificant units along a residual-decreasing direction.
* *Second-order statistics for directions.*  pHd (Li, 1992), SIR
  (Li, 1991); tensor-moment recovery of two-layer networks (Janzamin
  et al., 2015).  Information-exponent theory (Ben Arous et al., 2021)
  explains why an even activation needs second-order information.
* *Gradient outer products.*  Recursive Feature Machines
  (Radhakrishnan et al., 2022); one-gradient-step feature learning
  (Ba et al., 2022).
* *Data-dependent random features.*  Sinha & Duchi (2016); Rudi &
  Rosasco (2017); leverage-score sampling (Li et al., 2019).  These
  reweight within a kernel family.
* *Ternary projections and integer ELMs.*  Achlioptas (2003); Li,
  Hastie & Church (2006); an integer-only ELM with ternary input
  weights and integerized output weights (Machado et al., arXiv 2026,
  ReLU).  **Multiplier-free inference is an engineering payoff, not a
  novelty claim.**
* *Hinge location by queries.*  Rolnick & Kording (2020); Jagielski
  et al. (2020); shallow mimic students (Ba & Caruana, 2014).
* *Adaptive predictive coding.*  MPEG-4 ALS RLS–LMS cascade (Huang
  et al., 2008); DeepZip; NNCP; Vovk's online ridge bound (1997).
* *Neuron turnover.*  ReDo (Sokar et al., 2023) recycles dormant
  units by activation; continual backprop (Dohare et al., 2024) by a
  contribution utility; neither uses the exact refit loss increase.

**What the pass did not find** (absence of evidence only):

1. residual pHd eigenvectors used to propose hidden-unit directions for
   fixed-feature or `abs` networks;
2. the propose–accumulate–select loop with exact OBS selection from
   normal-equation statistics;
3. a fixed-feature predictor with an analytic readout used as a
   lossless compressor;
4. an exact-utility criterion for neuron turnover.

Positioning that follows: the contribution to test is the **proposal
distribution and the closed loop**, with OBS as the known exact
selector; the comparison arms must therefore include OP-ELM-style
ranking and random-proposal grow/prune, which the growth panel already
has in the form of the random-proposal arm.

## NS-1 interpretation and the one-shot probe (2026-11-22, exploratory)

Reading the archived NS-1 tables beyond the registered gate (secondary
arms are descriptive, not preregistered):

- **Selection is necessary; pHd proposals alone are not better
  neurons.** `phd_randkeep` (pHd proposals, uniform retention) ends
  near the random-proposal arm on Fashion-MNIST and CIFAR-10. The gain
  comes from a heavier tail of useful candidates that OBS picks out.
- **Iteration matters little.** `phd_frozen` (bases from the
  generation-0 residual only) is within 0.2 pp of the refreshed arm on
  Fashion-MNIST and equal on MNIST.
- **What the residual adds is dataset-dependent.** On MNIST unsupervised
  PCA directions and within-class-shuffled residuals recover most of the
  gain; on Fashion-MNIST PCA is no better than random and sample-level
  residual pairing carries most of it; CIFAR-10 sits between.
- **Spatial locality dominates on CIFAR-10.** Random receptive-field
  proposals + OBS (53.5 %) beat global pHd (50.2 %); patch pHd adds
  only about 0.5 pp (54.0 %). The global-family equivalent widths
  (about 6000–8000 at k = 1000) are therefore relative to hypersphere
  baselines only.

`explorations/neuron_selection/probe4_oneshot.py` then asked whether
the 12-generation loop is needed at all. Validation accuracy only (test
metrics computed by the shared helper were discarded), hypersphere
family, k = 1000, seeds 0–2, 8 threads; time is the whole trial.

| Pool → 1000, one OBS pass | MNIST val % | Fashion val % | time |
|---|---:|---:|---:|
| random 4000 | 95.46 | 87.24 | 9 s |
| random 13000 | 95.81 | 87.64 | 89 s |
| label-pHd 4000 (no model; class-moment contrasts) | 96.80 | 87.49 | 12 s |
| label-pHd 13000 | 97.20 | 87.90 | 91 s |
| residual-pHd 4000 | 97.14 | 88.45 | 12 s |
| residual-pHd 13000 | 97.34 | 88.88 | 92 s |
| residual-pHd 4000, three rounds | 97.35 | 88.99 | 35 s |
| NS-1 loop, 12 generations (same seeds, val) | 97.34 | 89.09 | 170–380 s (6 threads) |

Single seed, k = 5000, two rounds from a 10000 pool (about 2 min):
MNIST 98.23 % val (random-proposal control 97.43 %), Fashion-MNIST
90.12 % val (control 89.54 %).

Working conclusions, all requiring confirmation:

1. The method reduces to **sample a supervised candidate pool once (or
   2–3 times), prune once with OBS**: seconds, not minutes, and one or
   two extra data passes.
2. Model-free label-pHd uses only additive per-class input moments
   `(n_c, mu_c, Sigma_c)`. It works on MNIST but adds almost nothing
   over random on Fashion-MNIST, so continual/streaming variants
   cannot rely on it alone; they need residual moments.
3. At k = 5000 the validation numbers are in the range of the tuned
   MLP test accuracies in SSUP-I (98.10 % / 89.39 %), but the splits,
   training-set sizes and metric differ; an equal-protocol comparison
   is the next confirmatory target.
