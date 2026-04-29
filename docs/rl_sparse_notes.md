# Sparse-reward NEF-RL — exploration notes

Companion to `docs/rl_clean_sweeps.md`. This document records the
sparse-reward exploration that grew out of the rl_clean program: what
question we were after, which mechanisms we tried, what worked,
what failed and why, and what the residual open questions are.

These notes are intended to be the canonical record. Procedural
shortcuts taken while co-running with the rl_clean P3b/P8 sweeps
mean the *timings* in this work are not directly comparable to the
clean sweep numbers, but the *qualitative claims* below are
multi-seed and reproducible from the JSONs in `results/rl_sparse/`
plus the toy regression in `results/rl_sparse/sparse_chain.json`.

## 1. Framing

### 1.1 What "sparse reward" breaks for NEF-FQI

Plain NEF-FQI (`src/leenef/rl.py`, `NEFFQIAgent`) is faithful to the
core leenef program: fixed encoders, per-action analytical decoder
solve `D_a = (Φ_aᵀΦ_a + αI)⁻¹ Φ_aᵀY_a`, RLS-decayed sufficient
statistics, n-step returns to keep `Y` model-independent. On dense
control tasks (CartPole, LunarLander) it is competitive with
gradient-trained baselines while running on CPU.

Sparse reward kills this not at the function-approximation step but
at the *target-signal* step. With reward concentrated on a single
terminal transition and γ<1 propagating it through long horizons:

- `AᵀY ≈ 0` until the goal is reached. The analytical solve
  faithfully reports "every state has roughly the same value" —
  which is true given the data, but useless for control.
- TD bootstrap could push value backward from the goal, but our
  closed-form solve fits *exactly* to whatever bootstrap targets it
  receives, amplifying noise. This is the same instability that
  drove the rl_clean program toward Monte-Carlo n-step targets in
  the first place (`docs/rl_notes.md` §6, §8.8).

So the core question for this work was:

> How do we put non-trivial structure into `Y` for sparse-reward
> tasks, using only the primitives that already define leenef
> (`AᵀA`, `AᵀY`, fixed encoders, additive sufficient statistics,
> optional pooling)?

### 1.2 The leenef essence

Distilled from `streaming.py`, `layers.py` (`partial_fit` /
`solve_accumulated` / `continuous_fit`), the CL accumulator results,
and the working pieces of `rl.py`:

1. Encoders are **fixed** (random or data-adapted).
2. Decoders fall out of **one** analytic step.
3. `AᵀA` and `AᵀY` are the universe; everything is an additive
   combination of sufficient statistics.
4. `Y` is the only place where task structure enters.
5. Time is collapsed by **pooling**, not recurrence (streaming
   classifier).
6. **Per-class additivity** is what makes CL work: each output
   column owns its own decoder, so accumulating new task statistics
   never interferes with old columns.

Anything bolted onto NEF-RL needed to live within (1)–(6). No
gradient-trained auxiliary networks, no target-network schedules,
no learning rate.

## 2. The mechanism stack

Three mechanisms, each a re-use of existing leenef machinery:

### D1 — NEF Successor Features (`NEFSuccessor`, `NEFSFAgent`)

`src/leenef/rl_sf.py`. The same fixed encoder φ. A second analytical
solve produces a successor-feature decoder

```
ψ_a(s) ≈ E[ Σ_k γ^k φ(s_{t+k}) | s, a ]
```

where each *output column* is one encoder feature. The reward map
is a separate `n_neurons`-vector regression `w_a = (Φ_aᵀΦ_a + αI)⁻¹
Φ_aᵀ r`. Action value is then the bilinear form

```
Q(s, a) = ψ(s, a) · w_a
```

Both ψ and w_a use the same `(Φ_aᵀΦ_a + αI)⁻¹` cached by the FQI
solve. The targets for ψ are Monte-Carlo n-step successor sums on φ
— the direct analog of n-step on r — which keeps us in the
non-bootstrapped regime that already worked for the rl_clean line.

ψ is **reward-free**: it captures connectivity/reachability of state
space. When the sparse reward finally fires, only the small w_a
solve has to refit; ψ is already correct everywhere.

### D2 — Leverage novelty bonus

The Gram inverse `(Φ_aᵀΦ_a + αI)⁻¹` is already built. The
Mahalanobis leverage

```
novelty(s, a) = 1 / sqrt( φ(s)ᵀ (Φ_aᵀΦ_a + αI)⁻¹ φ(s) )
```

gives us a free RND-style novelty signal: large in
under-represented regions, small where we have lots of samples.

We add `β · novelty` to the n-step return *inside the target* so
intrinsic reward enters `AᵀY` at the sufficient-statistic level
(no replay rewrite). One scalar (`novelty_beta`) controls scale; no
auxiliary network, no second optimizer. Implemented in `NEFSFAgent`.

### D4 — Hindsight goal conditioning + `eval_goal` projection

`NEFHindsightSFAgent` (subclass of `NEFSFAgent`). Encoder input is
expanded to `(state, goal)`. Goals are sampled from a HER-style
buffer of recently visited states. The same Ψ machinery from D1
fits one decoder over the joint `(state, goal)` space.

The decisive — and small — addition is the **`eval_goal`
projection**. At evaluation the agent is given the *actual task
goal* `g*`, and we score actions by

```
Q(s, a; g*) = ψ(s, a) · u(φ([s; g*])),  u(x) = x / ‖x‖
```

This is a one-line eval-time change. It works because the SF
decomposition factorizes Q into a backbone (ψ) and a goal direction
(u). When training only ever saw HER-relabeled goals, ψ has still
learned the *connectivity* of state space; projecting onto the
direction of the true goal is enough to read out a useful policy
even though the extrinsic reward never propagated very far on its
own.

No extra solve. No reward shaping. No gradient.

## 3. Headline results

All numbers below are from multi-seed runs on three seeds {0, 1, 2}
under the rl_clean seed protocol (warmup/training/eval policy fixed
in `benchmarks/run_rl_sparse.py`). Forget factor and n_neurons noted
per row; the rl_clean base config drifted during this work, so
sparse-reward runs used a smaller `n_neurons=2000` to keep the
per-action Ψ cost tractable.

### 3.1 MountainCar-v0

| Variant | best_eval (3 seeds) | robust_eval | final_eval |
|---|---|---|---|
| FQI baseline (n=6000) | −200, −200, −200 | −200, −200, −200 | −200, −200, −200 |
| SF (n=2000) | −200, −200, −200 | −200, −200, −200 | −200, −200, −200 |
| SF + novelty β=0.3 | −200, −200, −200 | −200, −200, −200 | −200, −200, −200 |
| Hindsight (commit-mode, no eval_goal) | −200, −200, −200 | −200, −200, −200 | −200, −200, −200 |
| **Hindsight + eval_goal=(0.5, 0.0)** | **−132.6, −154.0, −138.4** | **−159.1, −158.4, −156.7** | **−137.0, −146.2, −153.0** |

`results/rl_sparse/mc_hindsight_evalgoal.json` is the headline
sparse-reward result. The HER buffer + Ψ trained against relabeled
goals; at eval, the policy is read out by projecting ψ onto the
direction of φ([0.5, 0.0]) — the at-rest top of the right hill.
3/3 seeds solve the task.

The intermediate negative results are themselves informative:

- FQI, SF, SF+novelty all stuck at −200 because under default ε
  the random walk simply does not reach the goal in 200 steps.
  Novelty modifies *targets*, not *behavior*; on MC's exploration
  bottleneck it cannot help on its own.
- Hindsight without `eval_goal` is *also* stuck — diagnostics
  showed termination_rate ≈10% in the final window, i.e. the agent
  *was* reaching the goal during training, but the extrinsic w_a
  alone could not propagate the signal under
  hindsight-driven exploration.

The eval_goal projection is what closes the loop.

### 3.2 Acrobot-v1

| Variant | best_eval | robust_eval | final_eval | comment |
|---|---|---|---|---|
| FQI baseline (n=6000) | −301, −354, −442 | −354, −500, −481 | −293, −500, −500 | seed 0 partial, others stuck |
| SF / SF+novelty (n=2000) | all ≥ −267 | all −500 | all −500 | exploration barrier |
| Hindsight + eval_goal, ff=0.999 | −253, −156, −217 | −392, −370, −484 | −286, −303, −500 | reaches goal; **regresses** |
| **Hindsight + eval_goal, ff=0.9999** | **−122.8, −106.2, −124.8** | **−178.5, −186.1, −166.9** | **−164.4, −116.4, −203.2** | **3/3 stable** |

Acrobot needs longer credit horizons than MC. With ff=0.999 the SF
backbone fades faster than the credit window; all three seeds find
the goal, but ψ then forgets it and the policy regresses toward
−500 by ep 300. With ff=0.9999 the slower-decaying ψ retains
successor structure across the long horizon and the same hindsight
+ eval_goal stack delivers a stable solve.

`forget_factor` is the principled stability lever here, not an
exploration knob. This is consistent with the rl_clean finding that
a slower forget factor correlates with stability on long-credit
tasks; sparse reward sharpens that dependency.

### 3.3 LunarLander-v3 (dense regression)

| Variant | best_eval | comment |
|---|---|---|
| FQI (n=2000) | −19.6, −50.1, −142.1 | matches dense baseline at scaled n |
| Hindsight (n=2000) | −306.3, −305.7, −236.9 | materially worse |

This is the expected and welcome failure: hindsight + eval_goal is
a sparse-reward tool layered on Ψ. It should *not* be applied to
dense tasks. The HER buffer pollutes Ψ with goal-relabeled
trajectories that a dense reward would not have produced, and the
eval-time projection onto u(g*) discards information that the
extrinsic reward already supplies.

This regression is documented so future automation does not pick
hindsight as a default; on dense reward, plain FQI or plain SF
remains the right tool.

### 3.4 MiniGrid-DoorKey-5x5

| Variant | best_eval | seeds | comment |
|---|---|---|---|
| Hindsight + random_direction goal | 0.0 | 3/3 stuck | clean negative |

`MiniGrid-DoorKey-5x5-v0` exposes only an image observation. We
wrap it (`_MiniGridFlatNorm` in `benchmarks/run_rl_sparse.py`) to a
flattened normalized float32 vector so the Box-input agents work
unchanged. Without a proper task-goal handle (i.e. without an
observation captured *at* the goal cell), the eval_goal projection
has nothing to project onto, and the 7-action discrete space
defeats hindsight-only exploration on its own.

This is a clean negative result: the missing piece is a programmatic
way to obtain an "observation at the goal cell" on image envs. With
that handle in place the same D1+D4+eval_goal stack should apply.
See §6.

## 4. The toy sparse-chain benchmark

`benchmarks/run_sparse_chain.py` runs the three variants
(`nef_fqi`, `nef_sf`, `nef_sf_novelty`) on a deterministic K-state
chain MDP (`leenef.rl_diag.SparseChainEnv`) where the closed-form
optimal Q is known via value iteration (`SparseChainEnv.optimal_q`).
The pre-existing
`tests/test_rl_sf.py::test_successor_solve_recovers_chain_optimal_q`
unit test already exercises the SF *solver* against Q*; this driver
exercises the *full agent* over training, plus a derived metric for
the eval_goal projection.

Sweep (3 seeds × 3 sizes × 200 episodes), summary numbers from
`results/rl_sparse/sparse_chain.json`:

| Variant | K | goal_reached | policy_correct | first_success (mean) | q_max_abs_err |
|---|---|---|---|---|---|
| nef_fqi | 5  | 3/3 | 3/3 | 0.3 | 0.33 |
| nef_fqi | 10 | 3/3 | 3/3 | 4.0 | 0.40 |
| nef_fqi | 20 | 2/3 | 0/3 | 6.5 | 0.65 |
| nef_sf  | 5  | 3/3 | 3/3 | 0.3 | 0.81 |
| nef_sf  | 10 | 3/3 | 3/3 | 4.7 | 0.81 |
| nef_sf  | 20 | 2/3 | 2/3 | 6.5 | 0.87 |
| **nef_sf_novelty** | 5  | 3/3 | 3/3 | 0.3 | 0.66 |
| **nef_sf_novelty** | 10 | 3/3 | 3/3 | 4.7 | 0.67 |
| **nef_sf_novelty** | 20 | **3/3** | **3/3** | 18.7 | 0.71 |

The eval_goal projection failed (0/9) for the chain — see §5 for
why; that failure is a *finding*, not a regression.

Two observations:

1. **D2 (novelty) clearly rescues exploration at K=20.** Plain SF
   reaches the goal in 2/3 seeds and produces a correct ranking
   policy in 2/3; SF + novelty reaches the goal in 3/3 and gives a
   correct policy in 3/3. The mean first_success rises (18.7 vs
   6.5) because the metric averages only over successful seeds; the
   *fraction* that ever finds the goal is the headline.
2. **Plain NEF-FQI fails K=20 policy** despite reaching the goal
   in 2/3 seeds. With γ=0.9 and 19 hops, γ^19 ≈ 0.135; the
   backed-up reward signal is too weak relative to the ε-greedy
   data distribution for the closed-form solve to recover the
   correct ranking. This is exactly the failure mode SF was
   designed to address — and on chains where eval_goal does not
   apply (see below), SF + novelty is the variant that survives.

The reward Q error stays elevated for SF (~0.81 vs the Q* range
[0, 1]). This is a known-direction artifact: the per-action reward
weight `w_a` is fit from terminal transitions only, so it is biased
positive, and the product ψ·w is amplified at goal-adjacent states.
The *ranking* across actions is still correct in 3/3 SF+novelty
seeds at K=20, which is what matters for control. The toy is
useful as a regression test for the *policy*; treating
`q_max_abs_err` as a tightness check requires fitting w_a from a
larger reward-mixing buffer than this toy provides.

## 5. Where eval_goal does not apply: the chain artifact

On the toy chain, projecting ψ onto u(φ([1.0])) (the goal at the
right end) gave 0/9 policy_correct. This is a *meaningful* result,
not a regression, and it bounds the regime of the eval_goal
mechanism.

Why the projection fails on the chain:

- From state s = K−2, action RIGHT terminates immediately. The
  successor representation ψ(s, RIGHT) is essentially a single
  φ(K−1) term (the terminal feature, before discounting cuts off).
- Action LEFT keeps the episode alive for many steps. ψ(s, LEFT) is
  a many-term sum Σ_k γ^k φ(s_k) over a long trajectory.
- The two ψ vectors therefore differ by 5–10× in magnitude, and
  the magnitude difference dominates the inner product with u(g*).
  Direction information is buried.

This does *not* happen on MC or Acrobot because both actions in
those envs produce trajectories of comparable length until goal
attainment — so magnitude differences in ψ reflect direction
differences, not horizon differences. The eval_goal projection
implicitly assumes that the action horizons being compared are
comparable; when they are not (e.g. a "stop here" action in an
otherwise-running rollout), the projection loses sign.

Implication: the eval_goal mechanism is **not universal**. It
applies cleanly to control tasks with bounded action effects on
horizon; it does not apply to MDPs with one-shot terminating
actions of very different horizon than the alternatives. Future
extensions could (a) normalize ψ by `‖ψ‖` before projection — a
one-line change, but it bakes a strong prior into the readout; or
(b) split ψ into "trajectory" and "terminal" components and project
each separately. We did *not* implement these; the chain is on the
record as the place where we noticed the limit.

## 6. What is *not* in this work and why

These were considered and consciously deferred:

- **Bootstrapped SF targets.** Rejected on the same grounds that
  bootstrapped Q targets are rejected for analytical solves: the
  closed-form fit amplifies bootstrap noise. We use Monte-Carlo
  n-step on φ throughout.
- **Per-task multi-head decoders (D5).** Effectively a discretized
  D4. D4 with `eval_goal` already gives us what D5 would, with one
  ψ matrix instead of one decoder column per task.
- **Trajectory-level pooling (D3).** The streaming-classifier
  transplant for RL — pool a window of φ(s) and regress trajectory
  return — is an attractive direction but is not a drop-in
  replacement for the mechanism that closes credit assignment;
  selection of point-wise actions from a trajectory-level Q is
  awkward. Left for future work.
- **Reward shaping.** Intrinsic novelty (D2) is tagged separately
  and ablated; never folded into the reported reward.
- **Knob sweeps over broadness, min_relative, recentering** as a
  sparse-reward fix. Per `docs/rl_review.md` §2.5, the bottleneck
  is mechanistic. Confirmed empirically: changing `n_neurons`
  6000→2000 did not move plain FQI/SF off the wall on MC, and the
  hindsight + eval_goal stack worked at n=2000.
- **Programmatic goal-obs for image envs.** The MiniGrid path
  needs an "observation at the goal" hook so eval_goal has a
  direction to project onto. Identified as the next concrete
  follow-up.
- **Single-seed claims.** Everything in §3 is multi-seed.

## 7. Practical defaults

For someone reaching for the sparse-reward stack today:

| Knob | Value | Rationale |
|---|---|---|
| Agent | `nef_sf_hindsight` (`benchmarks/run_rl_sparse.py`) | D1+D4 stack |
| `n_neurons` | 2000 | per-action ψ memory tractable; n=6000 hits OOM at typical ψ sizes |
| `forget_factor` | 0.9999 on long-horizon tasks (Acrobot), 0.999 on short (MC) | slower decay retains ψ across the credit window |
| `eval_goal` | task-specific state vector at the desired goal | MC: `[0.5, 0.0]`; Acrobot: `[-1, 0, 1, 0, 0, 0]` |
| `novelty_beta` | 0 by default; raise to 0.3 only if exploration is the bottleneck | D2 is *additive* on top of D1+D4; on MC/Acrobot D4 alone suffices |
| `goal_buffer_size` / `goal_horizon` | hindsight-typical (cf. driver defaults) | larger buffers help on slower envs |

When the env is dense, drop hindsight: §3.3 documents the
regression. When the env's actions have wildly different horizons
(see §5), do not rely on eval_goal projection; fall back to
training-time goal injection or to D2-augmented exploration.

## 8. Open questions / follow-ups

1. **Programmatic goal-obs capture for image envs.** The MiniGrid
   negative result is solvable: place the agent at the goal cell
   (or use the env's `mission`/grid representation) to construct an
   observation vector usable as `eval_goal`. Once that handle is in
   place, rerun MiniGrid-DoorKey-5x5 with the same D1+D4+eval_goal
   stack. Stretch: MiniGrid-FourRooms.
2. **Generalization study of eval_goal.** Sweep eval_goal *near*
   but not *at* the optimal terminal state: how robust is the
   projection? On MC, what happens if we eval against
   `[0.45, 0.0]` or `[0.5, 0.07]`? We expect a smooth degradation;
   measuring it would map the basin of the mechanism.
3. **Ψ stability vs forget_factor scan.** Acrobot showed a
   qualitative jump from ff=0.999 to ff=0.9999. A small scan
   (0.999, 0.9995, 0.9999, 0.99995) would locate the boundary and
   tell us whether to expose forget_factor as a per-task default
   in the driver.
4. **Magnitude-normalized ψ projection** (§5). One-line change:
   `Q(s, a; g*) = (ψ(s, a) / ‖ψ(s, a)‖) · u(g*)`. Tradeoff: bakes
   a strong prior. Worth a chain + MC + Acrobot ablation; if it
   does not hurt MC/Acrobot, it could be the default and would
   buy us applicability on toy chains and on terminating-action
   envs.
5. **Coupling to rl_clean base-config drift.** The sparse-reward
   stack inherits its base config from rl_clean. Each rl_clean
   adoption (notably forget_factor and gamma) should trigger a
   re-validation of MC/Acrobot under the new base. The toy
   sparse-chain regression test makes this cheap to check.
6. **Toy sparse-chain as a regression test.** `sparse_chain.json`
   is the cheapest self-contained signal that mechanism changes do
   not break SF or D2. Cost is on the order of a minute total;
   adding it to a CI smoke would catch sign errors and off-by-one
   bugs in n-step successor targets that are otherwise invisible
   inside seed noise on real benchmarks.

## 9. File map

| Artifact | Role |
|---|---|
| `src/leenef/rl.py` | `NEFFQIAgent` (D0 baseline) |
| `src/leenef/rl_sf.py` | `NEFSuccessor`, `NEFSFAgent`, `NEFHindsightSFAgent` (D1+D2+D4) |
| `src/leenef/rl_diag.py` | `SparseChainEnv`, `optimal_q()`, leverage diagnostics |
| `benchmarks/run_rl_sparse.py` | Multi-env driver; `nef_fqi` / `nef_sf` / `nef_sf_hindsight` variants; `--eval-goal`; MiniGrid wrapper |
| `benchmarks/run_sparse_chain.py` | Toy chain regression driver — Q* + policy + eval_goal grading |
| `tests/test_rl_sf.py` | `test_successor_solve_recovers_chain_optimal_q` and the eval_goal/extrinsic invariants |
| `results/rl_sparse/*.json` | Multi-seed run dumps (one per condition) |
| `results/rl_sparse/sparse_chain.json` | Toy regression sweep |

## 10. One-paragraph summary

For sparse-reward control, the leenef-faithful stack is **D1**
(NEFSuccessor: a second analytical solve on the same encoder, with
Monte-Carlo n-step targets on φ instead of r) + **D4** (hindsight
goal-conditioning on `(state, goal)`) + the **`eval_goal`
projection** (read out actions at evaluation by projecting ψ onto
the direction of φ at the true task goal). Plain D1 and plain D2
do not break MC's exploration wall on their own; the decisive
mechanism is D4 with eval_goal, which costs one extra encoder
input dimension and one normalized inner product at eval time.
Acrobot needs `forget_factor=0.9999` for the same stack to be
stable. The mechanism does **not** transfer to MDPs whose actions
have wildly different time horizons (toy chain in §5), and it
should **not** be applied to dense reward (LunarLander regression
in §3.3). Multi-seed evidence is in `results/rl_sparse/`; the toy
regression is in `results/rl_sparse/sparse_chain.json` via
`benchmarks/run_sparse_chain.py`.
