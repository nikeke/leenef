# Sparse-reward NEF-RL — exploration notes

Companion to `docs/SSUP-I.md` (single-layer / streaming supervised
learning) and `docs/SSUP-II.md` (continual learning by accumulation).
These two papers fix the leenef program; the present document is the
record of carrying that same program into sparse-reward RL, with an
eye toward a future SSUP-III.

The supervised and CL papers established a vocabulary — six numbered
principles **P1–P6** — that we will reuse below. P1 (fixed encoders),
P2 (one analytic decoder solve), P3 ($A^\top A$ and $A^\top Y$ as the
complete sufficient state), P4 ($Y$ is where the task enters), P5
(time collapsed by pooling, not by recurrence), and P6 (per-class
additivity for class-incremental CL). The mechanisms below are all
expressions of P1–P4 in an RL context; P5 and P6 have natural RL
analogs but were not the bottleneck for the experiments reported
here.

**Context note (rebase, May 2026).** When this work began, RL had a
parallel `rl_clean` program with its own base configuration and
sweep notes. During the SSUP-I/II finalization, that program was
consolidated and moved to `attic/docs/rl_clean_*.md` and
`attic/results/rl_clean/` as part of the clean restart that
restructured `benchmarks/`, `docs/`, `results/`, and `notebooks/`
into the `sl_clean / cl_clean / rl_clean` layout. The RL **agent**
(`src/leenef/rl.py`, `NEFAVRAgent`) was not changed semantically; only
its alpha-regularization plumbing was centralized through the new
`solvers.regularized_normal_matrix` + `solvers.solve_spd` helpers,
selecting the trace-scaled policy `alpha=(α, "trace")` that
NEF-AVR was already using inline. The `rl_clean` numbers from
`attic/docs/rl_clean_sweeps.md` therefore still describe the same
agent; they are *historical context*, not parallel work to be
matched.

These notes are intended to be the canonical record for the
sparse-reward line going forward. Reproducible from the JSONs in
`results/rl_sparse/` plus the toy regression in
`results/rl_sparse/sparse_chain.json`. Procedural caveats from
exploration (small-n network, co-running CPU) are flagged in line.

## 1. Framing

### 1.1 What "sparse reward" breaks for NEF-AVR

Plain NEF-AVR (`src/leenef/rl.py`, `NEFAVRAgent`) is faithful to the
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
  the first place (`attic/docs/rl_notes.md` §6, §8.8).

So the core question for this work was:

> How do we put non-trivial structure into `Y` for sparse-reward
> tasks, using only the primitives that already define leenef
> (`AᵀA`, `AᵀY`, fixed encoders, additive sufficient statistics,
> optional pooling)?

### 1.2 The six principles, read in RL

SSUP-I §1 and SSUP-II §1 define the six principles. Each has a direct
sparse-reward reading; the mechanism stack in §2 is just the
disciplined application of P1–P4.

- **P1 — Encoders are fixed.** In RL, the encoder φ(s, a) is
  selected once and never adapts to the reward signal. Sparse
  reward cannot break φ; it can only fail to write into the
  decoder.
- **P2 — One analytic decoder solve.** Per-action FQI fits
  $D_a = (\Phi_a^\top \Phi_a + \alpha I)^{-1} \Phi_a^\top Y_a$ in
  one step. There is no learning rate to tune away the
  pathological-loss-landscape effects that sparse reward typically
  produces in gradient methods; instead the failure mode is
  visible directly in $Y_a$ (cf. §1.1).
- **P3 — $A^\top A$ and $A^\top Y$ are the complete sufficient
  state.** RL n-step returns, replay buffers, and online RLS
  decay are all expressions of "*additively combine sufficient
  statistics across visits*". The successor decoder (D1 below) is
  a second sufficient-statistics solve on the *same* $A^\top A$.
  D2 (novelty) is read directly off $(A^\top A + \alpha I)^{-1}$
  with no extra storage.
- **P4 — Targets $Y$ are the only place where task structure
  enters.** Hindsight relabeling (D4) writes the task into the
  *(state, goal)* axis of the encoder input and into $Y$; the
  `eval_goal` projection is a one-line eval-time use of the same
  axis, no extra solve.
- **P5 — Time is collapsed by pooling.** The streaming classifier
  of SSUP-I §6 has a natural RL analog (pool windows of φ(s) over
  a rollout, regress trajectory return). We did *not* exercise
  this in the present round (cf. §6); it remains the most
  attractive deferred direction.
- **P6 — Per-class additivity.** The RL analog is *per-action*
  additivity: each action owns its own $A_a^\top A_a$, $A_a^\top
  Y_a$. NEF-AVR already obeys this, and it is exactly what makes
  the D1 successor decoder a drop-in second statistic on the same
  encoder.

So the core question for this work, restated in SSUP vocabulary:

> Under P1–P4, can we put non-trivial structure into $Y$ for
> sparse-reward tasks without breaking P3's "one-analytic-solve"
> discipline? In particular, can we keep one shared $A^\top A$ /
> $A^\top Y$ inversion at the heart of the agent and modulate the
> task signal entirely through $Y$ — without target networks,
> gradient descent, or reward shaping?

Everything in §2 is constructed inside that constraint. No
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

## 3. Headline results (exploratory, n=2000, 3 seeds)

> **Status caveat.** Every number in §3 is from a 3-seed
> exploration at `n_neurons=2000` to keep the per-action Ψ memory
> tractable while co-running with the SSUP-II finalization on the
> same CPU. The SSUP paper series reports its headlines at 10
> seeds and full-width networks; the analog for this work is the
> P10–P12 plan in §10 below (10-seed, n=6000+, fresh-base
> rerun). The mechanism conclusions (which D-stack works on which
> env, where eval_goal applies) are robust across seeds in the
> 3-seed range and are taken as load-bearing; the specific
> numbers should be reproduced at full width before they go into a
> paper.

All numbers below are from multi-seed runs on three seeds {0, 1, 2}
under the same seed protocol as the attic'd `rl_clean` line
(warmup/training/eval policies fixed in `benchmarks/run_rl_sparse.py`).
Forget factor and n_neurons noted per row.

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
(`nef_avr`, `nef_sf`, `nef_sf_novelty`) on a deterministic K-state
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
2. **Plain NEF-AVR fails K=20 policy** despite reaching the goal
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
  sparse-reward fix. Per `attic/docs/rl_review.md` §2.5, the bottleneck
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

## 8. Open questions / follow-ups, with likelihood-of-payoff estimates

Each follow-up below is rated on two axes: **likelihood of producing
an interesting finding** (high / med / low) and **cost** (small if a
day or less of CPU + coding, medium if a few days, large if it
needs a full sweep or new infrastructure). "Interesting" here means
"changes the mechanism story, adds an env to the working set, or
sharpens a paper-worthy claim", not "moves seed-mean by 5%".

### 8.A — Replication / hygiene (these are the §10 P10–P12 plan)

1. **Full-width, 10-seed replication of MC + Acrobot.** Rerun the
   `nef_sf_hindsight` + `eval_goal` headline at `n_neurons=6000`
   (the SSUP-I/II default range) across 10 seeds. Adopts the
   SSUP-II reporting bar.
   *Likelihood: very high (necessary for credibility).
   Cost: medium.* This is the work without which §3 numbers
   cannot enter a paper.

2. **Toy sparse-chain as CI smoke.** Add `pytest` invocation of
   `benchmarks/run_sparse_chain.py` with K=5, 200 ep, 1 seed,
   asserting policy_correct=1. ~10 s wall.
   *Likelihood: low (it just locks in a regression target).
   Cost: small.*

3. **Coupling to whatever replaces rl_clean.** The attic'd
   `rl_clean` base config (n=6000, γ=0.98, α=0.1, n_step=100,
   ff=0.999, broadness=0) was the dense-reward starting point.
   When the future SSUP-III ([rl_clean's successor]) settles on
   its own dense-reward base, mirror it here and rerun §3.
   *Likelihood: high (any base shift can invalidate ff defaults).
   Cost: small once the new base lands.*

### 8.B — Likely mechanism wins

4. **Magnitude-normalized ψ projection** (§5 caveat).  
   Change: `Q(s,a;g*) = (ψ(s,a)/‖ψ(s,a)‖) · u(g*)`. One-line patch
   in `NEFHindsightSFAgent._eval_goal_q`. If it makes the toy
   chain solvable without hurting MC/Acrobot, it becomes the
   default and we gain the terminating-action regime.
   *Likelihood: high (the artifact in §5 is mechanical and
   normalization is the obvious fix).
   Cost: small.*

5. **Programmatic goal-obs capture for image envs (MiniGrid).**
   Programmatic placement of the agent at the goal cell to read
   off an observation usable as `eval_goal`, or use the env's
   `mission`-conditioned representation. With that handle, the
   D1+D4+eval_goal stack should apply to DoorKey-5x5 and
   FourRooms.
   *Likelihood: high (the negative result in §3.4 is purely a
   missing handle, not a mechanism flaw).
   Cost: medium (needs MiniGrid wrapper work, possibly a
   reset-to-cell shim).*

6. **`forget_factor` scan on Acrobot.** Acrobot went from "all
   −500" to "−106 to −125 best" between ff=0.999 and ff=0.9999.
   A scan over {0.999, 0.9995, 0.9999, 0.99995, 0.99999} maps the
   boundary and tells us if `forget_factor` should be exposed as
   a per-task default (e.g. tied to `n_step` or episode length).
   *Likelihood: medium-high (the gap is large, so a smooth curve
   would be informative; a sharp threshold even more so).
   Cost: small at 3 seeds, medium at 10.*

### 8.C — Plausibility studies (might surprise either way)

7. **eval_goal *generalization* sweep.** On MC, evaluate the
   trained Ψ against goals near but not at `[0.5, 0.0]`:
   `[0.45, 0.0]`, `[0.5, 0.07]`, `[0.5, −0.07]`, `[0.4, 0.0]`.
   Maps the basin of the projection mechanism; tells us whether
   `eval_goal` is a "point-target" mechanism or a "direction"
   mechanism in φ-space. On Acrobot do the same with the natural
   continuous parametrization of "upright".
   *Likelihood: medium (could either reveal a smooth basin —
   nice paper figure — or a brittle point — also interesting,
   because it would imply Ψ has memorized the exact relabel
   target).
   Cost: small (eval-time only, no retraining).*

8. **Sparser hindsight (`goal_horizon` ablation).** Currently we
   relabel against any state visited later in the episode. Sweep
   the relabeling horizon and the buffer composition. The
   question is whether the eval_goal projection benefits from
   *dense* hindsight (Ψ sees many directions) or from *sparse*
   hindsight near the eventual task goal.
   *Likelihood: medium (could clarify why D2 was unnecessary on
   MC/Acrobot — D4's HER may already be doing the exploration
   work).
   Cost: small.*

9. **A second sparse env with a *natural* goal observation.** The
   "Reacher" sparse-reward family in `gym-robotics`, or a
   ManiSkill-2 reach task, has `(state, goal)` natively in the
   observation. This bypasses the MiniGrid handle problem.
   Confirms that D1+D4+eval_goal is not MC/Acrobot-specific.
   *Likelihood: medium-high if installation is tractable
   (`gym-robotics` is the cleanest); same mechanism story should
   apply.
   Cost: medium (env wiring + driver pass).*

### 8.D — Speculative / SSUP-III material

10. **Trajectory-level pooling for RL (P5 transplant from
    SSUP-I §6).** Pool a window of φ(s_t) over a rollout into a
    fixed-size feature and regress trajectory return. Action
    selection from a trajectory-level Q is awkward (we don't have
    point-wise Q), but trajectory-Q could be used as a *baseline*
    or a *credit signal* alongside per-step FQI; or for option
    selection in a hierarchical setup. This is the SSUP-III
    analog of the streaming classifier.
    *Likelihood: medium (very leenef-faithful; nontrivial design
    work to be useful for control; potentially a paper-grade
    contribution).
    Cost: large.*

11. **Per-task additivity (P6 transplant) for *task-incremental*
    RL.** Each task adds to per-action $A_a^\top A_a$ /
    $A_a^\top Y_a$; on a *new* task, the (state, task-id) encoder
    input adds a new column to Ψ. This is the natural RL analog
    of SSUP-II's "forgetting axis has length zero by construction"
    claim. CL-RL is a notoriously hard benchmark family.
    *Likelihood: medium-high for a clear claim ("no forgetting
    in the analytic readout"); medium for absolute return on
    standard CL-RL benchmarks where the feature ceiling will
    bite.
    Cost: large (needs a task-incremental driver and 2+ task
    suites).*

12. **Bootstrap-free SF *with* multi-step pooling.** Currently SF
    uses Monte-Carlo n-step on φ. SSUP-I's pooling
    (`StreamingNEFClassifier`) suggests pooling φ over a window
    *before* regressing — turning the per-step successor solve
    into a windowed solve. Could give a stable middle ground
    between fully-MC (high variance) and fully-bootstrapped
    (analytical fit amplifies noise).
    *Likelihood: medium.
    Cost: medium.*

13. **`broadness` / `min_relative` sweeps from SSUP-I in the RL
    setting.** SSUP-I §5 mapped the regularization regimes; we
    have not tested whether sparse RL is in the same regime.
    Sparse-reward Y_a is heavily zero, so the trace of Φ_aᵀΦ_a
    dominates the effective α and may push us toward
    over-regularization.
    *Likelihood: medium-low (`attic/docs/rl_review.md` §2.5 argued the bottleneck
    is mechanistic, not regularization). But the trace-scaled α
    centralization on main is a good moment to recheck.
    Cost: small.*

14. **PuckWorld / Pendulum-sparse / FrozenLake.** A small sparse
    benchmark suite beyond MC and Acrobot would harden the
    mechanism claim. PuckWorld is closest to MC's spirit but with
    a continuous-direction goal; sparse-Pendulum tests the
    eval_goal mechanism on a periodic state space.
    *Likelihood: medium (potential for genuine extension or for
    discovering a new failure mode).
    Cost: medium.*

## 9. Replication plan (P10–P12)

The §3 numbers are *exploratory*. Before any of them enter SSUP-III
they need to be redone at the SSUP-I/II reporting bar. Concrete
plan, in order:

- **P10 — Full-width replication.** `nef_sf_hindsight` +
  `eval_goal` on MC and Acrobot at `n_neurons=6000` (one
  configuration, fresh dense-reward base inherited from whatever
  has replaced `attic/rl_clean`). 3 seeds first to spot
  regressions, then expand to 10.
- **P11 — 10-seed headline.** Adopt 10 seeds for the MC and
  Acrobot rows in §3.1 / §3.2 once P10 has confirmed
  qualitative agreement. Report mean ± SE on best_eval,
  robust_eval, final_eval; quote 3 seeds explicitly *only* for
  the LunarLander dense regression (which is a sanity check, not
  a headline).
- **P12 — Toy chain regression in CI.** Add a short pytest entry
  that runs `nef_sf_novelty` on K=5 with one seed and asserts
  policy_correct=1 and goal_reached=1. Catches sign / off-by-one
  drift in n-step successor targets that is invisible inside
  seed noise on the real envs.

The likelihood that P10 will *reproduce* the §3 mechanism story is
high — the 3-seed signal is qualitatively decisive, and the
mechanism (D1+D4+eval_goal) does not depend on `n_neurons` in any
sensitive way. The likelihood that the specific numbers will
*shift* with the new dense-reward base is also high; the bar to
clear is the SSUP-II reporting standard, not the exploratory
numbers in §3.

## 10. Big-picture reflection — what remains stuck, and what isn't

Reading §1–§9 as a whole, here is where I currently stand on the
sparse-reward question. This is the section to read if §3 looks
like a paper-grade win and you wonder why the rest of the document
is hedged.

### 10.1 What works and why

The D1+D4+`eval_goal` stack solves MC and Acrobot from sparse
reward, with three seeds, without gradient descent, without a
target network, without reward shaping, and using **one** $A^\top
A$ inversion per action. That is the result. It works because of
the SSUP-I/II setup, not in spite of it:

- The fixed encoder φ([s; g]) projects the joint state-goal pair
  into a fixed high-dimensional space. Hindsight relabeling
  populates this space with the *connectivity* of trajectories,
  not with reward-driven targets.
- The successor decoder Ψ is a P3 statistic over φ: $\Psi_a =
  (\Phi_a^\top \Phi_a + \alpha I)^{-1} \Phi_a^\top \Phi^+_a$, where
  $\Phi^+_a$ is the Monte-Carlo n-step accumulation of φ. No
  reward enters Ψ — it is purely *trajectory-derived*.
- The `eval_goal` projection at evaluation time reads off the
  policy by asking "which action's ψ best aligns with the
  encoder direction of the true goal?" One linear projection, no
  retraining.

This is the cleanest expression of P4 ("Y is the only place where
task structure enters") that I have seen in this work. Ψ is
task-blind; the task enters only at eval, only as a one-line
inner product, and only with the goal vector φ([0.5, 0.0]) or
φ([−1, 0, 1, 0, 0, 0]).

### 10.2 What is genuinely stuck

Three distinct blockers — each different in character.

- **Image observations have no goal handle.** MiniGrid is stuck
  not because the mechanism is wrong but because we cannot point
  at an "observation at the goal". This is a *plumbing* problem,
  not a mechanism problem. Follow-up #5 in §8.B is the path.
- **Terminating-action / heterogeneous-horizon MDPs.** The chain
  (§5) breaks `eval_goal` projection because magnitudes of ψ
  reflect *horizon* rather than *direction*. Follow-up #4 in
  §8.B (length-normalized ψ projection) is the obvious fix; it
  may break MC/Acrobot in ways we don't yet understand, hence the
  ablation. *This is the one mechanism caveat that I think could
  be either trivial or deep.*
- **Long-horizon credit and forgetting.** Acrobot already needed
  ff=0.9999 (vs 0.999 on MC). Anything materially longer than
  Acrobot (500-step episodes) will need a different decay regime,
  or a different statistic altogether. The honest next test
  here is a sparse Pendulum, not LunarLander-sparse — Pendulum's
  geometry is closest to Acrobot's.

### 10.3 What is *not* stuck but looks scary

- **Sample efficiency** is a known-unscary problem in this line
  of work. The §3 MC numbers come from ≤ 300 episodes of 200 max
  steps. That is competitive with dense-reward gradient
  baselines on MC, and the per-action Ψ solve runs in seconds.
- **Stability across seeds.** All three §3 wins (MC, Acrobot
  ff=0.9999) are 3/3 seed-stable. The mechanism is not
  fragile; numbers vary by ~20% across seeds but the qualitative
  outcome is fixed.
- **Sensitivity to `forget_factor`.** Acrobot wanted 0.9999, MC
  wanted 0.999. That is a *finding*, not a knob to tune away.
  Follow-up #6 in §8.B will tell us whether to expose it as a
  per-task default.

### 10.4 Why the SSUP-III paper isn't ready yet

Three things missing, in priority order:

1. **10 seeds and a full-width network**, per §9. The §3 numbers
   are exploratory; SSUP-II's bar is 10 seeds on
   `nef_accumulate`. Until P10/P11 are done the paper has nothing
   to lead with.
2. **A second working env beyond MC and Acrobot.** Either a
   solved MiniGrid (follow-up #5 in §8.B) or a sparse-Pendulum
   (follow-up #14 in §8.D). One more env converts "mechanism
   works on two envs" to "mechanism works on a family"; this is
   the difference between a workshop note and a paper.
3. **An ablation table over D1 vs D1+D2 vs D1+D4 vs
   D1+D4+eval_goal**, on each working env, at full width and 10
   seeds. The exploratory data in §3 already implies the
   ablation, but a clean, paper-grade ablation has not been run.

The toy chain (§4) is also load-bearing for the *mechanism caveat*
story: it shows where `eval_goal` does not apply and gives us a
cheap regression target for future SF / novelty changes. It is not
a paper headline, but it is the cleanest single piece of mechanism
clarity in this work.

### 10.5 The SSUP-III shape, tentatively

If P10/P11 land, the shape of the paper is roughly:

- §1: Six principles in RL, parallel to SSUP-I §1 and SSUP-II §1.
  P1–P4 are load-bearing; P6 has an RL analog (per-action
  additivity in $A_a^\top A_a$); P5 is the speculative
  direction (§8.D).
- §2: NEF-AVR as the dense-reward base, faithful to the
  attic'd `rl_clean` line; one analytic solve per action.
- §3: Where sparse reward breaks the dense recipe (the
  Y-signal failure of §1.1), and three mechanisms (D1 / D2 / D4)
  each a re-use of an existing leenef primitive.
- §4: The `eval_goal` projection as the decisive *eval-time*
  mechanism, the one-line addition that turns hindsight-trained
  Ψ into a useful policy.
- §5: Results on MC + Acrobot + (one more env from §8.B/§8.D),
  10 seeds, full width; ablations; the toy chain caveat from §5
  as the mechanism-boundary section.
- §6: Discussion: what the eval_goal projection tells us about
  the geometry of Ψ, and why hindsight + projection in an
  analytic-solve setting is not the same operation as hindsight
  + gradient descent.

This is recognizably a SSUP-III: it sits between SSUP-I's
"single analytic step" and SSUP-II's "sufficient statistics
accumulate", and it lands a third claim — "$Y$-only modulation
via hindsight + eval-time projection" — that completes the family.

## 11. File map

| Artifact | Role |
|---|---|
| `src/leenef/rl.py` | `NEFAVRAgent` (D0 baseline) — unchanged in semantics on main; alpha plumbing centralized through `solvers.regularized_normal_matrix` + `solvers.solve_spd` after `8cd844c` |
| `src/leenef/rl_sf.py` | `NEFSuccessor`, `NEFSFAgent`, `NEFHindsightSFAgent` (D1+D2+D4). Still uses an inline regularize+chol pattern; cleanup to use the new helpers is mechanical and not behavior-changing |
| `src/leenef/rl_diag.py` | `SparseChainEnv`, `optimal_q()`, leverage diagnostics |
| `benchmarks/run_rl_sparse.py` | Multi-env driver; `nef_avr` / `nef_sf` / `nef_sf_hindsight` variants; `--eval-goal`; MiniGrid wrapper |
| `benchmarks/run_sparse_chain.py` | Toy chain regression driver — Q* + policy + `eval_goal` grading |
| `tests/test_rl_sf.py` | `test_successor_solve_recovers_chain_optimal_q` and the `eval_goal` / extrinsic invariants |
| `results/rl_sparse/*.json` | Multi-seed run dumps (one per condition) |
| `results/rl_sparse/sparse_chain.json` | Toy regression sweep |
| `docs/SSUP-I.md` | Single-layer + streaming supervised paper; P1–P6 vocabulary |
| `docs/SSUP-II.md` | CL-by-accumulation paper; P3 + P6 as the load-bearing principles for additive sufficient-statistics learning |
| `attic/docs/rl_*.md` | Historical RL notes from the pre-restart `rl_clean` line — context, not parallel work |

## 12. One-paragraph summary

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
