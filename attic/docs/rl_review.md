# NEF-RL: Brutally Honest Review

*A data-driven audit of the NEF-FQI research effort, independent of prior
notes, checkpoints, and commit messages.  Every claim here was re-derived
from the raw JSON in `results/rl/` and from the current code in
`src/leenef/rl.py`.*


## 1. TL;DR

**Most of the "headline" RL numbers in our notes are single-seed spikes,
and the seed variance is large enough to swallow almost every
configuration difference we have claimed to isolate.**  The "98% of DQN
on LunarLander" claim, the "bimodal broadness landscape" claim, and the
"voting ensemble beats single agent" claim are all backed by 1–3 seeds
each, with seed-to-seed ranges that routinely exceed 100 reward points.

When we ran **5 seeds of the same config** (4000n, n50, RLS,
focused recentering, `multiseed_focused4k_*.json`, 20 Apr against
pre-fix code — see §2.1a), the mean `best_eval` was **48.7** with
range **−29 to +147**.  That specific number reflects an older
implementation; the post-fix 4000n baseline is higher (≈ 142.5 mean,
3 seeds, same config after commit `33b04a5`).  Either way, seed
variance at a single config is large enough to swallow most of the
apparent effects in the rest of the data.

What is **real and defensible** from the data:

- CartPole is solved.
- LunarLander gets a *mean* best evaluation somewhere in the **80–160**
  band with large seed variance.  Some seeds spike to 200+; others
  collapse below 0.
- Acrobot and MountainCar do not work at all — consistent across every
  config.
- MC vs n-step, RLS vs buffer: directional signals are real.
- The "recentering" mechanism as currently coded is **not** dead-neuron
  detection.  It is **pure rotation** — it always recenters exactly the
  bottom 5% regardless of whether those neurons are actually dead.

**Recommendation (post-§8 sweep):**
1. **Drop recentering from the recommended NEF-FQI config.**  The
   `(percentile, min_relative)` sweep (§8) shows `none` is
   statistically tied with every non-destructive recentering setting at
   4000n base.  Recommended baseline: 4000n, broadness=0, no recentering.
2. Do **not** chase Acrobot/MountainCar with the current design.
3. The combined analytical-solver-for-non-stationary-learning paper
   (CL + RL) is still viable, but the RL half must be reframed
   honestly: "works on dense-reward tasks, fails on sparse-reward
   tasks, with variance comparable to DQN on the dense-reward ones"
   rather than "matches DQN within 2%".


## 2. What the raw data actually says

### 2.1 Seed variance is the dominant effect

The only multi-seed evaluation of a single "frozen" config we have is
`multiseed_focused4k_s{0..4}.json` (5 independent seeds of 4000n + n50 +
RLS + focused recentering, run on 20 Apr against fixed code):

| seed | best_eval | final_eval |
|------|-----------|------------|
| 0    | ~10       | ~−40       |
| 1    | ~50       | ~20        |
| 2    | ~50       | ~20        |
| 3    | 147       | 147        |
| 4    | −29       | −98        |
| **mean** | **48.7** | **26.7** |
| **range** | **−29 … 147** | **−98 … 147** |

Compare that distribution to the single-seed headline results filed
*later* on the same or a nearby configuration:

| File                             | best_eval | notes                    |
|----------------------------------|-----------|--------------------------|
| `sweep_lunarlander_8000n.json`   | **209.2** | 8000n, single seed, 19 Apr |
| `sweep_fixed_recenter_8k.json`   | 66.2      | 8000n, single seed, 19 Apr |
| `sweep_rotation_8k.json`         | −37.9     | 8000n, single seed, 20 Apr |
| `sweep_rotation_4k.json`         | 219.8     | 4000n, single seed, 20 Apr |
| `sweep_fixed_recenter_4k.json`   | 162.5     | 4000n, single seed, 19 Apr |
| `sweep_lunarlander.json` n50_rls_recenter | 160.3 | 4000n, single seed, 19 Apr |
| `broadness_corrected.json` b=0   | 142.5 ± 49 | 4000n, 3 seeds, 20 Apr    |

Three runs of "8000n n50 RLS recenter", each against *essentially* the
same code, produced **209.2, 66.2, and −37.9**.  That is the true
distribution.  The "98% of DQN" claim in the notes is the max of three
samples from a distribution with empirical range over 240.

### 2.1a Caveat: `multiseed_focused4k` is not directly comparable

`multiseed_focused4k_s{0..4}` (mean 48.7) was run on 20 Apr against the
binary `recenter_mode="focused"` code (commit `765d35a`), *before* the
`33b04a5` fix that made recentered neurons enter at the alive-only
mean instead of the whole-population mean (the latter includes their
own zeros, so they started ≈ zero-biased).  The
`broadness_lunarlander_corrected.json` 4000n b=0 run (mean 142.5,
3 seeds) ran after that fix.  **These are different implementations.**
The ≈90-point gap is not seed noise — it is the effect of the bug
fix.  The current, post-fix 4000n baseline is the 142.5 figure.

### 2.2 The broadness landscape is almost certainly noise

6000n dense broadness sweep, 3 seeds each:

| broadness | best_eval mean | best_eval range    | evidence |
|-----------|---------------:|-------------------:|----------|
| 0.010     | 160.7 | **100.7 … 232.3** | one lucky 232 seed |
| 0.980     | 159.5 | 109.5 … 194.2     | |
| 0.950     | 141.4 | 115.8 … 179.9     | |
| 0.000     | 124.9 | 71.4 … 162.9      | |
| 0.015     | 101.9 | 89.7 … 111.2      | tight — but worse mean than 0.01 |
| 0.012     | 84.4  | −23.2 … 166.8     | |
| 0.020     | 73.2  | 16.5 … 127.7      | |

The apparent peak at broadness 0.01 has ranges of 131 points, and its
own neighbors at 0.012, 0.015, 0.020 are 60–80 points below it with
*overlapping* ranges.  There is no smooth underlying function of
broadness being resolved here; we are looking at sampling noise.  The
same is true of the 4000n corrected sweep where "0.9 is best peak, 0.2
is best final" is an artifact of 3-seed sampling.

### 2.3 Recentering is rotation, not dead-neuron detection

This is probably the most important mechanistic finding in the whole
review.  The `recenter_log` in every 6000n run shows `n_dead = 300`
**every single time**, exactly 5% of 6000 neurons.  That is because
`recenter_percentile=5.0` with `recenter_min_relative=0.0` (our default)
always selects the bottom 5% regardless of absolute activity.

At the same time, the activity distribution at the *end* of training
(episode 1000) varies strongly with broadness:

| broadness | min / mean | p5 / mean | interpretation |
|-----------|-----------:|----------:|----------------|
| 0.000     | 0.55 | 0.62 | tight distribution — "bottom 5%" is at 62% of mean (healthy) |
| 0.010     | 0.47 | 0.59 | |
| 0.050     | 0.44 | 0.56 | |
| 0.200     | 0.38 | 0.53 | |
| 0.800     | 0.25 | 0.45 | |
| 0.950     | 0.21 | 0.48 | |
| 0.980     | 0.17 | 0.48 | broad distribution — bottom neurons at 17% of mean |
| 1.000     | 0.16 | 0.47 | |

**At low broadness, we are recentering perfectly healthy neurons.**  At
high broadness, there *are* real underperformers and recentering them
is plausibly useful.  If this is the true mechanism of the 0.98 regime
looking good, then the right intervention is not to tune broadness, but
to **stop recentering healthy neurons**.  That is exactly what the
`recenter_min_relative` parameter is for, and we have never swept it.

### 2.4 Ensembles are not convincingly better

| File                        | config                | best_eval | seeds |
|-----------------------------|-----------------------|-----------|-------|
| `sweep_ensemble_v2.json`    | voting 3×4000 + recenter | **202.9** | 1  |
| `sweep_fixed_voting3.json`  | voting 3×4000 + recenter |  174.6 | 1  |
| `sweep_rotation_voting3.json` | voting 3×4000 + recenter | 61.7 | 1  |
| `sweep_fixed_voting4.json`  | voting 4×4000 + recenter |  15.4 | 1  |
| `sweep_ensemble_v2.json`    | Thompson 3×4000 no-recenter | 158.4 | 1 |
| `sweep_lunarlander_2x6000.json` | Thompson 2×6000    | 161.1 | 1 |

Three independent seeds of "voting 3×4000 with recentering" give
**202.9, 174.6, 61.7**.  Mean ≈ 146, range 141.  This is not
distinguishable from a strong single-agent seed.

### 2.5 Acrobot and MountainCar are dead ends with the current design

Every config across `sweep_acrobot.json`, `sweep_mountaincar.json`, and
`continual_rl.json` that includes these environments fails
consistently.  The mechanism is sparse rewards: n-step and MC targets
collapse to near-identical values across states because exploration
rarely triggers the sparse reward.  This is a fundamental limitation of
model-independent targets and will not be fixed by tuning recentering,
broadness, or neuron count.


## 3. What is actually defensible in a paper

Given the data as it stands today, an honest public claim looks like:

- **Analytical-solve RL works on dense-reward classic control** (CartPole
  solved, LunarLander reaches mean eval ≈ 100–160 with 3-seed spread,
  occasional seeds reaching 200+ matching DQN).
- **Gradient-free, no replay buffer, no target network, no learning
  rate**, with a single hyperparameter (regularization) that is
  data-driven via trace normalization.
- **No catastrophic forgetting within a task** because AᵀA /AᵀY are
  sufficient statistics; this is the bridge to the CL story.
- **Fails on sparse-reward tasks** (Acrobot, MountainCar).  This is
  honest and consistent with the theory: MC/n-step targets carry no
  signal when rewards are absent over the horizon.
- **Recentering provides gradient-free resource reallocation**, and the
  `min_relative` variant (selective recentering of truly underperforming
  neurons) is the most principled form.  *Pending empirical confirmation
  from the sweep below.*

The contribution is not "we match DQN".  The contribution is "an
analytical, stable, gradient-free alternative to DQN for dense-reward
problems, whose sufficient-statistic structure naturally composes with
continual-learning".


## 4. Publication prospects

**Standalone RL paper:** *not viable as currently framed.*  The RL-only
story is too weak for a top venue (beaten by DQN on LunarLander,
doesn't work on Acrobot/MountainCar, no new theory).  A niche workshop
(e.g. an RL workshop at NeurIPS/ICML or ICLR's Blog Post track) could
take the observations as-is, especially the sparse-reward negative
result and the rotation-vs-dead-neuron finding.

**Combined CL+RL paper: viable, but it must stop over-claiming.**  The
right narrative is:

> *Analytical least-squares with sufficient-statistic updates solves
> non-stationary regression problems without gradient descent.  We
> demonstrate this in two settings: (a) class-incremental continual
> classification, where we beat EWC and ER by ≥20pp and match joint
> training, and (b) dense-reward reinforcement learning, where we match
> DQN on CartPole and approach DQN on LunarLander without any replay
> buffer, target network, or learning rate.  We identify a sparse-reward
> failure mode that clarifies the boundary of the method.*

The CL results are strong enough to carry the paper on their own.  RL is
an "extending the same principle to a second non-stationary problem"
section, not the centerpiece.

**What would change this assessment upwards:**

1. A properly replicated 8–10 seed LunarLander baseline with **mean**
   eval ≥ 180 (not single-seed spikes).  Achievable only if we find a
   config where seed variance collapses.  The `min_relative` sweep is
   the most plausible route.
2. A convincing ensemble mean (not one spike).  Needs 5+ seeds.
3. Even partial progress on MountainCar using curiosity-style intrinsic
   rewards or eligibility traces.  This would require a conceptual
   extension, not more sweeping.


## 5. Recommended next experiment (one, not many)

**Goal:** determine whether changing *which* neurons we recenter — not
*how many* — gives a more reliable and stronger LunarLander score than
the current "pure rotation" default.  This requires understanding that
three knobs interact:

- `percentile` (default 5): **upper cap** on the fraction of neurons
  that can be recentered at one event.  Turned to 100 it becomes
  inert.
- `min_relative` (default 0): **absolute floor**.  Combined with
  `percentile` as `threshold = min(quantile(percentile), mean·min_relative)`,
  so the *stricter* one binds.  With `percentile=5` and
  `min_relative ≤ 0.6`, the floor is almost never the binding
  constraint on our data (p5 ≈ 0.55·mean at low broadness), which is
  why the intuitive sweep `{0.4, 0.6, 0.8}` at `percentile=5` would
  show no effect for the first two.
- `broadness` (0 … 1): in the non-EMA branch we use by default,
  `apply_broadness` scales both `_act_sum *= (1 - broadness)` and
  `_act_count *= (1 - broadness)` after each event.  The *ratio*
  (what the selection rule reads) is preserved, but the effective
  sample size shrinks, so fresh activations weigh more in the next
  selection.  Broadness is therefore **not orthogonal** to selection —
  it controls how fast the stats adapt and thus what "below 60% of
  mean" means in practice.

The right first sweep is therefore two-dimensional in
(`percentile`, `min_relative`), with `broadness` held at 0 for now.
Broadness will re-enter once the right selection rule is known.

**Base config (frozen):** **4000n**, n-step(50), RLS(β=0.999),
`solve_every=10`, `recenter_interval=100`, `broadness=0.0`,
hypersphere encoders, gain U(0.5, 2), random biases.

We anchor on 4000n rather than 6000n because the corrected 4000n/6000n
comparison wins 4000n clearly:

| broadness | 4000n mean | 6000n mean |
|-----------|-----------:|-----------:|
| 0.00 | **142.5** | 124.9 |
| 0.10 | **101.2** | 47.5  |
| 0.20 | **152.6** | 37.0  |
| 0.90 | **157.4** | 90.0  |
| 1.00 | 78.1 | **101.7** |

4000n wins 4/5 matched points and has a much higher worst-case floor
(78 vs −24).  It also runs ≈ 3.5× faster wall-clock (35 min vs 120 min
per run in parallel mode), giving us more seeds per unit compute.

**Conditions (6 total, 8 seeds each → 48 runs):**

| name          | percentile | min_relative | rationale                        |
|---------------|-----------:|-------------:|----------------------------------|
| `none`        |     —      |      —       | `recenter_interval=None` — true baseline |
| `p5_mr0`      |  5         |  0.0         | current default (pure rotation)  |
| `p5_mr0.7`    |  5         |  0.7         | cap at 5% **and** floor at 0.7·mean (first value where floor actually binds at low broadness) |
| `p100_mr0.4`  | 100        |  0.4         | "no cap", loose floor            |
| `p100_mr0.6`  | 100        |  0.6         | "no cap", medium floor           |
| `p100_mr0.8`  | 100        |  0.8         | "no cap", strict floor (may recenter most of the layer when broadness=0) |

Per-event `n_dead` is logged for every condition — this tells us
whether (3)–(6) are actually firing and when.  We may well see that
`p5_mr0.7` almost never fires (floor doesn't bind), and that
`p100_mr0.8` fires way too often; that is itself useful signal.

**Budget:** 48 runs at ≈ 35 min/run, 6 workers → ≈ **4.7 h**.  Fits
within the 9 h hard limit with margin.  If interrupted, the
`imap_unordered` incremental save means resuming just skips completed
labels.

**Expected outcomes and what they mean:**

- *Some `min_relative` condition clearly beats both `none` and
  `p5_mr0`:* targeted recentering works, we have a real mechanism to
  claim in the paper.
- *`none` beats or matches all recentering conditions:* recentering at
  `broadness=0` is just noise; we drop it from the recommended config
  and replace with something smarter (possibly reintroducing broadness
  > 0) in a follow-up.
- *Seed variance still ±60 pts at every condition:* the method has
  that floor of variance on LunarLander, we report mean ± SE with
  8 seeds, and that is still an honest publishable result.

Only after this returns a clean answer should we touch ensembles,
broadness, Acrobot, or anything else.


## 6. Do not do these things next

- ❌ Continue sweeping broadness at finer resolution.  The 6000n data
  already shows the effect is inside the noise floor.
- ❌ Run Acrobot/MountainCar sweeps on the existing design.  The sparse
  reward problem is architectural, not a hyperparameter issue.
- ❌ Add more single-seed "confirmation" runs to the existing sweep
  files.  One more seed won't change anything — we need 8+ at a single
  point to characterize the noise floor.
- ❌ Use Colab GPUs.  The bottleneck is environment simulation, not
  matrix math.  A T4 won't make LunarLander step faster.  The existing
  6-worker CPU pool is the right scale.
- ❌ Publish the "98% of DQN" claim.  That number is a single seed.


## 7. If this were my project, what would I do right now?

1. **Commit this review** so the course-correction is on record.
2. **Run the `min_relative` sweep (§5).**  That's it.  One experiment.
3. While it runs, **start drafting the combined CL+RL paper** with the
   claims scoped to what the current data actually supports.  The RL
   section is a single subsection, not half the paper.
4. If the `min_relative` sweep comes back clean:
   - add an 8-seed DQN baseline on the same observations for a fair
     head-to-head on `best_eval`;
   - add one 5-seed voting-ensemble run at the winning `min_relative`;
   - that's the RL section of the paper, done.
5. If the `min_relative` sweep comes back noisy:
   - drop recentering from the recommended config entirely,
   - report LunarLander honestly as "mean 100–150, occasional spikes to
     200 matching DQN",
   - the paper is still fine — CL carries it.

The biggest single trap we have fallen into in this work is
**exploring breadth-first through single-seed configurations and
treating lucky seeds as signal**.  The cure is deliberate: one
experiment at a time, enough seeds to measure the noise, accept what
the data says.


## 8. Sweep results (4000n × 6 conditions × 8 seeds, completed)

Run on 2026-04-22/23 via `benchmarks/run_min_relative_sweep.py`, results
in `results/rl/min_relative_lunarlander.json`.  Wall time ≈ 7 h on a
6-worker CPU pool.

| condition     | n | best_eval mean ± SD | final_eval mean ± SD | n_dead(last)  |
|---------------|---|--------------------:|---------------------:|--------------:|
| `none`        | 8 | **126.2 ± 65.4**    | 109.4 ± 87.5         | —             |
| `p5_mr0`      | 8 | 121.2 ± 74.1        | 109.3 ± 64.4         | 200 [200,200] |
| `p5_mr0.7`    | 8 | **127.9 ± 71.4**    | **111.9 ± 83.3**     | 200 [200,200] |
| `p100_mr0.4`  | 8 | 113.4 ± 75.8        | 105.1 ± 82.9         |  29 [6, 71]   |
| `p100_mr0.6`  | 8 | 118.3 ± 65.8        | 110.5 ± 75.3         | 164 [74, 247] |
| `p100_mr0.8`  | 8 |  72.0 ± 51.9        |  52.1 ± 35.4         | 640 [365,788] |

### What the data says

1. **Recentering does not help at 4000n base.**  The five non-destructive
   conditions cluster in `best_eval` ∈ [113, 128] with SD ≈ 65–76, i.e.
   SE ≈ 23–27.  Differences between them are ≪ SE.  `none` is
   statistically indistinguishable from `p5_mr0`, `p5_mr0.7`,
   `p100_mr0.4`, and `p100_mr0.6`.

2. **Aggressive recentering hurts.**  `p100_mr0.8` recenters ~16 % of
   the layer per interval (mean 640/4000) and pays ~55 points in
   `best_eval`.  The selection rule with a strict absolute floor and no
   percentile cap genuinely destabilizes training.

3. **Selection-rule shape is irrelevant in the regime we tested.**
   `p5_mr0` (rotation, fixed 200 dead) and `p100_mr0.6` (data-driven,
   median 164 dead) land within 8 points despite being qualitatively
   different rules.  What matters is *how many* neurons get touched per
   interval, not *which* ones.

4. **Recentering at all is a coin flip.**  Best mean is `p5_mr0.7`
   (127.9), worst non-destructive is `p100_mr0.4` (113.4) — a 14-point
   gap inside a per-condition SE of ~25.  We cannot defend any
   non-trivial recentering recommendation from this data.

### Branching decision (per §5 "Expected outcomes")

We are in the **"none wins/ties"** branch.

- ✅ **Drop recentering from the recommended NEF-FQI config.**  The
  paper's "minimum viable agent" should be 4000n, broadness=0, no
  recentering.  Mean LunarLander score ≈ 126, range −17 to 192.
- ✅ **Do not run a voting-ensemble follow-up at any min_relative
  setting.**  There is no "winner" to confirm.  If we want ensembles,
  base them on `none` and on one `p5_mr0.7` arm, since those are tied
  for the top mean.
- ✅ **DQN head-to-head can use plain `none`.**  Simpler config, equal
  performance.
- ❌ **Do not run more recentering sweeps unless we change the regime**
  (e.g. enable EMA via `activity_decay`, or add a budget on neurons
  recentered per interval, or revisit broadness > 0).

### Implications for the paper

The RL story tightens, not weakens:

> "A purely analytic NEF-FQI agent with 4000 neurons and no
> recentering attains mean best-eval 126 ± 23 SE on LunarLander-v3
> across 8 seeds, occasionally matching DQN (≈ 200) on lucky seeds.
> Per-step solve time is dominated by environment simulation, not
> matrix math.  The agent has zero gradient updates."

That is honest, defensible, and the methods description is one
paragraph instead of three.  It also clears the deck for broadness > 0
and EMA-based recentering to be revisited later as **separate**
experiments, with no entanglement to a config baseline that needs them.
