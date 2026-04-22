# NEF-FQI: Brutally Honest Strategic Review

Purpose: cut through 18 checkpoints and 30+ result files to answer two
questions:

1. What does the raw data actually support (vs. what `rl_notes.md` claims)?
2. Is there a story worth finishing, and if so how?

This document intentionally disregards the headline numbers in
`rl_notes.md`, `AGENTS.md`, and commit messages, and only trusts what is
in `results/rl/*.json` read back from disk. All numbers below are
computed from raw JSON, not copied from prose.


## 1. TL;DR

- The method **works and is interesting on CartPole and LunarLander**, but
  essentially every "headline" number in `rl_notes.md` is a
  **single-seed result** in a regime with very large seed variance
  (σ often 50–120 on a reward scale of ~300). Those numbers should not
  be believed as tight estimates.
- The **best config we can defend from multi-seed data on LunarLander-v3
  is in the 140–160 range** (best_eval), clearly below DQN (~210).
  The claimed "2% gap" and "219.8 beating DQN" and "202.9 voting" are
  all single-seed runs, two of which have been explicitly falsified by
  later 3–5 seed follow-ups.
- **Most hyperparameters are still entangled**. Only n-step horizon,
  target mode (mc vs nstep vs td vs λ), and RLS β have been swept
  somewhat cleanly. Everything else (neuron count, recenter interval,
  percentile, min_relative, broadness, ensemble strategy, ε schedule,
  α, activity tracker) co-varies across sweeps.
- **Sparse-reward tasks (Acrobot, MountainCar) are a dead end for this
  formulation as it stands.** No config produced a non-failing result;
  this is a structural limitation of MC/n-step with undifferentiated
  returns, not a tuning issue.
- **Is this a dead end overall?** No — but the way it is currently
  framed ("beats DQN", "closes the gap to 2%") is a dead end.
  There is a real, modestly novel story around *gradient-free,
  buffer-free, analytically stable* RL on dense-reward episodic tasks
  with transparent mechanisms (LSPI-with-random-features + rotating
  data-driven recentering + voting ensembles). That story is worth
  finishing if and only if we stop chasing DQN and switch to defending
  an honest set of multi-seed numbers and an honest set of claims.


## 2. What we have, in one picture

Fixed for the entire review:

- **Target mode: n-step(50)** — clearly dominant on LunarLander over MC,
  TD, λ-returns, n=20/100 (target_modes.json, td_lambda.json, and
  every subsequent sweep).
- **RLS forgetting β=0.999** — every buffer-mode comparison we have
  (CartPole sweep, LunarLander sweep, lunar_scaling) loses to RLS.
- **Activation = abs, encoder = hypersphere, gain ~ U(0.5, 2.0)** —
  never swept in RL; inherited from the supervised defaults.
- **recenter_interval = 100, recenter_percentile = 5.0** — not swept
  independently in any multi-seed experiment.
- **γ = 0.99, α = 1e-2, solve_every = 10, ε 1.0→0.01 over 500 eps** —
  not swept in RL.

Still unresolved axes:

| Axis                    | Current defaults                    | Status                                  |
|-------------------------|-------------------------------------|-----------------------------------------|
| n_neurons               | 4000 / 6000 / 8000                  | non-monotonic in single-seed data; 3 seeds insufficient |
| broadness               | 0.0 (simplest) / 0.2 / 0.9 / 0.01 / 0.98 | 3-seed sweeps; multiple plausible optima |
| recenter_min_relative   | 0.0 (no floor)                      | **never used in any run so far**        |
| activity tracker        | cumulative mean                     | EMA tested once, hurts; not revisited   |
| ensemble strategy       | off / Thompson / voting             | single-seed only for all ensemble runs  |
| ensemble size × width   | 1 / 3×4k / 2×6k / 4×3k              | mostly single-seed                      |
| ε schedule              | 1.0→0.01, decay 500                 | 1.0→0.05 was used in §4.7 and not repeated |


## 3. Multi-seed reality check

The rl_notes.md top-line table lists best_eval 219.8, 209.2, 202.9,
186.3, 160.7, 159.5 from different points in the project. All are
**single seed** except the broadness sweeps.

### 3.1 The only runs we have with ≥3 seeds on LunarLander

Recomputed from `broadness_lunarlander_corrected.json` (4000n, 3 seeds)
and `broadness_lunarlander_6000_edges.json` (6000n, 3 seeds). All use
n50 + RLS(β=0.999) + recenter/100 at 5th percentile, alive-only mean
fair start, ε 1.0→0.01 over 500 eps. `mr` = `recenter_min_relative`.

|  n   |  broadness |  mr |  seeds |  best μ |   ±σ  |  min   |  max  | final μ |
|-----:|-----------:|----:|-------:|--------:|------:|-------:|------:|--------:|
| 4000 |   0.0      | 0.0 |   3    |  142.5  |  52.9 |  105.1 | 203.1 |   130.6 |
| 4000 |   0.1      | 0.0 |   3    |  101.2  |  49.2 |   45.9 | 140.0 |    99.3 |
| 4000 |   0.2      | 0.0 |   3    |  152.6  |  60.6 |   90.1 | 211.0 |   137.4 |
| 4000 |   0.3      | 0.0 |   3    |   96.8  |  90.2 |   -7.3 | 151.6 |    76.7 |
| 4000 |   0.5      | 0.0 |   3    |   91.4  |  53.0 |   44.8 | 149.0 |    91.4 |
| 4000 |   0.7      | 0.0 |   3    |  102.0  |  69.8 |   24.4 | 159.8 |    88.6 |
| 4000 |   0.9      | 0.0 |   3    |  157.4  |  25.9 |  128.9 | 179.5 |   104.9 |
| 4000 |   1.0      | 0.0 |   3    |   78.1  |  98.6 |  -29.0 | 165.1 |    63.7 |
| 6000 |   0.0      | 0.0 |   3    |  124.9  |  47.7 |   71.4 | 162.9 |   112.9 |
| 6000 |   0.01     | 0.0 |   3    |  160.7  |  66.6 |  100.7 | 232.3 |   144.3 |
| 6000 |   0.95     | 0.0 |   3    |  141.4  |  34.0 |  115.8 | 179.9 |   141.4 |
| 6000 |   0.98     | 0.0 |   3    |  159.5  |  44.4 |  109.5 | 194.2 |   159.5 |

And the only ≥3-seed evaluation of a pre-broadness recentering config
(`multiseed_focused4k_s{0..4}.json`, 4000n, "focused" mode, 5 seeds):

| label                            |  seeds  | best μ |  ±σ  |  min  |  max  | final μ |
|----------------------------------|:-------:|-------:|-----:|------:|------:|--------:|
| focused (pre-broadness, global reset off) |   5    |   48.7 | 75.4 |  -29.0 | 147.1 |   26.7 |

Key conclusions from this table alone, without consulting notes:

1. **The σ is huge.** At 3 seeds even the best 4000n setting (b=0.9)
   has σ ≈ 26 (95% CI roughly ±60). For most settings σ > 50. Any
   claim of "5 points better" is noise.
2. **The pre-broadness "focused" run (mean 48.7) falsifies the single-
   seed 219.8 headline.** The 219.8 was seed-lucky under code that
   has since been replaced.
3. **4000n `broadness=0.2` ties `broadness=0.9` within noise.** The
   §11.10 claim that the optimum is at 0.2 *and also at 0.9* is
   consistent with the data, but "the optimum is broadness X" is not
   defensible from 3 seeds — all of 0.0, 0.2, 0.9 are in one fuzzy
   best cluster.
4. **6000n does not clearly beat 4000n.** 6000n b=0.01 (160.7) edges
   4000n b=0.9 (157.4) by less than half a σ. The claimed 6000n
   "bimodality" with peaks at 0.01 and 0.98 is plausible but the
   1.0 point also scores 101.7, the 0.99 point scores 62.4, and 0.95
   scores 141.4 — so the "high-edge basin" looks more like noise plus
   one plateau around 0.95–0.98 than a sharp peak.
5. **Best-effort honest LunarLander claim, with 3 seeds:**
   *best_eval ≈ 150 ± 30 across a broad set of configurations.*
   That is meaningfully below DQN (~210).

### 3.2 Single-seed-only headline numbers (use with caution)

From `sweep_lunarlander_8000n.json`, `sweep_rotation_*.json`,
`sweep_ensemble_v2.json`:

|                config                 |   file                         | best | final |
|----------------------------------------|--------------------------------|------|-------|
| 8000n n50_rls_recenter (pre-fix)       | sweep_lunarlander_8000n.json   | 209.2| 209.2 |
| 4000n n50_rls_recenter (pre-fix)       | sweep_lunarlander.json         | 160.3| 160.3 |
| 4000n n50_rls_recenter *focused*       | sweep_rotation_4k.json         | 219.8| 219.8 |
| 8000n n50_rls_recenter *focused*       | sweep_rotation_8k.json         | -37.9| -37.9 |
| voting 3×4000 *focused*                | sweep_rotation_voting3.json    |  61.7|  30.5 |
| voting 3×4000 broad+recenter           | sweep_ensemble_v2.json         | 202.9| 201.4 |
| Thompson 3×4000 no-recenter            | sweep_ensemble_v2.json         | 158.4| 158.4 |
| voting 3×4000 no-recenter              | sweep_ensemble_v2.json         | -48.7| -50.0 |
| sweep_fixed_recenter_4k (min_rel=0.1)  | sweep_fixed_recenter_4k.json   | 162.5| 150.3 |
| sweep_fixed_recenter_8k (min_rel=0.1)  | sweep_fixed_recenter_8k.json   |  66.2|  66.2 |

The pattern that emerges once we stop trusting single seeds:
*every single-seed number > 180 is probably seed-lucky, and every
number < 0 with a config that looks reasonable is probably seed-
unlucky.* The multi-seed distribution makes both extremes easy to
produce.

### 3.3 Non-reproducibility lives inside the code

Checkpoint 028 already identified that `_recenter_dead()` uses the
global `torch.randint` generator, so "same seed" runs don't reproduce
identical recentering trajectories. This partly explains the seed
variance and also explains why the same config gave 219.8 once and
106.5 another time. **Any review of recentering mechanisms has to be
done with a seeded local generator first, otherwise we are measuring
something slightly different every time.**


## 4. Mechanism check: what does recentering actually do?

The recenter_log saved inside the broadness JSONs gives us the exact
per-neuron activity distribution at each recentering check. This is
the first time in the project we can actually *look* at whether
"dead neurons" exist.

### 4.1 Activity statistics over training (seed 0, 4000n, b=0.0)

|  ep  | min/mean | p5/mean | mean | std  |
|-----:|---------:|--------:|-----:|-----:|
| 100  |   0.186  |  0.396  | 1.07 | 0.47 |
| 200  |   0.366  |  0.472  | 1.08 | 0.45 |
| 300  |   0.406  |  0.499  | 1.06 | 0.42 |
| 500  |   0.456  |  0.559  | 1.09 | 0.40 |
| 700  |   0.438  |  0.551  | 1.17 | 0.44 |
| 1000 |   0.529  |  0.595  | 1.17 | 0.42 |

Under broadness=0.0 with the alive-only fair-start fix, the activity
distribution tightens monotonically: after about 300 episodes the
"bottom 5%" is already above 50% of the mean, i.e. there are **no
more dead neurons** in any meaningful sense. Further recentering is
pure rotation noise.

Under broadness=1.0 the fair-start plus full-decay makes the
distribution look perpetually fresh: min/mean hovers at 0.15–0.20
throughout. That's not because neurons are genuinely dead — it's
because the stats are wiped every 100 episodes and the short-window
distribution always has a ragged lower tail.

**So "dead neuron recentering" is a misnomer.** What we are actually
running is either:

- **Pure rotation** (broadness 0.0–0.2): refresh bottom 5% by long-
  term cumulative weakness, in an environment where the network
  quickly runs out of weak neurons.
- **Stats-washing rotation** (broadness 0.9–1.0): refresh bottom 5%
  by short-window noise, i.e. approximately random-subset replacement.
- **Something in between** (broadness 0.3–0.8): a mix that performs
  somewhere between these two, with no apparent dominant basin.

### 4.2 Why `recenter_min_relative` is worth another look

The user's hypothesis — use `min_relative` so only genuinely weak
neurons get recentered — is directly testable now. Current code:

    threshold = min(p_percentile, min_relative * mean)
    dead = act <= threshold

That is: `min_relative` can only *restrict* selection, it can't
expand it. With `min_relative = r`, the number of neurons recentered
at each check is `|{i : act_i <= min(p5, r · mean)}|`.

From the table above, for broadness=0.0:

- `min_relative = 0.35`: active at ep 100 (min/mean=0.19 → some below
  0.35·mean), tails off by ep 200 (min/mean≈0.37). Recentering
  effectively stops within 200 episodes.
- `min_relative = 0.45`: active through ep 400–500, then stops.
- `min_relative = 0.55`: active nearly the whole run.
- `min_relative = 0.0`: current behavior — pure rotation forever.

This is a genuinely new mechanism we have not tested:
**adaptive self-terminating recentering.** It is different from
"broad/focused" and from any broadness value — it changes the
*amount* of recentering as a function of how well the neurons are
distributed, instead of forcing a fixed 5% per check.

On the broadness data we have, this mechanism would have:

- recentered many neurons during the first ~200 episodes (when there
  is still a genuine tail of weak neurons), and
- stopped recentering afterwards, leaving the trained Q-function
  stable.

This fits naturally with the hypothesis that the rotation is
useful early (exploration, feature-space coverage) but harmful late
(pulls the solve off a good fit). It is also a cleaner story to
publish than "broadness=0.2 ties broadness=0.9 within noise".

### 4.3 Implementation caveat

The existing `dead_neuron_indices` uses `min(percentile, min_relative*mean)`.
With `percentile=5.0`, this caps the count at 5%, which is fine; but
it also means `min_relative` and `percentile` interact in a slightly
non-obvious way. For the min_relative experiment I would switch to:

    dead = (act <= min_relative * mean)
    if percentile is not None:
        dead = keep_at_most(dead, percentile_of_n)

i.e. "recenter only genuinely weak neurons, and at most `percentile`
of them per check". This cleanly separates the two knobs.


## 5. Headline claims vs. data

| Claim in notes                                  |  Data evidence                | Verdict |
|-------------------------------------------------|-------------------------------|---------|
| "n-step(50) is the best target mode"            | target_modes.json, 1 seed; consistent across later 3-seed sweeps | **Supported** (within the LunarLander/dense-reward regime). |
| "RLS >> buffer"                                 | sweep_lunarlander.json, rls_lunarlander.json, rls_cartpole.json | **Supported** (single seed each but consistent). |
| "TD(λ) destroys analytical solve"               | td_lambda.json, 1 seed       | **Directionally supported** (more seeds needed for rigor). |
| "LunarLander gap to DQN is 2%"                  | single-seed 209.2 on 8000n   | **Not supported.** Multi-seed best is ~150–160. |
| "219.8 beats DQN at 4000n focused"              | 1 lucky seed; 5 seeds: 48.7 mean | **Falsified.** |
| "Voting 3×4000 reaches 202.9, best ensemble"    | 1 seed                       | **Plausible but unverified.** |
| "Thompson 3×4000 reaches 186.3"                 | 1 seed, different ε schedule | **Plausible but unverified.** |
| "n_neurons is non-monotonic (4000=160, 6000=-24, 8000=209)" | 1 seed at each         | **Noise.** Within 3-seed σ of each other. |
| "Broadness=0.2 is the optimum"                  | 3 seeds                      | **Weakly supported**, tied with 0.9 (and with 0.0). |
| "Broadness=0.98 defines a high-edge basin at 6000n" | 3 seeds                  | **Weakly supported**, not a sharp peak. |
| "CartPole solved (500/500)"                     | rls_cartpole.json + sweep_cartpole.json, 1 seed each | **Supported.** |
| "Acrobot fails"                                 | sweep_acrobot.json, 5 configs | **Fully supported.** |
| "MountainCar fails"                             | sweep_mountaincar.json, 5 configs | **Fully supported.** |
| "Continual LunarLander: strong forward transfer A↔B, C destroys prior" | continual_rl.json, 1 seed | **Plausible but 1 seed only.** |


## 6. Is this a dead end?

**Not categorically — but the current pitch is.**

Framing as "NEF-FQI competes with / beats DQN on LunarLander" is not
defensible with the data we have. If we publish that, a reviewer
runs 5 seeds and we look silly.

The method does have genuine merits that a modest, honest paper can
stand on:

1. **Gradient-free, buffer-free online RL** that is numerically
   stable on dense-reward episodic tasks. No learning rate, no
   target network, no replay, no gradient clipping. This matters
   for pedagogy, for tiny embedded systems, and for researchers
   who want a glass-box LSPI-style baseline to compare against.
2. **A clean connection to LSPI / LSTD with random features**,
   plus a concrete mechanism (adaptive centers via recentering,
   forgetting via RLS) that addresses the classical weaknesses of
   fixed-feature LSPI.
3. **Forward transfer in continual RL** from shared-representation
   normal-equation accumulation. This is a natural extension of
   the CL story and does not require the RL method to beat DQN.
4. **Transparent, diagnosable behavior.** We can literally print
   the activity distribution and watch the recentering mechanism
   turn itself off as the network matures.

What the method is **not**:

- A DQN killer.
- A solution for sparse-reward RL (Acrobot, MountainCar).
- A scalable method for high-dimensional continuous state (Atari,
  MuJoCo). We have no evidence either way, but the O(n²·data) solve
  and the dependence on dense reward make it unlikely to carry.

### 6.1 Recommendation

Keep going, but **reframe and restrict**. The rest of this doc
proposes a concrete plan.


## 7. Recommended base configuration

One config, defended one axis at a time.

### 7.1 Base

| Setting               | Value       | Evidence                                       |
|-----------------------|-------------|------------------------------------------------|
| target_mode           | `nstep`     | target_modes.json, all later sweeps            |
| n_step                | 50          | target_modes.json (n20<n50, n100<n50)          |
| forget_factor (β)     | 0.999       | rls_lunarlander.json (β=0.999 > β=0.995)       |
| γ                     | 0.99        | not swept; inherited                           |
| α                     | 1e-2        | not swept; inherited                           |
| solve_every           | 10          | not swept; inherited                           |
| ε schedule            | 1.0 → 0.01, decay 500 | default; some evidence 0.05 helps Thompson |
| activation            | `abs`       | inherited; every RL run so far                 |
| encoder strategy      | `hypersphere` | inherited                                     |
| n_neurons             | **4000**    | most widely tested; no robust evidence for 6000/8000 |
| recenter_interval     | 100         | never swept                                    |
| recenter_percentile   | 5.0         | never swept                                    |
| activity tracking     | cumulative mean | EMA hurts (target_modes.json + sweep)      |
| recenter_min_relative | **0.0** for now | start with pure rotation baseline          |
| broadness             | **0.0**     | simplest; in the top 3-seed cluster with 0.2/0.9 |
| ensemble              | off         | single-agent first                             |

**Rationale for 4000n over 6000n/8000n:** 3-seed best means are
142.5 (4000, b=0) vs 124.9 (6000, b=0) vs single-seed 209.2
(8000, b=0 pre-fix). The only *multi-seed* evidence we have says
4000n is at least as good as 6000n at b=0, and 8000n has never been
tested on more than 1 seed under the corrected code path. 4000n also
trains in ~1500–2000s/run vs ~7000s for 8000n, which matters for
seed counts.

**Rationale for broadness=0.0:** cleanest interpretation (no decay
of survivor stats), cleanest story for a paper, in the top cluster
by 3-seed data, and the best candidate for combining with a
`min_relative` floor.


## 8. Plan

### Phase A — Honest baseline (do this first)

A.1 **Fix the recentering RNG.** Use a seeded local generator in
`_recenter_dead()` so the same seed gives the same recentering
trajectory. Without this the rest of the phase is built on sand.

A.2 **5-seed baseline run of the §7.1 base config on CartPole-v1
and LunarLander-v3.** 1000 episodes each. Record mean, std, min,
max for best_eval and final_eval. This replaces every headline
number in rl_notes.md.

A.3 **Confirm Acrobot-v1 and MountainCar-v0 still fail on this
config, 3 seeds each.** This is a negative-result tag, not a
tuning target.

Deliverable: a short "verified results" section pinned to the base
config with (μ ± σ, min, max) over 5 seeds.

### Phase B — Underperformer-only recentering

B.1 **Change `dead_neuron_indices` semantics** so `min_relative`
selects "neurons below r·mean" and `percentile` is just a cap:

    dead = (act <= min_relative * mean)
    if dead.numel() > n * percentile / 100:
        keep the weakest n*percentile/100

This makes the mechanism "recenter the weakest neurons, up to 5%,
but only if they're genuinely below the floor". At `min_relative=0`
it collapses to the current pure-rotation default.

B.2 **Sweep min_relative ∈ {0.0, 0.25, 0.35, 0.45, 0.55}** at the
base config, broadness=0.0, 5 seeds each.

B.3 **Also record `n_recentered` per check** so we can report the
effective mechanism — "recentering was active for the first ~N
episodes, then stopped" is a much better story than another
un-interpretable hyperparameter.

Expected outcome (not guaranteed): there is a setting around
min_relative ≈ 0.35–0.45 that matches or slightly beats pure
rotation at 4000n, with **the same or lower variance** and with
recentering self-terminating before episode 500. If yes, that is
the cleanest version of the "revive underperformers" story. If no,
we accept that rotation is the whole story and move on.

### Phase C — Voting ensemble on the locked config

Only after Phase A has produced a defensible single-agent baseline.

C.1 **3×base voting ensemble** (so 3×4000 = 12000 features total),
broadness taken from A.2 / B.2, **5 seeds**.

C.2 **3×base Thompson ensemble** under identical conditions for
comparison.

C.3 **Report ensemble numbers as (μ ± σ)**, and specifically compare
to the single-agent baseline at the same total feature budget (a
single 12000-neuron agent would be the right control — expensive
but needed to answer "is it the ensemble or just the capacity?").

### Phase D — Continual RL, already-promising story

D.1 **Sequential LunarLander A → B → C under the base config,
5 seeds.** Use β=1.0 for pure accumulation (best case) and
β=0.999 for mild forgetting; report both.

D.2 Accept whatever we find. The §4.8 findings are plausible;
multi-seed will tell us whether A=165 / B=178 / C=-69 is typical or
a lucky run. Even a degraded version of this result is still a
publishable forward-transfer / interference story.

### Phase E — Write-up

Once Phases A–D have landed, the paper / report story becomes:

- **"Gradient-free, buffer-free RL with adaptive random features."**
- **Claim 1:** CartPole solved with no replay buffer, no target
  network, no gradient steps, and no tuning (μ ± σ over 5 seeds).
- **Claim 2:** LunarLander reaches ~150 ± 30 best_eval at 4000
  neurons in ~30 minutes on CPU, with a stable, interpretable,
  self-terminating recentering mechanism. **Below DQN, but with
  properties DQN doesn't have.**
- **Claim 3:** Voting ensembles add M points (TBD) with N× the
  compute.
- **Claim 4:** Forward transfer on continual LunarLander variants.
- **Explicit negative results:** sparse-reward tasks (Acrobot,
  MountainCar) not yet addressed.

That is a small, honest, interesting paper. The path to it is the
plan above.

### What we do *not* do

- Do **not** run another single-seed sweep and publish the best
  number.
- Do **not** claim DQN parity.
- Do **not** start new hyperparameters until the current ones are
  pinned down.
- Do **not** try Atari / MuJoCo / pixel inputs.
- Do **not** continue exploring `broadness` as a continuous knob —
  the 3-seed evidence does not distinguish `0.0`, `0.2`, and `0.9`
  and the mechanism is confusing to explain. Pick `0.0` and move on.


## 9. What to do with `rl_notes.md`

`rl_notes.md` is a useful working diary but it currently reads as
load-bearing documentation. Specifically it presents single-seed
numbers as established facts (§5 summary table, §11.6 "4000n focused
beats DQN", §11.9 summary). After Phase A lands, it should be cut
down to:

- **§1–3** (concept, architecture, target modes): keep.
- **§4.x (results)**: replace all single-seed tables with 5-seed
  tables from Phase A–C. Delete §4.4 (non-monotonic neuron scaling)
  since it is a single-seed artifact. Reframe §4.7 as "ensembles are
  promising but unverified" until Phase C.
- **§5 summary**: replace with the Phase A/C verified numbers.
- **§6 analysis**: keep.
- **§7 related work**: keep.
- **§8 open questions**: most are closed; condense.
- **§9 sweep**: reframe as "ablation evidence for the base config",
  drop single-seed headline claims.
- **§10 neuron scaling**: delete (single-seed).
- **§11 recentering investigation**: condense to "the mechanism is
  rotation of the bottom percentile; broadness is a regularizer on
  the activity-tracking memory; `min_relative` is the way to make
  recentering self-terminate". Drop the "broad vs focused at 8000n"
  saga — it was a single-seed artifact.

This document (`rl_review.md`) is intentionally orthogonal and can
live alongside `rl_notes.md` until the rewrite happens.


## 10. Closing

The methodological sin here was accumulating single-seed headline
numbers and treating them as strong evidence as the implementation
was still being fixed (the recentering "bug" saga, the alive-only
fix, the broadness redefinition). Once you replot everything from
the actual JSON, the picture is a normal research picture: a
reasonable method in the 70–80% range of a strong baseline, with
large seed variance, lots of knobs that don't matter, and a couple
that do. That is still a paper. It is just not the paper we have
been telling ourselves we have.

If we commit to Phase A–D above, with multi-seed discipline, the
story is honest and the contribution is clear. If we keep chasing
single-seed peaks against DQN, the method is going to keep looking
just out of reach, because it actually is.
