# Sparse RL sweeps

This document tracks the first clean sparse-reward experiments for the NEF-RL
branch. These runs use the current `results/rl_clean/base_config.json` as the
dense-control starting point, but they write to the independent sparse namespace:

- `results/rl_sparse/`
- `result_namespace = "rl_sparse"`
- `runner_version = "rl-sparse-v1"`

The main question is whether sparse-reward failure is primarily a target-signal
problem and whether NEF-native dense auxiliary targets can help without turning
the method into a gradient-trained deep RL pipeline.

## Current plan status

The first implementation pass is complete:

- sparse-reward diagnostics are in place
- finite-horizon successor features are implemented and tested
- event-triggered finite traces are implemented and tested
- leverage novelty from normal-equation statistics is implemented and tested
- MountainCar and Acrobot hindsight/goal helper functions are implemented
- P1/P2 sparse sweeps completed with restartable drivers and clean metadata

The remaining work is not "finish the first plan" so much as choosing the next
research branch from the evidence below. The two branches are now clearly
different:

- **Acrobot**: successor features work, but stability and retention need work.
- **MountainCar**: the current mechanisms do not discover the sparse event at
  all, so the next step must be task-structured discovery/progress targets, not
  another small novelty-scale tweak.

## Reproducibility notes

Sparse runs use the same seed discipline as the clean RL restart:

- warmup resets: `100000 + seed + episode`
- training resets: `seed + episode`
- evaluation resets: `10000 + episode`
- reset seeds also seed the environment action space
- resumable results are reused only when namespace, runner version, full config,
  seed, seed policy, code version, and run context match

All P1/P2 result files below have zero error records. Raw `.log` files are
ignored because the JSON files contain the reproducible run records and the logs
are transient monitoring output.

## Phase P3 -- successor-readout geometry follow-up

Result files:

- `results/rl_sparse/geometry/sparse_chain_geometry.json`
- `results/rl_sparse/geometry_gpu/mountaincar_dot_n2000.json`
- `results/rl_sparse/geometry_gpu/mountaincar_cosine_n2000.json`
- `results/rl_sparse/geometry_gpu/mountaincar_future_cosine_n2000.json`

Design:

- Environment: `MountainCar-v0`
- Variant: full successor hindsight readout (`nef_sf_hindsight`)
- Seeds: 0-9
- Episodes: 300
- Evaluation: every 25 episodes, 5 deterministic eval episodes
- Width: 2000 neurons
- GPU runner code identity: `14f02d76a16aff1490f08fe12b5a28302b6896a0`
- Readout modes: original dot product, cosine-normalized successor, and
  cosine-normalized future increment.

| Readout mode | n | robust_eval mean +/- SE | best_eval mean +/- SE | final_eval mean +/- SE | Mean row time |
|---|---:|---:|---:|---:|---:|
| `dot` | 10 | -158.8 +/- 1.4 | -149.4 +/- 3.3 | -151.3 +/- 2.1 | 20.8 s |
| `cosine` | 10 | -200.0 +/- 0.0 | -200.0 +/- 0.0 | -200.0 +/- 0.0 | 22.4 s |
| `future_cosine` | 10 | -200.0 +/- 0.0 | -200.0 +/- 0.0 | -200.0 +/- 0.0 | 22.4 s |

Takeaways:

- The original dot-product readout exactly preserves the earlier
  MountainCar full-successor positive.
- Normalizing the successor vector is not a repair for MountainCar.  Both
  normalized modes collapse to the no-progress return across all seeds.
- The SparseChain magnitude artifact remains useful as a diagnostic, but
  on MountainCar successor magnitude appears to carry behaviorally useful
  temporal structure.

## Phase P1 -- mechanism screen

Result file: `results/rl_sparse/sparse_p1_mechanisms.json`

Design:

- Environments: `MountainCar-v0`, `Acrobot-v1`
- Seeds: 0, 1, 2
- Episodes: 300
- Evaluation: every 50 episodes, 10 deterministic eval episodes
- Workers/threads: 3 workers x 2 BLAS/Torch threads
- Mechanisms: dense FQI baseline, explicit success reward, event traces,
  successor features, and successor features with leverage novelty
- Dense-control base: `results/rl_clean/base_config.json` at run time
  (`n_neurons=6000`, `gamma=0.98`, `n_step=100`, `forget_factor=0.999`,
  `recenter_interval=null`)

| Env | Config | n | robust_eval mean +/- SE | best_eval mean | final_eval mean | final success |
|---|---|---:|---:|---:|---:|---:|
| `Acrobot-v1` | `fqi_event_trace_success` | 3 | -489.3 +/- 7.0 | -444.6 | -500.0 | 0.000 |
| `Acrobot-v1` | `fqi_nstep_env` | 3 | -491.3 +/- 8.7 | -459.1 | -499.1 | 0.033 |
| `Acrobot-v1` | `fqi_nstep_success` | 3 | -489.3 +/- 7.0 | -444.6 | -500.0 | 0.000 |
| `Acrobot-v1` | `sf_novelty_success_k1000` | 3 | -433.2 +/- 66.8 | -404.2 | -414.5 | 0.333 |
| `Acrobot-v1` | `sf_success_k1000` | 3 | -392.9 +/- 94.6 | -288.3 | -380.4 | 0.367 |
| `MountainCar-v0` | `fqi_event_trace_success` | 3 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |
| `MountainCar-v0` | `fqi_nstep_env` | 3 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |
| `MountainCar-v0` | `fqi_nstep_success` | 3 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |
| `MountainCar-v0` | `sf_novelty_success_k1000` | 3 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |
| `MountainCar-v0` | `sf_success_k1000` | 3 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |

Takeaways:

- Acrobot shows a real successor-feature signal: both successor variants beat
  all FQI/event-trace variants by a large margin.
- Event traces did not separate from the explicit success-reward FQI baseline.
- MountainCar remained fully flat across all P1 mechanisms. The blocker there
  appears to be reward discovery/exploration, not only reward propagation after
  discovery.

The decisive diagnostic difference is that Acrobot produces some successful
episodes under random/epsilon exploration, so a better target can exploit those
events. MountainCar did not produce any success events in P1; all return/target
statistics collapse to the maximum-length failure value.

## Phase P2 -- successor-feature refinement and MountainCar novelty

Acrobot result file: `results/rl_sparse/sparse_p2_acrobot_sf.json`

MountainCar result file: `results/rl_sparse/sparse_p2_mountaincar_explore.json`

Config files:

- `results/rl_sparse/configs/p2_acrobot_sf.json`
- `results/rl_sparse/configs/p2_mountaincar_explore.json`

Supervisor: `results/rl_sparse/p2_supervisor.sh`

Design:

- Acrobot: 5 seeds, 500 episodes, successor-feature horizon/capacity/novelty
  ablations.
- MountainCar: 6 seeds, 1000 episodes, stronger leverage novelty and slower
  epsilon decay.
- Workers/threads after P1 handoff: two concurrent sweeps, each with 2 workers x
  2 BLAS/Torch threads, for about 8 compute threads on the 12-thread host.

| Env | Config | n | robust_eval mean +/- SE | best_eval mean | final_eval mean | final success |
|---|---|---:|---:|---:|---:|---:|
| `Acrobot-v1` | `sf_h100_k1000` | 5 | -344.9 +/- 74.9 | -238.3 | -296.7 | 0.580 |
| `Acrobot-v1` | `sf_h100_k1000_nov003` | 5 | -331.6 +/- 69.4 | -269.9 | -319.8 | 0.540 |
| `Acrobot-v1` | `sf_h100_k1000_nov01` | 5 | -260.3 +/- 62.0 | -227.0 | -239.4 | 0.800 |
| `Acrobot-v1` | `sf_h100_k2000` | 5 | -400.8 +/- 39.2 | -348.5 | -394.8 | 0.400 |
| `Acrobot-v1` | `sf_h200_k1000` | 5 | -351.0 +/- 89.1 | -270.5 | -349.7 | 0.400 |
| `Acrobot-v1` | `sf_h50_k1000` | 5 | -449.3 +/- 50.7 | -325.7 | -447.8 | 0.160 |
| `MountainCar-v0` | `sf_h100_k1000_nov03_sloweps` | 6 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |
| `MountainCar-v0` | `sf_h100_k1000_nov1_sloweps` | 6 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |
| `MountainCar-v0` | `sf_h100_k1000_nov3_sloweps` | 6 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |
| `MountainCar-v0` | `sf_h200_k1000_nov1_sloweps` | 6 | -200.0 +/- 0.0 | -200.0 | -200.0 | 0.000 |

Takeaways:

- Acrobot P2 confirms successor features as the useful sparse-reward mechanism.
  The best P2 setting is `sf_h100_k1000_nov01`, with robust eval -260.3 +/- 62.0
  and final success 0.800. This is far above the P1 FQI/event-trace range
  around -489 to -491 robust eval.
- The best Acrobot run family still has high seed variance, so the next Acrobot
  step should focus on stability and retention rather than only improving peak
  eval. Promising knobs are mild behavior-only leverage novelty, reward-decoder
  regularization, and target/normal-equation forgetting.
- Increasing to 2000 neurons did not help in this setting. Capacity is not the
  first bottleneck for Acrobot.
- MountainCar remained completely flat even with stronger leverage novelty and
  slower epsilon decay. This strongly suggests that MountainCar needs a
  task-structured discovery mechanism, such as goal/progress targets or
  hindsight goals, before successor features can propagate useful sparse events.

### Acrobot seed-level behavior

The aggregate Acrobot numbers hide a useful pattern: many seeds discover
successful trajectories, but some configurations lose evaluation performance
after early success. That makes retention/stability the next problem.

For the best complete family, `sf_h100_k1000_nov01`:

| Seed | robust_eval | best_eval | final_eval | final success | first success episode | training successes |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -500.0 | -500.0 | -500.0 | 0.0 | 89 | 260 |
| 1 | -244.8 | -198.8 | -239.7 | 1.0 | 89 | 332 |
| 2 | -148.0 | -126.8 | -129.3 | 1.0 | 61 | 372 |
| 3 | -216.5 | -172.3 | -191.1 | 1.0 | 7 | 255 |
| 4 | -192.2 | -136.9 | -136.9 | 1.0 | 4 | 351 |

This is encouraging because four of five seeds converge to useful evaluation
policies. The failed seed is not a pure discovery failure: it still had 260
successful training episodes, but the learned evaluation policy remained at
-500.0. That points away from simple event discovery and toward decoder
stability, action-value ranking, or reward-map calibration after events are
found.

The plain `sf_h100_k1000` family has a similar split: seeds 2 and 3 reach
strong final policies, seed 4 is useful but weaker, and seeds 0/1 underperform
despite many training successes. The longer `sf_h200_k1000` setting can produce
excellent runs (seed 2 final eval -107.3), but it also collapses in more seeds,
so the extra horizon seems to increase variance more than reliability.

### MountainCar diagnostic conclusion

P2 tested novelty bonuses 0.3, 1.0, and 3.0 with slow epsilon decay, plus a
200-step successor horizon. Across all 24 MountainCar runs:

- no run had a success event
- every robust, best, and final evaluation was exactly -200.0
- every final success rate was 0.0

This is stronger evidence than "the value method is weak." With no successful
events, the reward decoder has no positive labels and the successor-feature
composition cannot assign value to the goal. Leverage novelty over fixed
state-action features did not create the momentum-building exploration pattern
needed by MountainCar, even at large behavior-only scales.

For MountainCar, the next mechanism should deliberately create dense
task-relevant supervised signal while remaining NEF-native. Good candidates are:

- achieved-goal/hindsight heads over position or height
- progress/event-proximity decoders for max-position improvement
- trajectory-window classifiers for "did this window increase reachable energy?"
- frontier-centered feature allocation around high-position/high-velocity states

Blindly increasing novelty scale, neuron count, or successor horizon is unlikely
to help without one of those discovery/progress signals.

## Mechanistic interpretation

P1/P2 support the original sparse-reward diagnosis with more precision:

1. When sparse events are occasionally discovered, successor features help
   because every transition trains a dense future-occupancy target. The sparse
   reward map can then compose with learned reachability instead of asking the
   Q decoder to infer long-horizon credit from mostly zero labels.
2. Event-triggered traces are not enough by themselves on these runs. They spread
   rare labels backward, but they do not create a dense model of what states are
   reachable from each action.
3. Leverage novelty is useful but only in the right regime. Mild Acrobot novelty
   (`0.1`) improves the best family; stronger MountainCar novelty still fails
   because the exploration problem is structured, not just under-sampled.
4. More capacity is not the first Acrobot bottleneck: `sf_h100_k2000` was worse
   than `sf_h100_k1000`. The current bottleneck looks like stability and
   calibration of solved successor/reward decoders.

## Next experiment branches

### P3A -- Acrobot stability

Keep `sf_h100_k1000_nov01` as the incumbent sparse Acrobot candidate. The next
Acrobot sweep should focus on reliability:

- repeat `sf_h100_k1000_nov01` with more seeds to estimate variance
- compare novelty 0.05, 0.1, and 0.2 around the current winner
- test reward-decoder regularization separately from successor-decoder
  regularization
- test slower or no forgetting for the reward map while keeping successor
  statistics adaptive
- log per-action Q margins at evaluation time so failed seeds can be separated
  into "wrong ranking" versus "all actions flat"

Adoption should require a current-control row in the same run context and should
prefer robust/final eval over isolated best eval.

### P3B -- MountainCar discovery

Do not spend the next MountainCar phase on more blind novelty scaling. The next
MountainCar sweep should introduce a task-structured NEF target:

- hindsight goals from achieved max position
- progress targets such as future max position or future height over a finite
  window
- a window-pooled success/progress classifier inspired by `StreamingNEFClassifier`
- optional frontier-centered features once progress/frontier states are logged

The key adoption metric for this phase is not final score at first. The first
gate is whether any configuration produces nonzero success events or a non-flat
progress/evaluation curve across seeds.

### Dense-control check

Before treating the Acrobot candidate as a general RL improvement, rerun a small
LunarLander-v3 dense-control check against the current `rl_clean` base. That
check should use the clean base available at run time and should not write into
`results/rl_clean/` unless it is part of the clean adoption process.

## Longer-horizon future work

These ideas are not immediate P3 requirements, but they fit the evidence and the
project's core style better than adding a gradient-trained critic.

### Stabilize solved successor control

The Acrobot failures after many successful training episodes suggest that the
reward map and successor decoder can discover useful structure but still rank
actions poorly at evaluation time. Possible NEF-native stabilizers:

- **Separate regularization for successor and reward decoders.** P2 used one
  `alpha` for both. The reward map is a sparse scalar regression problem, while
  successor prediction is a dense multi-output problem; tying their
  regularization may be unnecessarily brittle.
- **Reward-target normalization or class-balanced reward solves.** Sparse reward
  labels are extremely imbalanced. Solving the reward map with balanced event
  weighting or a normalized event-probability target may preserve the analytical
  solve while reducing all-zero dominance.
- **Action-margin diagnostics and conservative action selection.** Failed seeds
  should log whether the greedy action barely beats alternatives or whether the
  wrong action has a large margin. If margins are tiny, averaging, abstaining, or
  action-value smoothing may help without changing the learning rule.
- **Two-timescale statistics.** Keep successor statistics broad and slowly
  forgotten, but let the reward map adapt on a shorter or differently
  regularized timescale once events appear.
- **Decoder ensembles over the same accumulated features.** Multiple solved
  reward maps or successor decoders with different seeds/regularization could
  vote or average at action selection time, analogous to existing NEF ensembles
  but cheaper than training multiple full agents.

### Make goal discovery a supervised NEF problem

MountainCar needs a discovery signal before successor features have anything to
compose with. Future MountainCar work should turn progress into a supervised
target rather than hoping novelty finds the swing-up pattern:

- **Future max-position decoders.** Regress the maximum position reached over the
  next `K` steps from each state-action feature. This keeps the target dense and
  task-relevant while avoiding hand-shaped reward in the control objective.
- **Energy/height progress heads.** MountainCar and Acrobot both have simple
  physical progress coordinates. Predicting future height or energy could serve
  as a learned option score, then sparse success remains the final external
  criterion.
- **Hindsight goal heads.** Treat achieved max positions as goals after each
  episode and solve `(state, goal, action) -> reached-within-K` decoders. This is
  a natural multi-goal extension of the already implemented hindsight helpers.
- **Frontier replay without SGD.** Maintain additive sufficient statistics for
  frontier states separately from the main stream, so high-position or
  high-energy transitions are not washed out by thousands of flat failures.
- **Window-pooled trajectory classifiers.** Use the streaming classifier idea
  directly: encode short action/state windows, pool activities, and solve whether
  the window increased future reachable height. This may be a better fit for
  momentum-building tasks than pointwise Q regression.

### Use successor features as an analytical model, not only a value head

Once successor features are learned, they can support more than
`Q = psi @ reward_weight`:

- **One-step lookahead in successor space.** Use solved successor predictions to
  compare how actions change future occupancy, even before the sparse reward map
  is reliable.
- **Goal-conditioned reward maps.** Keep one successor decoder but solve many
  lightweight reward/goal maps on top, similar to continual-learning additive
  heads.
- **Option discovery from successor geometry.** Cluster or factor successor
  vectors to find temporally coherent regions/options. This would be a
  fixed-feature analogue of skill discovery without training a policy network.
- **Successor reconstruction diagnostics.** Track whether failed runs have poor
  successor prediction or good successor prediction with poor reward/action
  ranking. That distinction should guide whether to improve temporal targets or
  the reward head.

### Bring continual-learning accumulation ideas into RL more explicitly

The CL results suggest that additive sufficient statistics are not just an
implementation detail; they can be the method. Sparse RL could use that more
directly:

- **Per-event or per-goal normal equations.** Store separate `A^T A`/`A^T Y`
  blocks for discovered goals or event types, then combine decoders at decision
  time instead of forcing one global head to represent every sparse objective.
- **Conflict-aware decoder merging.** When a new sparse event class appears,
  measure whether its decoder conflicts with earlier behavior and merge with
  CL-style regularized accumulation rather than overwriting the global value
  head.
- **Accumulated prototypes for rare events.** Keep compact NEF prototype
  statistics around successful trajectories and use them to seed centers or
  goal heads in later runs.

### Evaluation improvements

Future sweeps should add diagnostics that make the next failure mode visible
without needing to inspect every log:

- first success episode and total training successes in summary tables
- final action-margin and Q-spread statistics
- reward-map positive/negative calibration curves
- successor prediction error on held-out episode suffixes
- novelty score at frontier, goal, and ordinary failure states
- per-seed sparkline or small JSON summary of eval history shape

These are especially important because Acrobot now has real but high-variance
signal. The next adoption decision should distinguish "occasionally excellent"
from "reliably useful."
