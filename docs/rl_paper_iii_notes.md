# Paper III RL research notes

These notes consolidate the dense-reward `rl_clean` line and the
sparse-reward `rl_sparse` line for the third SSUP paper. They are a
working research record, not manuscript prose: chronology, internal
paths, branch names, and caveats are allowed here so that the eventual
paper can stay clean and self-contained.

## 1. Primary source map

### Dense reward: session checkpoints

The dense RL work was done in session
`2b999ca8-cdfb-4414-8129-7a8d50921b96`. Checkpoints `012` through `042`
are load-bearing:

- `012`-`020`: prototype NEF-FQI, TD/FQI instability, MC and n-step
  target design, CartPole success, LunarLander partial success, and the
  first RLS/recentering ideas.
- `021`-`030`: pre-clean exploratory sweeps, single-seed spikes,
  recentering/broadness confusion, rotation-vs-dead-neuron audit, and
  why old `results/rl/` rows are design priors rather than evidence.
- `031`-`042`: reproducibility reset, clean seed policy, clean adoption
  rule, and the final dense `rl_clean` backbone.

The clean evidence should be treated as starting only after
`037-rl-reproducibility-reset.md`.

### Sparse reward: session checkpoints

The sparse RL work was done in session
`2521aa3c-1ef1-44bb-a267-ea370b4492f0`. Checkpoints `001` through `009`
are load-bearing:

- They record the mechanism ladder: sparse diagnostics, toy chain,
  successor features, leverage novelty, hindsight goal conditioning,
  and eval-goal projection.
- They record why sparse reward is a target-signal problem in this
  program: the fixed encoder can be adequate while `A^T Y` remains flat
  or absent.
- They record operational constraints: successor features were run at
  2000 neurons because per-action `Psi` scales quadratically in neuron
  count.

### Repo artifacts now live again

Dense clean artifacts:

- `docs/rl_clean_sweeps.md`
- `docs/generate_rl_clean_figures.py`
- `docs/figures/rl_clean_*.png`
- `results/rl_clean/`

Sparse artifacts from `sparse-reward-c`:

- `docs/rl_sparse_notes.md`
- `benchmarks/run_rl_sparse.py`
- `benchmarks/run_rl_diag.py`
- `benchmarks/run_sparse_chain.py`
- `src/leenef/rl_diag.py`
- `src/leenef/rl_sf.py`
- `tests/test_rl_diag.py`
- `tests/test_rl_hindsight.py`
- `tests/test_rl_sf.py`
- `tests/test_rl_sf_agent.py`
- `results/rl_sparse/*` exploratory hindsight/SF result JSONs

Unique sparse artifacts imported from `sparse-reward-g`:

- `docs/rl_sparse_sweeps.md`
- `results/rl_sparse/sparse_p1_mechanisms.json`
- `results/rl_sparse/sparse_p2_acrobot_sf.json`
- `results/rl_sparse/sparse_p2_mountaincar_explore.json`
- `results/rl_sparse/configs/p2_acrobot_sf.json`
- `results/rl_sparse/configs/p2_mountaincar_explore.json`
- `results/rl_sparse/p2_supervisor.sh`

Older pre-clean RL artifacts remain in `attic/docs/rl_notes.md`,
`attic/docs/rl_review.md`, `attic/docs/rl_multiseed_sweeps.md`, and
`attic/results/rl/`. Use them for mechanism history and failure
analysis only.

## 2. Dense reward evidence inventory

The dense reward line asks whether the same analytic readout that works
for supervised and continual learning can be used for control when the
targets are policy-induced returns.

### 2.1 Mechanism conclusion

Bootstrapped TD targets are the wrong first target for this analytic
solver. SGD can track moving targets by small steps; the closed-form
solve replaces the decoder with the global optimum for the current
target matrix. If the bootstrap targets shift between solves, the policy
can oscillate. MC and n-step targets remove the moving-target loop and
turn the problem back into stable regression on completed trajectories.

The clean dense-control configuration therefore uses n-step returns,
decayed sufficient statistics, and trace-scaled regularization:

```json
{
  "n_neurons": 6000,
  "gamma": 0.98,
  "alpha": 0.1,
  "epsilon_decay": 500,
  "solve_every": 10,
  "target_mode": "nstep",
  "n_step": 100,
  "forget_factor": 0.999,
  "broadness": 0.0,
  "recenter_interval": null
}
```

### 2.2 Clean phase summary

| Phase | Result | Evidence status |
|---|---|---|
| P1 gamma | `gamma=0.98` adopted; robust eval `183.0 +/- 11.6` over 5 seeds. | Paper-grade phase decision. |
| P3b horizon x capacity | `n100_k6000 = 216.7 +/- 6.2` robust eval; adopted over `4000@50` by `+33.7` with combined SE `13.2`. | Main dense headline candidate. |
| P4 cadence x forgetting | Kept `solve_every=10`, `forget_factor=0.999`; nearest runner-up did not clear adoption. | Paper-grade phase decision. |
| P5 epsilon | Kept `epsilon_decay=500`; slower/faster schedules collapsed or weakened. | Paper-grade phase decision. |
| P8 ensemble pilot | Thompson-trained voting readout `218.8 +/- 8.9`; only `+2.1` over single-agent incumbent with combined SE `10.9`. | Useful non-adoption result. |
| P8b voting exploration | Voting exploration failed, driven by catastrophic seed 0. | Useful negative result. |
| P7 final confirmation | Incumbent duplicate reproduced exactly; H2 runner-up tied but did not beat. | Determinism sentinel and final-base confirmation. |

### 2.3 Paper dense-control chain

The Paper III dense chain completed under one code identity,
`12385951c0a1898ff01c9f8e9730f91db0246d01`, with all planned rows
written under `results/rl_paper/dense/`.

| Panel | n | robust eval mean +/- SE | final eval mean +/- SE | best eval mean +/- SE | Interpretation |
|---|---:|---:|---:|---:|---|
| CartPole NEF MC sanity | 10 | `409.0 +/- 20.4` | `428.4 +/- 19.0` | `499.4 +/- 0.6` | Sanity anchor: the analytic control path reliably solves or nearly solves CartPole. |
| LunarLander n-step, no bootstrap | 5 | `-38.4 +/- 18.0` | `24.7 +/- 32.3` | `33.4 +/- 20.5` | Stable targets help but this small panel is not the final dense configuration. |
| LunarLander MC, no bootstrap | 5 | `-67.7 +/- 32.3` | `5.6 +/- 38.2` | `43.6 +/- 39.0` | MC-only targets are weaker and high variance. |
| LunarLander TD bootstrap | 5 | `-123.8 +/- 13.1` | `-72.4 +/- 3.8` | `-71.5 +/- 4.2` | Mechanistic negative: bootstrapped moving targets collapse for this analytic solve. |
| LunarLander in-repo DQN context | 10 | `129.3 +/- 21.7` | `149.6 +/- 14.0` | `196.3 +/- 14.1` | Useful matched-seed context, not a broad claim against DQN. |
| LunarLander final NEF-FQI | 10 | `201.3 +/- 9.8` | `210.0 +/- 8.4` | `216.2 +/- 9.9` | Central dense-control result: paper-grade 10-seed confirmation. |
| LunarLander Thompson ensemble, 3 members | 3 | `200.0 +/- 12.8` | `173.1 +/- 29.6` | `210.3 +/- 12.9` | Does not improve over the single final NEF-FQI row. |
| LunarLander voting ensemble, 3 members | 3 | `64.5 +/- 81.4` | `62.6 +/- 79.3` | `113.3 +/- 93.3` | Non-adoption: voting exploration is brittle, including one catastrophic seed. |

Dense interpretation:

- The dense-control headline should be the 10-seed final NEF-FQI
  LunarLander panel, supported by the bootstrap negative mini-panel.
- The DQN panel can be used as in-repository context: the final NEF-FQI
  row is clearly stronger here, but the paper should avoid implying a
  general DQN benchmark claim.
- Ensembles should be a closure/negative result, not a promoted
  improvement. Thompson exploration approximately matches the single
  agent; voting exploration is unstable.

### 2.4 Dense remaining cautions

- Treat the DQN row as context for this implementation and compute
  budget, not as a literature-level baseline sweep.
- The final dense result is strong enough for Paper III. Additional
  dense extensions should only be run if a manuscript argument genuinely
  needs them.
- The bootstrap negative is central because it explains *why* the final
  n-step target construction matters.

## 3. Sparse reward evidence inventory

The sparse reward line asks what happens when the return target is
mostly absent. The answer is not "increase the fixed feature map"; it is
"construct a better target while staying inside the sufficient-statistic
program."

### 3.1 Mechanism stack

`D1`: NEF successor features. The same fixed encoder `phi(s)` is used,
but the target is the n-step future occupancy vector
`sum_k gamma^k phi(s_{t+k})`. A second analytic solve produces
successor decoders `Psi_a`; a reward map `w_a` maps features to reward.
Then `Q(s,a) = psi(s,a) dot w_a`.

`D2`: leverage novelty. The inverse of the regularized normal matrix
already exists or can be recomputed from `A^T A`. The quadratic form
`phi^T (A^T A + alpha I)^-1 phi` exposes under-sampled regions without
an auxiliary network.

`D4`: hindsight goal conditioning plus eval-goal projection. The agent
learns a successor backbone over goal-conditioned behavior. At
evaluation, the task goal is projected into feature space and used as a
readout direction for `psi`, with no extra solve.

### 3.2 Sparse-reward-c notes outcome

From `docs/rl_sparse_notes.md`:

| Environment | Main result | Evidence status |
|---|---|---|
| MountainCar-v0 | Plain FQI, SF, SF+novelty, and hindsight without eval-goal stay at `-200`; hindsight + `eval_goal=(0.5, 0.0)` improves all 3 seeds. | Strong mechanism evidence; needs replication for paper headline. |
| Acrobot-v1 | Hindsight + eval-goal at `ff=0.999` finds signal but regresses; `ff=0.9999` gives stable useful runs across 3 seeds. | Strong mechanism evidence; retention boundary needs mapping. |
| LunarLander-v3 | Hindsight/eval-goal hurts dense control at 2000 neurons. | Important negative: sparse mechanism is not a dense default. |
| MiniGrid-DoorKey-5x5 | Random-direction goal is stuck at zero reward. | Clean negative until a programmatic goal-observation handle exists. |
| SparseChain | SF+novelty rescues the harder chain; eval-goal fails because successor-vector magnitude swamps direction when action horizons differ. | Useful toy mechanism diagnostic. |

### 3.3 Sparse-reward-g sweep outcome

From `docs/rl_sparse_sweeps.md`:

- P1 mechanism screen confirms that Acrobot has occasional sparse events
  for successor features to exploit, while MountainCar has no success
  events under the tested random/epsilon behavior.
- P2 Acrobot refinement shows `sf_h100_k1000_nov01` as the best P2
  family: robust eval `-260.3 +/- 62.0`, final success `0.800` over
  5 seeds. The high variance and a failed seed despite many successful
  training episodes point to decoder stability/action-ranking, not
  simple discovery.
- P2 MountainCar remains exactly flat across all 24 runs, reinforcing
  that it needs task-structured discovery/progress targets rather than
  more blind novelty scale.

These P1/P2 rows are valuable because they predate the eval-goal
breakthrough and show why the later mechanism was necessary.

### 3.4 GPU directional successor panel

The Colab G4 directional successor panel completed under code identity
`12385951c0a1898ff01c9f8e9730f91db0246d01`, using `n_neurons=6000`,
`n_step=100`, `alpha=0.1`, `solve_every=10`, CUDA tensors, and skipped
end-of-run eigensolve diagnostics. Directional readout is the
known-goal, memory-reduced variant; it is not a full replacement for
post-hoc arbitrary goal projection.

| Environment / cell | n | robust eval mean +/- SE | final eval mean +/- SE | best eval mean +/- SE | Interpretation |
|---|---:|---:|---:|---:|---|
| MountainCar directional, `ff=0.999` | 10 | `-197.0 +/- 1.7` | `-195.1 +/- 2.7` | `-186.8 +/- 6.1` | Does not reproduce the earlier 2000-neuron full-successor eval-goal result as a headline. A few seeds show partial progress, but most remain flat. |
| Acrobot directional, `ff=0.9999` | 10 | `-194.9 +/- 10.4` | `-193.0 +/- 23.2` | `-132.1 +/- 6.1` | Strong sparse-control positive for the directional known-goal variant. |
| Acrobot scan, `ff=0.999` | 5 | `-296.6 +/- 28.5` | `-268.4 +/- 41.8` | `-158.6 +/- 21.4` | Too much forgetting substantially hurts retention. |
| Acrobot scan, `ff=0.9995` | 5 | `-190.6 +/- 7.0` | `-201.5 +/- 30.3` | `-128.8 +/- 7.5` | Near the stable regime. |
| Acrobot scan, `ff=0.9999` | 5 | `-184.2 +/- 5.8` | `-166.8 +/- 17.5` | `-128.0 +/- 5.3` | Best scan setting; these rows duplicate the first five seeds of the 10-seed `ff=0.9999` panel. |

Immediate interpretation:

- Acrobot is now the paper-grade sparse-reward positive: the
  known-goal directional successor readout gives stable 10-seed signal
  at full 6000-neuron width, and the forget-factor scan explains the
  retention boundary.
- MountainCar should not be promoted as a directional full-width
  headline. The failed directional replication is itself informative:
  the earlier full-successor/random-direction behavior does not transfer
  to the cheaper known-goal directional regime.
- The CUDA path solves the OOM/feasibility problem for directional
  successors. Full successor matrices remain a separate quadratic-memory
  mechanism question.

### 3.5 MountainCar full-successor follow-up

The Colab follow-up reran the earlier positive MountainCar mechanism
rather than the directional known-goal shortcut: full successor readout
(`nef_sf_hindsight`), `goal_mode=random_direction`,
`goal_action_mode=commit`, `eval_goal=(0.5, 0.0)`, `novelty_beta=0.3`,
`goal_buffer_size=512`, `goal_horizon=25`, and `forget_factor=0.999`.
Both reruns used CUDA and code identity
`12385951c0a1898ff01c9f8e9730f91db0246d01`.

| Environment / cell | n | robust eval mean +/- SE | final eval mean +/- SE | best eval mean +/- SE | Mean row time | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| MountainCar full successor, random direction | 10 x 2000 | `-158.8 +/- 1.4` | `-151.3 +/- 2.1` | `-149.4 +/- 3.3` | `20.8 s` | Replicates the earlier positive across 10 seeds. |
| MountainCar full successor, random direction | 10 x 6000 | `-158.4 +/- 4.8` | `-155.0 +/- 7.1` | `-154.4 +/- 5.3` | `119.0 s` | No improvement from widening; more expensive and higher variance. |

Interpretation:

- MountainCar was not falsified by the failed directional 6000-neuron
  run. It depends on the full-successor/random-direction behavior
  mechanism, which provides exploration structure that the directional
  known-goal panel intentionally removed.
- The best paper-facing MountainCar evidence is the 2000-neuron
  full-successor panel: it is cheaper, more stable, and essentially tied
  with the 6000-neuron rerun.
- This creates a useful sparse contrast: Acrobot works well with the
  memory-reduced directional known-goal readout, while MountainCar needs
  the richer full-successor/random-direction mechanism.

### 3.6 Sparse gaps

- Present MountainCar carefully: it is a positive for full successor
  readout plus random-direction behavior, and a negative for the cheaper
  directional known-goal shortcut.
- Full successor features are feasible on Colab/GPU at 6000 neurons for
  MountainCar, but the cost is about 5.7x the 2000-neuron version with
  no accuracy gain in this panel. Treat widening as a cost/variance
  boundary, not an automatic improvement.
- Test magnitude-normalized eval-goal projection. The SparseChain
  failure mechanism is concrete enough that this is the cheapest likely
  new insight.
- Preserve the Acrobot forget-factor boundary as a result, not only a
  tuning note: `0.999` is clearly worse than `0.9995`/`0.9999` in the
  directional full-width panel.
- Decide whether MiniGrid gets a goal-observation follow-up or remains a
  negative boundary result.

## 4. Unified Paper III mechanism story

Papers I and II establish that `A^T A` and `A^T Y` are enough for fixed
feature supervised and continual learning. Paper III changes the nature
of `Y`.

In dense-control RL, the main problem is target stability. MC/n-step
returns produce a target matrix that the analytic solve can fit without
chasing its own bootstrap. The successful dense line is therefore not a
trick for exploration; it is a disciplined way to make the target stable
enough for a one-shot solve.

In sparse-control RL, the main problem is target absence. If the reward
rarely fires, `A^T Y` is zero or nearly constant and the analytic solve
correctly learns a useless value function. Successor features and
hindsight/eval-goal do not change the fixed encoder; they change what
gets written into `Y` so that useful structure exists before the sparse
extrinsic reward has propagated.

This gives Paper III a coherent arc:

1. Dense reward teaches us to avoid moving bootstrap targets.
2. Sparse reward teaches us to construct dense auxiliary target
   structure without abandoning analytic sufficient statistics.
3. The program succeeds when target construction is well matched to the
   task and fails cleanly when the target handle is missing or the
   successor readout assumptions break.

## 5. Remaining experiment priorities

Cost is qualitative and relative to this project. "Interesting" means
the result would change the story, sharpen a boundary, or make a claim
paper-grade.

| ID | Experiment | Interesting likelihood | Cost | Decision role |
|---|---|---|---|---|
| RL-D0 | Audit restored dense rows and current metadata/code identity. | Very high | Small | Completed: all dense-chain rows use identity `12385951c0a1898ff01c9f8e9730f91db0246d01`. |
| RL-D1 | Extend/confirm final LunarLander base. | High | Medium | Completed: 10-seed final NEF-FQI robust eval `201.3 +/- 9.8`. |
| RL-D2 | Clean matched-seed DQN/context baseline. | Medium-high | Medium | Completed as context: DQN robust eval `129.3 +/- 21.7`. |
| RL-D3 | Small TD/bootstrap negative mini-panel. | High | Small-medium | Completed: bootstrapped TD robust eval `-123.8 +/- 13.1`. |
| RL-D4 | CartPole dense sanity replication. | Medium | Small | Completed: robust eval `409.0 +/- 20.4`, best eval `499.4 +/- 0.6`. |
| RL-D5 | Ensemble/voting closure. | Low-medium | Medium | Completed: Thompson does not beat final single agent; voting is brittle. |
| RL-S0 | Audit sparse `c`/`g` rows and classify evidence status. | Very high | Small | Required before sparse claims. |
| RL-S1 | Replicate MC + Acrobot eval-goal results. | Very high | Medium-large | Completed: Acrobot directional positive; MountainCar full-successor positive. |
| RL-S2 | Successor-feature memory feasibility / compression sketch. | High | Medium | Directional CUDA path works; full successor works at 6000 on GPU but has clear cost/variance limits. |
| RL-S3 | Magnitude-normalized eval-goal projection. | High | Small | Cheapest likely mechanism win. |
| RL-S4 | Acrobot forget-factor scan. | Medium-high | Small-medium | Completed: `0.999` is much worse than `0.9995`/`0.9999`. |
| RL-S5 | Eval-goal generalization sweep. | Medium | Small | Nice figure if smooth; useful caveat if brittle. |
| RL-S6 | MiniGrid goal-observation handle. | Medium-high | Medium | Possible extension, but can become a time sink. |
| RL-S7 | SparseChain CI smoke. | Low | Small | Locks in regression tests. |
| RL-S8 | MountainCar progress-target branch. | Medium | Medium-large | Conceptually attractive, but not first unless MC becomes central. |

Current next package:

1. Write the Paper III outline around target construction:
   dense-control stability, sparse-control target absence, and
   successor/hindsight target structure.
2. Use Acrobot directional and MountainCar full-successor as the two
   sparse positives, with MountainCar directional as a boundary and
   not as a failed headline.
3. Decide whether to run the optional normalized eval-goal/SparseChain
   mechanism check before drafting, or leave it as future work.

## 6. Paper III outline seed

1. **Introduction.** RL as the paper where `Y` becomes the problem:
   returns are not labels, and policy changes the data distribution.
2. **Background.** FQI, LSTD/RLS, random features, successor features,
   HER, and intrinsic novelty. Be explicit that these families already
   exist; SSUP reframes and connects them.
3. **Dense-control method.** NEF-FQI, per-action sufficient statistics,
   trace-scaled ridge, n-step returns, RLS decay, seed policy.
4. **Dense-control results.** LunarLander final NEF-FQI, TD/bootstrap
   negative, matched in-repo DQN context, CartPole sanity, non-adopted
   ensemble result.
5. **Sparse-control method.** Successor features, leverage novelty,
   hindsight/eval-goal projection.
6. **Sparse-control results.** Acrobot directional positive,
   MountainCar full-successor positive, MountainCar directional
   boundary, SparseChain diagnostic, MiniGrid and dense-LunarLander
   negatives.
7. **Discussion.** Target construction as the RL analog of labels in SL
   and per-class additivity in CL; where the SSUP program helps and
   where it does not.
8. **Appendix.** Source trail, checkpoint references, metadata schemas,
   runner seed policy, environment versions, and result artifacts.
