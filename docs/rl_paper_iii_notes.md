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

### 2.3 Dense gaps

- The final LunarLander number is strong enough as a clean internal
  result, but if it is the central Paper III headline, extending or
  confirming it beyond the current 5-seed grid and 3-seed final
  sentinel would make the paper sturdier.
- DQN comparison language must be conservative unless we run a clean
  matched-seed baseline. The old DQN rows are context, not sufficient
  evidence for a "matches" or "beats" claim.
- The TD/bootstrap negative is mechanistically central. It may deserve
  a small clean mini-panel even if old notes already show the pattern.
- CartPole is cheap and useful as a sanity anchor, but not where the
  paper should spend its limited effort.

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

### 3.4 Sparse gaps

- Replicate MC and Acrobot eval-goal results under a final runner and
  classify whether 2000-neuron successor features are the intended
  paper setting or only a feasibility compromise.
- Test whether full-width successor features are possible. If not, make
  the quadratic-memory limit explicit and consider low-rank or compressed
  successor decoders as future work.
- Test magnitude-normalized eval-goal projection. The SparseChain
  failure mechanism is concrete enough that this is the cheapest likely
  new insight.
- Map the Acrobot forget-factor boundary. The jump from `0.999` to
  `0.9999` is too large to leave unexplained if Acrobot becomes a paper
  result.
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
| RL-D0 | Audit restored dense rows and current metadata/code identity. | Very high | Small | Required before writing dense claims. |
| RL-D1 | Extend/confirm final LunarLander base. | High | Medium | Needed if LunarLander is a headline number. |
| RL-D2 | Clean matched-seed DQN/context baseline. | Medium-high | Medium | Needed only for stronger DQN comparison language. |
| RL-D3 | Small TD/bootstrap negative mini-panel. | High | Small-medium | Supports the target-stability thesis. |
| RL-D4 | CartPole dense sanity replication. | Medium | Small | Useful anchor, not a main claim. |
| RL-D5 | Ensemble/voting closure. | Low-medium | Medium | Appendix only unless the user wants closure. |
| RL-S0 | Audit sparse `c`/`g` rows and classify evidence status. | Very high | Small | Required before sparse claims. |
| RL-S1 | Replicate MC + Acrobot eval-goal results. | Very high | Medium-large | Required for sparse headline numbers. |
| RL-S2 | Successor-feature memory feasibility / compression sketch. | High | Medium | Determines whether 2000 neurons is a principled setting or a limitation. |
| RL-S3 | Magnitude-normalized eval-goal projection. | High | Small | Cheapest likely mechanism win. |
| RL-S4 | Acrobot forget-factor scan. | Medium-high | Small-medium | Explains retention/stability boundary. |
| RL-S5 | Eval-goal generalization sweep. | Medium | Small | Nice figure if smooth; useful caveat if brittle. |
| RL-S6 | MiniGrid goal-observation handle. | Medium-high | Medium | Possible extension, but can become a time sink. |
| RL-S7 | SparseChain CI smoke. | Low | Small | Locks in regression tests. |
| RL-S8 | MountainCar progress-target branch. | Medium | Medium-large | Conceptually attractive, but not first unless MC becomes central. |

Recommended first package:

1. RL-D0 and RL-S0 evidence audit.
2. RL-S3 normalized eval-goal, because it directly tests a known failure
   mechanism.
3. RL-S4 Acrobot retention scan.
4. RL-S5 eval-goal generalization.

After those, choose either a credibility package (`RL-D1`, `RL-D2`,
`RL-S1`) or a mechanism package (`RL-S2`, `RL-S6`, `RL-S8`). Do not
upgrade appendix-only numbers unless the manuscript actually needs them.

## 6. Paper III outline seed

1. **Introduction.** RL as the paper where `Y` becomes the problem:
   returns are not labels, and policy changes the data distribution.
2. **Background.** FQI, LSTD/RLS, random features, successor features,
   HER, and intrinsic novelty. Be explicit that these families already
   exist; SSUP reframes and connects them.
3. **Dense-control method.** NEF-FQI, per-action sufficient statistics,
   trace-scaled ridge, n-step returns, RLS decay, seed policy.
4. **Dense-control results.** Clean LunarLander, TD/bootstrap negative,
   final base, non-adopted ensemble result.
5. **Sparse-control method.** Successor features, leverage novelty,
   hindsight/eval-goal projection.
6. **Sparse-control results.** MC/Acrobot positives, SparseChain
   diagnostic, MiniGrid and dense-LunarLander negatives.
7. **Discussion.** Target construction as the RL analog of labels in SL
   and per-class additivity in CL; where the SSUP program helps and
   where it does not.
8. **Appendix.** Source trail, checkpoint references, metadata schemas,
   runner seed policy, environment versions, and result artifacts.

