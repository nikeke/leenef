# Clean RL sweep namespace

This document tracks the **post-audit clean restart** for NEF-FQI RL sweeps.
New reproducibility-sensitive sweep outputs must go to the clean namespace:

- `results/rl_clean/`
- `results/rl_clean/base_config.json`
- `docs/figures/rl_clean_*.png`

The clean runner policy is:

1. One user-visible seed per run.
2. Warmup environment resets use `100000 + seed + episode`.
3. Training environment resets use `seed + episode`.
4. Evaluation environment resets use `10000 + episode`.
5. When a reset seed is provided, the environment action space is seeded to the
   same value.
6. Resumable sweep entries are only reused when all of the following match:
   `result_namespace`, `runner_version`, `config`, `seed`, and `seed_policy`.
7. Clean adoption uses `robust_eval = mean(eval_history[-3:])`; `best_eval` is
   diagnostic only.
8. Code identity is dirty-tree aware for source, benchmark, test, and clean
   figure-generation code, but it ignores generated clean result JSON and PNG
   artifacts so a resumable phase does not invalidate itself while writing
   outputs.

## Clean restart base config

The clean restart began from the last defensible pre-bug base, copied into
`results/rl_clean/base_config.json`:

```json
{
  "n_neurons": 4000,
  "gamma": 0.97,
  "alpha": 0.1,
  "epsilon_start": 1.0,
  "epsilon_end": 0.01,
  "epsilon_decay": 500,
  "solve_every": 10,
  "target_mode": "nstep",
  "n_step": 50,
  "forget_factor": 0.999,
  "fqi_iters": 1,
  "tau": 0.3,
  "broadness": 0.0,
  "recenter_interval": null
}
```

Current clean base after completed phases:

```json
{
  "n_neurons": 6000,
  "gamma": 0.98,
  "alpha": 0.1,
  "epsilon_start": 1.0,
  "epsilon_end": 0.01,
  "epsilon_decay": 500,
  "solve_every": 10,
  "target_mode": "nstep",
  "n_step": 100,
  "forget_factor": 0.999,
  "fqi_iters": 1,
  "tau": 0.3,
  "broadness": 0.0,
  "recenter_interval": null
}
```

## Base config evolution

| After phase | n_neurons | gamma | alpha | n_step | forget_factor | solve_every | eps_decay | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| start | 4000 | 0.97 | 0.1 | 50 | 0.999 | 10 | 500 | clean post-audit starting point |
| P1 gamma | 4000 | **0.98** | 0.1 | 50 | 0.999 | 10 | 500 | adopted over 0.97 by robust_eval |
| P3b H1 | **6000** | 0.98 | 0.1 | **100** | 0.999 | 10 | 500 | adopted over 4000@50 and rejected 8000/12000 edges |
| P4 H2 | 6000 | 0.98 | 0.1 | 100 | **0.999** | **10** | 500 | incumbent cadence/window remained best |
| P5 epsilon | 6000 | 0.98 | 0.1 | 100 | 0.999 | 10 | **500** | incumbent exploration decay remained best |

## Phase results

### P1 — gamma frame-setter

Result file: `results/rl_clean/gamma_lunarlander_v3.json`

Figures:

- `docs/figures/rl_clean_sweep_gamma.png`
- `docs/figures/rl_clean_sweep_summary.png`

| gamma | n | robust_eval mean ± SE | best_eval mean ± SE | final_eval mean ± SE |
|---:|---:|---:|---:|---:|
| 0.96 | 5 | 125.4 ± 6.5 | 155.5 ± 8.2 | 121.0 ± 14.8 |
| 0.97 | 5 | 160.4 ± 9.5 | 181.6 ± 10.2 | 161.8 ± 12.9 |
| **0.98** | 5 | **183.0 ± 11.6** | **204.2 ± 16.9** | **180.1 ± 11.5** |

Decision: adopt `gamma=0.98`. The robust-eval improvement over the incumbent
0.97 was `22.6`, with `SE_combined=15.0` (`ratio=1.50`), and both diagnostics
also favored 0.98. P2 alpha sanity is skipped because the gamma frame-setter did
not move to `gamma <= 0.96`.

### P3a — H1 diagonal pilot: horizon x capacity

Result file: `results/rl_clean/p3a_h1_pilot_lunarlander_v3.json`

Figure:

- `docs/figures/rl_clean_p3a_h1_pilot.png`

| config | n_step | n_neurons | n | robust_eval mean ± SE | best_eval mean | final_eval mean |
|---|---:|---:|---:|---:|---:|---:|
| `n25_k4000` | 25 | 4000 | 2 | 120.1 ± 4.0 | 141.8 | 133.6 |
| `n50_k4000` | 50 | 4000 | 2 | 172.6 ± 19.0 | 183.2 | 172.8 |
| `n50_k6000` | 50 | 6000 | 2 | 151.5 ± 17.0 | 181.1 | 165.9 |
| `n50_k8000` | 50 | 8000 | 2 | 171.8 ± 1.7 | 189.8 | 170.8 |
| **`n100_k8000`** | 100 | 8000 | 2 | **227.7 ± 6.3** | **233.9** | **232.2** |

Decision: P3a is **not** an adoption phase, but it cleanly triggered the
planned long-horizon/high-capacity reframe condition. The short-horizon
candidate `n25_k4000` was weak, the 6000-neuron tainted hint did not reproduce
in this pilot, and `n50_k8000` only tied the 4000-neuron incumbent. In contrast,
`n100_k8000` was strong on both seeds and both robust/final diagnostics.

Next action: run an edge-extended P3b rather than the original grid alone:
`{4000,6000,8000} x {50,100}` plus `8000@200` and `12000@100` if memory
allows. This preserves the scaling-law story while directly testing whether the
P3a winner is an edge artifact.

### P3b — H1 edge-extended grid: horizon x capacity

Result file: `results/rl_clean/p3b_h1_grid_lunarlander_v3.json`

Figure:

- `docs/figures/rl_clean_p3b_h1_grid.png`

| config | n_step | n_neurons | n | robust_eval mean ± SE | best_eval mean | final_eval mean |
|---|---:|---:|---:|---:|---:|---:|
| `n50_k4000` | 50 | 4000 | 5 | 183.0 ± 11.6 | 204.2 | 180.1 |
| `n100_k4000` | 100 | 4000 | 5 | 184.8 ± 2.9 | 195.5 | 182.1 |
| `n50_k6000` | 50 | 6000 | 5 | 172.1 ± 13.3 | 190.8 | 177.0 |
| **`n100_k6000`** | 100 | 6000 | 5 | **216.7 ± 6.2** | **232.8** | **212.3** |
| `n50_k8000` | 50 | 8000 | 5 | 152.6 ± 9.1 | 166.7 | 151.6 |
| `n100_k8000` | 100 | 8000 | 5 | 189.5 ± 20.0 | 198.6 | 187.3 |
| `n200_k8000` | 200 | 8000 | 5 | 158.1 ± 13.1 | 176.3 | 157.5 |
| `n100_k12000` | 100 | 12000 | 5 | 163.0 ± 11.9 | 183.6 | 176.2 |

Decision: adopt `n_neurons=6000` and `n_step=100`. This is the best
configuration in robust_eval and both diagnostics. It improves over the P1
incumbent `n50_k4000` by `+33.7` robust_eval (`SE_combined=13.2`) and over
`n50_k6000` by `+44.6` (`SE_combined=14.7`), so the clean H1 result supports a
horizon-capacity interaction rather than a pure capacity increase. The
high-capacity edge did not reproduce the P3a two-seed spike: `n100_k8000`,
`n200_k8000`, and `n100_k12000` all underperformed `n100_k6000`, and the
12000-neuron edge was also slower and more memory-intensive.

Next action: continue to P4 H2 using the new clean base
`n_neurons=6000, n_step=100`. Treat 8000/12000-neuron capacity as de-emphasized
unless a later phase reveals a new interaction that specifically justifies
returning to it.

### P4 — H2 grid: RLS cadence x forgetting window

Result file: `results/rl_clean/p4_h2_grid_lunarlander_v3.json`

| config | solve_every | forget_factor | n | robust_eval mean ± SE |
|---|---:|---:|---:|---:|
| `se5_ff995` | 5 | 0.995 | 5 | 182.2 ± 14.8 |
| `se5_ff998` | 5 | 0.998 | 5 | 196.3 ± 12.0 |
| `se5_ff999` | 5 | 0.999 | 5 | 200.4 ± 16.2 |
| `se5_ff9995` | 5 | 0.9995 | 5 | 193.8 ± 6.4 |
| `se10_ff995` | 10 | 0.995 | 5 | 214.4 ± 12.6 |
| `se10_ff998` | 10 | 0.998 | 5 | 181.8 ± 11.3 |
| **`se10_ff999`** | 10 | 0.999 | 5 | **216.7 ± 6.2** |
| `se10_ff9995` | 10 | 0.9995 | 5 | 192.7 ± 16.3 |
| `se25_ff995` | 25 | 0.995 | 5 | 200.2 ± 16.2 |
| `se25_ff998` | 25 | 0.998 | 5 | 184.1 ± 9.0 |
| `se25_ff999` | 25 | 0.999 | 5 | 192.9 ± 12.9 |
| `se25_ff9995` | 25 | 0.9995 | 5 | 179.0 ± 5.9 |

Decision: keep `solve_every=10` and `forget_factor=0.999`. Faster refresh
(`solve_every=5`) did not improve nonstationarity handling, slower refresh
(`25`) lost ground, and neither shorter nor longer forgetting windows beat the
incumbent under the combined-SE rule. `se10_ff995` was close but still below
the incumbent and had a larger SE.

### P5 — exploration decay

Result file: `results/rl_clean/epsilon_decay_lunarlander_v3.json`

Figure:

- `docs/figures/rl_clean_sweep_epsilon_decay.png`

| epsilon_decay | n | robust_eval mean ± SE | best_eval mean ± SE | final_eval mean ± SE |
|---:|---:|---:|---:|---:|
| 200 | 5 | -42.8 ± 53.1 | 43.4 ± 64.5 | -49.6 ± 55.2 |
| **500** | 5 | **216.7 ± 6.2** | **232.8 ± 8.4** | **212.3 ± 10.6** |
| 1000 | 5 | 123.8 ± 8.5 | 149.4 ± 7.7 | 117.0 ± 12.4 |
| 2000 | 5 | -122.3 ± 13.2 | -69.8 ± 8.8 | -129.8 ± 23.4 |

Decision: keep `epsilon_decay=500`. Faster exploitation (`200`) and slower
exploration decay (`1000`, `2000`) both underperformed the clean incumbent by a
large margin, so the result is an exploration-schedule failure rather than a
signal for recentering or capacity changes.

### P8 — 3-member ensemble pilot

Result file: `results/rl_clean/p8_ensemble_pilot_lunarlander_v3.json`

Configuration: three `6000`-neuron members, Thompson exploration during
training, seeds 0-2, and both mean-Q and majority-voting readouts evaluated
from the same trained ensembles.

| readout | n | robust_eval mean ± SE | values |
|---|---:|---:|---|
| mean-Q | 3 | 212.3 ± 8.8 | 195.7, 215.3, 225.8 |
| voting | 3 | 218.8 ± 8.9 | 201.6, 223.1, 231.6 |

Decision: promising but not adopted. Voting readout beats mean-Q on average and
two individual seeds exceed the single-agent incumbent, but the voting mean is
only `+2.1` robust_eval above the clean `6000@100` incumbent while
`SE_combined=10.9`. Under the clean adoption rule this is a statistical tie,
not an improvement. The result is still useful: it suggests that deployment
readout matters, and a larger ensemble confirmation should prioritize voting
readout if ensembles are revisited.

### P8b — voting-exploration ensemble pilot

Result file:
`results/rl_clean/p8_voting_ensemble_pilot_lunarlander_v3.json`

Configuration: same three `6000`-neuron members and seeds as P8, but with
majority-voting exploration during training instead of Thompson exploration.
Mean-Q and majority-voting readouts were again evaluated from each trained
ensemble.

| readout | n | robust_eval mean ± SE | values |
|---|---:|---:|---|
| mean-Q | 3 | 41.9 ± 70.1 | -98.0, 120.2, 103.6 |
| voting | 3 | 41.9 ± 69.8 | -97.0, 124.3, 98.3 |

Decision: reject voting exploration for the current clean base. This phase
answers a different question from P8: P8 tested Thompson-trained ensembles with
different deployment readouts, while P8b changed the exploration policy itself.
Voting exploration is far below both the single-agent incumbent (`216.7 ± 6.2`)
and the Thompson-exploration ensemble pilot (`218.8 ± 8.9` with voting
readout). The failure is driven by a catastrophic seed 0 run, so there is no
case for extending this branch before the final single-agent confirmation.

### P7 — final single-agent confirmation

Result file: `results/rl_clean/p7_final_confirm_lunarlander_v3.json`

Configuration: the final clean single-agent base, a duplicated incumbent
sentinel with the same scientific configuration, and the closest H2 runner-up
(`solve_every=10, forget_factor=0.995`), all on seeds 0-2.

| config | n | robust_eval mean ± SE | values |
|---|---:|---:|---|
| `incumbent` | 3 | 220.9 ± 10.3 | 219.1, 204.1, 239.5 |
| `incumbent_duplicate` | 3 | 220.9 ± 10.3 | 204.1, 219.1, 239.5 |
| `h2_runnerup_se10_ff995` | 3 | 219.6 ± 10.6 | 239.1, 217.1, 202.5 |

Decision: keep the final clean base unchanged. The duplicate incumbent exactly
reproduces the incumbent values up to row order, which is the intended
determinism sentinel. The closest H2 runner-up is a statistical tie but remains
slightly lower (`-1.3` robust_eval) and does not clear the adoption rule. This
confirms the post-clean-restart default as `gamma=0.98`, `n_step=100`,
`n_neurons=6000`, `solve_every=10`, `forget_factor=0.999`,
`epsilon_decay=500`, and no recentering.

## Historical results that are stale for sweep adoption

Treat these as **historical only**. Do not use them for skip/reuse decisions or
 cross-phase comparisons in the clean restart:

- `results/rl/solve_every_lunarlander.json`
- `results/rl/n_neurons_lunarlander.json`
- `results/rl/gamma_lunarlander.json`
- any downstream phase run on top of the stale `n_neurons=6000` adoption

More broadly, the pre-audit `results/rl/*.json` sweep artifacts remain useful as
qualitative history, but they are not the source of truth for resumable clean
reruns.

## Notes on intentionally uncontrolled paths

- Benchmark plotting jitter is now deterministic, but it is only a
  visualization aid and does not affect run metrics.
- User-provided augmentation callbacks in convolutional benchmarks remain
  caller-controlled; reproducibility depends on the callback honoring the
  provided outer benchmark seed policy.
