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
  "n_neurons": 4000,
  "gamma": 0.98,
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

## Base config evolution

| After phase | n_neurons | gamma | alpha | n_step | forget_factor | solve_every | eps_decay | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| start | 4000 | 0.97 | 0.1 | 50 | 0.999 | 10 | 500 | clean post-audit starting point |
| P1 gamma | 4000 | **0.98** | 0.1 | 50 | 0.999 | 10 | 500 | adopted over 0.97 by robust_eval |

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
