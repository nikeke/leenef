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

## Clean restart base config

The clean restart begins from the last defensible pre-bug base, copied into
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
