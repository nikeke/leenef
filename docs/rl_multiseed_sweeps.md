# NEF-FQI multi-seed hyperparameter sweeps

Living log of the rigorous, multi-seed isolation sweeps we run on
LunarLander-v3 to understand each NEF-FQI knob in isolation. Earlier,
exploratory single-seed sweeps live in `docs/rl_review.md`; this document
covers the post-`rl_review.md` §8 sweeps where every value is re-run with
8 fresh seeds and a strict statistical adoption rule.

## Methodology

- **Environment**: LunarLander-v3, 1000 episodes per run, eval every
  50 episodes (50-episode greedy rollouts).
- **Seeds**: 8 per value (seeds 0..7), unless noted.
- **Parallelism**: 6 workers via `multiprocessing.Pool` (CPU-only).
- **Driver**: `session/files/rl_hp_driver.sh` chains all phases
  unattended; logs progress and regenerates plots after each phase.
- **Sweeper**: `benchmarks/run_hp_sweep.py` — single-knob
  `--knob NAME --values v1 v2 …`, varies one parameter at a time
  while reading the rest from `results/rl/base_config.json`.
- **Analyzer**: `benchmarks/analyze_rl_sweep.py` — picks the value with
  the highest mean `best_eval`, **adopts it as the new base only iff**
  `Δmean > sqrt(SE_winner² + SE_current²)`. Atomic write
  (`.tmp` + `os.replace`).
- **Plotter**: `benchmarks/plot_rl_sweeps.py` — per-knob PNG (mean line
  with min–max bars, jittered seed dots) plus a summary panel. Outputs
  to `results/rl/figures/sweep_<knob>.png` and `sweep_summary.png`.
- **Base config**: `results/rl/base_config.json` is the single source of
  truth, mutated atomically between phases. Snapshots for each completed
  phase are appended to the table below.
- **Adoption rule rationale**: 1·SE_combined ≈ 68% confidence on the
  difference (assuming independence). Tighter than naive "pick best
  mean", looser than 1.96 SE so we don't get permanently stuck on a
  noisy local optimum. Ties default to the incumbent (no churn).

### What we record per run

`results/rl/<knob>_lunarlander.json` keyed by `<knob>=<value>_s<seed>`,
each entry has: `value`, `seed`, `best_50`, `final_50`, `best_eval`,
`final_eval`, `time_s`, full `eval_history` (per 50 ep), full per-episode
`rewards`, and the snapshotted `config` used for that run. Resumable —
the sweeper skips labels already present.

## Phase ordering

The driver runs the eight phases in this fixed order; each subsequent
phase reads the (possibly updated) base config:

1. **A1** `n_step` ∈ {10, 20, 50, 100}
2. **A2** `forget_factor` ∈ {0.99, 0.995, 0.999, 0.9995}
3. **A3** `alpha` ∈ {1e-4, 1e-3, 1e-2, 1e-1}
4. **B1** `solve_every` ∈ {1, 5, 10, 20}
5. **B2** `tau` ∈ {0.1, 0.3, 0.5, 1.0}
6. **C1** `n_neurons` ∈ {2000, 3000, 4000, 6000, 8000}
7. **D1** `gamma` ∈ {0.97, 0.99, 0.995}
8. **D2** `epsilon_decay` ∈ {200, 500, 1000, 2000}

Note: `fqi_iters` was originally Phase B2 but dropped — it is a no-op
when `forget_factor` is set (the agent dispatches to the RLS
`_solve_from_stats` path which performs a single solve regardless).

## Base config evolution

| After phase | n_neurons | n_step | forget_factor | alpha | solve_every | tau | gamma | eps_decay | Notes |
|---|---|---|---|---|---|---|---|---|---|
| start | 4000 | 50 | 0.999 | 0.01 | 10 | 0.3 | 0.99 | 500 | derived from §8 of `rl_review.md` |
| A1 (n_step) | 4000 | **50** | 0.999 | 0.01 | 10 | 0.3 | 0.99 | 500 | kept (best winner, within SE of 100) |
| A2 (forget_factor) | 4000 | 50 | **0.999** | 0.01 | 10 | 0.3 | 0.99 | 500 | kept (best mean, well above 0.9995) |
| A3 (alpha) | 4000 | 50 | 0.999 | **0.1** | 10 | 0.3 | 0.99 | 500 | **adopted** — Δ=73.3 on best ≫ SE_comb=39.7; on final Δ=46.3 just above SE_comb=41.8 |
| A3b (alpha extension) | 4000 | 50 | 0.999 | **0.1** | 10 | 0.3 | 0.99 | 500 | kept — extension values 0.3, 1.0, and 3.0 were all worse |

## Phase results

### A1 — `n_step`

| value | n | best_eval (mean ± SE) | final_eval (mean ± SE) |
|---|---|---|---|
| 10  | 8 |  38.9 ± 15.1 | -84.5 ± 11.7 |
| 20  | 8 |  79.0 ±  6.9 |  55.0 ± 12.8 |
| **50**  | 8 | **147.3 ± 18.2** | **143.4 ± 17.6** |
| 100 | 8 | 127.0 ± 26.7 | 121.8 ± 25.9 |

Plot: `results/rl/figures/sweep_n_step.png`

**Verdict**: 50 wins on both metrics. 100 is within SE on `best_eval`
(Δ = 20.3, SE_combined = √(18.2² + 26.7²) ≈ 32.3, so well within) — kept
50 as the incumbent. Short horizons (10, 20) are clearly worse;
bootstrapping bias dominates the variance benefit at this neuron count.

**Open question raised by user**: at 4000 neurons, n_step=50 already
beats 100 — possibly the function approximator is the bottleneck for
longer horizons. C1 (n_neurons sweep with n_step fixed at 50) won't
answer this directly; a follow-up 2-D sweep `(n_neurons × n_step)` at
{4000, 8000} × {50, 100, 200} would isolate the entanglement.

### A2 — `forget_factor`

| value | n | best_eval (mean ± SE) | final_eval (mean ± SE) |
|---|---|---|---|
| 0.99    | 8 |  74.2 ± 16.1 |  56.0 ± 23.2 |
| 0.995   | 8 | 130.7 ± 20.5 | 116.0 ± 20.2 |
| **0.999**   | 8 | **163.4 ± 26.1** | **141.4 ± 32.6** |
| 0.9995  | 8 | 138.7 ± 10.2 | 130.5 ± 11.4 |

Plot: `results/rl/figures/sweep_forget_factor.png`

**Verdict**: 0.999 wins on both metrics. 0.9995 has notably tighter SE
(more stable) but lower mean; we'd need more seeds to know whether the
gap is real. Kept 0.999. Effective memory ≈ 1000 transitions, which
neatly matches n_step=50 × solve_every=10 cadence.

### A3 — `alpha`

| value | n | best_eval (mean ± SE) | final_eval (mean ± SE) |
|---|---|---|---|
| 0.0001 | 8 | -56.0 ±  5.8 | -64.8 ±  5.3 |
| 0.001  | 8 | -16.7 ±  9.6 | -35.7 ± 12.6 |
| 0.01   | 8 |  96.3 ± 24.1 |  81.9 ± 22.7 |
| **0.1**    | 8 | **169.6 ± 31.6** | **128.2 ± 35.1** |

Plot: `results/rl/figures/sweep_alpha.png`

**Verdict**: 0.1 wins decisively on `best_eval` (Δ=73.3 vs SE_comb=39.7).
On `final_eval` the win is narrower (Δ=46.3 vs SE_comb=41.8) but still
above the adoption threshold. **Adopted: alpha 0.01 → 0.1.** All
downstream phases now run with alpha=0.1.

**Caveat**: 0.1 is the upper edge of our sweep; the argmax could be
higher still. **Resolution**: scheduled as **A3b** (extension over
{0.3, 1.0, 3.0}) to run **before B1** rather than after the linear
sweep. Reason: if a higher alpha wins, all downstream phases done at
0.1 would have to be redone. Better to spend one extra phase now than
risk reissuing five.

**Caveat 2**: B1, B2, C1, D1, D2 will all run at the new alpha=0.1.
Earlier intuitions formed around alpha=0.01 do not transfer — re-read
results below in context of the new base.

### A3b — `alpha` extension

| value | n | best_eval (mean ± SE) | final_eval (mean ± SE) |
|---|---|---|---|
| **0.1** | 8 | **169.6 ± 31.6** | **128.2 ± 35.1** |
| 0.3 | 8 | 77.7 ± 31.5 | -1.6 ± 37.2 |
| 1.0 | 8 | -76.9 ± 10.4 | -114.9 ± 5.0 |
| 3.0 | 8 | -118.0 ± 8.0 | -161.0 ± 6.4 |

Plot: `results/rl/figures/sweep_alpha.png`

**Verdict**: 0.1 remains the winner after extending the regularization
sweep upward. The apparent promise of 0.3 was seed-fragile: seeds 0 and 5
learned, but the 8-seed mean fell far below 0.1 and final performance
collapsed. Alpha values 1.0 and 3.0 are clearly over-regularized.
**Locked: alpha=0.1** for B1+.

**Operational note**: before B1 we paused the driver to validate the
`rl-cpu-optimizations` branch (`37daab2`). The parity gate reuses the
existing A3 alpha=0.1 old-code seeds as the paired baseline and runs the
optimized branch on the same seeds.

#### CPU optimization parity gate

Optimized branch: `37daab2` / main `7703a10`.
Result file: `results/rl/parity_rl_opt_alpha01.json`.

| implementation | n | best_eval (mean ± SE) | final_eval (mean ± SE) | mean time/run |
|---|---:|---:|---:|---:|
| old code, alpha=0.1 | 8 | 169.6 ± 31.6 | 128.2 ± 35.1 | 2908 s |
| optimized RLS batching | 8 | 129.4 ± 40.4 | 87.9 ± 47.8 | 827 s |

Combined-SE parity check:

| metric | Δ optimized-old | combined SE | Verdict |
|---|---:|---:|---|
| best_eval | -40.2 | 51.3 | within SE |
| final_eval | -40.3 | 59.3 | within SE |

**Decision**: adopt the optimization for B1+ because both primary metrics
remain inside the original statistical-parity bar while runtime improves by
about 3.5× under these loaded runs.

**Caveat**: paired seed deltas are not clean (`best_eval` paired delta
-40.2 ± 33.5), driven mostly by seeds 6 and 7. Treat B1's early results
as a smoke test for possible semantic drift: if all solve_every values look
uniformly worse than the old alpha=0.1 baseline, stop and reassess rather
than blindly continuing.

### B1 — `solve_every` (in progress)

### ~~B2 — `tau`~~ (SKIPPED — provably no-op)

In `target_mode='nstep'` (our base), `solve()` dispatches to
`_solve_from_stats` which never reads `W_target`. `select_action`
reads only `W`. The Polyak update at line 893 maintains a write-only
copy. All 4 tau values would produce bit-identical training. Removed
from the driver, saves ~3-4 h.

If we ever swap to `target_mode='td'`, B2 must be re-introduced (then
`W_target` is read by `_solve_from_buffer` for bootstrapping).

## Scratch observations / hypotheses

### envpool postmortem (do not retry without a different approach)

We spent a session porting `run_hp_sweep.py` to envpool, hoping for ~3-5×
wall-clock speedup from C++ batched env stepping. Result: **envpool is
slower than gym `mp.Pool`** at any tested parallelism. Profile of a single
agent at the production base config (4000n, n_step=50, solve_every=10):

| component       | per-step  | share |
|-----------------|----------:|------:|
| env.step (gym)  | 0.04 ms   | 0.7%  |
| agent.update    | 4.58 ms   | 77.2% |
| agent.end_ep    | 1.29 ms   | 21.7% |
| agent.select    | 0.02 ms   | 0.3%  |
| env.reset       | 0.01 ms   | 0.1%  |

Upper bound on per-process envpool benefit (env.step → 0): **1.01×**.
Box2D physics is a rounding error next to NEF agent compute (encoding +
AᵀA accumulation + periodic 4000³ RLS solves). envpool optimizes the
wrong layer.

The conventional wisdom "env stepping dominates RL wall time" applies to
SGD-trained MLPs on Box2D/MuJoCo. It does **not** apply to
analytically-solved NEF-FQI at our scale. Verify before reaching for
envpool/GPU on any future RL extension.

Profile script: `benchmarks/profile_nef_rl_step.py`. Branch:
`envpool-port` (preserved, not merged).

### Other observations

- **H1: optimal `n_step` scales with `n_neurons`.** Motivation: longer
  horizons reduce bootstrap bias but raise target variance; a richer
  function approximator absorbs that variance. Current evidence: 1
  point (at 4000n, 50 ≫ 100 on final, ~tied on best). To confirm and
  derive a default `n_step = f(n_neurons)`, we need a 2-D sweep:
  {2000, 4000, 8000} × {25, 50, 100, 200} n_step, 8 seeds → 96 runs.
  Argmax shift confirms; flat surface refutes.

- **H2: optimal `forget_factor` ≈ 1 - 1/(k·n_step·solve_every).**
  Motivation: RLS effective memory should match the data-freshness
  window. At (n_step=50, solve_every=10), optimum 0.999 corresponds to
  window ≈ 1000 ≈ 2·n_step·solve_every. To confirm and derive a default
  `forget_factor = g(n_step, solve_every)`, sweep forget_factor across
  ≥3 pairs spanning a decade of `n_step·solve_every`, e.g. (20,5),
  (50,10), (100,20), 8 seeds → 120 runs. A clean log-log line in
  `log(1-f*) vs log(n_step·solve_every)` with slope ≈ -1 confirms.

If both hold we ship two derived defaults and remove three coupled
knobs from the user-facing surface. **Both hypotheses require new 2-D
sweeps** — the current 1-D phases cannot answer them, since each varies
one knob with the others fixed at the post-A2 base.

- The recentering branch is fully retired (see `rl_review.md` §8). All
  multi-seed sweeps below assume `recenter_interval=None`,
  `broadness=0`. Broadness may resurface later as part of a "scale-down
  underperforming neurons" mechanism, but not in this sweep series.
- `fqi_iters` is inert under RLS — confirmed in `src/leenef/rl.py`:
  `solve()` dispatches to `_solve_from_stats` when `forget_factor is not
  None`, which performs exactly one solve regardless of `fqi_iters`.
  Only the buffer-replay path honors it. Document this in code if a
  reader is likely to be confused.
- `n_neurons=8000` at `forget_factor=0.999` allocates a per-action
  float64 RLS stats tensor of ~512 MB; with 4 actions × 6 workers ≈
  12 GB. Watch for OOM in C1; may need to drop `--parallel` for that
  phase.

## Maintenance

- After each phase: append a row to "Base config evolution"; add the
  per-phase results table; embed/reference the new PNG; record any
  surprises or follow-up ideas in "Scratch observations".
- Cross-check: the analyzer might keep the incumbent even when a
  competitor's mean is higher (within SE_combined). Always note
  *why* (Δ, SE_combined) so future readers can re-evaluate with more
  data.
- When a phase changes the base, re-state which downstream phases'
  earlier intuitions might no longer hold, so the next reader knows
  whether to re-investigate.
- Old single-seed sweeps in `docs/rl_review.md` may have been at a
  different base than these multi-seed runs — never compare the two
  directly without checking the configs.
- If we ever rerun an earlier phase with the latest base (e.g. after C1
  changes `n_neurons`), add a new row labeled `A1@8000` etc. — don't
  overwrite the original.
