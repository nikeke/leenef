# envpool port — handoff brief

You are continuing work on the **leenef** repo in the worktree
`/home/amnesia/Persistent/src/leenef-envpool/` on branch
`envpool-port`. The main worktree at `/home/amnesia/Persistent/src/leenef`
on `main` is busy running an unattended NEF-FQI hyperparameter sweep —
**do not touch it**. All your work happens in this worktree.

Read `AGENTS.md` first for general project conventions (lint, commit
style, internet via proxychains, etc.). Read `docs/rl_review.md`
sections 1-4 only for RL background (skip §5+ unless curious).

## Goal

Add an **envpool-based fast path** for `benchmarks/run_hp_sweep.py`
that runs N independent NEF-FQI agents in a single Python process,
stepping all their environments with one batched `envpool` call.
Target: 2-3× total wall-clock speedup vs. the current `multiprocessing`
+ gym path, **without changing experiment semantics** (results from
the new path must be statistically indistinguishable from gym for the
same seed and config). CPU-only; no GPU work in this task.

The motivation is two upcoming 2-D sweeps (96 + 120 runs) for
hypotheses H1 (`n_step ↔ n_neurons`) and H2 (`forget_factor ↔ n_step ·
solve_every`). On the gym path those cost ~36 h on the local 12-core
machine. With envpool we expect ~12-15 h.

## Constraints

1. **Do not modify `src/leenef/rl.py`.** The `NEFFQIAgent` API stays
   exactly as is. You wrap it. (If you find a real bug, file a TODO
   in the handoff doc — don't fix it on this branch.)
2. **Do not modify the existing `benchmarks/run_hp_sweep.py`.** Add a
   new file (e.g. `benchmarks/run_hp_sweep_envpool.py`) or a `--backend
   {gym,envpool}` flag to a new module. The gym path must keep working
   identically.
3. **JSON output schema must match exactly.** Same keys, same value
   semantics — `results/rl/<knob>_lunarlander.json` has the canonical
   schema. The analyzer (`benchmarks/analyze_rl_sweep.py`) and plotter
   (`benchmarks/plot_rl_sweeps.py`) must work unchanged on envpool
   outputs. See "JSON schema" below.
4. **Resumability preserved.** Skip already-completed labels (existing
   pattern: `if label in existing: continue`).
5. **Per-agent seeds must be honored.** Each agent in the batch runs a
   different seed. Numpy + torch + envpool seeds all need to be set
   per-agent. The current gym path uses `seed=seed + ep` for env
   resets — preserve the equivalent.
6. **Lint clean before commit:** `ruff check` and `ruff format --check`
   on `benchmarks/`. CI enforces both.

## What you're replacing (gym path summary)

`benchmarks/run_hp_sweep.py:run_experiment` does:

```python
env = gym.make("LunarLander-v3")
agent = NEFFQIAgent(d_obs, n_actions, seed=seed, **agent_kwargs)
agent.warmup(env=warmup_env, n_episodes=20)         # 20 random episodes for stats
for ep in range(n_episodes):                        # n_episodes = 1000
    obs, _ = env.reset(seed=seed + ep)
    while not done:
        action = agent.select_action(obs)           # epsilon-greedy
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
    agent.end_episode()                             # triggers solve every solve_every
    rewards.append(total_reward)
    if (ep+1) % 50 == 0:
        eval_ret = evaluate_greedy(agent, env_name, 20, seed=10000)
```

Currently a sweep runs 32 such experiments via `multiprocessing.Pool`
with 6 workers (each in its own process, each with 1 gym env).

## What you're building (envpool path)

A single process hosts **N** NEFFQIAgent instances and **one** envpool
batch of N envs. The hot loop:

```python
batch_obs = pool.reset()                            # shape (N, 8)
done_flags = np.zeros(N, dtype=bool)
while not all_done:                                 # one episode per agent
    actions = np.array([agents[i].select_action(batch_obs[i]) for i in range(N)])
    next_obs, rewards, term, trunc, info = pool.step(actions)
    for i in range(N):
        if not done_flags[i]:
            agents[i].update(batch_obs[i], actions[i], rewards[i],
                             next_obs[i], term[i] or trunc[i])
            if term[i] or trunc[i]:
                done_flags[i] = True
                agents[i].end_episode()
    batch_obs = next_obs
```

Two ways to handle batch episode endings — pick whichever envpool
semantics suggest:

- **Synchronous-batch mode**: wait for all N episodes to finish, then
  reset the whole batch. Simpler. Wastes some work (faster envs idle
  waiting for slower ones).
- **Auto-reset mode** (envpool default for some envs): envs auto-reset
  on done; agent's "episode" boundaries come from `info` flags. Less
  wasted work but trickier accounting (per-agent episode counters).

Recommend **synchronous-batch** for the first cut. envpool's batched
step is so cheap that the idle waste is small (~10-30 % of steps in
the worst case for LunarLander where episode length varies 100-1000
steps).

### Choosing N

- `select_action` is per-agent serial Python (~50 µs at 4000n on CPU).
- envpool `step` for N=8 LunarLanders is ~0.3 ms total.
- Per-agent `update` (the dominant cost) is ~5 ms at 4000n with RLS.

So per env-step the wall time is ~5 ms × N (sequential agent updates)
+ 0.3 ms (env step). Linear in N. **N gives no time speedup over a
mp.Pool of N processes** if all agents are gym/CPU bound. The win
comes from **eliminating the multiprocessing overhead** (process
spawn ~10 s × N + per-step IPC) and from envpool's faster env stepping
(~5×). Net expected speedup: **~2-3× vs. 6-process gym pool**, more if
N > 6 (since multiprocessing was capped at 6 to avoid contention).

Suggested N for first benchmark: **8** (the default 8-seed sweep
matches one envpool batch per knob value). Then try 16 and pick the
winner.

## Step plan

### Step 1: install envpool + verify LunarLander works (~30 min)

```bash
source venv/bin/activate
proxychains pip install envpool
python -c "
import envpool
env = envpool.make('LunarLander-v2', env_type='gymnasium', num_envs=4, seed=0)
obs, info = env.reset()
print(obs.shape, obs.dtype)
"
```

LunarLander-v2 vs v3: envpool may not have v3 yet. **Verify the
underlying physics matches gym v3** by running both with the same seed
and a fixed action sequence; compare reward sums. If physics differs,
record exactly how (envpool API note, observation/reward scaling) and
either work around or stop and report. Mismatch is a blocker — the
whole point is parity with the existing `n_step`, `forget_factor`
results, which were on `LunarLander-v3`.

If `LunarLander-v3` is unavailable in envpool, options:
- Use v2 and confirm the agent reaches similar reward ranges (~+150
  best_eval at the gym base config). Document the env version diff.
- Abandon envpool for LunarLander; reassess.

### Step 2: sanity match (~2 h)

Write `benchmarks/_envpool_smoke.py` (gitignored or kept under tests/):

1. Run **one** NEFFQIAgent with the envpool path for 200 episodes,
   seed 0, base config from `results/rl/base_config.json`.
2. Run the same agent with the existing gym path for the same seed.
3. Compare rolling reward curves. They will not be bit-exact (random
   resets differ between the two env implementations), but trends and
   means should agree within seed noise (rough check: rolling mean at
   ep 200 within ±30 of gym).

If they agree → proceed. If not → debug seeding, observation scaling,
reward scaling, episode-termination semantics. Common gotchas:
envpool's `reset()` semantics, auto-reset on done, the `info` dict
shape for episode boundaries.

### Step 3: batched runner (~3 h)

Write `benchmarks/run_hp_sweep_envpool.py`. Mirror the CLI of
`benchmarks/run_hp_sweep.py`:

```
python benchmarks/run_hp_sweep_envpool.py \
    --knob n_step --values 10 20 50 100 \
    --seeds 0 1 2 3 4 5 6 7 \
    --batch 8                   # NEW: envpool batch size
```

Behavior:

- Group runs into envpool batches. Simplest: one batch per
  `(knob_value)`, batch size = number of seeds. So 4 values × 1 batch
  of 8 agents = 4 batches sequentially. (Or run all 32 as one batch
  of 32 if that's faster — measure.)
- Each batch: instantiate N NEFFQIAgents with their respective seeds,
  one envpool with `num_envs=N, seed=seed_master`. Per-agent env seeds
  via envpool's `seed` API or by replaying `seed + ep` semantics — see
  envpool docs.
- Save each agent's run_dict to `results/rl/<knob>_lunarlander.json`
  immediately after that agent finishes its 1000 episodes (or after
  its batch finishes, if simpler). Atomic write via `.tmp` + replace.
- Skip labels already in the JSON.

### Step 4: parity sweep (~3 h)

Re-run **n_step phase only** (32 runs) on the envpool path. Compare
mean ± SE per value to the gym numbers in
`results/rl/n_step_lunarlander.json`:

| value | gym best_eval | envpool best_eval | match? |
|---|---|---|---|
| 10 | 38.9 ± 15.1 | ? | within SE? |
| 20 | 79.0 ± 6.9 | ? | |
| 50 | 147.3 ± 18.2 | ? | |
| 100 | 127.0 ± 26.7 | ? | |

Save envpool results to `results/rl/n_step_lunarlander_envpool.json`
(do NOT overwrite the gym file). Document the comparison table in
`docs/envpool_port_handoff.md` as an addendum at the bottom.

If parity holds (all four values' means within `√(SE_gym² +
SE_envpool²)` of each other) → port is verified. Commit and PR.

If parity fails on >1 value → investigate. Likely culprits: env
stepping semantics, observation normalization, reward shaping
differences in envpool vs gym. Don't merge until you understand why.

### Step 5: timing report

In the same handoff addendum, record:

- total wall time for the envpool n_step phase
- vs. the gym phase wall time (read from
  `results/rl/n_step_lunarlander.json`'s `time_s` per run, summed and
  divided by 6 since the gym path used 6-way parallelism)
- estimated time for a 96-run 2-D sweep at this throughput
- chosen N (envpool batch size)

## JSON schema (must match)

`results/rl/<knob>_lunarlander.json` is a flat dict keyed by
`<knob>=<value>_s<seed>`. Each entry:

```json
{
  "label": "n_step=50_s3",
  "seed": 3,
  "knob": "n_step",
  "value": 50,
  "best_50": 178.3,
  "final_50": 143.1,
  "best_eval": 198.7,
  "final_eval": 152.2,
  "time_s": 3340.5,
  "config": { ...full agent_kwargs after _knob/_value popped... },
  "eval_history": [{"episode": 50, "eval": -480.5}, ...],
  "rewards": [-108.5, -213.6, ..., -91.0]
}
```

`config` should reflect the actual agent kwargs used (the value
override applied). `time_s` in envpool can be either per-agent (clock
elapsed for that agent's training, but agents share a batch so it's
identical for all in a batch) or per-batch divided by N. Document
which you chose.

## Out of scope (do not do)

- GPU. CPU-only.
- Vectorizing across the agent dim (one shared encoder/RLS for all N
  agents). That changes experiment semantics.
- Modifying `rl.py`, `analyze_rl_sweep.py`, `plot_rl_sweeps.py`,
  `base_config.json`.
- Touching the running driver, its log, or anything in
  `session/files/` of the main worktree.
- Porting `NEFFQIEnsemble`. Single agents only for this task.
- Optimizing `select_action` or `update` themselves.

## Files you'll create (proposed)

- `benchmarks/run_hp_sweep_envpool.py` — main deliverable
- `tests/test_envpool_parity.py` — small parity test (or
  `benchmarks/_envpool_smoke.py` if it's not unit-test-shaped)
- additions to `docs/envpool_port_handoff.md` (this file): an addendum
  with the parity table, timing numbers, and any deviations / known
  issues

## How to know you're done

- [ ] envpool installs and stepping a LunarLander batch works
- [ ] one-agent parity smoke test passes
- [ ] `run_hp_sweep_envpool.py` runs to completion on the n_step phase
- [ ] envpool n_step results match gym n_step within SE_combined for
      all four values
- [ ] addendum documents speedup and any caveats
- [ ] `ruff check && ruff format --check` clean on `benchmarks/`,
      `tests/`, and any new code
- [ ] Commit on `envpool-port` branch with a Conventional Commit
      message + body listing what was added and the verification
      result. Push the branch.

After your branch lands, the main session will pick up the worktree's
HEAD, run the 2-D H1/H2 sweeps via the new path, and merge to main
once those produce the expected speedup in production.

## Useful pointers

- envpool docs: https://envpool.readthedocs.io/
- gym → envpool migration: envpool's `step()` returns `(obs, reward,
  term, trunc, info)` like gymnasium when `env_type='gymnasium'`.
- Per-action RLS state size (sanity for memory): n_neurons² × 8 bytes
  × n_actions = 4000² × 8 × 4 = 512 MB per agent at 4000n. N=8 batch =
  4 GB. N=16 batch = 8 GB. Machine has 30 GB; keep N ≤ 16 to stay
  comfortable when other things are running.
- Existing `agent_kwargs` produced by `run_hp_sweep.py` already
  excludes `_knob` and `_value` before passing to `NEFFQIAgent`. Mirror
  that.
- The driver's current run uses `forget_factor=0.999` (RLS path); your
  port must work for that path. The buffer-replay path (`fqi_iters`)
  is dormant.

Good luck. When done, ping the main session by writing a brief status
into the addendum and pushing the branch.

---

## Addendum — port status

### Implementation

- `benchmarks/run_hp_sweep_envpool.py` — main runner.  CLI mirrors
  `benchmarks/run_hp_sweep.py` (`--knob`, `--values`, `--seeds`,
  `--episodes`, `--base-config`, `--output`, plus `--batch`,
  `--max-episode-steps`, and `--envpool-master-seed`).  Hosts N
  `NEFFQIAgent`s in one process and steps them via a single envpool
  batch.
- `benchmarks/_envpool_smoke.py` — single-agent gym↔envpool comparison
  helper (Step 2-style spot check, optional).
- `tests/test_envpool_parity.py` — schema, seed-stream, label-invariance,
  eval-cadence, and partial-completion tests; one `slow`-marked
  end-to-end batch test.

### Implementation choices

- **Label `s` is pinned to envpool sub-env index `s`.**  Empirically
  verified that envpool's per-sub-env RNG stream depends only on
  `(master_seed, sub_env_index)` and is **independent of `num_envs`**
  (see `test_envpool_seed_stream_independent_of_pool_size`).  Combined
  with a fixed `--envpool-master-seed` (default 0), this gives
  label-stable trajectories regardless of `--batch`, resume state, or
  which other labels share a batch.  `pool_size = max(args.seeds)+1`
  for every batch in a sweep.  Sub-envs at indices not in the current
  batch are stepped with placeholder action 0; their results are
  discarded.
- **Synchronous-batch with explicit per-env reset.**  envpool's
  default auto-reset semantics return the post-reset *first-action*
  observation rather than the true reset state.  After any env reports
  `term or trunc`, the runner calls `pool.reset(env_id=[slot])` and
  uses the returned obs as the agent's next `select_action` input —
  matching gym's "agent decides on the true reset state" semantics
  even though the per-episode RNG seed is *not* reset to `seed+ep`
  (envpool has no per-reset seed API; the gym path's
  `env.reset(seed=seed+ep)` cannot be replicated).
- **Truncation at 1000 steps** (`max_episode_steps=DEFAULT_MAX_EPISODE_STEPS`)
  to match gym's `LunarLander-v3` default; envpool defaults to
  `INT_MAX`.
- **Pre-seeded warmup.**  `NEFFQIAgent.warmup` calls
  `env.reset()` without a seed, which would default to system entropy.
  The runner pre-seeds each warmup env with the agent's `seed` so the
  warmup trajectory is deterministic per label, preserving label-
  invariance across batches.  This is a *different* deterministic
  warmup than the gym path's (which is non-deterministic in its own
  subprocess); statistical parity is what gets checked, not byte-
  identical trajectories.
- **`time_s` is per-agent wall clock from start of batched training to
  that agent's last-episode completion.**  Agents that finish their
  episodes earlier than batchmates record a shorter `time_s`;
  remaining envs continue stepping (we only skip `select_action` /
  `update` for finished agents).  Warmup time is excluded, matching
  the gym path.
- **Eager incremental persistence.**  `run_envpool_batch` is a
  generator that yields each agent's run dict the moment that agent
  finishes its `n_episodes`; `main()` writes it via `_atomic_write_json`
  before draining the next agent.  A mid-batch crash therefore cannot
  lose results from agents that already finished, and the failure
  branch only marks *unfinished* labels as errors.
- **Default output path.**  Defaults to
  `results/rl/<knob>_lunarlander_envpool.json` (note the `_envpool`
  suffix) so accidentally omitting `--output` cannot clobber the gym
  path's canonical results file.
- **`info["env_id"]` alignment guard.**  After each `pool.reset` /
  `pool.step`, if envpool returns `env_id` in `info` we assert it is
  a permutation of `0..pool_size-1` so any future async/partial
  envpool mode that returns rows out of order trips loudly rather
  than silently mis-routing observations.
- **Worktree-shared editable install workaround.**  Both new scripts
  prepend the worktree root to `sys.path` before importing
  `benchmarks.*`, so a sibling worktree's `pip install -e` doesn't
  shadow the local `benchmarks/` package.

### LunarLander-v3 in envpool

`envpool 1.2.0` exposes `LunarLander-v3` via the gymnasium API and
returns the standard `(obs, reward, term, trunc, info)` tuple.
Spot-checks (noop policy, fixed seed) show physics close to but not
bit-identical with gym v3 — episode lengths and reward sums are within
seed-noise of one another.  Gym is `LunarLander-v3` (Box2D in Python);
envpool is its own C++ port.  Trajectory-level identity with the gym
path is therefore impossible; statistical parity over seeds is the bar.

### Lint and tests

`ruff check` and `ruff format --check` both pass on the benchmarks
and tests trees.  `pytest tests/test_envpool_parity.py` is green
(7 passed): schema, seed-stream-independent-of-num_envs, end-to-end
batch, **label-invariance across batch composition**, eval-cadence,
and eager-yield partial-completion.

### Brutal-review fixes

The first revision of this port had several silent-correctness bugs
flagged by an internal review.  The current commit addresses all of
them:

| Severity | Issue | Fix |
|----------|-------|-----|
| CRITICAL | Per-label trajectory depended on the agent's position in the envpool batch (envpool seeds sub-envs as `master+i`).  Resume / `--batch` / non-contiguous seeds would silently change a label's training trajectory. | Pin label `s` to sub-env index `s` with a fixed master seed; verified empirically that the per-sub-env stream is invariant under `num_envs`. |
| CRITICAL | `agent.warmup` reseeds the env from system entropy, so warmup trajectories drifted across batch compositions. | Pre-seed each warmup env with the agent's `seed` before calling `agent.warmup`. |
| HIGH     | Default `--output` collided with the gym path's canonical file (`results/rl/<knob>_lunarlander.json`). | Default to `…_envpool.json`. |
| MEDIUM   | `info["env_id"]` discarded; assumed envpool returned rows in order. | Assert `env_id` is a permutation of `0..pool_size-1` whenever envpool surfaces it. |
| MEDIUM   | Batch failure tagged every label `error`, even ones that had already finished. | Generator-based `run_envpool_batch` with `on_complete` callback persists each agent's results immediately; failure branch only errors out unfinished labels. |
| MEDIUM   | Tests were schema-only and would have passed even with the bugs above. | Added explicit tests for seed-stream-vs-num_envs, label-invariance across batch composition, eval cadence, and eager-yield. |

### Parity sweep — TO DO

The original brutal-review-failing v1 implementation had a 32-run
parity sweep underway; it was killed and discarded (the buggy seeding
made its results unreliable as a gym-vs-envpool comparison).  The
clean parity sweep needs to be re-run on this fixed implementation:

```
python -u benchmarks/run_hp_sweep_envpool.py --knob n_step \
    --values 10 20 50 100 --seeds 0 1 2 3 4 5 6 7 \
    --episodes 1000 --batch 8 \
    --output results/rl/n_step_lunarlander_envpool.json
```

When complete, fill in the table below from the runner's `SUMMARY`
block:

| value | gym best_eval (mean ± SE) | envpool best_eval (mean ± SE) | within `√(SE_g²+SE_e²)`? |
|------:|---------------------------|-------------------------------|--------------------------|
| 10    | 38.9 ± 15.1               | _TBD_                         | _TBD_                    |
| 20    | 79.0 ± 6.9                | _TBD_                         | _TBD_                    |
| 50    | 147.3 ± 18.2              | _TBD_                         | _TBD_                    |
| 100   | 127.0 ± 26.7              | _TBD_                         | _TBD_                    |

Fail criterion (per the handoff): >1 value outside the combined-SE
window blocks merge.

### Timing — TO DO

To be filled in after the parity sweep completes:

- envpool n_step phase total wall time: _TBD_
- gym n_step phase reference (sum of per-run `time_s` ÷ 6 for the
  6-way `mp.Pool`): _TBD_
- speedup: _TBD_
- chosen N: 8 (one envpool batch per knob value).
- estimated time for a 96-run 2-D sweep at this throughput: _TBD_
