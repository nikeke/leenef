# Reproducibility audit handoff

This handoff is for a separate Copilot session working in a new branch and
worktree. The goal is to audit every source of stochasticity in the codebase
before we restart the NEF-FQI hyperparameter sweeps.

## Suggested setup

From the main repository:

```bash
git worktree add -b reproducibility-audit /home/amnesia/Persistent/src/leenef-repro-audit main
cd /home/amnesia/Persistent/src/leenef-repro-audit
source /home/amnesia/Persistent/src/leenef/venv/bin/activate
```

Do not start long RL sweeps in the audit branch. The output should be code,
tests, and documentation only. The main session will merge this branch and
then restart the sweep sequence.

## Background

We found a serious bug in `benchmarks/run_hp_sweep.py`: training episodes
were seeded, but the 20 warmup episodes were not. That meant two runs with
the same `(config, seed)` could begin from different normalization state,
initial decoder solves, and RNG consumption. It invalidated downstream
comparisons where the same configuration appeared under two different knob
labels.

Current `main` already contains a partial fix:

- `NEFFQIAgent.warmup(..., seed=...)`
- `NEFFQIEnsemble.warmup(..., seed=...)`
- `run_hp_sweep.py` uses `warmup_seed = 100000 + seed`
- HP results include `runner_version="seeded-warmup-v1"` and `warmup_seed`
- stale HP results are not skipped unless runner version and config match

Treat this as a starting point, not a completed audit.

## Audit scope

Search the whole repository for every stochastic source and either make it
explicitly seeded or document why it is intentionally uncontrolled. At
minimum, audit these patterns:

```bash
rg "torch\\.(rand|randn|randint|randperm|multinomial|normal|bernoulli)" .
rg "torch\\.Generator|manual_seed|seed\\(" .
rg "np\\.random|default_rng|random\\." .
rg "env\\.reset|action_space\\.sample|action_space\\.seed" .
rg "DataLoader|shuffle=True|random_state|KMeans|PCA" .
rg "multiprocessing|Pool\\(" benchmarks src tests
```

Specific areas to inspect:

- `src/leenef/encoders.py`: encoder generation must always receive and use a
  `torch.Generator` or equivalent deterministic seed path.
- `src/leenef/layers.py`, `networks.py`, `ensemble.py`, `conv.py`,
  `recurrent.py`, `streaming.py`: initialization, centers selection,
  ensemble/member seeds, PCA/k-means randomness, and any sampling of batches
  or patches.
- `src/leenef/rl.py`: feature initialization, warmup, recentering
  (`torch.randint` with `_torch_rng`), exploration RNG, ensemble RNG, and
  every path that touches Gym environments.
- `benchmarks/*.py`: every runner should have a clear seed contract. Gym
  environments should reset with explicit seeds during warmup, training, and
  evaluation; if an action space is sampled directly, seed it too.
- `tests/*.py`: tests should not depend on ambient global RNG state.

## Desired seed policy

Implement or document a consistent policy:

1. Each experiment has one user-visible seed.
2. All substreams are derived deterministically from that seed, with named
   offsets or a small helper. For example:
   - feature/agent seed: `seed`
   - warmup environment seed: `100000 + seed`
   - training episode seed: `seed + episode`
   - evaluation seed: fixed eval base plus episode offset
   - recentering torch generator: derived from the agent seed
3. Reproducibility metadata is saved with each result:
   - runner version
   - full config
   - main seed
   - warmup seed policy
   - evaluation seed policy
   - code version if cheap to include
4. Resumable sweep tools must not silently reuse results generated under an
   older seed policy, older runner version, or different config.

If exact bitwise determinism is not feasible for a path, state the limitation
clearly in code comments or docs and make the result files carry enough
metadata to avoid accidental comparisons.

## Tests to add

Add focused, fast tests rather than long benchmarks:

1. `NEFFQIAgent.warmup(seed=...)` passes the expected seed sequence into a
   fake environment.
2. `NEFFQIEnsemble.warmup(seed=...)` does the same.
3. `benchmarks.run_hp_sweep.run_experiment(...)` is deterministic on a tiny
   environment/config: calling it twice with the same seed and config returns
   identical `rewards`, `eval_history`, `best_eval`, and `final_eval`.
4. A stale HP result without the current runner version is rerun rather than
   skipped.
5. A completed HP result with the current runner version and identical config
   is skipped.
6. The same config under two different knob labels produces identical metrics
   for the same seed in a small smoke setup.

Avoid tests that require a 1000-episode LunarLander run.

## Acceptance criteria

Before handing the branch back:

```bash
ruff check src/leenef/ tests/ benchmarks/
ruff format --check src/leenef/ tests/ benchmarks/
pytest tests/test_rl.py -q
pytest -q
```

Also include a short summary in the final response listing:

- every stochastic source found
- which ones were fixed
- which ones remain intentionally uncontrolled, if any
- any result files or prior sweep phases that should be considered stale

## Things not to do

- Do not restart the HP sweep.
- Do not delete existing result JSON files.
- Do not reset or rewrite unrelated commits.
- Do not make broad research-design changes in this branch; keep the scope to
  reproducibility, seed policy, metadata, and tests.

