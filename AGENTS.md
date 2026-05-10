# AGENTS.md — leenef

> **Status — clean restart in progress.**
>
> The library source (`src/leenef/`) and core tests (`tests/`) are
> stable.  Everything else — `benchmarks/`, `docs/`, `results/`,
> `notebooks/` — has been moved to `attic/` while the SSUP
> three-paper series (Supervised → Continual → Reinforcement Learning)
> is rebuilt with rl_clean-grade reproducibility.
>
> The paper outline lives in
> [`attic/docs/paper_i_outline.md`](attic/docs/paper_i_outline.md); the synthesized
> insights story in [`attic/docs/paper_insights.md`](attic/docs/paper_insights.md);
> the running sweep execution log in
> [`docs/sl_clean_sweeps.md`](docs/sl_clean_sweeps.md); follow-up
> experiment specs in [`attic/docs/b1_followups.md`](attic/docs/b1_followups.md).
> Per-session work-in-progress notes live under
> `~/.copilot/session-state/<session>/` (gitignored, may be wiped on
> compaction); anything that should outlast a session must be migrated
> into `docs/`.  New infrastructure
> (`benchmarks/sl_clean/`, `results/sl_clean/`, `docs/sl_clean_sweeps.md`)
> is in place.
>
> The sections below labeled **(ATTIC EPOCH — pending rewrite)** describe
> the prior infrastructure and are kept for reference until the new
> equivalents are written.  Do not run their commands (the referenced
> scripts and result paths now live under `attic/`).

## Commands

```bash
source venv/bin/activate

# Tests
pytest                                        # full suite
pytest tests/test_core.py -q                  # core module tests
pytest tests/test_core.py::TestNEFLayer -q    # single test class
pytest -k test_fit_identity -q                # single test by name

# Lint — BOTH commands must pass before every commit
ruff check src/leenef/ tests/ benchmarks/        # lint (E/F/W/I rules)
ruff format --check src/leenef/ tests/ benchmarks/  # format check
# Auto-fix formatting:
ruff format src/leenef/ tests/ benchmarks/

# Install after changing pyproject.toml
pip install -e '.[dev]'

# sl_clean sweeps (Paper I)
# All driver invocations need PYTHON= explicit because no `python` binary
# exists on PATH (only `python3` and the venv).
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_mnist
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_fashion
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_cifar10
# B7 follow-ups (DC-fold mechanism falsifiers)
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_cifar10_centered
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_mnist_decentered
```

> Pre-Phase-A2 benchmark, figure-generation, and other sweep-driver
> commands have moved to `attic/`; only the `sl_clean` block above is
> live.  See `docs/sl_clean_sweeps.md` for the running phase log.

## SSUP paper series (Papers I–II)

Paper I owns the `benchmarks/sl_clean/`, `results/sl_clean/`,
`docs/sl_clean_sweeps.md` trio.  Paper II owns the parallel
`benchmarks/cl_clean/`, `results/cl_clean/`, `docs/cl_clean_sweeps.md`,
and `docs/cl_clean_handover.md` set.  Both stacks are modeled after the
rl_clean infrastructure.

**Documents you should read before non-trivial work:**

- [`attic/docs/paper_i_outline.md`](attic/docs/paper_i_outline.md) — paper structure
  (the six principles, §-by-§ skeleton).
- [`attic/docs/paper_insights.md`](attic/docs/paper_insights.md) — synthesized
  story (frame coefficients, DC framing, α-as-function-of-n, manifold
  view, H8 linear collapse).  This is the **spine** of Paper I.
- [`docs/sl_clean_sweeps.md`](docs/sl_clean_sweeps.md) — chronological
  execution log of B-phases (configs, sentinels, dates, raw verdicts).
- [`docs/cl_clean_notes.md`](docs/cl_clean_notes.md) and
  [`docs/cl_clean_sweeps.md`](docs/cl_clean_sweeps.md) — Paper II CL
  result synthesis and execution log.
- [`docs/SSUP-I.md`](docs/SSUP-I.md) and
  [`docs/SSUP-II.md`](docs/SSUP-II.md) — current manuscripts; keep
  their abstracts synchronized with the corresponding README block
  quotes.
- [`attic/docs/b1_followups.md`](attic/docs/b1_followups.md) — queued
  H1-completion experiment specs (B1.4 / B1.5 DC-mechanism falsifiers).
- [`attic/docs/technical_report.md`](attic/docs/technical_report.md) —
  pre-restart single-seed TR.  Useful for understanding earlier
  results/defaults but **claims must be re-validated** under the
  rl_clean-style multi-seed protocol before reuse.
- [`../leenef-sparse-reward-c/docs/rl_sparse_notes.md`](../leenef-sparse-reward-c/docs/rl_sparse_notes.md)
  — canonical sparse-reward notes; §1.2 is the clearest existing
  spelling-out of the leenef base principles (relevant for Paper III
  and as background for Paper I framing).
- [`../leenef-sparse-reward-g/docs/rl_sparse_sweeps.md`](../leenef-sparse-reward-g/docs/rl_sparse_sweeps.md)
  — sparse-reward sweep log.

**Operational rules:**

- Every result row must carry a `code_identity` SHA (HEAD or
  `<sha>-dirty-<sha256[:12]>`).  Rows whose identity no longer matches
  the current tree are kept on disk but excluded from aggregates.
- Every dataset row must carry `dataset_metadata.raw_hashes` (SHA-256
  of on-disk bytes) and `processed_hashes` (SHA-256 of the
  preprocessed float32 tensor).  See `benchmarks/sl_clean/datasets.py`.
- Sweeps are launched detached via `setsid ... < /dev/null &; disown`
  (NOT `nohup ... &` — that has been killed by session shutdown in
  practice).  Confirm only one runner PID is alive after launch — a
  duplicate writer will race on the result JSON.
- All randomness inside the harness uses explicit `torch.Generator`
  instances seeded from a per-stream sub-RNG
  (`_make_subrngs(seed)`).  Never rely on the global default RNG.
- Encoders + gains are derived from sub-RNGs that depend ONLY on the
  seed, never on the contrasted factor — so the same seed produces
  bit-identical encoders across e.g. all three bias modes.
- Alpha handling is centralized in `src/leenef/solvers.py`.  The default
  ridge policy is trace-scaled (`alpha * trace(A^T A) / n`, floored at
  `alpha`).  Normal-equation APIs should receive an `alpha` policy pair
  such as `(1e-2, "trace")`; fixed-alpha Woodbury/cached-inverse APIs
  should use the name `fixed_alpha`.  Do not add raw `torch.linalg.solve`,
  `cholesky`, `cholesky_ex`, `inv`, `lstsq`, or `torch.cholesky_solve`
  calls outside `src/leenef/solvers.py` unless the alpha policy is
  unrelated to NEF/RL ridge regression and documented at the call site.

**Manuscript rules:**

- Keep the current abstracts for each SSUP paper in `README.md`.  When a
  manuscript abstract changes materially, update the corresponding README
  quote in the same commit; if a shared claim changes, check both abstract
  quotes before committing.
- Write abstracts and main text from the finished-paper reader's
  perspective.  Do not call experiments, panels, figures, or results
  "new", "recent", "upcoming", or similar merely because they were added
  in the current editing session; reserve chronological language for
  sweep logs and handover notes.
- Once a paper repeats the six principles internally, principle references
  inside that paper should be internal (`P3`, `P6`), not qualified as
  "Paper I, P3" or similar.
- The paper must be self-contained. Do **not** refer readers to
  repository paths, notes files, result JSON filenames, driver scripts,
  or other repo-internal files in the main text. Appendix A may list
  artifact paths only when they are explicitly anchored to the GitHub
  repository URL: `https://github.com/nikeke/leenef`.
- Avoid internal experiment labels such as B1/B4/C5 in the manuscript
  unless they are formally defined and reused as reader-facing protocol
  names. Prefer descriptive phrases ("the raw-dataset bias panel",
  "the tuned MLP grid", "the recurrent Colab panel").
- Use "end-to-end wall-clock time" for measured timing comparisons.
  Avoid ambiguous "runs faster" phrasing when the measured quantity is
  fitting or training time; say "uses N× less wall-clock time" instead.
  Use "fit" for analytic decoder solves and "train" for gradient-based
  optimization.
- Do not overuse "honest/honestly" as a rhetoric marker in the paper.
  State the limitation or caveat directly; reserve honesty language for
  places where it genuinely clarifies the methodology.
- Use `g_i` or "gain" for per-neuron gains in the paper. Reserve `α`
  for regularization strength and trace-scaled regularization policy so
  §2 does not introduce a second unrelated alpha.
- In Markdown paper manuscripts, write formulas as real TeX math
  (`$...$` inline, `$$...$$` display). Do **not** encode formulas as
  inline code, Unicode math glyphs, or plain-text pseudo-math. Brace
  multi-letter subscripts/superscripts (`d_{\mathrm{in}}`,
  `\alpha_{\mathrm{eff}}`), keep code spans for identifiers only
  (`batch_acc`, config keys, paths), and avoid raw Unicode Greek in
  headings/captions where PDF text fonts or bookmarks may drop glyphs.
- Before committing a rebuilt PDF, run the generalized builder with
  `--verbose` and fix every `Missing character:` warning:
  `bash docs/build_ssup_pdf.sh --verbose docs/SSUP-I.md docs/SSUP-I.pdf`
  or the analogous `SSUP-II.md` command.  The usual fixes are to move
  Greek letters, comparison signs, superscripts/subscripts, and
  equivalence symbols into TeX math (`$\alpha$`, `$\le$`, `$10^{-3}$`,
  `$\equiv$`) rather than leaving Unicode glyphs in table cells,
  headings, or captions.

**Live phase log:** `docs/sl_clean_sweeps.md`.

## Watchdog pattern for long detached sweeps

When launching multi-hour sweeps, follow this pattern so the CPU never
silently idles and so completion auto-notifies:

1. **Detach the actual work** with `setsid ... < /dev/null &; disown`.
   Sentinel file path lives in `results/<namespace>/<phase>_done`.
2. **Driver script** (`benchmarks/.../run_<phase>.sh`) sequences
   sub-phases (typically: row sequential → permuted+modes parallel
   via `& wait`).  Touches the sentinel last.
3. **Watcher script** (`wait_for_<phase>.sh`) is launched as a
   foreground sync bash command.  Polls the sentinel; stall-detects
   via max(mtime) over the active log files; exits 0 on done, 1 on
   stall.  When the sync command exits, the runtime sends a
   completion notification — *that* is what wakes the agent up.
4. **Parallelization rule.**  If a single phase saturates ≤ 50 % of
   cores (BLAS-thread limited), schedule a second independent phase
   to run alongside it.  Confirm via `ps -o pcpu` that combined
   utilisation ≥ 80 % of available cores before walking away.
5. **Do not rely on `nohup` alone** — the runtime kills nohup'd
   processes on session shutdown in practice.  `setsid` survives.
6. **Do NOT edit `sl_hp_driver.sh` (or any bash driver) while a run
   that uses it is in flight.**  Bash re-reads the script file mid-
   execution and will fail with `unexpected EOF` part-way through a
   case clause if the file changes.  The B3 phase paid this tax once;
   B4 went out of its way to prepare driver edits but defer them
   until the sentinel landed.  Pattern: register new experiments in
   Python (which workers cache after their initial import), prepare a
   *new* `run_<phase>.sh` that doesn't reuse the live driver, or
   simply wait for `<phase>_done` before editing the driver.

Reference: `benchmarks/sl_clean/{run_b2_panels.sh,
wait_for_b2_panels.sh}` and the B2 launch in
`docs/sl_clean_sweeps.md`.

7. **Watchdogs cannot survive CLI sandbox restarts.** The Copilot
   CLI runs inside `bwrap --die-with-parent`, so every "detached"
   process started from inside the sandbox dies when the CLI
   restarts.  The actual long-running sweep workers — launched
   on the host before the CLI session — survive, but in-sandbox
   monitor processes do not.  Pattern: launch the sweep on the
   host (outside the sandbox), then re-arm the watchdog from
   inside any CLI session as needed.  Don't rely on a watchdog
   surviving across CLI restarts.

8. **Sandbox cannot see host process IDs.** `ps`, `/proc`, and
   any PID-based tools inside the sandbox only see the ~7 procs
   in the sandbox itself, not the actual sweep workers running
   on the host.  Cannot kill host processes from inside.  Use
   file-mtime-based liveness checks (the sweep JSON files mtime
   updates on every cell completion).  Ask the user to kill
   if needed.

9. **Watch appended logs with row counts, not completion strings.**
   Relaunched drivers append to existing logs, so old "phase done"
   lines can fool a naive watcher. A robust watcher should check the
   current JSON for the expected `code_identity`, expected value set,
   and expected row count, while using max(JSON/log mtime) only for
   stall detection.

10. **Do not assume historical worktree names still exist.** The
    Phase-C GPU worktree was temporary and may be removed. Prefer the
    branch/ref name (`paper-i/phase-c-experiments`, GPU-suite branch)
    and verify with `git branch --contains` or `git worktree list`
    before looking for a sibling directory.

## Reproducibility hazards (must not regress)

- **Default RNG vs explicit Generator.** Every place that draws
  randomness must use an explicit `torch.Generator()` seeded from
  `_make_subrngs(seed)` (or equivalent).  Past sweeps were silently
  irreproducible because some helper used the global default RNG.
- **Encoders+gains shared across factor levels.** When contrasting
  factor X (e.g. bias_mode), all RNG streams that don't depend on X
  must be functions of the seed alone, so they're bit-identical across
  X levels.  This both isolates causality and gives ~3× speedup.
- **Detached launches.** Use `setsid ... < /dev/null &; disown`, not
  `nohup ... &`.  After launch, immediately `ps -ef | grep` for the
  runner — a stale instance + the new instance racing on the same
  result JSON is the most common silent failure.
- **GPT-5.5 review pass.** Before launching any long sweep that will
  consume hours of CPU/GPU, dispatch a `task` agent with model
  `gpt-5.5` to review the hot path for optimization opportunities
  (matmul shapes, dtype churn, missing in-place ops) **and**
  reproducibility hazards (default-RNG draws, scatter-with-collision
  patterns).  See `7703a1010c10bbcf0c11d821792c3aad386ef6e7` for the
  precedent.  GPT-5.5's pattern-matching here is genuinely
  complementary to the main agent's coding strengths.
- **Scatter-write with possibly-colliding indices is non-deterministic.**
  `W[rows, cols] = vals` where `rows`/`cols` come from independent
  `randint` draws is non-deterministic on collisions because PyTorch
  does not specify scatter write-order.  Use `randperm(N*N)[:k]` and
  decode to `(row, col)` to guarantee unique positions.  Caught in
  `benchmarks/sl_clean/baselines/esn.py` (commit `19ed80d`).
- **Mid-stream fixed-alpha changes with cached Woodbury inverses are silent
  bugs.**  Both `NEFLayer.continuous_fit_encoded` and
  `StreamingNEFClassifier.continuous_fit` reject changing `fixed_alpha`
  while a `_M_inv` is cached for an earlier `fixed_alpha`.  Callers must
  call `refresh_inverse()` or `reset_continuous()` first.  See
  commits `b895794`, `45e2db9`.
- **Cached `nn.Linear`/`nn.LSTM` constructors consume the global RNG**
  before deterministic re-init overwrites the weights.  Currently
  results don't depend on this (we overwrite), but any subsequent
  global-RNG draw in the same process becomes seed-dependent on the
  Linear/LSTM construction order.  Either save/restore global RNG
  state across construction or avoid relying on it downstream.
  See `benchmarks/sl_clean/baselines/{mlp,lstm}.py`.
- **`nn.Dropout` (and built-in `nn.LSTM(dropout=...)`) uses the global
  RNG** with no `generator=` kwarg.  Base configs set dropout=0; if
  you raise it for an experiment, implement explicit-Generator masks
  manually (or accept that runs become irreproducible).

## Colab / GPU suite lessons

- Fresh Colab runtimes do not have MNIST raw IDX files. The GPU suite
  must download and unpack the raw files into `data/MNIST/raw` before
  loading datasets, because the harness hashes raw bytes directly for
  reproducibility instead of relying on torchvision's dataset object.
- Prefer a single zipped result artifact for notebook download. Multiple
  `files.download()` calls are fragile in Colab; bundle JSON panels,
  logs, and exact config copies into one archive.
- **Follow the Paper-I notebook pattern literally.** A Colab launcher
  notebook should have the same basic six-cell shape as
  `colab/sl_clean/recurrent_smnist_gpu_suite.ipynb`:
  1. markdown intro with the suite purpose and preferred GPU;
  2. a tiny CUDA-check cell (`import torch`, assert CUDA, print device);
  3. a shell-style clone/setup cell using notebook magics, not Python
     subprocess wrappers:
     `%cd /content`, `!rm -rf leenef`, `!git clone --depth 1 --branch <ref> ...`,
     `%cd leenef`, `!git log -1 ...`, `!test -f <expected runner>`;
  4. a dedicated install cell: `!pip install -q -e '.[dev]'`;
  5. a dedicated run cell: `!python -u <suite.py> ...` (optionally with
     a commented-out quick/smoke variant above the full command);
  6. a final zip-and-download cell that asserts the expected result/log
     files exist, writes one archive, includes the GPU config snapshots,
     and calls `files.download(...)`.
- **In the notebook itself, prefer shell-style `!` commands over
  `subprocess.run(...)` wrappers.** The standard pattern is easier to
  compare across suites and tends to show live command output in Colab
  more reliably.
- For C5-style recurrent row-sMNIST suites, G4 is the preferred Colab
  GPU if available at the documented ~5 CU/hr rate; L4 is the fallback.
  A100/H100 are unnecessary unless a future panel becomes genuinely
  memory-bandwidth or memory-capacity bound.

## Technical report as design reference (ATTIC EPOCH — pending rewrite)

Before designing new experiments, planning enhancements, adding features, or
exploring research directions, **read
[`attic/docs/technical_report.md`](attic/docs/technical_report.md)** to
understand:

- existing results, defaults, and the claims they support
- window-size regimes, regularization sweet spots, and encoder-strategy
  tradeoffs already established by prior sweeps
- which configurations have already been tried and what worked or failed
- the narrative spine (title, abstract, Sections 1/6/7) so new work fits
  the existing story or consciously revises it

New Colab suites, benchmark parameters, and code changes should be
**consistent with the TR's findings** unless the explicit goal is to
challenge or extend them.  When results change the story, update the TR
(following the maintenance rules below).

## Running long commands

Do **not** pipe long-running commands (benchmarks, training runs, sweeps)
through buffering filters like `grep`, `grep -v`, `tail -f | grep`, etc.
These filters buffer stdout and can hide all progress output for tens of
minutes, making it impossible to tell whether the command is stuck or
working.  Instead:

- Use `python -u` (unbuffered stdout) when running Python scripts.
- If you need to filter output, redirect to a file and grep afterward.
- For interactive monitoring, run the command directly without pipes.

**Never pipe through `head` or `head -N`.**  `head` reads N lines and
then exits, sending SIGPIPE to the upstream process.  Python handles
SIGPIPE by raising `BrokenPipeError` on the next `print()`.  If the
script uses `multiprocessing`, child workers inherit the broken pipe
and all their `print()` calls silently fail.  More critically, output
appears to stop entirely even though the processes continue running,
making it impossible to monitor progress.

**Python multiprocessing and stdout buffering.**  When using
`multiprocessing.Pool` with `map()` or `imap_unordered()`, child
workers inherit the parent's stdout.  Python buffers stdout by default
when it's not a TTY (e.g., when piped).  To ensure progress output
from workers appears promptly:

- Run with `python -u` to disable stdout buffering, or
- Use `flush=True` in `print()` calls inside workers, or
- Set `PYTHONUNBUFFERED=1` in the environment.

When running sweep scripts that save results to JSON, prefer
`imap_unordered` over `map` so results are saved incrementally as
each worker finishes, rather than waiting for all workers to complete.

When sweep parameters can differ by small numeric amounts (for example
`broadness=0.01` vs `0.02` vs `0.05`), do **not** round labels or JSON
keys so aggressively that distinct settings collide.  Analysis/plotting
scripts should also tolerate **partial** JSON files so interrupted or
paused sweeps can still be summarized and resumed later.

For **unattended multi-phase sweeps**, prefer a detached driver script that
resumes from the existing JSON outputs, logs clear phase boundaries and
summary snapshots to a dedicated log file, and regenerates charts or other
derived artifacts after each completed phase.  That makes it safe to stop
and resume long pipelines without losing already finished runs.

## Colab benchmark suite rules (ATTIC EPOCH — pending rewrite)

- If you change a Colab benchmark suite or its launcher workflow, **commit and
  push immediately** after the relevant checks pass so Colab can rerun against
  `main` without waiting for later cleanup.
- Colab suites must **flush result files after every completed experiment**.
  JSON/CSV outputs should be updated incrementally during the run, not only at
  the end, so interrupted sweeps still preserve completed results.

## RL multi-seed sweep notes (ATTIC EPOCH — pending rewrite)

Long-running, multi-seed isolation sweeps for NEF-FQI are tracked in
`docs/rl_multiseed_sweeps.md`. After each phase of `rl_hp_driver.sh`
finishes (or when designing a new sweep), update that document:

- append a row to "Base config evolution" recording the new
  `base_config.json` snapshot
- add the per-phase results table with `n`, mean ± SE for both
  `best_eval` and `final_eval`
- reference the regenerated `results/rl/figures/sweep_<knob>.png`
- record the adoption decision (Δ vs `sqrt(SE_winner² + SE_current²)`)
  and *why* the analyzer kept or replaced the incumbent
- log open questions or hypotheses raised by the data (e.g. knob
  entanglements that need a 2-D follow-up)

Plots from these sweeps live in `results/rl/figures/`, **not** in the
top-level `results/rl/` directory. When checking for new charts, look
in the `figures/` subfolder.

Never compare these multi-seed numbers directly to older single-seed
results in `docs/rl_review.md` without checking that the base config
matches — the multi-seed series uses `recenter_interval=None`,
`broadness=0`, and a strict statistical adoption rule that earlier
sweeps did not.

## Technical report maintenance (ATTIC EPOCH — pending rewrite)

The technical report (TR) is `docs/technical_report.md`. Treat it as a
globally coupled document: later edits often change earlier framing,
summary tables, figure captions, and even the title.

### Primary files

- `docs/technical_report.md` — main report text
- `docs/generate_figures.py` — authoritative figure data, labels, and plot
  text
- `docs/figures/*.png` — generated assets; regenerate them whenever figure
  data, labels, or captions change

### Narrative spine and cross-section propagation

- The **title, abstract, Section 1, Section 6, and Section 7** are the
  narrative spine. When headline results, defaults, or claims change,
  revisit all of them, not just the local section you edited.
- Do not stop at the directly touched paragraph or table. Propagate changes
  through:
  - table highlights
  - captions and figure references
  - cross-references between sections
  - summary/recommendation tables
  - comparison prose (`beats`, `matches`, `almost matches`, etc.)
  - README or other docs if they repeat the same claim
- After a major results edit, do a final top-down coherence pass over the
  abstract, introduction, conclusions, and title together.

### Style and numeric conventions

- Use **American English everywhere** in the report, documentation, and
  code-facing prose. Common conversions:
  - `normalise` / `normalisation` -> `normalize` / `normalization`
  - `initialise` / `initialisation` -> `initialize` / `initialization`
  - `optimise` / `optimisation` -> `optimize` / `optimization`
  - `regularise` / `regularised` -> `regularize` / `regularized`
  - `colour` -> `color`
  - `behaviour` -> `behavior`
  - `modelling` -> `modeling`
  - `neighbourhood` -> `neighborhood`
- Round this project's reported **percentages to one decimal place**
  everywhere in the TR:
  - prose
  - tables
  - table highlights
  - figure labels and legends
  - captions
  - summary bullets
- **Exceptions:** keep externally cited literature values at the precision
  used in the source paper (for example McDonnell et al.'s `99.17%`), and
  keep raw console/log output unrounded when reproducing it verbatim.
- After rounding, re-check the meaning of every comparison:
  - ties created by rounding must be highlighted equally
  - a previous "better than" may become "matches"
  - a previous "matches" may need to become "almost matches"
  - if two rows round to the same result and one is slower or otherwise
    dominated, usually remove or de-emphasize the weaker row
- Do not use space-separated thousands in the TR. Use `12000`, `20000`,
  `60000`, not `12 000`, `20 000`, `60 000`.
- Keep hardware qualification explicit. Default phrasing: **results are on
  CPU unless stated otherwise**. If speed is a headline point and both CPU
  and GPU measurements exist, mention both. Do not merge CPU accuracy and
  GPU timing into a single unqualified claim.

### Content placement and structure

- Background sections should stay conceptual. Implementation-specific
  paragraphs belong in methods/implementation sections, not in conceptual
  background.
- Benchmark tables and quantitative comparisons belong in results or
  discussion sections, not in method descriptions.
- When moving content between sections, update all references to it.
- Thin empirical sections should usually be strengthened with a familiar
  baseline or comparison, not left as isolated internal numbers.
- Summary tables should avoid duplicate scenarios and keep columns
  semantically consistent across rows.
- Check table highlighting carefully: every rounded tie for the best value
  should be highlighted, and no highlight should imply unsupported
  superiority.

### Code/report alignment

- Keep the TR synchronized with actual code defaults and capabilities:
  - 7 encoder strategies (`hypersphere`, `gaussian`, `sparse`,
    `receptive_field`, `whitened`, `class_contrast`, `local_pca`)
  - single-layer default activation `abs`
  - recurrent default activation `relu`
  - data-driven biases via `centers=...`
  - GPU-friendly accumulate+solve path in addition to Woodbury updates
- Avoid blanket "random" language. In this project, encoders are **fixed**
  but may be random or data-adapted; biases may be data-driven. Use
  mechanism-specific wording instead of oversimplifying.
- If figure data changes, update `docs/generate_figures.py`, regenerate the
  PNGs, then update in-text references and captions in the TR.
- If a headline claim changes, re-check whether the title still fits the
  revised abstract/introduction/conclusions.

## Architecture

This project implements the Neural Engineering Framework (NEF) of Eliasmith
and Anderson for
machine learning using rate-based neurons on top of PyTorch.

A **NEF layer** has three stages:

1. **Encode** — fixed unit encoders project the input into a
   high-dimensional neuron space: `a = activation(gain * ((x − d) · e))`
   where *e* is a fixed direction (often random, but potentially
   data-adapted) and *d* is a reference point (center) sampled from
   training data.
2. **Activate** — a nonlinear activation (default: `abs`) models the
   neuron firing rate.
3. **Decode** — output weights (decoders) map activities to the target:
   `y = a @ D`.

Biases are derived from centers: `bias = −gain · (d · e)`.  Default
configuration: **abs** activation, **hypersphere** encoders, **per-neuron
gain** U(0.5, 2.0), **data-driven biases** via `centers=x_train`.
Recurrent layers default to **relu** (abs causes gradient explosion in BPTT).

The key insight: **encoders are fixed (random or data-adapted); decoders
are solved analytically** via regularized least-squares
(`layer.fit(x, targets)`). This avoids gradient-based training for a
single layer entirely.

For multi-layer networks (`NEFNetwork` in `networks.py`), hidden layers
use encode-only (activities as inter-layer representation) and only the
output layer decodes.  Five training strategies are supported:

- **Greedy** (`fit_greedy`) — fixed hidden encoders, analytic output
  decoders.  Fastest, no gradient computation.
- **Hybrid** (`fit_hybrid`) — alternate analytic decoder solves with
  gradient updates to all encoders/biases.
- **Target propagation** (`fit_target_prop`) — replaces backprop with
  layer-local targets via analytical representational decoders (NEF
  inverse models) and difference target propagation.  Single-layer
  gradients only; no gradient flows between layers.
- **End-to-end** (`fit_end_to_end`) — standard SGD on all parameters,
  initialized via a greedy NEF solve.

### Module roles

- `encoders.py` — encoder generation strategies, each registered in
  `ENCODER_STRATEGIES` dict.  Use `make_encoders(n, dim, strategy=...)`.
  Strategies: `hypersphere` (default), `gaussian`, `sparse`,
  `receptive_field` (local image patches for spatial locality),
  `whitened` (PCA-subspace projection adapts to data covariance),
  `class_contrast` (directions from one class to nearest other class),
  `local_pca` (top eigenvector of each neuron's local neighborhood).
- `activations.py` — rate neuron models, registered in `ACTIVATIONS` dict.
  Use `make_activation(name, ...)`.
- `solvers.py` — decoder solvers, registered in `SOLVERS` dict.
  Use `solve_decoders(A, targets, method=...)`.
- `layers.py` — `NEFLayer(nn.Module)` ties them together.
  `forward()` runs the full pipeline; `fit()` solves decoders analytically.
  `partial_fit()` / `solve_accumulated()` enable incremental/online learning
  via normal-equation accumulation.
  `continuous_fit()` / `continuous_fit_encoded()` apply rank-k Woodbury
  updates to the cached system inverse, producing up-to-date decoders
  after every batch.  The inverse is stored in float64 to prevent drift.
  `refresh_inverse()` recomputes the inverse exactly from accumulated
  statistics; `reset_continuous()` clears all Woodbury state.
  Accepts optional `centers=` training data to derive data-driven biases
  (`bias = -gain * (d · e)`), placing each neuron around a training sample.
  Gain can be scalar, range tuple, or per-neuron tensor (`_gain` buffer).
  `encoder_kwargs=` dict passes strategy-specific args (e.g. `image_shape`).
- `ensemble.py` — `NEFEnsemble(nn.Module)` wraps N independent NEFLayers
  with different random seeds.  Combines via probability averaging (`mean`)
  or majority voting (`vote`).  Ideal for exploiting the fast analytic
  solve: 20 members × 2s = 40s, still faster than one gradient-trained MLP.
- `networks.py` — `NEFNetwork(nn.Module)` stacks NEFLayers with the five
  training strategies above.  Also supports `hybrid_e2e` (hybrid → E2E).
- `recurrent.py` — `RecurrentNEFLayer(nn.Module)` implements the NEF
  decode-then-re-encode feedback loop for temporal sequences.  State
  decoders close the recurrent loop; output decoders produce the task
  prediction at the final timestep.  Four strategies (greedy/hybrid/
  target-prop/E2E); only E2E produces competitive results — greedy and
  hybrid struggle because random encoders compound state feedback noise
  across timesteps.  Target propagation through time (TPTT) uses state
  decoders as inverse models to propagate targets backward through
  timesteps.  `solve_from_normal_equations` in `solvers.py` avoids
  materializing the full T×B activity matrix during greedy solve.
  Supports data-driven biases via `centers=` (first timestep + zero
  state).
- `streaming.py` — `StreamingNEFClassifier(nn.Module)` classifies
  variable-length temporal sequences using a delay-line reservoir approach.
  Overlapping windows of K consecutive timesteps are encoded through fixed
  NEF neurons, mean-pooled over time, and decoded to class labels.
  Supports batch `fit()`, continuous Woodbury `continuous_fit()`, and
  GPU-friendly `accumulate()` + `solve()` (float32, no Woodbury inverse).
  `encode_sequence()` chunks internally to limit peak memory for large
  models.  Achieves 98.6% on sMNIST-row without gradient descent.
- `conv.py` — gradient-free convolutional feature extraction pipeline.
  `ConvNEFStage` learns PCA (or k-means) filters from data patches and
  applies them as fixed conv2d filters + abs activation + pool.
  `patch_size` accepts int (square) or `(h, w)` tuple (rectangular) for
  oriented filters.  `ConvNEFPipeline` stacks stages (sequential or
  parallel) with spatial pyramid pooling and a NEFLayer classification
  head.  Supports feature standardization (`standardize=True`), global
  contrast normalization (`gcn=True`), local contrast normalization
  (`lcn_kernel=5`), and data augmentation (`augment_fn`, `n_augment`).
  `parallel=True` enables multi-scale extraction with different patch
  sizes concatenated along the channel dimension.  `ConvNEFEnsemble`
  wraps N pipelines with different seeds.  All gradient-free: PCA
  filters are data-derived buffers, decoders are solved analytically
  via partial_fit/solve_accumulated.

## Conventions

### Pre-commit checks

**Before every commit**, run both lint and format checks and fix any
errors.  CI enforces both; commits that fail either check will break the
build.

```bash
ruff check src/leenef/ tests/ benchmarks/
ruff format --check src/leenef/ tests/ benchmarks/
```

If `ruff format --check` fails, run `ruff format src/leenef/ tests/ benchmarks/`
to auto-fix, then stage the formatted files before committing.

### Registry pattern

Encoders, activations, and solvers each use a `dict` registry
(`ENCODER_STRATEGIES`, `ACTIVATIONS`, `SOLVERS`) keyed by string name, with
a `make_*` / `solve_*` factory function.  New variants should be added to
the registry dict; callers select by string name.

### Decoders as `nn.Parameter`

Decoders are stored as `nn.Parameter(requires_grad=False)` so they
participate in `state_dict` and model saving, but are not updated by
gradient optimizers by default.  `fit()` writes to `.data` directly.

### Tensor-only, no Python loops

All computation must use batched PyTorch tensor operations
(`torch.linalg`, `@`, broadcasting).  Never loop over neurons or samples
in Python.

## Available GPU resources

The user can provision any of the following GCP GPUs on demand.  Choose
the cheapest option that fits the workload; don't request A100/H100
unless the benchmark genuinely needs the memory or bandwidth.

| GPU | VRAM | FP32 TFLOPS | Mem BW | Best for |
|-----|------|-------------|--------|----------|
| T4 | 16 GB GDDR6 | 8.1 | 320 GB/s | Quick sweeps, small models, CI |
| L4 | 24 GB GDDR6 | 30 | 300 GB/s | Medium matmuls, ConvNEF, EWC baselines |
| G4 (RTX PRO 6000 Blackwell) | 96 GB GDDR7 | ~100 | ~900 GB/s | Large neuron counts (50k+), big AᵀA, ImageNet-scale |
| A100 | 80 GB HBM2e | 19.5 (156 TF16) | 2 TB/s | HBM-bandwidth-bound ops, very large batch solves |
| H100 (PCIe) | 80 GB HBM2e | 51 (~1000 TF16) | 2 TB/s | Only if A100 is bottleneck; overkill for most NEF work |

### When to use a GPU

NEF's computational profile is dominated by three operations:

1. **Encoding** `A = act(gain * (x @ E^T) + bias)` — O(N × d_in × n_neurons).
   CPU-bound above ~5000 neurons with d_in ≥ 3072.
2. **Accumulation** `AᵀA += A^T @ A` — O(n_neurons² × N/batch).
   Memory-bound; benefits from HBM above ~20000 neurons.
3. **Solve** `(AᵀA + αI)⁻¹ AᵀY` — O(n_neurons³).
   GPU wins above ~10000 neurons.

**Rules of thumb:**
- MNIST (784-dim), ≤5000 neurons → CPU is fine (~2–10 s)
- CIFAR-10/100 (3072-dim), 2000–5000 neurons → CPU is fine (~5–30 s)
- CIFAR-10/100, 10000+ neurons → **T4 or L4**
- EWC baselines (per-sample Fisher) → GPU helps regardless of NEF size
- Large sweeps or Colab suites → **L4** minimum (best cost/perf)
- 50000+ neurons or ImageNet-scale → **G4**, **A100**, or **H100**

### Colab compute unit costs

Costs are in Google Colab compute units (CU) per hour.  $10 buys 100 CU
($0.10/CU).  Verified against GCP/Colab pricing as of mid-2026.

| GPU | CU/hr | $/hr | Notes |
|-----|-------|------|-------|
| T4 | 1.8–2.0 | $0.18–0.20 | Sometimes free on Colab; cheapest paid option |
| L4 | ~4.8 | ~$0.48 | Best cost/perf for medium workloads |
| G4 | ~5 | ~$0.50 | Available on Colab; check actual CU rate |
| A100 (40 GB) | ~12 | ~$1.20 | HBM bandwidth advantage for large solves |
| A100 (80 GB) | ~15 | ~$1.50 | Only if 40 GB is insufficient |
| H100 | ~20 | ~$2.00 | Available on Colab; check actual CU rate |

All five GPUs (T4, L4, G4, A100, H100) are available in the Colab
"Change runtime type" dialog.  No GCE VMs needed.
