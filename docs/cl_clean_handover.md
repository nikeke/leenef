# cl_clean handover — instructions for the executing Copilot

> Sister document to `docs/sl_clean_sweeps.md` and the Paper-I
> Phase-C handover used by `paper-i/phase-c-experiments`.  Anything
> in **AGENTS.md** still applies; this file only adds Paper-II
> specifics.

## Branch + worktree

* Scaffolding branch (this session): **`paper-ii/cl-scaffold`**.
* Execution branch (your branch): **`paper-ii/cl-experiments`**,
  created off `paper-ii/cl-scaffold` after the user has reviewed
  the scaffold.
* Worktree path: `../leenef-paper-ii` (the user creates it via
  `git worktree add`).

The scaffolding session **does not commit** to your branch and does
not modify `.git/worktrees/`.  If you need a code change to the
scaffolding (e.g., a runner bug), commit it on
`paper-ii/cl-experiments` and the user negotiates the merge back
into the scaffold branch.

## Allowed paths (you may write here)

```
benchmarks/cl_clean/**
results/cl_clean/**
docs/cl_clean_sweeps.md
docs/cl_clean_notes.md
notebooks/cl_clean/**
tests/test_cl_clean.py
```

## Forbidden paths (do **not** modify)

* `benchmarks/sl_clean/**`, `results/sl_clean/**` — Paper I.
* `docs/SSUP-I.md`, `docs/sl_clean_sweeps.md`,
  `docs/paper_insights.md`, `docs/build_ssup_pdf.sh`.
* `src/leenef/**` — any change here invalidates `code_identity`
  for every in-flight panel.  If a library bug shows up, surface
  it; do not patch silently.

## Phase order

1. **B1** joint-equivalence (4 panels × 10 seeds).  Sentinel per
   panel: `results/cl_clean/<experiment>_done`.
2. **B2** cross-method comparison.  Reuses the panels from B1;
   adds the MLP cells.
3. **B3** capacity matrix (Permuted-MNIST grid + Permuted-CIFAR-10
   spot panel).
4. **B4** center-adaptation falsifier.
5. **B5** solver-mode + α regime.
6. **C1, C2** ConvNEF-CL panels — *blocked* on Paper I v2 ConvNEF
   chapter landing.

After each phase sentinel:

* commit results JSON + log to `paper-ii/cl-experiments`;
* append a phase block to `docs/cl_clean_sweeps.md` with
  configs, dates, max-abs-gap, mean ± SE, and adoption verdicts;
* append a paragraph to `docs/cl_clean_notes.md` if the panel
  changed the story;
* hand back to the user.

## Kickoff sequence

```bash
cd ../leenef-paper-ii
source venv/bin/activate    # share the parent's venv via $VIRTUAL_ENV symlink
pytest tests/test_cl_clean.py -q  # sanity
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/cl_clean/cl_hp_driver.sh nef_continual_split_mnist
```

## Driver edits

**Do not edit `cl_hp_driver.sh` while a phase that uses it is in
flight.**  Bash re-reads the script file mid-execution.  If you need
to register a new panel: add it as a new `case` clause and only land
the change between phases, or copy the driver to
`run_<phase>.sh` for the next phase.

## Code-identity discipline

Every result row carries `code_identity`.  If you change anything
under `CODE_IDENTITY_PATHS` (see `cl_repro.py`), every in-flight row
becomes "old code" and the analyzer will exclude it from aggregates.
Land code changes between phases, never mid-phase.

## Reproducibility hazards (recap of AGENTS.md)

* Explicit `torch.Generator()` seeded from `_make_subrngs(seed)`.
* Encoders + gains depend only on the seed, never on the contrasted
  factor.
* Detached launches via `setsid ... < /dev/null & disown`.
* Watchdogs do not survive CLI sandbox restarts; re-arm per
  session.
* Dataset: every JSON row must carry `dataset_metadata.raw_hashes`
  and `dataset_metadata.processed_hashes`, and (for CL) the
  `stream_metadata` block emitted by `make_task_stream`.

## Hand-back protocol

When `B1..B5` and (later) `C1..C2` sentinels have all landed:

1. Tag the head of `paper-ii/cl-experiments` as
   `paper-ii/cl-data-frozen`.
2. Hand back to the scaffolding session for figure generation,
   number cross-check, and manuscript drafting.
