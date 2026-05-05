# cl_clean notes — Paper II (Continual Learning)

The Paper II analog of `attic/docs/paper_insights.md`: an append-only
file where we record observations, wording experiments, and any
quantitative pattern that might end up shaping the manuscript.  The
phase log lives in `cl_clean_sweeps.md`; this file is the *story*
side of the same coin.

The trilogy is "SSUP" (Sufficient Statistics as the Universal
Primitive), of which this is part II.  Style continuity from Paper I
applies (American English, 1 dp, `g_i`, real TeX math, self-contained
manuscript).

## Spine

The same regularized least-squares solve $D = (A^\top A + \alpha I)^{-1}
A^\top Y$ that Paper I used for static SL, applied to a non-stationary
*data stream* with a fixed encoder, exhibits a striking property:
$A^\top A$ and $A^\top Y$ are **additive** over data partitions.  Hence
sequentially calling `partial_fit` over a stream of tasks and finally
`solve_accumulated` produces the *same* decoder as a single fit on
the union of tasks.  The "forgetting" axis has zero length by
construction.

Three things make this paper interesting (and not just an "of course"):

1. The capacity law $n / (T \cdot c) \approx 10$ is empirical, not
   a free corollary of additivity — it tells us how much neuron
   budget per output dimension this regime asks for.
2. The competitive comparison: at ~85 % retention, NEF-accumulate
   beats MLP-EWC on class-incremental splits by 50+ pp because EWC
   collapses, and matches or beats MLP-joint at much lower wall
   clock.
3. The ConvNEF extension (Paper I v2) lifts the visual ceiling
   without breaking the additivity, so the property survives onto
   features that are themselves learned, just not online.

## Story so far (to be filled in as panels land)

* Phase A scaffolding committed on `paper-ii/cl-scaffold`.
* No multi-seed numbers exist yet under `cl_clean/`.  All numbers
  in `attic/docs/continual_learning_notes.md` are single-seed and
  must be re-validated before the manuscript cites them.
* Phase C now has a committed Colab package under `colab/cl_clean/`.
  Important design choice: the notebook does **not** carry experiment
  logic.  It clones the scaffold branch, writes GPU config snapshots,
  runs the normal `cl_phase_runner`, and bundles the resulting JSON/log
  files.  The actual ConvNEF-CL experiment lives under
  `benchmarks/cl_clean/`, so the GPU rows carry the same `code_identity`
  discipline as the CPU panels.
