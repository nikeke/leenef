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
* Split-MNIST is the first multi-seed validation point, and it sharpens
  the paper's central claim immediately: over 10 seeds,
  `nef_accumulate` and `nef_joint` are numerically identical
  (`max|Δ| = 0.0000 pp`), so the "no forgetting by construction"
  statement survives contact with the actual benchmark rather than just
  the algebra.
* The baseline picture is harsher than the old single-seed notes
  suggested.  On the class-incremental split, `mlp_finetune` and
  `mlp_ewc` both collapse to ~20.0% final accuracy, while replay climbs
  back to 86.5% but still trails the NEF pair by 10.5 pp.  That is a
  cleaner story than "NEF forgets less": on this regime, sufficient-
  statistics accumulation lands on the joint solution exactly, while
  Fisher-style protection does not remain competitive.
