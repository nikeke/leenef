# cl_clean notes — Paper II (Continual Learning)

The Paper II analog of `attic/docs/paper_insights.md`: an append-only
file where we record observations, wording experiments, and any
quantitative pattern that might end up shaping the manuscript.  The
phase log lives in `cl_clean_sweeps.md`; this file is the *story*
side of the same coin.

The paper series is "SSUP" (Sufficient Statistics as the Universal
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
* Phase C now has a committed Colab package under `colab/cl_clean/`.
  Important design choice: the notebook does **not** carry experiment
  logic.  It clones the scaffold branch, writes GPU config snapshots,
  runs the normal `cl_phase_runner`, and bundles the resulting JSON/log
  files.  The actual ConvNEF-CL experiment lives under
  `benchmarks/cl_clean/`, so the GPU rows carry the same `code_identity`
  discipline as the CPU panels.
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
* Split-CIFAR-10 keeps the same structural story but with a narrower
  accuracy gap to the joint MLP upper bound.  `nef_accumulate` again
  matches `nef_joint` exactly over all 10 seeds, and it crushes the
  continual SGD baselines, but `mlp_joint` still leads by 2.4 pp in
  absolute accuracy.  That suggests the paper should frame the visual
  result as "exact continual equivalence plus a strong efficiency story,"
  not as a blanket claim that the fixed-encoder NEF always dominates the
  best gradient-trained upper bound on harder images.
* Split-CIFAR-100 pushes the story back toward NEF.  On the harder
  100-class split, the NEF pair still matches exactly and now edges past
  `mlp_joint` by 0.3 pp while using about 5× less end-to-end wall-clock
  time.  A useful draft framing is that the fixed-encoder regime is not
  uniformly best on every visual task, but its combination of exact
  continual equivalence and competitive-or-better final accuracy becomes
  stronger as the class-incremental problem gets harsher.
* Permuted-MNIST is the important counterweight.  The additivity claim
  still survives untouched — `nef_accumulate` and `nef_joint` are again
  exactly identical over all 10 seeds — but the competitive story turns
  mixed.  Here `mlp_ewc` and `mlp_joint` both beat the NEF pair in final
  accuracy, while NEF still beats replay and massively beats the reset
  control.  That suggests Paper II should separate two claims cleanly:
  exact continual equivalence is structural, whereas competitive accuracy
  depends on the regime and is strongest on class-incremental splits.
* B3 turns that structural claim into a quantitative scaling law.  On
  the multi-seed Permuted-MNIST matrix, the approximate 80%-accuracy
  frontier lands almost exactly on `(500,5)`, `(1000,10)`, `(2000,20)`,
  `(5000,50)`, `(10000,100)`, i.e. about `100 neurons/task` or
  $n / (T \cdot 10) \approx 10$ for a 10-way head.  The four grid-extreme
  spot checks (`t5/t100 × n500/n10000`) all keep the accumulate-vs-joint
  gap at 0.0000 pp, so the paper can say two things at once: the
  sufficient-statistics identity stays exact at the extremes, and the
  practical question is how much neuron budget preserves useful accuracy.
* The CIFAR-10 confirmation is the important limiter on how boldly to
  state that law.  Capacity still scales monotonically with neurons and
  degrades smoothly with tasks, but even `10000` neurons only reaches
  `43.0%` at 20 tasks, so the `100 neurons/task` constant is not
  universal across datasets.  That pushes the manuscript toward wording
  like "an MNIST-level capacity law with dataset-dependent constants,"
  rather than presenting the ratio as a domain-agnostic theorem.
* The B4 center sweep sharpens the "fixed representation" part of the
  argument.  Removing data-driven centers entirely costs about 1 pp on
  Split-MNIST and 1.5 pp on Split-CIFAR-10, so centering matters; but
  continually expanding or recomputing centers does not buy a robust
  gain over `first_task`, and the old `growing` control collapses
  outright.  That leaves a cleaner message than "center adaptation
  helps": a single first-task center bank is already the stable
  sufficient-statistics regime.
* B5 closes the solver question cleanly.  On both MNIST panels the best
  Woodbury rows only beat `accumulate_a0.001` by a few hundredths of a
  percentage point, which stays inside the 1-SE adoption rule, while fit
  time inflates by roughly 6-9x.  On Split-CIFAR-10 the story changes in
  $\alpha$, not in solver family: `accumulate_a0.01` is the clear winner,
  the best Woodbury rows land within about 0.02 pp of it, and they are
  still far slower.  So the paper should frame B5 as "solver-equivalent
  when tuned, not solver-superior," with a dataset-dependent regularizer
  sweet spot rather than a case for online Woodbury as the new default.
* The online refresh diagnostics matter rhetorically because they remove
  an easy objection.  The best `woodbury_online` rows refresh back to the
  exact-solve decoder with max coefficient drift only on the order of
  $10^{-5}$ and essentially zero refreshed-accuracy gap, so the online
  method is not losing because it drifts numerically; it is losing on fit
  time.
* Phase C changes the visual story substantially.  The imported L4 GPU
  ConvNEF panel lifts Split-CIFAR-10 from the flat NEF floor
  (`49.5%`) to `72.9%` at 10000 neurons and `74.7%` at 20000, so the
  "visual extension" chapter is not cosmetic — it buys 23-25 pp on the
  same task with the same code identity.  That is the strongest current
  evidence that the Paper-II continual story can inherit a much better
  visual front end without giving up the analytic-decoder framing.
* The more delicate C1 result is that continual ConvNEF does not lose to
  the joint ConvNEF upper bound in final accuracy on this panel: the
  `all_data` continual path and the one-shot `joint` fit are exactly tied
  at each neuron count, while the joint fit is 2-3× faster.  So Phase C
  reinforces the same structural theme as B1 — the continual sufficient-
  statistics path can still land on the joint solution — but now in a
  regime where the overall accuracy level is much higher.
