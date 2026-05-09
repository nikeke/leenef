# Sufficient Statistics as the Universal Primitive (Part II)

## Continual Learning Without Forgetting, by Construction

Nikša Bosnić  
nikeke@otpisani.net[^ai]

[^ai]: GitHub Copilot (powered by Anthropic Claude Opus and OpenAI
GPT large language models) was used extensively during code
development, drafting, editing, and analysis for this work.  All
final content, judgments, and errors remain the author's
responsibility.

---

## Abstract

The same regularized least-squares solve
$D = (A^\top A + \alpha I)^{-1} A^\top Y$ that Paper I used for static
supervised learning is, under fixed-encoder NEF features, *automatically*
a continual learner.  Because $A^\top A$ and $A^\top Y$ are additive
over data partitions, sequential `partial_fit` over a non-stationary
task stream produces the **same** decoder as joint training over the
union of tasks; the "forgetting" axis has length zero by construction.
We verify this empirically across four standard benchmarks at multi-seed
rigor: on Split-MNIST, Permuted-MNIST, Split-CIFAR-10, and Split-CIFAR-100,
the per-seed accuracy of `nef_accumulate` matches `nef_joint` to
**0.000000 pp** over 10 seeds.  Comparing against EWC, Experience Replay,
and joint-training MLPs, sufficient-statistics accumulation
**collapses the class-incremental forgetting modes** of the gradient
baselines (Split-MNIST: NEF 97.0 % vs MLP-EWC and MLP-finetune
~20.0 %), **edges past the joint MLP upper bound on Split-CIFAR-100**
(NEF 24.15 ± 0.08 % vs MLP-joint 23.81 ± 0.13 %, ~4.5× less wall-clock
time), and **lands within 1.0 pp** of joint-MLP on Split-MNIST and
within 2.4 pp on Split-CIFAR-10.  A capacity scan on Permuted-MNIST
shows that worst-case retention scales smoothly with neuron budget per
task, with an MNIST-level "~100 neurons per task for ~80 % retention"
operating point and a confirmed dataset dependence on harder visual
splits.  Online Woodbury and offline accumulation match within 0.05 pp
at the tuned $\alpha$ while accumulation runs **5–9× faster**, and
center-bank policies turn out not to matter once a single first-task
bank is in place.  A gradient-free ConvNEF front-end lifts the
Split-CIFAR-10 ceiling from 49.5 % to **74.7 ± 0.1 %** without breaking
additivity: the continual `partial_fit` path is bit-identical in
accuracy to the one-shot joint fit at every neuron count tested.  These
results are not "another CL method"; they are the structural consequence
of treating $A^\top A$, $A^\top Y$ as the universe (Paper I, P3) and
letting `partial_fit` be the only operation a learner ever performs on
a task stream.

---

## §1 — Introduction

Catastrophic forgetting is the dominant failure mode of stochastic
gradient descent on a non-stationary input stream.  The standard
remedies — regularization on parameter drift (EWC; Kirkpatrick et al.,
2017; SI; Zenke et al., 2017), replay buffers (Experience Replay;
Robins, 1995; iCaRL; Rebuffi et al., 2017; MIR; Aljundi et al., 2019),
and parameter isolation (PackNet; Mallya & Lazebnik, 2018) — are all
mitigations layered on top of SGD.  They share an architecture in
which forgetting *can* happen in principle, and the method is doing
something to make it happen less.

This paper takes the opposite stance.  We adopt the analytic primitive
of Paper I — fixed encoders, one regularized least-squares solve over
the resulting features — and observe that it has no forgetting axis to
manage in the first place.  Two facts conspire.  First, the per-class
columns of the targets matrix $Y$ are *independent*: only the column
that owns class $c$ is non-zero on samples of class $c$, so adding new
training data for a fresh class does not touch the old class-decoder
columns through $A^\top Y$.  Second, and more importantly, the Gram
matrix $A^\top A$ — the matrix that *is* shared across all decoder
columns — is itself **additive over data partitions**:

$$
A^\top A \;=\; \sum_{t=1}^T A_t^\top A_t,
\qquad
A^\top Y \;=\; \sum_{t=1}^T A_t^\top Y_t,
$$

where $A_t$ is the activity matrix of the $t$-th task's training
samples through the *fixed* encoder.  Sequentially calling
`partial_fit` over a stream of tasks and finally `solve_accumulated`
yields exactly the same decoder as one fit on the union — not "to
within float precision" as a numerical aspiration, but bit-identical
in test accuracy as long as both sides use the same `A_t` (and we run
both sides under one code identity to guarantee that).

The point of this paper is to take that algebraic fact through to a
benchmark-grade evaluation, name what it does and does not say, and
characterize where it stops being useful.  Concretely:

- **What does and does not change with the data stream.**  The
  decoder $D$ is the unique minimizer of the same quadratic objective
  whether the data arrived in five tasks or one, so by construction
  the model has the *same* test accuracy on a task seen long ago as
  it would have had if it were trained on that task only yesterday.
  The phrase "knowledge of old tasks" reduces to "the columns of
  $A^\top Y$ that those tasks contributed", and they sit in the
  accumulator as a faithful record.  In the continual-learning
  literature, "average final accuracy" is one number with a lot
  of room to hide a lot of asymmetric behavior; here it is the same
  number you would get by ignoring the stream.
- **Where the regime is competitive and where it is not.**  Paper I
  established that the analytic primitive is competitive on image
  classification with strong analytic-feature priors (ConvNEF,
  receptive-field encoders) and uncompetitive against deep CNNs.
  We carry that limit straight into Paper II.  On class-incremental
  splits — the regime where SGD baselines collapse most dramatically —
  the analytic primitive wins by a wide margin because it does not
  collapse at all; on domain-incremental Permuted-MNIST it loses to
  EWC and joint MLP because the static-image accuracy ceiling of a
  single-layer NEF is below theirs.  The honest reading is that
  this paper's contribution is structural and the absolute level
  inherits whatever Paper I established.
- **Capacity, not just possibility.**  "No forgetting" is necessary
  but not sufficient.  We measure the multi-seed Permuted-MNIST
  capacity matrix $\{n_{\text{neurons}}\} \times \{n_{\text{tasks}}\}$
  and find an operating curve that, for that benchmark, keeps test
  accuracy above $\sim 80\,\%$ at roughly $100$ neurons per task
  (i.e. about $10$ neurons per output dimension per task for a 10-way
  head).  On harder visual splits even $10\,000$ neurons stay below
  half that retention level, which is the empirical limiter on how
  general we are willing to make the rule.
- **The visual ceiling moves cleanly.**  Phase C imports a
  gradient-free convolutional feature stack (the ConvNEF of Paper I,
  §5.5) as the front end, fits its analytic NEF head on a stream
  of tasks via the same `partial_fit`, and shows that the additive
  property survives the change of features.  The Split-CIFAR-10
  continual ceiling moves from 49.5 % to 74.7 % at 20 000 neurons,
  and the continual fit is bit-identical to the joint fit at every
  neuron count tested.

The contribution is therefore the joint-equivalence theorem and its
multi-seed validation, the scaling law, the ConvNEF extension, and
the negative results that say where this stops being a serious
proposition (Permuted-MNIST domain shift, harder visual splits,
adaptive features that need gradients).  We argue at the end that
this is not an isolated CL method but a continuation of the same
"sufficient statistics as the universal primitive" thesis: Paper I
showed it on the static side, this paper shows that the data-stream
axis has length zero under the same primitive, and Paper III will
show that the target-stream axis does too.

### What we are *not* claiming

We are not claiming this is the strongest continual learner across
all regimes.  Paper I's image-accuracy ceiling carries through:
single-layer NEF cannot match a deep CNN on raw CIFAR even when the
deep CNN is trained jointly, and our ConvNEF is a Coates-style 2011
feature stack rather than a modern encoder.  We are also not
claiming that this primitive trivially extends to feature-adaptive
continual learning.  When the front end itself must change across
tasks, the additive identity holds only with respect to the fixed
component, and Paper I's negative result on purely analytic depth is
the limiter.  We document the working operating regime and label the
boundary; we do not paper over either side of it.

---

## §2 — Background

### Catastrophic forgetting and SGD

Catastrophic forgetting (McCloskey & Cohen, 1989; French, 1999) is
the empirical observation that a connectionist model trained
sequentially on tasks $\mathcal{T}_1, \mathcal{T}_2, \dots$ tends
to lose accuracy on $\mathcal{T}_i$ once $\mathcal{T}_{i+1}$ is
trained.  The mechanism is well understood: SGD optimizes the
*current batch's* loss, parameter updates are local and
non-conservative across tasks, and a non-stationary input
distribution effectively redefines the loss landscape as time goes
on.  The deep learning literature has produced four broad families
of mitigations.

- **Regularization-based methods** add a penalty to the loss that
  resists drift on parameters deemed important to old tasks.
  Elastic Weight Consolidation (EWC; Kirkpatrick et al., 2017)
  uses the diagonal Fisher information; Synaptic Intelligence (SI;
  Zenke et al., 2017) uses path-integrated gradient magnitudes;
  Memory-Aware Synapses (MAS; Aljundi et al., 2018) uses
  output-sensitivity; Online EWC (Schwarz et al., 2018) accumulates
  Fisher across tasks.  All of these slow drift on "important"
  parameters but do not stop it.
- **Replay-based methods** keep a buffer of old samples (Experience
  Replay; Robins, 1995) or a generator for them
  (Generative Replay; Shin et al., 2017), and interleave old samples
  with new ones during training.  iCaRL (Rebuffi et al., 2017),
  GEM (Lopez-Paz & Ranzato, 2017), MIR (Aljundi et al., 2019), and
  GSS (Aljundi et al., 2019) refine the storage and selection
  policies.  Replay works well in practice but trades memory for
  performance and is not online in the strict sense.
- **Parameter-isolation methods** dedicate disjoint parameters to
  each task.  PackNet (Mallya & Lazebnik, 2018), Progressive Nets
  (Rusu et al., 2016), and HAT (Serrà et al., 2018) all variants on
  this theme.  These prevent forgetting by structurally avoiding
  shared parameters but require a task identifier at evaluation
  and grow the model with the stream.
- **Bayesian methods** maintain a posterior over parameters and
  update it with each task's data.  Variational Continual Learning
  (Nguyen et al., 2018) is the most influential variant; the
  Laplace approximation gives the tractable Gaussian-posterior
  formulation that EWC implicitly uses.

These four families share an architecture: SGD on a non-stationary
loss, augmented with one mechanism that *resists* the loss landscape
moving.  The accuracy on old tasks is a quantity the method tries
to preserve, and the failure mode is that it cannot fully succeed.

### Sufficient-statistic accumulation as a fourth family

The analytic primitive of Paper I sits outside the SGD frame.
With fixed encoders, the loss

$$
\mathcal{L}(D) \;=\; \frac{1}{N} \| A D - Y \|_F^2 + \alpha \|D\|_F^2
$$

is convex and quadratic in $D$.  Its minimizer

$$
D^\star \;=\; (A^\top A + N \alpha I)^{-1} A^\top Y
$$

depends on $A^\top A$ and $A^\top Y$ only (we absorb $N$ into an
$\alpha_{\mathrm{eff}}$ as in Paper I §4 and drop the factor for the
remainder of the paper).  Both $A^\top A$ and $A^\top Y$ are sums
over training samples, hence sums over any partition into tasks.
The sequential update rule is then trivially:

$$
G \mathrel{+}= A_t^\top A_t,
\qquad
B \mathrel{+}= A_t^\top Y_t,
\qquad
D^\star \;=\; (G + \alpha I)^{-1} B.
$$

`partial_fit(t)` adds the new task's $A_t^\top A_t$ and
$A_t^\top Y_t$ to the running accumulators.  `solve_accumulated`
returns the unique global minimizer of the joint regularized
loss over all tasks seen so far.  Old tasks' contributions are
not approximated, distilled, or replayed: their sufficient
statistics are sitting in the accumulator unchanged.

This puts the present method into a fourth, structural family
sometimes called **sufficient-statistic** or **online-ridge**
methods (in classical statistics; see e.g. Hoerl & Kennard, 1970;
the streaming version is the recursive least squares of Söderström
& Stoica, 1988).  The closest deep-learning relatives are the
random-feature ELM with online updates (ELM-Online; Liang et al.,
2006), Bayesian linear regression with random features (which is
the conjugate Gaussian-posterior special case), and the analytic
inverse models of LSTD in reinforcement learning (Bradtke & Barto,
1996).  None of these have been benchmark-scale evaluated on
modern continual-learning splits with multi-seed discipline against
EWC- and replay-style baselines.  The contribution of this paper
is to do exactly that, with the discipline carried over from
Paper I and `rl_clean`.

### A note on the four CL evaluation protocols

The continual learning literature distinguishes (van de Ven &
Tolias, 2019):

- **task-incremental**: the task identifier $t$ is given at
  evaluation, the model only has to discriminate within
  $\mathcal{T}_t$;
- **domain-incremental**: the input distribution changes but the
  output space is the same (e.g. Permuted-MNIST: ten output
  classes throughout, only the pixel ordering changes per task);
- **class-incremental**: new classes arrive over time, no task
  identifier is given at evaluation, the model must rank all
  classes seen so far (e.g. Split-MNIST, Split-CIFAR-10/100);
- **task-free / online streaming**: no task boundaries are
  exposed at all.

Sufficient-statistic accumulation does not need task IDs at any
point; the additive identity is unconditional.  We therefore
evaluate on the *hardest* of the supervised regimes
(class-incremental for the splits, domain-incremental for the
permuted variant) and report no task-ID accuracies.  Streaming
without exposed boundaries reduces to setting `partial_fit`'s
mini-batch to one sample, which we use as a sanity check.

### The reservoir computing analog

The fixed-encoder + analytic-decoder pattern is also the
architecture of reservoir computing (Jaeger, 2001;
Maass et al., 2002).  In the reservoir-computing literature the
analytic decoder is more often a one-shot batch ridge regression
rather than an online accumulator.  The continual interpretation
of the same architecture is implicit but rarely made explicit;
see e.g. the "online learning in echo state networks" line of
Jaeger & Haas (2004) for the closest explicit statement.  This
paper differs from that line in its choice of feature space (the
NEF interpretation of Paper I §3 rather than a random recurrent
network), in its evaluation discipline (multi-seed,
code-identity-tagged, against modern CL baselines rather than
single-seed against linear baselines), and in its scope (the
non-stationary stream is the *data*, not the input signal of a
recurrent system).

### Inheritance from Paper I

Everything in Paper I about encoder choice, bias placement,
activation, regularization, accumulate-and-solve numerics, and
ConvNEF feature extraction carries through unchanged.  Paper I is
the prerequisite; the present paper inherits its operating defaults
(`abs` activation, hypersphere encoders, data-driven first-task
biases, $\alpha$ at $10^{-3}$ for raw-MNIST-scale features and
$10^{-2}$ for Split-CIFAR-10 features) and its numerical discipline
(float64 accumulators, float32 features, code-identity-tagged
results).  We do not re-derive any of them here; we point to the
Paper I sections that own each design choice and use them.  The
six principles of Paper I §1 — in particular **P3** ("$A^\top A$
and $A^\top Y$ are the universe") and **P6** ("per-class additivity
is what makes continual learning work") — are the load-bearing
claims of Paper II as a whole.

---
