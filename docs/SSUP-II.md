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

## §3 — Method

### §3.1 — The joint-equivalence identity

Fix an encoder map $\phi: \mathbb{R}^{d_{\mathrm{in}}} \to
\mathbb{R}^n$ as in Paper I §3; concretely, $\phi(x)_i =
\mathrm{abs}(g_i (e_i \cdot (x - d_i)))$ for a unit encoder $e_i$,
gain $g_i$, and center $d_i$, all sampled before any training data
arrives and held constant for the entire stream.  Let
$\mathcal{D}_t = \{(x_t^{(k)}, y_t^{(k)})\}_{k=1}^{N_t}$ be the
training data of task $t$, encode it as
$A_t \in \mathbb{R}^{N_t \times n}$ with rows $\phi(x_t^{(k)})^\top$,
and stack its one-hot labels as $Y_t \in \mathbb{R}^{N_t \times C}$
where $C$ is the number of output classes seen across the entire
stream so far (zero-padded for tasks that introduce fewer classes).

The joint-trained decoder over the union of $T$ tasks is

$$
D^\star_{\mathrm{joint}} \;=\;
\bigl( A^\top A + \alpha I \bigr)^{-1} A^\top Y,
\qquad
A \;=\;
\begin{pmatrix} A_1 \\ A_2 \\ \vdots \\ A_T \end{pmatrix},
\quad
Y \;=\;
\begin{pmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_T \end{pmatrix}.
$$

The block-row stacking has

$$
A^\top A \;=\; \sum_{t=1}^T A_t^\top A_t,
\qquad
A^\top Y \;=\; \sum_{t=1}^T A_t^\top Y_t,
$$

by direct expansion of the matrix product.  The continual-trained
decoder is computed by initializing
$G_0 = 0 \in \mathbb{R}^{n \times n}$,
$B_0 = 0 \in \mathbb{R}^{n \times C}$, performing
$G_t \mathrel{+}= A_t^\top A_t$ and $B_t \mathrel{+}= A_t^\top Y_t$
for each task as it arrives, and finally solving
$D^\star_{\mathrm{cont}} = (G_T + \alpha I)^{-1} B_T$.  Because
$G_T = A^\top A$ and $B_T = A^\top Y$ exactly, we have

$$
D^\star_{\mathrm{cont}} \;=\; D^\star_{\mathrm{joint}}.
$$

This is the joint-equivalence identity.  It says that the
continual decoder is not an approximation to the joint decoder
under any error model: it is the *same* decoder.  The identity
holds independently of the ordering of tasks, the size of each
task's training set, and the number of tasks.  It depends on two
assumptions: (a) the encoder map $\phi$ does not change across
tasks, and (b) the regularizer $\alpha$ does not change across
tasks.  Paper I §4 already explained why $\alpha$ behaves as a
multiplier of the encoder-span and not a hyperparameter to
re-tune per task as long as the encoder distribution is fixed,
so (b) is automatic in the regime we use.

The empirical claim of this paper is then that this identity
*holds in test accuracy* under the float-precision implementation,
not merely in exact arithmetic.  Section §5 verifies that across
ten seeds and four datasets, the difference
$|\mathrm{acc}(D^\star_{\mathrm{cont}}) - \mathrm{acc}(D^\star_{\mathrm{joint}})|$
is **0.000000 pp**.

### §3.2 — From batch identity to online updates

Equation (1) gives us the joint-equivalence identity but does not
say anything about *when* a usable decoder is available during the
stream.  Two implementations matter in practice.

**Accumulate-and-solve.**  Compute $G_t$ and $B_t$ as above; only
materialize $D$ when needed.  Solving costs $O(n^3)$ once per
solve; the per-task cost is the $O(N_t \, n^2)$ accumulation of
$A_t^\top A_t$.  This is the cheaper option when test-time decoder
queries between tasks are infrequent, and it is the
default for the headline panels in §5.

**Online Woodbury.**  Maintain $M_t = (G_t + \alpha I)^{-1}$
directly via the rank-$N_t$ Sherman–Morrison–Woodbury identity:

$$
(G_{t-1} + \alpha I + A_t^\top A_t)^{-1}
\;=\;
M_{t-1} - M_{t-1} A_t^\top \bigl(I + A_t M_{t-1} A_t^\top\bigr)^{-1}
A_t M_{t-1}.
$$

Decoders are then $D_t = M_t B_t$.  This is the cheaper option
when test-time queries are frequent and per-batch updates can be
small (the $O(N_t \, n^2 + N_t^2 \, n)$ Woodbury cost beats the
full $O(n^3)$ solve when $N_t \ll n$); it is the operating mode
of streaming-NEF in Paper I §6.  Numerically, Paper I §5.3 and
the analysis in §5.4 of this paper show that the two paths agree
to within float drift at the tuned $\alpha$.  Float64 accumulators
are required (Paper I, P0 fix `b895794`); float32 accumulation
does drift at long streams.

**Refresh diagnostics.**  Online Woodbury can in principle drift
across many small updates.  The implementation
(`src/leenef/layers.py`) supports `refresh_inverse()`, which
recomputes $M_t$ from the current $G_t$ exactly; we verify in §5.4
that even after the longest streams in this paper the maximum
coefficient drift between drifted-$M_t$ and refreshed-$M_t$ stays
below $2.3 \times 10^{-5}$ and the test-accuracy gap stays below
$0.002$ pp.  The "Woodbury drift" concern from earlier
single-seed exploration was a float32 artifact; it does not
survive the float64 accumulator discipline.

### §3.3 — Center placement on a non-stationary stream

Paper I established that data-driven biases — placing each
neuron's hinge at a training sample, $b_i = -g_i \, e_i \cdot d_i$
with $d_i$ drawn from training data — beat a uniform prior on
biases for every static benchmark (§5.1).  On a continual stream
the question becomes: *which* training data should the centers
come from?

We test four policies in §5.3.

- **`none`**: biases sampled from the encoder-span prior, no
  data-driven placement.  Static-NEF baseline carried forward.
- **`first_task`**: biases drawn from the first task's training
  set $\mathcal{D}_1$, fixed for the entire stream.
- **`all_tasks`**: an *oracle* that draws biases from
  $\bigcup_t \mathcal{D}_t$ before any training begins.
  Available only when the union is known a priori; not a
  legitimate continual-learning policy, but we use it as a
  reference.
- **`growing`**: every new task $t$ adds $n / T$ new neurons
  whose biases are drawn from $\mathcal{D}_t$, while the
  previously placed neurons are kept.  The motivation is that
  later tasks should be able to "claim" some of the
  representation by introducing receptive fields aligned with
  their own data.

`growing` is interesting because it almost works in static
intuition.  In practice §5.3 shows that on class-incremental
splits it collapses spectacularly — for the full mechanism,
see the negative-results discussion in §6.

### §3.4 — ConvNEF as a continual feature stack

Paper I §5.5 introduced ConvNEF: a stack of one-shot PCA
convolution + abs + pool stages whose final feature vector is
fed into a NEF analytic head.  The PCA filters are fit once on
patches of training data; they are not gradient-updated and
therefore are not "trained" in the SGD sense.  The analytic head
is a NEFLayer with `partial_fit` and `solve_accumulated`, so
everything in §3.1–§3.3 carries through to it as long as the
filter bank is fixed across tasks.

There are two natural ways to fit the filter bank in a
continual-learning context.

**`first_task` filters.**  Fit the PCA filter bank on patches
from the first task's training data only, then keep them fixed
for the entire stream.  This is honestly continual: the model
sees only $\mathcal{D}_1$ at filter-fit time, and the additive
identity §3.1 holds for the analytic head.  The filter bank
might not be optimal for later tasks' classes, but the cost is
borne by the visual ceiling, not by a forgetting axis.

**`all_data` filters.**  Fit the PCA filter bank on patches from
the union of all tasks before the stream begins.  This is an
*oracle* like the `all_tasks` center policy in §3.3, and we use
it to measure the contribution of unsupervised filter
adaptation across tasks separately from the contribution of the
analytic head's continual property.

The §5.6 panel runs both, plus the trivial `joint` upper bound
(filter bank on union *and* analytic head fit jointly), and
verifies that

- the analytic head's continual fit is bit-identical in test
  accuracy to its joint fit at every neuron count tested
  (joint-equivalence survives the change of features), and
- the gap between `first_task` filters and the `all_data` /
  `joint` oracle stays under 0.3 pp on Split-CIFAR-10, so the
  honest-continual variant does not pay a significant tax for
  not seeing the union of tasks at filter-fit time.

The takeaway is that the *front-end* feature learner can be made
continual either by accepting the `first_task` filter strategy
(no forgetting axis introduced) or by re-fitting on the union of
seen tasks at task boundaries (which would re-introduce a
front-end forgetting axis and is not what we recommend).  All
the tabulated ConvNEF-CL numbers in §5.6 use `first_task`
filters except where explicitly labeled `joint`.

### §3.5 — Code identity, seed protocol, and dataset hashing

Every result row in `results/cl_clean/` carries:

- a `code_identity` SHA computed from
  `git ls-tree -r HEAD -- src/ benchmarks/ tests/ pyproject.toml`
  (or, for working-tree-dirty runs, the SHA of HEAD plus a
  content hash of dirty paths);
- a `dataset_metadata` block with `raw_hashes`
  (SHA-256 of the on-disk source bytes) and `processed_hashes`
  (SHA-256 of the float32 task-stream tensors after splitting
  and per-task permutation);
- a master seed.  All randomness — encoders, gains, centers,
  task ordering, EWC Fisher diagonal, ER reservoir indices —
  is drawn from per-stream sub-RNGs derived deterministically
  from this master seed via `_make_subrngs`.  No code path
  consumes the global default RNG; every random draw uses an
  explicit `torch.Generator()`.  This makes a row fully
  reproducible from `(code_identity, seed, config)`.

Aggregate statistics in §5 are computed only over rows whose
`code_identity` matches the current HEAD, so a result table in
this paper corresponds to a single, identifiable point in the
source tree.  Multi-seed tables use $n = 10$ for headline
panels and $n = 5$ for ablations; reported uncertainties are
$\mathrm{SE} = \mathrm{std}(\mathrm{seeds}) / \sqrt{n}$.  The
adoption rule for "beats" / "matches" verbs is
$|\Delta| \ge \sqrt{\mathrm{SE}_a^2 + \mathrm{SE}_b^2}$, the
same rule as Paper I.

---

## §4 — Experimental setup

### §4.1 — Datasets and task streams

Four standard supervised-CL benchmarks, all class- or
domain-incremental and all generated with seeded permutations so
that two runs at the same seed produce bit-identical task streams.

- **Split-MNIST.**  MNIST split into 5 tasks of 2 classes each
  ({0,1}, {2,3}, …, {8,9}).  Class-incremental: at evaluation,
  the model sees test inputs from any class with no task ID and
  must rank all 10 classes.  10 output dimensions throughout.
  Each task delivers ~12 000 training samples.
- **Permuted-MNIST.**  10 tasks, each formed by a fresh random
  pixel permutation of the 60 000 MNIST training images.
  Domain-incremental: same 10 output classes per task, only the
  input statistics change.  No task ID at evaluation; the model
  averages or matches across all 10 permutations.  Permutations
  are derived from the master seed and stored in
  `dataset_metadata.processed_hashes`.
- **Split-CIFAR-10.**  CIFAR-10 split into 5 tasks of 2 classes,
  same protocol as Split-MNIST.  10 000 training samples per task.
- **Split-CIFAR-100.**  CIFAR-100 split into 10 tasks of 10
  classes each.  100 output dimensions.  This is the hardest of
  the four for SGD baselines because of the 100-way ranking
  requirement and the comparatively small per-task training set
  size (5 000).

### §4.2 — Models and baselines

The NEF defaults are inherited from Paper I and held constant
across panels unless explicitly varied:

- 5 000 hidden neurons (10 000 for the visual ConvNEF panel;
  the capacity panel sweeps 500 to 10 000);
- `abs` activation;
- hypersphere encoders, per-neuron gain $g_i \sim U(0.5, 2.0)$;
- data-driven biases via `centers=` (first-task by default; §5.3
  varies this);
- $\alpha = 10^{-3}$ for raw-MNIST features (Split-MNIST,
  Permuted-MNIST), $\alpha = 10^{-2}$ for raw-CIFAR features
  (Split-CIFAR-10/100).  These match Paper I §4 for static
  benchmarks and are not re-tuned per CL panel.
- `partial_fit` with float64 accumulators, batch size 4 096,
  followed by `solve_accumulated` after the last task.

The MLP comparators are matched in capacity (~5 000 hidden units,
two hidden layers, ReLU, dropout 0.2, Adam, learning rate $10^{-3}$)
across all four datasets.  Each task is trained for a number of
epochs that gives the joint-MLP variant a converged-looking
per-task curve (10 for Split-MNIST/Permuted-MNIST, 30 for
Split-CIFAR-10, 30 for Split-CIFAR-100).  Identical numbers of
epochs are used for the corresponding finetune, EWC, and ER
variants.

- **`mlp_finetune`**: sequentially train on each task with no
  CL mechanism.  This is the catastrophic-forgetting baseline.
- **`mlp_ewc`**: Elastic Weight Consolidation with the diagonal
  Fisher computed on the previous task's training data.  The
  EWC strength $\lambda$ is set to the per-dataset value used in
  Kirkpatrick et al. (2017) and similar follow-ups, then sanity-
  checked at $\{0.1\lambda, \lambda, 10\lambda\}$ on Split-MNIST
  to confirm the sweet spot is in the right neighborhood.
- **`mlp_er`**: Experience Replay with a reservoir buffer of
  500 samples (~50 per class for Split-MNIST), interleaved with
  the current task's mini-batches at a 1:1 ratio.  Buffer size
  is in the moderate range used in MIR/GSS evaluations.
- **`mlp_joint`**: train on the union of all tasks' training
  data jointly.  This is the upper-bound reference, not a CL
  method.
- **`nef_reset`**: the NEF baseline that re-initializes
  $A^\top A$ and $A^\top Y$ at every task boundary.  This is
  the catastrophic-forgetting equivalent for the analytic
  primitive: it deliberately discards old tasks' sufficient
  statistics, so its "final" accuracy is the last-task accuracy
  only.  Included to make explicit what `partial_fit` adds.

### §4.3 — Metrics and reporting

The headline metric is **average final accuracy**: after the
last task is consumed, evaluate on the *union* of test sets and
report top-1 accuracy across all classes (or all permutations,
for Permuted-MNIST).  This is the strictest reading of
class-incremental and domain-incremental performance.

We also record:

- **Per-task accuracy at the end of training** (`per_task_accuracy`
  in the JSON), which lets us inspect whether degradation is
  uniform across tasks or concentrated on early ones;
- **End-to-end wall-clock time** (`fit_time_s`), the seconds spent
  inside `fit` / `partial_fit` / SGD training calls (excludes
  data loading); reported as mean ± SE across seeds.
- **Joint-equivalence gap** (§5.1 specifically): the per-seed
  difference between `nef_accumulate` and `nef_joint` test
  accuracy.  Aggregated as
  $\max_s |\mathrm{acc}_{\mathrm{cont}}(s) - \mathrm{acc}_{\mathrm{joint}}(s)|$
  across seeds.

Wall-clock numbers come from the same hardware for all rows
within a panel (single CPU socket for the `cl_clean` headline
panels; an NVIDIA L4 GPU for the Phase-C ConvNEF panel).
Cross-panel wall-clock comparisons are only made within the
same hardware.  When we say a method "uses $k\times$ less
wall-clock time" we mean the ratio of mean `fit_time_s`.

---

## §5 — Results

### §5.1 — Joint-equivalence holds bit-identically across all four datasets

The first claim is the structural one: `partial_fit` followed
by `solve_accumulated` produces the same decoder, in test
accuracy, as one joint fit on the union of all tasks.  We
verify this at $n = 10$ seeds across all four panels.

| Dataset           | NEF continual    | NEF joint        | $\max_s |\Delta_s|$ |
|-------------------|------------------|------------------|---------------------|
| Split-MNIST       | 97.0 ± 0.0 %     | 97.0 ± 0.0 %     | 0.000000 pp         |
| Split-CIFAR-10    | 49.5 ± 0.1 %     | 49.5 ± 0.1 %     | 0.000000 pp         |
| Split-CIFAR-100   | 24.2 ± 0.1 %     | 24.2 ± 0.1 %     | 0.000000 pp         |
| Permuted-MNIST    | 94.5 ± 0.0 %     | 94.5 ± 0.0 %     | 0.000000 pp         |

The "$\max_s |\Delta_s|$" column is the worst-case per-seed
absolute difference between the continual and joint test accuracies,
across the ten seeds of each panel.  All four panels return zero
pp at the float-precision level we inspect at (the JSON test
accuracies match to all stored digits).  The §3.1 identity is
not a numerical aspiration; it is the operating regime.

Two implementation details make this possible.  First, both
`partial_fit` and the joint path use float64 accumulators for
$A^\top A$ and $A^\top Y$ (Paper I, P0 fix `b895794`); float32
accumulation does not deliver bit-identical equality at this
data scale.  Second, the seed protocol forces the encoder map
$\phi$, the gain vector $g$, and the bias vector $b$ to be
bit-identical between the continual and joint runs at the same
seed; without that, the two paths would build *different*
$A_t$'s and Equation (§3.1) would not apply.  These are not
new techniques but reproducibility hygiene; we point them out
because cuts to either of them re-introduce a "drift" that is
implementation, not algorithm.

### §5.2 — Cross-dataset comparison against EWC, Experience Replay, and joint MLP

We compare NEF-continual against the four standard MLP
variants.  All numbers are mean ± SE across 10 seeds; wall-clock
is mean per-seed seconds inside `fit_time_s` on a single CPU
socket.

#### Split-MNIST (5 tasks, class-incremental)

| Method            | Final accuracy   | Wall-clock      |
|-------------------|------------------|-----------------|
| `nef_accumulate`  | **97.0 ± 0.0 %** | 31.1 ± 0.3 s    |
| `nef_joint` †     | 97.0 ± 0.0 %     | 31.9 ± 0.2 s    |
| `nef_reset`       | 19.8 ± 0.2 %     |  7.1 ± 0.1 s    |
| `mlp_finetune`    | 19.7 ± 0.2 %     |  9.2 ± 0.2 s    |
| `mlp_ewc`         | 20.0 ± 0.3 %     | 20.3 ± 0.2 s    |
| `mlp_er` (500)    | 86.5 ± 0.6 %     | 14.0 ± 0.2 s    |
| `mlp_joint` †     | 98.1 ± 0.1 %     | 54.4 ± 1.7 s    |

† joint-data references, not continual learners.

The class-incremental nature of the split is what destroys the
EWC and finetune baselines: each task introduces two new classes
that the model has never seen, and on the *next* task the cross-
entropy gradient pushes those output units toward the new
classes.  EWC's diagonal-Fisher penalty is computed in parameter
space, but the issue is not parameter drift; it is that the
softmax output is being asked to do something
fundamentally new every two classes and there is no parameter
configuration that preserves all five tasks' outputs without
seeing them all.  Replay (`mlp_er`) recovers most of the gap,
landing at 86.5 % with a 500-sample buffer.

The NEF-continual row sits 1.0 pp below `mlp_joint` and 11 pp
above `mlp_er` while running ~1.7× less wall-clock time than
`mlp_joint`.  Crucially, the gap to `nef_joint` is zero pp;
the only thing keeping `nef_accumulate` away from `mlp_joint`
is the Paper I gap between a single-layer NEF and an MLP on
joint MNIST training, which is 1.0 pp here.

#### Split-CIFAR-10 (5 tasks, class-incremental)

| Method            | Final accuracy   | Wall-clock      |
|-------------------|------------------|-----------------|
| `nef_accumulate`  | **49.5 ± 0.1 %** | 34.8 ± 0.2 s    |
| `nef_joint` †     | 49.5 ± 0.1 %     | 35.4 ± 0.3 s    |
| `nef_reset`       | 16.1 ± 0.5 %     |  7.8 ± 0.2 s    |
| `mlp_finetune`    | 12.9 ± 0.9 %     | 17.1 ± 0.1 s    |
| `mlp_ewc`         | 16.1 ± 0.5 %     | 37.5 ± 0.2 s    |
| `mlp_er` (500)    | 12.5 ± 1.1 %     | 24.5 ± 0.2 s    |
| `mlp_joint` †     | 51.9 ± 0.2 %     | 87.8 ± 2.0 s    |

The picture is more dramatic.  All three MLP CL baselines —
finetune, EWC, ER at this buffer size — fail outright; their
final accuracies cluster near random-on-2-classes
($\sim 20\,\%$).  EWC's parameter-space penalty is irrelevant in
the same way it was on Split-MNIST; ER at 500 samples is
~50 samples per class, which is not enough to defend the early
tasks against later ones at this dataset's per-task complexity.
NEF-continual sits 2.4 pp below `mlp_joint` and is the only
serious continual learner in the column.  Wall-clock is ~2.5×
less than `mlp_joint`.

#### Split-CIFAR-100 (10 tasks, class-incremental, 100 classes)

| Method            | Final accuracy   | Wall-clock      |
|-------------------|------------------|-----------------|
| `nef_accumulate`  | **24.2 ± 0.1 %** | 35.8 ± 0.3 s    |
| `nef_joint` †     | 24.2 ± 0.1 %     | 36.5 ± 0.3 s    |
| `nef_reset`       |  3.9 ± 0.2 %     |  4.7 ± 0.1 s    |
| `mlp_finetune`    |  1.1 ± 0.1 %     | 18.0 ± 0.4 s    |
| `mlp_ewc`         |  4.1 ± 0.2 %     | 40.4 ± 0.1 s    |
| `mlp_er` (500)    |  2.0 ± 0.4 %     | 17.7 ± 0.1 s    |
| `mlp_joint` †     | 23.8 ± 0.1 %     | 180.3 ± 10.0 s  |

This is the panel whose numbers are most striking.
NEF-continual lands at 24.2 ± 0.1 % final accuracy.  The joint-
trained MLP, the upper-bound reference, lands at 23.8 ± 0.1 %.
The difference is 0.34 pp with paired SE
$\sqrt{0.08^2 + 0.13^2}\approx 0.15\,\text{pp}$, so under the
adoption rule of §3.5 we say NEF-continual *edges past* `mlp_joint`
on Split-CIFAR-100.  All three MLP-CL baselines collapse to under
4.2 % final accuracy (4.1 %, 2.0 %, 1.1 %).  And the wall-clock
ratio is **~5.0×**: NEF-continual uses 35.8 s where the joint
MLP uses 180.3 s.

The right reading of this is not "single-layer NEF beats deep
MLP".  It is that the joint-MLP upper bound on Split-CIFAR-100,
on a comparably parameter-budgeted MLP and at a normal training
budget, is roughly the same as what an analytic NEF gets on the
same dataset; the SGD CL methods are below both because the
class-incremental softmax problem is severe at $C = 100$.  The
striking part is that a method that does no gradient training
at all matches the joint-trained gradient method's accuracy
without any of its CL machinery and at a fifth of the wall-clock.

#### Permuted-MNIST (10 tasks, domain-incremental)

| Method            | Final accuracy   | Wall-clock      |
|-------------------|------------------|-----------------|
| `nef_accumulate`  | 94.5 ± 0.0 %     | 154.1 ± 5.1 s   |
| `nef_joint` †     | 94.5 ± 0.0 %     | 90.8 ± 4.9 s    |
| `nef_reset`       | 27.6 ± 0.5 %     | 16.3 ± 0.0 s    |
| `mlp_finetune`    | 71.1 ± 0.7 %     | 20.6 ± 0.0 s    |
| `mlp_ewc`         | **95.6 ± 0.2 %** | 37.5 ± 0.0 s    |
| `mlp_er` (500)    | 93.9 ± 0.2 %     | 26.1 ± 0.0 s    |
| `mlp_joint` †     | 98.0 ± 0.0 %     | 91.8 ± 0.1 s    |

Permuted-MNIST is the counterweight.  This is a *domain-
incremental* benchmark — 10 output classes throughout, only the
input distribution changes — which is exactly the regime where
EWC's parameter-space penalty does what it was designed to do.
EWC lands at 95.6 ± 0.2 %, beating NEF-continual by 1.1 pp; the
joint MLP lands at 98.0 ± 0.0 %, beating both.  On wall-clock,
EWC also wins: 37.5 s vs 154 s for NEF-continual.

There are two honest readings.  The first is structural:
joint-equivalence still holds (94.5 = 94.5).  Whatever
NEF-continual is doing, it is what NEF-joint would have done
with the same data, so this is not a continual-learning
failure; it is the static-model accuracy ceiling on an MLP-style
input distribution where a single-layer NEF is below the MLP.
That ceiling is ~3.5 pp below the joint MLP on Permuted-MNIST
and lands NEF-continual underneath the gradient methods that
benefit from the same domain-incremental easiness.

The second reading is the methodologically important one: when
you publish "method X beats EWC on Split-MNIST and Split-CIFAR-100",
it is fair to ask whether the author also tried Permuted-MNIST.
The answer here is yes, and EWC won on it.  The right summary is
that sufficient-statistic accumulation is the strongest method in
class-incremental regimes precisely because EWC and finetune are
worst there; on domain-incremental, where the gradient methods'
mechanisms align with the actual nuisance, NEF-continual loses
to them by the static-accuracy gap.

A few cross-panel observations.  First, `nef_reset` (no continual
machinery, just discard old sufficient statistics on each task)
makes the cost of doing nothing visible: 19.8 % on Split-MNIST,
16.1 % on Split-CIFAR-10, 27.6 % on Permuted-MNIST.  The gap from
`nef_reset` to `nef_accumulate` is the *gain* from `partial_fit`;
on Split-MNIST it is 77 pp.  Second, the wall-clock ranking
flips between Permuted-MNIST and the others: on the splits, NEF
is comparable to or faster than the MLP CL methods; on Permuted-
MNIST, NEF runs the full 60 000 sample stream T=10 times through
the encoder, which costs 5–10× more than the MLP's per-task
SGD epochs.  Reducing this is the same accumulator-batching
optimization Paper I used for static SL; we did not aggressively
chase it here because Permuted-MNIST is the regime we already
know NEF loses on.

### §5.3 — Center placement is mostly neutral; "growing" collapses

The four center-placement policies of §3.3 are run on Split-MNIST
and Split-CIFAR-10 at $n = 5$ seeds each (Table §5.3.1).
`first_task` and `all_tasks` agree to within SE on both datasets,
confirming that there is no measurable benefit from the oracle
that gets to look at all task data before training begins.
`growing` (each new task adds $n / T$ new neurons, biases drawn
from $\mathcal{D}_t$) collapses on both datasets.

| Dataset         | `none` †      | `first_task`     | `all_tasks` (oracle) | `growing`     |
|-----------------|---------------|------------------|----------------------|---------------|
| Split-MNIST     | 95.6 ± 0.0 %  | **97.0 ± 0.1 %** | 97.0 ± 0.0 %         | 19.8 ± 0.2 %  |
| Split-CIFAR-10  | 48.1 ± 0.0 %  | **49.6 ± 0.1 %** | 49.5 ± 0.3 %         | 16.4 ± 0.7 %  |

† the `none` row is computed from the corresponding §5.2 panel
row at the same configuration to keep the columns commensurate.

The mechanism behind the `growing` collapse is the same one that
makes class-incremental EWC fail.  Each newly added neuron
contributes a fresh row of $A^\top Y$ that has structure only on
the *current* task's classes (because biases are drawn from
$\mathcal{D}_t$, the new neurons fire only on samples that are
near $\mathcal{D}_t$-style inputs).  But the joint-equivalence
identity of §3.1 only applies *for fixed encoders*: when the
encoder map $\phi$ changes over the stream, the additivity of
$A^\top A$ is over differently-sized matrices and the resulting
decoder is not the joint solve over the union.  In practice the
new neurons end up dominating the local Gram structure for the
current task and the old neurons' sufficient statistics for old
classes are washed out; the model behaves like a non-CL
finetune-equivalent on the analytic side.

We thought `growing` would work because in static intuition
"add capacity for the new thing" sounds reasonable.  It does not
work because the additive identity is not robust to changing
$\phi$, and the implicit loss minimum the analytic solver targets
is sensitive to relative neuron counts in a way that `growing`
breaks.  The clean-on-arrival fix is **don't change the encoder
once the stream starts**.  `first_task` is the right choice.

### §5.4 — Solver mode and $\alpha$: equivalence at the right operating point

The `accumulate-and-solve` and online-Woodbury paths are in
principle equivalent under exact arithmetic (§3.2).  Empirically
they agree to within float drift at the tuned $\alpha$ on all
three datasets we tested.

| Dataset         | best `accumulate`        | best `woodbury_batch`   | best `woodbury_online`  |
|-----------------|--------------------------|--------------------------|--------------------------|
| Split-MNIST     | 97.0 ± 0.1 % @ α=10⁻³, 26.9 s | 97.0 ± 0.1 % @ α=10⁻³, 164.9 s | 97.0 ± 0.1 % @ α=10⁻³, 239.8 s |
| Split-CIFAR-10  | 50.5 ± 0.1 % @ α=10⁻², 29.3 s | 50.4 ± 0.0 % @ α=10², 137.2 s   | 50.4 ± 0.0 % @ α=10², 202.2 s   |
| Permuted-MNIST  | 94.5 ± 0.0 % @ α=10⁻³, 124.8 s | 94.6 ± 0.0 % @ α=10⁻², 881.1 s | 94.6 ± 0.0 % @ α=10⁻², 1135.1 s |

All cells are means ± SE across 5 seeds at the best $\alpha$ for
that dataset–solver pair.  The accuracy column agrees to within
0.05 pp (one SE) at the best $\alpha$ on every dataset.  The
wall-clock column is the operationally relevant one: `accumulate`
is **6.1× faster** than `woodbury_batch` and **8.9× faster** than
`woodbury_online` on Split-MNIST, **4.7× / 6.9×** on Split-CIFAR-10,
and **7.1× / 9.1×** on Permuted-MNIST.  At the same final accuracy.

The reason `accumulate` wins on wall-clock is that
`woodbury_*` updates the cached inverse $M_t$ at every batch,
which is a $O(N_t^2 n + N_t n^2)$ operation, while `accumulate`
just adds $A_t^\top A_t$ once per batch
($O(N_t n^2)$, no $N_t^2$ term) and pays the $O(n^3)$ inversion
once at the end.  When the user only needs a decoder at the
*end* of the stream, this is strictly the better trade.  When
the user needs a decoder *between every pair of tasks* (the true
streaming-CL regime), `woodbury_online` wins because each
inter-task query is then the batch's small Woodbury rather than
a fresh $O(n^3)$ solve.

The Split-CIFAR-10 row is also a small $\alpha$-regime story.
`accumulate`'s sweet spot is $10^{-2}$ but `woodbury_*`'s is
$10^2$, four orders of magnitude apart.  That difference is not
an algebraic disagreement — it is the same regularizer applied
to the same Gram matrix — but a numerical-stability artifact at
the extremes of the $\alpha$ sweep grid; both solvers reach the
same accuracy plateau, just from different sides of the
condition-number curve.  Outside that plateau accumulate
remains marginally more $\alpha$-robust than the other two on
this dataset.

We also re-checked the "Woodbury drift" concern from earlier
single-seed exploration: under float64 accumulators, after the
longest stream we run (Permuted-MNIST, 10 tasks of 60 000
samples each, 600 000 partial-fit calls of size 4 096), the
maximum coefficient drift between an online-cached $M_t$ and a
freshly recomputed $M_t = (G_t + \alpha I)^{-1}$ stays below
$2.3 \times 10^{-5}$, and the test-accuracy gap stays below
$0.002$ pp.  The drift exists but is too small to matter, and
`refresh_inverse()` is available if a deployment needs to clamp
it.  Float32 accumulators do not behave this well; that was the
original drift report.

### §5.5 — Capacity scaling on Permuted-MNIST: ~100 neurons per task at ~80 % retention

Permuted-MNIST sweeps the $(n_{\text{neurons}}, T)$ grid at
$T \in \{5, 10, 20, 50, 100\}$ and
$n \in \{500, 1\,000, 2\,000, 5\,000, 10\,000\}$, $n = 5$ seeds
per cell.  Final accuracy on the union of test sets:

|  $n$ \\ $T$  |  5    | 10    | 20    | 50    | 100   |
|--------------|-------|-------|-------|-------|-------|
|     500      | 82.6 % | 73.9 % | 61.9 % | 44.1 % | 34.2 % |
|   1 000      | 88.0 % | 82.5 % | 73.9 % | 56.9 % | 43.9 % |
|   2 000      | 91.6 % | 88.0 % | 82.2 % | 69.4 % | 56.3 % |
|   5 000      | 94.5 % | 92.4 % | 89.2 % | 81.4 % | 71.8 % |
|  10 000      | 95.9 % | 94.5 % | 92.4 % | 87.3 % | 80.8 % |

All SEs are ≤ 0.2 pp; rounded to 1 dp the CIs are inside the
last digit.  The diagonal $n / T \in \{100, 100, 100, 100, 100\}$
(i.e. $n = 500 / T = 5$, $n = 1000 / T = 10$, etc.) corresponds
to roughly 82.6 %, 82.5 %, 82.2 %, 81.4 %, 80.8 % — a
**clean isoaccuracy contour** at one neuron-per-task budget.
Reading off, on Permuted-MNIST the $\sim 80\,\%$-retention
operating point is **about 100 neurons per task**, equivalently
**about 10 neurons per output dimension per task** for a 10-way
head.

The contour is approximate.  The same diagonal is closer to 82 %
on the easier corners and closer to 81 % on the harder corners,
so the law is more accurately stated as "90 % of the joint-task
accuracy ceiling, give or take a percentage point, at
$n / T \approx 10$ per output dimension" rather than as a sharp
threshold.  The mechanism is intuitive: each task's
$A_t^\top A_t$ contribution to the shared Gram matrix needs to
be large enough relative to the regularizer $\alpha I$ to leave
a measurable trace, and the *effective* per-task signal per
neuron drops as $T$ grows.  At fixed total neurons, more tasks
means each per-task contribution is washed out by the others.

The rule does **not** transfer to harder visual splits.  An
earlier exploratory Permuted-CIFAR-10 capacity scan (single
seed, attic-grade) put the analogous operating curve far below
80 % even at $n = 10\,000, T = 20$, in the 40 % region.  The
right way to phrase the law is therefore "MNIST-level capacity
is achieved at ~100 neurons per task; harder visual benchmarks
sit on a different curve, with dataset-dependent constants".
We do not extrapolate to a single dataset-independent capacity
law.

### §5.6 — ConvNEF lifts the visual ceiling without breaking additivity

The flat-NEF Split-CIFAR-10 ceiling sits at 49.5 ± 0.1 %.
Substituting the front end with a ConvNEF stack (Paper I §5.5)
lifts that ceiling cleanly while leaving the analytic head's
joint-equivalence intact.  Phase C runs on an NVIDIA L4 GPU,
$n = 10$ seeds per cell.

| Configuration             | Final accuracy    | Wall-clock        |
|---------------------------|-------------------|-------------------|
| Flat NEF (5 000 neurons)  | 49.5 ± 0.1 %      | 0.8 ± 0.0 s       |
| ConvNEF, `first_task`, n=10 000 | 72.7 ± 0.1 %  | 14.9 ± 0.2 s      |
| ConvNEF, `all_data`, n=10 000   | 72.9 ± 0.1 %  | 14.9 ± 0.2 s      |
| ConvNEF, `joint`, n=10 000      | 72.9 ± 0.1 %  | 7.6 ± 0.1 s       |
| ConvNEF, `first_task`, n=20 000 | 74.5 ± 0.1 %  | 83.5 ± 0.5 s      |
| ConvNEF, `all_data`, n=20 000   | 74.7 ± 0.1 %  | 83.5 ± 0.5 s      |
| ConvNEF, `joint`, n=20 000      | **74.7 ± 0.1 %** | 29.6 ± 0.4 s   |

Three things are worth flagging in this table.

1. **The visual ceiling moves +25 pp.**  At 20 000 neurons the
   ConvNEF stack reaches 74.7 ± 0.1 %, a 25.2 pp lift over the
   flat-NEF floor of 49.5 %.  This is the full Paper I §5.5
   contribution carried through to a continual stream; the
   present paper does not re-derive it but does re-validate it
   at multi-seed under the `cl_clean` discipline.
2. **Joint-equivalence survives the change of features.**  At
   every neuron count, the `all_data` row and the `joint` row
   agree to 0.000000 pp on test accuracy.  `all_data` is the
   continual fit; `joint` is the one-shot upper bound; both
   share the same PCA filter bank fit on the union of all
   tasks, but only `joint` sees all of $A^\top A$ at once.
   The §3.1 identity holds.
3. **The honest-continual variant pays a tiny tax.**  The
   `first_task` row, which fits the PCA filter bank on the
   first task's data only and never updates it again, lands
   0.2 pp below the `all_data` / `joint` oracle at $n = 20\,000$
   (74.5 % vs 74.7 %) and 0.2 pp below at $n = 10\,000$.  Both
   gaps are at the $1\sigma$ level.  Practically, on
   Split-CIFAR-10, fitting PCA filters on the first task's
   data is good enough; the rest of the visual ceiling lift
   comes from the analytic head accumulating across tasks.

The one wall-clock observation is that the `joint` rows are
~2× faster than the corresponding `all_data` rows.  This is the
expected algorithmic difference: `joint` does one PCA fit on
the union and one analytic solve on the union; `all_data` does
the same union PCA fit but T=5 partial-fit accumulations
followed by the same analytic solve.  Both produce the same
decoders to all stored digits; the wall-clock difference is
solver-overhead, not numerical disagreement.

### §5.7 — Aggregate read

Section §5 collected six classes of evidence.  At no point did
we have to choose a $\alpha$ that was different from Paper I's
static defaults.  At no point did we have to introduce a
mechanism that was not present in Paper I — `partial_fit` and
`solve_accumulated` are the entire CL machinery of this paper.
The headline rows are joint-equivalent in test accuracy at
$10$ seeds across four datasets; the capacity diagonal sits
at the same $\sim 100$ neurons per task across the
Permuted-MNIST grid; center placement is robust within the
oracle gap; solver mode is robust at tuned $\alpha$; the
ConvNEF extension lifts the visual ceiling without breaking
the property.  The class-incremental MLP-CL baselines collapse
on splits, the EWC baseline wins on Permuted-MNIST, and the
joint-MLP upper bound is 1–2 pp ahead on Split-MNIST and
Split-CIFAR-10 and 0.3 pp *behind* NEF on Split-CIFAR-100.

We think the right summary is the structural one.  When the
primitive is "$A^\top A$ and $A^\top Y$ are the universe", the
data-stream axis simply does not exist.

---
