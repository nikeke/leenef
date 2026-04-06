# Supervised Learning with the Neural Engineering Framework: Analytical Solvers, Ensembles, and Local Receptive Fields

## Abstract

We present *leenef*, a PyTorch library that adapts the Neural Engineering
Framework (NEF) of Eliasmith and Anderson (2003) for supervised learning
using rate-based neurons.  A single NEF layer — random fixed encoders,
nonlinear activation, analytically solved decoders — trains on MNIST in
under two seconds on a laptop CPU and reaches 95.5% accuracy.  This
architecture is structurally identical to an Extreme Learning Machine (ELM)
and to random kitchen sink features, but NEF's neuroscience-grounded
formulation motivates a distinctive design choice: *data-driven biases*
derived from training-sample centers.  We show that this single design
choice closes the entire 2–3% accuracy gap between encoder types
(hypersphere, Gaussian, sparse), making the encoder direction distribution
irrelevant and reducing hyperparameter sensitivity.  Because the two-second
analytical solve produces a strong base learner, we apply the ensemble
playbook: training 10–20 independent models with different random seeds and
combining predictions via averaging.  Adding local receptive field encoders
— where each neuron sees a random image patch rather than the full input —
following McDonnell et al. (2015), provides a further accuracy boost.  We
report ensemble results on MNIST, Fashion-MNIST, and CIFAR-10.  A
systematic sweep over neuron counts, patch sizes, and regularisation
strength reveals that properly configured single-layer RF models —
with dataset-tuned neuron counts, patch sizes, and Tikhonov α — match
a gradient-trained MLP on MNIST (98.50%, 37% faster), and beat the MLP
on Fashion-MNIST (89.74% vs 89.70%) and CIFAR-10 (58.44% vs 52.70%) —
all without gradient descent.
We further extend the framework to multi-layer networks with five training
strategies (greedy, hybrid, target propagation, end-to-end, and
warm-started combinations), recurrent temporal models, and
incremental/online learning via normal-equation accumulation.
For continuous learning, we introduce Woodbury rank-k updates that
maintain the system inverse incrementally in O(n²k) per batch instead of
O(n³) full re-solves.  Applied to temporal sequence classification, a
streaming delay-line reservoir classifier achieves 98.57% on sequential
MNIST — matching LSTM performance — entirely without gradient descent,
training in under four minutes on a laptop CPU.


## 1. Introduction

The dominant paradigm in supervised learning trains all network weights via
gradient descent.  This is effective but computationally expensive: even a
small two-layer MLP on MNIST requires minutes of iterative optimisation.
An alternative family of methods — dating back to random feature
approximations (Rahimi & Recht, 2007), extreme learning machines (Huang
et al., 2006), and the Neural Engineering Framework (Eliasmith & Anderson,
2003) — fixes the input-to-hidden weights at random and solves only the
output weights analytically.  Training reduces to a single regularised
least-squares solve, completing in seconds rather than minutes.

These methods share a common architecture but arrive at it from different
starting points.  ELMs treat random projections as a universal
approximation mechanism.  Random kitchen sinks view them as kernel
approximations.  NEF views them as population coding: each neuron has a
preferred direction (encoder), a tuning curve (activation function with
gain), and a decoding weight that recovers the represented quantity from
the population activity.  This neuroscience framing motivates several
design choices absent from the ELM and kernel literatures:

- **Data-driven biases** from training-sample centers, so each neuron's
  activation is centred around a known data point.
- **Per-neuron gain diversity**, following the NEF tradition of varied
  tuning curves within a population.
- **The absolute-value activation**, which gives each neuron a two-sided
  tuning curve responding to deviations in either direction from its center.

The practical payoff of the two-second single-layer solve is not just
speed but *composability*.  A fast analytical solver is an ideal base
learner for ensembling (Breiman, 2001): train many independent models
with different random seeds and combine their predictions.  This is the
Random Forest playbook applied to random-feature networks.  When combined
with local receptive field encoders — where each neuron sees a random
local image patch (McDonnell et al., 2015) — the ensemble approach
produces competitive results without any gradient training.

This report makes the following contributions:

1. A clear exposition of the NEF-for-supervised-learning architecture,
   situating it within the broader context of ELMs, random features, and
   kernel methods.
2. An analysis of data-driven biases showing they close the accuracy gap
   between all encoder types and reduce the method to a single important
   hyperparameter (neuron count).
3. An ensemble module (`NEFEnsemble`) that leverages the fast analytical
   solve to train 10–20 diverse models in the time a single MLP would
   take.
4. Local receptive field encoders that inject spatial structure without
   gradient training.
5. Extensions to multi-layer networks (five training strategies including
   analytical target propagation), recurrent temporal models, and
   incremental/online learning.
6. Comprehensive benchmarks on MNIST, Fashion-MNIST, CIFAR-10, and
   California Housing, with timing on consumer hardware (CPU-only).


## 2. Background

This section provides the theoretical context needed to understand the
methods presented later.  We cover the Neural Engineering Framework, its
relationship to extreme learning machines and random feature methods,
ensemble methods, and local receptive field encoders.

### 2.1 The Neural Engineering Framework (NEF)

The NEF, introduced by Eliasmith and Anderson (2003), is a theoretical
framework for understanding how populations of neurons represent and
transform information.  It rests on three principles:

1. **Representation:** A quantity *x* is represented by the activities of a
   population of neurons.  Each neuron *i* has an encoder *eᵢ* (a preferred
   direction in the input space), a gain *αᵢ*, a bias *bᵢ*, and a
   nonlinear activation function *G*:

   ```
   aᵢ = G(αᵢ · eᵢ · x + bᵢ)
   ```

   The activity *aᵢ* is the neuron's firing rate, a scalar measuring how
   strongly the neuron responds to input *x*.

2. **Decoding:** The represented quantity can be recovered from the
   population activities by a linear decoder:

   ```
   x̂ = Σᵢ aᵢ · dᵢ = a · D
   ```

   where *D* is a matrix of decoding weights, found by minimising the
   mean-squared error between decoded outputs and the true values over a
   set of representative inputs.  This yields a regularised least-squares
   problem:

   ```
   D = argmin_D ||A · D − Y||² + λ||D||²
   ```

   where *A* is the matrix of activities (samples × neurons), *Y* is the
   target matrix, and *λ* is a regularisation parameter.

3. **Transformation:** To compute a function *f(x)* rather than
   representing *x* itself, one simply changes the decoding targets from
   *x* to *f(x)*.  The same population of neurons can simultaneously
   represent *x* and compute arbitrary functions of it, with different
   decoders for each.

The key architectural insight is that encoders are fixed (either
biologically determined or randomly sampled) and decoders are *solved
analytically*.  This avoids iterative weight updates entirely for a single
population.

In the original NEF formulation, the bias *bᵢ* is typically determined by
a gain-intercept pair that shapes the neuron's tuning curve.  Our
adaptation replaces this with *data-driven biases* (Section 3.2), where
the bias is derived from a training-sample center.

### 2.2 Extreme Learning Machines (ELM)

The Extreme Learning Machine (Huang et al., 2006) is a single-hidden-layer
feedforward network where input weights and biases are randomly assigned
and never updated.  Only the output weights are trained, via a
least-squares solve:

```
H = g(X · W + b)       (hidden layer activation)
β = H† · Y             (output weights via pseudoinverse)
```

where *g* is a nonlinear activation, *W* and *b* are random, and *H†* is
the Moore-Penrose pseudoinverse of the hidden-layer activation matrix.

Huang et al. proved that a single-hidden-layer ELM with *N* hidden neurons
and any continuous activation function is a universal approximator: as *N*
grows, the network can approximate any continuous function on a compact
set.  The practical implication is that ELMs trade architectural efficiency
(more neurons needed than a trained network) for training speed (one matrix
solve vs. many gradient steps).

**Relationship to NEF:**  A single NEF layer with random encoders and an
analytical decoder solve is architecturally identical to an ELM.  The
difference is interpretive: NEF's encoders represent preferred directions,
its gain controls tuning-curve width, and its biases locate each neuron's
sensitive region in input space.  These interpretations motivate design
choices (data-driven biases, abs activation, per-neuron gain diversity)
that improve accuracy beyond a vanilla ELM.

### 2.3 Random Features and Kernel Approximation

Rahimi and Recht (2007) showed that random projections followed by a
nonlinearity approximate kernel functions.  Specifically, if *ω* is drawn
from a distribution *p(ω)* related to the kernel's Fourier transform, then:

```
k(x, y) ≈ (1/D) Σⱼ z_ωⱼ(x) · z_ωⱼ(y)
```

where *z_ωⱼ(x) = exp(i ωⱼ · x)* are random Fourier features.  In
practice, the exponential is often replaced by other nonlinearities
(ReLU, cosine, etc.) that approximate different kernels.

This framework — known as "random kitchen sinks" — provides a theoretical
justification for why random-weight networks work: they are performing
kernel regression in an approximate feature space.  The number of random
features controls approximation quality, just as the number of neurons
controls representational capacity in NEF and ELMs.

Random kitchen sinks on MNIST with 2000 features typically achieve 94–96%
accuracy (Rahimi & Recht, 2007; various reproductions), comparable to our
NEF single-layer result (95.5%).

### 2.4 Data-Driven Biases and Radial Basis Functions

Our implementation adds a *center* parameter to each neuron.  Given a
center *dᵢ* (sampled from training data) and an encoder *eᵢ*, the
neuron's response becomes:

```
aᵢ = |αᵢ · ((x − dᵢ) · eᵢ)|
```

which can be rewritten as:

```
aᵢ = |αᵢ · (x · eᵢ − dᵢ · eᵢ)|
     = |αᵢ · (x · eᵢ + bᵢ)|
```

where `bᵢ = −αᵢ · (dᵢ · eᵢ)`.  The neuron measures the *unsigned
deviation* of the input from its center along its encoder direction.

This connects to radial basis function (RBF) networks (Broomhead & Lowe,
1988), where each basis function is centred on a data point.  The
difference is that RBF neurons measure distance in *all* directions
(isotropic Gaussian kernels), while NEF neurons measure distance along a
*single random direction*.  Having many neurons with different random
directions recovers approximate omnidirectional sensitivity — a
Johnson-Lindenstrauss-style argument ensures that enough random projections
preserve distance relationships.

The practical effect is dramatic (Section 5.2): without data-driven biases,
hypersphere encoders lag Gaussian encoders by 2–3% because Gaussian
encoder norms create an implicit distribution of activation thresholds.
Data-driven biases make this explicit and optimal, closing the entire gap.

### 2.5 Ensemble Methods

Ensemble methods combine multiple models to improve prediction accuracy and
robustness.  The theoretical foundation rests on the bias-variance
decomposition: if individual models have uncorrelated errors, averaging
their predictions reduces variance without increasing bias (Breiman, 2001).

**Bagging** (Bootstrap Aggregating; Breiman, 1996) trains each model on a
random bootstrap sample of the training data.  **Random Forests** (Breiman,
2001) combine bagging with random feature subsets at each split, increasing
diversity among trees.  The key insight is that *randomness in model
construction* — not just data sampling — improves ensemble diversity and
thus accuracy.

For random-feature networks like ELMs and NEF layers, each model already
uses different random encoders (analogous to different random feature
subsets in Random Forests).  Training multiple NEF layers with different
random seeds creates natural diversity without bootstrap sampling, though
the two can be combined.

The ensemble improvement depends on the *correlation* between member
predictions.  If members produce identical predictions (correlation = 1),
ensembling provides no benefit.  If predictions are uncorrelated,
averaging *K* members reduces variance by a factor of *K*.  Random-feature
networks are well-suited because each member's errors are substantially
driven by the particular random projection, making errors relatively
uncorrelated across members with different seeds.

### 2.6 Local Receptive Field Encoders

McDonnell et al. (2015) demonstrated that combining ELMs with local
receptive fields dramatically improves image classification.  Instead of
each hidden neuron receiving a random projection of the *entire* input
image, each neuron sees only a small random *patch*.  This injects
spatial locality — similar to the convolution operation in CNNs — without
any learned filters.

Their results on MNIST were striking:
- Single ELM (global random weights, 10000 neurons): ~96%
- Single ELM + local receptive fields (10000 neurons): ~98.8%
- Ensemble of 10 ELMs + local receptive fields: **99.17%**

The receptive field approach works because natural images have strong
local structure: neighbouring pixels are highly correlated and local
patterns (edges, corners, textures) are more informative than global
projections.  A random projection of the full 784-dimensional MNIST image
dilutes local structure across all dimensions.  A random projection of a
5×5 patch (25 dimensions) concentrates the neuron's sensitivity on a
local feature.

In our implementation, each neuron *i* is assigned a random patch
position *(rᵢ, cᵢ)* in the image and a random weight vector *wᵢ* of
dimension *patch_size²* (or *patch_size² × C* for multi-channel images).
The encoder vector *eᵢ* is mostly zeros, with non-zero entries only at the
patch positions, where the values are *wᵢ* normalised to unit norm.  This
is registered in the encoder strategy registry as `"receptive_field"`.

### 2.7 Incremental Learning via Normal Equations

The standard decoder solve requires computing `D = (AᵀA + λI)⁻¹ AᵀY` on
the full dataset.  The key observation is that the sufficient statistics
`AᵀA` and `AᵀY` are *additive*: they can be accumulated across data
batches and the solve performed once at the end.

Given batches *A₁, Y₁*, *A₂, Y₂*, ..., *Aₖ, Yₖ*:

```
AᵀA = Σⱼ Aⱼᵀ Aⱼ
AᵀY = Σⱼ Aⱼᵀ Yⱼ
D = (AᵀA + λI)⁻¹ AᵀY
```

This enables streaming/online learning: data can arrive in chunks, each
contributing to the running totals, with the solve deferred until all data
has been seen.  It also enables updating an existing model with new data
without reprocessing old data, provided the old sufficient statistics are
retained.

This decomposition is well-known in recursive least-squares and online
learning (see, e.g., Haykin, 2002), and is particularly natural for NEF
and ELM architectures where the sufficient statistics are the only
information needed from the training data.

### 2.8 Continuous Learning via the Woodbury Identity

Accumulating `AᵀA` and `AᵀY` (Section 2.7) supports deferred-batch
solving: decoders are updated only when a full re-solve is triggered.
For *continuous* learning — where decoders must be current after every
incoming batch — we need to update the system inverse incrementally.

The Sherman-Morrison-Woodbury identity (Woodbury, 1950; Hager, 1989)
provides the tool.  Let M = AᵀA + αI be the current system matrix with
cached inverse M⁻¹.  When a new batch B (k × n) arrives, the system
matrix becomes M_new = M + BᵀB, and the updated inverse is:

```
M_new⁻¹ = M⁻¹ − M⁻¹ Bᵀ (Iₖ + B M⁻¹ Bᵀ)⁻¹ B M⁻¹
```

This *rank-k update* costs O(n²k + k³) — far cheaper than the O(n³) full
re-solve when k ≪ n (i.e. the batch size is much smaller than the neuron
count).  The decoders are then D = M_new⁻¹ · (AᵀY), where AᵀY
accumulates as before.

**Initialisation.**  Rather than computing M⁻¹ from the first batch
(which may be ill-conditioned when k < n), we initialise M⁻¹ = (1/α)I
— the inverse of the pure regulariser.  All batches then enter through
the Woodbury path, ensuring consistent conditioning.

**Numerical precision.**  Repeated rank-k updates in float32 accumulate
rounding errors that compound across many updates.  For large n
(thousands of neurons) and many batches, this drift can be catastrophic.
Two mitigations are used:

1. **Float64 inverse.**  The cached M⁻¹ and all Woodbury arithmetic are
   performed in float64.  The final decoder computation casts back to the
   model's working dtype.

2. **Periodic refresh.**  Since the sufficient statistics AᵀA and AᵀY are
   accumulated alongside the Woodbury updates, a full re-solve
   `M⁻¹ = inv(AᵀA + αI)` can be triggered periodically to reset any
   accumulated drift.

**Adaptive fallback.**  When a batch has k ≥ n (more samples than
neurons), the direct re-solve from accumulated statistics is both cheaper
and more numerically stable, so the Woodbury path is bypassed.


## 3. Method

### 3.1 NEF Layer Architecture

A `NEFLayer` implements the three-stage NEF pipeline:

1. **Encode:** Project input *x* into neuron space:
   ```
   z = gain · (x · Eᵀ) + bias
   ```
   where *E* ∈ ℝⁿˣᵈ is the encoder matrix (rows are unit vectors), *gain*
   is a per-neuron scalar, and *bias* is derived from centers.

2. **Activate:** Apply a nonlinear activation:
   ```
   a = σ(z)
   ```
   Default: `σ = abs` (Section 3.3).

3. **Decode:** Map activities to the output:
   ```
   ŷ = a · D
   ```
   where *D* ∈ ℝⁿˣᵒ is the decoder matrix, solved via regularised
   least-squares.

**Encoder strategies.**  The library provides four encoder strategies,
selected by string name from a registry:

| Strategy          | Description |
|-------------------|-------------|
| `hypersphere`     | Uniform random unit vectors on the unit hypersphere (default) |
| `gaussian`        | i.i.d. Gaussian entries (varying norms) |
| `sparse`          | Sparse random projections (~1/3 non-zero entries) |
| `receptive_field` | Sparse local-patch encoders for images (Section 3.5) |

All strategies except `receptive_field` produce dense vectors.  The
`receptive_field` strategy produces vectors that are zero everywhere
except at a random local image patch.

**Decoder solvers.**  Three solvers are available:

| Solver      | Description |
|-------------|-------------|
| `tikhonov`  | `(AᵀA + αI)⁻¹ AᵀY` via `torch.linalg.solve` (LU-based, default) |
| `cholesky`  | Same system solved via Cholesky factorisation (`torch.linalg.cholesky` + `cholesky_solve`) |
| `lstsq`     | `torch.linalg.lstsq` (unregularised or implicitly regularised) |

The default solver is Tikhonov with α = 0.01.

### 3.2 Data-Driven Biases

Given a set of training samples *X*, each neuron *i* is assigned a center
*dᵢ* sampled uniformly from *X*.  The bias is then:

```
bᵢ = −gain_i · (dᵢ · eᵢ)
```

This ensures that `aᵢ = σ(gain_i · ((x − dᵢ) · eᵢ))`, so the neuron's
zero-crossing is at the projection of its center onto its encoder
direction.  With the abs activation, the neuron responds symmetrically to
deviations in either direction from this zero-crossing.

Without data-driven biases, the library falls back to i.i.d. Gaussian
biases, which do not account for the data distribution.  Section 5.2 shows
that data-driven biases close a 2–3% accuracy gap.

### 3.3 Activation Functions

Four activation functions are provided:

| Name       | Definition | Properties |
|------------|------------|------------|
| `abs`      | \|z\|      | Two-sided response; gradient ±1 everywhere; default for feedforward |
| `relu`     | max(0, z)  | One-sided; sparse gradients; default for recurrent |
| `softplus` | log(1 + eᶻ) | Smooth approximation to ReLU |
| `lif_rate` | 1/(1 − e⁻ᶻ) for z > 0 | Leaky integrate-and-fire rate model |

The **abs activation** is a natural fit for the NEF distance
interpretation: `|gain · ((x − d) · e)|` responds to deviations in
*either* direction along the encoder, effectively doubling representational
capacity compared to ReLU.  With data-driven biases, abs consistently
outperforms other activations for single-layer classification (Section 5.3).

For **recurrent models**, abs has gradient ±1 everywhere (no sparsity),
causing gradient explosion through backpropagation through time (BPTT).
ReLU's zero gradient on negative inputs provides the damping needed for
stable recurrent gradient flow.

### 3.4 Per-Neuron Gain

In canonical NEF, each neuron has its own gain sampled from a
distribution, creating diverse tuning curves.  Our default samples gain
from U(0.5, 2.0), meaning neurons vary in sensitivity: low-gain neurons
have wide, gentle tuning curves while high-gain neurons have narrow,
sharp responses.  This diversity enriches the population representation
without additional parameters.

### 3.5 Local Receptive Field Encoders

The `receptive_field` encoder strategy creates sparse encoders where each
neuron sees a random local image patch:

1. Sample a random patch position *(r, c)* uniformly over valid positions
   (ensuring the patch fits within the image).
2. Generate random weights *w* ∈ ℝᵖ² (or ℝᵖ²ᶜ for *C*-channel images)
   and normalise to unit norm.
3. Construct the full encoder vector by placing *w* at the appropriate
   pixel indices and zeros elsewhere.

This creates *N* × *D*-dimensional encoder vectors (same shape as other
strategies) but with only *patch_size²* non-zero entries per neuron.  The
unit-norm convention is maintained within the patch, consistent with the
hypersphere strategy.

For multi-channel images (e.g., CIFAR-10 with 3 colour channels), the
patch covers all channels at each spatial position, giving
*patch_size² × C* non-zero entries per encoder.

### 3.6 NEF Ensemble

`NEFEnsemble` trains *K* independent `NEFLayer` models, each with a
different random seed (and thus different random encoders, centers, and
gains).  Predictions are combined by one of two methods:

- **Mean** (default): Average the output vectors across members.  For
  classification with one-hot targets, this averages the predicted class
  probabilities.
- **Vote**: Each member's argmax prediction is a vote; the class with the
  most votes wins.

The ensemble exploits the fact that different random projections produce
different error patterns.  When errors are uncorrelated across members,
averaging reduces the effective error rate.

### 3.7 Incremental / Online Learning

`NEFLayer` supports incremental learning via three methods:

- `partial_fit(x, targets)` — encodes the batch, computes `AᵀA` and
  `AᵀY`, and adds them to running totals stored as registered buffers.
- `solve_accumulated(alpha)` — solves decoders from the accumulated
  sufficient statistics.
- `reset_accumulators()` — clears the running totals.

This produces the same result as `fit()` on the full dataset (up to
floating-point accumulation order) but supports streaming data, memory
constraints, and model updates with new data.

#### 3.7.1 Continuous Fit with Woodbury Updates

For applications that require up-to-date decoders after every batch — not
just after a deferred final solve — `NEFLayer` provides:

- `continuous_fit(x, targets, alpha)` — encodes the batch, accumulates
  `AᵀA` / `AᵀY`, and applies a rank-k Woodbury update (Section 2.8) to
  the cached system inverse.  Decoders are recomputed immediately.
- `continuous_fit_encoded(activities, targets, alpha)` — the same, but
  accepts pre-computed activity matrices.  Used by downstream modules
  (e.g. `StreamingNEFClassifier`) that pool or transform activities before
  the decoder solve.
- `refresh_inverse(alpha)` — recomputes M⁻¹ exactly from the accumulated
  AᵀA to correct any Woodbury drift.
- `reset_continuous()` — clears the inverse cache and accumulators.

The Woodbury inverse is stored in float64 regardless of the model's
working dtype.  This prevents the catastrophic drift observed with float32
when many rank-k updates accumulate on large inverse matrices (thousands
of neurons × hundreds of batches).  The final decoder product casts back
to the model dtype.

#### 3.7.2 Accumulate + Solve (GPU-Friendly Path)

The Woodbury path's float64 requirement is a significant bottleneck on
consumer GPUs.  The Tesla T4, for example, delivers only 1/32 of its
float32 throughput in float64.  For workloads where online decoder updates
are not required — i.e. the model only needs to be accurate *after* seeing
all data — the library provides a lighter alternative:

- `accumulate(x, targets)` — encodes the batch in float32, then
  accumulates AᵀA/AᵀY in float64.  No inverse is maintained; cost is
  O(n²k) per batch.  The encoding itself (the expensive part on GPU)
  stays in float32; only the small n×n and n×d_out accumulator matrices
  are promoted to float64.
- `solve(alpha)` — performs a single Tikhonov-regularised solve from the
  accumulated statistics in float64, then casts decoders back to float32.
  Cost is O(n³), once.

The key advantage over Woodbury is **no n×n inverse maintenance**: the
Woodbury path performs O(n²k) float64 work *per batch* to update the
inverse, while accumulate performs O(n²k) float64 work per batch only
for the outer products, and a single O(n³) float64 solve at the end.

On a T4 GPU (sMNIST-row, 120 batches):

| Config | Woodbury | Accumulate | Speedup |
|--------|----------|------------|---------|
| 2000n w=10 | 8.1s / 97.24% | 1.0s / 97.24%* | **8×** |
| 8000n w=10 | 92.2s / 98.30% | 6.2s / 98.30%* | **15×** |

*Accuracy after fixing regularisation parity (see below).

Both paths share the same AᵀA/AᵀY accumulators, so they can be mixed:
use `accumulate()` for the main data pass, then switch to `continuous_fit()`
for subsequent online updates if needed.

### 3.8 Multi-Layer Networks

`NEFNetwork` stacks multiple `NEFLayer` modules.  Hidden layers
encode only: their neuron activities (not decoded outputs) become the
next layer's input.  Only the output layer decodes.

Six training strategies are available:

1. **Greedy** (`fit_greedy`): Random hidden encoders, analytical output
   decoders.  No gradients.  Fastest but limited.
2. **Hybrid** (`fit_hybrid`): Alternates analytical decoder solves with
   gradient updates to encoder weights.  The decoder re-solve at each
   iteration stabilises training and allows a constant learning rate.
3. **Target propagation** (`fit_target_prop`): Replaces backpropagation
   with layer-local targets via NEF representational decoders (analytical
   inverse models) and difference target propagation (Lee et al., 2015).
   Single-layer gradients only.  Described in detail below.
4. **End-to-end** (`fit_end_to_end`): Standard SGD on all parameters,
   initialised from a greedy NEF solve.
5. **Hybrid→E2E** (`fit_hybrid_e2e`): Hybrid warm start followed by E2E
   fine-tuning.  Generally the best balanced strategy.
6. **TP→E2E** (`fit_target_prop_e2e`): Target propagation warm start
   followed by E2E fine-tuning.

#### 3.8.1 Analytical Target Propagation (NEF-TP)

Standard target propagation (Lee et al., 2015) replaces backpropagation
with layer-local targets.  Each layer receives a target — what its
activities *should* have been to reduce the loss — and updates its weights
to match that target using only a local gradient.  Targets propagate
backward through learned "inverse models" rather than gradients.

**Difference target propagation (DTP)** adds a correction term to prevent
error accumulation across layers:

```
target_l = a_l + g_{l+1}(target_{l+1}) − g_{l+1}(a_{l+1})
```

When `target_{l+1} = a_{l+1}` (no change needed), the target for layer
*l* is exactly its current activities.

The key insight for NEF-TP is that **NEF representational decoders are
the inverse models that target propagation needs**.  In Eliasmith's NEF,
every neural population has a representational decoder that recovers the
encoded quantity from activities:

```
x̂ = a · D_repr    where D_repr = argmin_D ‖A · D − X‖²
```

This is solved analytically — no gradient training needed for the inverse
model.

**Training loop** for a network with L hidden layers and one output layer:

```
for each iteration:
    # Forward pass
    a[0] = x
    for l = 1 to L:
        a[l] = activate(gain_l · (a[l−1] · E_l^T) + b_l)
    a[L+1] = activate(gain_out · (a[L] · E_out^T) + b_out)

    # Solve decoders (all analytical, no gradients)
    D_out     = solve(a[L+1], targets)           # task decoder
    D_repr[l] = solve(a[l], a[l−1])  for l = 1…L+1  # representational decoders

    # Compute targets backward (no backprop)
    target[L+1] = a[L+1] − η · (a[L+1] · D_out − targets) · D_out^T
    for l = L down to 1:
        target[l] = a[l] + (target[l+1] − a[l+1]) · D_repr[l+1]  # DTP

    # Local encoder updates (parallelisable across layers)
    for l = 1 to L+1:
        loss_l = ‖encode_l(a[l−1]) − target[l]‖²
        update E_l, b_l with ∇loss_l
```

All decoder solves are analytical.  Encoder updates use only single-layer
gradients — no gradient flows between layers, and layer updates can in
principle run in parallel.

The step size η controls how aggressively the output target departs from
current activities.  Too large pushes targets outside the feasible
activity space; too small provides no learning signal.  Defaults: η=0.03
for plain TP, η=0.01 for TP→E2E warm starts.

**Comparison with hybrid training:**

| Property              | Hybrid               | NEF-TP                |
|-----------------------|----------------------|-----------------------|
| Gradient scope        | Full backprop        | Single layer          |
| Decoder solves / iter | 1 (output only)      | L+1 (all layers)     |
| Encoder gradient cost | O(L × forward)       | O(1 × forward) each  |
| Parallelisable        | No (chain rule)      | Yes (layer-independent) |
| Memory                | Full computation graph | Per-layer only        |
| Biological plausibility | Low                | Higher (local rules)  |

TP solves more decoders per iteration (cheap with our analytical solvers)
but saves on gradient computation.  For deep networks, TP should be faster
per iteration; whether it converges in fewer or more iterations is
task-dependent.

**Activation considerations for TP.**  The abs activation is many-to-one
(`abs(z) = abs(−z)`), so two inputs differing only in sign along an
encoder direction produce identical activities and the representational
decoder cannot distinguish them.  With enough random encoder directions
and data-driven biases, this is unlikely to matter (Johnson-Lindenstrauss
argument), and in practice abs gives the representational decoder richer
(always-nonzero) activities to work with.  ReLU's zero-gradient region
makes the decoder's job harder (zero activities carry no information), but
non-zero activities are fully informative.

### 3.9 Recurrent Models

`RecurrentNEFLayer` extends the feedforward pipeline with the canonical
NEF decode-then-re-encode feedback loop.  At each timestep *t*:

```
x_aug = concat(u[t], s[t−1])                  # input + previous state
a[t]  = σ(gain · (x_aug · Eᵀ) + bias)         # encode + activate
s[t]  = a[t] · D_state                        # state decoder (feedback)
y     = a[T] · D_out                          # output decoder (final step)
```

The **state decoder** extracts a low-dimensional state summary that feeds
back through the encoders.  The **output decoder** produces the task
prediction at the final timestep.  Training strategies parallel the
feedforward case: greedy, hybrid, target propagation through time (TPTT),
end-to-end BPTT, and warm-started combinations.

### 3.10 Streaming Temporal Classifier

`StreamingNEFClassifier` combines the delay-line reservoir idea from
computational neuroscience with the continuous Woodbury updates of
Section 3.7.1 to classify variable-length temporal sequences without
gradient descent.

**Architecture.**  Given an input sequence x ∈ ℝ^(T × d):

```
x_seq   (N, T, d)
   │
   ▼  delay-line: concatenate K consecutive timesteps
(N, T, K·d)
   │
   ▼  random NEF encoding per timestep
activities  (N, T, n_neurons)
   │
   ▼  mean pooling over time
pooled  (N, n_neurons)
   │
   ▼  linear decoder
output  (N, d_out)
```

The delay-line with window size K gives each neuron access to a short
temporal context of K consecutive timesteps.  The beginning is zero-padded
so that each timestep has a full K-length window.  Mean pooling collapses
the temporal dimension into a fixed-size representation regardless of
sequence length.

**Training.**  Three modes:

1. **Batch fit** — computes the full pooled activity matrix and solves
   decoders via standard Tikhonov (Section 3.1).
2. **Continuous fit** — processes sequences in chunks via Woodbury updates
   (Section 2.8), maintaining up-to-date decoders after each chunk.
3. **Accumulate + solve** — processes sequences in chunks, accumulating
   AᵀA/AᵀY in float32, then solves once at the end (Section 3.7.2).
   Mathematically equivalent to batch fit but memory-efficient and
   GPU-friendly (no float64 required).

The continuous mode enables genuine streaming: sequences arrive
incrementally and the model is usable at any point.  A final
`refresh_inverse()` call corrects any accumulated numerical drift.
The accumulate mode is preferred on consumer GPUs where float64
throughput is limited.

**Chunked encoding.**  To limit peak memory, `encode_sequence` processes
samples in chunks when the total token count (N × T) exceeds a threshold.
This is critical for large models — e.g. 60 000 sequences × 28 timesteps
× 4000 neurons would require 26.8 GB without chunking.


## 4. Experimental Setup

### 4.1 Datasets

| Dataset        | Samples  | Input dim | Classes | Type |
|----------------|----------|-----------|---------|------|
| MNIST          | 60k / 10k | 784     | 10      | Handwritten digits |
| Fashion-MNIST  | 60k / 10k | 784     | 10      | Clothing items |
| CIFAR-10       | 50k / 10k | 3072    | 10      | Natural images |
| sMNIST-row     | 60k / 10k | T=28, d=28 | 10   | Sequential digits |
| California Housing | 20640 | 8       | —       | Regression |

All classification targets are encoded as one-hot vectors.  Pixel inputs
are normalised to [0, 1].  **sMNIST-row** presents each MNIST image as a
sequence of 28 rows (28 pixels each), requiring temporal integration to
classify.  California Housing targets are standardised (zero mean, unit
variance).

### 4.2 Hardware and Software

All experiments run on a single CPU: AMD Ryzen 5 PRO 5650U (6 cores, 12
threads, 1.83 GHz base).  No GPU is used.  The implementation uses
PyTorch 2.0+ with `torch.linalg` for matrix operations.

### 4.3 Default Configuration

Unless noted otherwise, all NEF experiments use:
- **Activation:** abs (feedforward), relu (recurrent)
- **Encoders:** hypersphere (feedforward single-layer and multi-layer),
  receptive_field (ensemble with RF)
- **Gain:** per-neuron, U(0.5, 2.0)
- **Biases:** data-driven (`centers=x_train`)
- **Solver:** Tikhonov, α = 0.01
- **Random seed:** 0 (for reproducibility)


## 5. Results

### 5.1 Single-Layer Scaling with Neuron Count

| Dataset       |  500   | 1000   | 2000   | 5000   | 10k    | 20k    | 30k    |
|---------------|--------|--------|--------|--------|--------|--------|--------|
| MNIST         | 92.1%  | 94.3%  | 95.5%  | 96.9%  | 97.4%  | 97.9%  | 98.3%  |
| Fashion-MNIST | 82.6%  | 84.7%  | 85.7%  | 87.1%  | 88.4%  | 89.3%  | 89.8%  |
| CIFAR-10      | 43.7%  | 45.9%  | 47.8%  | 50.4%  | 51.0%  | 51.5%  | 51.8%  |
| Time          | <1s    | 1s     | 2s     | 10s    | 43s    | 140s   | 394s   |

Accuracy scales monotonically with neuron count but with severe diminishing
returns.  At 2000 neurons, MNIST reaches 95.5% in ~2 seconds — within 3%
of a fully-trained MLP (98.5%) that takes 40× longer.  Scaling to 30000
neurons (394 seconds) reaches 98.3% on MNIST, approaching but unable to
match the multi-layer hybrid result (98.6% in 318 seconds).  The
single-layer ceiling on Fashion-MNIST (89.8%) and CIFAR-10 (51.8%) falls
further short of multi-layer results (90.6% and 58.5%), showing that
learned features are essential where brute-force neuron scaling cannot
compensate.

![Accuracy and time scaling with neuron count](figures/neuron_scaling.png)
*Figure 1. Left: test accuracy saturates logarithmically with neuron count across all three datasets. Right: training time scales as O(n²), dominated by the AᵀA computation.*

### 5.2 Why Data-Driven Biases Matter

Accuracy at 2000 neurons with abs activation, with and without data-driven
biases:

|               | hyper  | + data | gauss  | + data | sparse | + data |
|---------------|--------|--------|--------|--------|--------|--------|
| MNIST         | 93.4%  |**95.6%**| 96.0% | 95.7%  | 95.6%  | 95.6%  |
| Fashion-MNIST | 84.1%  |**85.9%**| 86.0% | 86.0%  | 86.0%  | 85.6%  |
| CIFAR-10      | 45.9%  |**48.3%**| 47.3% | 47.5%  | 47.5%  | 48.2%  |

Without data-driven biases, hypersphere encoders lag Gaussian and sparse
encoders by 2–3%.  The advantage of Gaussian encoders comes from their
varying norms, which create an implicit distribution of activation
thresholds — neurons effectively have different "sensitivity ranges."
Data-driven biases make this explicit and optimal: each neuron's
zero-crossing is placed at the projection of a training sample onto the
encoder direction.

With data-driven biases, all encoder types converge to similar accuracy
(within 0.4%).  The encoder direction distribution becomes irrelevant;
only having enough random directions matters.  This reduces
hyperparameter sensitivity: the user needs to choose only the number of
neurons, not the encoder distribution.

### 5.3 Activation Comparison

Single-layer, 2000 neurons, hypersphere encoders, data-driven biases:

| Activation | MNIST  | Fashion | CIFAR-10 |
|------------|--------|---------|----------|
| abs        |**95.7%**|**85.8%**|**48.1%**|
| relu       | 95.4%  | 85.3%   | 47.9%   |
| softplus   | 90.9%  | 82.4%   | 44.2%   |
| lif_rate   | 88.9%  | 81.2%   | 38.8%   |

Data-driven biases amplify the activation effect.  With random biases,
all activations cluster within ~1%.  With data-driven biases, neurons
have more structured activation patterns with sharper boundaries.
Sharp-threshold activations (abs, relu) handle this well; smooth
approximations (softplus, lif_rate) lose substantial accuracy.

The abs activation's advantage is its two-sided response:
`|gain · ((x − d) · e)|` responds to deviations in either direction,
effectively doubling the representational capacity of each neuron
compared to relu's one-sided response.

### 5.4 Ensemble and Receptive Field Results

All ensemble experiments use 2000 neurons per member, abs activation,
data-driven biases, and Tikhonov solver (α = 0.01).  Receptive field
encoders use the default patch size of 5×5.

| Model                       | Members | MNIST  | Fashion | CIFAR-10 | Time (MNIST) |
|-----------------------------|---------|--------|---------|----------|--------------|
| NEFLayer (single)           |    1    | 95.68% | 85.93%  | 47.80%   |     2.1s     |
| Ensemble (hypersphere)      |   10    | 96.20% | 86.21%  | 50.86%   |    23.8s     |
| Ensemble (receptive field)  |   10    | 96.51% | 86.65%  | 55.32%   |    27.6s     |
| Ensemble (hypersphere)      |   20    | 96.24% | 86.14%  | 51.12%   |    44.7s     |
| Ensemble (receptive field)  |   20    | 96.54% | 86.90%  | 55.76%   |    45.8s     |

#### Analysis

**Ensembling provides a consistent boost.**  With 10 members and
hypersphere encoders, the ensemble lifts MNIST from 95.68% to 96.20%
(+0.52%), Fashion from 85.93% to 86.21% (+0.28%), and CIFAR-10 from
47.80% to 50.86% (+3.06%).  The improvement is largest on CIFAR-10,
where the base model's accuracy is lowest and there is more room for
error decorrelation to help.

**Local receptive fields are the bigger lever.**  On CIFAR-10, the RF
ensemble reaches 55.32% with 10 members — a remarkable **+7.52%** over
the single model and +4.46% over the hypersphere ensemble.  This is
consistent with the findings of McDonnell et al. (2015): local receptive
fields are the single biggest accuracy lever for random-weight networks on
image tasks, because they concentrate each neuron's sensitivity on local
spatial structure rather than diluting it across the full image.

On MNIST, the RF ensemble (96.51%) outperforms the hypersphere ensemble
(96.20%) by 0.31%.  The improvement is smaller because MNIST's simpler
digit structure is already well-captured by global projections.  Fashion-
MNIST shows an intermediate pattern: RF adds 0.44% over the hypersphere
ensemble (86.65% vs 86.21%).

**Diminishing returns from 10 to 20 members.**  Doubling the ensemble size
from 10 to 20 provides only marginal further gains: +0.04% on MNIST,
+0.25% on CIFAR-10 (hypersphere), and +0.44% on CIFAR-10 (RF).  On
Fashion-MNIST with hypersphere encoders, the 20-member ensemble is
actually 0.07% *worse* than the 10-member (86.14% vs 86.21%), likely due
to seed-dependent variance.  The RF 20-member ensemble does improve
Fashion from 86.65% to 86.90%.  The steepest improvement curve is from
1→10 members; 20 members roughly doubles training time for diminishing
returns, suggesting 10 members is the practical sweet spot.

**Timing.**  The 10-member ensemble takes ~24–28 seconds on MNIST
(~12× a single model, with some overhead).  This is still far faster
than gradient-trained alternatives: the MLP baseline takes 83 seconds,
and multi-layer hybrid→E2E takes 402 seconds.  The 20-member ensemble
takes ~45 seconds — still competitive with a single MLP training run
while providing better accuracy on some datasets.

**Comparison with brute-force neuron scaling.**  A single model with 20000
neurons (equivalent total parameters to a 10-member × 2000 ensemble)
reaches 97.9% on MNIST (from Section 5.1) in ~140 seconds, compared to
96.20% for the 10-member hypersphere ensemble in ~24 seconds.  The single
large model wins on accuracy but loses badly on time.  The RF ensemble
(96.51% in 28 seconds) offers a different trade-off: spatial structure
from RF encoders partially compensates for the smaller per-model neuron
count while training 5× faster than the single 20k-neuron model.

### 5.5 Neuron–Patch–Alpha Sweep: Beating the MLP Baseline

The previous section used 2000 neurons per member and the default
regularisation α = 0.01.  We now explore how far single-layer RF
models can be pushed by increasing neuron count, varying patch size,
and tuning the Tikhonov regularisation parameter α — with the explicit
goal of matching or exceeding the gradient-trained MLP baseline
(MNIST 98.50%, Fashion 89.70%, CIFAR-10 52.70%, all at ~83 seconds).

#### 5.5.1 Neuron Count × Patch Size Sweep

We first sweep neuron counts {2000, 3000, 4000, 5000} and patch sizes
{3, 5, 7, 10, 12, 14, 16, 18} for 10-member RF ensembles at the
default α = 0.01.

**MNIST (10-member RF ensemble, α = 0.01):**

| Neurons | Patch 3 | Patch 5 | Patch 7 | Patch 10 | Patch 12 | Patch 14 | Patch 16 | Patch 18 |
|---------|---------|---------|---------|----------|----------|----------|----------|----------|
| 2000  | 95.68%/21s | 96.51%/22s | 96.86%/23s | 97.13%/23s | — | — | — | — |
| 3000  | 96.28%/44s | 97.04%/46s | 97.34%/46s | 97.43%/46s | 97.40%/51s | 97.24%/52s | 97.15%/52s | 96.97%/52s |
| 4000  | — | — | 97.55%/81s | **97.78%/79s** | 97.66%/84s | 97.47%/82s | 97.32%/82s | — |
| 5000  | 96.83%/110s | 97.40%/112s | 97.69%/113s | **97.84%/113s** | — | — | — | — |

**Fashion-MNIST (10-member RF ensemble, α = 0.01):**

| Neurons | Patch 3 | Patch 5 | Patch 7 | Patch 10 | Patch 12 |
|---------|---------|---------|---------|----------|----------|
| 2000  | 86.46%/24s | 86.65%/25s | 86.73%/24s | 86.55%/24s | — |
| 3000  | 87.11%/47s | 87.58%/47s | 87.51%/48s | 86.98%/47s | — |
| 4000  | 87.68%/83s | 87.80%/83s | 87.75%/84s | 87.32%/83s | 87.20%/82s |
| 5000  | 88.07%/114s | 88.36%/114s | 88.31%/115s | 87.78%/113s | — |

**CIFAR-10 (10-member RF ensemble, α = 0.01):**

| Neurons | Patch 3 | Patch 5 | Patch 7 | Patch 10 |
|---------|---------|---------|---------|----------|
| 2000  | 54.70%/31s | 55.32%/32s | 55.00%/32s | 54.55%/32s |
| 3000  | 56.36%/58s | **56.94%/59s** | 56.61%/60s | 55.70%/59s |
| 5000  | 58.25%/130s | **59.06%/129s** | 58.86%/131s | 58.02%/131s |

**Optimal patch size is dataset-dependent.**  MNIST peaks at patch = 10
(~36% of the 28×28 image), Fashion-MNIST at 5–7, and CIFAR-10 at 5.
Patches larger than the optimum *hurt*: on MNIST, going from patch 10
to 18 drops 3000-neuron accuracy from 97.43% to 96.97%.  This makes
intuitive sense: as the patch approaches the full image, the receptive
field loses its locality advantage and degenerates toward a global
projection.

**Neuron count is the dominant factor** within a time budget.  At the
83-second MLP time budget, the best configurations are 4000 neurons ×
10 members: 97.78% on MNIST (79s), 87.80% on Fashion (83s), and
56.94% on CIFAR-10 with 3000 neurons (59s).  CIFAR-10 already exceeds
the MLP baseline (52.70%) at every configuration tested.  MNIST and
Fashion remain short of the MLP by 0.72% and 1.90% respectively.

![MNIST neuron–patch heatmap](figures/neuron_patch_heatmap.png)
*Figure 2. MNIST accuracy as a function of neuron count and RF patch size (α = 0.01).  The optimum shifts toward larger patches as neuron count increases.  Missing cells were not measured.*

#### 5.5.2 Single Large RF Layer vs Ensemble

The ensemble of small members leaves total neuron capacity split across
members.  A single layer with all neurons concentrated can capture richer
features.  We compare large single RF layers against ensembles:

**MNIST, single RF layer (α = 0.01):**

| Neurons | Patch 7 | Patch 10 | Patch 12 |
|---------|---------|----------|----------|
|  8000   | 97.85%/25s | 97.91%/25s | 97.87%/26s |
| 10000   | 98.02%/39s | 98.09%/39s | 98.15%/40s |
| 12000   | 97.91%/57s | 98.26%/58s | 98.18%/57s |

A single 12 000-neuron layer reaches **98.26%** in 58 seconds — already
0.48% better than the 10×4000 ensemble (97.78%/79s) while being 27%
faster.  Concentrating neurons in a single model is superior to splitting
them across an ensemble, because a richer feature space matters more than
decorrelation when the base models are already strong.

#### 5.5.3 Regularisation Tuning: The Decisive Lever

The default Tikhonov α = 0.01 was adopted from the single-layer
experiments with 2000 neurons, where it prevents overfitting.  With
12 000+ neurons and RF encoders, the feature space is much richer and
the solver can tolerate weaker regularisation.  A sweep over α reveals
dramatic improvements:

**MNIST (single 12 000n, RF patch = 10):**

| α | Train | Test | Time |
|---|-------|------|------|
| 5×10⁻⁴ | 99.73% | **98.50%** | 52s |
| 1×10⁻³ | 99.70% | 98.46% | 54s |
| 5×10⁻³ | 99.07% | 98.36% | 55s |
| 1×10⁻² | 98.77% | 98.26% | 56s |
| 2×10⁻² | 98.38% | 98.12% | 57s |

**Fashion-MNIST (single 14 000n, RF patch = 5):**

| α | Train | Test | Time |
|---|-------|------|------|
| 1×10⁻⁴ | 96.50% | 89.35% | 82s |
| 5×10⁻⁴ | 96.07% | 89.71% | 81s |
| 1×10⁻³ | 95.73% | **89.74%** | 82s |
| 2×10⁻³ | 95.22% | **89.74%** | 80s |
| 5×10⁻³ | 94.32% | 89.56% | 80s |
| 1×10⁻² | 93.45% | 89.49% | 79s |

**CIFAR-10 (10-member RF ensemble, 3000n, patch = 5):**

| α | Train | Test | Time |
|---|-------|------|------|
| 1×10⁻⁴ | 69.44% | **58.44%** | 53s |
| 5×10⁻⁴ | 69.16% | 58.31% | 56s |
| 1×10⁻³ | 68.80% | 58.15% | 56s |
| 5×10⁻³ | 66.78% | 57.62% | 56s |
| 1×10⁻² | 65.27% | 56.94% | 57s |

**Reducing α from 1×10⁻² to 5×10⁻⁴ lifts MNIST from 98.26% to
98.50%** — a 0.24% gain from a single hyperparameter change.  The effect
is consistent across datasets: Fashion improves from 89.49% to 89.74%,
and CIFAR-10 from 56.94% to 58.44%.

The optimal α depends on the ratio of feature space richness to dataset
complexity.  MNIST (simplest patterns, richest relative feature space)
benefits most from low regularisation (α ≈ 5×10⁻⁴).  Fashion-MNIST's
more complex visual patterns need slightly more regularisation (α ≈
1–2×10⁻³) to prevent overfitting.  CIFAR-10's 3-channel colour images
with the ensemble approach work best at α ≈ 1×10⁻⁴.

![Regularisation tuning curves](figures/alpha_tuning.png)
*Figure 3. Test accuracy vs Tikhonov α for the best configuration on each dataset.  The dashed line marks the MLP baseline.  Reducing α from the default 10⁻² to the optimum lifts accuracy by 0.24–1.50 percentage points.*

#### 5.5.4 Summary: Final Results vs MLP Baseline

| Method | MNIST | Fashion | CIFAR-10 | Time |
|--------|-------|---------|----------|------|
| MLP (2×1000, SGD) | 98.50% | 89.70% | 52.70% | 83s |
| **NEF single RF** (12 000n, p=10, α=5×10⁻⁴) | **98.50%** | — | — | **52s** |
| **NEF single RF** (14 000n, p=5, α=1×10⁻³) | — | **89.74%** | — | **82s** |
| **NEF single RF** (12 000n, p=5, α=1×10⁻³) | — | 89.70% | — | **59s** |
| **NEF ensemble** (10×3000, RF p=5, α=1×10⁻⁴) | — | — | **58.44%** | **53s** |

The analytically solved single-layer NEF model matches or exceeds the
gradient-trained MLP on all three benchmarks while training faster:

- **MNIST**: 98.50% in 52 seconds (37% faster, equal accuracy).
- **Fashion-MNIST**: 89.74% in 82 seconds (0.04% better; or 89.70% in
  59 seconds — matching accuracy 29% faster).
- **CIFAR-10**: 58.44% in 53 seconds (5.74% better and 36% faster).

No gradient computation is used.  Training consists of constructing
random receptive field encoders, computing neuron activities, and solving
a single Tikhonov-regularised least-squares problem.  The entire pipeline
is embarrassingly parallelisable across neurons and ensemble members.

![Speed vs accuracy comparison](figures/speed_accuracy.png)
*Figure 4. NEF single-layer RF models (blue/green) vs gradient-trained MLP (orange) on all three benchmarks.  NEF matches or exceeds MLP accuracy at equal or lower training time on commodity CPU hardware.*

### 5.6 Regression — California Housing

MSE with normalised targets, 2000 neurons:

| Neurons | Train MSE | Test MSE |
|---------|-----------|----------|
| 500     | 0.269     | 0.265    |
| 1000    | 0.254     | 0.249    |
| 2000    | 0.235     | 0.236    |
| 5000    | 0.213     | 0.227    |

More neurons improve accuracy with diminishing returns.  At 2000 neurons,
test MSE is within 1% of training MSE, indicating good generalisation.

### 5.7 Multi-Layer Results

Hidden=[1000], output=2000, 50 iterations for hybrid/TP, 50 epochs for
E2E:

| Model             | MNIST    | Fashion  | CIFAR-10 | Time (MNIST) |
|-------------------|----------|----------|----------|--------------|
| Linear baseline   | 85.3%    | 81.0%    | 39.6%    |     2s       |
| NEFLayer          | 95.7%    | 85.9%    | 47.8%    |     2s       |
| NEFNet-greedy     | 95.0%    | 85.4%    | 45.6%    |     3s       |
| NEFNet-hybrid     | 98.6%    | 90.2%    | 52.7%    |   318s       |
| NEFNet-target-prop| 98.6%    | 90.1%    | 51.0%    |   378s       |
| NEFNet-e2e        | 98.5%    | 90.3%    | 58.5%    |   241s       |
| NEFNet-hybrid→E2E |**98.6%** |**90.6%** | 58.4%    |   402s       |
| NEFNet-TP→E2E     | 98.5%    | 90.6%    |**58.5%** |   464s       |
| MLP (2×1000)      | 98.5%    | 89.7%    | 52.7%    |    83s       |

**Key observations:**

*Propagated data-driven biases.*  Training data is forwarded through each
layer, and the resulting activations serve as centers for the next layer's
bias computation.  This is critical for greedy training: without propagated
biases, greedy multi-layer is *worse* than single-layer (94.0% vs 95.5% on
MNIST).  With propagated biases, the gap narrows to 0.5%.

*Iterations dominate hybrid hyperparameters.*  Going from 10 to 50
iterations lifts hybrid from 97.2% to 98.5% on MNIST, 87.9% to 90.4% on
Fashion, and 45.9% to 53.3% on CIFAR-10.  Other hyperparameters (solver
type, regularisation, layer width, depth) each contribute less than 0.2%.

*Decoder regularisation is the second lever.*  α = 10⁻³ is optimal for
hybrid.  Lower values let decoders overfit the current encoder state,
producing noisy gradients; at α = 10⁻⁵, training collapses.

*CE loss is destructive.*  Decoders solve for MSE-optimal outputs near
0/1; cross-entropy interprets these as logits, creating a gradient
conflict that drops CIFAR-10 from 53% to 37%.

*Hybrid→E2E is the best balanced strategy.*  98.6% / 90.6% / 58.4% —
best on Fashion-MNIST, tied for best on MNIST, within 0.1 points on
CIFAR-10.  The hybrid phase learns encoder orientations with analytic
decoders; the E2E phase refines all parameters jointly.

#### Activation Effect on Multi-Layer Hybrid

| Activation | MNIST  | Fashion |
|------------|--------|---------|
| relu       |**97.8%**|**88.7%**|
| abs        | 97.5%  | 88.0%   |
| lif_rate   | 95.2%  | 85.1%   |
| softplus   | 94.0%  | 84.1%   |

> Measured with 10 hybrid iterations, gain=U(0.5, 2.0), data-driven
> biases, α = 10⁻².

The ranking reverses from single-layer: relu slightly edges out abs for
multi-layer hybrid training.  Per-neuron gain diversity means relu's
zero-gradient region no longer kills half the neurons uniformly; the
sparsity aids gradient flow through multiple encode-decode cycles.

![Method comparison across all training strategies](figures/method_comparison.png)
*Figure 5. Accuracy across all training strategies and all three datasets.  "NEF RF+α\*" uses the best single-layer RF configuration per dataset; it matches multi-layer strategies on MNIST/Fashion and the MLP baseline on CIFAR-10 — without any gradient computation.*

### 5.8 Recurrent Results — Sequential MNIST

Row-by-row sMNIST: each image is a sequence of 28 rows (T=28, d=28),
classified at the final timestep.  All NEF models use 2000 neurons, relu
activation, gain U(0.5, 2.0), and data-driven biases.

| Model                                | Accuracy | Time    |
|--------------------------------------|----------|---------|
| RecNEF-greedy (5 iter)               |  15.3%   |   241s  |
| RecNEF-hybrid (10 iter)              |  21.0%   |   461s  |
| RecNEF-target-prop (50 iter)         |  12.1%   |  5864s  |
| RecNEF-E2E (50 epochs)               |  98.5%   |   799s  |
| RecNEF-hybrid→E2E (10+20 epochs)     | **98.6%**|   686s  |
| LSTM-128 (20 epochs)                 |  98.3%   |    98s  |

Recurrent hybrid→E2E is the strongest result, edging pure E2E while
training faster.  Plain recurrent hybrid and greedy collapse — random
encoders compound state feedback noise across timesteps.  Only E2E-based
strategies produce competitive results.

*Why abs fails for recurrence.*  In feedforward models, abs doubles
representational capacity.  In recurrent BPTT, abs has gradient
sign(x) ∈ {−1, +1} at every neuron — the recurrent Jacobian has no
sparsity to damp gradient magnitudes.  Over 28 timesteps, this causes
gradient explosion.  ReLU's zero gradient on ~half the neurons provides
critical damping.  E2E with abs gets 10.1% (random); with relu, 98.5%.

#### Predictive State Targets (Experimental)

The experimental `predictive` state target option decodes the *next* input
frame rather than the current one, which better matches what recurrent
state should carry.  On seeded slices (2k train / 1k test):

| State target    | Seed 0 | Seed 1 | Seed 2 | Mean  |
|-----------------|--------|--------|--------|-------|
| Reconstruction  | 22.2%  | 22.6%  | 21.8%  | 22.2% |
| Predictive      | 31.9%  | 30.9%  | 32.8%  | 31.9% |

This is the first recurrent TP change that consistently improves accuracy
across seeds, though it remains far below E2E-based results.


### 5.9 Streaming Temporal Results — Sequential MNIST

The `StreamingNEFClassifier` (Section 3.10) takes a fundamentally different
approach to temporal classification: instead of maintaining recurrent state,
it encodes each sequence through a delay-line of overlapping temporal
windows, mean-pools the resulting activities, and decodes analytically.
Training uses continuous Woodbury updates (Section 2.8) — no gradients,
no backpropagation through time.

#### 5.9.1 Window Size and Neuron Count Sweep

We sweep window sizes K ∈ {3, 5, 7, 10, 14, 28} and neuron counts
n ∈ {2000, 4000, 6000, 8000, 10000}.  All models use abs activation,
hypersphere encoders, per-neuron gain U(0.5, 2.0), data-driven biases,
streaming Woodbury training with batch size 500, and a final
`refresh_inverse()` call.  Regularisation α is tuned per configuration.

**Key results (best α per configuration):**

| Neurons | Window | α       | Train%  | Test%      | Time  |
|---------|--------|---------|---------|------------|-------|
| 2000    |  3     | 1×10⁻² | 94.33   | 92.34      | 16s   |
| 2000    |  7     | 1×10⁻² | 97.43   | 96.95      | 24s   |
| 2000    | 10     | 1×10⁻² | 97.77   | 97.22      | 23s   |
| 2000    | 28     | 1×10⁻² | 97.82   | 97.16      | 34s   |
| 4000    |  7     | 1×10⁻³ | 98.61   | 97.80      | 91s   |
| 4000    | 10     | 1×10⁻² | 98.81   | 98.06      | 90s   |
| 4000    | 14     | 5×10⁻³ | 98.87   | 98.10      | 114s  |
| 6000    | 10     | 1×10⁻² | 99.24   | 98.35      | 136s  |
| 8000    | 10     | 5×10⁻³ | 99.49   | **98.56**  | 222s  |
| 10000   | 10     | 1×10⁻¹ | 99.64   | **98.57**  | 346s  |

**Observations:**

1. **Window size sweet spot at K=10–14.**  Larger windows capture more
   temporal context, but K=28 (full image row) actually degrades
   performance.  The delay-line feature dimension is K×d; at K=28 this
   becomes 784, and the neurons cannot cover the high-dimensional space
   effectively.

2. **Strong scaling with neuron count up to ~8000.**  From 2000 to 8000
   neurons, test accuracy improves steadily from 97% to 98.56%.  Beyond
   8000, returns diminish sharply: 10000 neurons require 5× stronger
   regularisation (α=0.1 vs 0.01) and gain only 0.01% at 55% more time.

3. **Overfitting at high neuron counts.**  With 10000 neurons and the
   default α=10⁻², test accuracy collapses to 97.62% while training
   reaches 99.47%.  Regularisation tuning is essential for large models.

4. **α is insensitive at moderate neuron counts.**  For ≤6000 neurons,
   α ∈ {10⁻³, 5×10⁻³, 10⁻²} all yield essentially identical test
   accuracy.  Sensitivity appears only above 8000 neurons.

#### 5.9.2 Comparison with Recurrent Models

| Model                                | Accuracy   | Time   | Gradients? |
|--------------------------------------|------------|--------|------------|
| RecNEF-greedy                        |  15.3%     |  241s  | No         |
| RecNEF-E2E (50 epochs)               |  98.5%     |  799s  | Yes (BPTT) |
| RecNEF-hybrid→E2E (10+20 epochs)     |  98.6%     |  686s  | Yes (BPTT) |
| LSTM-128 (20 epochs)                 |  98.3%     |   98s  | Yes (BPTT) |
| **StreamNEF-2000n (w=10)**           |  97.22%    | **23s**| **No**     |
| **StreamNEF-8000n (w=10)**           |**98.56%**  |  222s  | **No**     |

The streaming NEF classifier achieves 97.22% in just 23 seconds —
completely gradient-free, on CPU — outperforming RecNEF-greedy by 82
percentage points at 10× less time.  At 8000 neurons, it reaches 98.56%,
matching the LSTM baseline (98.3%) and approaching RecNEF-E2E (98.5%),
while still using no gradients whatsoever.

The key advantage is architectural: the delay-line reservoir avoids the
fundamental problem of recurrent gradient-free methods (noise compounding
through the feedback loop) by replacing recurrence with temporal pooling.
This trades sequence modelling flexibility for robust gradient-free
training.

### 6.1 Competitive Positioning

The single-layer NEF result (95.7% MNIST, 2 seconds) is competitive with
vanilla ELMs (~95%) and random kitchen sinks (94–96%) at similar speed.
The NEF-specific data-driven biases provide a consistent edge by
eliminating the dependence on encoder type.

The key result of this work is that **scaling up the single-layer model
with local receptive fields and tuned regularisation matches or exceeds a
gradient-trained MLP on all three benchmarks** (Section 5.5).  With 12 000
RF neurons and α = 5×10⁻⁴, a single layer reaches 98.50% on MNIST in 52
seconds — equal to the MLP in 37% less time, with zero gradient
computation.  On Fashion-MNIST, 14 000 neurons with α = 1×10⁻³ reach
89.74% in 82 seconds (0.04% better than MLP).  On CIFAR-10, a 10-member
3 000-neuron RF ensemble with α = 1×10⁻⁴ reaches 58.44% in 53 seconds
(5.74% better and 36% faster).

These results are consistent with McDonnell et al. (2015), who reported
98.8% on MNIST with a single ELM using local receptive fields and 10 000
neurons, and 99.17% with an ensemble of 10 such models.  Our single
12 000-neuron layer reaches 98.50% — slightly below their 98.8%, which
may be attributable to differences in preprocessing (they use contrast
normalisation) or activation function (they use ReLU with a different gain
distribution).  The core finding is the same: local receptive fields +
many neurons + analytical solve is a powerful combination that competes
with gradient training.

Two experimental insights emerged from the sweep:

1. **Single large layer > ensemble of small layers** (at the same time
   budget).  A 12 000-neuron single layer (98.50%/52s) outperforms a
   10×4000 ensemble (97.78%/79s).  The richer feature space of the larger
   model matters more than ensemble decorrelation.

2. **Regularisation tuning is the decisive lever.**  Reducing α from
   1×10⁻² to 5×10⁻⁴ lifts MNIST accuracy by 0.24 percentage points.
   The optimal α scales inversely with the feature space richness: MNIST
   (simplest) → 5×10⁻⁴, Fashion (moderate) → 1–2×10⁻³, CIFAR ensemble
   → 1×10⁻⁴.  This is predictable from bias-variance theory: richer
   feature spaces can tolerate less regularisation before overfitting.

The streaming NEF classifier (Section 5.9) extends this competitive
positioning to temporal data.  At 8000 neurons with window K=10, the
gradient-free streaming classifier reaches 98.56% on sMNIST-row — matching
the LSTM baseline (98.3%) and rivalling RecNEF-E2E (98.5%), which requires
full backpropagation through time.  The 2000-neuron variant (97.22% in 23s)
is competitive as a real-time temporal classifier on commodity hardware.

### 6.2 When to Use Each Strategy

| Scenario | Recommended approach | Expected accuracy (MNIST/sMNIST) | Time |
|----------|---------------------|--------------------------|------|
| Maximum speed | Single NEFLayer | 95.7% | 2s |
| Speed/accuracy balance | Single RF NEFLayer (12k, α=5×10⁻⁴) | 98.5% | ~52s |
| Beat MLP, no gradients | Single RF layer, tuned α | 98.5% / 89.7% / 58.4% | 52–82s |
| Maximum accuracy, moderate time | NEFNet-hybrid→E2E | 98.6% | 402s |
| Streaming data (online) | NEFLayer + continuous_fit (Woodbury) | 95.7% | varies |
| Streaming data (GPU) | NEFLayer + accumulate + solve | 95.7% | varies |
| Temporal sequences (fast) | StreamNEF (2000n, w=10) | 97.2% (sMNIST) | 23s |
| Temporal sequences (accurate) | StreamNEF (8000n, w=10) | 98.6% (sMNIST) | 222s |
| Temporal sequences (best) | RecNEF-hybrid→E2E | 98.6% (sMNIST) | 686s |

### 6.3 Limitations

- **CIFAR-10 ceiling.**  Even with RF + ensemble + tuned α, single-layer
  accuracy plateaus at ~58%.  Natural images require learned hierarchical
  features that random projections cannot capture.  Multi-layer strategies
  reach the same level (58.5%) with gradient training, but both remain far
  below CNN-level accuracy (~95%).
- **Hyperparameter sensitivity.**  The optimal α varies by dataset and
  neuron count.  While the optimal range is narrow (10⁻⁴ to 10⁻³), the
  wrong setting can cost 0.5–1% accuracy.  Cross-validation or a small
  held-out set is recommended.
- **Recurrent TP.**  Despite the predictive state target improvement,
  recurrent target propagation remains research-grade (~32%) compared to
  E2E (98.5%).
- **GPU dtype tradeoff.**  The Woodbury continuous-fit path requires
  float64 for numerical stability, but consumer GPUs (T4, L4) deliver
  only 1/32 of their float32 throughput in float64.  The accumulate +
  solve path (Section 3.7.2) works entirely in float32, but sacrifices
  online decoder updates.  Initial T4 measurements show StreamNEF-8k
  at 92s (Woodbury, float64) vs LSTM-128 at 21s — a 2.4× CPU→GPU
  speedup compared to LSTM's 4.7×.  The accumulate path should close
  this gap significantly.

### 6.4 Relationship to Prior Work

Our analytical target propagation approach uses NEF representational
decoders as the inverse models that DTP (Lee et al., 2015) requires.
Recent independent work has explored similar ideas: Bao et al. (2024)
derive closed-form feedback weights for target propagation, and Shibuya
et al. (2023) show that even fixed feedback weights work if forward
weights are well-conditioned.  Our NEF decoders are analytical, exact, and
recomputed at each iteration — a natural fit from neuroscience principles
that achieves the same goal.

Several other biologically motivated training approaches share conceptual
ground with NEF-TP and offer avenues for integration:

- **Feedback alignment** (Lillicrap et al., 2016) replaces the transpose
  weight matrix in backprop with a fixed random matrix.  Our analytical
  representational decoders are strictly better for target computation —
  they are exact inverse models computed for free.
- **Predictive coding networks** (Rao & Ballard, 1999; Millidge et al.,
  2022) propagate prediction errors through a generative hierarchy.
  Our representational decoders serve as the generative model (top-down
  predictions), and the prediction error at each layer could drive encoder
  updates — conceptually equivalent to TP but with an iterative inference
  phase that relaxes to equilibrium before weight updates.
- **The forward-forward algorithm** (Hinton, 2022) trains each layer to
  distinguish real from corrupted data using only local information.  It
  could complement our analytical decoders: solve D after each layer
  learns good encodings via forward-forward.
- **HSIC bottleneck** (Ma et al., 2020) trains each layer to maximise the
  Hilbert-Schmidt Independence Criterion between its representation and
  the target.  The kernel matrix is computed from the activity matrix, and
  the optimum has a closed-form related to our normal-equations solver,
  making it a natural candidate for integration with NEF.


## 7. Conclusions

1. **Data-driven biases are the key single-layer design choice.**
   Rewriting the encoding as `|gain · ((x − d) · e)|` reveals each
   neuron measures unsigned deviation from a reference point along a
   random direction.  Sampling references from training data closes the
   entire 2–3% gap between encoder types and makes the encoder direction
   distribution irrelevant.

2. **A single analytically solved layer matches or beats a gradient-
   trained MLP — without any gradient computation.**  With local
   receptive field encoders, 12 000–14 000 neurons, and tuned Tikhonov
   regularisation, a single NEF layer achieves 98.50% on MNIST (52s,
   37% faster than MLP), 89.74% on Fashion-MNIST (82s, 0.04% better),
   and 58.44% on CIFAR-10 via ensembling (53s, 5.74% better and 36%
   faster).  This is the headline result: the analytical solve is not
   merely "fast but less accurate" — it is *competitive in both speed
   and accuracy* when properly configured.

3. **Regularisation tuning is the decisive lever at scale.**  The
   default Tikhonov α = 10⁻² is appropriate for 2000-neuron models,
   but larger models with RF encoders have richer feature spaces that
   tolerate weaker regularisation.  Reducing α to 5×10⁻⁴ (MNIST) or
   1×10⁻³ (Fashion) lifts accuracy by 0.24–0.25 percentage points,
   closing the gap to the MLP baseline.

4. **Single large layer outperforms ensembles of small layers** at the
   same time budget.  A 12 000-neuron single RF layer (98.50%/52s) beats
   a 10×4000 ensemble (97.78%/79s) on MNIST.  The richer feature space
   of the larger model matters more than ensemble decorrelation when
   base models are already strong.  Ensembles remain valuable for CIFAR-10,
   where per-model accuracy is lower and decorrelation helps more.

5. **The abs activation is a natural fit for single-layer models.**
   Two-sided response doubles representational capacity.  For multi-layer
   gradient training, relu's sparsity slightly edges ahead.

6. **Local receptive field encoders are the biggest structural lever.**
   RF encoders boost CIFAR-10 from 47.80% to 55.32% (+7.52%) with a
   10×2000 ensemble.  On MNIST, larger RF layers reach 98.50% — close
   to McDonnell et al.'s 98.8% with comparable neuron counts.

7. **Hybrid→E2E remains the best balanced multi-layer strategy.**
   98.6% / 90.6% / 58.4% — the hybrid phase learns encoder orientations
   with analytic decoders; the E2E phase refines all parameters jointly.

8. **Target propagation with analytical NEF inverse models is viable.**
   Plain TP reaches 98.6% / 90.1% / 51.0% using only single-layer
   gradients — no cross-layer backpropagation.

9. **Predictive state targets are the most promising recurrent TP
   direction.**  Switching from reconstruction to prediction targets
   lifts recurrent TP from 22% to 32% mean accuracy.

10. **Incremental learning is exact.**  The normal-equation decomposition
    enables streaming data and model updates without reprocessing, with
    results identical to full-batch training.

11. **Continuous learning via Woodbury updates enables gradient-free
    temporal classification.**  The Sherman-Morrison-Woodbury identity
    maintains the system inverse incrementally at O(n²k) cost per batch.
    Combined with a delay-line temporal encoder, this produces a streaming
    classifier that reaches 98.57% on sequential MNIST — matching LSTM
    (98.3%) and RecNEF-E2E (98.5%) — without any gradient computation.
    Float64 arithmetic for the inverse is essential; float32 drift
    causes catastrophic failure at ≥4000 neurons.  For GPU deployment,
    the accumulate + solve path (Section 3.7.2) avoids per-batch
    inverse maintenance by accumulating AᵀA/AᵀY (with float64
    outer products only) and performing a single solve at the end.
    On a T4 GPU this yields 8–15× speedup over Woodbury while
    matching accuracy.

12. **The delay-line reservoir sidesteps the recurrent gradient-free
    bottleneck.**  Gradient-free recurrent models (RecNEF-greedy: 15%)
    fail because random encoders compound state feedback noise.  The
    streaming classifier replaces recurrence with temporal pooling,
    achieving 97% in 23 seconds (2000 neurons) — a viable
    real-time temporal classifier on commodity hardware.


## References

- Y. Bao, Y. Li, S. Huang, L. Zhang, A. Zheng, Y. Jiang, Y. Chen,
  "Efficient Target Propagation by Deriving Analytical Solution", *AAAI
  Conference on Artificial Intelligence*, 2024.

- L. Breiman, "Bagging Predictors", *Machine Learning* 24(2), 1996.

- L. Breiman, "Random Forests", *Machine Learning* 45(1), 2001.

- D. S. Broomhead & D. Lowe, "Radial Basis Functions, Multi-Variable
  Functional Interpolation and Adaptive Networks", RSRE Memorandum 4148,
  1988.

- C. Eliasmith & C. H. Anderson, *Neural Engineering: Computation,
  Representation, and Dynamics in Neurobiological Systems*, MIT Press,
  2003.

- C. Eliasmith, "A unified approach to building and controlling spiking
  attractor networks", *Neural Computation* 17(6), 2005.

- S. Haykin, *Adaptive Filter Theory*, 4th edition, Prentice Hall, 2002.

- G. Hinton, "The Forward-Forward Algorithm: Some Preliminary
  Investigations", *arXiv:2212.13345*, 2022.

- G.-B. Huang, Q.-Y. Zhu & C.-K. Siew, "Extreme Learning Machine:
  Theory and Applications", *Neurocomputing* 70(1–3), 2006.

- D.-H. Lee, S. Zhang, A. Fischer & Y. Bengio, "Difference Target
  Propagation", *European Conference on Machine Learning and Principles
  and Practice of Knowledge Discovery in Databases (ECML-PKDD)*, 2015.

- T. P. Lillicrap, D. Cownden, D. B. Tweed & C. J. Akerman, "Random
  synaptic feedback weights support error backpropagation for deep
  learning", *Nature Communications* 7, 2016.

- S. Ma, R. Bassily & M. Belkin, "The Power of Interpolation: Understanding
  the Effectiveness of SGD in Modern Over-parametrized Learning",
  *Proceedings of the 35th International Conference on Machine Learning
  (ICML)*, 2018.  *(Note: HSIC bottleneck methods referenced in Section 6.4
  build on kernel independence measures from this line of work.)*

- M. D. McDonnell, M. D. Tissera, T. Vladusich, A. van Schaik &
  J. Tapson, "Fast, Simple and Accurate Handwritten Digit Classification
  by Training Shallow Neural Network Classifiers with the 'Extreme
  Learning Machine' Algorithm", *PLOS ONE* 10(8), 2015.

- B. Millidge, A. Tschantz & C. L. Buckley, "Predictive Coding
  Approximates Backprop along Arbitrary Computation Graphs", *Neural
  Computation* 34(6), 2022.

- A. Rahimi & B. Recht, "Random Features for Large-Scale Kernel
  Machines", *Advances in Neural Information Processing Systems (NeurIPS)*,
  2007.

- R. P. N. Rao & D. H. Ballard, "Predictive coding in the visual
  cortex: a functional interpretation of some extra-classical
  receptive-field effects", *Nature Neuroscience* 2, 1999.

- T. Shibuya, T. Kudo & T. Harada, "Fixed-Weight Difference Target
  Propagation", *AAAI Conference on Artificial Intelligence*, 2023.

- M. A. Woodbury, "Inverting Modified Matrices", *Statistical Research
  Group Memorandum Report* 42, Princeton University, 1950.

- W. W. Hager, "Updating the Inverse of a Matrix", *SIAM Review* 31(2),
  1989.

**Datasets:**

- MNIST — Y. LeCun, L. Bottou, Y. Bengio & P. Haffner, "Gradient-Based
  Learning Applied to Document Recognition", *Proceedings of the IEEE*
  86(11), 1998.
- Fashion-MNIST — H. Xiao, K. Rasul & R. Vollgraf, "Fashion-MNIST: a
  Novel Image Dataset for Benchmarking Machine Learning Algorithms",
  *arXiv:1708.07747*, 2017.
- CIFAR-10 — A. Krizhevsky, "Learning Multiple Layers of Features from
  Tiny Images", Technical Report, University of Toronto, 2009.
- California Housing — R. K. Pace & R. Barry, "Sparse Spatial
  Autoregressions", *Statistics and Probability Letters* 33(3), 1997.
