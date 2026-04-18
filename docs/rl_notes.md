# NEF-FQI: Reinforcement Learning with Analytical Solvers

Research notes for NEF-based reinforcement learning experiments.

Hardware: CPU = AMD Ryzen 5 PRO 5650U (6C/12T, 30 GB RAM), no GPU.


## 1. Concept

Replace DQN's gradient-based Q-network updates with NEF's analytical
least-squares solve.  Both approaches approximate Q(s, a); the difference
is *how* the approximation weights are updated:

| Aspect            | DQN                        | NEF-FQI                       |
|-------------------|----------------------------|-------------------------------|
| Features          | Learned (hidden layers)    | Fixed random projections      |
| Weight updates    | SGD on mini-batches        | Analytical solve on buffer    |
| Target network    | Needed (stability)         | Not needed (MC mode)          |
| Replay buffer     | Required                   | Required                      |
| Learning rate     | Sensitive hyperparameter   | None                          |
| Solve frequency   | Every step                 | Every N episodes              |

The connection to classical RL is direct: NEF-FQI with TD targets is
equivalent to LSTD (Least-Squares Temporal Difference) with random
features.  With MC targets, it becomes batch Monte Carlo policy
evaluation with random features and ε-greedy improvement.


## 2. Architecture

### NEFFeatures (fixed encoder)

Encodes state s ∈ ℝ^d into φ(s) ∈ ℝ^n:

    φ(s) = |gain * (normalize(s) @ E^T) + bias|

- **Encoders E**: unit vectors sampled uniformly on the hypersphere (fixed)
- **Gain**: per-neuron, sampled from U(0.5, 2.0)
- **Bias**: random, sampled from U(-1, 1)
- **Activation**: abs (absolute value)
- **Normalization**: online running mean/std of observed states

### NEFFQIAgent

- Replay buffer (deque, 50000 capacity)
- Per-action weight vectors W_a ∈ ℝ^n (one per action)
- Solves every `solve_every` episodes
- ε-greedy exploration with exponential decay

### Q-value computation

    Q(s, a) = φ(s) @ W_a

Action selection: argmax_a Q(s, a) with ε-greedy.

### Decoder solve (per action)

For each action a, collect all (s_i, target_i) where action_i = a:

    Φ_a = [φ(s_i)]       (n_a × n_neurons)
    y_a = [target_i]      (n_a × 1)
    W_a = (Φ_a^T Φ_a + α I)^{-1} Φ_a^T y_a

Gram matrix computed in float64 for numerical stability.
Regularization: α = reg * trace(Φ^T Φ) / n_neurons, clamped to min α.


## 3. Target Modes

### Monte Carlo (MC) — default

Stores complete episode transitions in a buffer.  At episode end, computes
discounted returns G_t = Σ_{k=0}^{T-t-1} γ^k r_{t+k} and inserts
(s_t, a_t, G_t) into the replay buffer.

**Advantages:** No bootstrapping, no target network, no oscillation.
Direct regression on actual outcomes.

**Disadvantages:** High variance.  Fails on sparse-reward tasks where
most episodes produce near-identical returns (no signal to differentiate
states).  Requires episodic tasks.

### Temporal Difference (TD)

Standard bootstrapped targets: y = r + γ max_a Q_target(s', a).
Uses a target weight matrix W_target updated via Polyak averaging
(τ = 0.3 soft blend after each solve).

**Advantages:** Lower variance, works on continuing tasks.

**Disadvantages:** Bootstrapping instability.  The analytical solve
replaces W entirely each time (globally optimal for current targets), but
targets shift between solves → oscillation.  Five iterations tested on
CartPole; none fully stable.

### The fundamental insight

**Bootstrapping is the root cause of instability with analytical
solvers.**  SGD makes small updates that smoothly track shifting targets.
The analytical solve replaces weights wholesale — if targets shift even
slightly between solves, the new solution may be far from the old one,
causing policy oscillation.

MC eliminates this entirely: targets are fixed, ground-truth episode
returns.  The analytical solve converges monotonically as the buffer
grows.  No deadly triad, no target network, no Polyak averaging.


## 4. Results

### 4.1 CartPole-v1

**NEF-FQI solves CartPole; DQN does not.**

| Method      | Neurons | Episodes | Best eval | Final eval | Time    |
|-------------|---------|----------|-----------|------------|---------|
| NEF-FQI MC  | 2000    | 500      | **500.0** | 420.5      | 220.4s  |
| DQN 128×2   | —       | 500      | 332.4     | 122.5      | 50.0s   |

NEF-FQI reaches the maximum score of 500 (environment ceiling) from
episode 350 onward.  DQN peaks at 332 but never stabilizes.

**TD mode iteration history** (all CartPole, 2000-4000 neurons):

| Version       | Peak  | Final | Key change                    |
|---------------|-------|-------|-------------------------------|
| v1 (baseline) | 292.4 | 140.1 | No target network             |
| v2 (conserv.) | 118.8 | ~105  | tau=0.8 Polyak on W directly  |
| v3 (FQI iter) | 434.4 | ~111  | fqi_iters=3, solve_batch=20k  |
| v4 (hard tgt) | 125.5 | ~125  | W_target hard update / 20 eps |
| v4b (Polyak)  | 228.3 | 142.2 | tau=0.3 Polyak target         |
| v5 (low γ)    | 194.2 | ~132  | γ=0.95, 4000 neurons          |
| **MC mode**   |**500.0**|**486.7**| **No bootstrapping**       |

The contrast is stark: every TD variant peaks and collapses.  MC mode
converges and stays converged.

### 4.2 Acrobot-v1

**DQN wins; MC fails on sparse reward.**

| Method      | Neurons | Episodes | Best eval  | Final eval | Time   |
|-------------|---------|----------|------------|------------|--------|
| NEF-FQI MC  | 4000    | 500      | -410.8     | -500.0     | —      |
| DQN 128×2   | —       | 500      | **-141.7** | -160.4     | —      |

Acrobot gives -1 per step with a 500-step timeout.  Most MC episodes
produce returns ≈ -100, providing no signal to differentiate good from
bad states.  This is a fundamental MC limitation on sparse-reward tasks,
not an NEF-specific failure.

### 4.3 LunarLander-v3

**DQN wins clearly; NEF-FQI learns slowly but does not solve.**

| Method      | Neurons | Episodes | Best eval  | Final eval | Time     |
|-------------|---------|----------|------------|------------|----------|
| NEF-FQI MC  | 8000    | 1000     | 40.8       | 14.1       | 3590.3s  |
| DQN 256×2   | —       | 1000     | **214.0**  | 208.8      | 520.7s   |

LunarLander has shaped rewards (landing bonus, crash penalty, fuel cost)
so MC should work.  NEF-FQI does learn (from -181 to +41) but is very
noisy and 7× slower.  DQN solves it (≥200) by episode 800.

The 8D state space may be too high for 8000 random features to cover
adequately.  Also, MC returns have high variance in LunarLander due to
the wide reward range (crash ≈ -100, successful landing ≈ +200).

### 4.4 MountainCar-v0

Not tested.  Expected to fail for the same reason as Acrobot: sparse
reward (-1 per step, 200-step timeout).


## 5. Summary Table

| Environment    | NEF-FQI (MC) | DQN     | Winner  | Why                           |
|----------------|--------------|---------|---------|-------------------------------|
| CartPole-v1    | **500.0**    | 332.4   | NEF-FQI | Dense reward, low-dim state   |
| Acrobot-v1     | -410.8       |**-141.7**| DQN    | Sparse reward → MC fails      |
| LunarLander-v3 | 40.8         |**214.0**| DQN     | High-dim, high-variance MC    |


## 6. Analysis

### Where NEF-FQI excels

1. **Dense, low-dimensional reward environments.**  CartPole is the
   ideal case: 4D state, +1 per step, clear signal in MC returns.
2. **No hyperparameter tuning for the solve.**  No learning rate, no
   target network update frequency, no gradient clipping.
3. **Stability once converged.**  Once the buffer contains enough good
   episodes, the analytical solve produces a stable policy.  No
   catastrophic forgetting of good strategies.

### Where NEF-FQI fails

1. **Sparse reward.**  MC requires episodes that vary in return to learn
   which states are better.  If all episodes look the same (timeouts),
   there is no signal.
2. **High-dimensional state spaces.**  Random features in 8D (LunarLander)
   may need far more neurons than in 4D (CartPole).  The curse of
   dimensionality applies to random projections.
3. **Speed.**  The per-action analytical solve on the full buffer is
   O(n² · buffer_size) per solve.  DQN's SGD mini-batch updates are much
   cheaper per step.
4. **MC variance.**  Even with shaped rewards, MC returns can have very
   high variance (e.g., LunarLander's -100 to +200 range), making the
   regression target noisy.

### The bootstrapping dilemma

This is the core theoretical tension:

- **MC targets** are unbiased but high-variance and require dense rewards.
- **TD targets** are biased but lower-variance, and work with sparse
  rewards — but the analytical solve's wholesale weight replacement
  amplifies bootstrapping instability.
- **SGD** (DQN) handles this by making small incremental updates that
  smooth over target shifts.  The analytical solve does not have this
  "damping" effect.

Possible solutions not yet explored:
- **TD(λ):** Blend MC and TD targets via eligibility traces
- **Reward shaping:** Transform sparse rewards to give MC more signal
- **Feature adaptation:** Learn encoders (violates the "fixed encoder"
  principle but may be necessary for higher dimensions)
- **Ensemble averaging:** Multiple random projections, average Q-values
  to reduce variance


## 7. Relationship to Existing Work

NEF-FQI with MC targets is essentially **batch Monte Carlo control with
random features**.  This is a well-studied approach in the RL literature:

- **LSPI** (Lagoudakis & Parr, 2003): Least-squares policy iteration
  with fixed features.  NEF-FQI is LSPI with random features and MC
  evaluation.
- **Random Fourier Features for RL** (Rahimi & Recht, 2007 + followers):
  Random features for value function approximation.  Our approach uses
  the abs-valued NEF encoding rather than RFF.
- **Fitted Q-Iteration** (Ernst et al., 2005): Batch Q-learning with
  function approximation.  NEF-FQI is FQI with random NEF features.

The novelty is modest: the specific combination of NEF encoding (abs
activation, hypersphere encoders, per-neuron gain) with MC returns and
batch analytical solve.  The CartPole result is clean but not groundbreaking.

The more interesting angle for a paper would be **continual RL**: the
normal-equation accumulation (AᵀA, AᵀY) naturally extends across task
boundaries without forgetting.  This connects directly to the CL story.


## 8. Open Questions and Next Steps

### 8.1 Online Woodbury updates (Step 2)

Replace the full-buffer solve with episode-by-episode Woodbury updates.
Each episode adds rank-k updates to (AᵀA)⁻¹.  This eliminates the
replay buffer entirely — one pass, no storage.

Question: does Woodbury's numerical drift matter in RL where targets are
inherently noisy?  The CL experiments showed Woodbury ≈ accumulate when
α is well-tuned.

### 8.2 Ensemble exploration (Step 3)

Train N NEFFeature encoders with different random seeds.  Each produces
a different Q-estimate.  Explore where they disagree (high variance =
high uncertainty).  This is Thompson sampling with random-feature
posterior approximation.

### 8.3 Continual RL (Step 4)

Sequential tasks with changing dynamics or rewards.  The AᵀA/AᵀY
accumulation should retain knowledge across tasks.  This is the CL+RL
synthesis — the paper's potential climax.

### 8.4 TD(λ) targets

Blend MC and TD to get the best of both.  λ=1 is pure MC, λ=0 is pure
TD.  Intermediate values may avoid both MC's high variance and TD's
bootstrapping instability.

### 8.5 Scaling neurons with state dimension

The results suggest a scaling law: neurons needed grows rapidly with
state dimension.  CartPole (4D, 2000 neurons) works; LunarLander
(8D, 8000 neurons) doesn't.  Need systematic scaling experiments.
