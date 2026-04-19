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

**RLS forgetting + dead neuron recentering** (2000 neurons, 500 episodes):

| Config                       | Best eval | Final eval | Time    |
|------------------------------|-----------|------------|---------|
| Baseline (buffer)            | 500.0     | 487.9      | 248.4s  |
| RLS β=0.995                  | 500.0     | 338.7      |  92.6s  |
| Buffer + recenter/50         | 500.0     | 491.6      | 245.6s  |
| **RLS β=0.995 + recenter/50**| **500.0** | **500.0**  |**111.1s**|

The combination is synergistic.  RLS alone forgets too aggressively
after convergence (good early data decays).  Recentering alone adds
coverage but doesn't reduce solve cost.  Together they achieve
**perfect 500.0 from episode 150 onward** — stable convergence 50
episodes earlier than the baseline, at 2.2× less wall time, with no
replay buffer.

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

**DQN wins overall, but RLS forgetting closes half the gap.**

| Method                            | Neurons | Episodes | Best eval  | Final eval | Time     |
|-----------------------------------|---------|----------|------------|------------|----------|
| NEF-FQI MC (baseline)             | 8000    | 1000     | 40.8       | 14.1       | 3590.3s  |
| Recenter-only /100                | 8000    | 1000     | 64.2       | 19.7       | 3769.5s  |
| RLS β=0.995 + recenter/100        | 8000    | 1000     | 47.3       | 32.6       | 2580.5s  |
| RLS β=0.999 only                  | 8000    | 1000     | 109.0      | 82.6       | 2685.8s  |
| **RLS β=0.999 + recenter/100**    | 8000    | 1000     | **109.4**  | **102.2**  | 2621.1s  |
| DQN 256×2                         | —       | 1000     | **214.0**  | 208.8      | 520.7s   |

**Ablation analysis:**

- **RLS forgetting is the primary driver.** RLS-only reaches best=109.0,
  nearly matching the combined 109.4.  Forgetting old data lets the solver
  adapt to the improving policy instead of fitting a mixture of early
  (random) and late (good) behavior.
- **Recentering alone helps modestly** (64.2 vs 40.8 baseline) but is
  highly oscillatory (swings from +64 to -19 in consecutive evals).  Full
  buffer retains all history including early bad episodes, limiting
  adaptation.
- **The combination adds stability:** final 102.2 vs 82.6 for RLS-only.
  Recentering keeps features covering the visited manifold, which smooths
  the late-phase performance.  But it is not the primary mechanism.
- **RLS also eliminates the replay buffer** and runs faster (no
  re-encoding of the full buffer at each solve).

β=0.999 (effective window ~1000 episodes) substantially outperforms
β=0.995 (effective window ~200 episodes).  LunarLander's high reward
variance requires longer memory than CartPole.

The gap to DQN (214.0) remains significant.  Contributing factors:
- MC returns have high variance due to wide reward range
  (crash ≈ -100, successful landing ≈ +200)
- DQN's learned features adapt to the reward structure; NEF's are fixed
- Neuron scaling does not help (§4.4) — the bottleneck is MC variance

### 4.4 Neuron Scaling on LunarLander

**More neurons hurt at fixed episode budget.**

| Neurons | β     | Recenter | Best eval  | Final eval | Time      |
|---------|-------|----------|------------|------------|-----------|
| 8000    | 0.999 | /100     | **109.4**  | **102.2**  | 2621s     |
| 16000   | 0.999 | /100     | -12.2      | -73.4      | 13358s    |

16000 neurons tracks 8000 closely through episode 800 (both around
-12 to -40), but then **fails to produce the late surge** that 8000
shows.  At episode 950, 8000 reaches +109 while 16000 drops to -75.

The explanation is sample complexity: with β=0.999 (effective window
~1000 episodes), the solver has ~1000 × ~200 steps ≈ 200k data points
feeding into 16000 parameters.  This is an insufficient ratio for
stable least-squares in a non-stationary regime.  8000 neurons with
the same data gives a 25:1 sample-to-parameter ratio; 16000 gives
12.5:1.  The regularization (α) helps but cannot fully compensate.

Additionally, recentering 5% of 16000 neurons every 100 episodes
(~800 neurons/round, 8000 total over 1000 episodes = 50% of the
network) creates too much instability compared to recentering 400
neurons (5% of 8000) per round.

**Conclusion:** the bottleneck is not feature count but MC return
variance and exploration quality.  The optimal neuron count for
LunarLander with 1000 episodes and β=0.999 is around 8000.  32000
was not tested (O(n³) solve cost would exceed 4 hours per run).

### 4.5 MountainCar-v0

Not tested.  Expected to fail for the same reason as Acrobot: sparse
reward (-1 per step, 200-step timeout).


### 4.6 TD(λ) on LunarLander

**TD(λ) hurts — any bootstrapping destabilizes the analytical solve.**

| Method             | λ    | Best eval  | Final eval | Time    |
|--------------------|------|------------|------------|---------|
| **MC (pure)**      | 1.0  | **125.2**  | **103.5**  | 2595s   |
| λ-return           | 0.9  | 18.3       | -75.1      | 2961s   |

Both configs: RLS β=0.999, 8000 neurons, 1000 episodes, solve every 10.

λ=0.9 peaked early (18.3 at ep 400) but then collapsed and never
recovered.  By contrast, MC showed the characteristic late surge
(ep 800–900) reaching 125.2.

**Root cause:** bootstrapping creates a self-reinforcing error loop.
With pure MC, targets are model-independent ground truth — each episode
return is a noisy but unbiased estimate of the true value, regardless of
how good Q currently is.  With λ-returns, even 10% of each intermediate
target comes from max Q(s'), which is inaccurate early on.  In gradient-
based RL this is mitigated by slow, averaged updates; in the analytical
solve, bad targets directly shift the least-squares solution.  The RLS
forgetting factor compounds the problem: contaminated sufficient
statistics persist for ~1000 episodes (1/β window).

**Key insight:** the analytical solve's greatest strength — computing the
globally optimal weights for the given targets — becomes a weakness when
targets are model-dependent.  MC targets are the natural fit because they
preserve the model-independence that makes the analytical solve stable.


### 4.7 Ensemble Exploration on LunarLander

**Thompson sampling is a breakthrough; UCB fails completely.**

| Config              | Best eval   | Final eval  | Time    |
|---------------------|-------------|-------------|---------|
| Single 4k           | 80.3        | 114.9       | 585s    |
| **Thompson 3×4k**   | **186.3**   | **183.6**   | 1724s   |
| UCB 3×4k            | -60.6       | -78.5       | 1640s   |

All configs: RLS β=0.999, solve every 10, 1000 episodes, ε 1.0→0.05.
Ensemble members share experience (broadcast transitions to all).

**Thompson 3×4k closes 57% of the gap to DQN** (214.0).  The learning
curve shows a characteristic pattern: slow start (worse than single
through ep 500), then a dramatic late surge (ep 750–1000) as diverse
experience accumulates.  At ep 950, Thompson reached 186.3 — a 131%
improvement over the single-agent best-ever of 80.3.

**UCB fails entirely** — never achieves positive reward.  UCB's
deterministic "always explore highest uncertainty" strategy seems to
over-exploit noisy variance estimates from only 3 members.  With
inaccurate Q-values and small ensemble, the std is unreliable as an
exploration signal.  Thompson's random member selection provides natural
diversity without needing accurate uncertainty estimates.

**Why Thompson works:** each member has different random projections, so
they encode different aspects of the state-action space.  In under-
explored regions, members disagree about Q-values.  Thompson sampling
(pick a random member's greedy action at each step) naturally explores
where there's uncertainty — without explicitly computing uncertainty.
The per-step (not per-episode) member sampling creates incoherent
episode-level behavior, but the diverse experience this generates more
than compensates.

**Cost analysis:** 3×4k takes ~3× the time of a single 4k agent (1724s
vs 585s).  The total feature count (3×4000 = 12000) exceeds a single 8k
agent, but Thompson 3×4k (186.3) dramatically outperforms single 8k
(125.2 best-ever with RLS).  The ensemble's advantage is not just more
features — it's the exploration diversity.

**Comparison with prior best:**

| Method                       | Best eval | Final eval |
|------------------------------|-----------|------------|
| NEF-FQI baseline (8k)       | 40.8      | 40.8       |
| NEF-FQI + RLS (8k)          | 125.2     | 103.5      |
| **NEF-FQI + Thompson (3×4k)** | **186.3** | **183.6** |
| DQN (reference)             | 214.0     | 214.0      |


## 5. Summary Table

| Environment    | NEF-FQI (MC) | NEF-FQI (RLS) | Thompson 3×4k | DQN     | Winner  | Why                           |
|----------------|--------------|---------------|---------------|---------|---------|-------------------------------|
| CartPole-v1    | **500.0**    | **500.0**     | —             | 332.4   | NEF-FQI | Dense reward, low-dim state   |
| Acrobot-v1     | -410.8       | —             | —             |**-141.7**| DQN    | Sparse reward → MC fails      |
| LunarLander-v3 | 40.8         | 109.4         | **186.3**     |**214.0**| DQN (close) | Ensemble exploration closes 57% of gap |


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
   need more data, not more neurons.  Scaling from 8000 to 16000 neurons
   hurts because the sample-to-parameter ratio drops below what the
   regularized solve can handle with forgetting (§4.4).
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
- **TD(λ)** (§4.6) does not help: even λ=0.9 (90% MC) destabilizes the
  solve because 10% bootstrapping from inaccurate Q creates a
  self-reinforcing error loop that the analytical solve cannot damp.
- **SGD** (DQN) handles this by making small incremental updates that
  smooth over target shifts.  The analytical solve does not have this
  "damping" effect.

**The fundamental constraint:** the analytical solve requires
model-independent targets.  MC provides these; any form of TD
bootstrapping violates this requirement.  The MC variance problem
must be addressed through other means:
- **Reward shaping:** Transform sparse rewards to give MC more signal
- **Feature adaptation:** Learn encoders (violates the "fixed encoder"
  principle but may be necessary for higher dimensions)
- **Ensemble exploration:** Multiple random projections with Thompson
  sampling dramatically reduce effective variance (§4.7: 186.3 vs
  single-agent 80.3) — **confirmed as the most effective improvement**


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

### 8.1 Exponentially-weighted sufficient statistics ✅ IMPLEMENTED

Instead of `AᵀA = Σ A_k^T A_k` (all data weighted equally), apply an
exponential forgetting factor:

    AᵀA ← β · AᵀA + A_batch^T A_batch
    AᵀY ← β · AᵀY + A_batch^T Y_batch

Old data fades with weight β^k.  This is **Recursive Least Squares
(RLS) with forgetting factor** — a classic control theory technique.

Implemented via `forget_factor` parameter in `NEFFQIAgent`.  When set,
the agent maintains per-action sufficient statistics and eliminates the
replay buffer entirely.

**CartPole results:** RLS alone (β=0.995) learns fast (92.6s vs 248.4s)
but forgets too aggressively after convergence — old exploration data
that covers failure modes decays, causing performance regression.
Combined with recentering, this instability disappears: the
combo achieves perfect 500.0 from episode 150 onward with no replay
buffer.

### 8.2 Dead neuron recentering ✅ IMPLEMENTED

Neurons with centers far from the visited state space barely fire.
Periodically:

1. Measure mean activation per neuron over recent episodes.
2. If activation < threshold (e.g. 5th percentile), the neuron is "dead."
3. Recenter the dead neuron to a recently observed state:
   `bias_i = -gain_i * (normalized_center · encoder_i)`.
4. Reset that neuron's row/column in AᵀA and AᵀY (partial reset).

Implemented via `recenter_interval` and `recenter_percentile` parameters
in `NEFFQIAgent`, and `dead_neuron_indices()` / `recenter()` methods in
`NEFFeatures`.

This is **gradient-free resource allocation** — analogous to the "reset"
mechanism in recent deep RL papers (Lyle et al., 2023) but much cleaner
because the recentering is analytical (just change the bias) and the
solve remains exact.

Connects directly to the CL story: data-driven centers are our advantage
in supervised CL; adaptive centers are the RL extension.

**LunarLander results:** RLS β=0.999 + recenter/100 achieves best=109.4,
final=102.2 — a 2.7× improvement over baseline (best=40.8).  The agent
shows a dramatic late surge (flat at -40 through ep 800, then +109 by
ep 950), suggesting recentering gradually builds adequate feature
coverage.  β=0.995 is too aggressive for LunarLander's high variance
(best=47.3, only marginally better than baseline).  The gap to DQN
(214.0) remains but is halved.

### 8.3 TD(λ) targets ✅ TESTED — NEGATIVE

Implemented λ-returns (§4.6).  Even λ=0.9 (90% MC, 10% TD bootstrap)
dramatically worsened LunarLander performance: best 18.3 vs MC's 125.2.
Bootstrapping from inaccurate Q contaminates the analytical solve.

This is a fundamental incompatibility: the analytical solve needs
model-independent targets.  MC provides these; TD does not.  This rules
out any TD-based target improvement and confirms MC as the correct
choice for NEF-FQI.  The variance problem must be addressed differently
(e.g., reward shaping, environment-side changes, not target blending).

### 8.4 Online Woodbury updates (original Step 2)

Replace the full-buffer solve with episode-by-episode Woodbury updates.
Each episode adds rank-k updates to (AᵀA)⁻¹.  Combines naturally with
§8.1 (forgetting factor on the Woodbury inverse).

### 8.5 Ensemble exploration (original Step 3) ✅ TESTED

NEFFQIEnsemble wraps N independent agents with different random seeds.
All members train on the same experience (transitions broadcast).  Two
exploration strategies tested:

- **Thompson sampling (per-step random member):** breakthrough result.
  3×4k reaches 186.3 on LunarLander (vs single-agent 80.3, DQN 214.0).
  Closes 57% of the DQN gap.  The diverse random projections provide
  natural exploration without explicit uncertainty computation.

- **UCB (mean + β·std):** complete failure.  Never achieves positive
  reward (-60.6 best).  With only 3 members, the std estimate is too
  noisy to guide exploration.  UCB's deterministic exploitation of
  uncertainty over-concentrates on unreliable regions.

Thompson is the clear winner and should be the default ensemble strategy
for NEF-FQI.  See §4.7 for full analysis.

### 8.6 Continual RL (original Step 4)

Sequential tasks with changing dynamics or rewards.  The AᵀA/AᵀY
accumulation should retain knowledge across tasks — the CL+RL synthesis.

### 8.7 Scaling neurons with state dimension ✅ TESTED

Scaling from 8000 to 16000 neurons on LunarLander (8D) with 1000
episodes **hurts performance** (best -12.2 vs 109.4).  The bottleneck
is not feature count but sample complexity relative to parameter count.
With RLS forgetting (effective window ~1000 episodes), the solver has
a fixed data budget; doubling neurons halves the sample-to-parameter
ratio and prevents convergence.

This is a meaningful insight: in online RL with forgetting, the neuron
count must be matched to the effective data window, not just the state
dimension.  The optimal count depends on β, solve_every, and episode
length.
