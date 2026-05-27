# The SSUP three-paper series, in one page

The three papers study a simple question: how far can we get if a learning
system keeps its internal representation fixed and learns only a final linear
readout?  In ordinary machine learning, a model is often trained by repeatedly
nudging many internal parameters with gradient descent.  Here the central
object is different.  Inputs are first mapped into a large fixed set of
nonlinear "neural activities" collected in a matrix $A$.  The only fitted
quantity is a decoder $D$, found in one regularized least-squares solve:

$$
D = (A^\top A + \alpha I)^{-1} A^\top Y.
$$

The useful fact is that the learner's state is almost entirely summarized by
two matrices.  $A^\top A$ records the geometry of the fixed representation:
which activity patterns look similar or different.  $A^\top Y$ records what
those activities are being asked to predict: labels, task outputs, returns, or
successor targets.  Both are sums over examples, so they can be accumulated,
merged, updated online, or split by action.  The series calls this the
"sufficient statistics as a universal primitive" view.  The word "universal"
is deliberately tested rather than assumed.

A useful picture is that each fixed neuron asks one simple geometric question.
Choose a training example $d_i$ and a direction $e_i$, like drawing an arrow
through that example.  For a new input $x$, the neuron measures how far $x$ has
moved away from $d_i$ along that arrow, and then forgets the sign:
$a_i = |g_i e_i \cdot (x - d_i)|$.  The gain $g_i$ just changes the ruler's
scale.  One such measurement is weak.  Thousands of them form an
over-representation: many more simple measurements than the input dimension
strictly requires.  This only helps because of the nonlinearity, here the
absolute value.  If the neuron kept the signed linear measurement, then any
linear readout of many neurons would collapse back into one ordinary linear
formula.  The extra neurons would be redundant.  The absolute value folds space
at many data-centered locations, so a final linear readout can combine many
folded distance measurements into a piecewise-linear decision rule.

**Paper I: supervised learning.**  The first paper asks what this primitive
can do in the ordinary setting where each input comes with a known target, such
as an image label.  The method is a rate-based specialization of the Neural
Engineering Framework: fixed encoders turn inputs into many nonlinear
activities, and the decoder is solved analytically.  The paper's main lesson is
that the primitive is fast, clean, and surprisingly useful, but not magic.
Static image classifiers work well enough to be informative, especially when
the fixed features contain the right visual bias, but generic fixed encoders
usually trail tuned gradient-trained networks.  The strongest positive result
is temporal: Stream-NEF summarizes short windows of a sequence, pools them, and
solves one readout.  On row-wise sequential MNIST it comes close to a tuned LSTM
while using less wall-clock time.  The strongest negative result is equally
important: simply stacking analytic nonlinear layers does not automatically
produce the kind of useful depth that gradient-trained neural networks get.
Paper I therefore establishes the primitive, its geometry, its speed, and its
limits in supervised learning.

**Paper II: continual learning.**  The second paper asks what happens when data
arrives as a stream of tasks rather than all at once.  Many neural networks
"forget" old tasks because gradient updates for the new task overwrite
parameters that mattered for the old one.  With fixed features and additive
sufficient statistics, the situation is algebraically different.  If the target
semantics remain stable, then accumulating $A^\top A$ and $A^\top Y$ task by
task gives the same decoder as training once on the union of all tasks.  In the
paper's terms, the forgetting axis has length zero relative to the joint
fixed-feature model.  This identity is verified across standard split and
permuted image benchmarks, and the paper compares it with common continual
learning baselines.  The result is strongest in class-incremental settings,
where each class writes into stable decoder columns.  The limitation is also
clear: removing forgetting from the readout does not remove the accuracy ceiling
imposed by the fixed feature map.  Paper II is therefore less a new trick than
an accounting result: if the learner's state is the right pair of sums, then
continual learning becomes preservation of sufficient statistics.

**Paper III: reinforcement learning.**  The third paper moves to control
problems, where the target $Y$ is no longer given by a teacher.  A control agent
must construct targets from rewards, future returns, goals, or successor
features.  This is where the series' central distinction becomes sharp:
$A^\top A$ can still be accumulated, but $A^\top Y$ is now the main design
problem.  The paper shows that analytic fixed-feature control can work when the
targets are stable and task-matched.  A non-bootstrapped NEF action-value
regression variant reaches the LunarLander reference region under the paper's
protocol and passes an easier CartPole sanity check.  Sparse-reward problems
need additional target structure: successor-style targets produce useful signal
for MountainCar without solving it, and a directional successor readout gives a
stable Acrobot signal.  The failures are part of the result.  One-step
bootstrapped targets collapse in the tested LunarLander setting; merely adding
more width does not fix sparse control; mismatched goal handles stay flat.
Paper III therefore does not claim a general-purpose RL algorithm.  It shows
where the same sufficient-statistics interface can support control, and where
target construction becomes the real learning problem.

Taken together, the papers make a bounded claim.  Fixed encoders plus analytic
decoders are not a replacement for all of modern machine learning.  They are a
compact, auditable primitive with unusually clear algebra.  In supervised
learning, the primitive fits labels quickly; in continual learning, its
additivity removes one source of forgetting; in reinforcement learning, it
exposes target construction as the hard part.  The value of the series is not
that every result wins, but that the same two matrices keep explaining both the
successes and the failures.
