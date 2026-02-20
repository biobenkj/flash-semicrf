# FAQ Part 6: Conceptual Bridge for Genomics & Biostatistics Researchers

This document maps semi-CRF concepts to the statistical and genomic
frameworks you already know. If you're comfortable with HMMs,
posterior probabilities, and Bayesian inference but "marginals" and
"partition function" in the ML sense feel unfamiliar, start here.

---

## Connecting to what you already know

### Q: I know HMMs from genomics (ChromHMM, GeneMark, Augustus). How does a semi-CRF relate?

An HMM defines a **generative** model: it specifies P(observations, hidden states)
jointly, then uses Bayes' rule to infer the hidden states. A CRF defines a
**discriminative** model: it directly parameterizes P(hidden states | observations)
without modeling how the observations were generated.

In practice, the inference algorithms are nearly identical. Both use
forward-backward to compute posteriors (marginals) and Viterbi to find
the best path. The difference is in *training*:

| | HMM | CRF / Semi-CRF |
|---|---|---|
| **Models** | P(observations, states) | P(states \| observations) |
| **Training** | EM (Baum-Welch) or MLE | Gradient descent on conditional log-likelihood |
| **Features** | Emission distributions (e.g., Gaussian) | Arbitrary neural network outputs |
| **Inference** | Forward-backward, Viterbi | Forward-backward, Viterbi (same algorithms) |

A **semi-CRF** extends the CRF by modeling *segments* (contiguous runs of the
same state with explicit duration) rather than single positions. This is
analogous to the relationship between an HMM and a hidden semi-Markov model
(HSMM, also called a generalized HMM or GHMM):

| | Position-level | Segment-level |
|---|---|---|
| **Generative** | HMM | HSMM / Generalized HMM |
| **Discriminative** | Linear-chain CRF | Semi-CRF |

If you've used Augustus or GeneMark, you've used a generalized HMM — a
generative semi-Markov model with hand-crafted features. A semi-CRF is the
discriminative counterpart, where the features come from a neural network
(like Mamba or a BiLSTM) and the model is trained end-to-end.

### Q: What does "discriminative" buy me over a generative model?

Two things:

1. **You don't need to model the observations.** An HMM must specify
   P(nucleotide | hidden state), which requires assumptions about emission
   distributions. A CRF takes encoder representations as input and only
   models the structured output. The encoder can be arbitrarily complex
   (a transformer, Mamba, etc.) without changing the inference algorithm.

2. **The training objective directly optimizes prediction accuracy.** An HMM
   trained with maximum likelihood optimizes the joint P(x, y). A CRF
   trained with conditional log-likelihood optimizes P(y | x) directly,
   which is what you actually care about for prediction. This is the same
   argument for why logistic regression can outperform naive Bayes even
   though they use the same model family.

### Q: What is ChromHMM, really, in this framework?

ChromHMM is a hidden Markov model (not semi-Markov) trained with EM on
binarized ChIP-seq data. It models chromatin states at fixed-size genomic
bins (typically 200bp). Because it's an HMM, not an HSMM or semi-CRF, it
has no explicit duration model — state durations emerge implicitly from
the self-transition probability. This is why ChromHMM states don't have
well-calibrated duration distributions, and why post-hoc segment merging
is common in ChromHMM workflows.

A semi-CRF replaces this with explicit duration modeling: the parameter
$\mathcal{B}_{k,c}$ directly scores "a segment of label $c$ lasting $k$
positions." This is a fundamentally different (and richer) inductive bias.

---

## Marginals: connecting ML and statistics terminology

### Q: In statistics, a marginal is obtained by integrating out variables from a joint distribution. What's being "integrated" in a semi-CRF?

Exactly the same thing, but the joint distribution is over **segmentations**
rather than continuous parameters.

**The sample space:** All valid segmentations of your sequence. A valid
segmentation tiles the sequence into non-overlapping, contiguous segments,
each with a label and a duration. For a sequence of length T with C labels
and maximum duration K, the number of valid segmentations is exponential
in T.

**The joint distribution:** Each segmentation $y$ gets a probability:

$$P(y \mid x) = \frac{1}{Z} \exp\!\Bigl(\sum_{i} \psi(s_i, k_i, c_i, c_{i-1})\Bigr)$$

where $\psi$ is the edge potential (content score + transition + duration
bias), and $Z = \sum_{y'} \exp(\text{score}(y'))$ sums over all valid
segmentations. This is a Gibbs distribution — the same exponential family
form you see in statistical mechanics and in Bayesian posteriors with
log-linear models.

**The marginal:** The marginal probability of a specific local configuration —
say, "a segment of label $c$ with duration $k$ ending at position $t$,
preceded by label $c'$" — is obtained by summing the joint distribution
over all segmentations that contain that configuration:

$$\mu(t, k, c, c') = \sum_{\substack{y \,:\, \text{segment} \\ (t,k,c,c') \in y}} P(y \mid x)$$

This is marginalization in the classical sense: summing (integrating) the
joint over all other variables (all the other segments in the segmentation),
keeping one local piece fixed.

### Q: How is this summation tractable? There are exponentially many segmentations.

The forward-backward algorithm. This is the same algorithm used in HMMs, extended
to handle segments of variable duration.

- $\alpha_t(c)$ (the forward message) accumulates the total probability mass
  of all valid partial segmentations ending at position $t$ in state $c$.
  It sums over all valid "pasts."

- $\beta_t(c)$ (the backward message) accumulates the total probability mass
  of all valid partial segmentations from position $t$ onward in state $c$.
  It sums over all valid "futures."

The marginal then factors as:

$$\mu(t, k, c, c') \propto \exp\!\bigl(\alpha_{t-k}(c') + \psi(t, k, c, c') + \beta_t(c)\bigr)$$

This works because of the Markov property: given the current segment's
identity, the past and future are conditionally independent. So the product
$\alpha \times \text{edge} \times \beta$ correctly accounts for all valid
complete segmentations containing this specific segment — without
enumerating them.

Normalizing by $Z$ gives the true marginal probability:

$$\mu(t, k, c, c') = \exp\!\bigl(\alpha_{t-k}(c') + \psi(t, k, c, c') + \beta_t(c) - \log Z\bigr)$$

**The analogy to Bayesian computation:** Just as Markov chain Monte Carlo
(MCMC) lets you compute posterior expectations without enumerating all
parameter configurations, the forward-backward algorithm lets you compute
marginals without enumerating all segmentations. The difference is that
forward-backward is *exact* (no sampling error) because the Markov
structure permits a closed-form dynamic programming solution. MCMC is
needed when the model lacks this factorization.

### Q: Are these marginals the same as posterior probabilities?

Yes. In the CRF framework, $P(y \mid x)$ is already a conditional
(posterior) distribution — conditioned on the observed sequence $x$. The
marginal $\mu(t, k, c, c')$ is the posterior probability of a specific
local configuration, marginalizing over all other structural decisions.

If you're used to Bayesian language: the "prior" is implicit in the model
parameters (transition scores, duration bias). The "likelihood" is the
emission/content score from the encoder. The "posterior" is the normalized
Gibbs distribution $P(y \mid x)$. The marginals are posterior marginals.

The key difference from a Bayesian treatment of model *parameters* is that
here we're computing posteriors over the *output structure* (the
segmentation), not over the model's weights. The model weights are point
estimates found by gradient descent, not integrated over.

---

## Training: how the partition function enters the loss

### Q: How does training a semi-CRF relate to maximum likelihood?

The standard CRF training objective is the conditional log-likelihood:

$$\ell(\theta) = \log P(y^* \mid x) = \text{score}(y^*) - \log Z$$

where $y^*$ is the ground-truth segmentation. This is maximum likelihood
estimation of $P(y \mid x)$, which is the discriminative analog of MLE
in generative models.

The gradient of this loss is:

$$\nabla_\theta \ell = \nabla_\theta \text{score}(y^*) - \mathbb{E}_{P(y|x)}[\nabla_\theta \text{score}(y)]$$

The first term pushes the model to increase the score of the correct
segmentation. The second term (the expected gradient under the model's
current distribution) pushes the model to decrease the scores of
everything else, weighted by how probable the model currently thinks
each segmentation is. The marginals computed by forward-backward are
exactly what's needed to evaluate this expectation.

This is the same structure as the EM algorithm's M-step: the expected
sufficient statistics under the posterior. The difference is that CRF
training computes these expectations exactly (via forward-backward) and
takes gradient steps, while EM alternates between exact E-steps and
closed-form M-steps. In genomics terms: Baum-Welch for an HMM and
gradient-based CRF training are solving the same optimization problem
from different sides.

### Q: What does it mean that the CRF layer provides a "structured training signal"?

When you train with per-position cross-entropy (as with a softmax
classifier), each position's loss is computed independently. The gradient
for position $t$ depends only on the prediction and label at position $t$.

When you train with CRF loss, the gradient at every position depends on
the *global* segmentation structure. The marginals $\mu(t, k, c, c')$
reflect how every segment interacts with every other segment in the
sequence. This means the encoder receives gradients that are informed by
segment-level structure (durations, transitions, boundary consistency),
not just per-position labels.

Concretely: if the encoder produces emissions that are correct per-position
but inconsistent at segment boundaries (e.g., predicting exon for 47
positions, then intron for 1 position, then exon for 52 positions), the
CRF loss will penalize this because the marginals assign low probability
to 1-position intron segments. The encoder learns to smooth its
representations to avoid producing structurally implausible outputs. A
per-position loss has no mechanism to provide this feedback.

---

## Exact vs. approximate inference

### Q: Why does "exact" matter? In statistics, we use MCMC approximations all the time.

Exact inference matters for three reasons:

1. **Correct gradients.** The CRF training loss requires $\log Z$, and
   its gradient requires the marginals. If these are approximate, the
   gradient is biased, and the model may learn to exploit the approximation
   rather than the true structure. This is analogous to how a biased MCMC
   sampler produces biased posterior estimates, except that here the bias
   compounds across thousands of gradient steps during training.

2. **Calibrated marginals.** If you use marginals for downstream decisions
   ("the probability that this region is an exon is 0.93"), you want that
   number to be correct under the model. Approximate inference gives you
   marginals under the *approximation*, which may not correspond to the
   model's actual distribution.

3. **No approximation hyperparameters.** Approximate methods (beam search,
   filtering, pruning) introduce parameters — beam width, filtering
   threshold, pruning criterion — that interact with training in
   hard-to-predict ways. Exact inference means the only hyperparameters
   are the model's own (K, architecture choices). One fewer thing to tune,
   one fewer thing to justify.

The semi-CRF is one of the cases where exact inference *is* tractable
(unlike, say, loopy graphical models where you're forced to approximate).
The forward-backward algorithm runs in $O(TKC^2)$ time, which is
polynomial. The question is whether the constant factors make it practical
at the scale you need — and that's what flash-semicrf is designed to solve.

### Q: How does this compare to MCMC or variational inference?

| Method | Exact? | When to use |
|---|---|---|
| **Forward-backward** (CRF/semi-CRF) | Yes | Chain-structured or tree-structured models |
| **MCMC** (Metropolis, Gibbs sampling) | Asymptotically | Models with complex dependencies, no DP structure |
| **Variational inference** (mean field, etc.) | No (biased) | Large models where exact inference is intractable |
| **Beam search** | No | Decoding (finding good sequences), not training |

For semi-CRFs, exact inference is tractable, so there's no reason to
approximate. The challenge is making the exact computation *fast enough*
at genomic scale, which is what the streaming algorithm addresses.

---

## The centering trick: a biostatistics perspective

### Q: The emission centering looks like it adds a data-dependent prior. Is this an empirical Bayes procedure?

It has the structure of one, yes. Here's the connection.

In classical empirical Bayes (Efron, 2010), you estimate prior
hyperparameters from the data rather than specifying them a priori.
The estimated prior then regularizes individual-level estimates toward the
population mean.

Emission centering does something analogous at the sequence level. The
baseline $\nu_{b,c}$ (the mean emission for label $c$ across the
sequence) is a data-dependent quantity that shifts the effective duration
model:

$$\mathcal{B}^{\text{eff}}_{k,c} = \mathcal{B}_{k,c} - \nu_{b,c} \cdot k$$

The learned duration bias $\mathcal{B}_{k,c}$ captures *population-level*
duration preferences (across all training sequences). The per-sequence
correction $-\nu_{b,c} \cdot k$ adjusts for sequence-specific emission
statistics. Labels that are unusually prevalent in a given sequence
(high $\nu_{b,c}$) get penalized for long segments; rare labels get a
relative boost.

This is shrinkage: individual sequence-level behavior is pulled toward
the population prior, with the strength of shrinkage proportional to
the deviation of the sequence's emission statistics from the population
mean.

### Q: When does centering NOT help?

Two settings:

1. **Balanced label proportions.** When all labels are equally prevalent,
   $\nu_{b,c}$ is approximately the same for all labels, and the
   centering shift is approximately label-independent. In this case,
   it reduces to a path-invariant transformation that doesn't change
   the model's distribution over segmentations.

2. **CRF upstream of another model.** If the semi-CRF's marginals are
   used as *input features* for a downstream model (rather than as the
   final output), per-label centering alters the marginals in a
   sample-specific way that the downstream model hasn't been trained to
   expect. In this setting, use path-invariant centering (e.g., subtract
   the per-position maximum across labels) to preserve the canonical
   semi-CRF posterior.

---

## Practical mappings

### Q: I'm used to thinking about P-values and confidence intervals. What's the semi-CRF equivalent?

The semi-CRF gives you:

- **Marginal probabilities** at each position: "the posterior probability
  that a segment boundary falls at position $t$ is 0.87." This is a
  direct probability statement, not a P-value (no null hypothesis).

- **The Viterbi path**: the single most probable segmentation. Analogous
  to a maximum a posteriori (MAP) estimate.

- **Entropy**: a measure of the model's overall uncertainty about the
  segmentation. High entropy means many segmentations have similar
  probability; low entropy means the model is confident.

The marginals are better calibrated than classifier confidence scores
(softmax probabilities) because they account for structural constraints.
A softmax classifier might say "position 47 is 95% exon," but a semi-CRF
marginal says "across all valid segmentations, 95% of the probability
mass has position 47 inside an exon segment" — a much stronger statement
because it incorporates duration and transition consistency.

### Q: What's the analog of a likelihood ratio test?

The partition function $Z$ plays the role of the marginal likelihood
(model evidence) in Bayesian model comparison. If you train two semi-CRF
models (say, one with K=100 and one with K=500) on the same data, the
model with higher $\log P(y^* \mid x) = \text{score}(y^*) - \log Z$
on held-out data is the better model, analogous to comparing Bayes
factors. This gives you a principled way to select K rather than
treating it as an arbitrary hyperparameter.

---

**See also:** [The Basics](01-basics.md) · [Jargon Decoder](03-jargon-decoder.md) · [Streaming supplement](../manuscript/streaming_algorithm_supplement.tex)
