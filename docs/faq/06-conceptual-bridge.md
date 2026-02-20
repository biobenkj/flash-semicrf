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

|                  | HMM                          | CRF / Semi-CRF                                   |
| ---------------- | ---------------------------- | ------------------------------------------------- |
| **Models**       | P(observations, states)      | P(states \| observations)                         |
| **Training**     | EM (Baum-Welch) or MLE       | Gradient descent on conditional log-likelihood     |
| **Features**     | Emission distributions (e.g., Gaussian) | Arbitrary neural network outputs          |
| **Inference**    | Forward-backward, Viterbi    | Forward-backward, Viterbi (same algorithms)        |

A **semi-CRF** extends the CRF by modeling *segments* (contiguous runs of the
same state with explicit duration) rather than single positions. This is
analogous to the relationship between an HMM and a hidden semi-Markov model
(HSMM, also called a generalized HMM or GHMM):

|                    | Position-level    | Segment-level           |
| ------------------ | ----------------- | ----------------------- |
| **Generative**     | HMM               | HSMM / Generalized HMM  |
| **Discriminative** | Linear-chain CRF  | Semi-CRF                |

If you've used Augustus or GeneMark, you've used a generalized HMM — a
generative semi-Markov model with hand-crafted features (Stanke & Waack, 2003).
A semi-CRF (Sarawagi & Cohen, 2004) is the discriminative counterpart, where
the features come from a neural network (like Mamba or a BiLSTM) and the model
is trained end-to-end (Kong et al., 2016).

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
   which is what you actually care about for prediction (Lafferty et al.,
   2001). This is the same argument for why logistic regression can
   outperform naive Bayes even though they use the same model family.

### Q: What is ChromHMM, really, in this framework?

ChromHMM (Ernst & Kellis, 2012) is a hidden Markov model (not semi-Markov)
trained with EM on binarized ChIP-seq data. It models chromatin states at
fixed-size genomic bins (typically 200bp). Because it's an HMM, not an
HSMM or semi-CRF, it has no explicit duration model — state durations
emerge implicitly from the self-transition probability. This is why
ChromHMM states don't have well-calibrated duration distributions, and
why post-hoc segment merging is common in ChromHMM workflows.

A semi-CRF replaces this with explicit duration modeling: the parameter
`B(k, c)` directly scores "a segment of label `c` lasting `k` positions."
This is a fundamentally different (and richer) inductive bias.

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

**The joint distribution:** Each segmentation `y` gets a probability:

```
P(y | x) = (1/Z) * exp( sum_i  psi(s_i, k_i, c_i, c_{i-1}) )
```

where `psi` is the edge potential (content score + transition + duration
bias), and `Z = sum_{y'} exp(score(y'))` sums over all valid
segmentations. This is a Gibbs distribution — the same exponential family
form you see in statistical mechanics and in Bayesian posteriors with
log-linear models.

**The marginal:** The marginal probability of a specific local configuration —
say, "a segment of label `c` with duration `k` ending at position `t`,
preceded by label `c'`" — is obtained by summing the joint distribution
over all segmentations that contain that configuration:

```
mu(t, k, c, c') = sum_{y : segment (t,k,c,c') in y}  P(y | x)
```

This is marginalization in the classical sense: summing (integrating) the
joint over all other variables (all the other segments in the segmentation),
keeping one local piece fixed.

### Q: How is this summation tractable? There are exponentially many segmentations

The forward-backward algorithm (Rabiner, 1989). This is the same algorithm
used in HMMs, extended to handle segments of variable duration.

- `alpha_t(c)` (the forward message) accumulates the total probability mass
  of all valid partial segmentations ending at position `t` in state `c`.
  It sums over all valid "pasts."

- `beta_t(c)` (the backward message) accumulates the total probability mass
  of all valid partial segmentations from position `t` onward in state `c`.
  It sums over all valid "futures."

The marginal then factors as:

```
mu(t, k, c, c') ~ exp( alpha_{t-k}(c') + psi(t, k, c, c') + beta_t(c) )
```

This works because of the Markov property: given the current segment's
identity, the past and future are conditionally independent. So the product
`alpha * edge * beta` correctly accounts for all valid complete
segmentations containing this specific segment — without enumerating them.

Normalizing by Z gives the true marginal probability:

```
mu(t, k, c, c') = exp( alpha_{t-k}(c') + psi(t, k, c, c') + beta_t(c) - log Z )
```

**The analogy to Bayesian computation:** Just as Markov chain Monte Carlo
(MCMC) lets you compute posterior expectations without enumerating all
parameter configurations, the forward-backward algorithm lets you compute
marginals without enumerating all segmentations. The difference is that
forward-backward is *exact* (no sampling error) because the Markov
structure permits a closed-form dynamic programming solution. MCMC is
needed when the model lacks this factorization.

### Q: Are these marginals the same as posterior probabilities?

Yes. In the CRF framework, `P(y | x)` is already a conditional
(posterior) distribution — conditioned on the observed sequence `x`. The
marginal `mu(t, k, c, c')` is the posterior probability of a specific
local configuration, marginalizing over all other structural decisions.

If you're used to Bayesian language: the "prior" is implicit in the model
parameters (transition scores, duration bias). The "likelihood" is the
emission/content score from the encoder. The "posterior" is the normalized
Gibbs distribution `P(y | x)`. The marginals are posterior marginals.

The key difference from a Bayesian treatment of model *parameters* is that
here we're computing posteriors over the *output structure* (the
segmentation), not over the model's weights. The model weights are point
estimates found by gradient descent, not integrated over.

---

## Training: how the partition function enters the loss

### Q: How does training a semi-CRF relate to maximum likelihood?

The canonical CRF training objective is the conditional log-likelihood:

```
L(theta) = log P(y* | x) = score(y*) - log Z
```

where `y*` is the ground-truth segmentation. This is maximum likelihood
estimation of `P(y | x)`, which is the discriminative analog of MLE
in generative models.

In flash-semicrf, emissions are globally centered before cumulative sums
for numerical stability at large `T`. As described in the streaming
supplement, this introduces a sequence-dependent adjustment to segment
scores (an adaptive duration prior), so the optimized objective is the
centered model used by this implementation. The gradient structure below
is unchanged, but absolute reported loss values can shift (and may become
negative), so compare trends within the same implementation and rely on
task metrics for cross-library comparisons.

The gradient of this loss is:

```
grad_theta L = grad_theta score(y*) - E_{P(y|x)}[ grad_theta score(y) ]
```

The first term pushes the model to increase the score of the correct
segmentation. The second term (the expected gradient under the model's
current distribution) pushes the model to decrease the scores of
everything else, weighted by how probable the model currently thinks
each segmentation is. The marginals computed by forward-backward are
exactly what's needed to evaluate this expectation.

This has the same expectation-of-sufficient-statistics structure used in
EM-style updates. But the objective is different: Baum-Welch for HMMs
maximizes the generative joint likelihood `P(x, y)`, while CRF training
maximizes the discriminative conditional likelihood `P(y | x)`.

### Q: What does it mean that the CRF layer provides a "structured training signal"?

When you train with per-position cross-entropy (as with a softmax
classifier), each position's loss is computed independently. The gradient
for position `t` depends only on the prediction and label at position `t`.

When you train with CRF loss, the gradient at every position depends on
the *global* segmentation structure. The marginals `mu(t, k, c, c')`
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

This insight — that the CRF loss shapes the encoder's learned
representations, not just the output — was first demonstrated for
semi-CRFs by Kong et al. (2016).

---

## Exact vs. approximate inference

### Q: Why does "exact" matter? In statistics, we use MCMC approximations all the time

Exact inference matters for three reasons:

1. **Correct gradients.** The CRF training loss requires `log Z`, and
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
   one fewer thing to justify. Contrast with Zaratiana et al. (2023), who
   achieve speed via filtering but introduce threshold parameters that
   alter the model's distribution.

The semi-CRF is one of the cases where exact inference *is* tractable
(unlike, say, loopy graphical models where you're forced to approximate).
The forward-backward algorithm runs in O(TKC^2) time, which is
polynomial. The question is whether the constant factors make it practical
at the scale you need — and that's what flash-semicrf is designed to solve.

### Q: How does this compare to MCMC or variational inference?

| Method                                 | Exact?          | When to use                                         |
| -------------------------------------- | --------------- | --------------------------------------------------- |
| **Forward-backward** (CRF/semi-CRF)   | Yes             | Chain-structured or tree-structured models           |
| **MCMC** (Metropolis, Gibbs sampling)  | Asymptotically  | Models with complex dependencies, no DP structure    |
| **Variational inference** (mean field) | No (biased)     | Large models where exact inference is intractable    |
| **Beam search**                        | No              | Decoding (finding good sequences), not training      |

For semi-CRFs, exact inference is tractable, so approximation is an
engineering tradeoff rather than a mathematical necessity. The challenge
is making the exact computation *fast enough* at genomic scale, which is
what the streaming algorithm addresses.

---

## The centering trick: a biostatistics perspective

### Q: The emission centering looks like it adds a data-dependent prior. Is this an empirical Bayes procedure?

It has the structure of one, yes. Here's the connection.

In classical empirical Bayes (Efron, 2010), you estimate prior
hyperparameters from the data rather than specifying them a priori.
The estimated prior then regularizes individual-level estimates toward the
population mean.

Emission centering does something analogous at the sequence level. The
baseline `v(b, c)` (the mean emission for label `c` across the
sequence) is a data-dependent quantity that shifts the effective duration
model:

```
B_eff(k, c) = B(k, c) - v(b, c) * k
```

The learned duration bias `B(k, c)` captures *population-level*
duration preferences (across all training sequences). The per-sequence
correction `-v(b, c) * k` adjusts for sequence-specific emission
statistics. Labels that are unusually prevalent in a given sequence
(high `v(b, c)`) get penalized for long segments; rare labels get a
relative boost.

This is shrinkage: individual sequence-level behavior is pulled toward
the population prior, with the strength of shrinkage proportional to
the deviation of the sequence's emission statistics from the population
mean.

### Q: When does centering NOT help?

When all labels are approximately equally prevalent. In that case,
`v(b, c)` is approximately the same for all labels, and the centering
shift is approximately label-independent. It reduces to a path-invariant
transformation that doesn't change the model's distribution over
segmentations.

In ablation experiments on synthetic sequences with balanced label
proportions (25% each), all centering modes produced nearly identical
segmentations, confirming that the adaptive prior activates specifically
in response to label imbalance rather than acting as a blanket
regularizer.

---

## Practical mappings

### Q: I'm used to thinking about P-values and confidence intervals. What's the semi-CRF equivalent?

The semi-CRF gives you:

- **Marginal probabilities** at each position: "the posterior probability
  that a segment boundary falls at position `t` is 0.87." This is a
  direct probability statement, not a P-value (no null hypothesis).

- **The Viterbi path**: the single most probable segmentation. Analogous
  to a maximum a posteriori (MAP) estimate.

- **Entropy**: a measure of the model's overall uncertainty about the
  segmentation. High entropy means many segmentations have similar
  probability; low entropy means the model is confident. flash-semicrf
  computes entropy in two ways: *exactly* via the entropy semiring
  (Li & Eisner, 2009) when the edge tensor fits in memory (T < ~10K),
  or *approximately* from boundary marginals at any scale via the
  streaming Triton kernel. The streaming kernel itself computes log Z
  and marginals; entropy is derived downstream, not directly from the
  kernel.

The marginals can be better calibrated than classifier confidence scores
(softmax probabilities) because they account for structural constraints.
A softmax classifier might say "position 47 is 95% exon," but a semi-CRF
marginal says "across all valid segmentations, 95% of the probability
mass has position 47 inside an exon segment" — a much stronger statement
because it incorporates duration and transition consistency.

### Q: What's the analog of a likelihood ratio test?

The closest analog is comparing held-out conditional log-likelihood, not
Bayes factors. In a CRF, `Z(x)` is a conditional normalizer for each
input `x`, not Bayesian model evidence.

If you train two semi-CRF models (for example, `K=100` vs `K=500`) on the
same held-out set, compare:

```
score(y*) - log Z
```

(equivalently the centered loss `log Z - score(y*)` used in flash-semicrf,
where lower is better within this implementation). The model with better
held-out conditional likelihood is preferred.

---

## References

Papers cited in this document, organized by topic. These are the foundational
works that connect the genomics, statistics, and ML perspectives discussed above.

**CRFs and semi-CRFs (the core models):**

- Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional Random Fields:
  Probabilistic Models for Segmenting and Labeling Sequence Data. *ICML*.
  — The original CRF paper. Introduced the discriminative alternative to HMMs.

- Sarawagi, S. & Cohen, W. W. (2004). Semi-Markov Conditional Random Fields
  for Information Extraction. *NeurIPS*.
  — The original semi-CRF paper. Extended CRFs to model segments with explicit
  duration.

- Kong, L., Dyer, C., & Smith, N. A. (2016). Segmental Recurrent Neural Networks.
  *ICLR*.
  — First integration of semi-CRFs as a differentiable layer in a neural network,
  trained end-to-end with gradient descent.

- Zaratiana, U., et al. (2023). Filtered Semi-Markov CRF. *Findings of EMNLP*.
  — Approximate semi-CRF inference via pruning. Contrast with the exact
  inference approach in flash-semicrf.

**HMMs and HSMMs in genomics (what you already know):**

- Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition. *Proceedings of the IEEE*, 77(2), 257-286.
  — The standard HMM reference. Forward-backward, Viterbi, Baum-Welch.

- Durbin, R., Eddy, S., Krogh, A., & Mitchison, G. (1998). *Biological Sequence
  Analysis: Probabilistic Models of Proteins and Nucleic Acids*. Cambridge
  University Press.
  — The standard reference for HMMs and profile HMMs in computational biology.

- Ernst, J. & Kellis, M. (2012). ChromHMM: automating chromatin-state discovery
  and characterization. *Nature Methods*, 9(3), 215-216.
  — The widely-used HMM for chromatin state annotation. Position-level HMM,
  no explicit duration model.

- Stanke, M. & Waack, S. (2003). Gene prediction with a hidden Markov model and
  a new intron submodel. *Bioinformatics*, 19(suppl\_2), ii215-ii225.
  — Augustus gene finder. Uses a generalized HMM (GHMM) with explicit
  intron/exon length distributions — the generative analog of a semi-CRF.

**Semirings and expectation semirings:**

- Eisner, J. (2002). Parameter estimation for probabilistic finite-state
  transducers. *ACL*.
  — Introduced the expectation semiring framework: same DP algorithm, different
  algebraic operations, different quantities computed.

- Li, Z. & Eisner, J. (2009). First- and Second-Order Expectation Semirings
  with Applications to Minimum-Risk Training on Translation Forests. *EMNLP*.
  — Extended expectation semirings to compute entropy, variance, and risk.
  The `EntropySemiring` in flash-semicrf is based on this work.

**Structured prediction libraries:**

- Rush, A. M. (2020). Torch-Struct: Deep Structured Prediction Library. *ACL*.
  — The prior art for semi-CRF implementation in PyTorch. Uses pre-computed
  edge tensors with semiring abstraction. Does not scale to genomic lengths
  due to O(TKC^2) memory for edge materialization.

**Statistical foundations:**

- Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for
  Estimation, Testing, and Prediction*. Cambridge University Press.
  — The empirical Bayes framework referenced in the centering discussion.

---

**See also:** [The Basics](01-basics.md) · [Jargon Decoder](03-jargon-decoder.md) · [Streaming algorithm supplement](../manuscript/streaming_algorithm_supplement.pdf)
