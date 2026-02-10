# Parameter guide: T, K, C

This guide expands on the three core semi-CRF parameters and how they map to
common genomics setups.

## T = Sequence length

T is the sequence length in positions. In genomics this could be:
- base pairs (1 bp = 1 position)
- tokens (for example 4/8/16 bp per token, k-mers, pooled stride)

Intuition: T is the width of context you decode in one shot.

Examples:
- Single-gene locus decoding: T might be the span of a gene plus flanks.
- Chunked genome scanning: T is your chunk size (with overlaps).
- Transcript-level decoding: T might be the span covering one transcript model.

Why it matters:
- Larger T uses more context and reduces edge effects.
- Vectorized linear scan time grows roughly linearly with T.

## K = Maximum segment duration

Semi-CRFs predict segments, not per-base labels. K is the max duration you
consider when forming a segment that ends at position t.

In genomics, segment lengths correspond to:
- Exon length
- Intron length
- UTR length
- Intergenic/background length
- TE element length (or chunks)

Intuition: K sets the longest one-piece region the decoder can model without
splitting it.

Common strategies:
- Cap and split background-like labels (long regions become multiple segments).
- Use label-specific K values (shorter for exons/UTRs, longer for introns).

In practice, pick K using a quantile (p95/p99) of observed lengths rather than
max length.

## C = Number of segment labels

C is the label set size, your segment-level annotation vocabulary.

Common choices:
- Coarse (C ~ 3): exon, intron, intergenic/background
- Gene-structure (C ~ 4-8): split exon into CDS/UTR, plus intron, intergenic
- Rich (C ~ 10-30+): strand-split labels, more biotypes, TE classes, signals

Intuition: C controls how detailed the segmentation is.

## The decoder's decision at each step

At the end of position t, the semi-CRF asks:
- Did a segment end here?
- If yes, which label c (one of C) and which duration d (1..K)?
- What label did we transition from?

So:
- T controls how many decisions you make.
- K controls how far back you can look.
- C controls how many segment types exist.

## Practical examples

Gene structure annotation:
- T = locus/chunk length you decode in one shot
- K = max exon/intron/background segment without splitting
- C = exon/intron/UTR/etc label set

TE annotation:
- T = chunk length
- K = max TE segment length you treat as one element (or chunk)
- C = TE families/superfamilies plus background

## Choosing parameters

1. Pick T based on your inference unit (gene locus or genome chunk).
2. Pick C based on your desired label granularity.
3. Pick K based on segment lengths you want to model as single segments.
4. Pick a duration distribution based on your data regime (see below).

## Choosing a duration distribution

Once you've chosen K, the next question is: **how should the model score different
durations within that range?** The `duration_distribution` parameter on
`SemiMarkovCRFHead` controls this. There are six options:

| Distribution | Parameters | Shape | When to use |
| --- | --- | --- | --- |
| `"learned"` (default) | K × C | Fully free | Plenty of training data, no strong prior on segment lengths |
| `"geometric"` | C | Exponential decay | Segments tend to be short; longer is monotonically less likely |
| `"poisson"` | C | Peaked at λ | Each class has a characteristic length (e.g. typical exon ~150 bp) |
| `"negative_binomial"` | 2 × C | Peaked, flexible variance | Like Poisson but need independent control of mean and spread |
| `"uniform"` | 0 | Flat | Duration should not influence the model (ablation baseline) |
| `CallableDuration` | user-defined | Anything | You have external histograms or domain-specific priors |

**Important observation:** `"learned"` allocates K × C free parameters — one per
duration-class pair. When K is large (e.g. K=3000 for introns) and training data
is limited, many of those parameters correspond to rare long durations that get
little gradient signal. In that regime, a parametric distribution (geometric,
Poisson, negative binomial) acts as a strong regularizer, encoding structural
assumptions about segment lengths with far fewer parameters (C or 2C).

### Decision flowchart

1. **Start with `"learned"` (the default).** It is the most flexible and works
   well when you have enough data to support K × C duration parameters.
2. **If you see overfitting or K is large relative to your dataset**, switch to a
   parametric form. Inspect the learned `duration_bias` matrix after training:
   - If it looks like exponential decay per class → try `"geometric"`
   - If each class peaks at a characteristic length → try `"poisson"`
   - If peaks vary in width across classes → try `"negative_binomial"`
3. **If you have external duration statistics** (e.g. known exon length
   distributions from annotation databases), use `CallableDuration` to inject
   them directly.
4. **For ablation experiments**, use `"uniform"` to measure how much duration
   modeling contributes to your task.

### Example

```python
from flash_semicrf import SemiMarkovCRFHead

# Default: fully learned duration bias
head = SemiMarkovCRFHead(num_classes=5, max_duration=200, hidden_dim=256)

# Geometric: good for speech/NLP where short segments dominate
head = SemiMarkovCRFHead(
    num_classes=5, max_duration=200, hidden_dim=256,
    duration_distribution="geometric",
)

# Poisson: good when each class has a typical length
head = SemiMarkovCRFHead(
    num_classes=5, max_duration=200, hidden_dim=256,
    duration_distribution="poisson",
)
```
