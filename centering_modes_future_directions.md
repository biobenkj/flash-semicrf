# Emission Centering Modes: Design Notes and Future Directions

## Current Implementation

All code paths in `torch-semimarkov` apply a single centering strategy before constructing prefix sums. In `nn.py` (lines 199, 342) and `uncertainty.py` (lines 186, 317):

```python
scores_float = scores_float - scores_float.mean(dim=1, keepdim=True)
```

This subtracts the per-label mean over the full padded sequence dimension. The centered emissions then enter `torch.cumsum` to produce `cum_scores`, which are consumed by all downstream backends (Triton streaming, exact via `semimarkov.py`, binary tree sharded). No backend performs its own centering — `cum_scores` arrives pre-centered.

### What centering does

The centering subtracts a per-label, per-sequence constant $\nu_{b,c}$ from each emission before building the prefix sums. For a segment $[s, s+k)$ with label $c$, the effective score becomes:

$$\text{content}(c, s, k) = \sum_{u=s}^{s+k-1} f_\theta(u, c) - \nu_{b,c} \cdot k$$

This has two effects:

1. **Numerical**: Cumulative sums grow as $O(\sqrt{T})$ instead of $O(T)$, preventing catastrophic cancellation in float32 at genomic scale.
2. **Modeling**: The $-\nu_{b,c} \cdot k$ term acts as a data-dependent duration prior that penalizes long segments of globally prevalent labels. Equivalent to an effective duration model $\mathcal{B}^{\text{eff}}_{k,c} = \mathcal{B}_{k,c} - \nu_{b,c} \cdot k$.

---

## Proposed: Configurable Centering Modes

A `centering` parameter on `SemiMarkovCRFHead` would let users select the centering strategy. The implementation cost is low — it's a single preprocessing step that modifies `scores_float` before `torch.cumsum`, identical across all backends.

### Mode 1: `"mean"` (current default)

```python
# Current behavior
shift = scores_float.mean(dim=1, keepdim=True)  # (B, 1, C)
scores_float = scores_float - shift
```

Subtracts the per-label mean over all $T$ positions (including padding). Introduces the adaptive duration prior described above.

**When to use**: Default for encoder → CRF decoder architectures, especially genomic segmentation where label imbalance is severe and the duration prior provides useful regularization. Best when all sequences in a batch have equal length (padding fraction = 0).

### Mode 2: `"masked_mean"`

```python
# Mean over valid tokens only
mask = torch.arange(T, device=scores_float.device).unsqueeze(0) < lengths.unsqueeze(1)
mask = mask.unsqueeze(-1).float()  # (B, T, 1)
shift = (scores_float * mask).sum(dim=1, keepdim=True) / lengths.unsqueeze(-1).unsqueeze(-1)
scores_float = scores_float - shift
```

Subtracts the per-label mean computed only over valid (non-padding) positions. Same adaptive duration prior as `"mean"`, but the baseline accurately reflects the actual sequence content rather than being diluted by padding emissions.

**When to use**: Variable-length batches where the padding fraction varies significantly across batch elements. Requires `lengths` to be available at centering time (it already is — passed through `forward()`).

### Mode 3: `"position"` (path-invariant)

```python
# Per-position, label-agnostic shift
shift = scores_float.max(dim=-1, keepdim=True).values  # (B, T, 1)
scores_float = scores_float - shift
```

Subtracts a per-position scalar shared across all labels. This is path-invariant: every valid segmentation covers each position exactly once, so the total shift cancels in the partition function. The resulting model is identical to an uncentered semi-CRF — no implicit duration prior.

**When to use**: When CRF marginals serve as input features for a downstream model (e.g., Mamba state-space model consuming posterior label probabilities). The marginals reflect the canonical semi-CRF distribution without data-dependent bias. Also appropriate when you want the duration model to be entirely controlled by the explicit `duration_bias` parameter, with no interference from centering.

**Important caveat**: While path-invariant centering preserves the model distribution exactly, it provides weaker numerical stabilization. The per-position max removes cross-label scale differences but does not center the temporal drift. Cumulative sums can still grow as $O(T)$ if emissions have a nonzero temporal mean. At $T > 100\text{K}$, this may require combining with checkpoint normalization (already present in the Triton kernel) for adequate stability.

### Mode 4: `"reconstruct"` (semantics-preserving with full stability)

```python
# Center for prefix-sum stability, reconstruct at score time
shift = scores_float.mean(dim=1, keepdim=True)  # (B, 1, C) — stored for reconstruction
scores_float = scores_float - shift
cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)
# At segment scoring: (cum_scores[t+k, c] - cum_scores[t, c]) + shift[b, 0, c] * k
```

Builds prefix sums on centered residuals (for $O(\sqrt{T})$ cumsum magnitude) but adds back $\nu_{b,c} \cdot k$ when computing each segment's content score. The resulting model is mathematically identical to an uncentered semi-CRF — no implicit duration prior — while retaining the full numerical benefit of centering.

**When to use**: When you want the clean numerical properties of centering with no modeling side effects. The correction term $\nu_{b,c} \cdot k$ requires passing the stored baseline through to the kernel, which adds a small amount of complexity.

**Implementation note**: This mode requires modifying the segment score computation inside the Triton kernel (or the edge tensor construction for exact backends). The kernel would need to accept a `baseline` tensor of shape $(B, C)$ and add `baseline[b, c] * k` to each segment score. The streaming PyTorch backend and exact backend can implement this straightforwardly. For the Triton kernel, the correction is a single multiply-add per $(t, k, c)$ triple — negligible cost. The baseline tensor would be passed alongside `cum_scores`, `transition`, and `duration_bias` as an additional kernel argument.

### Mode 5: `"none"`

```python
# No centering
# scores_float is used as-is
```

Raw emissions enter the cumsum directly. Only safe for short sequences ($T < 1000$) or when emissions are known to be small-magnitude.

**When to use**: Debugging, unit testing, or when comparing against reference implementations that don't center. Also appropriate when the encoder explicitly produces zero-mean emissions (e.g., via LayerNorm as the final layer).

---

## Proposed API

```python
crf = SemiMarkovCRFHead(
    num_classes=24,
    max_duration=100,
    hidden_dim=512,
    centering="mean",  # "mean", "masked_mean", "position", "reconstruct", "none"
)
```

### Implementation Skeleton

The change is localized to the cumulative score construction in `nn.py`. All centering modes produce a `cum_scores` tensor of the same shape — downstream backends are completely unaware of which mode was used (with the exception of `"reconstruct"`, which requires a kernel-level change).

```python
def _build_cum_scores(self, scores: Tensor, lengths: Tensor) -> Tensor:
    """Build cumulative scores with configurable centering."""
    batch, T, C = scores.shape
    scores_float = scores.float()

    if T > 1:
        if self.centering == "mean":
            scores_float = scores_float - scores_float.mean(dim=1, keepdim=True)

        elif self.centering == "masked_mean":
            mask = torch.arange(T, device=scores.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            shift = (scores_float * mask).sum(dim=1, keepdim=True)
            shift = shift / lengths.float().unsqueeze(-1).unsqueeze(-1)
            scores_float = scores_float - shift

        elif self.centering == "position":
            shift = scores_float.max(dim=-1, keepdim=True).values
            scores_float = scores_float - shift

        elif self.centering == "reconstruct":
            self._baseline = scores_float.mean(dim=1, keepdim=True)  # save for kernel
            scores_float = scores_float - self._baseline

        elif self.centering == "none":
            pass

    cum_scores = torch.zeros(batch, T + 1, C, dtype=torch.float32, device=scores.device)
    cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)
    return cum_scores
```

The same function replaces the inline centering in `_build_edge_tensor`, `forward`, `viterbi`, and `uncertainty.py`.

---

## Interaction with Semirings

The Triton streaming kernel currently supports two semirings: **log** (partition function, marginals) and **max** (Viterbi decoding). The exact backend via `semimarkov.py` supports arbitrary semirings through the `torch-struct` framework (log, max, sampling, entropy, etc.) but requires materializing the full $(B, T, K, C, C)$ edge tensor — $O(TKC^2)$ memory.

The centering mode and the semiring are orthogonal choices, but they interact in one important way:

**Log semiring**: Centering affects the partition function and marginals. The `"mean"` and `"masked_mean"` modes change the model (adaptive duration prior). The `"position"` and `"reconstruct"` modes preserve the model exactly.

**Max semiring (Viterbi)**: Centering affects which path is selected as the MAP estimate. Per-label centering (`"mean"`, `"masked_mean"`) biases Viterbi toward shorter segments for prevalent labels, which is the same regularization effect as in the log semiring. Path-invariant centering (`"position"`) does not affect the Viterbi path (the argmax is invariant to per-position shifts that are shared across labels). The `"reconstruct"` mode also preserves the Viterbi path.

**Sampling and entropy semirings** (exact backend only): These semirings compute expectations under the model distribution. Per-label centering changes the distribution and therefore changes the samples and entropy values. If exact distributional properties matter, `"reconstruct"` or `"position"` should be used.

---

## Interaction with Duration Distributions

The `duration_distribution` parameter (`"learned"`, `"geometric"`, `"poisson"`, `"negbin"`, `"uniform"`) controls the explicit duration bias $\mathcal{B}_{k,c}$. Centering adds an implicit component $-\nu_{b,c} \cdot k$. The two interact:

- **Learned duration** can absorb the average effect of centering during training, since $\mathcal{B}_{k,c}$ has $K \times C$ free parameters. The centering then primarily regularizes per-sample variation.
- **Parametric durations** (geometric, Poisson, negative binomial) have far fewer parameters. The centering adds a linear-in-$k$ component that the parametric form may not be able to represent. For geometric distributions (which are already exponential in $k$), the linear centering term dominates at large $k$, effectively imposing a hard penalty on long segments regardless of the geometric rate parameter.
- **Uniform duration** ($\mathcal{B}_{k,c} = 0$) makes the centering effect most visible: the effective duration model is purely $-\nu_{b,c} \cdot k$, a data-dependent linear decay.

Users combining parametric duration distributions with per-label centering should be aware of this interaction. The `"reconstruct"` or `"position"` modes eliminate it entirely.

---

## Recommended Defaults by Application

| Application | Centering | Duration | Rationale |
|---|---|---|---|
| Genomic segmentation (fixed-length windows) | `"mean"` | `"learned"` | Adaptive prior helps with label imbalance; equal lengths make unmasked mean exact |
| Genomic segmentation (variable-length) | `"masked_mean"` | `"learned"` | Same benefits, baseline reflects actual sequence content |
| CRF marginals as downstream features | `"position"` or `"reconstruct"` | any | Preserves canonical distribution for downstream model |
| Benchmarking against reference implementations | `"none"` or `"reconstruct"` | `"learned"` | No implicit model changes for fair comparison |
| Short sequences ($T < 1000$) | `"none"` | any | Centering unnecessary at this scale |
| Parametric duration + long sequences | `"reconstruct"` | `"geometric"` / `"poisson"` | Avoids interaction between centering and parametric form |

---

## Implementation Priority

1. **`"masked_mean"`** — lowest-effort, highest-value change. One-line modification to replace `.mean()` with a masked mean. No kernel changes needed.
2. **`"position"`** — also a one-line change. Useful for the CRF-as-features use case.
3. **`"reconstruct"`** — requires passing the baseline tensor to the kernel and adding a multiply-add to the segment score computation. Moderate effort but provides the cleanest semantics.
4. **`"none"`** — trivial (skip the subtraction). Useful for testing.
