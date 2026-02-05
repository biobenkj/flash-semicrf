# FAQ Part 5: Troubleshooting

### Q: I'm getting NaN values. What should I check?

The most common causes are:

1. **Cumulative scores are in float16 instead of float32** — always use `.float()` before cumsum
2. **Input emissions contain NaN or Inf values** — check `torch.isfinite(hidden_states).all()`
3. **Scores are not zero-centered** before taking the cumulative sum, causing precision drift at large T

Quick diagnostic:
```python
# Check these before calling the semi-CRF
assert torch.isfinite(cum_scores).all(), "cum_scores contains NaN/Inf"
assert (cum_scores[:, 0, :] == 0).all(), "cum_scores boundary must be zero"
assert cum_scores.dtype == torch.float32, "cum_scores must be float32"
```

### Q: I'm running out of GPU memory. What should I do?

First, make sure you're using the streaming backend (`backend="streaming"` or `backend="auto"`). If you're still OOM, the bottleneck is likely your encoder, not the semi-CRF layer. Further options:

- **Reduce K** — use a p95 quantile instead of p99; compute is O(TKC²)
- **Reduce batch size** — streaming memory scales linearly with batch
- **Use checkpoint semirings** — `CheckpointSemiring(LogSemiring)` trades 2× compute for reduced memory in the DP itself
- **Profile your encoder** — for most architectures, the encoder dominates memory, not the CRF layer

### Q: My gradients vary between runs.

This is expected when using the Triton backward kernel. The variance is caused by `atomic_add` operations combined with floating-point non-associativity (see [Engineering Decisions](04-engineering-decisions.md#q-why-is-the-backward-pass-non-deterministic)). The variance is small (~1–2% for transitions) and does not affect convergence.

If you need exact reproducibility for an ablation study, use the pure PyTorch reference backend, which is deterministic but slower:

```python
from torch_semimarkov.streaming.pytorch_reference import semi_crf_streaming_forward_pytorch
```

### Q: Training loss went negative. Is something wrong?

Not necessarily. The streaming implementation zero-centers emission scores before computing cumulative sums. Because logsumexp is nonlinear, this centering shifts the partition function and gold score by different amounts, which can make the reported NLL negative on easy data. This does **not** indicate invalid probabilities — it's an artifact of the internal representation.

**How to verify everything is fine:** if accuracy metrics (frame accuracy, segment F1) look correct and are improving during training, the model is working as intended.

### Q: The first training step is very slow.

The Triton kernel is compiled (JIT) on first use, which can take several seconds. Subsequent calls reuse the cached compiled kernel and will be fast. This is a one-time cost per kernel configuration (per combination of C, K, and tile size).

### Q: How do I verify that the library is computing correct results?

Run the gradient check script:

```bash
python benchmarks/practical_demonstration/synthetic_genomics/gradient_check.py
```

This compares the Triton kernel's forward partition and all gradients against the pure PyTorch reference implementation.

| Gradient | Metric | Threshold |
|----------|--------|-----------|
| Forward partition | Max relative error | < 10⁻⁴ |
| `grad_transition` | Max relative error | < 10⁻² |
| `grad_duration_bias` | Max relative error | < 10⁻² |
| `grad_cum_scores` | Mean absolute error | < 10⁻³ |

`grad_cum_scores` uses mean absolute error because max relative error can be inflated at positions where the reference gradient is near-zero.

### Q: I set K too large and it's very slow. Can I recover without retraining?

Compute is O(TKC²), so halving K roughly halves runtime. If most of your segments are much shorter than K, you can retrain with a smaller K (based on data quantiles) with minimal accuracy loss. In practice, the model rarely uses the full K range because the duration bias learns to penalize very long segments.

### Q: NLL values differ between torch-semimarkov and pytorch-crf. Which is wrong?

Neither. The two implementations define slightly different probability models (different handling of the first position — see [Linear CRF Equivalence](../linear_crf_equivalence.md)). Both are valid. **Compare accuracy, not NLL values**, when benchmarking across implementations.

### Q: The `backend="exact"` mode fails or gives NaN on short sequences.

The exact methods require materializing the full edge tensor and can have numerical edge cases on some inputs. For production use, prefer streaming methods which are more robust:

```python
# Prefer this (always works):
entropy = model.compute_entropy_streaming(hidden, lengths)

# Over this (may fail for some inputs):
entropy = model.compute_entropy_exact(hidden, lengths)
```

### Q: Marginals don't sum to 1 over classes.

**Clarification:** boundary marginals represent P(boundary at position t), not P(label at position t). These are different quantities. For per-position label distributions that sum to 1 over the class dimension, use:

```python
position_marginals = model.compute_position_marginals(hidden, lengths)
# This sums to 1 over the class dimension
```

---

> **Still stuck?** Open an issue on the GitHub repository with: (1) your T, K, C values, (2) the error message or unexpected behavior, (3) whether you're using streaming or exact backend, and (4) your GPU model and available memory.

---

**See also:** [The Basics](01-basics.md) · [Jargon Decoder](03-jargon-decoder.md) · [Streaming internals](../streaming_internals.md) · [Uncertainty guide](../uncertainty_and_focused_learning.md)
