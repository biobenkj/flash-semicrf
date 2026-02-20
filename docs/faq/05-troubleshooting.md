# FAQ Part 5: Troubleshooting

### Q: I'm getting NaN values. What should I check?

These issues are more common when working directly with lower-level streaming APIs and manually constructed tensors. When using the high-level CRF heads (`SemiMarkovCRFHead`), many of these cases are validated or prevented earlier.

The most common causes are:

1. **Manual low-level preprocessing used float16 cumulative scores** — flash-semicrf’s high-level API builds `cum_scores` in float64, but if you construct `cum_scores` yourself, use float32 or float64 (float64 recommended)
2. **Input emissions contain NaN or Inf values** — check `torch.isfinite(hidden_states).all()`
3. **Scores are not zero-centered** before taking the cumulative sum, causing precision drift at large T

Quick diagnostic:
```python
# Check these before calling the semi-CRF
assert torch.isfinite(cum_scores).all(), "cum_scores contains NaN/Inf"
assert (cum_scores[:, 0, :] == 0).all(), "cum_scores boundary must be zero"
assert cum_scores.dtype in (torch.float32, torch.float64), "cum_scores must be float32 or float64"
```

### Q: I'm running out of GPU memory. What should I do?

First, make sure you're using the streaming backend (`backend="streaming"` or `backend="auto"`). If you're still out of memory (OOM), the bottleneck may be your encoder and not necessarily the semi-CRF layer. Further options:

- **Reduce K** — use a p95 quantile instead of p99; compute is O(TKC²)
- **Reduce batch size** — streaming memory scales linearly with batch
- **Use checkpoint semirings (exact backend)** — `CheckpointSemiring(LogSemiring)` trades ~2× compute for reduced memory in edge-tensor DP
- **Profile your encoder** — for most architectures, the encoder dominates memory, not the CRF layer

### Q: My gradients vary between runs.

This is expected when using the Triton backward kernel. The variance is caused by `atomic_add` operations combined with floating-point non-associativity (see [Engineering Decisions](04-engineering-decisions.md#q-why-is-the-backward-pass-non-deterministic)). The variance is small (~1–2% for transitions) and does not affect convergence.

If you need exact reproducibility for an ablation study, use the pure PyTorch reference backend, which is deterministic but slower:

```python
# Disable Triton to force the pure PyTorch streaming path
loss = model.compute_loss(hidden, lengths, labels, backend="streaming", use_triton=False)
```

### Q: Training loss went negative. Is something wrong?

Not necessarily. The streaming implementation zero-centers emission scores before computing cumulative sums. Because logsumexp is nonlinear, this centering shifts the partition function and gold score by different amounts, which can make the reported NLL negative on easy data. This does **not** indicate invalid probabilities — it's an artifact of the internal representation.

**How to verify everything is fine:** if accuracy metrics (frame accuracy, segment F1) look correct and are improving during training, the model is working as intended.

### Q: The first training step is very slow.

The Triton kernel is compiled (JIT) on first use, which can take several seconds. Subsequent calls reuse the cached compiled kernel and will be fast. This is a one-time cost per kernel configuration (per combination of C, K, and tile size).

### Q: How do I verify that the library is computing correct results?

Use a layered validation workflow:

1. **Ground-truth correctness harness (recommended first):**

```bash
python scripts/validate_correctness.py
```

This runs implementation-independent checks:
- self-consistency invariants on marginals
- finite-difference checks against autograd
- multi-backend training convergence agreement

2. **Low-level Triton vs PyTorch gradient parity check:**

```bash
python benchmarks/practical_demonstration/synthetic_genomics/gradient_check.py
```

This compares Triton kernel forward/gradients against the pure PyTorch streaming reference implementation.

| Gradient | Metric | Threshold |
|----------|--------|-----------|
| Forward partition | Max relative error | < 10⁻⁴ |
| `grad_transition` | Max relative error | < 10⁻² |
| `grad_duration_bias` | Max relative error | < 10⁻² |
| `grad_cum_scores` | Mean absolute error | < 10⁻³ |

`grad_cum_scores` uses mean absolute error because max relative error can be inflated at positions where the reference gradient is near-zero.

3. **Backend equivalence tests (pytest):**

```bash
pytest tests/test_backend_equivalence.py -q
pytest tests/test_nn.py -k "streaming_exact_equivalence or compute_loss_backend_routing" -q
```

These verify agreement across backend implementations at the module/test-suite level.

### Q: Which existing tests should I run for common troubleshooting scenarios?

Use targeted tests for the symptom you see:

- **NaN/Inf input and shape issues** — `pytest tests/test_validation.py tests/test_error_handling.py -q`
- **Numerical stability at long sequence lengths** — `pytest tests/test_numerical_stability_clinical.py tests/test_boundary_precision.py -q`
- **Triton vs PyTorch kernel parity** — `pytest tests/test_streaming_triton.py tests/test_triton_marginals.py -q`
- **Backend equivalence (streaming/exact/tree variants)** — `pytest tests/test_backend_equivalence.py tests/test_partition_equivalence.py -q`
- **High-level CRF head routing/equivalence checks** — `pytest tests/test_nn.py -k "streaming_exact_equivalence or compute_loss_backend_routing" -q`
- **Uncertainty and marginal behavior** — `pytest tests/test_uncertainty_quantification.py -q`

If you want one broader end-to-end check instead of targeted tests, run:

```bash
python scripts/validate_correctness.py
```

### Q: I set K too large and it's very slow. Can I recover without retraining?

Compute is O(TKC²), so halving K roughly halves runtime. If most of your segments are much shorter than K, you can retrain with a smaller K (based on data quantiles) with minimal accuracy loss. In practice, the model rarely uses the full K range because the duration bias learns to penalize very long segments.

### Q: NLL values differ between flash-semicrf and pytorch-crf. Which is wrong?

Neither. The two implementations define slightly different probability models (different handling of the first position — see [Linear CRF Equivalence](../internals/linear_crf_equivalence.md)). Both are valid. **Compare accuracy, not NLL values**, when benchmarking across implementations.

### Q: The exact edge-tensor methods fail, OOM, or give NaN. What should I do?

The exact methods require materializing the full edge tensor and can have numerical edge cases on some inputs. For production use, prefer streaming methods which are more robust:

```python
# Prefer this for long sequences / production:
entropy = model.compute_entropy_streaming(hidden, lengths)

# Use exact entropy when edge tensors fit comfortably in memory:
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

**See also:** [The Basics](01-basics.md) · [Jargon Decoder](03-jargon-decoder.md) · [Streaming internals](../internals/streaming_internals.md) · [Uncertainty guide](../guides/uncertainty_and_focused_learning.md)
