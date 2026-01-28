# Sentinel: PyTorch Reference Backward (K >= 3)

**Verified against:** `src/torch_semimarkov/streaming/pytorch_reference.py` @ commit `40fe66b`
**Linked tests:** `tests/test_streaming.py::TestStreamingBackward::test_backward_produces_finite_gradients`

## Summary

The PyTorch reference backward computes gradients using the forward-backward algorithm with checkpointing. Same algorithm as Triton backward but implemented in pure PyTorch for CPU fallback and reference testing.

## Shape Legend

- `B` = Batch size
- `T` = Sequence length
- `K` = Maximum segment length
- `C` = Number of classes/labels

## Entry Points

| Function | File:Line | Called When |
|----------|-----------|-------------|
| `SemiCRFStreaming.backward()` | autograd.py:74 | Backward through SemiCRFStreaming |
| `semi_crf_streaming_backward_pytorch()` | pytorch_reference.py:844 | Called from autograd |

## Data Flow

```
Inputs (from ctx.saved_tensors):
  cum_scores: (B, T+1, C)
  transition: (C, C) or (K, C, C)
  duration_bias: (K, C)
  lengths: (B,)
  ring_checkpoints: (B, num_ckpts, K, C)
  partition: (B,)

Outputs:
  grad_cum_scores: (B, T+1, C)    <- Per-batch
  grad_transition: (B, C, C)       <- Per-batch (reduced in autograd)
  grad_duration_bias: (B, K, C)    <- Per-batch (reduced in autograd)
  grad_proj_start: (B, T, C) or None
  grad_proj_end: (B, T, C) or None
```

## Algorithm Overview

Same as Triton backward: process checkpoint segments in reverse order.

### Phase 1: Alpha Recomputation

```python
# For segment starting at checkpoint ckpt_idx
alpha_segment = recompute_alpha_segment(
    ring_checkpoints[:, ckpt_idx, :, :],  # Starting ring state
    cum_scores, transition, duration_bias,
    seg_start, seg_end, K
)  # Returns alpha values for segment
```

### Phase 2: Beta + Gradient Accumulation

```python
for t in range(seg_end-1, seg_start-1, -1):
    # Compute beta[t] from beta[t+1:t+K+1]
    beta_t = logsumexp over k of (beta[t+k] + edge[t -> t+k])

    # Accumulate gradients via marginals
    for k in range(1, K+1):
        log_marginal = alpha[t] + edge + beta[t+k] - partition
        marginal = exp(clamp(log_marginal, -80, 80))

        grad_cum_scores[t:t+k] += marginal
        grad_transition += marginal
        grad_duration_bias[k] += marginal
```

## Gradient Reduction in Autograd

The backward function returns per-batch gradients. Autograd reduces them:

```python
# autograd.py:138-143
if grad_transition.ndim == 3:  # (batch, C, C)
    grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
else:  # (batch, K, C, C)
    grad_transition = torch.einsum("bkij, b -> kij", grad_transition, grad_output)

grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)
```

## Critical Invariants

| Invariant | Check |
|-----------|-------|
| Partition finite | `torch.isfinite(partition).all()` (autograd.py:88) |
| Gradients finite | `torch.isfinite(grad_cum_scores).all()` (autograd.py:115) |
| Log marginal clamped | `clamp(log_marginal, min=-80, max=80)` |

## Known Issues

| Issue | Severity | Resolution |
|-------|----------|------------|
| Slow on GPU | Medium | Use Triton backward |
| Memory for large T | Medium | Checkpointing reduces peak |

## Version History

- **2026-01-27**: Initial trace @ commit `40fe66b`
