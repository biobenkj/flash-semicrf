# Sentinel: K=1 Linear CRF Fast Path

**Verified against:** `src/torch_semimarkov/streaming/pytorch_reference.py` @ commit `f865fed`
**Linked tests:** `tests/test_streaming.py::TestStreamingK1::test_streaming_k1_gradient_flow`

## Summary

The K=1 fast path handles linear CRF (segment length = 1 only). No ring buffer needed - O(batch×C) memory. This is a specialized optimization for the common case where segments have unit length.

## Shape Legend

- `B` = Batch size
- `T` = Sequence length
- `C` = Number of classes/labels

## Entry Points

| Function | File:Line | Called When |
|----------|-----------|-------------|
| `LinearCRFStreaming.apply()` | autograd.py:600 | K=1, needs_grad, no boundaries |
| `linear_crf_forward_pytorch()` | pytorch_reference.py:26 | Forward pass |
| `linear_crf_backward_pytorch()` | pytorch_reference.py:86 | Backward pass |
| `linear_crf_viterbi_pytorch()` | pytorch_reference.py:195 | Max semiring (Viterbi) |

## Dispatch Conditions

```python
# autograd.py:594-616
if K == 1:
    if proj_start is not None or proj_end is not None:
        # Fall through to K>=3 path (boundaries not supported)
        pass
    elif needs_grad:
        return LinearCRFStreaming.apply(cum_scores, transition, duration_bias, lengths, semiring)
    else:
        # Inference
        if semiring == "max":
            scores, _ = linear_crf_viterbi_pytorch(...)
            return scores
        else:
            return linear_crf_forward_pytorch(...)
```

## Why K=1 is Special

1. **No ring buffer aliasing**: K=1 means all `t % 1 = 0`, causing ring buffer corruption
2. **Simpler recurrence**: Only one previous state matters (alpha[t-1])
3. **No duration loop**: No need to iterate over k = 1..K

## Algorithm (Forward)

```python
def linear_crf_forward_pytorch(cum_scores, transition, lengths, duration_bias):
    # pytorch_reference.py:26-83

    # alpha: (B, C) - only current timestep needed
    alpha = torch.zeros(batch, C)

    for t in range(1, T + 1):
        # Content score from cumsum difference
        emission = cum_scores[:, t, :] - cum_scores[:, t-1, :] + dur_bias  # (B, C)

        # Linear CRF recurrence:
        # alpha[t, c_dst] = logsumexp_{c_src}(alpha[t-1, c_src] + trans[c_src, c_dst]) + emission[c_dst]
        alpha_new = torch.logsumexp(alpha.unsqueeze(-1) + transition, dim=-2) + emission

        # Update alpha only for active sequences
        active_mask = (t <= lengths).view(batch, 1)
        alpha = torch.where(active_mask, alpha_new, alpha)

        # Capture final alpha at sequence endpoints
        final_mask = (t == lengths).view(batch, 1)
        final_alpha = torch.where(final_mask, alpha_new, final_alpha)

    return torch.logsumexp(final_alpha, dim=-1)  # (B,)
```

## Algorithm (Backward)

```python
def linear_crf_backward_pytorch(cum_scores, transition, lengths, log_Z, duration_bias):
    # pytorch_reference.py:86-192

    # Forward pass: store all alpha values
    alpha_all = torch.full((batch, T + 1, C), NEG_INF)
    alpha_all[:, 0, :] = 0.0
    for t in range(1, T + 1):
        # ... standard forward recurrence

    # Backward pass: compute all beta values
    beta_all = torch.full((batch, T + 1, C), NEG_INF)
    # Initialize beta at final positions
    for b in range(batch):
        beta_all[b, lengths[b].item(), :] = 0.0

    for t in range(T - 1, -1, -1):
        # beta[t, c_src] = logsumexp_{c_dst}(trans[c_src, c_dst] + emission[c_dst] + beta[t+1, c_dst])
        beta_new = torch.logsumexp(
            transition.unsqueeze(0) + (emission_next + beta_all[:, t + 1, :]).unsqueeze(-2),
            dim=-1,
        )

    # Compute gradients via marginals
    # Edge marginal: P(c_src -> c_dst at time t)
    log_marginal = alpha[t-1, c_src] + trans[c_src, c_dst] + emission[c_dst] + beta[t, c_dst] - log_Z
    marginal = exp(clamp(log_marginal, -80, 80))

    # Accumulate gradients (per-batch for reduction in autograd)
    grad_cum_scores[:, t, :] += marginal.sum(dim=-2)
    grad_cum_scores[:, t - 1, :] -= marginal.sum(dim=-2)
    grad_transition += marginal
    grad_duration_bias += marginal.sum(dim=-2)
```

## Autograd Gradient Reduction

The backward function returns per-batch gradients. Autograd reduces them:

```python
# autograd.py:380-384
grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
if grad_duration_bias is not None:
    grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)
```

## Memory Comparison

| Path | Memory | Why |
|------|--------|-----|
| K>=3 ring buffer | O(batch × K × C) | Ring buffer + checkpoints |
| K=1 fast path | O(batch × C) | Only current alpha |

## Critical Invariants

| Invariant | Check |
|-----------|-------|
| K must be 1 | Dispatch enforces |
| No boundaries | If boundaries, falls to K>=3 |
| duration_bias[0] used | Index 0 for k=1 |
| Per-batch gradients | transition/duration_bias returned as (B, ...) |

## Known Issues

| Issue | Severity | Resolution | Commit |
|-------|----------|------------|--------|
| Ring buffer aliasing with K=1 | Critical | Dispatch to this path | `870bd1f` |
| Boundaries not supported | Low | Falls to K>=3 path | - |

## Version History

- **2026-02-02**: Updated line numbers (Viterbi at 195, dispatch at 594-616); documented per-batch gradient convention
- **2026-01-27**: Initial trace @ commit `09e86ed`
