# Sentinel: K=2 Specialized Path

**Verified against:** `src/flash_semicrf/streaming/pytorch_reference.py` @ commit `52c3f9e`
**Linked tests:** `tests/test_streaming_triton.py::TestKSpecificPaths::test_k2_forward_correctness`

## Summary

The K=2 fast path handles the case where segments can be length 1 or 2. Uses explicit 2-step history instead of a ring buffer, avoiding the fragile alternating-slot pattern that would occur with K=2 ring indexing.

## Shape Legend

- `B` = Batch size
- `T` = Sequence length
- `C` = Number of classes/labels

## Entry Points

| Function | File:Line | Called When |
|----------|-----------|-------------|
| `SemiCRFK2Streaming.apply()` | autograd.py:626 | K=2, needs_grad, no boundaries |
| `semi_crf_k2_forward_pytorch()` | pytorch_reference.py:275 | Forward pass |
| `semi_crf_k2_backward_pytorch()` | pytorch_reference.py:338 | Backward pass |
| `semi_crf_k2_viterbi_pytorch()` | pytorch_reference.py:468 | Max semiring (Viterbi) |

## Dispatch Conditions

```python
# autograd.py:620-642
if K == 2:
    if proj_start is not None or proj_end is not None:
        # Fall through to K>=3 path (boundaries not supported)
        pass
    elif needs_grad:
        return SemiCRFK2Streaming.apply(cum_scores, transition, duration_bias, lengths, semiring)
    else:
        # Inference
        if semiring == "max":
            scores, _, _ = semi_crf_k2_viterbi_pytorch(...)
            return scores
        else:
            return semi_crf_k2_forward_pytorch(...)
```

## Why K=2 is Special

1. **Ring buffer fragility**: K=2 means `t % 2` alternates 0,1,0,1... Any off-by-one causes corruption
2. **Explicit history**: Store alpha[t-1] and alpha[t-2] explicitly, no modular arithmetic
3. **Cleaner code**: Two explicit variables instead of ring buffer indexing

## Algorithm (Forward)

```python
def semi_crf_k2_forward_pytorch(cum_scores, transition, duration_bias, lengths):
    # pytorch_reference.py:275-335

    # Explicit 2-step history
    alpha_prev1 = torch.zeros(batch, C)   # alpha[t-1]
    alpha_prev2 = torch.full(..., NEG_INF)  # alpha[t-2] (invalid for t<2)

    for t in range(1, T + 1):
        # k=1: segment from t-1 to t
        emission_k1 = cum_scores[:, t, :] - cum_scores[:, t-1, :] + duration_bias[0]
        score_k1 = torch.logsumexp(alpha_prev1.unsqueeze(-1) + transition, dim=-2) + emission_k1

        # k=2: segment from t-2 to t (only if t >= 2)
        if t >= 2:
            emission_k2 = cum_scores[:, t, :] - cum_scores[:, t-2, :] + duration_bias[1]
            score_k2 = torch.logsumexp(alpha_prev2.unsqueeze(-1) + transition, dim=-2) + emission_k2
            # Combine scores from both durations
            alpha_new = torch.logsumexp(torch.stack([score_k1, score_k2], dim=-1), dim=-1)
        else:
            alpha_new = score_k1

        # Shift history
        alpha_prev2 = alpha_prev1.clone()
        alpha_prev1 = torch.where(active_mask, alpha_new, alpha_prev1)

        # Capture at sequence end
        final_alpha = torch.where(final_mask, alpha_new, final_alpha)

    return torch.logsumexp(final_alpha, dim=-1)
```

## Algorithm (Backward)

```python
def semi_crf_k2_backward_pytorch(cum_scores, transition, duration_bias, lengths, log_Z):
    # pytorch_reference.py:338-465

    # Forward pass: store all alpha values
    alpha_all = torch.full((batch, T + 1, C), NEG_INF)
    alpha_all[:, 0, :] = 0.0
    for t in range(1, T + 1):
        # ... k=1 and k=2 forward recurrence

    # Backward pass: compute all beta values
    beta_all = torch.full((batch, T + 1, C), NEG_INF)
    for b in range(batch):
        beta_all[b, lengths[b].item(), :] = 0.0

    for t in range(T - 1, -1, -1):
        beta_new = NEG_INF
        # Contribution from k=1 segments ending at t+1
        if t + 1 <= T:
            contrib_k1 = logsumexp(trans + emission_k1 + beta[t+1])
            beta_new = logsumexp([beta_new, contrib_k1])
        # Contribution from k=2 segments ending at t+2
        if t + 2 <= T:
            contrib_k2 = logsumexp(trans + emission_k2 + beta[t+2])
            beta_new = logsumexp([beta_new, contrib_k2])

    # Compute gradients via marginals (per-batch for reduction in autograd)
    for t in range(1, T + 1):
        # k=1 edges
        log_marginal_k1 = alpha[t-1] + trans + emission_k1 + beta[t] - log_Z
        marginal_k1 = exp(clamp(log_marginal_k1, -80, 80))
        grad_cum_scores[:, t, :] += marginal_k1.sum(dim=-2)
        grad_cum_scores[:, t - 1, :] -= marginal_k1.sum(dim=-2)
        grad_transition += marginal_k1
        grad_duration_bias[:, 0, :] += marginal_k1.sum(dim=-2)

        # k=2 edges (if t >= 2)
        if t >= 2:
            log_marginal_k2 = alpha[t-2] + trans + emission_k2 + beta[t] - log_Z
            marginal_k2 = exp(clamp(log_marginal_k2, -80, 80))
            grad_cum_scores[:, t, :] += marginal_k2.sum(dim=-2)
            grad_cum_scores[:, t - 2, :] -= marginal_k2.sum(dim=-2)
            grad_transition += marginal_k2
            grad_duration_bias[:, 1, :] += marginal_k2.sum(dim=-2)
```

## Autograd Gradient Reduction

The backward function returns per-batch gradients. Autograd reduces them:

```python
# autograd.py:472-473
grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)
```

## Memory Comparison

| Path | Memory | Why |
|------|--------|-----|
| K>=3 ring buffer | O(batch * K * C) | Ring buffer + checkpoints |
| K=2 explicit | O(batch * C * 2) | Two explicit alpha tensors |

## Critical Invariants

| Invariant | Check |
|-----------|-------|
| K must be 2 | Dispatch enforces |
| No boundaries | If boundaries, falls to K>=3 |
| duration_bias[0] for k=1, [1] for k=2 | Index = k-1 |
| alpha_prev2 only valid for t >= 2 | Check `if t >= 2` |
| Per-batch gradients | transition/duration_bias returned as (B, ...) |

## Known Issues

| Issue | Severity | Resolution | Commit |
|-------|----------|------------|--------|
| Ring buffer fragility with K=2 | Critical | Dispatch to this path | `870bd1f` |
| Boundaries not supported | Low | Falls to K>=3 path | - |

## Version History

- **2026-02-02**: Updated line numbers (dispatch at 620-642, apply at 626); documented per-batch gradient convention
- **2026-01-27**: Initial trace @ commit `09e86ed`
