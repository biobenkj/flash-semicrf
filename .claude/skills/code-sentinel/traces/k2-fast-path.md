# Sentinel: K=2 Specialized Path

**Verified against:** `src/torch_semimarkov/streaming/pytorch_reference.py` @ commit `09e86ed`
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
| `SemiCRFK2Streaming.apply()` | autograd.py:616 | K=2, needs_grad, no boundaries |
| `semi_crf_k2_forward_pytorch()` | pytorch_reference.py:275 | Forward pass |
| `semi_crf_k2_backward_pytorch()` | pytorch_reference.py:338 | Backward pass |
| `semi_crf_k2_viterbi_pytorch()` | pytorch_reference.py:468 | Max semiring (Viterbi) |

## Dispatch Conditions

```python
# autograd.py:610-631
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
        scores_all = []

        # k=1: segment from t-1 to t
        content_k1 = cum_scores[:, t, :] - cum_scores[:, t-1, :]
        segment_k1 = content_k1 + duration_bias[0, :]
        edge_k1 = segment_k1.unsqueeze(-1) + transition.T
        scores_k1 = alpha_prev1.unsqueeze(-2) + edge_k1
        scores_all.append(scores_k1)

        # k=2: segment from t-2 to t (only if t >= 2)
        if t >= 2:
            content_k2 = cum_scores[:, t, :] - cum_scores[:, t-2, :]
            segment_k2 = content_k2 + duration_bias[1, :]
            edge_k2 = segment_k2.unsqueeze(-1) + transition.T
            scores_k2 = alpha_prev2.unsqueeze(-2) + edge_k2
            scores_all.append(scores_k2)

        # Combine durations
        scores_stacked = torch.stack(scores_all, dim=1)
        alpha_t = torch.logsumexp(torch.logsumexp(scores_stacked, dim=-1), dim=1)

        # Shift history
        alpha_prev2 = alpha_prev1
        alpha_prev1 = alpha_t

        # Capture at sequence end
        ...

    return torch.logsumexp(final_alpha, dim=-1)
```

## Memory Comparison

| Path | Memory | Why |
|------|--------|-----|
| K>=3 ring buffer | O(batch × K × C) | Ring buffer + checkpoints |
| K=2 explicit | O(batch × C × 2) | Two explicit alpha tensors |

## Critical Invariants

| Invariant | Check |
|-----------|-------|
| K must be 2 | Dispatch enforces |
| No boundaries | If boundaries, falls to K>=3 |
| duration_bias[0] for k=1, [1] for k=2 | Index = k-1 |
| alpha_prev2 only valid for t >= 2 | Check `if t >= 2` |

## Known Issues

| Issue | Severity | Resolution | Commit |
|-------|----------|------------|--------|
| Ring buffer fragility with K=2 | Critical | Dispatch to this path | `870bd1f` |
| Boundaries not supported | Low | Falls to K>=3 path | - |

## Version History

- **2026-01-27**: Initial trace @ commit `09e86ed`
