# Sentinel: K=1 Linear CRF Fast Path

**Verified against:** `src/torch_semimarkov/streaming/pytorch_reference.py` @ commit `40fe66b`
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
| `LinearCRFStreaming.apply()` | autograd.py:590 | K=1, needs_grad, no boundaries |
| `linear_crf_forward_pytorch()` | pytorch_reference.py:26 | Forward pass |
| `linear_crf_backward_pytorch()` | pytorch_reference.py:86 | Backward pass |
| `linear_crf_viterbi_pytorch()` | pytorch_reference.py:166 | Max semiring (Viterbi) |

## Dispatch Conditions

```python
# autograd.py:584-605
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
        content = cum_scores[:, t, :] - cum_scores[:, t-1, :]  # (B, C)
        segment_score = content + duration_bias[0, :]  # k=1 uses index 0

        # edge[c_dst, c_src] = segment_score[c_dst] + transition[c_src, c_dst]
        edge = segment_score.unsqueeze(-1) + transition.T  # (B, C_dst, C_src)

        # scores = alpha[c_src] + edge[c_dst, c_src]
        scores = alpha.unsqueeze(-2) + edge  # (B, C_dst, C_src)

        # logsumexp over c_src
        alpha = torch.logsumexp(scores, dim=-1)  # (B, C)

        # Capture at sequence end
        if t == lengths[b]:
            final_alpha[b] = alpha[b]

    return torch.logsumexp(final_alpha, dim=-1)  # (B,)
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

## Known Issues

| Issue | Severity | Resolution | Commit |
|-------|----------|------------|--------|
| Ring buffer aliasing with K=1 | Critical | Dispatch to this path | `870bd1f` |
| Boundaries not supported | Low | Falls to K>=3 path | - |

## Version History

- **2026-01-27**: Initial trace @ commit `40fe66b`
