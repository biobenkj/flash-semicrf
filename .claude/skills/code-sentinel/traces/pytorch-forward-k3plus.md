# Sentinel: PyTorch Reference Forward (K >= 3)

**Verified against:** `src/torch_semimarkov/streaming/pytorch_reference.py` @ commit `09e86ed`
**Linked tests:** `tests/test_streaming.py::TestStreamingForward::test_forward_produces_finite_values`

## Summary

The PyTorch reference forward computes the Semi-CRF partition function on CPU using the same O(KC) ring buffer algorithm as Triton. This is the fallback path when Triton is unavailable or when running on CPU.

## Shape Legend

- `B` = Batch size
- `T` = Sequence length (time steps)
- `K` = Maximum segment length
- `C` = Number of classes/labels

## Entry Points

| Function | File:Line | Called When |
|----------|-----------|-------------|
| `SemiCRFStreaming.forward()` | autograd.py:33 | K>=3, needs_grad, no Triton/CPU |
| `semi_crf_streaming_forward_pytorch()` | pytorch_reference.py:601 | Direct call or from autograd |

## Dispatch Conditions

```python
# autograd.py:636
can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda

# autograd.py:654-664
if needs_grad:
    if not can_use_triton:
        return SemiCRFStreaming.apply(...)  # This path
```

## Data Flow

```
Inputs:
  cum_scores: (B, T+1, C)               <- Zero-centered cumsum
  transition: (C, C) or (K, C, C)       <- Transition matrix
  duration_bias: (K, C)                 <- Duration bias
  lengths: (B,)                         <- Sequence lengths
  proj_start: (B, T, C) optional        <- Start boundaries
  proj_end: (B, T, C) optional          <- End boundaries

Allocates:
  alpha_ring: (B, K, C)                 <- Ring buffer (NOT padded to C_PAD)
  ring_checkpoints: (B, num_ckpts, K, C) <- Checkpoints for backward
  final_alpha: (B, C)                   <- Captured at sequence end

Outputs:
  partition: (B,)                       <- Log partition function
  ring_checkpoints: (B, num_ckpts, K, C)
  checkpoint_interval: int
```

## Algorithm Steps

### 1. Initialization (pytorch_reference.py:641-653)

```python
# Ring buffer - NOT padded (unlike Triton)
alpha_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)
alpha_ring[:, 0, :] = 0.0  # Initial state

# Checkpoint storage
ring_checkpoints = torch.full((batch, num_checkpoints, K, C), NEG_INF, ...)
ring_checkpoints[:, 0, :, :] = alpha_ring  # Initial checkpoint
```

### 2. Main Forward Loop (pytorch_reference.py:656-710)

```python
for t in range(1, T + 1):
    active_mask = t <= lengths
    k_eff = min(K, t)

    scores_all = []
    for k in range(1, k_eff + 1):
        start = t - k
        ring_idx = start % K
        alpha_prev = alpha_ring[:, ring_idx, :]  # (B, C_src)

        # Compute edge on-the-fly
        edge_block = compute_edge_block_streaming(
            cum_scores, transition, duration_bias, start, k, proj_start, proj_end
        )  # (B, C_dest, C_src)

        scores = alpha_prev.unsqueeze(-2) + edge_block  # (B, C_dest, C_src)
        scores_all.append(scores)

    scores_stacked = torch.stack(scores_all, dim=1)  # (B, k_eff, C_dest, C_src)

    # Reduction
    if semiring == "log":
        scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
        alpha_t = torch.logsumexp(scores_over_src, dim=1)
    else:  # max
        scores_over_src = torch.max(scores_stacked, dim=-1)[0]
        alpha_t = torch.max(scores_over_src, dim=1)[0]

    # Update ring buffer
    ring_idx_t = t % K
    alpha_ring[:, ring_idx_t, :] = torch.where(active_mask.view(B, 1), alpha_t, alpha_ring[:, ring_idx_t, :])

    # Checkpoint at intervals
    if t % checkpoint_interval == 0:
        ckpt_idx = t // checkpoint_interval
        ring_checkpoints[:, ckpt_idx, :, :] = alpha_ring

    # Capture final alpha
    is_final = (t == lengths)
    final_alpha = torch.where(is_final.view(B, 1), alpha_t, final_alpha)
```

### 3. Final Reduction (pytorch_reference.py:712-714)

```python
partition = torch.logsumexp(final_alpha, dim=-1)  # (B,)
```

## Differences from Triton

| Aspect | PyTorch | Triton |
|--------|---------|--------|
| C padding | No padding | Padded to C_PAD (power of 2) |
| Device | CPU or CUDA | CUDA only |
| Precision | Native dtype | Same, but kernel ops may differ |
| NEG_INF guards | Via torch.where | Explicit guards in kernel |

## Critical Invariants

| Invariant | Math | Python Check |
|-----------|------|--------------|
| Ring buffer aliasing | alpha[t] overwrites alpha[t-K] | `t % K` indexing |
| Variable length mask | Only update active sequences | `active_mask = t <= lengths` |
| Zero-centered warning | cum_scores endpoints < 1000 | pytorch_reference.py:622-630 |

## Known Issues

| Issue | Severity | Frequency | Resolution |
|-------|----------|-----------|------------|
| Non-zero-centered cumsum | High | When user forgets | Warning at pytorch_reference.py:622-630 |
| Slow on GPU | Medium | Always (vs Triton) | Use Triton when available |

## Version History

- **2026-01-27**: Initial trace @ commit `09e86ed`
