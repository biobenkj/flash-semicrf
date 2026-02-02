# Sentinel: PyTorch Reference Forward (K >= 3)

**Verified against:** `src/torch_semimarkov/streaming/pytorch_reference.py` @ commit `f865fed`
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
# autograd.py:646
can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda

# autograd.py:664-674
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
  ring_checkpoints: (B, num_ckpts, K, C) <- Checkpoints for backward (NORMALIZED)
  log_norm_checkpoints: (B, num_ckpts)   <- Cumulative normalization factors
  final_alpha: (B, C)                   <- Captured at sequence end
  accum_log_norm: (B,)                  <- Running normalization accumulator

Outputs:
  partition: (B,)                       <- Log partition function
  ring_checkpoints: (B, num_ckpts, K, C) <- Normalized ring buffer states
  checkpoint_interval: int
  log_norm_checkpoints: (B, num_ckpts)   <- For backward pass numerical stability
```

## Algorithm Steps

### 1. Initialization (pytorch_reference.py:662-670)

```python
# Ring buffer - NOT padded (unlike Triton)
alpha_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)
alpha_ring[:, 0, :] = 0.0  # Initial state

# Checkpoint storage
ring_checkpoints = torch.full((batch, num_checkpoints, K, C), NEG_INF, ...)
ring_checkpoints[:, 0, :, :] = alpha_ring  # Initial checkpoint

# Log normalization tracking for T>100K stability
log_norm_checkpoints = torch.zeros((batch, num_checkpoints), ...)
accum_log_norm = torch.zeros(batch, ...)
```

### 2. Main Forward Loop (pytorch_reference.py:673-780)

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

    # === CHECKPOINT + NORMALIZATION (Flash Attention pattern) ===
    if t % checkpoint_interval == 0:
        ckpt_idx = t // checkpoint_interval
        # 1. Find max alpha for normalization shift
        shift = alpha_t.max(dim=-1, keepdim=True)[0]
        # 2. Accumulate normalization factor
        accum_log_norm += shift.squeeze(-1)
        # 3. Normalize alpha_t
        alpha_t = alpha_t - shift
        # 4. Normalize ALL K slots in ring buffer
        for k_slot in range(K):
            alpha_ring[:, k_slot, :] -= shift
        # 5. Save normalized checkpoint
        ring_checkpoints[:, ckpt_idx, :, :] = alpha_ring
        # 6. Save cumulative normalization
        log_norm_checkpoints[:, ckpt_idx] = accum_log_norm

    # Capture final alpha
    is_final = (t == lengths)
    final_alpha = torch.where(is_final.view(B, 1), alpha_t, final_alpha)
```

### 3. Final Reduction (pytorch_reference.py:782-791)

```python
# Add back cumulative normalization
raw_partition = torch.logsumexp(final_alpha, dim=-1)
partition = raw_partition + accum_log_norm
```

## Flash Attention-Style Normalization

At checkpoint boundaries, alpha values are normalized to prevent unbounded growth at extreme T (100K+):

1. **Shift extraction**: `shift = max(alpha_t)` per batch element
2. **Accumulation**: `accum_log_norm += shift` tracks cumulative scale
3. **Normalization**: All ring buffer slots shifted by `-shift`
4. **Checkpoint**: Normalized state saved for backward
5. **Final correction**: Partition = logsumexp(final_alpha) + accum_log_norm

This prevents the ~2.5×T log-sum growth that causes overflow at T>100K.

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
| Zero-centered warning | cum_scores endpoints < 1000 | pytorch_reference.py:630-638 |
| Normalization consistency | All K slots shifted together | Loop over k_slot in range(K) |

## Known Issues

| Issue | Severity | Frequency | Resolution |
|-------|----------|-----------|------------|
| Non-zero-centered cumsum | High | When user forgets | Warning at pytorch_reference.py:630-638 |
| Slow on GPU | Medium | Always (vs Triton) | Use Triton when available |

## Version History

- **2026-02-02**: Updated line numbers; documented Flash Attention-style normalization; log_norm_checkpoints now returned as 4th value
- **2026-01-27**: Initial trace @ commit `09e86ed`
