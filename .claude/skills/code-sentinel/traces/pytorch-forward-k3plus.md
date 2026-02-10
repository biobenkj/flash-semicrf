# Sentinel: PyTorch Reference Forward (K >= 3)

**Verified against:** `src/flash_semicrf/streaming/pytorch_reference.py` @ commit `52c3f9e`
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

### 1. Initialization (pytorch_reference.py:660-668)

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

### 2. Main Forward Loop (pytorch_reference.py:671-774)

```python
for t in range(1, T + 1):
    # ACTIVE MASKING: Include t == lengths to compute alpha at final position
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

    # Update ring buffer (ONLY FOR ACTIVE SEQUENCES)
    ring_idx_t = t % K
    alpha_ring[:, ring_idx_t, :] = torch.where(
        active_mask.view(batch, 1), alpha_t, alpha_ring[:, ring_idx_t, :]
    )

    # === CHECKPOINT + NORMALIZATION (Flash Attention pattern) ===
    if t % checkpoint_interval == 0:
        ckpt_idx = t // checkpoint_interval
        if ckpt_idx < num_checkpoints:
            # 1. Find max alpha for normalization (MASKED FOR ACTIVE ONLY)
            alpha_for_norm = torch.where(
                active_mask.view(batch, 1), alpha_t, torch.full_like(alpha_t, NEG_INF)
            )
            shift = alpha_for_norm.max(dim=-1, keepdim=True)[0]

            # Guard against all-NEG_INF case
            shift = torch.where(shift < NEG_INF + 1.0, torch.zeros_like(shift), shift)

            # 2. Update cumulative normalization (ONLY FOR ACTIVE)
            accum_log_norm = torch.where(
                active_mask, accum_log_norm + shift.squeeze(-1), accum_log_norm
            )

            # 3. Normalize alpha_t register (ONLY FOR ACTIVE)
            alpha_t = torch.where(active_mask.view(batch, 1), alpha_t - shift, alpha_t)

            # 4. Normalize ALL K slots in ring buffer (ONLY FOR ACTIVE)
            for k_slot in range(K):
                alpha_ring[:, k_slot, :] = torch.where(
                    active_mask.view(batch, 1),
                    alpha_ring[:, k_slot, :] - shift,
                    alpha_ring[:, k_slot, :],
                )

            # 5. Re-update current slot with normalized alpha_t
            alpha_ring[:, ring_idx_t, :] = torch.where(
                active_mask.view(batch, 1), alpha_t, alpha_ring[:, ring_idx_t, :]
            )

            # 6. Save normalized ring buffer (ONLY FOR ACTIVE)
            for k_slot in range(K):
                ring_checkpoints[:, ckpt_idx, k_slot, :] = torch.where(
                    active_mask.view(batch, 1),
                    alpha_ring[:, k_slot, :],
                    ring_checkpoints[:, ckpt_idx, k_slot, :],
                )

            # 7. Save cumulative log normalization (ONLY FOR ACTIVE)
            log_norm_checkpoints[:, ckpt_idx] = torch.where(
                active_mask, accum_log_norm, log_norm_checkpoints[:, ckpt_idx]
            )

    # Capture final alpha at t == lengths
    is_final = t == lengths
    if is_final.any():
        final_alpha = torch.where(is_final.view(batch, 1), alpha_t, final_alpha)
```

### 3. Final Reduction (pytorch_reference.py:776-784)

```python
# Add back cumulative normalization
if semiring == "log":
    raw_partition = torch.logsumexp(final_alpha, dim=-1)
    partition = raw_partition + accum_log_norm
else:  # max
    partition = torch.max(final_alpha, dim=-1)[0] + accum_log_norm
```

## Active Masking Pattern

The key change in this version is consistent active masking throughout variable-length batch handling:

1. **active_mask = t <= lengths**: Includes the final position (`t == lengths`)
2. **Ring buffer updates**: Only update slots for active sequences
3. **Checkpoint normalization**: All 7 steps apply `active_mask` to prevent phantom updates for ended sequences
4. **Final capture**: Uses `t == lengths` (not `t == lengths - 1`)

This ensures:
- Sequences that have ended don't receive spurious normalization shifts
- Checkpoints contain correct state for variable-length backward pass
- `accum_log_norm` only accumulates shifts for active time steps

## Flash Attention-Style Normalization

At checkpoint boundaries, alpha values are normalized to prevent unbounded growth at extreme T (100K+):

1. **Shift extraction**: `shift = max(alpha_t)` per batch element (masked for active only)
2. **Accumulation**: `accum_log_norm += shift` tracks cumulative scale (only for active)
3. **Normalization**: All ring buffer slots shifted by `-shift` (only for active)
4. **Checkpoint**: Normalized state saved for backward (only for active)
5. **Final correction**: Partition = logsumexp(final_alpha) + accum_log_norm

This prevents the ~2.5*T log-sum growth that causes overflow at T>100K.

## Differences from Triton

| Aspect | PyTorch | Triton |
|--------|---------|--------|
| C padding | No padding | Padded to C_PAD (power of 2) |
| Device | CPU or CUDA | CUDA only |
| Precision | Native dtype | Same, but kernel ops may differ |
| NEG_INF guards | Via torch.where | Explicit guards in kernel |
| Active masking | torch.where per-op | Mask parameter in tl.load/tl.store |

## Critical Invariants

| Invariant | Math | Python Check |
|-----------|------|--------------|
| Ring buffer aliasing | alpha[t] overwrites alpha[t-K] | `t % K` indexing |
| Variable length mask | Only update active sequences | `active_mask = t <= lengths` |
| Zero-centered warning | cum_scores endpoints < 1000 | pytorch_reference.py:630-638 |
| Normalization consistency | All K slots shifted together | Loop over k_slot in range(K) |
| Inactive sequence preservation | No phantom shifts | `torch.where(active_mask, ...)` |

## Known Issues

| Issue | Severity | Frequency | Resolution |
|-------|----------|-----------|------------|
| Non-zero-centered cumsum | High | When user forgets | Warning at pytorch_reference.py:630-638 |
| Slow on GPU | Medium | Always (vs Triton) | Use Triton when available |

## Version History

- **2026-02-05**: Minor documentation fixes (comment formatting); no functional changes; updated to commit `52c3f9e`
- **2026-02-02**: Updated for active masking changes; all checkpoint operations now properly masked for variable-length sequences; line numbers updated for commit `d9aff99`
- **2026-02-02**: Updated line numbers; documented Flash Attention-style normalization; log_norm_checkpoints now returned as 4th value
- **2026-01-27**: Initial trace @ commit `09e86ed`
