# Sentinel: PyTorch Reference Backward (K >= 3)

**Verified against:** `src/torch_semimarkov/streaming/pytorch_reference.py` @ commit `f865fed`
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
| `SemiCRFStreaming.backward()` | autograd.py:77 | Backward through SemiCRFStreaming |
| `semi_crf_streaming_backward_pytorch()` | pytorch_reference.py:915 | Called from autograd |

## Data Flow

```
Inputs (from ctx.saved_tensors):
  cum_scores: (B, T+1, C)
  transition: (C, C) or (K, C, C)
  duration_bias: (K, C)
  lengths: (B,)
  ring_checkpoints: (B, num_ckpts, K, C)   <- NORMALIZED checkpoints
  log_norm_checkpoints: (B, num_ckpts)     <- Cumulative normalization factors
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

### Phase 1: Alpha Recomputation (pytorch_reference.py:984-1028)

```python
# For segment starting at checkpoint ckpt_idx
alpha_segment.fill_(NEG_INF)
alpha_ring = ring_checkpoints[:, ckpt_idx, :, :].clone()

# Store alpha[seg_start] at local position 0
alpha_segment[:, 0, :] = alpha_ring[:, seg_start % K, :]

# Recompute forward through segment
for t in range(seg_start + 1, seg_end):
    # Standard forward recurrence
    alpha_t = logsumexp over k of (alpha_prev + edge)
    alpha_ring[:, t % K, :] = alpha_t
    alpha_segment[:, local_t, :] = alpha_t
```

### Phase 2: Beta + Gradient Accumulation (pytorch_reference.py:1030-1185)

Uses relative log-marginal computation (Flash Attention pattern) for numerical stability:

```python
for t in range(seg_end-1, seg_start-1, -1):
    log_norm_at_ckpt = log_norm_checkpoints[:, ckpt_idx]  # Load shift factor

    for k in range(1, max_k + 1):
        end_pos = t + k
        beta_next = beta_ring[:, end_pos % K, :]

        # === Relative log-marginal pattern (Flash Attention) ===
        # Step 1: Clamp inputs
        alpha_t_safe = clamp(alpha_t, -1e6, 1e6)
        beta_next_safe = clamp(beta_next, -1e6, 1e6)
        edge_block_safe = clamp(edge_block, -1e6, 1e6)

        # Step 2: Compute log_joint (WITHOUT log_Z subtraction)
        log_joint = alpha_t_safe + edge_block_safe + beta_next_safe

        # Step 3: Find local reference
        local_ref = max(log_joint)

        # Step 4: Relative log-marginal (bounded in (-inf, 0])
        log_marginal_rel = log_joint - local_ref

        # Step 5: Unnormalized marginal (bounded in (0, 1])
        marginal_unnorm = exp(log_marginal_rel)

        # Step 6: CRITICAL - Compute scale using log_norm_at_ckpt
        # This bridges normalized alpha and full log_Z
        log_scale = local_ref + log_norm_at_ckpt - partition

        # Step 7: Defensive clamping
        log_scale_clamped = clamp(log_scale, -700, 0)
        scale = exp(log_scale_clamped)

        # Step 8: Final marginal
        marginal = marginal_unnorm * scale

        # Accumulate gradients
        grad_cum_scores[:, end_pos, :] += marginal.sum(dim=-1)
        grad_cum_scores[:, t, :] -= marginal.sum(dim=-1)
        grad_transition += marginal.transpose(-1, -2)
        grad_duration_bias[:, k-1, :] += marginal.sum(dim=-1)
```

## Why Relative Log-Marginal?

At extreme T (100K+), alpha values grow unbounded (~2.5 per step). Even with normalization:

- Normalized alpha: ~0 to 50 (small after shift)
- Cumulative shift (log_norm_at_ckpt): ~125,000 at midpoint
- Beta: ~125,000 (backward mass)
- True log_Z: ~250,000

Direct computation of `exp(alpha + edge + beta - log_Z)` would overflow because intermediate values exceed float64 range.

The relative pattern:
1. Computes `local_ref = max(log_joint)` (~125k)
2. Computes `marginal_unnorm = exp(log_joint - local_ref)` (bounded 0-1)
3. Computes `scale = exp(local_ref + log_norm - log_Z)` (~exp(0) = 1)
4. Final `marginal = marginal_unnorm * scale` stays bounded

## Gradient Reduction in Autograd

The backward function returns per-batch gradients. Autograd reduces them:

```python
# autograd.py:143-148
if grad_transition.ndim == 3:  # (batch, C, C)
    grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
else:  # (batch, K, C, C)
    grad_transition = torch.einsum("bkij, b -> kij", grad_transition, grad_output)

grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)
```

## Critical Invariants

| Invariant | Check |
|-----------|-------|
| Partition finite | `torch.isfinite(partition).all()` (autograd.py:92) |
| Gradients finite | `torch.isfinite(grad_cum_scores).all()` (autograd.py:120) |
| Relative log-marginal | Uses `log_norm_at_ckpt` to bridge normalized alpha and full log_Z |
| Scale clamping | `clamp(log_scale, min=-700, max=0)` prevents exp overflow |
| Per-batch gradients | transition/duration_bias returned as (B, ...) for einsum reduction |

## Known Issues

| Issue | Severity | Resolution |
|-------|----------|------------|
| Slow on GPU | Medium | Use Triton backward |
| Memory for large T | Medium | Checkpointing reduces peak |

## Version History

- **2026-02-02**: Updated line numbers (backward now at 915); documented relative log-marginal pattern with 8-step breakdown; added per-batch gradient convention
- **2026-01-27**: Initial trace @ commit `09e86ed`
