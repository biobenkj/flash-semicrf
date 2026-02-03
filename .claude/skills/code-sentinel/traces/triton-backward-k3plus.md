# Sentinel: Triton Backward Kernel (K >= 3)

**Verified against:** `src/torch_semimarkov/streaming/triton_backward.py` @ commit `49d9d61`
**Linked tests:** `tests/test_streaming_triton.py::TestTritonGradients`, `tests/test_streaming_k_boundaries.py::TestK3TritonBoundary`

## Summary

The Triton backward kernel computes gradients for the Semi-CRF partition function using the forward-backward algorithm with memory-efficient streaming checkpoints. For each checkpoint segment (processed in reverse), it recomputes alpha values forward, then computes beta backward while accumulating gradients.

**Numerical Stability (T > 100K)**: Uses relative log-marginal computation (Flash Attention pattern) to prevent overflow when alpha + beta >> log_Z at extreme scale. The log_norm_checkpoints from forward pass are used to bridge the gap between normalized alpha values and the full-scale partition function.

## Shape Legend

- `B` = Batch size
- `T` = Sequence length (time steps)
- `K` = Maximum segment length
- `C` = Number of classes/labels
- `C_PAD` = Padded class count (next power of 2 >= C)
- `SEGMENT_SIZE` = `CHECKPOINT_INTERVAL + K`

## Entry Points

| Function | File:Line | Called When |
|----------|-----------|-------------|
| `SemiCRFStreamingTriton.backward()` | autograd.py:213 | Backward through SemiCRFStreamingTriton |
| `launch_streaming_triton_backward()` | triton_backward.py:879 | Main launcher |
| `semi_crf_streaming_backward_kernel()` | triton_backward.py:54 | The Triton kernel |

## Data Flow

```
Inputs (from forward):
  cum_scores: (B, T+1, C)                <- Original input
  transition: (C, C) or (K, C, C)        <- Original input
  duration_bias: (K, C)                  <- Original input
  lengths: (B,)                          <- Original input
  log_Z: (B,)                            <- Partition from forward
  ring_checkpoints: (B, num_ckpts, K, C_PAD) <- Saved ring states (normalized)
  log_norm_checkpoints: (B, num_ckpts)   <- Cumulative normalization factors
  grad_output: (B,)                      <- Upstream gradient

Launcher allocates (triton_backward.py:1005-1030):
  alpha_buffer: (B, SEGMENT_SIZE, C_PAD) <- Recomputed alpha within segment
  beta_ring: (B, K, C_PAD)               <- Beta ring buffer
  grad_cum_scores: (B, T+1, C_PAD) float32 <- Output gradient (float32 sufficient with log_norm)
  grad_tr_workspace: (B, C, C) or (B, K, C, C) float32 <- Per-batch accumulator
  grad_db_workspace: (B, K, C) float32   <- Per-batch accumulator
  NOTE: Kernel-internal accumulators use tl.float64 for log-sum-exp operations

Outputs:
  grad_cum_scores: (B, T+1, C) original dtype <- Scaled by grad_output
  grad_transition: (C, C) or (K, C, C)   <- Reduced via einsum
  grad_duration_bias: (K, C)             <- Reduced via einsum
  grad_proj_start: (B, T, C) or None     <- If boundaries used
  grad_proj_end: (B, T, C) or None       <- If boundaries used
```

## Algorithm Overview

The backward pass processes checkpoint segments in **reverse order**:

```
Segment n-1: [ckpt_{n-1} * interval, T]
Segment n-2: [ckpt_{n-2} * interval, ckpt_{n-1} * interval]
...
Segment 0:   [0, ckpt_1 * interval]
```

For each segment:

### Phase 1: Alpha Recomputation

```python
# Load ring buffer checkpoint for segment start
ring_buffer = ring_checkpoints[:, ckpt_idx, :, :]

# Recompute alpha values forward through segment
for t in range(seg_start, seg_end):
    alpha[t] = logsumexp_k(alpha[t-k] + edge[t-k -> t])
```

### Phase 2: Beta Computation + Gradient Accumulation

```python
# Load cumulative log normalization factor for this checkpoint
log_norm_at_ckpt = log_norm_checkpoints[ckpt_idx]

# Beta backward while accumulating gradients
for t in range(seg_end-1, seg_start-1, -1):  # t = seg_end-1, ..., seg_start
    beta_t = logsumexp_k(beta[t+k] + edge[t -> t+k])

    # Compute marginal probabilities using relative log-marginal (Flash Attention pattern)
    for k in range(1, K+1):
        # Step 1: Compute log_joint (without log_Z subtraction)
        log_joint = alpha[t] + edge[t, t+k] + beta[t+k]

        # Step 2: Find local reference and compute relative marginal
        local_ref = max(log_joint)
        marginal_unnorm = exp(log_joint - local_ref)

        # Step 3: Apply scale factor using log_norm_at_ckpt
        # This bridges normalized alpha (~0) to full-scale log_Z (~250k at T=100k)
        log_scale = local_ref + log_norm_at_ckpt - log_Z
        scale = exp(clamp(log_scale, min=-700, max=0))

        # Step 4: Final marginal
        marginal = marginal_unnorm * scale * grad_output

        # Accumulate gradients
        grad_cum_scores[t:t+k] += marginal
        grad_transition += marginal
        grad_duration_bias[k] += marginal
```

## Critical Invariants

| Invariant | Math | Python Check |
|-----------|------|--------------|
| Marginal sum | sum(marginals) = 1 per sequence | `assert marginals.sum() approx 1` |
| Alpha-beta product | alpha[t] + beta[t] = log_Z (at valid t) | Validated implicitly |
| Gradient scaling | grad = marginal * grad_output | grad_output applied in kernel |
| Float64 accumulation | Accumulators use float64 | Prevents precision loss |

## Resource Budget

| Metric | Value |
|--------|-------|
| **Expected dtype** | float32 inputs, float32 accumulators (kernel uses tl.float64 internally) |
| **Accumulator dtype** | float32 for workspace tensors (triton_backward.py:1013); tl.float64 in kernel |
| **grad_cum_scores dtype** | float32 (reverted from float64; log_norm keeps values bounded) |
| **Alpha buffer** | `(B, SEGMENT_SIZE, C_PAD) * 4` bytes |
| **Beta ring** | `(B, K, C_PAD) * 4` bytes |
| **Grad workspace** | `(B, C, C) * 8` or `(B, K, C, C) * 8` bytes (float64) |
| **Grid Dim** | `(batch,)` - one program per batch element |
| **num_warps** | Default 4; range 2-8 |
| **TILE_C** | 16 (constexpr for tiling) |

## Recomputation Logic

| What | Status | Why |
|------|--------|-----|
| `ring_checkpoints` | Loaded | Saved in forward (normalized) |
| `log_norm_checkpoints` | Loaded | Cumulative normalization factors |
| `alpha[seg_start:seg_end]` | Recomputed | From checkpoint, forward through segment |
| `beta` values | Computed | Backward from final position |
| `edge` potentials | Recomputed | On-the-fly from cum_scores |
| `log_marginal` | Computed | Relative log-marginal with scale correction |

**Memory tradeoff**: sqrt(T*K) checkpoints, recompute O(checkpoint_interval) forward within each segment.

## Numerical Guards

| Location | Guard | Purpose |
|----------|-------|---------|
| autograd.py:228-235 | `torch.isfinite(partition)` | Validate partition before backward |
| triton_backward.py (kernel) | Clamp log_scale | Prevent exp() overflow: `clamp(log_scale, min=-700, max=0)` |
| autograd.py:264-277 | `torch.isfinite(grad_*)` | Validate all backward outputs |

## Gradient Reduction

The Triton kernel produces **per-batch** gradients for shared parameters. Autograd reduces them:

```python
# autograd.py lines 236-250 (Triton path)
# Triton backward already scales by grad_output internally

# Per-batch workspaces are reduced via einsum after kernel returns
# (This happens in the launcher, not autograd)
```

In the launcher (triton_backward.py:1159-1180):
```python
# Slice workspaces back to actual class count C
# Convert accum_dtype (float32) → original dtype if different
grad_cum_scores = grad_cum_scores[:, :, :C].to(dtype)
grad_transition = torch.einsum("bkij, b -> kij", grad_tr_workspace[:, :, :C, :C], grad_output)
# or for non-duration-dependent: "bij, b -> ij"
grad_duration_bias = torch.einsum("bkc, b -> kc", grad_db_workspace[:, :, :C], grad_output)
```

## Known Issues

| Issue | Severity | Frequency | Resolution | Commit |
|-------|----------|-----------|------------|--------|
| Duration-dependent transition off-by-one | Critical | HAS_DURATION_TRANSITIONS | Use `dur_idx` not `k` for indexing | `09e86ed` |
| @triton.autotune corruption | Critical | Multi-config benchmark | Removed autotune decorator | See DEBUGGING_NAN.md |
| Float32 accumulator overflow | Resolved | Long sequences, large C | log_norm_checkpoints keeps values bounded; kernel uses tl.float64 internally | d7b802c |
| Wrong checkpoint_interval | Critical | Mismatched forward/backward | Pass same interval to both | autograd.py:244 |
| grad_output not scaled | Medium | Wrong gradient magnitude | Triton kernel scales internally | - |
| int32/int64 comparison failure | Critical | Variable-length batches | Cast seq_len to int32 | `49d9d61` |

## Debugging: Gradient Mismatch

Compare against PyTorch reference:

```python
# In test, compare gradients
grad_triton = SemiCRFStreamingTriton.apply(...)
grad_pytorch = SemiCRFStreaming.apply(...)

# Check each gradient component
for name, t_grad, p_grad in [
    ('cum_scores', t_grad_cum, p_grad_cum),
    ('transition', t_grad_tr, p_grad_tr),
    ('duration_bias', t_grad_db, p_grad_db),
]:
    diff = (t_grad - p_grad).abs().max()
    print(f"{name}: max diff = {diff:.6e}")
```

## Debugging: NaN in Backward

Insert in `launch_streaming_triton_backward()` after kernel:

```python
# After kernel returns (before workspace reduction)
if not torch.isfinite(grad_cum_scores_ws).all():
    torch.save({
        'cum_scores': cum_scores.clone(),
        'transition': transition.clone(),
        'log_Z': log_Z.clone(),
        'ring_checkpoints': ring_checkpoints.clone(),
        'grad_output': grad_output.clone(),
        'grad_cum_scores_ws': grad_cum_scores_ws.clone(),
    }, f'backward_nan_{time.time()}.pt')
    raise RuntimeError("NaN in backward workspace")
```

## Version History

- **2026-02-02**: Fixed int32/int64 type mismatch for seq_len comparison; seq_len now loaded as int32 to match loop variable type from tl.range; updated to commit `49d9d61`
- **2026-02-02**: Reverted workspace accumulators from float64 to float32 @ `94652ad`; log_norm_checkpoints keeps values bounded within checkpoint blocks, so float32 is sufficient; kernel-internal accumulators still use tl.float64 for log-sum-exp safety
- **2026-02-01**: Added log_norm_checkpoints support for T=100k+ stability; relative log-marginal computation (Flash Attention pattern); removed epsilon from logsumexp
- **2026-01-28**: Fixed duration-dependent transition indexing (k -> dur_idx = k-1), added K boundary tests
- **2026-01-27**: Initial trace @ commit `09e86ed`
