# Sentinel: Triton Forward Kernel (K >= 3)

**Verified against:** `src/torch_semimarkov/streaming/triton_forward.py` @ commit `49d9d61`
**Linked tests:** `tests/test_streaming_triton.py::TestTritonBasic`, `tests/test_streaming_k_boundaries.py::TestK3TritonBoundary`

## Summary

The Triton forward kernel computes the Semi-CRF partition function on GPU using O(KC) ring buffer memory. Edge potentials are computed on-the-fly from cumulative scores via prefix-sum decomposition. This is the production path for genome-scale sequences (T > 10K) with K >= 3.

**Numerical Stability (T > 100K)**: Uses Flash Attention-style checkpoint normalization to prevent alpha values from growing unbounded at extreme sequence lengths. Log normalization factors are tracked cumulatively and added back to produce the correct partition function.

## Shape Legend

- `B` = Batch size
- `T` = Sequence length (time steps)
- `K` = Maximum segment length
- `C` = Number of classes/labels
- `C_PAD` = Padded class count (next power of 2 >= C)

## Active Assumptions

### Mechanically Verified

These are verified automatically via `python3 verify-assumptions.py triton-forward-k3plus`.

| ID | Assumption | Verification |
|----|------------|--------------|
| A1 | Ring buffer uses `t % K` write indexing | anchor: RING_BUFFER_WRITE |
| A2 | NEG_INF guard pattern exists | anchor: NEGINF_GUARD |
| A3 | Checkpoint save occurs at interval boundaries | anchor: CHECKPOINT_SAVE |
| A5 | Kernel launched via `launch_streaming_triton_kernel` | anchor: KERNEL_LAUNCH |

### Agent-Verified (on trace load)

These require human/agent judgment when loading the trace.

| ID | Assumption          | Verification Guidance                                                    |
|----|---------------------|--------------------------------------------------------------------------|
| A4 | C_PAD is power of 2 | Check C_PAD assignment in launcher (line 975) uses `_next_power_of_2(C)` |

## Entry Points

| Function | File:Line | Called When |
|----------|-----------|-------------|
| `SemiCRFStreamingTriton.forward()` | autograd.py:166 | K>=3, GPU, needs_grad, use_triton |
| `launch_streaming_triton_kernel()` | triton_forward.py:902 | Direct inference or from autograd |
| `semi_crf_streaming_scan_kernel()` | triton_forward.py:84 | Log semiring (partition function) |
| `semi_crf_streaming_scan_kernel_max()` | triton_forward.py:449 | Max semiring (Viterbi) |

## Dispatch Conditions

```python
# autograd.py:641
can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda

# autograd.py:643-652
if needs_grad:
    if can_use_triton:
        return SemiCRFStreamingTriton.apply(...)  # This path
```

## Data Flow

```
Inputs:
  cum_scores: (B, T+1, C) float32     <- Zero-centered cumsum of projected scores
  transition: (C, C) or (K, C, C)     <- Label transition matrix
  duration_bias: (K, C)               <- Duration-specific bias
  lengths: (B,)                       <- Sequence lengths
  proj_start: (B, T, C) optional      <- Start boundary scores
  proj_end: (B, T, C) optional        <- End boundary scores

Launcher allocates (triton_forward.py:1000-1018):
  partition: (B,)                     <- Output
  ring_buffer: (B, K, C_PAD)          <- Live ring buffer (initialized: [0,:,0,:C]=0, rest=NEG_INF)
  ring_checkpoints: (B, num_ckpts, K, C_PAD) <- Saved states for backward
  log_norm_checkpoints: (B, num_ckpts)       <- Cumulative normalization factors

Kernel produces:
  partition: (B,)                     <- Log partition function
  ring_checkpoints: (B, num_ckpts, K, C_PAD) <- Checkpoints at interval boundaries
  log_norm_checkpoints: (B, num_ckpts)       <- For numerical stability at extreme T
```

## Algorithm Steps

### 1. Buffer Initialization (triton_forward.py:1003-1018)

```python
ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)
ring_buffer[:, 0, :C] = 0.0  # alpha[0, c] = 0.0 for all valid labels

ring_checkpoints = torch.full((batch, num_ckpts, K, C_PAD), NEG_INF, ...)
ring_checkpoints[:, 0, 0, :C] = 0.0  # Initial state at checkpoint 0

log_norm_checkpoints = torch.zeros((batch, num_checkpoints), device=device, dtype=dtype)
```

### 2. Kernel Launch (triton_forward.py:1042-1084)

Grid: `(batch,)` - one program per batch element

### 3. Main Forward Loop (triton_forward.py:208-420)

```
for t in range(1, T + 1):  # t = 1, 2, ..., T
    # ACTIVE MASKING: Include t == seq_len to compute alpha at final position
    # NOTE: Both t and seq_len are cast to int32 for consistent Triton comparison
    active = t.to(tl.int32) <= seq_len  # seq_len loaded as int32

    alpha_t = full([C_PAD], NEG_INF)

    for k in range(1, K + 1):  # k = 1, 2, ..., K
        k_valid = (k <= t) & (k <= K)
        start_pos = t - k

        # 3a. Read alpha[start_pos] from ring buffer (lines 223-227)
        # MASKED: only load for active & k_valid
        ring_k_idx = start_pos % K
        alpha_prev = load(ring_buffer[ring_k_idx], mask=active & k_valid)  # (C_PAD,)

        # 3b. Compute edge on-the-fly (lines 229-292)
        # All loads MASKED with active & k_valid
        content_score = cum_scores[t] - cum_scores[start_pos]  # (C_PAD,)
        segment_score = content_score + duration_bias[k-1]     # (C_PAD,)
        if HAS_BOUNDARIES:
            segment_score += proj_start[start_pos] + proj_end[t-1]
        edge_block = segment_score[:, None] + transition_block  # (C_PAD, C_PAD)

        # 3c. Compute scores and logsumexp over c_src (lines 299-312)
        scores = alpha_prev[None, :] + edge_block  # (C_PAD, C_PAD)
        score_for_k = logsumexp(scores, axis=1)    # (C_PAD,) - with NEG_INF guard

        # 3d. Accumulate into alpha_t via logsumexp (lines 314-323)
        alpha_t = logsumexp_2way(alpha_t, score_for_k)  # with NEG_INF guard

    # 3e. Mask inactive sequences (line 326)
    alpha_t = where(active & c_mask, alpha_t, NEG_INF)

    # 3f. Store to ring buffer (lines 329-334)
    # MASKED: only store for active sequences
    ring_t_idx = t % K
    store(ring_buffer[ring_t_idx], alpha_t, mask=active & c_mask)

    # 3g. Checkpoint with normalization (lines 340-413)
    if t % CHECKPOINT_INTERVAL == 0:
        ckpt_idx = t // CHECKPOINT_INTERVAL

        # 3g-i. Find max for normalization (MASKED for active only)
        alpha_for_norm = where(active & c_mask, alpha_t, NEG_INF)
        max_val = max(alpha_for_norm)
        # WORKAROUND: Use `active` directly instead of NEG_INF comparison
        # The comparison `max_val < (NEG_INF + 1.0)` failed silently in Triton
        shift = where(active, max_val, 0.0)

        # 3g-ii. Update cumulative normalization
        # shift is 0 for inactive, so no phantom updates
        accum_log_norm = accum_log_norm + shift

        # 3g-iii. Normalize alpha_t register
        alpha_t = alpha_t - shift

        # 3g-iv. Normalize entire ring buffer (MASKED for active only)
        for k_norm in range(K):
            ring_buffer[k_norm] = where(active, ring_buffer[k_norm] - shift, ring_buffer[k_norm])

        # 3g-v. Save normalized state to checkpoint (MASKED for active only)
        for k_save in range(K):
            save_mask = (ckpt_idx < NUM_CKPTS) & active & c_mask
            store(ring_checkpoints[ckpt_idx, k_save], ring_buffer[k_save], mask=save_mask)

        # 3g-vi. Save cumulative log normalization (MASKED for active only)
        if (ckpt_idx < NUM_CKPTS) & active:
            store(log_norm_checkpoints[ckpt_idx], accum_log_norm)

    # 3h. Capture final alpha at sequence end (lines 418-422)
    # NOTE: Cast t to int32 for consistent comparison with seq_len
    is_final = t.to(tl.int32) == seq_len
    final_alpha = where(is_final & c_mask, alpha_t, final_alpha)
```

### 4. Final Reduction (triton_forward.py:428-451)

```python
# Compute raw partition from normalized final_alpha
final_alpha_masked = where(c_mask, final_alpha, NEG_INF)
max_val = max(final_alpha_masked)
raw_partition = logsumexp(final_alpha)  # with NEG_INF guard

# Add back cumulative normalization to get true partition function
partition = raw_partition + accum_log_norm

store(out_ptr + batch_idx, partition)
```

## Active Masking Pattern

The key change in this version is consistent active masking throughout variable-length batch handling:

1. **active = t.to(tl.int32) <= seq_len**: Includes the final position (`t == seq_len`); both operands cast to int32
2. **Load masks**: All `tl.load` calls use `mask=active & k_valid & c_mask`
3. **Store masks**: All `tl.store` calls use `mask=active & c_mask`
4. **Checkpoint normalization**: All 6 steps check `active` to prevent phantom updates; uses `active` directly for shift (not NEG_INF comparison)
5. **Final capture**: Uses `t.to(tl.int32) == seq_len` (not `t == seq_len - 1`)

This ensures:
- Sequences that have ended don't receive spurious normalization shifts
- Checkpoints contain correct state for variable-length backward pass
- `accum_log_norm` only accumulates shifts for active time steps

## Ring Buffer Mechanics

```
Position:  0    1    2    3    4    5    6    7    (K=3)
Ring idx:  0    1    2    0    1    2    0    1
           ^
           Initial: alpha[0] = 0.0

At t=4:
  - Read alpha[3] from idx 0 (3 % 3 = 0)
  - Read alpha[2] from idx 2 (2 % 3 = 2)
  - Read alpha[1] from idx 1 (1 % 3 = 1)
  - Write alpha[4] to idx 1 (4 % 3 = 1)

Critical: alpha[t] overwrites alpha[t-K]. Never read alpha[t-K] after writing alpha[t].
```

## Checkpoint Mechanics

```
checkpoint_interval = max(sqrt(T * K), K)  # From _compute_checkpoint_interval

Example: T=100, K=10, interval=32
  Checkpoint 0: ring state at t=0 (initial)
  Checkpoint 1: ring state at t=32
  Checkpoint 2: ring state at t=64
  Checkpoint 3: ring state at t=96

Backward recomputes forward between checkpoints using saved ring states.
```

## Critical Invariants

| Invariant | Math | Python Check |
|-----------|------|--------------|
| Log-partition bounds | Z >= max_y s(x,y) | `assert partition >= viterbi_score` |
| Ring buffer aliasing | alpha[t] overwrites alpha[t-K] | `t % K` write, `(t-k) % K` read |
| Prefix-sum init | cum_scores[0] = 0 | `assert (cum_scores[:, 0, :] == 0).all()` |
| C_PAD power of 2 | C_PAD = 2^ceil(log2(C)) | Required for `tl.arange` |
| checkpoint_interval >= K | interval >= K | Required for correct backward |
| Active masking | Only update active sequences | All ops use `active` mask |

## Resource Budget

| Metric | Value |
|--------|-------|
| **Expected dtype** | float32 (float16 causes overflow in LogSumExp) |
| **Accumulator dtype** | float32 always |
| **SRAM Usage** | `C_PAD * K * 4` bytes per block (ring buffer) |
| **Grid Dim** | `(batch,)` - one program per batch element |
| **num_warps** | Default 4; range 2-8 |
| **Triton Special** | No atomic adds; use `tl.sum` reduction |

## Numerical Guards

| Location | Guard | Purpose |
|----------|-------|---------|
| triton_forward.py:309-313 | `is_all_neginf = max_scores < (NEG_INF + 1.0)` | Prevent undefined logsumexp when all inputs are NEG_INF |
| triton_forward.py:321-327 | `is_both_neginf` check | Guard two-way logsumexp accumulation |
| triton_forward.py:355-360 | `shift = where(active, max_val, 0.0)` | Checkpoint normalization uses `active` directly (NEG_INF comparison workaround) |
| triton_forward.py:432-438 | Final logsumexp guard | Prevent NaN in partition output |
| autograd.py:228 | `torch.isfinite(partition)` | Validate before backward |

## NEG_INF Handling Pattern (Flash Attention Style)

```python
# The critical guard pattern used throughout (triton_forward.py:309-313):
# Flash Attention pattern: no epsilon needed inside log
# The NEG_INF guard handles the zero case
max_val = tl.max(scores, axis=...)
is_all_neginf = max_val < (NEG_INF + 1.0)  # NEG_INF = -1e9
max_val_safe = tl.where(is_all_neginf, 0.0, max_val)
log_sum = tl.log(tl.sum(tl.exp(scores - max_val_safe[..., None]), axis=...))
result = tl.where(is_all_neginf, NEG_INF, max_val + log_sum)
```

**Why**: Without this guard, `scores - max_val` yields 0 when all inputs are NEG_INF, causing `exp(0) = 1` instead of preserving NEG_INF. The `+ 1e-10` epsilon was removed following Flash Attention's pattern - the NEG_INF guard already handles the degenerate case.

**WORKAROUND (checkpoint normalization)**: The NEG_INF comparison `max_val < (NEG_INF + 1.0)` failed silently in Triton for variable-length sequences due to unclear type/comparison issues. For checkpoint normalization shift computation, the kernel now uses `shift = tl.where(active, max_val, 0.0)` instead of checking `is_all_neginf`. This is semantically equivalent since inactive sequences have all-NEG_INF alpha values.

## Recomputation Logic (For Backward Pass)

| What | Saved? | Notes |
|------|--------|-------|
| `cum_scores` | Yes | ctx.save_for_backward |
| `transition` | Yes | ctx.save_for_backward |
| `duration_bias` | Yes | ctx.save_for_backward |
| `ring_checkpoints` | Yes | Checkpointed alpha states (normalized) |
| `log_norm_checkpoints` | Yes | Cumulative normalization factors |
| `partition` | Yes | Validated before backward |
| **Alpha values** | Partial | Only checkpoints; recompute between them |
| **Edge potentials** | No | Recomputed on-the-fly in backward |
| **LogSumExp intermediates** | No | Recomputed during segment-wise backward |

## Known Issues

| Issue | Severity | Frequency | Resolution | Commit |
|-------|----------|-----------|------------|--------|
| Duration-dependent transition off-by-one | Critical | HAS_DURATION_TRANSITIONS | Use `dur_idx` not `k` for indexing | `09e86ed` |
| K=1/K=2 ring buffer aliasing | Critical | Always | Dispatch to specialized paths | `870bd1f` |
| @triton.autotune corruption | Critical | Multi-config benchmark | Removed autotune decorator | See DEBUGGING_NAN.md |
| Float16 overflow | High | Long sequences | Force float32 inputs | - |
| Non-power-of-2 C | Medium | Always | Pad to C_PAD | triton_forward.py:975 |
| int32/int64 comparison failure | Critical | Variable-length batches | Cast seq_len and t to int32 | `49d9d61` |
| NEG_INF comparison failure | Critical | Variable-length + checkpointing | Use `active` directly for shift | `49d9d61` |

## Debugging: Log-partition bounds violation

Insert after line 1087 in `triton_forward.py`:

```python
# In launcher, after kernel returns
if (partition < viterbi_score).any():
    torch.save({
        'cum_scores': cum_scores.clone(),
        'transition': transition.clone(),
        'duration_bias': duration_bias.clone(),
        'lengths': lengths.clone(),
        'partition': partition,
        'ring_checkpoints': ring_checkpoints.clone(),
    }, f'partition_violation_{time.time()}.pt')
```

## Version History

- **2026-02-02**: Fixed int32/int64 type mismatch in Triton comparisons; `seq_len` loaded as int32, `t` cast to int32 for `active` and `is_final` comparisons; checkpoint normalization now uses `active` directly instead of NEG_INF comparison (workaround for silent comparison failure); updated to commit `49d9d61`
- **2026-02-02**: Updated for active masking changes; all kernel operations now properly masked for variable-length sequences; checkpoint normalization uses `active` to prevent phantom shifts; line numbers updated for commit `d9aff99`
- **2026-02-02**: Anchored to commit `d7b802c`; corrected line references for A4 verification, dispatch conditions, buffer init, kernel launch, main loop, final reduction, numerical guards
- **2026-02-01**: Added Flash Attention-style checkpoint normalization for T=100k+ stability; removed epsilon from logsumexp; return type now includes `log_norm_checkpoints`
- **2026-01-28**: Fixed duration-dependent transition indexing (k -> dur_idx = k-1), added K boundary tests
- **2026-01-27**: Initial trace @ commit `09e86ed`
