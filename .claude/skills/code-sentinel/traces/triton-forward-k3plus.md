# Sentinel: Triton Forward Kernel (K >= 3)

**Verified against:** `src/torch_semimarkov/streaming/triton_forward.py` @ commit `UNCOMMITTED`
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

| ID | Assumption | Verification Guidance |
|----|------------|----------------------|
| A4 | C_PAD is power of 2 | Check C_PAD assignment in launcher (~line 881) uses `2 ** math.ceil(math.log2(C))` |

## Entry Points

| Function | File:Line | Called When |
|----------|-----------|-------------|
| `SemiCRFStreamingTriton.forward()` | autograd.py:166 | K>=3, GPU, needs_grad, use_triton |
| `launch_streaming_triton_kernel()` | triton_forward.py:889 | Direct inference or from autograd |
| `semi_crf_streaming_scan_kernel()` | triton_forward.py:84 | Log semiring (partition function) |
| `semi_crf_streaming_scan_kernel_max()` | triton_forward.py:441 | Max semiring (Viterbi) |

## Dispatch Conditions

```python
# autograd.py:636
can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda

# autograd.py:640-652
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

Launcher allocates (triton_forward.py:988-1007):
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

### 1. Buffer Initialization (triton_forward.py:988-1007)

```python
ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)
ring_buffer[:, 0, :C] = 0.0  # alpha[0, c] = 0.0 for all valid labels

ring_checkpoints = torch.full((batch, num_ckpts, K, C_PAD), NEG_INF, ...)
ring_checkpoints[:, 0, 0, :C] = 0.0  # Initial state at checkpoint 0

log_norm_checkpoints = torch.zeros((batch, num_checkpoints), device=device, dtype=dtype)
```

### 2. Kernel Launch (triton_forward.py:1028-1076)

Grid: `(batch,)` - one program per batch element

### 3. Main Forward Loop (triton_forward.py:206-420)

```
for t in range(1, T + 1):  # t = 1, 2, ..., T
    active = t <= seq_len
    alpha_t = full([C_PAD], NEG_INF)

    for k in range(1, K + 1):  # k = 1, 2, ..., K
        k_valid = (k <= t) & (k <= K)
        start_pos = t - k

        # 3a. Read alpha[start_pos] from ring buffer (line 220-227)
        ring_k_idx = start_pos % K
        alpha_prev = load(ring_buffer[ring_k_idx])  # (C_PAD,)

        # 3b. Compute edge on-the-fly (lines 229-291)
        content_score = cum_scores[t] - cum_scores[start_pos]  # (C_PAD,)
        segment_score = content_score + duration_bias[k-1]     # (C_PAD,)
        if HAS_BOUNDARIES:
            segment_score += proj_start[start_pos] + proj_end[t-1]
        edge_block = segment_score[:, None] + transition_block  # (C_PAD, C_PAD)

        # 3c. Compute scores and logsumexp over c_src (lines 299-311)
        scores = alpha_prev[None, :] + edge_block  # (C_PAD, C_PAD)
        score_for_k = logsumexp(scores, axis=1)    # (C_PAD,) - with NEG_INF guard

        # 3d. Accumulate into alpha_t via logsumexp (lines 316-324)
        alpha_t = logsumexp_2way(alpha_t, score_for_k)  # with NEG_INF guard

    # 3e. Store to ring buffer (lines 329-335)
    ring_t_idx = t % K
    store(ring_buffer[ring_t_idx], alpha_t)

    # 3f. Checkpoint with normalization (lines 338-400)
    if t % CHECKPOINT_INTERVAL == 0:
        ckpt_idx = t // CHECKPOINT_INTERVAL

        # 3f-i. Normalize alpha for numerical stability at extreme T
        shift = max(alpha_t)  # Find normalization factor
        accum_log_norm += shift  # Track cumulative shift
        alpha_t = alpha_t - shift  # Normalize current alpha

        # 3f-ii. Normalize entire ring buffer for consistency
        for k_norm in range(K):
            ring_buffer[k_norm] -= shift

        # 3f-iii. Save normalized state to checkpoint
        for k_save in range(K):
            store(ring_checkpoints[ckpt_idx, k_save], ring_buffer[k_save])
        store(log_norm_checkpoints[ckpt_idx], accum_log_norm)

    # 3g. Capture final alpha at sequence end (lines 405-408)
    if t == seq_len:
        final_alpha = alpha_t
```

### 4. Final Reduction (triton_forward.py:419-439)

```python
# Compute raw partition from normalized final_alpha
raw_partition = logsumexp(final_alpha, axis=0)  # Reduce over labels

# Add back cumulative normalization to get true partition function
partition = raw_partition + accum_log_norm

store(out_ptr + batch_idx, partition)
```

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
| triton_forward.py:304-311 | `is_all_neginf = max_scores < (NEG_INF + 1.0)` | Prevent undefined logsumexp when all inputs are NEG_INF |
| triton_forward.py:316-324 | `is_both_neginf` check | Guard two-way logsumexp accumulation |
| triton_forward.py:340-400 | Checkpoint normalization | Prevent alpha overflow at extreme T (100k+) |
| triton_forward.py:419-430 | Final logsumexp guard | Prevent NaN in partition output |
| autograd.py:228-235 | `torch.isfinite(partition)` | Validate before backward |

## NEG_INF Handling Pattern (Flash Attention Style)

```python
# The critical guard pattern used throughout (triton_forward.py:304-311):
# Flash Attention pattern: no epsilon needed inside log
# The NEG_INF guard handles the zero case
max_val = tl.max(scores, axis=...)
is_all_neginf = max_val < (NEG_INF + 1.0)  # NEG_INF = -1e9
max_val_safe = tl.where(is_all_neginf, 0.0, max_val)
log_sum = tl.log(tl.sum(tl.exp(scores - max_val_safe[..., None]), axis=...))
result = tl.where(is_all_neginf, NEG_INF, max_val + log_sum)
```

**Why**: Without this guard, `scores - max_val` yields 0 when all inputs are NEG_INF, causing `exp(0) = 1` instead of preserving NEG_INF. The `+ 1e-10` epsilon was removed following Flash Attention's pattern - the NEG_INF guard already handles the degenerate case.

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
| Non-power-of-2 C | Medium | Always | Pad to C_PAD | triton_forward.py:881 |

## Debugging: Log-partition bounds violation

Insert after line 367 in `triton_forward.py`:

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

- **2026-02-01**: Added Flash Attention-style checkpoint normalization for T=100k+ stability; removed epsilon from logsumexp; return type now includes `log_norm_checkpoints`
- **2026-01-28**: Fixed duration-dependent transition indexing (k -> dur_idx = k-1), added K boundary tests
- **2026-01-27**: Initial trace @ commit `09e86ed`
