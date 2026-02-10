# Streaming Semi-CRF Internals

Technical reference for working with the streaming inference module.

## Overview

The streaming module (`flash_semicrf.streaming`) provides memory-efficient Semi-CRF inference by computing edge potentials on-the-fly rather than materializing the full edge tensor.

### The Memory Problem

Standard Semi-CRF inference requires an edge tensor of shape `(batch, T, K, C, C)`:
- T = sequence length (e.g., 100,000 for genomic sequences)
- K = maximum segment duration (e.g., 1,000)
- C = number of states (e.g., 24)

This requires **O(T × K × C²)** memory, which quickly exceeds GPU capacity for genome-scale sequences.

### The Streaming Solution

The streaming approach reduces memory to **O(K × C)** by:
1. Using a **ring buffer** for forward/backward messages
2. Computing edge potentials **on-the-fly** from cumulative scores
3. **Checkpointing** the ring buffer state for the backward pass
4. **Per-checkpoint log normalization** for numerical stability at extreme T

### Module Structure

| File | Purpose |
|------|---------|
| [autograd.py](../src/flash_semicrf/streaming/autograd.py) | Public API, autograd functions, and K-dispatch logic |
| [triton_forward.py](../src/flash_semicrf/streaming/triton_forward.py) | Triton forward kernels (log/max semiring) |
| [triton_backward.py](../src/flash_semicrf/streaming/triton_backward.py) | Triton backward kernel with loop tiling |
| [triton_cache.py](../src/flash_semicrf/streaming/triton_cache.py) | Triton cache validation utilities |
| [pytorch_reference.py](../src/flash_semicrf/streaming/pytorch_reference.py) | Pure PyTorch reference implementation |
| [constants.py](../src/flash_semicrf/streaming/constants.py) | Shared constants (NEG_INF) |

---

## K-Dispatch Logic

The streaming module automatically routes to optimized implementations based on K:

| K Value | Implementation | Memory | Notes |
|---------|---------------|--------|-------|
| K=1 | Linear CRF fast path | O(batch × C) | No ring buffer, no duration loop |
| K=2 | Specialized 2-step path | O(batch × C) | Explicit history, no ring buffer |
| K≥3 | Triton streaming kernel | O(K × C) | Ring buffer architecture |

**Important observation**: The Triton kernels require K≥3 because the ring buffer architecture assumes meaningful separation between timesteps. K=1 causes ring buffer aliasing (all `t % 1 = 0`), and K=2 has ring buffer fragility.

From [autograd.py:602-656](../src/flash_semicrf/streaming/autograd.py#L602-L656):

```python
# K=1 Fast Path: Linear CRF (no ring buffer, no duration loop)
if K == 1:
    return LinearCRFStreaming.apply(...)

# K=2 Fast Path: Explicit 2-step history (no ring buffer)
if K == 2:
    return SemiCRFK2Streaming.apply(...)

# K>=3: Triton streaming kernel (ring buffer architecture)
can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda
if can_use_triton:
    return SemiCRFStreamingTriton.apply(...)
else:
    return SemiCRFStreaming.apply(...)  # Pure PyTorch fallback
```

---

## Tensor Conventions

### End-Position Indexing

The streaming module uses **end-position indexing**:
- `t` = segment **end** position (inclusive, 1-indexed in the algorithm)
- `k` = segment **duration**
- Segment covers positions `[t-k, t-1]` (0-indexed)

This differs from start-position indexing where `t` would be the start.

```
Segment with t=5, k=3:
  positions: [2, 3, 4]  (0-indexed)
  start_pos = t - k = 2
  end_pos = t - 1 = 4
```

### Destination-First Edge Convention

Edge tensors use **destination-first** ordering:
```python
edge[..., c_dst, c_src]  # Transition FROM c_src TO c_dst
```

This means:
- `transition[c_src, c_dst]` stores score for c_src → c_dst
- But when computing edges, we use `transition.T` to get `[c_dst, c_src]` layout

### Cumulative Score Layout

```python
cum_scores: (batch, T+1, C)
```

- Index 0 is the boundary (all zeros)
- Index `t` contains cumulative sum through position `t-1`
- Content score for segment `[start, end]` = `cum_scores[end+1] - cum_scores[start]`

---

## Ring Buffer Architecture

### Memory Layout

**Forward pass** uses a ring buffer indexed by `t % K`:

```python
alpha_ring: (batch, K, C)
```

**Backward pass** uses a larger ring buffer indexed by `t % (2*K)`:

```python
beta_ring: (batch, 2*K, C)
```

The backward pass requires `2*K` slots because it needs to look forward by up to K positions while also storing beta values at position t. The `2*K` size ensures no aliasing between read and write positions.

### Forward Ring Buffer

Forward messages use `t % K` indexing:

- `alpha_ring[:, t % K, :]` stores α values at position `t`
- Only the most recent `K` positions are kept
- Older values are overwritten as the scan progresses

```
Time:     0   1   2   3   4   5   6   7   8   ...
Ring idx: 0   1   2   0   1   2   0   1   2   ...  (K=3)
          ↑           ↑           ↑
          overwritten by t=3, 6, 9, ...
```

### Backward Ring Buffer

Backward messages use `t % (2*K)` indexing to avoid aliasing:
- `beta_ring[:, t % (2*K), :]` stores β values at position `t`
- Looking forward: `beta_ring[:, (t+k) % (2*K), :]` for k ∈ [1, K]
- With `2*K` slots, `t % (2*K)` never equals `(t+k) % (2*K)` for k ∈ [1, K]

### Why O(KC) Memory

Standard forward pass stores all α values: **O(T × C)** memory.

With max segment duration K, position `t` only depends on positions `[t-K+1, t-1]`. The ring buffer exploits this by storing only the K most recent values (forward) or 2K values (backward).

---

## Edge Computation On-the-Fly

### The Decomposed Potential

Edge potentials are computed on-the-fly using prefix-sum decomposition:

```
edge[c_dst, c_src] = content_score[c_dst] + duration_bias[k, c_dst] + transition[c_src, c_dst]
```

Where the content score is derived from cumulative scores:
```
content_score = cum_scores[t, c_dst] - cum_scores[t-k, c_dst]
```

### Code Location

From [triton_forward.py:232-295](../src/flash_semicrf/streaming/triton_forward.py#L232-L295) (kernel implementation):

```python
# Content score via cumulative sum difference
cum_end = tl.load(cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c, ...)
cum_start = tl.load(cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c, ...)
content_score = cum_end - cum_start  # (C_PAD,)

# Add duration bias (duration k uses index k-1)
dur_idx = k - 1
dur_bias = tl.load(duration_bias_ptr + dur_idx * stride_db_k + c_idx * stride_db_c, ...)
segment_score = content_score + dur_bias  # (C_PAD,)

# Build edge block
edge_block = segment_score[:, None] + transition_block  # (C_PAD, C_PAD)
```

### Duration-Dependent Transitions

The streaming module supports both **static** and **duration-dependent** transitions:

**Static transitions** `(C, C)`: Same transition matrix for all segment durations

```python
transition[c_src, c_dst]  # Single matrix
```

**Duration-dependent transitions** `(K, C, C)`: Different transitions per duration

```python
transition[k, c_src, c_dst]  # k = 0, 1, ..., K-1 (duration k+1 uses index k)
```

The kernel detects the shape at launch and uses the appropriate loading pattern:

From [triton_forward.py:283-293](../src/flash_semicrf/streaming/triton_forward.py#L283-L293):

```python
# For duration-dependent transitions, load transition[dur_idx] inside the loop
# Duration k uses index k-1 (same convention as PyTorch reference)
if HAS_DURATION_TRANSITIONS:
    transition_block = tl.load(
        transition_ptr
        + dur_idx * stride_tr_k
        + c_dst_idx * stride_tr_dst
        + c_src_idx * stride_tr_src,
        mask=c_mask_2d,
        other=0.0,
    )  # (C_PAD, C_PAD) - transition[dur_idx].T
```

---

## Code-to-Math Correspondence

| Code Variable | Math Notation | Shape | Description |
|--------------|---------------|-------|-------------|
| `cum_scores[:, t, c]` | S_t,c | (batch, T+1, C) | Cumulative projected scores |
| `transition[c_src, c_dst]` | T_c',c | (C, C) | Static label transitions (src → dst) |
| `transition[k, c_src, c_dst]` | T_k,c',c | (K, C, C) | Duration-dependent transitions |
| `duration_bias[k, c]` | B_k,c | (K, C) | Duration-specific label bias |
| `alpha_ring[:, t%K, :]` | α̃_t | (batch, C) | Log-forward messages (ring buffer) |
| `beta_ring[:, t%(2K), :]` | β̃_t | (batch, C) | Log-backward messages (2K ring buffer) |
| `ring_checkpoints[:, i, :, :]` | Ω_i | (batch, K, C) | Saved ring buffer state at checkpoint i |
| `log_norm_checkpoints[:, i]` | N_i | (batch,) | Cumulative log normalization at checkpoint i |
| `log_Z` | log Z | (batch,) | Log partition function |

### Mathematical Notation

The edge potential (log-domain) with static transitions:
```
ψ̃(t, k, c, c') = (S_t,c - S_{t-k},c) + B_k,c + T_c',c
```

With duration-dependent transitions:
```
ψ̃(t, k, c, c') = (S_t,c - S_{t-k},c) + B_k,c + T_{k,c',c}
```

Forward recurrence:
```
α̃_t(c) = logsumexp_{k,c'} [ α̃_{t-k}(c') + ψ̃(t, k, c, c') ]
```

---

## Forward Pass Walkthrough

### Kernel Entry Point

From [triton_forward.py:83-135](../src/flash_semicrf/streaming/triton_forward.py#L83-L135):

```python
@triton.jit
def semi_crf_streaming_scan_kernel(
    cum_scores_ptr,      # (batch, T+1, C)
    transition_ptr,      # (C, C) or (K, C, C)
    duration_bias_ptr,   # (K, C)
    lengths_ptr,         # (batch,)
    proj_start_ptr,      # (batch, T, C) - optional
    proj_end_ptr,        # (batch, T, C) - optional
    out_ptr,             # (batch,) - partition function
    ring_ptr,            # (batch, K, C_PAD) - live ring buffer
    ring_ckpt_ptr,       # (batch, num_ckpts, K, C_PAD) - checkpoints
    log_norm_ckpt_ptr,   # (batch, num_ckpts) - cumulative log normalization
    ...
):
```

### Initialization

The ring buffer and checkpoint 0 are pre-initialized by the launcher:

From launcher code:

```python
# Ring buffer initialized in launcher
ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)
ring_buffer[:, 0, :C] = 0.0  # alpha[0, c] = 0.0 for all valid labels

# Checkpoint storage initialized
ring_checkpoints = torch.full(
    (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=dtype
)
ring_checkpoints[:, 0, 0, :C] = 0.0  # Initial state at checkpoint 0

# Log normalization checkpoints
log_norm_checkpoints = torch.zeros(batch, num_checkpoints, device=device, dtype=dtype)
```

### Main Loop Structure

From [triton_forward.py:207-338](../src/flash_semicrf/streaming/triton_forward.py#L207-L338):

```python
# Cumulative log normalization factor for numerical stability at extreme T
accum_log_norm = tl.zeros((), dtype=tl.float64)

for t in tl.range(1, T + 1):
    # Cast t to int32 to match seq_len type for consistent comparison
    active = t.to(tl.int32) <= seq_len
    alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float64)

    for k in tl.range(1, K + 1):
        k_valid = (k <= t) & (k <= K)
        start_pos = t - k
        ring_k_idx = start_pos % K

        # Load alpha_prev from ring buffer
        alpha_prev = tl.load(ring_base + ring_k_idx * stride_ring_k + ...)

        # Compute edge on-the-fly (prefix-sum decomposition)
        content_score = cum_end - cum_start
        segment_score = content_score + dur_bias
        edge_block = segment_score[:, None] + transition_block

        # Accumulate via logsumexp
        scores = alpha_prev[None, :] + edge_block
        score_for_k = logsumexp(scores, axis=1)  # Over c_src
        alpha_t = logsumexp_2way(alpha_t, score_for_k)  # Over k

    # Store to live ring buffer
    ring_t_idx = t % K
    tl.store(ring_base + ring_t_idx * stride_ring_k + ..., alpha_t, ...)
```

### Logsumexp Reduction Pattern

The kernel uses a two-step logsumexp with NEG_INF guards:

1. **Over source labels** (c_src): `logsumexp(scores, axis=1)`
2. **Over durations** (k): Accumulated incrementally via guarded logsumexp

From [triton_forward.py:304-326](../src/flash_semicrf/streaming/triton_forward.py#L304-L326):

```python
# Logsumexp over c_src (axis=1) with NEG_INF guard
max_scores = tl.max(scores, axis=1)
is_all_neginf = max_scores < (NEG_INF + 1.0)
max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
log_sum_exp = tl.log(tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1))
score_for_k = tl.where(is_all_neginf, NEG_INF, max_scores + log_sum_exp)

# Accumulate into alpha_t via guarded logsumexp
max_alpha = tl.maximum(alpha_t, score_for_k)
is_both_neginf = (alpha_t < (NEG_INF + 1.0)) & (score_for_k < (NEG_INF + 1.0))
max_alpha_safe = tl.where(is_both_neginf, 0.0, max_alpha)
log_sum_exp_acc = tl.log(
    tl.exp(alpha_t - max_alpha_safe) + tl.exp(score_for_k - max_alpha_safe)
)
alpha_t = tl.where(is_both_neginf, NEG_INF, max_alpha + log_sum_exp_acc)
```

The NEG_INF guards prevent undefined arithmetic when all inputs are `-1e9` (would produce `exp(0) = 1` incorrectly).

---

## Checkpointing for Backward Pass

### Why Checkpointing?

The ring buffer only stores the K most recent α values. For the backward pass, we need α values at all positions. Options:
1. **Store all α**: O(TC) memory - defeats the purpose
2. **Recompute from scratch**: O(T²KC²) time - too slow
3. **Checkpoint + recompute**: O(T/S × KC) memory, O(TKC²) time

We use option 3 with checkpoint interval S = √(T×K).

### Checkpoint Storage

```python
ring_checkpoints: (batch, num_ckpts, K, C)
log_norm_checkpoints: (batch, num_ckpts)
```

At each checkpoint position `i × S`, we save the entire ring buffer state and the cumulative log normalization factor.

### Checkpoint Interval Calculation

From [pytorch_reference.py:15-18](../src/flash_semicrf/streaming/pytorch_reference.py#L15-L18):

```python
def _compute_checkpoint_interval(T: int, K: int) -> int:
    """Optimal interval minimizes total memory.

    Memory = (T/S) × K × C + S × C + K × C
    Taking d/dS = 0 gives S* = sqrt(T × K)
    """
    optimal = int(math.sqrt(T * K))
    return max(K, optimal)  # At least K
```

### Saving Checkpoints with Normalization

From [triton_forward.py:339-414](../src/flash_semicrf/streaming/triton_forward.py#L339-L414):

```python
should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
ckpt_idx = t // CHECKPOINT_INTERVAL
if should_checkpoint:
    # ===== NORMALIZATION STEP (for numerical stability at extreme T) =====
    # Following Flash Attention pattern: normalize at checkpoints
    
    # 1. Find max alpha value for normalization (active sequences only)
    alpha_for_norm = tl.where(active & c_mask, alpha_t, NEG_INF)
    max_val = tl.max(alpha_for_norm)
    shift = tl.where(active, max_val, 0.0)
    
    # 2. Update cumulative normalization factor
    accum_log_norm = accum_log_norm + shift
    
    # 3. Normalize alpha_t register
    alpha_t = alpha_t - shift
    
    # 4. Normalize ALL K slots in ring buffer
    for k_norm in tl.range(0, K):
        ring_val = tl.load(ring_base + k_norm * stride_ring_k + ...)
        ring_val_shifted = ring_val - shift
        tl.store(ring_base + k_norm * stride_ring_k + ..., ring_val_shifted, mask=active & c_mask)
    
    # 5. Save normalized ring buffer to checkpoint
    for k_save in tl.range(0, K):
        ring_val = tl.load(ring_base + k_save * stride_ring_k + ...)
        tl.store(ring_ckpt_base + ckpt_idx * stride_ckpt_n + k_save * stride_ckpt_k + ..., ring_val)
    
    # 6. Save cumulative log normalization factor
    if (ckpt_idx < NUM_CKPTS) & active:
        tl.store(log_norm_ckpt_ptr + batch_idx * stride_lnc_b + ckpt_idx * stride_lnc_n, accum_log_norm)
```

---

## Per-Checkpoint Log Normalization

### The Problem: Numerical Overflow at T=100k

At extreme sequence lengths (T ≥ 100,000), the forward-backward algorithm accumulates log-space values that grow proportionally with T. For a sequence with mean emission score μ ≈ -0.3 per position:

```text
α[T] ≈ μ × T ≈ -0.3 × 100,000 ≈ -30,000
log_Z ≈ -30,000 (similar magnitude)
```

The problem manifests in the backward pass when computing marginals:

```text
log_marginal = α[t] + edge + β[t'] - log_Z
scale = exp(log_marginal)
```

**Without normalization**, the forward-backward identity `α[t] + β[t] ≈ log_Z` can drift due to accumulated floating-point error:

| Metric | Expected | Observed (Broken) |
|--------|----------|-------------------|
| `α[t] + β[t]` | ≈ log_Z | 330k - 360k |
| `log_Z` | - | 257k |
| `log_scale` | ≈ 0 | +70k to +100k |
| `exp(log_scale)` | ≈ 1 | **Inf** |

This causes `scale` to overflow, producing NaN gradients.

### The Solution: Checkpoint-Boundary Normalization

Following the **Flash Attention** pattern, we normalize the ring buffer at checkpoint boundaries and track the cumulative normalization factor:

```text
Forward pass at checkpoint i:
  1. shift = max(α_t)           # Find normalization constant
  2. α_t ← α_t - shift          # Normalize current alpha
  3. ring_buffer ← ring_buffer - shift  # Normalize all K slots
  4. accum_log_norm += shift    # Track cumulative shift
  5. Save ring_buffer to checkpoint
  6. Save accum_log_norm to log_norm_checkpoints[i]

Final partition:
  log_Z = logsumexp(α_T) + accum_log_norm
```

The backward pass restores the identity by adding `log_norm_at_ckpt`:

```text
Backward pass at segment from checkpoint i:
  1. Load log_norm_at_ckpt from log_norm_checkpoints[i]
  2. Recompute α from checkpoint (normalized values)
  3. Compute β backward
  4. Marginal: log_scale = (α + edge + β) + log_norm_at_ckpt - log_Z
                                            ^^^^^^^^^^^^^^^^
                                            Restores the true α magnitude
```

**Why this works**: At checkpoint i (roughly position t = i × interval), the accumulated shift `log_norm ≈ μ × t`. Adding it back bridges the gap:

```text
Example at t=50,000 (midpoint of T=100k):
  normalized_α ≈ 0 (after shift)
  log_norm ≈ 125,000 (accumulated shift)
  β ≈ 125,000 (backward mass)
  log_Z ≈ 250,000

  log_scale = (0 + edge + 125k) + 125k - 250k ≈ 0  ✓
```

### Memory Impact

The additional storage is negligible:

```python
log_norm_checkpoints: (batch, num_checkpoints)
# At T=100k with ~14 checkpoints: 14 × batch × 8 bytes = 112 bytes/sequence (float64)
```

### Literature Grounding

This technique synthesizes established methods:

1. **Flash Attention (Dao et al., 2022)** — Online softmax with running max/sum normalization, correcting at tile boundaries. Our checkpoint normalization follows the same pattern.

2. **HMM Scaling (Rabiner, 1989)** — Classical forward-backward algorithms use "scaling coefficients" at each timestep to prevent underflow. We apply the same principle at checkpoint granularity.

3. **CTC (Graves et al., 2006)** — Connectionist Temporal Classification uses similar log-space normalization for numerical stability.

4. **Gradient Checkpointing (Chen et al., 2016)** — Saving intermediate state for memory-efficient backprop; we extend this to include normalization state.

### Variable-Length Batch Handling

#### The Problem: Phantom Normalization for Ended Sequences

When processing batches with variable-length sequences, sequences that end before `T_max` must not accumulate normalization shifts at later checkpoints. Without proper masking, phantom shifts would be added:

```text
Batch with lengths = [50, 75, 100], checkpoint_interval = 25:

Without masking (BROKEN):
  Sequence 0 (L=50):
    ckpt 1 (t=25): shift = max(α) → valid
    ckpt 2 (t=50): shift = max(α) → valid (final for this sequence)
    ckpt 3 (t=75): shift = max(NEG_INF) = NEG_INF → WRONG! Adds phantom shift
    ckpt 4 (t=100): shift = max(NEG_INF) = NEG_INF → WRONG!

  Result: partition[0] = logsumexp(α_50) + accum_log_norm (includes phantom shifts)
          partition[0] ≈ expected + 2×NEG_INF ≈ -2e9 → COMPLETELY WRONG
```

#### The Solution: Active Sequence Masking

From [triton_forward.py:354-361](../src/flash_semicrf/streaming/triton_forward.py#L354-L361):

```python
# 1. Find max alpha value for normalization (over valid C only)
#    For inactive sequences, use NEG_INF to produce shift=0
alpha_for_norm = tl.where(active & c_mask, alpha_t, NEG_INF)
max_val = tl.max(alpha_for_norm)

# WORKAROUND: Use `active` directly to determine if we should apply shift
shift = tl.where(active, max_val, 0.0)

# 2. Update cumulative normalization factor
#    shift is 0 for inactive sequences, so no phantom updates
accum_log_norm = accum_log_norm + shift
```

**Important observation**: For inactive sequences, `shift = 0` and `accum_log_norm` freezes at the correct value.

---

## Backward Pass Walkthrough

### Two-Phase Algorithm with Loop Tiling

The backward pass processes segments in reverse order with **loop tiling** to reduce register pressure:

From [triton_backward.py:307-326](../src/flash_semicrf/streaming/triton_backward.py#L307-L326):

```python
for ckpt_idx_loop in tl.range(0, NUM_CKPTS):
    ckpt_idx = NUM_CKPTS - 1 - ckpt_idx_loop  # Reverse order
    seg_start = ckpt_idx * CHECKPOINT_INTERVAL
    seg_end = min((ckpt_idx + 1) * CHECKPOINT_INTERVAL, T)

    # Load cumulative log normalization factor for this checkpoint
    log_norm_at_ckpt = tl.load(log_norm_ckpt_ptr + batch_idx * stride_lnc_b + ckpt_idx * stride_lnc_n)

    # Compute segment-specific workspace base pointers for deterministic accumulation
    grad_tr_ws_seg = grad_tr_ws_base + ckpt_idx * stride_gtw_seg
    grad_db_ws_seg = grad_db_ws_base + ckpt_idx * stride_gdbw_seg

    # Phase 1: Recompute alpha from checkpoint
    # Phase 2: Compute beta backward + accumulate gradients (TILED)
```

### Phase 1: Alpha Recomputation

Load ring buffer state from checkpoint, then recompute forward through the segment. Alpha buffer is **segment-isolated** to prevent race conditions:

From [triton_backward.py:340-366](../src/flash_semicrf/streaming/triton_backward.py#L340-L366):

```python
# Alpha buffer has segment dimension: (batch, num_segments, SEGMENT_SIZE, C_PAD)
alpha_buf_seg = alpha_buf_base + ckpt_idx * stride_ab_seg

# Load checkpoint and restore to buffer position 0
for k_slot in tl.range(0, K):
    alpha_val = tl.load(ring_ckpt_base + ckpt_idx * stride_ckpt_n + k_slot * stride_ckpt_k + ...)
    if k_slot == seg_start % K:
        tl.store(alpha_buf_seg + 0 * stride_ab_t + ..., alpha_val)

# CRITICAL: Memory barrier ensures writes visible across all warps
tl.debug_barrier()

# Recompute alpha for positions seg_start+1 to seg_end
for local_t in tl.range(1, SEGMENT_SIZE):
    t = seg_start + local_t
    if t < seg_end and t < seq_len:
        # Same forward recurrence as main kernel
        ...
```

### Phase 2: Tiled Beta Backward + Gradients

The marginal computation requires a `(C_PAD × C_PAD)` matrix. At C_PAD=64, this demands ~384 registers/thread. With num_warps=4+, this exceeds available registers and causes spilling.

**Solution**: Process c_dst dimension in tiles of TILE_C (typically 16):

From [triton_backward.py:510-730](../src/flash_semicrf/streaming/triton_backward.py#L510-L730):

```python
for t_offset in tl.range(0, CHECKPOINT_INTERVAL):
    t = seg_end - 1 - t_offset
    if t >= seg_start and t < seq_len and t >= 0:
        alpha_t = tl.load(alpha_buf_seg + local_t * stride_ab_t + ...)
        new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float64)

        for k in tl.range(1, K + 1):
            end_pos = t + k
            if end_pos <= seq_len:
                # === PASS 1: Compute global statistics across all tiles ===
                global_max = tl.full((), NEG_INF, dtype=tl.float64)
                global_sum_exp = tl.zeros((), dtype=tl.float64)

                for c_dst_tile_start in tl.static_range(0, C_PAD, TILE_C):
                    # Load (TILE_C, C_PAD) tile of log_joint
                    # Update global_max and global_sum_exp via online algorithm
                    ...

                # Compute scale factor bridging normalized alpha and full log_Z
                log_scale = global_max_safe + log_norm_at_ckpt - log_Z
                scale = tl.exp(log_scale_clamped)

                # === PASS 2: Compute marginals using global statistics ===
                for c_dst_tile_start in tl.static_range(0, C_PAD, TILE_C):
                    # Load tile, compute marginal = exp(log_joint - global_max) * scale
                    # Accumulate gradients per-tile
                    ...

        # Store beta[t] to ring buffer (uses 2*K slots)
        t_ring_idx = t % (2 * K)
        tl.store(beta_ring_base + t_ring_idx * stride_br_k + ..., new_beta)
```

### Why Two-Pass Tiling?

The two-pass algorithm ensures all tiles use **identical normalization**:

1. **Pass 1**: Compute `global_max` and `global_sum_exp` across ALL tiles using online logsumexp (Flash Attention pattern)
2. **Pass 2**: Compute marginals using the global statistics

Without this, each tile would use its own local normalization, causing incorrect marginals when marginals span multiple tiles (10-400% errors observed in testing).

### Beta Ring Buffer (2K slots)

The backward pass uses `t % (2*K)` indexing to prevent aliasing:

```python
# Initialize at final position
final_ring_idx = final_pos % (2 * K)
for k_init in tl.range(0, 2 * K):
    is_final = k_init == final_ring_idx
    init_val = tl.where(is_final & c_mask, 0.0, NEG_INF)
    tl.store(beta_ring_base + k_init * stride_br_k + ...)

# During backward: read from (t+k) % (2*K), write to t % (2*K)
end_ring_idx = end_pos % (2 * K)
beta_next = tl.load(beta_ring_base + end_ring_idx * stride_br_k + ...)
```

---

## Gradient Semantics

### Per-Batch vs Shared Parameters

There are two types of parameters:

1. **Per-batch parameters** (cum_scores, proj_start, proj_end):
   - Shape includes batch dimension
   - Gradients scaled by `grad_output[batch_idx]` **inside** the kernel

2. **Shared parameters** (transition, duration_bias):
   - Shape does NOT include batch dimension
   - Accumulated per-batch-per-segment in **workspace buffers**, then reduced via einsum

### Per-Segment Workspace Pattern

The Triton backward kernel uses per-segment workspace buffers to eliminate cross-segment atomic contention:

```python
# Workspace shapes include segment dimension for isolation
# Static transitions: (batch, num_segments, C_PAD, C_PAD)
# Duration-dependent: (batch, num_segments, K, C_PAD, C_PAD)
grad_tr_workspace = torch.zeros(batch, num_segments, K, C_PAD, C_PAD, dtype=torch.float64)
grad_db_workspace = torch.zeros(batch, num_segments, K, C_PAD, dtype=torch.float64)

# Each segment writes to its own workspace slice
grad_tr_ws_seg = grad_tr_ws_base + ckpt_idx * stride_gtw_seg
grad_db_ws_seg = grad_db_ws_base + ckpt_idx * stride_gdbw_seg
```

### The Einsum Reduction Pattern

After kernel execution, host-side reduction combines per-segment workspaces:

From [triton_backward.py:1447-1466](../src/flash_semicrf/streaming/triton_backward.py#L1447-L1466):

```python
# 1. Sum over segments (deterministic order)
grad_tr_workspace = grad_tr_workspace[:, :, :, :C, :C].sum(dim=1)  # (batch, K, C, C)
grad_db_workspace = grad_db_workspace[:, :, :, :C].sum(dim=1)      # (batch, K, C)

# 2. Weighted reduction via einsum
# Notation: b=batch, s=segment, k=duration, i=src_state, j=dst_state
grad_output_f64 = grad_output.to(torch.float64)

if has_duration_transitions:
    grad_transition = torch.einsum("bkij, b -> kij", grad_tr_workspace, grad_output_f64)
else:
    grad_transition = torch.einsum("bij, b -> ij", grad_tr_workspace, grad_output_f64)

grad_duration_bias = torch.einsum("bkc, b -> kc", grad_db_workspace, grad_output_f64)
```

### Why Float64 for Shared Parameter Gradients?

Shared parameter gradients (`grad_transition`, `grad_duration_bias`) accumulate contributions from all T×K positions. With T=100k and K=1000, this means ~10^8 atomic additions per gradient element. Float32 accumulation would introduce O(sqrt(T×K×C)) error per element, leading to ~1e-3 relative error — visible in training. Float64 reduces this to ~1e-10 (negligible).

Per-position gradients (`grad_cum_scores`) don't accumulate across T, so float32 is sufficient.

---

## Numerical Considerations

### Float64 Internal Computation

All internal computation uses **float64** for numerical stability at extreme sequence lengths. This includes:

**Forward kernel:**
- `alpha_t`, `final_alpha`: float64 registers
- `accum_log_norm`: float64 scalar
- `ring_buffer`: float64 tensor `(batch, K, C_PAD)`
- `ring_checkpoints`: float64 tensor `(batch, num_ckpts, K, C_PAD)`
- `log_norm_checkpoints`: float64 tensor `(batch, num_ckpts)`
- `partition`: float64 output

From [triton_forward.py:974](../src/flash_semicrf/streaming/triton_forward.py#L974):
```python
dtype = torch.float64  # Internal computation in float64 for numerical stability
```

**Backward kernel:**
- `alpha_buffer`, `beta_ring`: float64 working memory
- `grad_tr_workspace`, `grad_db_workspace`: float64 (accumulated across T)
- Kernel-internal accumulators: `tl.float64` for logsumexp safety

From [triton_backward.py:1212](../src/flash_semicrf/streaming/triton_backward.py#L1212):
```python
dtype = torch.float64  # Must match ring_checkpoints dtype from forward pass
```

### Selective Precision for Gradients

Gradient tensors use **selective precision** based on their accumulation pattern:

From [triton_backward.py:1276-1285](../src/flash_semicrf/streaming/triton_backward.py#L1276-L1285):
```python
# NUMERICAL STABILITY: Selective precision for gradient tensors.
# - grad_cum_scores: O(B*T*C) but per-position (no cross-T accumulation) -> float32 OK
# - grad_transition, grad_duration_bias: small but accumulated across all T -> need float64
# The kernel-internal accumulators still use tl.float64 for log-sum-exp safety.

grad_cum_scores = torch.zeros(batch, T_plus_1, C_PAD, device=device, dtype=torch.float32)
grad_tr_workspace = torch.zeros(..., dtype=torch.float64)
grad_db_workspace = torch.zeros(..., dtype=torch.float64)
```

| Gradient | Dtype | Rationale |
|----------|-------|-----------|
| `grad_cum_scores` | float32 | Per-position, no cross-T accumulation |
| `grad_transition` | float64 | Accumulated across all T×K positions |
| `grad_duration_bias` | float64 | Accumulated across all T positions |
| `grad_proj_start/end` | float32 | Per-position, no cross-T accumulation |

### Autograd Dtype Handling

The autograd function manages dtype conversion between user-facing float32 and internal float64:

From [autograd.py:203-224](../src/flash_semicrf/streaming/autograd.py#L203-L224):
```python
# Cast partition back to input dtype for return, but keep float64 for backward
partition_f64 = partition  # Keep float64 for backward pass
partition_return = partition.to(cum_scores.dtype)  # Return float32 to user

# Save float64 version for backward
ctx.save_for_backward(
    ...
    partition_f64,  # <-- float64 for backward numerical stability
    ...
)

return partition_return  # <-- float32 for user
```

### Zero-Centering for Cumulative Scores

Before computing cumulative scores, zero-center the projected scores to prevent magnitude growth:

```python
projected = projected - projected.mean(dim=1, keepdim=True)
cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
cum_scores[:, 1:, :] = torch.cumsum(projected.float(), dim=1)
```

**Why this matters**: Without zero-centering, cumulative scores can reach magnitudes of O(T) which reduces effective precision. Zero-centering keeps values bounded regardless of T.

### NEG_INF Masking

Invalid positions are masked with `NEG_INF = -1e9` (not `-inf` to avoid NaN in gradients):

```python
# From constants.py
NEG_INF = -1e9

# Masking pattern in kernel
alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)
```

The threshold `-1e9` is chosen to be:
- Small enough to be dominated by any valid score in logsumexp
- Large enough that `exp(NEG_INF - max)` underflows to 0 rather than producing NaN

### NEG_INF Guards in Logsumexp

When all logsumexp inputs are NEG_INF, the standard computation `max + log(sum(exp(x - max)))` breaks:
- `max = NEG_INF`
- `x - max = 0` (instead of staying at NEG_INF)
- `exp(0) = 1`, `log(C) ≈ 3` → wrong result

The kernel guards against this:

```python
max_scores = tl.max(scores, axis=1)
is_all_neginf = max_scores < (NEG_INF + 1.0)
max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
log_sum_exp = tl.log(tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1))
score_for_k = tl.where(is_all_neginf, NEG_INF, max_scores + log_sum_exp)
```

### Log-Marginal Clamping

Before `exp()` in the backward pass, log-marginals are clamped to prevent float64 overflow:

```python
log_scale_clamped = tl.minimum(log_scale, 0.0)  # Upper bound: scale ≤ 1
log_scale_clamped = tl.maximum(log_scale_clamped, -700.0)  # Prevent underflow
scale = tl.exp(log_scale_clamped)
```

The bounds are chosen because `exp(710) ≈ inf` in float64.

### Forward-Backward Identity at Scale

At extreme sequence lengths (T ≥ 100k), accumulated floating-point error can violate the forward-backward identity `α[t] + β[t] ≈ log_Z`, causing marginal scale factors to overflow. This is solved by **per-checkpoint log normalization** — see the dedicated section [Per-Checkpoint Log Normalization](#per-checkpoint-log-normalization) for details.

---

## Triton Cache Validation

The module includes cache validation utilities to detect when Triton kernel configuration has changed:

From [triton_cache.py](../src/flash_semicrf/streaming/triton_cache.py):

```python
class TritonConfig(NamedTuple):
    """Configuration values that affect Triton kernel compilation."""
    num_warps: int

def validate_triton_cache(config: TritonConfig, warn: bool = True) -> bool:
    """Check if Triton cache matches current config.
    
    Returns True if cache is valid, False on mismatch (with warning).
    """
```

This helps users avoid stale cache issues when changing `num_warps` or other configuration.

---

## Performance Characteristics

### When Streaming Beats Pre-computed Edges

Memory bandwidth is the bottleneck, not compute:
- **Pre-computed**: Load O(T×K×C²) edge tensor from memory
- **Streaming**: Compute edges from O(T×C) cumulative scores

For large K, streaming is faster even with the extra computation.

### Benchmark Comparison (NVIDIA L40S)

| Configuration | pre-computed edges | streaming | Advantage |
|---------------|-------------------|-----------|-----------|
| K=100, batch=64 | 127ms, 14GB | 38ms, 6MB | 3.35× faster, 2,393× less memory |
| K=500, batch=32 | 330ms, 35GB | 224ms, 3MB | 1.48× faster, 11,795× less memory |

### Batch Scaling

Streaming memory scales as O(batch × T × C), not O(batch × T × K × C²):
- Can process larger batches
- Linear scaling with batch size

---

## Debugging Tips

### Common Issues

1. **Shape mismatch with boundaries**: `proj_start` and `proj_end` must have shape `(batch, T, C)`, not `(batch, T+1, C)`

2. **Gradient NaN**: Usually caused by:
   - Not zero-centering before cumulative sums
   - Using float16 for cum_scores
   - At T ≥ 100k: forward-backward identity violation (see [Per-Checkpoint Log Normalization](#per-checkpoint-log-normalization))

3. **Wrong partition value**: Check that `cum_scores[:, 0, :]` is all zeros (the boundary condition)

4. **K < 3 with Triton kernels**: Use the dispatch logic in `semi_crf_streaming_forward()` which automatically routes K=1 and K=2 to specialized implementations

### Verification Against Reference

Use the PyTorch reference for debugging:

```python
from flash_semicrf.streaming.pytorch_reference import semi_crf_streaming_forward_pytorch

# Compare results (both return 4 values: partition, ring_ckpts, interval, log_norm_ckpts)
partition_triton, _, _, _ = launch_streaming_triton_kernel(...)
partition_ref, _, _, _ = semi_crf_streaming_forward_pytorch(...)

assert torch.allclose(partition_triton, partition_ref, atol=1e-4)
```

### Gradient Agreement Testing

A dedicated test script at `benchmarks/practical_demonstration/synthetic_genomics/gradient_check.py` compares Triton kernel gradients vs PyTorch reference at the low-level API.

**Important observation**: Both implementations now use log normalization checkpointing for numerical stability at extreme sequence lengths (T>100K). The PyTorch reference was updated to match the Triton kernel's Flash Attention-style checkpoint normalization pattern.

| Gradient | Metric | Threshold | Why |
| -------- | ------ | --------- | --- |
| Forward partition | Max relative error | < 1e-4 | Core correctness |
| grad_transition | Max relative error | < 1e-2 | Parameter gradient |
| grad_duration_bias | Max relative error | < 1e-2 | Parameter gradient |
| grad_cum_scores | **Mean absolute error** | < 1e-3 | See below |

**Why grad_cum_scores uses mean absolute error:**

The max relative error on grad_cum_scores can be large (~70x at T=10k) even when the implementation is correct. This occurs at positions where:

- Reference gradient is near-zero (~1e-3)
- Absolute difference is small (~7e-3)
- Division by near-zero inflates relative error to ~70

The mean absolute error (2.6e-4 at T=10k) correctly captures that most positions agree closely.

**Error scaling with T:**

| Sequence Length | Forward Rel Error | grad_transition Rel Error |
| --------------- | ----------------- | ------------------------- |
| T=1,000 | 1.1e-6 | 7.9e-4 |
| T=10,000 | 6.2e-7 | 6.7e-3 |

Error grows roughly linearly with T due to accumulated floating-point rounding differences between:

1. Triton parallel reduction vs Python sequential reduction
2. Floating-point non-associativity: `(a+b)+c ≠ a+(b+c)`

**Reference comparison notes:**

- Both Triton and PyTorch reference now use per-checkpoint log normalization
- Partitions and gradients should agree at all sequence lengths
- Small differences arise from parallel vs sequential floating-point reduction
- Use gradient_check.py to validate agreement at any T

### Known Limitation: Non-Determinism in Backward Pass

**Status:** Known limitation, accepted. Full determinism (two-phase kernel) is documented in the plan file for future implementation if needed.

**Behavior:** The Triton backward kernel produces slightly different gradients across runs due to `tl.atomic_add` operations. Multiple (t, k) iterations write to the same workspace entries, and GPU memory reordering combined with floating-point non-associativity causes variance.

**Magnitude:**

- `grad_transition`: ~1-2% relative difference across runs
- `grad_duration_bias`: ~7-14% relative difference across runs
- `grad_cum_scores`: Position-dependent, lower variance

**Impact on training:**

- Gradients are *correct* (match PyTorch reference within tolerance)
- Training will *converge* (variance is within normal SGD noise)
- Run-to-run reproducibility is *not guaranteed*

**When this matters:**

- Focused learning approaches (curriculum learning, hard example mining)
- Ablation studies requiring exact reproducibility
- Production model parity across retraining

**Partial mitigation (implemented):**

Per-segment workspace allocation eliminates cross-segment atomic contention. Each checkpoint segment writes to its own workspace slice, and the host performs a deterministic sum over segments before the final einsum reduction. This reduces variance but does not eliminate within-segment non-determinism.

**Full mitigation (documented, not implemented):**

The plan file documents a "Part D: Two-Phase Kernel" approach that would eliminate all atomics for `grad_transition` and `grad_duration_bias` by:

1. Storing full beta history per segment (~1 MB additional memory)
2. Separating gradient accumulation into a second pass with local register accumulators
3. Writing once per (segment, tile, tile) with no atomics

This would add ~1.5-2× compute overhead for marginal recomputation.

**Verification:**

```bash
# Test correctness (should pass)
python benchmarks/practical_demonstration/synthetic_genomics/gradient_check.py

# Test determinism (will show remaining variance)
python scripts/find_determinism.py
```
