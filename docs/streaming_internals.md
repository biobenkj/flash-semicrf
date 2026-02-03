# Streaming Semi-CRF Internals

Technical reference for developers working with the streaming inference module.

## Overview

The streaming module (`torch_semimarkov.streaming`) provides memory-efficient Semi-CRF inference by computing edge potentials on-the-fly rather than materializing the full edge tensor.

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

### Module Structure

| File | Purpose |
|------|---------|
| [autograd.py](../src/torch_semimarkov/streaming/autograd.py) | Public API and autograd functions |
| [triton_forward.py](../src/torch_semimarkov/streaming/triton_forward.py) | Triton forward kernels (log/max semiring) |
| [triton_backward.py](../src/torch_semimarkov/streaming/triton_backward.py) | Triton backward kernel |
| [pytorch_reference.py](../src/torch_semimarkov/streaming/pytorch_reference.py) | Pure PyTorch reference implementation |
| [constants.py](../src/torch_semimarkov/streaming/constants.py) | Shared constants (NEG_INF) |

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
- When computing edges, we use `transition.T` to get `[c_dst, c_src]` layout

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

```python
alpha_ring: (batch, K, C)
```

Forward messages use a ring buffer indexed by `t % K`:
- `alpha_ring[:, t % K, :]` stores α values at position `t`
- Only the most recent `K` positions are kept
- Older values are overwritten as the scan progresses

### Why O(KC) Memory

Standard forward pass stores all α values: **O(T × C)** memory.

With max segment duration K, position `t` only depends on positions `[t-K+1, t-1]`. The ring buffer exploits this by storing only the K most recent values.

```
Time:     0   1   2   3   4   5   6   7   8   ...
Ring idx: 0   1   2   0   1   2   0   1   2   ...  (K=3)
          ↑           ↑           ↑
          overwritten by t=3, 6, 9, ...
```

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

From [triton_forward.py](../src/torch_semimarkov/streaming/triton_forward.py) (kernel implementation):

```python
# Content score via cumsum difference
cum_end = tl.load(cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c, ...)
cum_start = tl.load(cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c, ...)
content_score = cum_end - cum_start  # (C_PAD,)

# Add duration bias
dur_bias = tl.load(duration_bias_ptr + k * stride_db_k + c_idx * stride_db_c, ...)
segment_score = content_score + dur_bias  # (C_PAD,)

# Build edge block
edge_block = segment_score[:, None] + transition_block  # (C_PAD, C_PAD)
```

---

## Code-to-Math Correspondence

| Code Variable | Math Notation | Shape | Description |
|--------------|---------------|-------|-------------|
| `cum_scores[:, t, c]` | S_t,c | (batch, T+1, C) | Cumulative projected scores |
| `transition[c_src, c_dst]` | T_c',c | (C, C) | Label transitions (src → dst) |
| `duration_bias[k, c]` | B_k,c | (K, C) | Duration-specific label bias |
| `alpha_ring[:, t%K, :]` | α̃_t | (batch, C) | Log-forward messages (ring buffer) |
| `beta_ring[:, t%K, :]` | β̃_t | (batch, C) | Log-backward messages (ring buffer) |
| `ring_checkpoints[:, i, :, :]` | Ω_i | (batch, K, C) | Saved ring buffer state at checkpoint i |
| `log_Z` | log Z | (batch,) | Log partition function |

### Mathematical Notation

The edge potential (log-domain):
```
ψ̃(t, k, c, c') = (S_t,c - S_{t-k},c) + B_k,c + T_c',c
```

Forward recurrence:
```
α̃_t(c) = logsumexp_{k,c'} [ α̃_{t-k}(c') + ψ̃(t, k, c, c') ]
```

---

## Forward Pass Walkthrough

### Kernel Entry Point

From [triton_forward.py](../src/torch_semimarkov/streaming/triton_forward.py):

```python
@triton.jit
def semi_crf_streaming_scan_kernel(...):
```

### Initialization

```python
# Ring buffer initialized in launcher (triton_forward.py:640-641)
ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)
ring_buffer[:, 0, :C] = 0.0  # alpha[0, c] = 0.0 for all valid labels
```

### Main Loop Structure

```python
for t in tl.range(1, T + 1):
    active = t <= seq_len
    alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)  # Accumulator

    for k in tl.range(1, tl.maximum(K, 2)):
        k_valid = (k <= t) & (k <= K - 1)
        start_pos = t - k
        ring_k_idx = start_pos % K

        # Load alpha_prev from ring buffer
        alpha_prev = tl.load(ring_base + ring_k_idx * stride_ring_k + ...)

        # Compute edge on-the-fly (see above)

        # Accumulate via logsumexp
        scores = alpha_prev[None, :] + edge_block
        score_for_k = logsumexp(scores, axis=1)  # Over c_src
        alpha_t = logsumexp_2way(alpha_t, score_for_k)  # Over k

    # Store to ring buffer
    ring_t_idx = t % K
    tl.store(ring_base + ring_t_idx * stride_ring_k + ..., alpha_t, ...)
```

### Logsumexp Reduction Pattern

The kernel uses a two-step logsumexp:
1. **Over source labels** (c_src): `logsumexp(scores, axis=1)`
2. **Over durations** (k): Accumulated incrementally via `logsumexp_2way`

```python
# Stable logsumexp accumulation
max_alpha = tl.maximum(alpha_t, score_for_k)
alpha_t = max_alpha + tl.log(
    tl.exp(alpha_t - max_alpha) + tl.exp(score_for_k - max_alpha)
)
```

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
```

At each checkpoint position `i × S`, we save the entire ring buffer state.

### Checkpoint Interval Calculation

From [pytorch_reference.py:25-45](../src/torch_semimarkov/streaming/pytorch_reference.py#L25-L45):

```python
def _compute_checkpoint_interval(T: int, K: int) -> int:
    """Optimal interval minimizes total memory.

    Memory = (T/S) × K × C + S × C + K × C
    Taking d/dS = 0 gives S* = sqrt(T × K)
    """
    optimal = int(math.sqrt(T * K))
    return max(K, optimal)  # At least K
```

### Saving Checkpoints

From the forward kernel:

```python
should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
ckpt_idx = t // CHECKPOINT_INTERVAL
if should_checkpoint:
    for k_save in tl.range(0, K):
        ring_val = tl.load(ring_base + k_save * stride_ring_k + ...)
        tl.store(ring_ckpt_base + ckpt_idx * stride_ckpt_n + k_save * stride_ckpt_k + ...)
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
# At T=100k with ~14 checkpoints: 14 × batch × 4 bytes = 56 bytes/sequence
```

### Literature Grounding

This technique synthesizes established methods:

1. **Flash Attention (Dao et al., 2022)** — Online softmax with running max/sum normalization, correcting at tile boundaries. Our checkpoint normalization follows the same pattern.

2. **HMM Scaling (Rabiner, 1989)** — Classical forward-backward algorithms use "scaling coefficients" at each timestep to prevent underflow. We apply the same principle at checkpoint granularity.

3. **CTC (Graves et al., 2006)** — Connectionist Temporal Classification uses similar log-space normalization for numerical stability.

4. **Gradient Checkpointing (Chen et al., 2016)** — Saving intermediate state for memory-efficient backprop; we extend this to include normalization state.

### Implementation Details

**Forward kernel** ([triton_forward.py](../src/torch_semimarkov/streaming/triton_forward.py)):

- Checkpoint normalization block: lines 330-380
- Final partition computation with `accum_log_norm`: line ~405

**Backward kernel** ([triton_backward.py](../src/torch_semimarkov/streaming/triton_backward.py)):

- Load `log_norm_at_ckpt` at segment start: line ~305
- Add to marginal computation: line ~650

**Autograd** ([autograd.py](../src/torch_semimarkov/streaming/autograd.py)):

- Save `log_norm_checkpoints` in forward: `save_for_backward`
- Pass to backward kernel: backward launcher call

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

The test failure manifests as ~1e9 difference in partition values for variable-length batches.

#### The Solution: Active Sequence Masking

Both PyTorch and Triton implementations mask the `active` flag when computing normalization shifts:

```text
At checkpoint boundary (t % interval == 0):
  1. active = (t <= seq_len[b])
  2. alpha_for_norm = where(active, α_t, NEG_INF)
  3. shift = max(alpha_for_norm)
  4. Guard: if shift == NEG_INF → shift = 0  # Inactive sequence
  5. accum_log_norm += shift  # Only non-zero for active sequences
```

**Key insight**: For inactive sequences, `alpha_for_norm` is all `NEG_INF`, so `shift = 0` and `accum_log_norm` freezes at the correct value.

**Triton Implementation** ([triton_forward.py](../src/torch_semimarkov/streaming/triton_forward.py)):

```python
# 1. Find max alpha value for normalization (over valid C only)
#    For inactive sequences, use NEG_INF to produce shift=0
alpha_for_norm = tl.where(active & c_mask, alpha_t, NEG_INF)
max_val = tl.max(alpha_for_norm)

# Guard against all-NEG_INF case (inactive sequence or numerical issue)
is_all_neginf = max_val < (NEG_INF + 1.0)
shift = tl.where(is_all_neginf, 0.0, max_val)

# 2. Update cumulative normalization factor
#    shift is 0 for inactive sequences, so no phantom updates
accum_log_norm = accum_log_norm + shift

# 3. Normalize alpha and ring buffer
alpha_t = alpha_t - shift
# ... normalize all K ring buffer slots ...

# 4. Save checkpoint with active masking
save_mask = (ckpt_idx < NUM_CKPTS) & active & c_mask
tl.store(ring_ckpt_ptr + ..., alpha_t, mask=save_mask)
tl.store(log_norm_ckpt_ptr + ..., accum_log_norm, mask=(ckpt_idx < NUM_CKPTS) & active)
```

**PyTorch Implementation** ([pytorch_reference.py](../src/torch_semimarkov/streaming/pytorch_reference.py)):

```python
# Normalize at checkpoint boundary
active_mask = (t <= lengths)  # (batch,)

# Compute shift only for active sequences
alpha_for_norm = torch.where(
    active_mask[:, None],
    alpha_t,
    torch.full_like(alpha_t, NEG_INF)
)
shift = alpha_for_norm.max(dim=-1).values  # (batch,)

# Guard against all-NEG_INF (inactive sequences)
is_all_neginf = shift < (NEG_INF + 1.0)
shift = torch.where(is_all_neginf, torch.zeros_like(shift), shift)

# Update accum_log_norm (frozen for inactive sequences)
accum_log_norm = accum_log_norm + shift

# Apply normalization
alpha_t = alpha_t - shift[:, None]
ring_buffer = ring_buffer - shift[:, None, None]
```

#### Comparison with Flash Attention

Flash Attention explicitly states: "Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor)." Our approach extends the Flash Attention normalization pattern to handle variable-length batches by:

1. **Masking active sequences** in the shift computation
2. **Zeroing shifts for ended sequences** via the `is_all_neginf` guard
3. **Freezing `accum_log_norm`** once a sequence ends

This is a novel extension of the Flash Attention technique to the batched variable-length Semi-CRF setting.

---

## Backward Pass Walkthrough

### Two-Phase Algorithm

From [triton_backward.py](../src/torch_semimarkov/streaming/triton_backward.py):

The backward pass processes segments in reverse order:

```python
for ckpt_idx in range(NUM_CKPTS - 1, -1, -1):
    seg_start = ckpt_idx * CHECKPOINT_INTERVAL
    seg_end = min((ckpt_idx + 1) * CHECKPOINT_INTERVAL, T)

    # Phase 1: Recompute alpha from checkpoint
    # Phase 2: Compute beta backward + accumulate gradients
```

### Phase 1: Alpha Recomputation

Load ring buffer state from checkpoint, then recompute forward through the segment:

```python
# Load from checkpoint
for k_slot in tl.range(0, K):
    alpha_val = tl.load(ring_ckpt_base + ckpt_idx * stride_ckpt_n + k_slot * stride_ckpt_k + ...)
    if k_slot == seg_start % K:
        tl.store(alpha_buf_base + 0 * stride_ab_t + ..., alpha_val)

# Recompute alpha for positions seg_start+1 to seg_end
for local_t in tl.range(1, SEGMENT_SIZE):
    t = seg_start + local_t
    if t < seg_end and t < seq_len:
        # Same forward computation as main kernel
        ...
```

### Phase 2: Beta Backward + Gradients

Compute beta backward while accumulating gradients:

```python
for t_offset in tl.range(0, CHECKPOINT_INTERVAL):
    t = seg_end - 1 - t_offset
    if t >= seg_start and t < seq_len:
        alpha_t = tl.load(alpha_buf_base + local_t * stride_ab_t + ...)
        new_beta = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

        for k in tl.range(1, tl.maximum(K, 2)):
            end_pos = t + k
            if end_pos <= seq_len:
                beta_next = tl.load(beta_ring_base + (end_pos % K) * stride_br_k + ...)

                # Compute edge
                edge_block = ...

                # Compute marginal
                log_marginal = alpha_t[None, :] + edge_block + beta_next[:, None] - log_Z
                marginal = tl.exp(log_marginal)

                # Accumulate gradients
                ...

                # Update beta
                scores_for_beta = edge_block + beta_next[:, None]
                new_beta = logsumexp_2way(new_beta, logsumexp(scores_for_beta, axis=0))

        tl.store(beta_ring_base + (t % K) * stride_br_k + ..., new_beta)
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
   - Accumulated per-batch **inside** kernel, scaled by `grad_output` **after** via einsum

### The Einsum Pattern

From [autograd.py:139-146](../src/torch_semimarkov/streaming/autograd.py#L139-L146):

```python
# Per-batch: scale each batch element
scale = grad_output.view(batch, 1, 1)
grad_cum_scores = grad_cum_scores * scale

# Shared: weighted sum via einsum
# grad_θ = Σ_b[grad_output[b] × grad_per_batch[b]]
grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)
```

### Why This Matters

The difference matters when `grad_output` varies across batch elements (e.g., weighted losses, masked sequences):

```python
# Correct:
grad_θ = Σ_b[grad_output[b] × marginal_sum[b]]

# Wrong (buggy):
grad_θ = Σ_b[marginal_sum[b]] × Σ_b[grad_output[b]]
```

The einsum approach is also memory-efficient, avoiding large intermediate tensors.

---

## Numerical Considerations

### Float32 Requirement

Cumulative scores **must** be float32 for numerical stability at T > 100K:

```python
# In the kernel docstring warning:
# cum_scores MUST be float32 for numerical stability at T > 100K.
# Zero-centering before cumsum is critical to prevent precision loss.
```

### Zero-Centering

Before computing cumulative scores, zero-center the projected scores:

```python
projected = projected - projected.mean(dim=1, keepdim=True)
cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
cum_scores[:, 1:, :] = torch.cumsum(projected.float(), dim=1)
```

This prevents cumulative scores from growing too large and losing precision.

### NEG_INF Masking

Invalid positions are masked with `NEG_INF = -1e9` (not `-inf` to avoid NaN in gradients):

```python
# From constants.py
NEG_INF = -1e9

# Masking pattern in kernel
alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)
```

### Forward-Backward Identity at Scale

At extreme sequence lengths (T ≥ 100k), accumulated floating-point error can violate the forward-backward identity `α[t] + β[t] ≈ log_Z`, causing marginal scale factors to overflow. This is solved by **per-checkpoint log normalization** — see the dedicated section [Per-Checkpoint Log Normalization](#per-checkpoint-log-normalization) for details.

---

## Performance Characteristics

### When Streaming Beats Pre-computed Edges

Memory bandwidth is the bottleneck, not compute:
- **Pre-computed**: Load O(T×K×C²) edge tensor from memory
- **Streaming**: Compute edges from O(T×C) cumulative scores

For large K, streaming is faster even with the extra computation.

### Benchmark Comparison (NVIDIA L40S)

| Configuration | triton_scan (pre-computed) | streaming | Advantage |
|---------------|---------------------------|-----------|-----------|
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
   - Not zero-centering before cumsum
   - Using float16 for cum_scores
   - At T ≥ 100k: forward-backward identity violation (see [Per-Checkpoint Log Normalization](#per-checkpoint-log-normalization))

3. **Wrong partition value**: Check that `cum_scores[:, 0, :]` is all zeros (the boundary condition)

### Verification Against Reference

Use the PyTorch reference for debugging:

```python
from torch_semimarkov.streaming.pytorch_reference import semi_crf_streaming_forward_pytorch

# Compare results (both return 4 values: partition, ring_ckpts, interval, log_norm_ckpts)
partition_triton, _, _, _ = launch_streaming_triton_kernel(...)
partition_ref, _, _, _ = semi_crf_streaming_forward_pytorch(...)

assert torch.allclose(partition_triton, partition_ref, atol=1e-4)
```

### Gradient Agreement Testing

A dedicated test script at `benchmarks/practical_demonstration/synthetic_genomics/gradient_check.py` compares Triton kernel gradients vs PyTorch reference at the low-level API.

**Key insight**: Both implementations now use log normalization checkpointing for numerical stability at extreme sequence lengths (T>100K). The PyTorch reference was updated to match the Triton kernel's Flash Attention-style checkpoint normalization pattern.

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
python src/torch_semimarkov/streaming/find_determinism.py
```
