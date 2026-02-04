# Debug Print Mask for Triton Backward Kernel

This document preserves the debug print "mask" used to debug the backward kernel. These prints were instrumental in finding and fixing the segment isolation and memory barrier bugs. They can be quickly added back for future debugging.

## Overview

The debug prints follow a hierarchical strategy:
1. **Checkpoint loading**: Verify checkpoint values being loaded from ring buffer
2. **Store verification**: Confirm values written to alpha_buffer
3. **Memory barrier verification**: Check all warps see consistent values after barrier
4. **Alpha recomputation**: Track forward pass recreation within segments
5. **Beta loading**: Verify beta values from ring buffer
6. **Segment boundary**: Compare normalization scales across segment boundaries
7. **Marginal computation**: Detailed breakdown of marginal calculation

All prints are **conditional on `batch_idx == 0`** to avoid overwhelming output from multi-batch runs. Most are further filtered to specific positions (e.g., `t == 1`, `ckpt_idx == 0`) to focus on critical test cases.

## Critical Principle: Scalar Reduction

**IMPORTANT**: Triton's `tl.device_print()` can only print scalars. To print tensor values, you must reduce them to scalars using:
- `tl.sum(tensor.to(tl.float32))` - Sum of all elements
- `tl.sum(tl.where(c_idx == 0, tensor, 0.0))` - Extract single element (e.g., c=0)
- `tl.sum((tensor < threshold).to(tl.int32))` - Count elements matching condition

## Debug Print Locations

### 1. Checkpoint Loading (Lines ~347-351)

**Purpose**: Verify checkpoint values being loaded from ring buffer

**Location**: Inside checkpoint loading loop, after loading `alpha_val`

**Condition**: `batch_idx == 0 and ckpt_idx == 0 and k_slot < 3`

**Code**:
```python
# DEBUG: Print checkpoint values for segment 0
if batch_idx == 0 and ckpt_idx == 0 and k_slot < 3:
    alpha_val_sum = tl.sum(alpha_val.to(tl.float32))
    tl.device_print("=== CHECKPOINT 0 DEBUG ===")
    tl.device_print("k_slot=:", k_slot)
    tl.device_print("alpha_val_sum=:", alpha_val_sum)
```

**What it shows**:
- Which k_slot is being loaded (0, 1, 2)
- Sum of alpha values for that slot
- Helps detect if checkpoint contains NEG_INF (sum would be very negative)

**When to use**: When suspecting checkpoint loading issues or ring buffer corruption

---

### 2. Alpha Buffer Store (Lines ~357-363)

**Purpose**: Verify what's being stored to alpha_buffer before the store operation

**Location**: Inside checkpoint loading loop, before `tl.store()` to alpha_buffer

**Condition**: `batch_idx == 0 and ckpt_idx == 0` (inside `k_slot == seg_start % K` block)

**Code**:
```python
# DEBUG: For segment 0, verify what we're about to store
if batch_idx == 0 and ckpt_idx == 0:
    stored_sum = tl.sum(alpha_val.to(tl.float32))
    seg_offset_bytes = ckpt_idx * stride_ab_seg
    tl.device_print("===SEG0 STORE===")
    tl.device_print("Storing to seg=0, t=0")
    tl.device_print("seg_offset_bytes=:", seg_offset_bytes)
    tl.device_print("Storing sum=:", stored_sum)
```

**What it shows**:
- Segment offset being used (should be 0 for ckpt_idx=0)
- Sum of values being stored
- Confirms store is happening with expected data

**When to use**: When debugging buffer indexing or segment isolation issues

---

### 3. Memory Barrier Verification (Lines ~376-393)

**Purpose**: **CRITICAL** - Verify all warps see consistent values after memory barrier

**Location**: Immediately after `tl.debug_barrier()`, after checkpoint loading loop

**Condition**: `batch_idx == 0 and ckpt_idx == 0`

**Code**:
```python
# DEBUG: Verify what's in the buffer after checkpoint loading (ONLY segment 0)
if batch_idx == 0 and ckpt_idx == 0:
    # Load immediately after store to verify
    verify_load = tl.load(
        alpha_buf_seg + 0 * stride_ab_t + c_idx * stride_ab_c,
        mask=c_mask,
        other=NEG_INF,
    )
    verify_sum = tl.sum(verify_load.to(tl.float32))
    verify_first = tl.sum(tl.where(c_idx == 0, verify_load, 0.0))
    neginf_count = tl.sum((verify_load < (NEG_INF + 1.0)).to(tl.int32))
    seg_offset_bytes = ckpt_idx * stride_ab_seg

    tl.device_print("===SEG0 VERIFY===")
    tl.device_print("Loading from seg=0, t=0")
    tl.device_print("seg_offset_bytes=:", seg_offset_bytes)
    tl.device_print("Loaded sum=:", verify_sum)
    tl.device_print("Loaded[c=0]=:", verify_first)
    tl.device_print("NEG_INF count=:", neginf_count)
```

**What it shows**:
- **CRITICAL**: How many threads see the stored value vs NEG_INF
- If you see different values from different threads (e.g., 32 threads see 0.0, 96 threads see NEG_INF), you have a memory visibility bug
- After memory barrier fix, should see all 128 threads with same value

**When to use**: When debugging warp-level memory coherency issues. This is the **store-verify pattern** documented in DEBUGGING_GUIDE.md.

---

### 4. Alpha Previous Load Debug (Lines ~411-423)

**Purpose**: Verify alpha_prev loading during alpha recomputation

**Location**: Inside alpha recomputation loop, before loading alpha_prev

**Condition**: `batch_idx == 0 and t == 1 and k == 1`

**Code**:
```python
# DEBUG: Print what we're about to load
if batch_idx == 0 and t == 1 and k == 1:
    tl.device_print("=== ABOUT TO LOAD ALPHA_PREV ===")
    tl.device_print("t=:", t)
    tl.device_print("k=:", k)
    tl.device_print("start_pos=:", start_pos)
    tl.device_print("local_start=:", local_start)
    tl.device_print("seg_start=:", seg_start)
    tl.device_print("ckpt_idx=:", ckpt_idx)
    # Check the buffer offset we're about to use
    seg_offset = ckpt_idx * stride_ab_seg
    t_offset = local_start * stride_ab_t
    tl.device_print("seg_offset=:", seg_offset)
    tl.device_print("t_offset=:", t_offset)
```

**What it shows**:
- Position (t, k) being computed
- Mapping from t to local_start (position within segment)
- Buffer offsets being used
- Helps verify indexing logic for alpha_buffer reads

**When to use**: When debugging alpha buffer indexing or local_t mapping issues

---

### 5. Alpha Previous Loaded Verification (Lines ~434-442)

**Purpose**: Verify alpha_prev after loading

**Location**: Inside alpha recomputation loop, after loading alpha_prev

**Condition**: `batch_idx == 0 and t == 1 and k == 1`

**Code**:
```python
# DEBUG: Verify what we loaded
if batch_idx == 0 and t == 1 and k == 1:
    alpha_prev_sum = tl.sum(alpha_prev.to(tl.float32))
    alpha_prev_first = tl.sum(tl.where(c_idx == 0, alpha_prev, 0.0))
    neginf_count = tl.sum((alpha_prev < (NEG_INF + 1.0)).to(tl.int32))
    tl.device_print("=== LOADED ALPHA_PREV ===")
    tl.device_print("alpha_prev_sum=:", alpha_prev_sum)
    tl.device_print("alpha_prev[c=0]=:", alpha_prev_first)
    tl.device_print("NEG_INF count=:", neginf_count)
```

**What it shows**:
- Sum of loaded alpha_prev
- Specific element (c=0) for comparison
- Count of NEG_INF values (should be 0 for valid positions)

**When to use**: Pair with "ABOUT TO LOAD" print to verify load succeeded

---

### 6. Alpha Recomputation Intermediate Values (Lines ~487-498)

**Purpose**: Detailed breakdown of alpha recomputation for specific position

**Location**: Inside alpha recomputation loop, after computing segment_score

**Condition**: `batch_idx == 0 and t == 1 and k == 1`

**Code**:
```python
# DEBUG: Print intermediate values for t=1, k=1
if batch_idx == 0 and t == 1 and k == 1:
    alpha_prev_sum = tl.sum(alpha_prev.to(tl.float32))
    cum_end_sum = tl.sum(cum_end.to(tl.float32))
    cum_start_sum = tl.sum(cum_start.to(tl.float32))
    content_sum = tl.sum(content_score.to(tl.float32))
    dur_sum = tl.sum(dur_bias.to(tl.float32))
    seg_sum = tl.sum(segment_score.to(tl.float32))
    tl.device_print("=== BWD RECOMP t=1 k=1 ===")
    tl.device_print("alpha_prev_sum=:", alpha_prev_sum)
    tl.device_print("cum_end_sum=:", cum_end_sum)
    tl.device_print("cum_start_sum=:", cum_start_sum)
    tl.device_print("content_sum=:", content_sum)
    tl.device_print("dur_sum=:", dur_sum)
    tl.device_print("seg_sum=:", seg_sum)
```

**What it shows**:
- Every component of the alpha update equation
- Helps identify which term is wrong (alpha_prev, content, duration, transition)
- Compare with PyTorch reference to find divergence point

**When to use**: When forward and backward alpha diverge - this pinpoints which component is wrong

---

### 7. Alpha Recomputation Result (Lines ~578-583)

**Purpose**: Track computed alpha_t values during recomputation

**Location**: Inside alpha recomputation loop, after computing alpha_t

**Condition**: `batch_idx == 0 and ckpt_idx == 0 and t <= 2`

**Code**:
```python
# DEBUG: Print alpha_t after recomputation
if batch_idx == 0 and ckpt_idx == 0 and t <= 2:
    alpha_t_sum = tl.sum(alpha_t.to(tl.float32))
    tl.device_print("=== ALPHA RECOMP DEBUG ===")
    tl.device_print("t=:", t)
    tl.device_print("local_t=:", local_t)
    tl.device_print("alpha_t_sum=:", alpha_t_sum)
```

**What it shows**:
- Alpha values for t=0, 1, 2 in segment 0
- Progression of alpha through recomputation
- Compare with forward pass to verify correctness

**When to use**: When verifying segment-level alpha recomputation matches forward pass

---

### 8. Alpha Loading for Marginal Computation (Lines ~600-607)

**Purpose**: Verify alpha values loaded for marginal/gradient computation

**Location**: After loading alpha_t for marginal computation

**Condition**: `batch_idx == 0 and (t == 1 or t == 26 or t == 27)`

**Code**:
```python
# DEBUG: Print alpha at key positions
if batch_idx == 0 and (t == 1 or t == 26 or t == 27):
    alpha_sum = tl.sum(alpha_t.to(tl.float32))
    tl.device_print("=== ALPHA DEBUG ===")
    tl.device_print("t=:", t)
    tl.device_print("local_t=:", local_t)
    tl.device_print("ckpt_idx=:", ckpt_idx)
    tl.device_print("alpha_sum=:", alpha_sum)
    tl.device_print("log_norm_at_ckpt=:", log_norm_at_ckpt)
```

**What it shows**:
- Alpha at segment boundaries (t=26/27 is checkpoint boundary for interval=27)
- Log normalization factor for this segment
- Helps debug segment boundary issues

**When to use**: When segment 0 has different errors than segment 1

---

### 9. Beta Loading Debug (Lines ~650-659)

**Purpose**: Verify beta values loaded from ring buffer

**Location**: Inside k-loop, after loading beta from ring buffer

**Condition**: `batch_idx == 0 and t == 1 and k == 1`

**Code**:
```python
# DEBUG: Print beta values
if batch_idx == 0 and t == 1 and k == 1:
    beta_check = tl.load(
        beta_ring_base + end_ring_idx * stride_br_k + c_idx * stride_br_c,
        mask=c_mask,
        other=NEG_INF,
    )
    beta_sum = tl.sum(beta_check.to(tl.float32))
    tl.device_print("=== BETA DEBUG t=:", t)
    tl.device_print("end_pos=:", end_pos)
    tl.device_print("end_ring_idx=:", end_ring_idx)
    tl.device_print("beta_sum=:", beta_sum)
```

**What it shows**:
- Beta ring buffer indexing (end_pos → end_ring_idx)
- Beta values being loaded
- Helps verify ring buffer wrapping logic

**When to use**: When suspecting beta ring buffer corruption or indexing issues

---

### 10. Segment Boundary Debug - Segment 0 (Lines ~811-819)

**Purpose**: Compare normalization scales at segment boundaries

**Location**: Inside marginal computation, after computing scale

**Condition**: `batch_idx == 0 and t == 26 and k == 1`

**Code**:
```python
# DEBUG: Segment boundary comparison
if batch_idx == 0 and t == 26 and k == 1:
    # Segment 0, needs beta[27] from segment 1
    tl.device_print("=== SEGMENT BOUNDARY DEBUG ===")
    tl.device_print("t=26 k=1: end_pos=", end_pos)
    tl.device_print("log_norm_at_ckpt=", log_norm_at_ckpt)
    tl.device_print("global_max=", global_max)
    tl.device_print("log_Z=", log_Z)
    tl.device_print("log_scale=", log_scale)
    tl.device_print("scale=", scale)
```

**What it shows**:
- Normalization parameters for position in segment 0 that uses beta from segment 1
- Helps detect scale mismatches between segments

**When to use**: When segment 0 has different error patterns than segment 1

---

### 11. Segment Boundary Debug - Segment 1 (Lines ~821-829)

**Purpose**: Reference normalization scales for segment 1

**Location**: Inside marginal computation, after computing scale

**Condition**: `batch_idx == 0 and t == 27 and k == 1`

**Code**:
```python
# DEBUG: Segment 1 reference
if batch_idx == 0 and t == 27 and k == 1:
    # Segment 1, computes beta[28]
    tl.device_print("=== SEGMENT 1 REFERENCE ===")
    tl.device_print("t=27 k=1: end_pos=", end_pos)
    tl.device_print("log_norm_at_ckpt=", log_norm_at_ckpt)
    tl.device_print("global_max=", global_max)
    tl.device_print("log_Z=", log_Z)
    tl.device_print("log_scale=", log_scale)
    tl.device_print("scale=", scale)
```

**What it shows**:
- Normalization parameters for adjacent position in segment 1
- Compare with segment 0 values to find scale mismatch

**When to use**: Pair with segment 0 print to compare normalization across boundaries

---

### 12. Detailed Marginal Breakdown (Lines ~961-983)

**Purpose**: Complete breakdown of marginal computation for specific position

**Location**: Inside marginal computation, after computing final marginal

**Condition**: `batch_idx == 0 and t == 1 and k == 1`

**Code**:
```python
# DEBUG: Detailed marginal breakdown for t=1, k=1
if batch_idx == 0 and t == 1 and k == 1:
    # Extract [0, 0] element from each tensor for detailed trace
    alpha_0 = tl.sum(tl.where((c_src_idx == 0) & (c_idx == 0), alpha_t_clamped, 0.0))
    beta_0 = tl.sum(tl.where(c_dst_idx == 0, beta_tile, 0.0))
    edge_00 = tl.sum(tl.where((c_dst_idx == 0) & (c_idx == 0), edge_tile_clamped, 0.0))
    log_joint_00 = alpha_0 + edge_00 + beta_0
    log_marginal_rel_00 = log_joint_00 - global_max_safe_0
    marginal_unnorm_00 = tl.exp(log_marginal_rel_00)
    marginal_final_00 = marginal_unnorm_00 * scale_0

    # Print extracted values
    tl.device_print("=== MARGINAL DEBUG t=:", t)
    tl.device_print("k=:", k)
    tl.device_print("alpha[0]=:", alpha_0)
    tl.device_print("beta[0]=:", beta_0)
    tl.device_print("edge[0,0]=:", edge_00)
    tl.device_print("log_joint[0,0]=:", log_joint_00)
    tl.device_print("log_joint-global_max=:", log_marginal_rel_00)
    tl.device_print("marginal_unnorm[0,0]=:", marginal_unnorm_00)
    tl.device_print("marginal_final[0,0]=:", marginal_final_00)
```

**What it shows**:
- **Every step** of marginal computation for a single (t, k, c_src, c_dst) tuple
- Helps identify which step produces wrong result
- Compare with PyTorch line-by-line to find divergence

**When to use**: When marginals are wrong but you don't know which component (alpha, beta, edge, scale) is the issue

---

## AWK Post-Processing Script

The debug output can be overwhelming (128 threads × many positions). Use the AWK script from `scripts/parse_debug_output.sh` to:

1. **Deduplicate**: Show unique (value, count) pairs instead of 128 identical lines
2. **Group by section**: Organize output by debug marker (===CHECKPOINT===, ===STORE===, etc.)
3. **Verify fixes**: Automated checks for memory barrier (all threads agree) and alpha recomputation (forward == backward)

**Usage**:
```bash
python scripts/debug_marginal_divergence.py > debug.log 2>&1
bash scripts/parse_debug_output.sh debug.log
```

See `scripts/parse_debug_output.sh` for the full AWK implementation.

## Adding Back Debug Prints

To quickly add these prints back:

1. **For comprehensive debugging**: Add all prints (useful for unknown bugs)
2. **For specific hypothesis**: Add only relevant sections:
   - Memory barrier issue → prints #2, #3
   - Alpha recomputation → prints #4, #5, #6, #7
   - Segment boundary → prints #8, #11
   - Marginal computation → print #12

## Key Debugging Patterns from This Session

### Store-Verify Pattern
```python
# 1. Store operation
tl.store(buffer + offset, data, mask=mask)

# 2. Memory barrier (if needed)
tl.debug_barrier()

# 3. Verification load
if DEBUG:
    verify = tl.load(buffer + offset, mask=mask, other=SENTINEL)
    verify_sum = tl.sum(verify.to(tl.float32))
    tl.device_print("verify_sum=:", verify_sum)
```

### Comparison Pattern
```python
# Compare Triton value with expected reference
if DEBUG:
    triton_sum = tl.sum(triton_value.to(tl.float32))
    expected_sum = 17.9544  # From PyTorch or previous run
    tl.device_print("triton=:", triton_sum)
    # Then manually compare with expected_sum in output
```

### Hierarchical Debugging
1. **Level 1**: Checkpoint values (inputs to segment)
2. **Level 2**: Alpha recomputation (forward pass within segment)
3. **Level 3**: Beta loading (backward pass inputs)
4. **Level 4**: Marginal computation (final outputs)

Start at Level 1 and work down until you find divergence.

## References

- [DEBUGGING_GUIDE.md](/.claude/skills/triton/references/DEBUGGING_GUIDE.md) - General Triton debugging methodology
- [scripts/parse_debug_output.sh](/../scripts/parse_debug_output.sh) - AWK script for parsing output
- [scripts/debug_marginal_divergence.py](/../scripts/debug_marginal_divergence.py) - Test script that generated this output
