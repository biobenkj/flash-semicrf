# Debug Print Mask for Triton Forward Kernel

This document preserves the debug print "mask" used to debug the forward kernel. These prints were used to compare forward pass values with backward recomputation, helping identify the memory barrier and segment isolation bugs.

## Overview

The forward kernel has simpler debug prints compared to backward, focusing on:
1. **Alpha computation details**: Intermediate values for specific (t, k) pairs
2. **Alpha verification**: Final alpha values at key positions

All prints are **conditional on `batch_idx == 0`** to avoid overwhelming output from multi-batch runs.

## Critical Principle: Scalar Reduction

**IMPORTANT**: Triton's `tl.device_print()` can only print scalars. To print tensor values, you must reduce them to scalars using:
- `tl.sum(tensor.to(tl.float32))` - Sum of all elements
- `tl.sum(tl.where(c_idx == 0, tensor, 0.0))` - Extract single element (e.g., c=0)
- `tl.sum((tensor < threshold).to(tl.int32))` - Count elements matching condition

## Debug Print Locations

### 1. Forward Pass Intermediate Values (Lines ~262-275)

**Purpose**: Detailed breakdown of alpha computation for comparison with backward recomputation

**Location**: Inside forward pass loop, after computing segment_score

**Condition**: `batch_idx == 0 and t == 1 and k == 1`

**Code**:
```python
# DEBUG: Compare forward with backward recomputation at t=1, k=1
if batch_idx == 0 and t == 1 and k == 1:
    alpha_prev_sum = tl.sum(alpha_prev.to(tl.float32))
    cum_end_sum = tl.sum(cum_end.to(tl.float32))
    cum_start_sum = tl.sum(cum_start.to(tl.float32))
    content_sum = tl.sum(content_score.to(tl.float32))
    dur_sum = tl.sum(dur_bias.to(tl.float32))
    seg_sum = tl.sum(segment_score.to(tl.float32))
    tl.device_print("=== FWD PASS t=1 k=1 ===")
    tl.device_print("alpha_prev_sum=:", alpha_prev_sum)
    tl.device_print("cum_end_sum=:", cum_end_sum)
    tl.device_print("cum_start_sum=:", cum_start_sum)
    tl.device_print("content_sum=:", content_sum)
    tl.device_print("dur_sum=:", dur_sum)
    tl.device_print("seg_sum=:", seg_sum)
```

**What it shows**:
- Every component of the alpha update equation for t=1, k=1
- alpha_prev (previous state)
- cum_end, cum_start (cumulative scores)
- content_score (cum_end - cum_start)
- dur_bias (duration bias)
- segment_score (content + duration)

**Comparison with backward**:
- Forward prints "=== FWD PASS t=1 k=1 ==="
- Backward prints "=== BWD RECOMP t=1 k=1 ===" (lines ~492-498 in triton_backward.py)
- Values should match exactly if recomputation is correct

**When to use**: When backward alpha recomputation doesn't match forward pass. Add this print in forward and the corresponding one in backward, then compare values to find where they diverge.

---

### 2. Forward Alpha Verification (Lines ~348-352)

**Purpose**: Track final alpha values at key positions for comparison with backward

**Location**: After computing alpha_t for position t

**Condition**: `batch_idx == 0 and (t == 1 or t == 26 or t == 27)`

**Code**:
```python
# DEBUG: Print forward alpha for comparison with backward recomputation
if batch_idx == 0 and (t == 1 or t == 26 or t == 27):
    alpha_t_sum = tl.sum(alpha_t.to(tl.float32))
    tl.device_print("=== FWD ALPHA DEBUG ===")
    tl.device_print("t=:", t)
    tl.device_print("alpha_t_sum=:", alpha_t_sum)
```

**What it shows**:
- Final alpha[t] values after logsumexp over k
- Positions t=1, 26, 27 chosen because:
  - t=1: First position after initial state
  - t=26/27: Checkpoint boundary (for interval=27)

**Comparison with backward**:
- Forward prints "=== FWD ALPHA DEBUG ==="
- Backward prints "=== ALPHA DEBUG ===" (lines ~602-607 in triton_backward.py)
- Values should match if:
  1. Checkpoints are saved correctly
  2. Backward recomputation logic is correct
  3. Segment isolation is working

**When to use**: When verifying that backward recomputation from checkpoints produces the same alpha values as forward pass. This was critical for debugging the memory barrier bug.

**Example from debugging session**:
```
# Forward output
=== FWD ALPHA DEBUG ===
t=: 1
alpha_t_sum=: 17.9544

# Backward output (BEFORE fix)
=== ALPHA DEBUG ===
t=: 1
alpha_t_sum=: 11.7400  ← WRONG! Different by -6.21

# Backward output (AFTER fix)
=== ALPHA DEBUG ===
t=: 1
alpha_t_sum=: 17.9544  ← CORRECT! Matches forward
```

---

### 3. Commented-Out Partition Function Prints (Lines ~465-466)

**Purpose**: Debug partition function computation (currently disabled)

**Location**: After computing final log_Z

**Condition**: Currently commented out

**Code**:
```python
# DEBUG: Print final partition function
# if batch_idx == 0:
#     tl.device_print("FWD log_Z=", partition)
#     tl.device_print("FWD max_final_alpha=", max_val)
```

**What it shows**:
- Final log partition function (log_Z)
- Maximum final alpha value before partition computation

**When to use**: Uncomment if debugging partition function issues or if forward log_Z doesn't match backward log_Z.

---

## AWK Post-Processing Script

Use the same AWK script as backward: `scripts/parse_debug_output.sh`

**Usage**:
```bash
python scripts/debug_marginal_divergence.py > debug.log 2>&1
bash scripts/parse_debug_output.sh debug.log
```

The script will:
1. Deduplicate repeated values (128 threads → unique values + counts)
2. Group output by debug markers
3. Compare forward vs backward alpha values automatically

## Adding Back Debug Prints

To quickly add these prints back:

1. **For forward/backward comparison**: Add prints #1 and #2
2. **For partition function debugging**: Uncomment print #3

## Key Debugging Pattern: Forward/Backward Comparison

When backward recomputation doesn't match forward:

1. **Add detailed component prints** (print #1 in both forward and backward)
   - Compare every component: alpha_prev, content, duration, etc.
   - Find which component diverges first

2. **Add alpha verification prints** (print #2 in both forward and backward)
   - Compare final alpha[t] values
   - Check multiple positions (start, middle, segment boundaries)

3. **Run with both prints active**:
   ```bash
   python scripts/debug_marginal_divergence.py > debug.log 2>&1
   bash scripts/parse_debug_output.sh debug.log
   ```

4. **Look for "FIX VERIFICATION SUMMARY" section**:
   ```
   CHECK 2: Alpha Recomputation
   Position t=1:
     Forward: 17.9544
     Backward: 17.9544
     Difference: 0.0000
   ✓ PASS: Forward and backward alpha values match exactly
   ```

## Example Debugging Session Output

```
=== FWD PASS t=1 k=1 ===
alpha_prev_sum=: 0.000000 (×128 threads)
cum_end_sum=: 225.123000 (×128 threads)
cum_start_sum=: 207.231000 (×128 threads)
content_sum=: 17.892000 (×128 threads)
dur_sum=: 15.234000 (×128 threads)
seg_sum=: 33.126000 (×128 threads)

=== BWD RECOMP t=1 k=1 ===
alpha_prev_sum=: 0.000000 (×128 threads)
cum_end_sum=: 225.123000 (×128 threads)
cum_start_sum=: 207.231000 (×128 threads)
content_sum=: 17.892000 (×128 threads)
dur_sum=: 15.234000 (×128 threads)
seg_sum=: 33.126000 (×128 threads)

→ All components match! ✓

=== FWD ALPHA DEBUG ===
t=: 1
alpha_t_sum=: 17.9544 (×128 threads)

=== ALPHA DEBUG ===
t=: 1
alpha_t_sum=: 17.9544 (×128 threads)

→ Final alpha matches! ✓
```

If values differ, the divergence point shows which component is wrong.

## Relationship with Backward Debug Prints

These forward prints are designed to pair with backward prints:

| Forward Print | Backward Print | Purpose |
|---------------|----------------|---------|
| FWD PASS t=1 k=1 (lines 262-275) | BWD RECOMP t=1 k=1 (backward lines 492-498) | Component-level comparison |
| FWD ALPHA DEBUG (lines 348-352) | ALPHA DEBUG (backward lines 602-607) | Final alpha comparison |

Always use them together when debugging recomputation issues.

## References

- [BACKWARD_DEBUG_PRINTS.md](./BACKWARD_DEBUG_PRINTS.md) - Backward kernel debug prints
- [DEBUGGING_GUIDE.md](/.claude/skills/triton/references/DEBUGGING_GUIDE.md) - General Triton debugging methodology
- [scripts/parse_debug_output.sh](/../scripts/parse_debug_output.sh) - AWK script for parsing output
- [scripts/debug_marginal_divergence.py](/../scripts/debug_marginal_divergence.py) - Test script that uses these prints
