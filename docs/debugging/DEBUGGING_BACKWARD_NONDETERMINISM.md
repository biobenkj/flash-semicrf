# Debugging Journey: Backward Pass Non-Determinism

**Date**: February 2026
**Duration**: ~4 hours across multiple sessions
**Severity**: Critical - 36% run-to-run differences in gradients, making training unstable
**Root Causes**: (1) Missing memory barrier after beta store, (2) Debug script comparison bug
**Status**: ✅ RESOLVED

---

## Executive Summary

This document chronicles the debugging journey to resolve non-determinism in the Triton backward kernel. The bug manifested as run-to-run variations of up to 36% in gradient values, with identical inputs producing different outputs on consecutive runs.

**Key Discoveries**:
1. **Beta ring buffer memory visibility**: The second tile (c=4-7) at position t=9 was reading stale NEG_INF values instead of beta values stored at t=10 by the first tile
2. **Three different global_max values** across runs from the same (t, k) position indicated a race condition
3. **Debug script comparison bug**: PyTorch returns per-batch gradients `(batch, C, C)`, Triton returns reduced gradients `(C, C)` - direct comparison was invalid

**Resolution Impact**: All runs now produce identical results, gradients match PyTorch within 2e-5.

---

## Table of Contents

1. [Initial Symptoms](#initial-symptoms)
2. [The Debugging Journey](#the-debugging-journey)
   - [Phase 1: Initial Investigation - Float64 Hypothesis](#phase-1-initial-investigation---float64-hypothesis)
   - [Phase 2: Debug Print Instrumentation](#phase-2-debug-print-instrumentation)
   - [Phase 3: The Breakthrough - Memory Visibility](#phase-3-the-breakthrough---memory-visibility)
   - [Phase 4: Gradient Correctness Investigation](#phase-4-gradient-correctness-investigation)
3. [Technical Deep Dive](#technical-deep-dive)
4. [Debugging Strategies That Worked](#debugging-strategies-that-worked)
5. [Lessons Learned](#lessons-learned)
6. [Prevention Strategies](#prevention-strategies)

---

## Initial Symptoms

### The Error Pattern

Running `find_determinism.py` revealed severe non-determinism in the backward pass:

```
=== Backward Pass Testing ===
T=20, C=8, K=10:
  rel_diff_cs: 348.26%
  rel_diff_tr: 335.92%
  rel_diff_db: 356.67%

[ASSESSMENT] FAIL - Severe non-determinism detected
```

### What Was Strange

1. **Forward pass was deterministic**: All forward configurations passed
2. **Large, erratic variations**: Not small floating-point noise, but massive 36%+ differences
3. **Same inputs, different outputs**: Identical random seed, yet different gradient values
4. **Consistent failure pattern**: The same configurations always failed (T=20,C=8,K=10 etc.)

### Initial Confusion

The error pattern suggested:
- Not a numerical precision issue (too large, too erratic)
- Not an algorithm bug (would be consistent)
- Something about execution timing or memory visibility
- Possibly race condition in tiled computation

---

## The Debugging Journey

### Phase 1: Initial Investigation - Float64 Hypothesis

**Theory**: The `tl.sum` and `tl.exp` operations might be accumulating floating-point errors that compound differently on each run.

**The Fix Attempt**: Cast to float64 before reduction operations:

```python
# Cast to float64 for deterministic accumulation
tile_sum_exp = tl.sum(tile_exp.to(tl.float64), axis=0)
```

**Result**: ❌ Made things worse! The non-determinism persisted and added unnecessary computation.

**Lesson**: Float64 doesn't help with race conditions; it only helps with numerical precision.

**Action**: Reverted the float64 changes.

### Phase 2: Debug Print Instrumentation

**Strategy**: Add comprehensive debug prints to trace values through the kernel and identify where non-determinism enters.

**Debug Points Added**:

1. **Beta Store (t=10)**:
```python
if batch_idx == 0 and t == 10:
    new_beta_sum = tl.sum(new_beta.to(tl.float32))
    tl.device_print("=== BETA STORE t=10 ===")
    tl.device_print("new_beta_sum=:", new_beta_sum)
    tl.device_print("t_ring_idx=:", t_ring_idx)
```

2. **Beta Load (t=9, k=1)**:
```python
if batch_idx == 0 and t == 9 and k == 1:
    beta_tile_sum = tl.sum(beta_tile.to(tl.float32))
    tl.device_print("=== BETA LOAD t=9 k=1 ===")
    tl.device_print("tile_start=:", c_dst_tile_start)
    tl.device_print("beta_tile_sum=:", beta_tile_sum)
    tl.device_print("end_ring_idx=:", end_ring_idx)
```

3. **Pass 1 Statistics**:
```python
if batch_idx == 0 and t == 9 and k == 1:
    tl.device_print("=== PASS1 DEBUG t=9 k=1 ===")
    tl.device_print("global_max=:", global_max)
    tl.device_print("global_sum_exp=:", global_sum_exp)
    tl.device_print("log_scale=:", log_scale)
    tl.device_print("scale=:", scale)
```

4. **Local Accumulators**:
```python
if batch_idx == 0 and t == 9:
    grad_cs_sum = tl.sum(grad_cs_t_local.to(tl.float32))
    tl.device_print("=== GRAD_CS_LOCAL t=9 ===")
    tl.device_print("grad_cs_t_local_sum=:", grad_cs_sum)
```

**Created Supporting Scripts**:
- `scripts/debug_backward_nondeterminism.py` - Runs failing config with PyTorch comparison
- `scripts/parse_debug_output.sh` - AWK script to deduplicate and analyze kernel output

### Phase 3: The Breakthrough - Memory Visibility

**Running the debug script on HPC revealed the smoking gun**:

```
=== BETA STORE (t=10) ===
new_beta_sum values (deduplicated):
  189.709089 (×5 runs, consistent!)

t_ring_idx values (deduplicated):
  0 (×5 runs, consistent!)

=== BETA LOAD (t=9, k=1) ===
tile_start values (deduplicated):
  0 (×5 runs)    ← First tile
  4 (×5 runs)    ← Second tile

beta_tile_sum values (deduplicated):
  tile_start=0: 74.3795 (correct!)
  tile_start=4: -4000000000.0 (NEG_INF!)  ← BUG!
```

**The Critical Pattern**:
- **First tile (c=0-3)**: Reads correct beta values (74.38)
- **Second tile (c=4-7)**: Reads NEG_INF (-4e9) - uninitialized memory!

**This caused different `global_max` values**:
```
=== PASS 1 GLOBAL STATISTICS ===
global_max values (deduplicated):
  57.7645 (×1 run)
  74.2859 (×2 runs)
  67.0252 (×2 runs)

→ THREE different values for the same (t, k) position!
```

**Root Cause Identified**: The beta values stored at t=10 were not visible to the second tile at t=9 due to missing memory barrier. Different runs had different timing, causing some runs to see stale values.

**The Fix**: Add `tl.debug_barrier()` after beta store:

```python
# Store new beta to ring buffer
tl.store(
    beta_ring_base + t_ring_idx * stride_br_k + c_idx * stride_br_c,
    new_beta,
    mask=c_mask,
)

# CRITICAL: Memory barrier ensures beta store is visible before next t iteration.
# Without this, the second tile (c=4-7) at t-1 may see stale NEG_INF values
# from ring buffer initialization instead of the values just stored at t.
# Debug analysis showed: tile_start=0 gets correct values, tile_start=4 gets NEG_INF.
tl.debug_barrier()
```

**Result**: ✅ Non-determinism FIXED!

```
=== PASS 1 GLOBAL STATISTICS ===
global_max values (deduplicated):
  57.743061 (×5 runs)  ← Single consistent value!

[PASS] All 5 runs produced identical results
```

### Phase 4: Gradient Correctness Investigation

**New Problem**: After fixing non-determinism, debug output showed apparent correctness issues:

```
=== Full Tensor Comparison vs PyTorch ===
  Run 0: max_diff_cs=4.947186e-06, max_diff_tr=6.550605e-01, max_diff_db=2.859037e+00
                                                       ↑ 65% error!        ↑ 286% error!
```

**Investigation**: Examined the PyTorch reference implementation:

```python
# PyTorch backward returns:
grad_transition: (batch, C, C)      # Per-batch gradients
grad_duration_bias: (batch, K, C)   # Per-batch gradients

# Triton backward returns:
grad_transition: (C, C)             # Reduced gradients (via einsum with grad_output)
grad_duration_bias: (K, C)          # Reduced gradients
```

**Root Cause**: The debug script was directly comparing tensors of different shapes!
- PyTorch: `(batch=2, C=8, C=8)` for grad_transition
- Triton: `(C=8, C=8)` for grad_transition
- NumPy/PyTorch broadcasting made the comparison "work" but gave wrong results

**The Fix**: Reduce PyTorch gradients before comparing:

```python
# IMPORTANT: PyTorch returns per-batch gradients (batch, C, C) and (batch, K, C)
# Triton returns reduced gradients (C, C) and (K, C) via einsum with grad_output
# Since grad_output = ones, the einsum is equivalent to sum over batch
pytorch_grad_tr_reduced = pytorch_grad_tr.sum(dim=0)  # (batch, C, C) -> (C, C)
pytorch_grad_db_reduced = pytorch_grad_db.sum(dim=0)  # (batch, K, C) -> (K, C)
```

**Final Result**: ✅ All gradients match!

```
=== Full Tensor Comparison vs PyTorch ===
  Run 0: max_diff_cs=4.947186e-06, max_diff_tr=4.470348e-06, max_diff_db=1.955032e-05
  Run 1: max_diff_cs=4.947186e-06, max_diff_tr=4.470348e-06, max_diff_db=1.955032e-05
  ... (all 5 runs identical)

[PASS] All 5 runs produced identical results
[PASS] Backward pass is deterministic
```

---

## Technical Deep Dive

### The Memory Visibility Bug

**GPU Memory Model**: In CUDA/Triton, stores from one thread/warp are NOT automatically visible to other threads without explicit synchronization.

**The Ring Buffer Pattern**:
```python
# Backward pass processes t in reverse order: T-1, T-2, ..., 1, 0
for t in reverse_range(T):
    for k in range(K):
        # Load beta[t+k] from ring buffer
        end_pos = t + k
        end_ring_idx = end_pos % (2 * K)

        # For each tile of c_dst dimension
        for c_dst_tile_start in tl.static_range(0, C_PAD, TILE_C):
            beta_tile = tl.load(beta_ring + end_ring_idx * stride + ...)
            # Use beta_tile for marginal computation...

    # Store updated beta[t] to ring buffer
    t_ring_idx = t % (2 * K)
    tl.store(beta_ring + t_ring_idx * stride + ..., new_beta)
    # ← MISSING BARRIER HERE caused the bug!
```

**Why Tile 0 Worked but Tile 1 Failed**:
- Both tiles read from the same ring buffer position
- Tile 0 happened to execute before tile 1 (scheduling luck)
- The store from previous t iteration completed in time for tile 0
- Tile 1 executed before the store was visible → read NEG_INF

**Why It Was Non-Deterministic**:
- GPU thread scheduling varies between runs
- Sometimes tile 1 got lucky and saw the store
- Sometimes it didn't → different global_max → different gradients

### The Tiled Two-Pass Algorithm

**The marginal computation uses two passes**:

```python
# Pass 1: Compute global statistics across all tiles
global_max = NEG_INF
global_sum_exp = 0.0

for tile in tiles:
    log_joint_tile = alpha + edge_tile + beta_tile
    tile_max = tl.max(log_joint_tile)

    # Online max update (Flash Attention pattern)
    new_global_max = max(global_max, tile_max)
    global_sum_exp = global_sum_exp * exp(global_max - new_global_max)
    global_sum_exp += sum(exp(log_joint_tile - new_global_max))
    global_max = new_global_max

# Pass 2: Use global statistics for all tiles
scale = exp(global_max + log_norm - log_Z)

for tile in tiles:
    marginal = exp(log_joint_tile - global_max) * scale
    # Accumulate gradients...
```

**Why Incorrect Beta Breaks Everything**:
- If tile 1's beta_tile contains NEG_INF, its `tile_max` is wrong
- `global_max` becomes wrong for that tile
- Scale computation uses wrong global_max
- All marginals and gradients are wrong

### Per-Batch vs Reduced Gradients

**PyTorch Design**: Returns gradients per batch element for flexibility:
```python
def backward(...):
    # grad_transition[b] = sum over (t, k) of marginal[b, t, k] * indicator
    return grad_cum_scores, grad_transition, grad_duration_bias
    #                       (batch, C, C)    (batch, K, C)
```

**Triton Design**: Uses einsum with grad_output for memory efficiency:
```python
# In kernel:
grad_tr_local = marginal * indicator  # Per-element contribution
grad_transition = einsum('b...,b->...', grad_tr_local, grad_output)
#                 Reduces batch dimension weighted by grad_output
```

**For grad_output = ones(batch)**: The einsum reduces to sum over batch dimension.

---

## Debugging Strategies That Worked

### 1. Hierarchical Debug Print Strategy

**Strategy**: Add prints at key checkpoints, not everywhere:

```
Level 1: Beta store (t=10)     ← Is the store happening correctly?
Level 2: Beta load (t=9, k=1)  ← Are we reading what was stored?
Level 3: Pass 1 statistics     ← Is global_max consistent?
Level 4: Scale computation     ← Is the normalization correct?
Level 5: Local accumulators    ← Are gradients accumulating correctly?
```

**Important Observation**: The bug revealed itself at Level 2 (beta load), showing tile 1 got wrong values.

### 2. Tile-Specific Analysis

**Pattern**: Print `c_dst_tile_start` along with values to identify tile-specific issues:

```python
tl.device_print("tile_start=:", c_dst_tile_start)
tl.device_print("beta_tile_sum=:", beta_tile_sum)
```

**What It Revealed**:
```
tile_start=0: beta=74.38 (correct)
tile_start=4: beta=-4e9  (NEG_INF = uninitialized)
```

### 3. AWK Post-Processing

**Problem**: Kernel prints from 128+ threads create overwhelming output.

**Solution**: AWK script to deduplicate and count occurrences:

```bash
gawk '
/=== BETA LOAD t=9 k=1 ===/ { in_beta_load = 1; next }
in_beta_load && /beta_tile_sum=:/ {
    match($0, /beta_tile_sum=: (-?[0-9.e+-]+)/, a)
    beta_load_vals[a[1]]++
}
END {
    print "beta_tile_sum values (deduplicated):"
    for (val in beta_load_vals) {
        printf "  %s (×%d threads)\n", val, beta_load_vals[val]
    }
}
' debug.log
```

**Result**: Instead of 1000 lines of raw output, see:
```
beta_tile_sum values (deduplicated):
  74.3795 (×64 threads)     ← Tile 0: correct
  -4000000000.0 (×64 threads)  ← Tile 1: wrong!
```

### 4. Cross-Run Comparison

**Key Diagnostic**: If values differ between runs with identical inputs, it's a race condition.

```
Run 1: global_max=57.76
Run 2: global_max=74.29
Run 3: global_max=67.03
Run 4: global_max=74.29
Run 5: global_max=57.76

→ Multiple values = non-determinism confirmed
```

### 5. Shape-Aware Tensor Comparison

**The mistake**: Directly comparing tensors without checking shapes:
```python
diff = (triton_grad_tr - pytorch_grad_tr).abs()  # Wrong! Different shapes!
```

**The fix**: Verify shapes and transform if needed:
```python
print(f"PyTorch shape: {pytorch_grad_tr.shape}")  # (batch, C, C)
print(f"Triton shape: {triton_grad_tr.shape}")    # (C, C)

pytorch_grad_tr_reduced = pytorch_grad_tr.sum(dim=0)  # Now same shape
diff = (triton_grad_tr - pytorch_grad_tr_reduced).abs()  # Correct!
```

---

## Lessons Learned

### 1. Memory Barriers Are Not Optional for Ring Buffers

**Lesson**: When writing to a ring buffer and reading from it in a tiled loop, add a memory barrier after the store.

**Pattern**:
```python
# Store to ring buffer
tl.store(ring_buffer + idx * stride + ..., value)

# MUST add barrier before any subsequent reads
tl.debug_barrier()

# Now reads will see the stored value
for tile in tiles:
    loaded = tl.load(ring_buffer + idx * stride + ...)  # Safe now
```

**Red Flag**: If tile 0 gets correct values but tile N gets NEG_INF/garbage, suspect missing barrier.

### 2. Non-Determinism Points to Race Conditions

**Lesson**: When identical inputs produce different outputs across runs, the bug is timing-related.

**Investigation Checklist**:
- [ ] Are there stores without subsequent barriers?
- [ ] Are different tiles/warps racing for shared memory?
- [ ] Are atomic operations being used correctly?
- [ ] Is there implicit ordering that isn't guaranteed?

### 3. Validate Debug Script Assumptions

**Lesson**: Debug scripts can have bugs too! Especially when comparing tensors of different shapes.

**Best Practice**: Always print shapes before comparison:
```python
print(f"Expected shape: {expected.shape}")
print(f"Actual shape: {actual.shape}")
assert expected.shape == actual.shape, "Shape mismatch!"
```

### 4. Per-Batch vs Reduced Gradient Semantics

**Lesson**: Different implementations may use different gradient conventions.

**Common Patterns**:
- PyTorch: Returns per-batch gradients for flexibility
- Triton/CUDA: Returns reduced gradients for memory efficiency
- einsum with grad_output: Implicitly reduces when grad_output is all-ones

**Documentation**: Always document gradient shapes in backward implementation.

### 5. The Value of Structured Debug Output

**Lesson**: Parsing kernel debug output manually is error-prone. Automate it.

**Best Practice**:
1. Use consistent markers: `=== SECTION NAME ===`
2. Use consistent format: `variable=: value`
3. Write AWK/grep scripts to parse and summarize
4. Check into repo for future debugging sessions

---

## Prevention Strategies

### 1. Ring Buffer Memory Barrier Checklist

**Add `tl.debug_barrier()` after**:
- [ ] Ring buffer stores that will be read in subsequent iterations
- [ ] Checkpoint stores that will be read by different tiles
- [ ] Any store where timing between store and load is not guaranteed

### 2. Tiled Computation Verification

**Test Strategy**: For tiled kernels, verify:
- [ ] All tiles get consistent input values
- [ ] Global statistics (max, sum) are same across tiles
- [ ] Results match non-tiled reference implementation

**Debug Print Pattern**:
```python
if DEBUG:
    tl.device_print("tile_start=:", tile_start)
    tl.device_print("input_sum=:", tl.sum(input_tile))
    # Should see same input_sum for same tile_start across runs
```

### 3. Gradient Shape Documentation

**Best Practice**: Document gradient shapes in function docstrings:

```python
def launch_streaming_triton_backward(...):
    """
    Returns:
        grad_cum_scores: (batch, T+1, C) - Per-element gradients
        grad_transition: (C, C) - REDUCED across batch via einsum
        grad_duration_bias: (K, C) - REDUCED across batch via einsum

    Note: PyTorch reference returns (batch, C, C) and (batch, K, C).
    To compare, reduce PyTorch outputs: pytorch_grad.sum(dim=0)
    """
```

### 4. Debug Script Validation

**Before trusting debug script output**:
- [ ] Print all tensor shapes
- [ ] Verify comparison is apples-to-apples
- [ ] Test with known-correct inputs
- [ ] Check for broadcasting behavior

---

## Files Modified

### Kernel Fix
**File**: `src/torch_semimarkov/streaming/triton_backward.py`

**Change**: Added memory barrier after beta store (lines 1106-1120):
```python
tl.store(
    beta_ring_base + t_ring_idx * stride_br_k + c_idx * stride_br_c,
    new_beta,
    mask=c_mask,
)

# CRITICAL: Memory barrier ensures beta store is visible before next t iteration.
tl.debug_barrier()
```

### Debug Script Fix
**File**: `scripts/debug_backward_nondeterminism.py`

**Change**: Reduce PyTorch gradients before comparison (lines 83-87):
```python
pytorch_grad_tr_reduced = pytorch_grad_tr.sum(dim=0)
pytorch_grad_db_reduced = pytorch_grad_db.sum(dim=0)
```

---

## Conclusion

This debugging session uncovered two critical issues:
1. **Missing memory barrier** after beta store caused tile 1 to read stale NEG_INF values
2. **Debug script comparison bug** made correct gradients appear wrong

Both bugs were systematic but manifested differently:
- Memory barrier bug: Non-determinism (timing-dependent race condition)
- Comparison bug: Consistent "errors" that were actually correct values compared wrong

**Final Status**: All runs produce identical results, all gradients match PyTorch within 2e-5.

**Key Takeaway**: In tiled GPU kernels with ring buffers, always add memory barriers after stores that will be read by subsequent loop iterations or different tiles. And always validate your validation scripts!

---

## References

- [DEBUGGING_SEGMENT_ISOLATION.md](DEBUGGING_SEGMENT_ISOLATION.md) - Related debugging journey (segment isolation bugs)
- [BACKWARD_DEBUG_PRINTS.md](BACKWARD_DEBUG_PRINTS.md) - Debug print restoration guide
- [parse_debug_output.sh](../../scripts/parse_debug_output.sh) - AWK parsing scripts
- [debug_backward_nondeterminism.py](../../scripts/debug_backward_nondeterminism.py) - Debug test script

---

**Document Status**: Complete
**Last Updated**: February 2026
**Author**: Debugging session with Claude Opus 4.5
