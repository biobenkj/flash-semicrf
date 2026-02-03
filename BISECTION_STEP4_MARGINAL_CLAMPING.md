# Bisection Step 4: Add Marginal > 1.0 Clamping

## Overview
This document describes how to add the marginal > 1.0 clamping from commit 07456c6.

**IMPORTANT**: This is the most suspicious change because it explicitly masks an underlying bug instead of fixing it.

## What is Marginal > 1.0 Clamping?

This adds a "defensive sanity check" that clamps any marginal values > 1.0 to 0.0. The commit comment admits:

> "Values > 1 indicate scale mismatch between forward and backward passes (e.g., at checkpoint boundaries). Clamping log_scale to ≤ 0 (line 753-755) usually prevents this, but we add this check for robustness."

**Translation**: There's a bug causing marginals > 1.0, and instead of fixing it, they're clamping invalid values to 0.

## Change Required

**Location**: After line 710 in [triton_backward.py](src/torch_semimarkov/streaming/triton_backward.py)

**Current code** (lines 708-712):
```python
                                    # Step 7: Final marginal = unnorm * scale
                                    marginal_tile = marginal_unnorm * scale
                                    marginal_tile = tl.where(tile_mask_2d, marginal_tile, 0.0)

                                    # === Accumulate gradients from this tile ===
```

**Replace with**:
```python
                                    # Step 7: Final marginal = unnorm * scale
                                    marginal_tile = marginal_unnorm * scale
                                    marginal_tile = tl.where(tile_mask_2d, marginal_tile, 0.0)

                                    # Defensive sanity check: reject invalid marginals > 1.0
                                    # Valid marginals are probabilities ∈ [0, 1]. Values > 1 indicate
                                    # scale mismatch between forward and backward passes (e.g., at
                                    # checkpoint boundaries). Clamping log_scale to ≤ 0 (line 753-755)
                                    # usually prevents this, but we add this check for robustness.
                                    marginal_tile = tl.where(
                                        marginal_tile > 1.0, 0.0, marginal_tile
                                    )

                                    # === Accumulate gradients from this tile ===
```

## Exact Edit Instructions

**Find**: The line with `marginal_tile = tl.where(tile_mask_2d, marginal_tile, 0.0)` (line 710)

**Insert after it**:
```python

                                    # Defensive sanity check: reject invalid marginals > 1.0
                                    # Valid marginals are probabilities ∈ [0, 1]. Values > 1 indicate
                                    # scale mismatch between forward and backward passes (e.g., at
                                    # checkpoint boundaries). Clamping log_scale to ≤ 0 (line 753-755)
                                    # usually prevents this, but we add this check for robustness.
                                    marginal_tile = tl.where(
                                        marginal_tile > 1.0, 0.0, marginal_tile
                                    )
```

## Testing

After making this change, run:
```bash
python test_triton_minimal.py
```

## Expected Outcomes

### If this IS the culprit (most likely):
- **Without marginal clamping**: Tests PASS (diff < 3e-5)
- **With marginal clamping**: Tests FAIL (diff ~ 0.1-0.4)

This would prove that:
1. The underlying algorithm is correct
2. Marginals > 1.0 are being generated due to numerical precision issues
3. Clamping them to 0.0 corrupts the gradient computation
4. **Root cause**: The "scale mismatch" mentioned in the comment - likely checkpoint normalization bug

### If this is NOT the culprit:
- Tests still FAIL with or without this change
- The bug is in one of the other changes (adaptive TILE_C or Mamba clamping)

## Diagnosis if This is the Culprit

If adding this clamping BREAKS the tests, the next steps are:

1. **Remove the clamping** (it masks the bug)
2. **Instrument the kernel** to find where marginals > 1.0 occur:
   ```python
   # Add after line 710
   if marginal_tile.max() > 1.0:
       tl.device_print("WARNING: marginal > 1.0 at segment", ckpt_idx, "position", t)
   ```
3. **Investigate checkpoint normalization**:
   - Are log_norm_checkpoints computed correctly in forward pass?
   - Are they applied correctly in backward pass?
   - Is there a boundary condition where scales aren't properly aligned?

4. **Check for float32 vs float64 precision issues**:
   - The comment mentions "line 753-755" for log_scale clamping
   - Might need float64 accumulation somewhere

## Verification

Check that the change was applied:
```bash
grep -n "marginal_tile > 1.0" src/torch_semimarkov/streaming/triton_backward.py
```

Should show the line where clamping is applied (around line 716-718).

## Why This is Most Suspicious

1. **Explicitly masks a bug**: Comment admits values > 1.0 "indicate scale mismatch"
2. **Corrupts semantics**: Marginals are probabilities - clamping to 0 is arbitrary
3. **No root cause fix**: Just patches symptoms at computation boundary
4. **Timing matches**: Tests started failing when this was added in 07456c6

If this is the culprit, the fix is to:
- **Remove the clamping**
- **Fix the checkpoint scale mismatch** that causes marginals > 1.0
- Tests should then pass without any defensive clamping
