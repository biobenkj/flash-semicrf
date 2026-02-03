# Bisection Master Plan: Find Culprit in Commit 07456c6

## Executive Summary

**Branch**: `bisect-2k-features` (based on 6ba4d02)

**Goal**: Identify which feature added in commit 07456c6 breaks the Triton marginals tests.

**Status**:
- ✅ Step 1 Complete: Reverted to 6ba4d02 + manually added 2K beta ring
- ⏳ Step 2 Pending: Test baseline on HPC
- ⏳ Steps 3-6 Pending: Bisect features

## Background

**Known Facts**:
1. ✅ 6ba4d02 (baseline): Tests PASS (small 3e-5 errors)
2. ✅ 6ba4d02 + manual 2K beta: Tests PASS (3e-5 errors) - **User verified**
3. ✗ 04060c1 (2K beta + alpha bugs): Tests FAIL (~0.5 errors)
4. ✗ 07456c6 (alpha bugs "fixed" + 3 new features): Tests FAIL (~0.1-0.4 errors)
5. ✗ 725cdb1 (current): Tests FAIL (~0.1-0.4 errors)

**Critical Insight**: Commit 07456c6 added THREE features along with the alpha bug fix:
1. Adaptive TILE_C (`_compute_tile_c` function)
2. Mamba clamping pattern (5 locations in logsumexp)
3. Marginal > 1.0 clamping (explicitly masks checkpoint scale mismatch bug)

**Hypothesis**: One of these three features is the culprit, NOT the 2K beta ring change.

## Bisection Strategy

Test each feature incrementally to identify which one breaks the tests.

### Step 1: Baseline ✅ COMPLETE

**Branch**: `bisect-2k-features`

**Changes**: 6ba4d02 + 2K beta ring only (5 manual edits)

**Expected**: Tests PASS (3e-5 errors)

**Verify on HPC**:
```bash
git checkout bisect-2k-features
python test_triton_minimal.py
```

If PASSES → proceed to Step 2
If FAILS → something went wrong, verify the 5 edits were correct

---

### Step 2: Add Adaptive TILE_C

**Guide**: [BISECTION_STEP2_ADAPTIVE_TILE_C.md](BISECTION_STEP2_ADAPTIVE_TILE_C.md)

**Changes**:
1. Add `_compute_tile_c` function (before line 889)
2. Add `tile_c = _compute_tile_c(C)` (before kernel launch)
3. Change `TILE_C=16` to `TILE_C=tile_c` (line 1164)

**Likelihood**: Low-Medium (changes tiling strategy, could affect atomics)

**Test**:
```bash
python test_triton_minimal.py
```

**Outcomes**:
- ✅ PASSES (diff < 3e-5) → Adaptive TILE_C is OK, proceed to Step 3
- ✗ FAILS (diff ~ 0.1-0.4) → Adaptive TILE_C is the culprit, go to Step 5

---

### Step 3: Add Mamba Clamping

**Guide**: [BISECTION_STEP3_MAMBA_CLAMPING.md](BISECTION_STEP3_MAMBA_CLAMPING.md)

**ONLY proceed if Step 2 PASSED**

**Changes**: 5 locations where logsumexp gets Mamba clamping:
1. Alpha recomputation (line ~474)
2. Beta tile sum (line ~824)
3. Beta online rescaling (line ~840)
4. Beta merging across k (line ~875)
5. (Optional) Marginals computation (line ~459)

**Likelihood**: Medium (changes numerical behavior of logsumexp, could amplify errors)

**Test**:
```bash
python test_triton_minimal.py
```

**Outcomes**:
- ✅ PASSES (diff < 3e-5) → Mamba clamping is OK, proceed to Step 4
- ✗ FAILS (diff ~ 0.1-0.4) → Mamba clamping is the culprit, go to Step 6

---

### Step 4: Add Marginal > 1.0 Clamping

**Guide**: [BISECTION_STEP4_MARGINAL_CLAMPING.md](BISECTION_STEP4_MARGINAL_CLAMPING.md)

**ONLY proceed if Step 3 PASSED**

**Changes**: Single 7-line addition after line 710

**Likelihood**: HIGH (explicitly masks checkpoint scale mismatch bug)

**Test**:
```bash
python test_triton_minimal.py
```

**Outcomes**:
- ✅ PASSES (diff < 3e-5) → None of the 3 features are the culprit (unexpected!)
- ✗ FAILS (diff ~ 0.1-0.4) → Marginal clamping is the culprit, go to Step 7

---

### Step 5: Diagnose Adaptive TILE_C Issue

**If adaptive TILE_C is the culprit:**

1. **Hypothesis**: Forcing multiple tiles at small C creates race conditions in atomic operations
2. **Test**: Try different TILE_C values (4, 8, 16, 32) to see which work
3. **Instrument**: Add device_print to show tile execution order
4. **Fix options**:
   - Revert to fixed TILE_C=16 (simplest)
   - Add synchronization between tiles
   - Use different accumulation strategy (non-atomic)

---

### Step 6: Diagnose Mamba Clamping Issue

**If Mamba clamping is the culprit:**

1. **Hypothesis**: Clamping differences changes numerical stability in unexpected way
2. **Test**: Remove clamping from one location at a time to find which breaks it
3. **Investigate**: Why does clamping cause 1000x error amplification?
4. **Fix options**:
   - Revert Mamba clamping (use original logsumexp)
   - Use different clamping threshold (e.g., -1e6 instead of 0.0)
   - Apply clamping only at specific locations, not all 5

---

### Step 7: Diagnose Marginal Clamping Issue ⭐ MOST LIKELY

**If marginal > 1.0 clamping is the culprit:**

This is the EXPECTED outcome because the commit comment explicitly admits there's a bug:

> "Values > 1 indicate scale mismatch between forward and backward passes (e.g., at checkpoint boundaries)"

**Diagnosis steps**:

1. **Remove the clamping** - it masks the real bug
2. **Instrument to find where marginals > 1.0 occur**:
   ```python
   max_marginal = marginal_tile.max()
   if max_marginal > 1.0:
       tl.device_print("marginal > 1:", max_marginal, "at seg", ckpt_idx, "pos", t)
   ```
3. **Check checkpoint normalization**:
   - Are `log_norm_checkpoints` from forward pass correct?
   - Are they applied correctly in backward pass?
   - Is there a segment boundary bug?

4. **Root cause hypotheses**:
   - Checkpoint `log_Z` values not properly aligned between segments
   - Numerical precision loss in log_norm accumulation
   - Segment boundary condition where scale isn't applied
   - Float32 vs float64 precision in checkpoint storage

5. **Fix approach**:
   - Fix the checkpoint scale mismatch
   - Tests should pass WITHOUT any defensive clamping
   - Marginals should naturally stay in [0, 1]

---

## Quick Reference

| Step | Feature | Files Changed | Lines | Complexity | Suspicion |
|------|---------|---------------|-------|------------|-----------|
| 1 | 2K beta ring | triton_backward.py | 5 edits | Low | ✅ Verified OK |
| 2 | Adaptive TILE_C | triton_backward.py | 3 edits | Low | Low-Medium |
| 3 | Mamba clamping | triton_backward.py | 5 locations | Medium | Medium |
| 4 | Marginal > 1.0 | triton_backward.py | 1 edit | Low | **HIGH** ⭐ |

## Testing Commands

```bash
# Baseline test
python test_triton_minimal.py

# Full test suite (after fix identified)
pytest tests/test_triton_marginals.py -v
pytest tests/test_streaming_k_boundaries.py -v
pytest tests/test_streaming_triton.py -v
```

## Current Status

```bash
# Branch
git branch
# Should show: * bisect-2k-features

# Changes
git diff 6ba4d02 --stat
# Should show: triton_backward.py | 5 changes (2K beta ring)

# Test baseline on HPC
python test_triton_minimal.py
# Expected: ✓ TEST PASSED, max diff ~3e-5
```

## Contacts and References

- Plan file: `~/.claude/plans/curious-roaming-crane.md`
- Git commits:
  - 6ba4d02: Working baseline
  - 04060c1: 2K beta + alpha bugs (broken)
  - 07456c6: Alpha fix + 3 features (still broken)
  - 725cdb1: Current main (still broken)

- Test files:
  - `test_triton_minimal.py`: Ultra-minimal test (batch=1, T=48, C=8, K=16)
  - `tests/test_triton_marginals.py`: Full test suite (10 tests)

## Decision Tree

```
Test baseline (6ba4d02 + 2K beta)
├─ PASS → Add adaptive TILE_C
│  ├─ PASS → Add Mamba clamping
│  │  ├─ PASS → Add marginal clamping
│  │  │  ├─ PASS → Unexpected! All features OK, bug elsewhere
│  │  │  └─ FAIL → Marginal clamping is culprit (EXPECTED) → Diagnose checkpoint scale mismatch
│  │  └─ FAIL → Mamba clamping is culprit → Diagnose logsumexp numerical stability
│  └─ FAIL → Adaptive TILE_C is culprit → Diagnose atomic contention / tiling race
└─ FAIL → Error in baseline setup, verify 2K beta ring edits

```

## Next Actions

1. **User**: Test baseline on HPC cluster
2. **If PASS**: Follow step 2 guide to add adaptive TILE_C
3. **If FAIL at any step**: Diagnose using corresponding section above
4. **When culprit found**: Remove problematic feature and fix underlying bug
5. **Verify fix**: Run full test suite

---

**Last Updated**: 2026-02-03
**Branch**: `bisect-2k-features`
**Current Step**: Testing baseline on HPC
