# Bisection Step 3: Add Mamba Clamping Pattern

## Overview
This document describes how to add the Mamba clamping pattern from commit 07456c6 to prevent overflow in logsumexp operations.

**IMPORTANT**: Only proceed with this step if adaptive TILE_C (Step 2) did NOT break the tests.

## What is Mamba Clamping?

The Mamba pattern clamps differences to `[−∞, 0]` before taking exp in logsumexp operations:

**Before (baseline)**:
```python
log_sum_exp = tl.log(tl.exp(x - max_val) + tl.exp(y - max_val))
```

**After (Mamba clamping)**:
```python
diff_x = tl.minimum(x - max_val, 0.0)
diff_y = tl.minimum(y - max_val, 0.0)
log_sum_exp = tl.log(tl.exp(diff_x) + tl.exp(diff_y))
```

This prevents overflow when differences are large positive numbers (which can happen due to floating-point rounding).

## Locations Requiring Changes

There are **5 locations** in the kernel where Mamba clamping needs to be applied:

### 1. Alpha Recomputation (lines ~474-477)
**Current**:
```python
                                log_sum_exp_acc = tl.log(
                                    tl.exp(alpha_t - max_alpha_safe)
                                    + tl.exp(score_for_k - max_alpha_safe)
                                )
```

**Replace with**:
```python
                                # Mamba pattern: clamp differences to [−∞, 0] before exp to prevent overflow
                                diff_alpha = tl.minimum(alpha_t - max_alpha_safe, 0.0)
                                diff_score = tl.minimum(score_for_k - max_alpha_safe, 0.0)
                                log_sum_exp_acc = tl.log(tl.exp(diff_alpha) + tl.exp(diff_score))
```

### 2. Beta Computation - Tile Sum (lines ~824)
**Current**:
```python
                                    sum_exp_tile = tl.sum(
                                        tl.exp(scores_for_beta_tile - max_tile_safe[None, :]),
                                        axis=0,
                                    )
```

**Replace with**:
```python
                                    # Mamba pattern: clamp differences to [−∞, 0] before exp to prevent overflow
                                    diff_tile = tl.minimum(
                                        scores_for_beta_tile - max_tile_safe[None, :], 0.0
                                    )
                                    sum_exp_tile = tl.sum(tl.exp(diff_tile), axis=0)
```

### 3. Beta Computation - Online Rescaling (lines ~835-841)
**Current**:
```python
                                    l_beta_k = tl.where(
                                        is_m_neginf,
                                        sum_exp_tile * tl.exp(max_tile - m_new_safe),
                                        l_beta_k * tl.exp(m_beta_k - m_new_safe)
                                        + sum_exp_tile * tl.exp(max_tile - m_new_safe),
                                    )
```

**Replace with**:
```python
                                    # Rescale previous sum and add new tile's contribution
                                    # Mamba pattern: clamp differences to [−∞, 0] before exp to prevent overflow
                                    diff_m_beta = tl.minimum(m_beta_k - m_new_safe, 0.0)
                                    diff_max_tile = tl.minimum(max_tile - m_new_safe, 0.0)
                                    l_beta_k = tl.where(
                                        is_m_neginf,
                                        sum_exp_tile * tl.exp(diff_max_tile),
                                        l_beta_k * tl.exp(diff_m_beta)
                                        + sum_exp_tile * tl.exp(diff_max_tile),
                                    )
```

### 4. Beta Merging Across k (lines ~873-876)
**Current**:
```python
                                log_sum_exp_new = tl.log(
                                    tl.exp(new_beta - max_new_safe) + tl.exp(beta_k - max_new_safe)
                                )
```

**Replace with**:
```python
                                # Mamba pattern: clamp differences to [−∞, 0] before exp to prevent overflow
                                diff_new_beta = tl.minimum(new_beta - max_new_safe, 0.0)
                                diff_beta_k = tl.minimum(beta_k - max_new_safe, 0.0)
                                log_sum_exp_new = tl.log(
                                    tl.exp(diff_new_beta) + tl.exp(diff_beta_k)
                                )
```

### 5. (Optional) Marginals Computation (line ~459)
This one might not be in the kernel - check if it exists in your current code:
```python
                                    tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1)
```

If it exists, apply the same pattern:
```python
                                    # Mamba pattern: clamp differences to [−∞, 0] before exp to prevent overflow
                                    diff_scores = tl.minimum(scores - max_scores_safe[:, None], 0.0)
                                    sum_exp = tl.sum(tl.exp(diff_scores), axis=1)
```

## Recommended Approach

Given the complexity of these changes (5 locations), consider:

1. **Option A**: Manually apply all 5 changes and test
2. **Option B**: Use git to cherry-pick just the Mamba clamping changes from 07456c6
3. **Option C**: Skip Mamba clamping for now and test marginal > 1.0 clamping first (simpler)

## Testing

After making these changes, run:
```bash
python test_triton_minimal.py
```

**Expected outcomes**:
1. **If test PASSES** (diff < 3e-5): Mamba clamping is not the culprit, proceed to marginal > 1.0 clamping
2. **If test FAILS** (diff ~ 0.1-0.4): Mamba clamping is the culprit, diagnose the issue

## Verification

Check that all changes were applied:
```bash
grep -n "Mamba pattern" src/torch_semimarkov/streaming/triton_backward.py
```

Should show 4-5 comment lines (one for each location).

```bash
grep -n "tl\.minimum.*- max.*0\.0" src/torch_semimarkov/streaming/triton_backward.py
```

Should show 8-10 lines (diff_alpha, diff_score, diff_tile, diff_m_beta, diff_max_tile, diff_new_beta, diff_beta_k, etc.).

## Alternative: Test by Reverting

If you want to test whether Mamba clamping is the culprit without manually applying all changes, you could:
1. Checkout commit 07456c6
2. Manually revert ONLY the Mamba clamping changes (opposite edits)
3. Test

This might be faster than applying all changes manually.
