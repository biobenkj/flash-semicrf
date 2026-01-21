# Lessons Learned: Writing Custom Triton Kernels

Internal notes from developing the torch-semimarkov Triton kernels.

---

## 1. Off-by-One Errors: The Silent Killer

### The Pattern
Off-by-one errors in loop bounds cause **partial correctness** - the kernel appears to work but silently skips edge cases (first/last positions, segment boundaries).

### Symptoms
- Forward pass looks correct, backward pass has wrong gradients
- Tests with uniform data pass, but edge positions fail
- ~70-95% of gradient elements match, but boundary positions are wrong/zero

### Root Causes Found

| Bug | Pattern | Fix |
|-----|---------|-----|
| Final position skipped | `t < length - 1` | `t < length` |
| Last segment ignored | `end_pos <= length - 1` | `end_pos <= length` |
| Beta init off by one | `final_pos = seq_len - 1` | `final_pos = seq_len` |
| Loop excludes boundary | `seg_start < seq_len - 1` | `seg_start < seq_len` |

### Prevention
1. **Think in terms of inclusive vs exclusive bounds** - document which you're using
2. **Test boundary positions explicitly** - position 0, position T-1, and T
3. **Add debug logging** that prints values at first/last positions during development

---

## 2. Keep PyTorch Reference and Triton in Sync

### The Problem
When you fix a bug in the PyTorch reference implementation, you **must** propagate it to the Triton kernel. We fixed off-by-one errors in `pytorch_reference.py` but forgot to update `triton_backward.py` - this caused 8 tests to fail months later.

### Solution
- Maintain a checklist: every bug fix location in PyTorch has a corresponding Triton location
- When fixing loops/bounds in one, grep for the same pattern in the other
- Consider having a single source of truth for constants like loop bounds

### Example
```python
# PyTorch reference (fixed):
active_mask = t < lengths  # NOT: t < (lengths - 1)

# Triton (also needs fixing):
if t >= seg_start and t < seq_len:  # NOT: t < seq_len - 1
```

---

## 3. Gradient Scaling for Shared vs Per-Batch Parameters

### The Bug
For parameters shared across the batch (e.g., transition matrix), the naive approach:
```python
grad = workspace.sum(dim=0) * grad_output.sum()  # WRONG
```
gives incorrect gradients when `grad_output` is non-uniform (e.g., masked losses).

### The Fix
Weight each batch element's contribution by its `grad_output` before summing:
```python
# Correct: weight before summing
grad = torch.einsum('bij, b -> ij', workspace, grad_output)
```

### Why Tests Didn't Catch It
All tests used `.sum().backward()` which produces uniform `grad_output = [1,1,...,1]`. With uniform weights, the bug is mathematically hidden:
```
BUGGY:  (m[0] + m[1]) * (1 + 1) = 2*(m[0] + m[1])
CORRECT: m[0]*1 + m[1]*1 = m[0] + m[1]
# Different by a factor, but proportional
```

### Prevention
Always test with non-uniform `grad_output`:
```python
def test_heterogeneous_grad_output():
    grad_output = torch.tensor([0.5, 2.0, 0.0])  # Non-uniform, includes zero
    partition.backward(grad_output)
```

---

## 4. Memory-Efficient Gradient Aggregation

### The Problem
Broadcasting creates large intermediate tensors:
```python
# Creates (batch, K, C, C) intermediate before summing
grad = (workspace * grad_output.view(-1, 1, 1, 1)).sum(dim=0)
# For K=1024, C=64, batch=16: 268MB intermediate
```

### The Solution
Use `einsum` to fuse multiply + reduce:
```python
# No intermediate allocation
grad = torch.einsum('bkij, b -> kij', workspace, grad_output)
```

### When It Matters
- Large `K` (max_duration) or `C` (num_classes)
- GPU memory constrained environments
- Batch sizes > 8-16

---

## 5. Duration/Index Conventions

### The Confusion
Is `duration_bias[k]` the bias for duration `k` or duration `k+1`? Is it 0-indexed or 1-indexed?

### Our Convention (after fixing bugs)
- `duration_bias[k, c]` = bias for duration `k`, class `c`
- Index 0 is unused (no segments of duration 0)
- Valid durations: 1 to K-1 (where K = `max_duration`)
- Durations >= K are clamped to K-1

### The Bug We Fixed
```python
# WRONG: 0-indexed
score += duration_bias[duration - 1, label]

# CORRECT: 1-indexed
score += duration_bias[duration, label]
```

### Prevention
Document your indexing convention explicitly and add assertions:
```python
assert 1 <= duration <= K - 1, f"Duration {duration} out of range"
```

---

## 6. Testing Strategies for Triton Kernels

### What Catches Bugs

| Test Type | What It Catches |
|-----------|-----------------|
| PyTorch vs Triton comparison | Implementation divergence |
| Non-uniform `grad_output` | Shared parameter scaling bugs |
| Explicit boundary positions | Off-by-one errors |
| Variable-length batches | Masking/padding bugs |
| Single-batch vs batched | Reduction bugs |
| Gradcheck (finite difference) | Backward correctness |

### Debug Test Pattern
When tests fail, add a diagnostic test that prints detailed comparisons:
```python
def test_debug_gradient_comparison():
    # Run both implementations
    triton_grad = compute_triton_grad(...)
    pytorch_grad = compute_pytorch_grad(...)

    # Print position-by-position comparison
    for t in range(T):
        print(f"Position {t}:")
        print(f"  Triton:  {triton_grad[0, t]}")
        print(f"  PyTorch: {pytorch_grad[0, t]}")
        print(f"  Diff:    {(triton_grad - pytorch_grad)[0, t]}")
```

### The "Mask Isolation" Test
This test directly catches the gradient scaling bug:
```python
def test_mask_isolation():
    # batch=2, mask out second sequence
    grad_output = torch.tensor([1.0, 0.0])

    # Run batched
    partition_batch.backward(grad_output)
    grad_batched = param.grad.clone()

    # Run single (first sequence only)
    partition_single.backward(torch.ones(1))
    grad_single = param.grad.clone()

    # Must be equal - masked sequence contributes nothing
    assert torch.allclose(grad_batched, grad_single)
```

---

## 7. Triton-Specific Gotchas

### Loop Bounds
Triton uses `tl.range()` which has different semantics than Python `range()` in some cases. Be explicit:
```python
# Python: range(1, K) when K=1 is empty
# Triton: tl.range(1, K) when K=1 is also empty
# Fix: tl.range(1, tl.maximum(K, 2))
```

### Ring Buffer Initialization
Move expensive initialization from kernel to launcher when possible:
```python
# Bad: Initialize in kernel (runs per block)
for k in tl.range(0, K):
    ring_buffer[k] = init_value

# Good: Initialize in launcher (runs once)
ring_buffer = torch.full((batch, K, C), NEG_INF, device=device)
ring_buffer[:, 0, :] = 0.0
```

### Workspace Tensors
Pre-allocate workspace tensors in the launcher rather than inside the kernel. The kernel should only read/write to pre-allocated memory.

---

## 8. Debugging Without Local GPU

When you don't have local GPU access:

1. **Add diagnostic tests** that print detailed comparisons
2. **Push to CI/HPC** and analyze the output
3. **Compare forward pass first** - if forward matches, the bug is in backward
4. **Check specific positions** - especially boundaries (0, T-1, T)
5. **Look for patterns** - all zeros at certain positions = loop bound issue

---

## 9. Documentation Checklist

For custom Triton kernels, document:

- [ ] Tensor conventions (indexing order, 0-indexed vs 1-indexed)
- [ ] Loop bound semantics (inclusive vs exclusive)
- [ ] Memory layout assumptions
- [ ] Which parameters are per-batch vs shared
- [ ] Expected shapes at each stage
- [ ] Numerical stability considerations (logsumexp, cumsum precision)

---

## Quick Reference: Common Bug Patterns

| Symptom | Likely Cause |
|---------|--------------|
| Last position has zero gradient | Loop bound `< length - 1` instead of `< length` |
| Gradients wrong by constant factor | Shared param scaling with uniform grad_output |
| Tests pass but production fails | Insufficient test coverage (uniform grad_output, no edge cases) |
| Forward correct, backward wrong | Off-by-one in backward loop bounds |
| PyTorch works, Triton doesn't | Bug fix not propagated between implementations |
| OOM on backward | Intermediate tensor from broadcasting (use einsum) |
