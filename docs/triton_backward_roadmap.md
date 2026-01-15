# Triton Backward Pass Implementation Roadmap

This document outlines the plan for implementing a custom Triton backward kernel for the semi-CRF log-partition function, eliminating the dependency on `torch.compile`.

## Motivation

The current implementation uses:
- **Forward**: Custom Triton kernel with O(KC) memory ring buffer (~45x speedup)
- **Backward**: `torch.compile` on PyTorch reference code

Problems with `torch.compile`:
- Unpredictable compilation times (minutes to hang indefinitely)
- Shape-dependent compilation (each unique T/K/C requires recompilation)
- Instability across PyTorch versions
- CPU-bound compilation bottleneck on HPC

A custom Triton backward would provide:
- Fast, predictable kernel compilation (seconds)
- No dependency on torch.compile behavior
- Potential for fused forward+backward optimization
- Consistent behavior across environments

---

## Mathematical Background

### Semi-CRF Log-Partition Forward

The forward pass computes:
```
Z = Σ_y exp(score(y))  →  log Z = logsumexp over all valid segmentations
```

Using the streaming scan with ring buffer:
```
α[t, c] = logsumexp over k in 1..K of:
    α[t-k, c'] + edge[t-k:t, k, c', c]
```

where `α[t, c]` is the log-sum of all paths ending at position t with label c.

### Backward: Gradient = Marginals

The gradient of log Z with respect to edge potentials is the **marginal probability** of each edge:

```
∂(log Z) / ∂(edge[t, k, c', c]) = P(segment (t, k, c', c) is used)
                                = exp(α[t, c'] + edge[t,k,c',c] + β[t+k, c] - log Z)
```

where:
- `α[t, c]` = forward (log-sum of paths ending at t with label c)
- `β[t, c]` = backward (log-sum of paths starting from t with label c)
- `log Z` = final partition value

### Backward DP Recurrence

The backward values are computed in reverse:
```
β[t, c] = logsumexp over k in 1..K, c' in 1..C of:
    edge[t, k, c, c'] + β[t+k, c']
```

Starting from `β[T, c] = 0` (or appropriate boundary).

---

## Implementation Strategy

### Option A: Recomputation (Gradient Checkpointing Style)
**Memory**: O(KC) — same as forward
**Compute**: 2x forward (recompute α during backward)

1. During backward, re-run forward to regenerate α values on-the-fly
2. Simultaneously compute β in reverse
3. At each position, compute gradient contribution using α, β, edge

**Pros**: Minimal memory, matches current forward memory profile
**Cons**: 2x compute for backward

### Option B: Checkpointed Segments
**Memory**: O(√T × KC)
**Compute**: ~1.5x forward

1. During forward, save α at every √T positions
2. During backward, recompute α only within segments
3. Compute β and gradients

**Pros**: Better compute/memory tradeoff
**Cons**: More complex implementation

### Option C: Full Storage (Baseline)
**Memory**: O(T × KC)
**Compute**: 1x forward + 1x backward

1. Save all α values during forward
2. Compute β in reverse pass
3. Compute gradients

**Pros**: Simplest implementation, fastest backward
**Cons**: Memory scales with T (defeats streaming advantage)

### Recommended: Option A (Recomputation)

For consistency with the O(KC) memory goal, Option A is recommended. The 2x compute cost is acceptable given:
- Forward is already very fast (Triton optimized)
- Memory is the primary constraint for long sequences
- Matches behavior of gradient checkpointing but in fused kernel

---

## Implementation Plan

### Phase 1: PyTorch Reference Backward
- [ ] Implement standalone backward function in PyTorch
- [ ] Verify gradient correctness against `torch.autograd.gradcheck`
- [ ] Benchmark memory usage and timing
- [ ] Document the exact computation pattern

### Phase 2: Triton Backward Kernel (Single Block)
- [ ] Port backward recurrence to Triton
- [ ] Implement reverse-direction ring buffer
- [ ] Handle boundary conditions (β initialization)
- [ ] Test on small shapes (T≤256, KC≤64)

### Phase 3: Fused Forward-Backward Kernel
- [ ] Combine forward recomputation with backward in single kernel
- [ ] Optimize memory access patterns
- [ ] Add gradient accumulation into output tensor
- [ ] Handle variable sequence lengths

### Phase 4: Multi-Block / Large Shape Support
- [ ] Extend to larger KC (block-parallel over C dimension)
- [ ] Handle T > single-block capacity
- [ ] Optimize for different GPU architectures

### Phase 5: Integration & Testing
- [ ] Create `torch.autograd.Function` wrapper
- [ ] Add to `semi_crf_triton_forward` with `use_custom_backward=True`
- [ ] Comprehensive gradient checking across shapes
- [ ] Benchmark against torch.compile version
- [ ] Update documentation

---

## Detailed Implementation Notes

### Ring Buffer for Backward

The backward needs access to β[t+1], β[t+2], ..., β[t+K] to compute β[t].

```python
# Backward ring buffer (similar to forward)
beta_ring = zeros(K, C)  # β values for positions t+1 to t+K

for t in reversed(range(T-1)):
    # Current β[t] computation
    beta_new = zeros(C)
    for c in range(C):
        for k in range(1, min(K, T-t)+1):
            for c_next in range(C):
                # edge[t, k-1, c, c_next] + β[t+k, c_next]
                contrib = edge[t, k-1, c, c_next] + beta_ring[(t+k) % K, c_next]
                beta_new[c] = logsumexp(beta_new[c], contrib)

    # Update ring buffer
    beta_ring[t % K, :] = beta_new
```

### Gradient Computation

At each position t, compute gradient contributions:
```python
for t in range(T-1):
    for k in range(1, K+1):
        for c_prev in range(C):
            for c in range(C):
                # Marginal = exp(α + edge + β - logZ)
                marginal = exp(
                    alpha[t, c_prev] +
                    edge[t, k-1, c_prev, c] +
                    beta[t+k, c] -
                    log_Z
                )
                grad_edge[t, k-1, c_prev, c] = marginal
```

### Triton Kernel Structure

```python
@triton.jit
def semi_crf_backward_kernel(
    edge_ptr,           # Input: (B, T-1, K, C, C)
    grad_edge_ptr,      # Output: (B, T-1, K, C, C)
    log_Z_ptr,          # Input: (B,) - partition values
    lengths_ptr,        # Input: (B,) - sequence lengths
    B, T, K, C,         # Dimensions
    BLOCK_C: tl.constexpr,
):
    # Block handles one batch element
    batch_idx = tl.program_id(0)

    # Initialize ring buffers for α (forward) and β (backward)
    # ... implementation details ...

    # Forward pass (recompute α)
    for t in range(T-1):
        # ... compute alpha[t] ...
        # Store in ring buffer

    # Backward pass (compute β and gradients)
    for t in reversed(range(T-1)):
        # ... compute beta[t] from ring buffer ...
        # ... compute gradient contribution ...
        # Update ring buffer
```

---

## Testing Strategy

### Unit Tests
- [ ] `test_backward_small`: T=8, K=2, C=2 — verify against analytical gradient
- [ ] `test_backward_gradcheck`: Use `torch.autograd.gradcheck` with float64
- [ ] `test_backward_vs_pytorch`: Compare against PyTorch reference backward

### Integration Tests
- [ ] `test_backward_variable_lengths`: Batched sequences with different lengths
- [ ] `test_backward_semirings`: Test both Log and Max semirings
- [ ] `test_backward_large_shapes`: T=1024, K=16, C=12

### Benchmark Tests
- [ ] Compare timing: custom backward vs torch.compile vs checkpointing
- [ ] Memory profiling: verify O(KC) memory usage
- [ ] Scaling tests: vary T while keeping KC fixed

---

## Files to Create/Modify

```
torch_semimarkov/
├── triton_scan.py              # Existing forward kernel
├── triton_backward.py          # NEW: Backward kernel
└── triton_autograd.py          # NEW: autograd.Function wrapper

tests/
└── test_triton_backward.py     # NEW: Backward tests

benchmarks/
└── benchmark_triton_backward.py # NEW: Backward benchmarks
```

---

## Open Questions

1. **Max semiring backward**: The backward for max is simpler (just argmax path), but needs different handling. Implement separately or unified?

2. **Numerical stability**: The marginal computation involves `exp(α + edge + β - logZ)`. For numerical stability, may need to compute in log-space and only exponentiate at the end.

3. **Memory layout**: Current edge tensor is `(B, T-1, K, C, C)`. Is this optimal for the backward access pattern, or should we consider transposed storage?

4. **Block size tuning**: What BLOCK_C values work best? Likely need power-of-2 ≥ C.

---

## References

- Current forward kernel: `torch_semimarkov/triton_scan.py`
- PyTorch reference: `torch_semimarkov/triton_scan.py::_semi_crf_forward_pytorch`
- Semi-CRF paper: Sarawagi & Cohen, "Semi-Markov Conditional Random Fields"
- Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/

---

## Progress Log

| Date | Status | Notes |
|------|--------|-------|
| 2025-01-14 | Planning | Created roadmap document |
| | | |

---

*Last updated: 2025-01-14*
