# Backends and Triton kernel

This project provides multiple semi-CRF inference backends with different
performance and memory profiles.

## Backend summary

| Backend | Time | DP memory | Parallel depth | Best for |
|---------|------|-----------|----------------|----------|
| `linear_scan_streaming` | O(TKC^2) | O(KC) | O(T) | **Default** - best memory, near-optimal speed |
| `linear_scan_vectorized` | O(TKC^2) | O(TKC) | O(T) | When memory permits (2-3x faster than scalar) |
| `linear_scan` | O(TKC^2) | O(TKC) | O(T) | Reference implementation |
| `binary_tree` | O(TKC^2 log T) | O(T(KC)^2) | O(log T) | Small KC only |
| `binary_tree_sharded` | O(TKC^2 log T) | O(T(KC)^2) | O(log T) | Reduced peak memory |
| `block_triangular` | O(TKC^2) | O(T(KC)^2) | O(log T) | Structured sparsity |

## Recommendation

**Default: `linear_scan_streaming`** - best for most use cases:
- O(KC) memory via ring buffer - always fits
- Within a few percent of vectorized speed
- Works across all tested configurations

Use `use_vectorized=True` when memory permits for 2-3x speedup over scalar scan.

Tree-based methods can exhaust GPU memory for KC > 150 because of O((KC)^3)
log-semiring temporaries.

## Triton fused streaming kernel (up to 45x speedup)

`torch_semimarkov.triton_scan.semi_crf_triton_forward` provides a fused O(T)
streaming scan that keeps the K x C frontier in fast memory. It mirrors the
streaming scan but collapses the loop into a single GPU kernel, yielding up to
45x speedup compared to the vectorized PyTorch implementation.

Behavior:
- Uses Triton automatically when available and the input is CUDA.
- Falls back to the PyTorch reference when Triton is missing or inputs are CPU.
- `validate=True` runs a float64 PyTorch reference for numerical checks.

Example:

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

partition = semi_crf_triton_forward(edge.cuda(), lengths.cuda())
```
