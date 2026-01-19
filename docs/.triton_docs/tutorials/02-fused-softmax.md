# Fused Softmax Tutorial: Complete Guide

## Overview

This tutorial demonstrates how to implement a high-performance softmax kernel using Triton that significantly outperforms PyTorch's native implementation for matrices where rows fit in GPU SRAM.

## Key Learning Objectives

The tutorial covers:
- Benefits of kernel fusion for bandwidth-limited operations
- Reduction operators in Triton
- Numerical stability techniques

## Motivation: Why Fusion Matters

A naive PyTorch softmax implementation requires multiple read/write cycles:

```python
def naive_softmax(x):
    """Row-wise softmax with numerical stability"""
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret
```

This approach reads 5MN + 2M elements and writes 3MN + 2M elements. A "fused" kernel reading X once and computing on-chip requires only MN bytes, yielding ~4x theoretical speedup.

## Optimized Kernel Implementation

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride,
                   output_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step,
                            num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        # Load with masking for irregular shapes
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # Numerical stability: subtract maximum
        row_minus_max = row - tl.max(row, axis=0)

        # Compute softmax
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # Write results
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

## Critical Implementation Details

**Power-of-Two Blocking**: Triton requires power-of-two block sizes. The kernel pads internally and uses masking to handle arbitrary column counts.

**Numerical Stability**: Subtracting the row maximum before exponentiation prevents overflow—softmax is invariant to constant shifts.

**Occupancy Tuning**: The wrapper function calculates optimal thread counts based on register usage and shared memory constraints.

## Validation

```python
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch)
```

Testing with irregular dimensions (1823×781) confirms padding mechanisms work correctly.

## Performance Benchmarks

The fusion approach achieves compelling results:

- **vs. PyTorch softmax**: Triton is approximately 4x faster
- **vs. naive implementation**: Triton shows 3-4x improvement across varying column counts
- **Scalability**: Performance remains consistent from 256 to 12,288 columns

The benchmark tested matrices with 4,096 rows across varying column widths, measuring throughput in GB/s. Triton maintained 1,400+ GB/s for large matrices while PyTorch fluctuated more significantly.

## Conclusion

This tutorial illustrates how "kernel fusion eliminates redundant memory operations by combining multiple computational stages within a single GPU kernel" while maintaining clarity and maintainability compared to hand-optimized CUDA code.
