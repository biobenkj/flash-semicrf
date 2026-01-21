# Vector Addition Tutorial - Triton Documentation

## Overview

This tutorial demonstrates how to implement vector addition using Triton, covering the programming model, kernel decoration, and validation best practices.

## Core Concepts

### The `@triton.jit` Decorator

Triton kernels are defined using the `@triton.jit` decorator. This marks functions as GPU kernels that can be compiled and executed on parallel hardware.

### Program Identification and Block Sizing

The kernel uses `tl.program_id(axis=0)` to identify which parallel instance is executing. As the documentation explains: "There are multiple 'programs' processing different data. We identify which program we are here."

Each program processes a block of elements determined by `BLOCK_SIZE`. For a 256-element vector with 64-element blocks, programs would handle elements `[0:64, 64:128, 128:192, 192:256]` respectively.

## Kernel Implementation

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)
```

### Memory Access Patterns

The kernel creates offset pointers for parallel data access. "Create a mask to guard memory operations against out-of-bounds accesses." This prevents reading/writing beyond vector boundaries when the size isn't divisible by block size.

## Wrapper Function

```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

The launch grid specifies "the number of kernel instances that run in parallel," analogous to CUDA grid configurations.

## Validation Results

Testing with 98,432 elements showed perfect accuracy: "The maximum difference between torch and triton is 0.0"

## Performance Benchmarking

The tutorial includes comparative benchmarks across vector sizes from 2^12 to 2^27 elements, measuring throughput in GB/s. Results demonstrate that Triton's implementation matches PyTorch's performance across all tested sizes, reaching approximately 1,684 GB/s at maximum scales.
