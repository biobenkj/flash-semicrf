# torch-semimarkov

Efficient Semi-Markov CRF Inference for PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/biobenkj/torch-semimarkov/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/biobenkj/torch-semimarkov/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/biobenkj/torch-semimarkov/branch/main/graph/badge.svg)](https://codecov.io/gh/biobenkj/torch-semimarkov)

## Overview

This library provides optimized implementations of Semi-Markov CRF inference algorithms, benchmarked and documented in:

> **Practical Semi-Markov CRF Inference for Genomic Sequence Annotation**
> Benjamin K. Johnson (2026)

**Key finding:** Memory, not time, is the binding constraint. Streaming linear scan is universally applicable across all genomic parameter regimes.

Highlights:
- Streaming scan with O(KC) memory (default) - within a few percent of vectorized speed
- Optional Triton fused kernel for up to 45x GPU speedup
- Vectorized scan available when memory permits (O(TKC) memory, 2-3x faster than scalar)

## Why Semi-Markov CRFs?

Semi-Markov CRFs extend linear-chain CRFs with explicit duration modeling:

```
psi(x_{s:e}, c', c, d) = psi_emission(x_{s:e}, c) + psi_transition(c', c) + psi_duration(c, d)
```

This provides three structural advantages:
1. **Guaranteed valid segmentations** - segments tile the sequence by construction
2. **Explicit duration modeling** - incorporate biological priors (exon lengths, TE sizes)
3. **Segment-level posteriors** - amenable to calibration and uncertainty quantification

## Installation

### Basic Installation (CPU)

```bash
pip install torch-semimarkov
```

### Development Installation

```bash
git clone https://github.com/benjohnson/torch-semimarkov.git
cd torch-semimarkov
pip install -e ".[dev]"
```

### With CUDA Support

For GPU acceleration with custom CUDA kernels:

```bash
pip install -e . --config-settings="--build-option=--cuda"
```

Or using environment variable:
```bash
TORCH_SEMIMARKOV_CUDA=1 pip install -e .
```

Note: Building the CUDA extension requires a CUDA toolkit (nvcc) available via
`CUDA_HOME`. If the toolkit is not found, the build is skipped.

### Optional Triton Kernel (GPU)

The fused streaming kernel uses [Triton](https://github.com/openai/triton) and is optional:

```bash
pip install triton
```

Triton is used automatically when available and the input is CUDA.

## Quick Start

```python
import torch
from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring

# Parameters
batch_size = 4
seq_length = 1000   # T
max_duration = 16   # K
num_classes = 6     # C

# Create model
model = SemiMarkov(LogSemiring)

# Edge potentials: (batch, T-1, K, C, C)
edge = torch.randn(batch_size, seq_length - 1, max_duration, num_classes, num_classes)
lengths = torch.full((batch_size,), seq_length)

# Forward pass (partition function)
# Uses streaming scan by default: O(KC) memory, within a few percent of vectorized speed
log_Z, _, _ = model.logpartition(edge, lengths=lengths)

# Backward pass for gradients
log_Z.sum().backward()
```

### Triton Fused Streaming Kernel (up to 45x speedup)

```python
from torch_semimarkov.triton_scan import semi_crf_triton_forward

edge = edge.cuda()
lengths = lengths.cuda()
partition = semi_crf_triton_forward(edge, lengths)  # uses Triton automatically on CUDA
```

## Documentation

- [Parameter guide: T, K, C](docs/parameter_guide.md)
- [Backends and Triton kernel](docs/backends.md)
- [Benchmarking](docs/benchmarks.md)
- [API reference](docs/api.md)
- [AI disclosure](docs/disclosure.md)

## Testing

The test suite includes ~2,900 lines of tests across 15 test files:

```bash
# Run full test suite (CPU-only by default)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=torch_semimarkov --cov-report=term-missing
```

### Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_backend_equivalence.py` | All backends produce identical results |
| `test_semimarkov_utils.py` | sum(), marginals(), to_parts()/from_parts() |
| `test_hsmm.py` | Hidden semi-Markov model integration |
| `test_numerical_gradients.py` | Gradient correctness via finite differences |
| `test_error_handling.py` | Invalid inputs and edge cases |
| `test_semimarkov_banded.py` | Banded backend operations |
| `test_cpu_only.py` | CPU fallback behavior |
| `test_partition_equivalence.py` | Streaming scan ring buffer correctness |
| `test_banded_matrix.py` | BandedMatrix unit tests |
| `test_semirings.py` | Semiring operations (Log, Max, Std, etc.) |
| `test_blocktriangular.py` | Block-triangular matmul |
| `test_banded_utils.py` | Bandwidth measurement, permutations |
| `test_checkpoint_semiring.py` | Gradient checkpointing |
| `test_helpers_cpu.py` | Base _Struct class |
| `test_triton_scan.py` | Triton kernel fallback |

Note: Tests run CPU-only by default. GPU tests require CUDA and are skipped in CI.

## Project Structure

```
torch-semimarkov/
├── src/torch_semimarkov/
│   ├── __init__.py              # Main exports (SemiMarkov, BandedMatrix, semirings)
│   ├── semimarkov.py            # SemiMarkov class with DP algorithms
│   │                            #   - _dp_standard (scalar linear scan)
│   │                            #   - _dp_standard_vectorized (vectorized, 2-3x faster)
│   │                            #   - _dp_scan_streaming (O(KC) memory, default)
│   │                            #   - _dp_blocktriangular (structured sparsity)
│   ├── helpers.py               # Base _Struct class for structured prediction
│   ├── banded.py                # CPU BandedMatrix implementation
│   ├── banded_utils.py          # Bandwidth measurement, permutation utilities
│   ├── blocktriangular.py       # Block-triangular matrix operations
│   ├── triton_scan.py           # Triton fused streaming scan (optional GPU)
│   ├── semirings/
│   │   ├── semirings.py         # Log, Max, Std, KMax, Entropy, CrossEntropy
│   │   └── checkpoint.py        # CheckpointSemiring, CheckpointShardSemiring
│   └── _genbmm/                 # CUDA extension (optional)
│       ├── genmul.py            # PyTorch autograd functions
│       ├── sparse.py            # BandedMatrix with CUDA
│       └── csrc/                # C++/CUDA kernel sources
├── tests/                       # ~2,900 LOC across 15 test files
│   ├── conftest.py              # Pytest config (CPU-only enforcement)
│   ├── test_backend_equivalence.py
│   ├── test_semimarkov_utils.py
│   ├── test_hsmm.py
│   ├── test_numerical_gradients.py
│   ├── test_error_handling.py
│   ├── test_semimarkov_banded.py
│   ├── test_cpu_only.py
│   ├── test_partition_equivalence.py
│   ├── test_banded_matrix.py
│   ├── test_semirings.py
│   ├── test_blocktriangular.py
│   ├── test_banded_utils.py
│   ├── test_checkpoint_semiring.py
│   ├── test_helpers_cpu.py
│   └── test_triton_scan.py
├── benchmarks/
│   ├── benchmark_backends.py    # Multi-backend timing comparison
│   ├── benchmark_grid.py        # Parameter sweep benchmarks
│   ├── benchmark_memory_analysis.py
│   └── plot_figures.py          # Paper figure generation
├── docs/
│   ├── api.md                   # API reference
│   ├── backends.md              # Backend overview and Triton kernel
│   ├── benchmarks.md            # Benchmark recipes
│   ├── parameter_guide.md       # T/K/C parameter guide
│   └── disclosure.md            # AI-assisted development disclosure
├── pyproject.toml               # Modern Python packaging
└── setup.py                     # CUDA extension build
```

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Streaming Scan** | ✅ Complete | O(KC) memory, default backend |
| **Vectorized Scan** | ✅ Complete | O(TKC) memory, 2-3x faster |
| **Binary Tree** | ✅ Complete | O(log N) depth, high memory for large KC |
| **Block-Triangular** | ✅ Complete | Exploits duration constraint sparsity |
| **Semirings** | ✅ Complete | Log, Max, Std, KMax, Entropy, CrossEntropy |
| **Checkpoint Semiring** | ✅ Complete | Memory-efficient gradients |
| **BandedMatrix (CPU)** | ✅ Complete | Lightweight prototyping |
| **CUDA Extension** | ✅ Optional | Builds when nvcc available |
| **Triton Kernel** | ✅ Optional | ~45x speedup on GPU |
| **Test Suite** | ✅ Comprehensive | ~2,900 LOC, 15 test files |

## Citation

If you use this library, please cite:

```bibtex
@article{johnson2026semimarkov,
  title={Practical Semi-Markov CRF Inference for Genomic Sequence Annotation},
  author={Johnson, Benjamin K.},
  journal={bioRxiv},
  year={2026}
}
```

## Acknowledgments

This library builds on:
- [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) by Alexander Rush
- [genbmm](https://github.com/harvardnlp/genbmm) for CUDA generalized batch matrix multiplication

## License

MIT License - see [LICENSE](LICENSE) for details.
