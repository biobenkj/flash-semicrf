"""
Setup script for torch-semimarkov with optional CUDA extension.

The CUDA extension (genbmm) provides accelerated generalized batch matrix
multiplication for log-semiring, max-semiring, and banded operations.

To build with CUDA support:
    pip install -e . --config-settings="--build-option=--cuda"

Or manually:
    python setup.py build_ext --inplace

Without CUDA, the package uses pure PyTorch fallbacks.
"""

import os
import sys
from pathlib import Path

from setuptools import setup

# Check for CUDA build request
BUILD_CUDA = (
    "--cuda" in sys.argv
    or os.environ.get("TORCH_SEMIMARKOV_CUDA", "0") == "1"
)

if "--cuda" in sys.argv:
    sys.argv.remove("--cuda")


def get_cuda_extensions():
    """Build CUDA extensions if requested and available."""
    if not BUILD_CUDA:
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        print("Warning: torch.utils.cpp_extension not available, skipping CUDA build")
        return []

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, skipping CUDA extension build")
            return []
    except Exception:
        print("Warning: Could not check CUDA availability, skipping CUDA build")
        return []

    csrc_dir = Path(__file__).parent / "src" / "torch_semimarkov" / "_genbmm" / "csrc"

    if not csrc_dir.exists():
        print(f"Warning: CUDA source directory not found at {csrc_dir}")
        return []

    sources = [
        str(csrc_dir / "matmul_cuda.cpp"),
        str(csrc_dir / "matmul_cuda_kernel.cu"),
        str(csrc_dir / "banded_cuda_kernel.cu"),
    ]

    # Check all source files exist
    for src in sources:
        if not Path(src).exists():
            print(f"Warning: CUDA source file not found: {src}")
            return []

    return [
        CUDAExtension(
            name="torch_semimarkov._genbmm._C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]


def main():
    ext_modules = get_cuda_extensions()

    # Only use cmdclass if we have extensions to build
    if ext_modules:
        from torch.utils.cpp_extension import BuildExtension
        cmdclass = {"build_ext": BuildExtension}
    else:
        cmdclass = {}

    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )


if __name__ == "__main__":
    main()
