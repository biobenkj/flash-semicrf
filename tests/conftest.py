"""
Pytest configuration for torch-semimarkov tests.

IMPORTANT: CPU-ONLY TESTING
---------------------------
This test suite is designed to run on CPU only. GPU support is not enabled
for CI due to cost constraints. The Triton kernels will automatically fall
back to CPU implementations when CUDA is not available.

All tests should:
1. Use CPU tensors (the default)
2. Not require CUDA to pass
3. Work correctly with the CPU fallback implementations
"""

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_cuda: mark test as requiring CUDA (will be skipped if not available)",
    )


@pytest.fixture(autouse=True)
def ensure_cpu_default():
    """
    Fixture that runs before each test to ensure we're using CPU.

    This is a documentation/verification fixture - it doesn't force CPU
    but warns if CUDA is being used unexpectedly.
    """
    # Just verify torch is available - tests should create CPU tensors by default
    assert torch.tensor([1.0]).device.type == "cpu", "Default device should be CPU"
    yield


@pytest.fixture
def cpu_device():
    """Fixture providing CPU device for explicit device specification."""
    return torch.device("cpu")


@pytest.fixture
def skip_if_no_cuda():
    """Fixture to skip tests that require CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
