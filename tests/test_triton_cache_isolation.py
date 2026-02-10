"""Test to verify Triton cache isolation between tests.

This test verifies that the clear_triton_cache fixture properly prevents
test order dependencies caused by Triton kernel cache contamination.
"""

import pytest
import torch

from tests.conftest import force_clear_triton_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTritonCacheIsolation:
    """Tests to verify Triton cache is properly cleared between tests."""

    def test_cache_baseline_small(self):
        """First test with small configuration to populate cache."""
        from flash_semicrf.streaming import HAS_TRITON

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        from flash_semicrf.streaming import (
            launch_streaming_triton_kernel,
            semi_crf_streaming_forward_pytorch,
        )

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        batch, T, K, C = 2, 50, 8, 4
        scores = torch.randn(batch, T, C, device="cuda")
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda", dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)
        transition = torch.randn(C, C, device="cuda") * 0.1
        duration_bias = torch.randn(K, C, device="cuda") * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )
        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_cache_different_large(self):
        """Second test with large configuration - should get fresh cache.

        This test uses the same configuration as test_triton_larger_sequence.
        If the cache isn't cleared between tests, this might fail due to
        reusing kernels compiled with parameters from test_cache_baseline_small.
        """
        from flash_semicrf.streaming import HAS_TRITON

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        # Explicitly clear cache to demonstrate the helper function usage
        force_clear_triton_cache()

        from flash_semicrf.streaming import (
            launch_streaming_triton_kernel,
            semi_crf_streaming_forward_pytorch,
        )

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        batch, T, K, C = 2, 500, 16, 8
        scores = torch.randn(batch, T, C, device="cuda")
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda", dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)
        transition = torch.randn(C, C, device="cuda") * 0.1
        duration_bias = torch.randn(K, C, device="cuda") * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )
        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Use same tolerance as test_triton_larger_sequence
        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-3, atol=1e-3, check_dtype=False
        )

    def test_cache_verify_determinism(self):
        """Verify that running the same config twice gives identical results.

        This test runs the same kernel twice in a row and verifies the results
        are identical (not just close, but exactly the same). This would fail
        if there's any non-determinism in the kernel or cache behavior.
        """
        from flash_semicrf.streaming import HAS_TRITON

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        from flash_semicrf.streaming import launch_streaming_triton_kernel

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        batch, T, K, C = 2, 100, 8, 4
        scores = torch.randn(batch, T, C, device="cuda")
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda", dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)
        transition = torch.randn(C, C, device="cuda") * 0.1
        duration_bias = torch.randn(K, C, device="cuda") * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        # Run twice with the same inputs
        partition_1, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        partition_2, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Should be exactly identical (not just close)
        assert torch.equal(
            partition_1, partition_2
        ), "Running the same kernel twice should give identical results"
