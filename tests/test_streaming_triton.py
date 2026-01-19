"""
Tests for the Triton streaming forward kernel.

Validates the Triton implementation against the PyTorch reference.
"""

import pytest
import torch

from torch_semimarkov.streaming import (
    semi_crf_streaming_forward_pytorch,
    HAS_TRITON,
)

# Conditionally import Triton functions
if HAS_TRITON:
    from torch_semimarkov.streaming import launch_streaming_triton_kernel


def create_golden_rule_inputs(batch, T, K, C, device="cpu", dtype=torch.float32, seed=42):
    """Create test inputs for the Golden Rule streaming API."""
    torch.manual_seed(seed)

    # Simulate projected encoder features
    projected = torch.randn(batch, T, C, device=device, dtype=dtype)
    # Zero-center (critical for numerical stability at large T)
    projected = projected - projected.mean(dim=1, keepdim=True)

    # Cumulative scores: (batch, T+1, C)
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
    cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

    # Transition matrix: (C, C)
    transition = torch.randn(C, C, device=device, dtype=dtype) * 0.1

    # Duration bias: (K, C)
    duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1

    # Lengths
    lengths = torch.full((batch,), T, dtype=torch.long, device=device)

    return cum_scores, transition, duration_bias, lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingKernel:
    """Test the Triton streaming forward kernel against PyTorch reference."""

    def test_triton_matches_pytorch_small(self):
        """Verify Triton kernel matches PyTorch for small inputs."""
        batch, T, K, C = 2, 20, 4, 4
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Triton kernel
        partition_triton, ring_ckpts, ckpt_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4
        )

    def test_triton_matches_pytorch_medium(self):
        """Verify Triton kernel matches PyTorch for medium inputs."""
        batch, T, K, C = 4, 100, 8, 6
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4
        )

    def test_triton_matches_pytorch_large_C(self):
        """Verify Triton kernel works with larger C (requires padding)."""
        batch, T, K, C = 2, 50, 6, 24  # C=24 -> C_PAD=32
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4
        )

    def test_triton_matches_pytorch_non_power_of_2_C(self):
        """Verify Triton kernel handles non-power-of-2 C values."""
        for C in [3, 5, 7, 11, 13, 17]:
            batch, T, K = 2, 30, 5
            cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
                batch, T, K, C, device="cuda"
            )

            partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores, transition, duration_bias, lengths, K
            )

            partition_triton, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K
            )

            torch.testing.assert_close(
                partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4,
                msg=f"C={C} failed"
            )

    def test_triton_variable_lengths(self):
        """Verify Triton kernel handles variable sequence lengths."""
        batch, T, K, C = 4, 50, 6, 4
        cum_scores, transition, duration_bias, _ = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )
        lengths = torch.tensor([T, T - 10, T - 20, T - 30], dtype=torch.long, device="cuda")

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4
        )

    def test_triton_short_sequences(self):
        """Verify Triton kernel handles sequences shorter than K."""
        K, C, batch = 10, 4, 2

        for T in [4, 6, 8, 12]:
            cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
                batch, T, K, C, device="cuda"
            )

            partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores, transition, duration_bias, lengths, K
            )

            partition_triton, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K
            )

            torch.testing.assert_close(
                partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4,
                msg=f"T={T} failed"
            )

    def test_triton_max_semiring(self):
        """Verify Triton max semiring matches PyTorch."""
        batch, T, K, C = 2, 50, 6, 4
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4
        )

    def test_triton_produces_finite_values(self):
        """Verify Triton kernel produces finite values."""
        batch, T, K, C = 4, 100, 8, 6
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_triton, ring_ckpts, ckpt_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        assert torch.isfinite(partition_triton).all(), "Partition contains non-finite values"

    def test_triton_checkpoints_saved(self):
        """Verify Triton kernel saves checkpoints correctly."""
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        _, ring_ckpts, ckpt_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Verify checkpoint shape
        expected_num_ckpts = (T + ckpt_interval - 1) // ckpt_interval + 1
        assert ring_ckpts.shape[1] == expected_num_ckpts, f"Expected {expected_num_ckpts} checkpoints"
        assert ring_ckpts.shape[2] == K
        assert ring_ckpts.shape[3] == C

        # Verify checkpoint 0 is initialized correctly (alpha[0] = 0, rest = -inf)
        ckpt_0 = ring_ckpts[:, 0, :, :]  # (batch, K, C)
        assert torch.allclose(ckpt_0[:, 0, :], torch.zeros_like(ckpt_0[:, 0, :]))

    def test_triton_larger_sequence(self):
        """Verify Triton kernel works with larger sequences."""
        batch, T, K, C = 2, 500, 16, 8
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Use slightly looser tolerance for longer sequences
        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-3, atol=1e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingBenchmark:
    """Benchmark tests for Triton streaming kernel."""

    @pytest.mark.parametrize("T", [100, 500, 1000])
    @pytest.mark.parametrize("K", [8, 16, 32])
    def test_benchmark_correctness(self, T, K):
        """Verify correctness across different T and K values."""
        batch, C = 4, 6
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-3, atol=1e-3,
            msg=f"T={T}, K={K} failed"
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
    elif not HAS_TRITON:
        print("Triton not available, skipping tests")
    else:
        print("Running Triton streaming kernel validation...")

        # Quick validation
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, lengths = create_golden_rule_inputs(
            batch, T, K, C, device="cuda"
        )

        print(f"\nInput shapes:")
        print(f"  cum_scores: {cum_scores.shape}")
        print(f"  transition: {transition.shape}")
        print(f"  duration_bias: {duration_bias.shape}")
        print(f"  lengths: {lengths}")

        # PyTorch reference
        partition_pytorch, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )
        print(f"\nPyTorch partition: {partition_pytorch}")

        # Triton kernel
        partition_triton, ring_ckpts, ckpt_interval = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        print(f"Triton partition:  {partition_triton}")
        print(f"Checkpoint interval: {ckpt_interval}")
        print(f"Ring checkpoints shape: {ring_ckpts.shape}")

        # Compare
        diff = (partition_triton - partition_pytorch).abs().max().item()
        print(f"\nMax difference: {diff:.6e}")

        if diff < 1e-4:
            print("PASSED: Triton matches PyTorch reference")
        else:
            print("FAILED: Triton does not match PyTorch reference")
