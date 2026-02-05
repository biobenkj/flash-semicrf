"""Tests for Triton boundary marginals implementation.

These tests verify that the Triton kernel for computing boundary marginals
matches the PyTorch reference implementation. They require CUDA and are
designed to be run on HPC with GPU compute.
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton tests"
)


@pytest.fixture
def cuda_device():
    """Fixture to ensure CUDA is available and return device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


class TestTritonMarginalsBasic:
    """Basic correctness tests for Triton marginals."""

    def test_triton_marginals_matches_pytorch(self, cuda_device):
        """Triton marginals should match PyTorch reference."""
        from torch_semimarkov.streaming import (
            HAS_TRITON,
            launch_streaming_triton_marginals,
            semi_crf_streaming_marginals_pytorch,
        )
        from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K = 4, 100, 8, 16

        # Setup - create cumulative scores
        scores = torch.randn(batch, T, C, device=cuda_device)
        scores = scores - scores.mean(dim=1, keepdim=True)  # Zero-center
        cum_scores = torch.zeros(batch, T + 1, C, device=cuda_device, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

        transition = torch.randn(C, C, device=cuda_device)
        duration_bias = torch.randn(K, C, device=cuda_device)
        lengths = torch.full((batch,), T, device=cuda_device, dtype=torch.long)

        # PyTorch reference (on CPU for comparison)
        pytorch_marginals, log_Z_pytorch = semi_crf_streaming_marginals_pytorch(
            cum_scores.cpu(),
            transition.cpu(),
            duration_bias.cpu(),
            lengths.cpu(),
            K,
        )

        # Triton (need forward pass first for checkpoints)
        log_Z_triton, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        triton_marginals = launch_streaming_triton_marginals(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z_triton,
            ring_ckpts,
            log_norm_ckpts,
            interval,
        )

        # Compare marginals
        torch.testing.assert_close(
            triton_marginals.cpu(),
            pytorch_marginals,
            rtol=0.01,
            atol=1e-5,
            msg="Triton marginals don't match PyTorch reference",
        )

        # Compare partition functions
        torch.testing.assert_close(
            log_Z_triton.cpu(),
            log_Z_pytorch,
            rtol=1e-5,
            atol=1e-5,
            check_dtype=False,
            msg="Partition functions don't match",
        )

    def test_triton_marginals_variable_lengths(self, cuda_device):
        """Marginals should handle variable sequence lengths correctly."""
        from torch_semimarkov.streaming import (
            HAS_TRITON,
            launch_streaming_triton_marginals,
            semi_crf_streaming_marginals_pytorch,
        )
        from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123)
        batch, T_max, C, K = 4, 100, 6, 12
        lengths_list = [100, 80, 60, 40]

        # Setup
        scores = torch.randn(batch, T_max, C, device=cuda_device)
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T_max + 1, C, device=cuda_device, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

        transition = torch.randn(C, C, device=cuda_device)
        duration_bias = torch.randn(K, C, device=cuda_device)
        lengths = torch.tensor(lengths_list, device=cuda_device, dtype=torch.long)

        # PyTorch reference
        pytorch_marginals, _ = semi_crf_streaming_marginals_pytorch(
            cum_scores.cpu(),
            transition.cpu(),
            duration_bias.cpu(),
            lengths.cpu(),
            K,
        )

        # Triton
        log_Z_triton, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        triton_marginals = launch_streaming_triton_marginals(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z_triton,
            ring_ckpts,
            log_norm_ckpts,
            interval,
        )

        # Compare
        torch.testing.assert_close(
            triton_marginals.cpu(),
            pytorch_marginals,
            rtol=0.01,
            atol=1e-5,
            msg="Variable length marginals don't match",
        )


class TestTritonMarginalsEdgeCases:
    """Edge case tests for Triton marginals."""

    def test_triton_marginals_k_equals_1(self, cuda_device):
        """Test with K=1 (only duration-1 segments)."""
        from torch_semimarkov.streaming import (
            HAS_TRITON,
            launch_streaming_triton_marginals,
            semi_crf_streaming_marginals_pytorch,
        )
        from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K = 2, 50, 4, 1

        scores = torch.randn(batch, T, C, device=cuda_device)
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=cuda_device, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

        transition = torch.randn(C, C, device=cuda_device)
        duration_bias = torch.randn(K, C, device=cuda_device)
        lengths = torch.full((batch,), T, device=cuda_device, dtype=torch.long)

        # PyTorch reference
        pytorch_marginals, _ = semi_crf_streaming_marginals_pytorch(
            cum_scores.cpu(),
            transition.cpu(),
            duration_bias.cpu(),
            lengths.cpu(),
            K,
        )

        # Triton
        log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        triton_marginals = launch_streaming_triton_marginals(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            interval,
        )

        torch.testing.assert_close(
            triton_marginals.cpu(),
            pytorch_marginals,
            rtol=0.01,
            atol=1e-5,
            msg="K=1 marginals don't match",
        )

    def test_triton_marginals_t_equals_k(self, cuda_device):
        """Test with T=K (sequence length equals max duration)."""
        from torch_semimarkov.streaming import (
            HAS_TRITON,
            launch_streaming_triton_marginals,
            semi_crf_streaming_marginals_pytorch,
        )
        from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K = 2, 16, 4, 16

        scores = torch.randn(batch, T, C, device=cuda_device)
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=cuda_device, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

        transition = torch.randn(C, C, device=cuda_device)
        duration_bias = torch.randn(K, C, device=cuda_device)
        lengths = torch.full((batch,), T, device=cuda_device, dtype=torch.long)

        # PyTorch reference
        pytorch_marginals, _ = semi_crf_streaming_marginals_pytorch(
            cum_scores.cpu(),
            transition.cpu(),
            duration_bias.cpu(),
            lengths.cpu(),
            K,
        )

        # Triton
        log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        triton_marginals = launch_streaming_triton_marginals(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            interval,
        )

        torch.testing.assert_close(
            triton_marginals.cpu(),
            pytorch_marginals,
            rtol=0.01,
            atol=1e-5,
            msg="T=K marginals don't match",
        )

    def test_triton_marginals_small_batch(self, cuda_device):
        """Test with batch=1."""
        from torch_semimarkov.streaming import (
            HAS_TRITON,
            launch_streaming_triton_marginals,
            semi_crf_streaming_marginals_pytorch,
        )
        from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K = 1, 100, 8, 16

        scores = torch.randn(batch, T, C, device=cuda_device)
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=cuda_device, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

        transition = torch.randn(C, C, device=cuda_device)
        duration_bias = torch.randn(K, C, device=cuda_device)
        lengths = torch.full((batch,), T, device=cuda_device, dtype=torch.long)

        # PyTorch reference
        pytorch_marginals, _ = semi_crf_streaming_marginals_pytorch(
            cum_scores.cpu(),
            transition.cpu(),
            duration_bias.cpu(),
            lengths.cpu(),
            K,
        )

        # Triton
        log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        triton_marginals = launch_streaming_triton_marginals(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            interval,
        )

        torch.testing.assert_close(
            triton_marginals.cpu(),
            pytorch_marginals,
            rtol=0.01,
            atol=1e-5,
            msg="Batch=1 marginals don't match",
        )

    def test_triton_marginals_large_k(self, cuda_device):
        """Test with large K (many possible durations)."""
        from torch_semimarkov.streaming import (
            HAS_TRITON,
            launch_streaming_triton_marginals,
            semi_crf_streaming_marginals_pytorch,
        )
        from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K = 2, 200, 4, 64

        scores = torch.randn(batch, T, C, device=cuda_device)
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=cuda_device, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

        transition = torch.randn(C, C, device=cuda_device)
        duration_bias = torch.randn(K, C, device=cuda_device)
        lengths = torch.full((batch,), T, device=cuda_device, dtype=torch.long)

        # PyTorch reference
        pytorch_marginals, _ = semi_crf_streaming_marginals_pytorch(
            cum_scores.cpu(),
            transition.cpu(),
            duration_bias.cpu(),
            lengths.cpu(),
            K,
        )

        # Triton
        log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        triton_marginals = launch_streaming_triton_marginals(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            interval,
        )

        torch.testing.assert_close(
            triton_marginals.cpu(),
            pytorch_marginals,
            rtol=0.01,
            atol=1e-5,
            msg="Large K marginals don't match",
        )


class TestTritonMarginalsIntegration:
    """Integration tests with UncertaintyMixin."""

    def test_uncertainty_mixin_uses_triton_on_cuda(self, cuda_device):
        """UncertaintyMixin should use Triton kernel on CUDA."""
        from torch_semimarkov.streaming import HAS_TRITON

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        from torch_semimarkov import UncertaintySemiMarkovCRFHead

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K, hidden_dim = 2, 100, 8, 16, 64

        model = UncertaintySemiMarkovCRFHead(
            num_classes=C,
            max_duration=K,
            hidden_dim=hidden_dim,
        ).to(cuda_device)

        hidden = torch.randn(batch, T, hidden_dim, device=cuda_device)
        lengths = torch.tensor([100, 80], device=cuda_device)

        # Compute marginals using streaming backend (should use Triton internally)
        marginals_streaming = model.compute_boundary_marginals(
            hidden, lengths, backend="streaming", normalize=False
        )

        # Verify shape
        assert marginals_streaming.shape == (batch, T)

        # Verify on correct device (compare type, not exact device index)
        assert marginals_streaming.device.type == cuda_device.type

        # Verify values are reasonable (probabilities should be positive)
        assert (marginals_streaming >= 0).all()

    def test_streaming_matches_exact_on_cuda(self, cuda_device):
        """Streaming and exact backends should match on CUDA."""
        from torch_semimarkov.streaming import HAS_TRITON

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        from torch_semimarkov import UncertaintySemiMarkovCRFHead

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K, hidden_dim = 2, 50, 6, 12, 32

        model = UncertaintySemiMarkovCRFHead(
            num_classes=C,
            max_duration=K,
            hidden_dim=hidden_dim,
        ).to(cuda_device)

        hidden = torch.randn(batch, T, hidden_dim, device=cuda_device)
        lengths = torch.full((batch,), T, device=cuda_device)

        # Compute with both backends
        marginals_streaming = model.compute_boundary_marginals(
            hidden, lengths, backend="streaming", normalize=False
        )
        marginals_exact = model.compute_boundary_marginals(
            hidden, lengths, backend="exact", normalize=False
        )

        # Should match closely
        torch.testing.assert_close(
            marginals_streaming,
            marginals_exact,
            rtol=0.01,
            atol=1e-5,
            msg="Streaming and exact marginals don't match on CUDA",
        )


class TestBackwardCompatibility:
    """Test that existing backward functionality still works."""

    def test_backward_still_computes_gradients(self, cuda_device):
        """Backward kernel should still compute gradients correctly."""
        from torch_semimarkov.streaming import (
            HAS_TRITON,
        )
        from torch_semimarkov.streaming.triton_backward import launch_streaming_triton_backward
        from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K = 4, 50, 6, 12

        scores = torch.randn(batch, T, C, device=cuda_device)
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=cuda_device, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

        transition = torch.randn(C, C, device=cuda_device)
        duration_bias = torch.randn(K, C, device=cuda_device)
        lengths = torch.full((batch,), T, device=cuda_device, dtype=torch.long)
        grad_output = torch.ones(batch, device=cuda_device)

        # Forward
        log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Backward with return_boundary_marginals=False (default)
        (
            grad_cs,
            grad_tr,
            grad_db,
            grad_ps,
            grad_pe,
            boundary_marginals,
        ) = launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            interval,
            grad_output,
            return_boundary_marginals=False,
        )

        # Gradients should be computed
        assert grad_cs is not None
        assert grad_tr is not None
        assert grad_db is not None

        # Boundary marginals should be None when not requested
        assert boundary_marginals is None

        # Gradients should be finite
        assert torch.isfinite(grad_cs).all()
        assert torch.isfinite(grad_tr).all()
        assert torch.isfinite(grad_db).all()

    def test_backward_with_marginals_still_computes_gradients(self, cuda_device):
        """Backward kernel with marginals should still compute gradients."""
        from torch_semimarkov.streaming import HAS_TRITON
        from torch_semimarkov.streaming.triton_backward import launch_streaming_triton_backward
        from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

        if not HAS_TRITON:
            pytest.skip("Triton not available")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch, T, C, K = 4, 50, 6, 12

        scores = torch.randn(batch, T, C, device=cuda_device)
        scores = scores - scores.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=cuda_device, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

        transition = torch.randn(C, C, device=cuda_device)
        duration_bias = torch.randn(K, C, device=cuda_device)
        lengths = torch.full((batch,), T, device=cuda_device, dtype=torch.long)
        grad_output = torch.ones(batch, device=cuda_device)

        # Forward
        log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Backward with return_boundary_marginals=True
        (
            grad_cs,
            grad_tr,
            grad_db,
            grad_ps,
            grad_pe,
            boundary_marginals,
        ) = launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            interval,
            grad_output,
            return_boundary_marginals=True,
        )

        # Gradients should still be computed
        assert grad_cs is not None
        assert grad_tr is not None
        assert grad_db is not None

        # Boundary marginals should be returned
        assert boundary_marginals is not None
        assert boundary_marginals.shape == (batch, T)

        # All should be finite
        assert torch.isfinite(grad_cs).all()
        assert torch.isfinite(grad_tr).all()
        assert torch.isfinite(grad_db).all()
        assert torch.isfinite(boundary_marginals).all()

        # Boundary marginals should be non-negative (probabilities)
        assert (boundary_marginals >= 0).all()
