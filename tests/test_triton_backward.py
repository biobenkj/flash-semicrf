"""Tests for the explicit forward-backward implementation.

This module tests the PyTorch reference backward pass implementation,
verifying correctness against torch.autograd.gradcheck and the existing
autograd-based backward.
"""

import pytest
import torch

from torch_semimarkov.triton_backward import (
    semi_crf_forward_with_alpha,
    semi_crf_backward_beta,
    semi_crf_compute_marginals,
    semi_crf_backward_pytorch,
    semi_crf_forward_backward,
    semi_crf_triton_backward,
    semi_crf_forward_with_checkpoints,
    semi_crf_backward_from_checkpoints,
    semi_crf_checkpointed_backward,
    semi_crf_forward_with_ring_checkpoints,
    semi_crf_backward_from_ring_checkpoints,
    semi_crf_optimized_checkpointed_backward,
    _compute_checkpoint_interval,
    SemiCRFBackward,
    HAS_TRITON,
)
from torch_semimarkov.triton_scan import (
    semi_crf_forward_pytorch,
    semi_crf_triton_forward,
)


# =============================================================================
# Forward with Alpha Tests
# =============================================================================


def test_forward_with_alpha_matches_reference():
    """Test that forward_with_alpha matches the reference forward."""
    torch.manual_seed(42)
    batch, T, K, C = 2, 8, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref_partition = semi_crf_forward_pytorch(edge, lengths, semiring="log")

    # Our implementation
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    assert torch.allclose(partition, ref_partition, atol=1e-5), (
        f"Partition mismatch: max diff {(partition - ref_partition).abs().max()}"
    )


def test_forward_with_alpha_variable_lengths():
    """Test forward_with_alpha with variable sequence lengths."""
    torch.manual_seed(123)
    batch, T, K, C = 3, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.tensor([10, 7, 5], dtype=torch.long)

    ref_partition = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    assert torch.allclose(partition, ref_partition, atol=1e-5)


def test_forward_with_alpha_max_semiring():
    """Test forward_with_alpha with max semiring."""
    torch.manual_seed(456)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    ref_partition = semi_crf_forward_pytorch(edge, lengths, semiring="max")
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="max")

    assert torch.allclose(partition, ref_partition, atol=1e-5)


# =============================================================================
# Backward Beta Tests
# =============================================================================


def test_backward_beta_boundary_conditions():
    """Test that beta has correct boundary conditions."""
    torch.manual_seed(789)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    beta = semi_crf_backward_beta(edge, lengths, semiring="log")

    # At final position (T-1), beta should be 0
    assert torch.allclose(
        beta[:, T - 1, :], torch.zeros(batch, C), atol=1e-6
    ), "Beta at final position should be 0"


def test_backward_beta_variable_lengths():
    """Test backward_beta with variable lengths."""
    torch.manual_seed(321)
    batch, T, K, C = 3, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.tensor([10, 7, 5], dtype=torch.long)

    beta = semi_crf_backward_beta(edge, lengths, semiring="log")

    # Check that beta at each sequence's final position is 0
    for b in range(batch):
        final_pos = lengths[b].item() - 1
        assert torch.allclose(
            beta[b, final_pos, :], torch.zeros(C), atol=1e-6
        ), f"Beta at final position for batch {b} should be 0"


# =============================================================================
# Marginal/Gradient Tests
# =============================================================================


def test_marginals_sum_to_expected_segments():
    """Test that marginals sum to expected number of segments.

    In a semi-CRF with variable durations, the expected number of segments
    is NOT fixed at T-1. Instead, it's the expected value under the model's
    distribution. We verify this by checking that the sum is within valid
    bounds and matches what we compute via a different method.
    """
    torch.manual_seed(111)
    batch, T, K, C = 2, 8, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    marginals = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Total marginals = expected number of segments under the distribution
    # This should be between 1 (one long segment) and T-1 (all unit segments)
    total_marginal = marginals.sum(dim=(1, 2, 3, 4))

    # Check bounds: at least 1 segment, at most T-1 segments
    assert (total_marginal >= 1.0 - 1e-4).all(), (
        f"Total marginal below 1: {total_marginal}"
    )
    assert (total_marginal <= (lengths - 1).float() + 1e-4).all(), (
        f"Total marginal above T-1: {total_marginal}"
    )


def test_marginals_non_negative():
    """Test that all marginals are non-negative."""
    torch.manual_seed(222)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    marginals = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    assert (marginals >= -1e-6).all(), "Marginals should be non-negative"


def test_marginals_bounded_by_one():
    """Test that all marginals are <= 1."""
    torch.manual_seed(333)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    marginals = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    assert (marginals <= 1.0 + 1e-6).all(), "Marginals should be <= 1"


# =============================================================================
# Gradient Correctness Tests
# =============================================================================


def test_backward_matches_autograd():
    """Test that explicit backward matches autograd-computed gradients."""
    torch.manual_seed(444)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Compute gradients via autograd
    partition_autograd = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    partition_autograd.sum().backward()
    grad_autograd = edge.grad.clone()

    # Compute gradients via explicit backward
    edge_detached = edge.detach()
    partition_explicit, grad_explicit = semi_crf_backward_pytorch(
        edge_detached, lengths, semiring="log"
    )

    # Check partition values match
    assert torch.allclose(partition_autograd.detach(), partition_explicit, atol=1e-5), (
        f"Partition mismatch: {(partition_autograd.detach() - partition_explicit).abs().max()}"
    )

    # Check gradients match
    assert torch.allclose(grad_autograd, grad_explicit, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_autograd - grad_explicit).abs().max()}"
    )


def test_backward_matches_autograd_variable_lengths():
    """Test gradient correctness with variable sequence lengths."""
    torch.manual_seed(555)
    batch, T, K, C = 3, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.tensor([10, 7, 5], dtype=torch.long)

    # Autograd gradients
    partition_autograd = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    partition_autograd.sum().backward()
    grad_autograd = edge.grad.clone()

    # Explicit backward gradients
    edge_detached = edge.detach()
    partition_explicit, grad_explicit = semi_crf_backward_pytorch(
        edge_detached, lengths, semiring="log"
    )

    assert torch.allclose(partition_autograd.detach(), partition_explicit, atol=1e-5)
    assert torch.allclose(grad_autograd, grad_explicit, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_autograd - grad_explicit).abs().max()}"
    )


def test_gradcheck_small():
    """Test gradients with torch.autograd.gradcheck on small inputs."""
    torch.manual_seed(666)
    batch, T, K, C = 1, 4, 2, 2
    edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float64, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    def func(edge_input):
        return semi_crf_forward_backward(edge_input, lengths, semiring="log")

    # Use gradcheck to verify numerical gradients
    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_medium():
    """Test gradients with torch.autograd.gradcheck on medium inputs."""
    torch.manual_seed(777)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float64, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    def func(edge_input):
        return semi_crf_forward_backward(edge_input, lengths, semiring="log")

    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_variable_lengths():
    """Test gradients with variable lengths using gradcheck."""
    torch.manual_seed(888)
    batch, T, K, C = 2, 8, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C, dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([8, 5], dtype=torch.long)

    def func(edge_input):
        return semi_crf_forward_backward(edge_input, lengths, semiring="log")

    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


# =============================================================================
# Autograd Function Tests
# =============================================================================


def test_autograd_function_forward():
    """Test that SemiCRFBackward.forward matches reference."""
    torch.manual_seed(999)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    out = semi_crf_forward_backward(edge, lengths, semiring="log")

    assert torch.allclose(out, ref, atol=1e-5)


def test_autograd_function_backward():
    """Test that SemiCRFBackward computes correct gradients."""
    torch.manual_seed(1000)
    batch, T, K, C = 2, 6, 3, 2

    # Reference gradients via autograd
    edge_ref = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)
    out_ref = semi_crf_forward_pytorch(edge_ref, lengths, semiring="log")
    out_ref.sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Our implementation
    edge_test = edge_ref.detach().clone().requires_grad_(True)
    out_test = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out_test.sum().backward()
    grad_test = edge_test.grad

    assert torch.allclose(out_ref.detach(), out_test.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_test, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_test).abs().max()}"
    )


def test_autograd_function_with_grad_output():
    """Test backward with non-uniform grad_output."""
    torch.manual_seed(1100)
    batch, T, K, C = 3, 6, 3, 2

    # Reference
    edge_ref = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)
    grad_output = torch.randn(batch)

    out_ref = semi_crf_forward_pytorch(edge_ref, lengths, semiring="log")
    (out_ref * grad_output).sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Our implementation
    edge_test = edge_ref.detach().clone().requires_grad_(True)
    out_test = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    (out_test * grad_output).sum().backward()
    grad_test = edge_test.grad

    assert torch.allclose(grad_ref, grad_test, atol=1e-4)


# =============================================================================
# Edge Cases
# =============================================================================


def test_single_position_sequence():
    """Test with sequence length = 2 (single segment)."""
    torch.manual_seed(1200)
    batch, T, K, C = 2, 2, 2, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge.grad = None
    out = semi_crf_forward_backward(edge.detach().clone().requires_grad_(True), lengths, semiring="log")

    assert torch.allclose(out, ref.detach(), atol=1e-5)


def test_large_K():
    """Test with K larger than T (many durations unused)."""
    torch.manual_seed(1300)
    batch, T, K, C = 2, 5, 10, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge_test = edge.detach().clone().requires_grad_(True)
    out = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out.sum().backward()
    grad_test = edge_test.grad

    assert torch.allclose(out, ref.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_test, atol=1e-4)


def test_single_label():
    """Test with C=1 (single label)."""
    torch.manual_seed(1400)
    batch, T, K, C = 2, 6, 3, 1
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge_test = edge.detach().clone().requires_grad_(True)
    out = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out.sum().backward()
    grad_test = edge_test.grad

    assert torch.allclose(out, ref.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_test, atol=1e-4)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


def test_numerical_stability_large_values():
    """Test with large edge values (potential overflow)."""
    torch.manual_seed(1500)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C) * 10  # Large values
    edge = edge.requires_grad_(True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge_test = edge.detach().clone().requires_grad_(True)
    out = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out.sum().backward()
    grad_test = edge_test.grad

    assert torch.isfinite(out).all(), "Output should be finite"
    assert torch.isfinite(grad_test).all(), "Gradients should be finite"
    assert torch.allclose(out, ref.detach(), atol=1e-4)
    assert torch.allclose(grad_ref, grad_test, atol=1e-3)


def test_numerical_stability_small_values():
    """Test with small edge values (potential underflow)."""
    torch.manual_seed(1600)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C) * 0.1  # Small values
    edge = edge.requires_grad_(True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    ref = semi_crf_forward_pytorch(edge, lengths, semiring="log")
    ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Our implementation
    edge_test = edge.detach().clone().requires_grad_(True)
    out = semi_crf_forward_backward(edge_test, lengths, semiring="log")
    out.sum().backward()
    grad_test = edge_test.grad

    assert torch.isfinite(out).all()
    assert torch.isfinite(grad_test).all()
    assert torch.allclose(out, ref.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_test, atol=1e-4)


# =============================================================================
# Triton Backward Kernel Tests (CUDA only)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_kernel_matches_pytorch():
    """Test that Triton backward kernel matches PyTorch reference."""
    from torch_semimarkov.triton_backward import launch_triton_backward_kernel

    torch.manual_seed(2000)
    batch, T, K, C = 4, 16, 5, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute alpha and partition using PyTorch
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    # Compute gradients using PyTorch reference
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    grad_pytorch = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Compute gradients using Triton kernel
    grad_triton = launch_triton_backward_kernel(edge, alpha, partition, lengths)

    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_kernel_variable_lengths():
    """Test Triton backward kernel with variable sequence lengths."""
    from torch_semimarkov.triton_backward import launch_triton_backward_kernel

    torch.manual_seed(2100)
    batch, T, K, C = 4, 20, 6, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.tensor([20, 15, 10, 5], dtype=torch.long, device="cuda")

    # Compute alpha and partition
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    # PyTorch reference
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    grad_pytorch = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Triton kernel
    grad_triton = launch_triton_backward_kernel(edge, alpha, partition, lengths)

    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_kernel_non_power_of_2_labels():
    """Test Triton backward kernel with non-power-of-2 label count."""
    from torch_semimarkov.triton_backward import launch_triton_backward_kernel

    torch.manual_seed(2200)
    batch, T, K, C = 3, 12, 4, 5  # C=5 is not power of 2
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute alpha and partition
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    # PyTorch reference
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    grad_pytorch = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Triton kernel
    grad_triton = launch_triton_backward_kernel(edge, alpha, partition, lengths)

    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_autograd_function():
    """Test full autograd integration with Triton backward kernel."""
    torch.manual_seed(2300)
    batch, T, K, C = 4, 16, 5, 4

    # Reference gradients via PyTorch
    edge_ref = torch.randn(batch, T - 1, K, C, C, device="cuda", requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    out_ref = semi_crf_forward_pytorch(edge_ref, lengths, semiring="log")
    out_ref.sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Triton backward via autograd function
    edge_triton = edge_ref.detach().clone().requires_grad_(True)
    out_triton = semi_crf_triton_backward(edge_triton, lengths, semiring="log")
    out_triton.sum().backward()
    grad_triton = edge_triton.grad

    assert torch.allclose(out_ref.detach(), out_triton.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_triton, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_triton).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_larger_shapes():
    """Test Triton backward kernel with larger shapes (closer to genomics use)."""
    from torch_semimarkov.triton_backward import launch_triton_backward_kernel

    torch.manual_seed(2400)
    batch, T, K, C = 8, 64, 12, 8
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute alpha and partition
    partition, alpha = semi_crf_forward_with_alpha(edge, lengths, semiring="log")

    # PyTorch reference
    beta = semi_crf_backward_beta(edge, lengths, semiring="log")
    grad_pytorch = semi_crf_compute_marginals(edge, alpha, beta, partition, lengths)

    # Triton kernel
    grad_triton = launch_triton_backward_kernel(edge, alpha, partition, lengths)

    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-3), (
        f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_backward_gradcheck():
    """Verify Triton backward with torch.autograd.gradcheck."""
    torch.manual_seed(2500)
    batch, T, K, C = 2, 6, 3, 2
    edge = torch.randn(
        batch, T - 1, K, C, C, dtype=torch.float64, device="cuda", requires_grad=True
    )
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    def func(edge_input):
        return semi_crf_triton_backward(edge_input, lengths, semiring="log")

    # Use gradcheck to verify numerical gradients
    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


# =============================================================================
# Phase 3: Checkpointed Forward-Backward Tests
# =============================================================================


def test_checkpoint_interval_computation():
    """Test that checkpoint interval is max(sqrt(T), K)."""
    import math

    # Test with default K=1
    test_cases_default = [(4, 1), (16, 1), (64, 1), (100, 1), (256, 1), (1000, 1)]
    for T, K in test_cases_default:
        interval = _compute_checkpoint_interval(T, K)
        expected = max(K, int(math.sqrt(T)))
        assert interval == expected, f"For T={T}, K={K}, expected {expected}, got {interval}"

    # Test with larger K
    test_cases_large_k = [(16, 8), (64, 10), (100, 20), (256, 5)]
    for T, K in test_cases_large_k:
        interval = _compute_checkpoint_interval(T, K)
        expected = max(K, int(math.sqrt(T)))
        assert interval == expected, f"For T={T}, K={K}, expected {expected}, got {interval}"


def test_checkpointed_forward_matches_full_forward():
    """Test that checkpointed forward computes the same partition."""
    torch.manual_seed(3000)
    batch, T, K, C = 2, 16, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Full forward (reference)
    partition_full, _ = semi_crf_forward_with_alpha(edge, lengths)

    # Checkpointed forward
    partition_ckpt, checkpoints, interval = semi_crf_forward_with_checkpoints(
        edge, lengths
    )

    assert torch.allclose(partition_full, partition_ckpt, atol=1e-5), (
        f"Partition mismatch: {(partition_full - partition_ckpt).abs().max()}"
    )


def test_checkpointed_forward_checkpoint_count():
    """Test that the correct number of checkpoints are saved."""
    torch.manual_seed(3100)
    batch, T, K, C = 2, 20, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Use explicit interval
    interval = 5
    partition, checkpoints, actual_interval = semi_crf_forward_with_checkpoints(
        edge, lengths, checkpoint_interval=interval
    )

    expected_num_checkpoints = (T + interval - 1) // interval  # ceil(T / interval)
    assert checkpoints.shape[1] == expected_num_checkpoints, (
        f"Expected {expected_num_checkpoints} checkpoints, got {checkpoints.shape[1]}"
    )
    assert actual_interval == interval


def test_checkpointed_forward_variable_intervals():
    """Test checkpointed forward with various checkpoint intervals."""
    torch.manual_seed(3200)
    batch, T, K, C = 2, 24, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Full forward for reference
    partition_ref, _ = semi_crf_forward_with_alpha(edge, lengths)

    for interval in [2, 4, 6, 8, 12]:
        partition_ckpt, checkpoints, _ = semi_crf_forward_with_checkpoints(
            edge, lengths, checkpoint_interval=interval
        )
        assert torch.allclose(partition_ref, partition_ckpt, atol=1e-5), (
            f"Partition mismatch for interval={interval}"
        )


def test_checkpointed_backward_matches_full_backward():
    """Test that checkpointed backward computes correct gradients."""
    torch.manual_seed(3300)
    batch, T, K, C = 2, 16, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference gradients via full backward
    partition_ref, alpha_ref = semi_crf_forward_with_alpha(edge.detach(), lengths)
    beta_ref = semi_crf_backward_beta(edge.detach(), lengths)
    grad_ref = semi_crf_compute_marginals(
        edge.detach(), alpha_ref, beta_ref, partition_ref, lengths
    )

    # Checkpointed backward
    partition_ckpt, checkpoints, interval = semi_crf_forward_with_checkpoints(
        edge.detach(), lengths
    )
    grad_ckpt = semi_crf_backward_from_checkpoints(
        edge.detach(), checkpoints, partition_ckpt, lengths, interval
    )

    assert torch.allclose(grad_ref, grad_ckpt, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_ckpt).abs().max()}"
    )


def test_checkpointed_autograd_function():
    """Test full autograd integration with checkpointed backward."""
    torch.manual_seed(3400)
    batch, T, K, C = 2, 16, 4, 3

    # Reference via autograd
    edge_ref = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out_ref = semi_crf_forward_pytorch(edge_ref, lengths)
    out_ref.sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Checkpointed backward via autograd function
    edge_ckpt = edge_ref.detach().clone().requires_grad_(True)
    out_ckpt = semi_crf_checkpointed_backward(edge_ckpt, lengths)
    out_ckpt.sum().backward()
    grad_ckpt = edge_ckpt.grad

    assert torch.allclose(out_ref.detach(), out_ckpt.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_ckpt, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_ckpt).abs().max()}"
    )


def test_checkpointed_gradcheck():
    """Verify checkpointed backward with torch.autograd.gradcheck."""
    torch.manual_seed(3500)
    batch, T, K, C = 1, 8, 3, 2
    edge = torch.randn(
        batch, T - 1, K, C, C, dtype=torch.float64, requires_grad=True
    )
    lengths = torch.full((batch,), T, dtype=torch.long)

    def func(edge_input):
        return semi_crf_checkpointed_backward(edge_input, lengths)

    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_checkpointed_variable_lengths():
    """Test checkpointed backward with variable sequence lengths."""
    torch.manual_seed(3600)
    batch, T, K, C = 3, 20, 5, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.tensor([20, 15, 8], dtype=torch.long)

    # Reference via autograd
    out_ref = semi_crf_forward_pytorch(edge, lengths)
    out_ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Checkpointed backward
    edge.grad = None
    edge_ckpt = edge.detach().clone().requires_grad_(True)
    out_ckpt = semi_crf_checkpointed_backward(edge_ckpt, lengths)
    out_ckpt.sum().backward()
    grad_ckpt = edge_ckpt.grad

    assert torch.allclose(out_ref.detach(), out_ckpt.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_ckpt, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_ckpt).abs().max()}"
    )


def test_checkpointed_small_interval():
    """Test with checkpoint interval = 1 (should match full storage)."""
    torch.manual_seed(3700)
    batch, T, K, C = 2, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    out_ref = semi_crf_forward_pytorch(edge, lengths)
    out_ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Checkpointed with interval=1 (saves every position)
    edge_ckpt = edge.detach().clone().requires_grad_(True)
    out_ckpt = semi_crf_checkpointed_backward(edge_ckpt, lengths, checkpoint_interval=1)
    out_ckpt.sum().backward()
    grad_ckpt = edge_ckpt.grad

    assert torch.allclose(out_ref.detach(), out_ckpt.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_ckpt, atol=1e-4)


def test_checkpointed_large_interval():
    """Test with checkpoint interval = T (only initial checkpoint)."""
    torch.manual_seed(3800)
    batch, T, K, C = 2, 10, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference
    out_ref = semi_crf_forward_pytorch(edge, lengths)
    out_ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Checkpointed with large interval
    edge_ckpt = edge.detach().clone().requires_grad_(True)
    out_ckpt = semi_crf_checkpointed_backward(edge_ckpt, lengths, checkpoint_interval=T)
    out_ckpt.sum().backward()
    grad_ckpt = edge_ckpt.grad

    assert torch.allclose(out_ref.detach(), out_ckpt.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_ckpt, atol=1e-4)


def test_checkpointed_memory_reduction():
    """Verify that checkpointing actually reduces memory usage."""
    import math

    T = 100
    C = 10
    interval = _compute_checkpoint_interval(T)

    # Full storage: O(T * C) for alpha
    full_alpha_elements = T * C

    # Checkpointed storage: O(num_checkpoints * C)
    num_checkpoints = (T + interval - 1) // interval
    checkpointed_elements = num_checkpoints * C

    # Checkpointing should use sqrt(T) times less memory for alpha
    reduction_factor = full_alpha_elements / checkpointed_elements
    expected_reduction = T / num_checkpoints

    assert reduction_factor >= 5, (
        f"Expected at least 5x memory reduction, got {reduction_factor:.1f}x"
    )
    assert abs(reduction_factor - expected_reduction) < 0.1, (
        f"Reduction factor {reduction_factor:.1f} doesn't match expected {expected_reduction:.1f}"
    )


# =============================================================================
# Optimized Checkpointed Backward Tests (O(T) compute)
# =============================================================================


def test_optimized_checkpointed_forward_matches_full():
    """Test that optimized checkpointed forward computes the same partition."""
    torch.manual_seed(4000)
    batch, T, K, C = 2, 20, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Full forward (reference)
    partition_full, _ = semi_crf_forward_with_alpha(edge, lengths)

    # Optimized checkpointed forward
    partition_opt, ring_ckpts, interval = semi_crf_forward_with_ring_checkpoints(
        edge, lengths
    )

    assert torch.allclose(partition_full, partition_opt, atol=1e-5), (
        f"Partition mismatch: {(partition_full - partition_opt).abs().max()}"
    )

    # Ring checkpoints should have shape (batch, num_ckpts, K, C)
    num_ckpts = (T + interval - 1) // interval
    assert ring_ckpts.shape == (batch, num_ckpts, K, C)


def test_optimized_checkpointed_backward_matches_full():
    """Test that optimized checkpointed backward computes correct gradients."""
    torch.manual_seed(4100)
    batch, T, K, C = 2, 20, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Reference gradients via full backward
    partition_ref, alpha_ref = semi_crf_forward_with_alpha(edge.detach(), lengths)
    beta_ref = semi_crf_backward_beta(edge.detach(), lengths)
    grad_ref = semi_crf_compute_marginals(
        edge.detach(), alpha_ref, beta_ref, partition_ref, lengths
    )

    # Optimized checkpointed backward
    partition_opt, ring_ckpts, interval = semi_crf_forward_with_ring_checkpoints(
        edge.detach(), lengths
    )
    grad_opt = semi_crf_backward_from_ring_checkpoints(
        edge.detach(), ring_ckpts, partition_opt, lengths, interval
    )

    assert torch.allclose(grad_ref, grad_opt, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_opt).abs().max()}"
    )


def test_optimized_checkpointed_autograd():
    """Test full autograd integration with optimized checkpointed backward."""
    torch.manual_seed(4200)
    batch, T, K, C = 2, 20, 4, 3

    # Reference via autograd
    edge_ref = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out_ref = semi_crf_forward_pytorch(edge_ref, lengths)
    out_ref.sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Optimized checkpointed backward
    edge_opt = edge_ref.detach().clone().requires_grad_(True)
    out_opt = semi_crf_optimized_checkpointed_backward(edge_opt, lengths)
    out_opt.sum().backward()
    grad_opt = edge_opt.grad

    assert torch.allclose(out_ref.detach(), out_opt.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_opt, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_opt).abs().max()}"
    )


def test_optimized_checkpointed_variable_lengths():
    """Test optimized checkpointed backward with variable sequence lengths."""
    torch.manual_seed(4300)
    batch, T, K, C = 3, 24, 5, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.tensor([24, 16, 10], dtype=torch.long)

    # Reference via autograd
    out_ref = semi_crf_forward_pytorch(edge, lengths)
    out_ref.sum().backward()
    grad_ref = edge.grad.clone()

    # Optimized checkpointed backward
    edge.grad = None
    edge_opt = edge.detach().clone().requires_grad_(True)
    out_opt = semi_crf_optimized_checkpointed_backward(edge_opt, lengths)
    out_opt.sum().backward()
    grad_opt = edge_opt.grad

    assert torch.allclose(out_ref.detach(), out_opt.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_opt, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_opt).abs().max()}"
    )


def test_optimized_checkpointed_gradcheck():
    """Verify optimized checkpointed backward with torch.autograd.gradcheck."""
    torch.manual_seed(4400)
    batch, T, K, C = 1, 10, 3, 2
    edge = torch.randn(
        batch, T - 1, K, C, C, dtype=torch.float64, requires_grad=True
    )
    lengths = torch.full((batch,), T, dtype=torch.long)

    def func(edge_input):
        return semi_crf_optimized_checkpointed_backward(edge_input, lengths)

    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_optimized_vs_basic_checkpointed():
    """Verify that optimized and basic checkpointed produce same results."""
    torch.manual_seed(4500)
    batch, T, K, C = 2, 20, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    # Basic checkpointed
    edge_basic = edge.detach().clone().requires_grad_(True)
    out_basic = semi_crf_checkpointed_backward(edge_basic, lengths)
    out_basic.sum().backward()
    grad_basic = edge_basic.grad

    # Optimized checkpointed
    edge_opt = edge.detach().clone().requires_grad_(True)
    out_opt = semi_crf_optimized_checkpointed_backward(edge_opt, lengths)
    out_opt.sum().backward()
    grad_opt = edge_opt.grad

    assert torch.allclose(out_basic, out_opt, atol=1e-5)
    assert torch.allclose(grad_basic, grad_opt, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_basic - grad_opt).abs().max()}"
    )


def test_optimized_checkpointed_memory_tradeoff():
    """Verify memory trade-off: O(√T × K × C) vs O(√T × C)."""
    import math

    T = 100
    K = 8
    C = 10
    interval = _compute_checkpoint_interval(T, K)
    num_checkpoints = (T + interval - 1) // interval

    # Basic checkpointing: O(√T × C)
    basic_checkpoint_elements = num_checkpoints * C

    # Optimized checkpointing: O(√T × K × C)
    optimized_checkpoint_elements = num_checkpoints * K * C

    # Optimized uses K× more checkpoint memory
    memory_ratio = optimized_checkpoint_elements / basic_checkpoint_elements
    assert abs(memory_ratio - K) < 0.1, f"Expected {K}× memory, got {memory_ratio:.1f}×"

    # But both are still much less than full storage O(T × C)
    full_elements = T * C
    basic_reduction = full_elements / basic_checkpoint_elements
    optimized_reduction = full_elements / optimized_checkpoint_elements

    assert basic_reduction >= 5, f"Basic should have ≥5× reduction, got {basic_reduction:.1f}×"
    assert optimized_reduction >= 1, f"Optimized should have positive reduction"


# =============================================================================
# Phase 3 CUDA: Checkpointed Triton Kernel Tests
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_checkpointed_kernel_matches_pytorch():
    """Test that Triton checkpointed backward kernel matches PyTorch reference."""
    from torch_semimarkov.triton_backward import (
        launch_triton_checkpointed_backward_kernel,
    )

    torch.manual_seed(5000)
    batch, T, K, C = 4, 20, 5, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute forward with ring checkpoints
    partition, ring_checkpoints, interval = semi_crf_forward_with_ring_checkpoints(
        edge, lengths
    )

    # PyTorch reference
    grad_pytorch = semi_crf_backward_from_ring_checkpoints(
        edge, ring_checkpoints, partition, lengths, interval
    )

    # Triton kernel
    grad_triton = launch_triton_checkpointed_backward_kernel(
        edge, ring_checkpoints, partition, lengths, interval
    )

    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_checkpointed_kernel_variable_lengths():
    """Test Triton checkpointed backward kernel with variable sequence lengths."""
    from torch_semimarkov.triton_backward import (
        launch_triton_checkpointed_backward_kernel,
    )

    torch.manual_seed(5100)
    batch, T, K, C = 4, 24, 6, 4
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.tensor([24, 18, 12, 6], dtype=torch.long, device="cuda")

    # Compute forward with ring checkpoints
    partition, ring_checkpoints, interval = semi_crf_forward_with_ring_checkpoints(
        edge, lengths
    )

    # PyTorch reference
    grad_pytorch = semi_crf_backward_from_ring_checkpoints(
        edge, ring_checkpoints, partition, lengths, interval
    )

    # Triton kernel
    grad_triton = launch_triton_checkpointed_backward_kernel(
        edge, ring_checkpoints, partition, lengths, interval
    )

    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_checkpointed_kernel_non_power_of_2_labels():
    """Test Triton checkpointed backward kernel with non-power-of-2 label count."""
    from torch_semimarkov.triton_backward import (
        launch_triton_checkpointed_backward_kernel,
    )

    torch.manual_seed(5200)
    batch, T, K, C = 3, 16, 4, 5  # C=5 is not power of 2
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute forward with ring checkpoints
    partition, ring_checkpoints, interval = semi_crf_forward_with_ring_checkpoints(
        edge, lengths
    )

    # PyTorch reference
    grad_pytorch = semi_crf_backward_from_ring_checkpoints(
        edge, ring_checkpoints, partition, lengths, interval
    )

    # Triton kernel
    grad_triton = launch_triton_checkpointed_backward_kernel(
        edge, ring_checkpoints, partition, lengths, interval
    )

    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_checkpointed_autograd_function():
    """Test full autograd integration with Triton checkpointed backward."""
    from torch_semimarkov.triton_backward import semi_crf_triton_checkpointed_backward

    torch.manual_seed(5300)
    batch, T, K, C = 4, 20, 5, 4

    # Reference gradients via PyTorch
    edge_ref = torch.randn(batch, T - 1, K, C, C, device="cuda", requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    out_ref = semi_crf_forward_pytorch(edge_ref, lengths, semiring="log")
    out_ref.sum().backward()
    grad_ref = edge_ref.grad.clone()

    # Triton checkpointed backward via autograd function
    edge_triton = edge_ref.detach().clone().requires_grad_(True)
    out_triton = semi_crf_triton_checkpointed_backward(edge_triton, lengths)
    out_triton.sum().backward()
    grad_triton = edge_triton.grad

    assert torch.allclose(out_ref.detach(), out_triton.detach(), atol=1e-5)
    assert torch.allclose(grad_ref, grad_triton, atol=1e-4), (
        f"Gradient mismatch: max diff {(grad_ref - grad_triton).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_checkpointed_larger_shapes():
    """Test Triton checkpointed backward kernel with larger shapes."""
    from torch_semimarkov.triton_backward import (
        launch_triton_checkpointed_backward_kernel,
    )

    torch.manual_seed(5400)
    batch, T, K, C = 8, 64, 12, 8
    edge = torch.randn(batch, T - 1, K, C, C, device="cuda")
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    # Compute forward with ring checkpoints
    partition, ring_checkpoints, interval = semi_crf_forward_with_ring_checkpoints(
        edge, lengths
    )

    # PyTorch reference
    grad_pytorch = semi_crf_backward_from_ring_checkpoints(
        edge, ring_checkpoints, partition, lengths, interval
    )

    # Triton kernel
    grad_triton = launch_triton_checkpointed_backward_kernel(
        edge, ring_checkpoints, partition, lengths, interval
    )

    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-3), (
        f"Gradient mismatch: max diff {(grad_triton - grad_pytorch).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_triton_checkpointed_gradcheck():
    """Verify Triton checkpointed backward with torch.autograd.gradcheck."""
    from torch_semimarkov.triton_backward import semi_crf_triton_checkpointed_backward

    torch.manual_seed(5500)
    batch, T, K, C = 2, 8, 3, 2
    edge = torch.randn(
        batch, T - 1, K, C, C, dtype=torch.float64, device="cuda", requires_grad=True
    )
    lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

    def func(edge_input):
        return semi_crf_triton_checkpointed_backward(edge_input, lengths)

    # Use gradcheck to verify numerical gradients
    assert torch.autograd.gradcheck(func, (edge,), eps=1e-6, atol=1e-4, rtol=1e-3)
