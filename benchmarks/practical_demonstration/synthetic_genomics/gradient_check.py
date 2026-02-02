#!/usr/bin/env python3
"""
Gradient Agreement Test for Streaming Semi-CRF

Directly compares Triton kernel gradients vs PyTorch reference to verify:
1. Forward pass partition agreement
2. Backward pass gradient agreement for cum_scores, transition, duration_bias

This is a low-level test that bypasses the high-level SemiMarkovCRFHead API
to isolate the kernel implementation.

Usage:
    # Quick smoke test (T=1000, forced small checkpoint interval)
    python gradient_check.py --seq-length 1000 --checkpoint-interval 100

    # T=10k with auto checkpoint interval
    python gradient_check.py --seq-length 10000

    # Larger scale (may take longer)
    python gradient_check.py --seq-length 50000 --batch-size 1

Notes:
    - PyTorch reference lacks per-checkpoint log normalization
    - At T > ~10k, partitions will diverge (expected)
    - Gradients should agree at T < 10k where overflow doesn't occur
"""

from __future__ import annotations

import argparse
import sys

import torch

# Import PyTorch reference implementations
from torch_semimarkov.streaming.pytorch_reference import (
    semi_crf_streaming_backward_pytorch,
    semi_crf_streaming_forward_pytorch,
)
from torch_semimarkov.streaming.triton_backward import launch_streaming_triton_backward

# Import Triton kernel launchers
from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel


def create_synthetic_inputs(
    batch: int,
    T: int,
    C: int,
    K: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic inputs for gradient checking.

    Args:
        batch: Batch size
        T: Sequence length
        C: Number of classes
        K: Maximum duration
        device: Device to use
        dtype: Data type (float32 required - Triton kernel uses float32 internally)

    Returns:
        cum_scores: (batch, T+1, C) cumulative emission scores
        transition: (C, C) transition matrix
        duration_bias: (K, C) duration log-potentials
        lengths: (batch,) sequence lengths
    """
    # Emissions via cumsum (ensures cum_scores[:, 0, :] = 0)
    emissions = torch.randn(batch, T, C, device=device, dtype=dtype) * 0.1
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
    cum_scores[:, 1:, :] = torch.cumsum(emissions, dim=1)

    # Transition and duration bias
    transition = torch.randn(C, C, device=device, dtype=dtype) * 0.1
    duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1

    lengths = torch.full((batch,), T, device=device, dtype=torch.long)

    return cum_scores, transition, duration_bias, lengths


def check_forward_agreement(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    checkpoint_interval: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compare forward pass results between Triton and PyTorch reference.

    Returns:
        partition_triton: Triton partition function
        ring_ckpts_triton: Triton ring checkpoints (normalized)
        ckpt_interval: Checkpoint interval used
        log_norm_ckpts: Log normalization checkpoints (Triton only)
        partition_ref: PyTorch reference partition function
        ring_ckpts_ref: PyTorch reference ring checkpoints (un-normalized)
    """
    # Triton forward
    partition_triton, ring_ckpts_triton, ckpt_interval, log_norm_ckpts = (
        launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            checkpoint_interval=checkpoint_interval,
        )
    )

    # PyTorch reference forward (doesn't have log_norm)
    partition_ref, ring_ckpts_ref, _ = semi_crf_streaming_forward_pytorch(
        cum_scores,
        transition,
        duration_bias,
        lengths,
        K,
        checkpoint_interval=checkpoint_interval,
    )

    # Compare partitions
    diff = (partition_triton - partition_ref).abs()
    rel_diff = diff / (partition_ref.abs() + 1e-8)

    print("Forward partition comparison:")
    print(f"  Triton:  {partition_triton.tolist()}")
    print(f"  PyTorch: {partition_ref.tolist()}")
    print(f"  Abs diff: {diff.max().item():.2e}")
    print(f"  Rel diff: {rel_diff.max().item():.2e}")

    # Check if log_norm is non-zero (indicates normalization happened)
    if log_norm_ckpts is not None:
        max_log_norm = log_norm_ckpts.abs().max().item()
        print(f"  Max log_norm_checkpoint: {max_log_norm:.2e}")
        if max_log_norm > 1e-6:
            print("  (Note: Normalization active, partitions may differ slightly)")

    # Verify checkpoint consistency: Triton normalized + log_norm should ≈ PyTorch raw
    # Only compare the actual C classes (Triton may have C_PAD)
    C_actual = cum_scores.shape[2]
    triton_ckpt_slice = ring_ckpts_triton[:, :, :, :C_actual]

    # Reconstruct "un-normalized" Triton checkpoints by adding back log_norm
    # log_norm_ckpts shape: (batch, num_ckpts)
    # ring_ckpts shape: (batch, num_ckpts, K, C)
    triton_unnorm = triton_ckpt_slice + log_norm_ckpts[:, :, None, None]

    ckpt_diff = (triton_unnorm - ring_ckpts_ref).abs()
    print("  Checkpoint reconstruction check (Triton+log_norm vs PyTorch):")
    print(f"    Max abs diff: {ckpt_diff.max().item():.2e}")
    print(f"    Mean abs diff: {ckpt_diff.mean().item():.2e}")

    return (
        partition_triton,
        ring_ckpts_triton,
        ckpt_interval,
        log_norm_ckpts,
        partition_ref,
        ring_ckpts_ref,
    )


def check_backward_agreement(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    partition_triton: torch.Tensor,
    partition_ref: torch.Tensor,
    ring_ckpts_triton: torch.Tensor,
    ring_ckpts_ref: torch.Tensor,
    log_norm_ckpts: torch.Tensor,
    checkpoint_interval: int,
    tolerance: float = 1e-2,  # Relaxed for normalized vs un-normalized comparison
) -> bool:
    """
    Compare backward pass gradients between Triton and PyTorch reference.

    IMPORTANT: Each implementation must use its own checkpoints:
    - Triton checkpoints are NORMALIZED (alpha shifted by log_norm)
    - PyTorch checkpoints are UN-NORMALIZED (raw alpha values)

    Returns:
        True if gradients agree within tolerance, False otherwise.
    """
    batch, T_plus_1, C = cum_scores.shape
    device = cum_scores.device
    dtype = cum_scores.dtype

    grad_output = torch.ones(batch, device=device, dtype=dtype)

    # Triton backward (uses normalized checkpoints + log_norm_ckpts)
    grad_cum_triton, grad_trans_triton, grad_dur_triton, _, _, _ = launch_streaming_triton_backward(
        cum_scores,
        transition,
        duration_bias,
        lengths,
        partition_triton,  # log_Z
        ring_ckpts_triton,
        log_norm_ckpts,
        checkpoint_interval,
        grad_output,
    )

    # PyTorch reference backward (uses un-normalized checkpoints)
    # CRITICAL: Use ring_ckpts_ref from PyTorch forward, NOT Triton's normalized checkpoints
    grad_cum_ref, grad_trans_ref_batch, grad_dur_ref_batch, _, _ = (
        semi_crf_streaming_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            partition_ref,  # Use ref partition for ref backward
            ring_ckpts_ref,  # Use ref checkpoints (un-normalized)
            checkpoint_interval,
        )
    )

    # PyTorch reference returns PER-BATCH gradients: (batch, C, C) and (batch, K, C)
    # Triton returns REDUCED gradients: (C, C) and (K, C)
    # Reduce PyTorch gradients to match (grad_output is all 1s, so just sum over batch)
    grad_trans_ref = grad_trans_ref_batch.sum(dim=0)
    grad_dur_ref = grad_dur_ref_batch.sum(dim=0)

    # Compare gradients
    def compare_grads(
        name: str, triton: torch.Tensor, ref: torch.Tensor
    ) -> tuple[float, float, float]:
        """Returns (max_abs_diff, max_rel_diff, mean_abs_diff)."""
        # Handle shape differences (Triton may have C_PAD)
        if triton.shape != ref.shape:
            # Slice Triton to match ref shape
            slices = tuple(slice(0, s) for s in ref.shape)
            triton = triton[slices]

        diff = (triton - ref).abs()
        rel_diff = diff / (ref.abs() + 1e-8)

        max_abs = diff.max().item()
        max_rel = rel_diff.max().item()
        mean_abs = diff.mean().item()

        print(f"  {name}:")
        print(f"    Max abs diff: {max_abs:.2e}")
        print(f"    Max rel diff: {max_rel:.2e}")
        print(f"    Mean abs diff: {mean_abs:.2e}")

        # Check for NaN/Inf
        if torch.isnan(triton).any() or torch.isinf(triton).any():
            print("    WARNING: Triton gradient contains NaN/Inf!")
        if torch.isnan(ref).any() or torch.isinf(ref).any():
            print("    WARNING: Reference gradient contains NaN/Inf!")

        return max_abs, max_rel, mean_abs

    print("\nBackward gradient comparison:")
    _, _, mean_abs_cum = compare_grads("grad_cum_scores", grad_cum_triton, grad_cum_ref)
    _, rel_trans, _ = compare_grads("grad_transition", grad_trans_triton, grad_trans_ref)
    _, rel_dur, _ = compare_grads("grad_duration_bias", grad_dur_triton, grad_dur_ref)

    # Updated acceptance criteria for normalized vs un-normalized comparison:
    # - grad_cum_scores: Mean absolute error (large relative errors at near-zero positions expected)
    # - grad_transition/duration_bias: Max relative error (actual parameter gradients)
    #
    # These thresholds account for:
    # 1. PyTorch reference lacks normalization (alpha values grow with T)
    # 2. Parallel reduction (Triton) vs sequential (Python) causes FP rounding differences
    # 3. Error accumulation scales roughly linearly with T
    cum_tol = 1e-3  # Mean absolute error tolerance for cum_scores
    param_tol = tolerance  # Relative error tolerance for parameter gradients

    cum_pass = mean_abs_cum < cum_tol
    trans_pass = rel_trans < param_tol
    dur_pass = rel_dur < param_tol
    passed = cum_pass and trans_pass and dur_pass

    print("\nAcceptance criteria (normalized vs un-normalized):")
    print(
        f"  grad_cum_scores mean abs: {mean_abs_cum:.2e} < {cum_tol:.0e} {'✓' if cum_pass else '✗'}"
    )
    print(
        f"  grad_transition max rel:  {rel_trans:.2e} < {param_tol:.0e} {'✓' if trans_pass else '✗'}"
    )
    print(
        f"  grad_duration_bias max rel: {rel_dur:.2e} < {param_tol:.0e} {'✓' if dur_pass else '✗'}"
    )
    print(
        f"\n{'PASS' if passed else 'FAIL'}: All criteria met"
        if passed
        else f"\n{'PASS' if passed else 'FAIL'}: Criteria not met"
    )

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Gradient agreement test for streaming Semi-CRF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seq-length", "-T", type=int, default=1000, help="Sequence length (default: 1000)"
    )
    parser.add_argument("--batch-size", "-B", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument(
        "--num-classes", "-C", type=int, default=3, help="Number of classes (default: 3)"
    )
    parser.add_argument(
        "--max-duration", "-K", type=int, default=50, help="Maximum segment duration (default: 50)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Force checkpoint interval (default: auto-computed)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Relative error tolerance for parameter gradients (default: 1e-2)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        print("WARNING: Triton kernels require CUDA, test will likely fail")
        args.device = "cpu"

    # Set seed
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Gradient Agreement Test for Streaming Semi-CRF")
    print("=" * 60)
    print("Parameters:")
    print(f"  T (seq_length):      {args.seq_length}")
    print(f"  B (batch_size):      {args.batch_size}")
    print(f"  C (num_classes):     {args.num_classes}")
    print(f"  K (max_duration):    {args.max_duration}")
    print(f"  checkpoint_interval: {args.checkpoint_interval or 'auto'}")
    print(f"  tolerance:           {args.tolerance:.0e}")
    print(f"  device:              {args.device}")
    print(f"  seed:                {args.seed}")
    print("=" * 60)
    print()

    # Create inputs
    cum_scores, transition, duration_bias, lengths = create_synthetic_inputs(
        args.batch_size,
        args.seq_length,
        args.num_classes,
        args.max_duration,
        device=args.device,
    )

    # Forward pass comparison
    (
        partition_triton,
        ring_ckpts_triton,
        ckpt_interval,
        log_norm_ckpts,
        partition_ref,
        ring_ckpts_ref,
    ) = check_forward_agreement(
        cum_scores,
        transition,
        duration_bias,
        lengths,
        args.max_duration,
        checkpoint_interval=args.checkpoint_interval,
    )

    print(f"\n(Using checkpoint_interval={ckpt_interval})")

    # Backward pass comparison
    # IMPORTANT: Each implementation uses its OWN checkpoints
    # - Triton: ring_ckpts_triton (normalized) + log_norm_ckpts
    # - PyTorch: ring_ckpts_ref (un-normalized)
    passed = check_backward_agreement(
        cum_scores,
        transition,
        duration_bias,
        lengths,
        args.max_duration,
        partition_triton,
        partition_ref,
        ring_ckpts_triton,
        ring_ckpts_ref,
        log_norm_ckpts,
        ckpt_interval,
        tolerance=args.tolerance,
    )

    print()
    print("=" * 60)
    if passed:
        print("TEST PASSED: Triton and PyTorch reference gradients agree")
    else:
        print("TEST FAILED: Gradient mismatch detected")
        print()
        print("Possible causes:")
        print("  1. At large T, PyTorch reference may overflow (lacks normalization)")
        print("  2. Implementation bug in Triton kernel")
        print("  3. Floating point precision issues")
        print()
        print("Try running with smaller --seq-length or --checkpoint-interval")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
