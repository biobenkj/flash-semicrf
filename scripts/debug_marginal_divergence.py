#!/usr/bin/env python3
"""Minimal debug script to identify marginal computation divergence.

Adds targeted debug prints to compare Triton vs PyTorch intermediate values.
"""

import sys

import torch

from torch_semimarkov.streaming import (
    launch_streaming_triton_marginals,
    semi_crf_streaming_marginals_pytorch,
)
from torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel

# Configuration matching the failing test
batch, T, C, K = 1, 48, 8, 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("MARGINAL DIVERGENCE DEBUG")
print("=" * 80)
print(f"Config: batch={batch}, T={T}, C={C}, K={K}")
print(f"Device: {device}")
print()

# Generate test data (same seed as test)
torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)

scores = torch.randn(batch, T, C, device=device)
scores = scores - scores.mean(dim=1, keepdim=True)
cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

transition = torch.randn(C, C, device=device)
duration_bias = torch.randn(K, C, device=device)
lengths = torch.full((batch,), T, device=device, dtype=torch.long)

print("Running PyTorch reference...")
pytorch_marginals, log_Z_pytorch = semi_crf_streaming_marginals_pytorch(
    cum_scores.cpu(), transition.cpu(), duration_bias.cpu(), lengths.cpu(), K
)

print(f"PyTorch log_Z: {log_Z_pytorch[0].item():.6f}")
print(f"PyTorch marginals sum: {pytorch_marginals[0].sum().item():.6f}")
print(
    f"PyTorch marginals range: [{pytorch_marginals[0].min().item():.6f}, {pytorch_marginals[0].max().item():.6f}]"
)
print()

if device.type != "cuda":
    print("CUDA not available - skipping Triton test")
    sys.exit(0)

print("Running Triton forward...")
log_Z_triton, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
    cum_scores, transition, duration_bias, lengths, K
)

print(f"Triton log_Z: {log_Z_triton[0].item():.6f}")
print(f"Checkpoint interval: {interval}")
print(f"log_norm_checkpoints: {log_norm_ckpts[0].tolist()}")
print()

print("Running Triton backward (marginals)...")
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

print(f"Triton marginals sum: {triton_marginals[0].sum().item():.6f}")
print(
    f"Triton marginals range: [{triton_marginals[0].min().item():.6f}, {triton_marginals[0].max().item():.6f}]"
)
print()

# Compare
diff = (triton_marginals.cpu() - pytorch_marginals).abs()[0]
ratio = (triton_marginals.cpu() / pytorch_marginals)[0]

print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Max absolute difference: {diff.max().item():.6e}")
print(f"Mean absolute difference: {diff.mean().item():.6e}")
print(f"Max ratio (Triton/PyTorch): {ratio.max().item():.6f}")
print(f"Min ratio (Triton/PyTorch): {ratio.min().item():.6f}")
print()

# Find positions with largest errors
threshold = 1e-4
error_indices = torch.where(diff > threshold)[0].tolist()

if error_indices:
    print(f"Positions with error > {threshold}:")
    print(f"  {'t':>3} | {'PyTorch':>10} | {'Triton':>10} | {'Diff':>10} | {'Ratio':>8} | Segment")
    print("-" * 75)

    for t in error_indices[:20]:  # Show first 20
        pt_val = pytorch_marginals[0, t].item()
        tr_val = triton_marginals[0, t].item()
        diff_val = diff[t].item()
        ratio_val = ratio[t].item()
        seg = 0 if t < interval else 1
        print(
            f"  {t:3d} | {pt_val:10.6f} | {tr_val:10.6f} | {diff_val:10.6e} | {ratio_val:8.4f} | Seg{seg}"
        )

    if len(error_indices) > 20:
        print(f"  ... and {len(error_indices) - 20} more")

    print()
    print(f"Total errors: {len(error_indices)} / {T} positions")

    # Analyze by segment
    seg0_errors = [t for t in error_indices if t < interval]
    seg1_errors = [t for t in error_indices if t >= interval]

    print("\nErrors by segment:")
    print(f"  Segment 0 (t=0-{interval-1}): {len(seg0_errors)} errors")
    print(f"  Segment 1 (t={interval}-{T-1}): {len(seg1_errors)} errors")
else:
    print("[PASS] All positions match within threshold!")
    print()
    print("SUCCESS: Triton matches PyTorch reference")

print()
print("NEXT STEPS:")
print("1. If errors exist, add tl.device_print to triton_backward.py for specific (t, k)")
print("2. Compare alpha, beta, edge, scale values between Triton and PyTorch")
print("3. Identify first divergence point")
