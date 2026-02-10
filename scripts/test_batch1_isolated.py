"""Isolated test for batch=1 marginals to identify the issue."""

import torch

from flash_semicrf.streaming import (
    HAS_TRITON,
    launch_streaming_triton_marginals,
    semi_crf_streaming_marginals_pytorch,
)
from flash_semicrf.streaming.triton_forward import launch_streaming_triton_kernel

if not HAS_TRITON:
    print("Triton not available")
    exit(1)

if not torch.cuda.is_available():
    print("CUDA not available")
    exit(1)

# Test configuration from failing test
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
batch, T, C, K = 1, 100, 8, 16

print(f"Testing batch={batch}, T={T}, C={C}, K={K}")

# Setup
device = torch.device("cuda")
scores = torch.randn(batch, T, C, device=device)
scores = scores - scores.mean(dim=1, keepdim=True)
cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

transition = torch.randn(C, C, device=device)
duration_bias = torch.randn(K, C, device=device)
lengths = torch.full((batch,), T, device=device, dtype=torch.long)

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

# Compare
print("\nPyTorch marginals stats:")
print(f"  Shape: {pytorch_marginals.shape}")
print(f"  Min: {pytorch_marginals.min():.6f}, Max: {pytorch_marginals.max():.6f}")
print(f"  Mean: {pytorch_marginals.mean():.6f}")
print(f"  First 10: {pytorch_marginals[0, :10].tolist()}")

print("\nTriton marginals stats:")
print(f"  Shape: {triton_marginals.shape}")
print(f"  Min: {triton_marginals.min():.6f}, Max: {triton_marginals.max():.6f}")
print(f"  Mean: {triton_marginals.mean():.6f}")
print(f"  First 10: {triton_marginals.cpu()[0, :10].tolist()}")

# Compute differences
diff = (triton_marginals.cpu() - pytorch_marginals).abs()
rel_diff = diff / (pytorch_marginals.abs() + 1e-8)

print("\nDifference stats:")
print(f"  Max abs diff: {diff.max():.6e}")
print(f"  Max rel diff: {rel_diff.max():.6f}")
print(f"  Mean abs diff: {diff.mean():.6e}")
print(f"  Mean rel diff: {rel_diff.mean():.6f}")

# Find positions with largest errors
top_errors = torch.topk(diff.flatten(), k=10)
print("\nTop 10 error positions:")
for i, (error, idx) in enumerate(zip(top_errors.values, top_errors.indices, strict=True)):
    batch_idx = idx // T
    t_idx = idx % T
    pytorch_val = pytorch_marginals[batch_idx, t_idx]
    triton_val = triton_marginals.cpu()[batch_idx, t_idx]
    print(
        f"  {i+1}. t={t_idx}: pytorch={pytorch_val:.6f}, triton={triton_val:.6f}, diff={error:.6e}"
    )

# Check tolerance
try:
    torch.testing.assert_close(
        triton_marginals.cpu(),
        pytorch_marginals,
        rtol=0.01,
        atol=1e-5,
        msg="Batch=1 marginals don't match",
    )
    print("\n[PASS] TEST PASSED")
except AssertionError as e:
    print(f"\n[FAIL] TEST FAILED: {e}")
