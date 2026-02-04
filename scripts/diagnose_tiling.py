#!/usr/bin/env python3
"""Diagnostic script to isolate the adaptive tiling bug.

Key hypothesis: The issue is related to how the tile loop processes c_dst
in chunks and merges results. Specifically, when TILE_C < C_PAD, multiple
tiles process the marginal computation, and there may be an issue with:

1. How local_ref is computed per-tile vs globally
2. How the online logsumexp for beta accumulates across tiles
3. How marginal_sum_all_k accumulates across tiles
"""


import torch

try:
    from src.torch_semimarkov.streaming import (
        HAS_TRITON,
        launch_streaming_triton_marginals,
        semi_crf_streaming_marginals_pytorch,
    )
    from src.torch_semimarkov.streaming.triton_forward import launch_streaming_triton_kernel
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in standalone mode - using inline definitions")
    HAS_TRITON = False


def _next_power_of_2_local(n: int) -> int:
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    p = 1
    while p < n:
        p *= 2
    return p


def _compute_tile_c_local(C: int) -> int:
    C_PAD = _next_power_of_2_local(C)
    if C_PAD <= 8:
        return 4
    elif C_PAD <= 16:
        return 8
    else:
        return 16


def diagnose_tiling():
    """Diagnose the tiling behavior for different C values."""
    print("=" * 70)
    print("DIAGNOSTIC: Adaptive TILE_C computation")
    print("=" * 70)

    test_cases = [4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 32]

    print(f"\n{'C':>4} | {'C_PAD':>6} | {'TILE_C':>7} | {'#Tiles':>7} | Notes")
    print("-" * 50)

    for C in test_cases:
        C_PAD = _next_power_of_2_local(C)
        TILE_C = _compute_tile_c_local(C)
        num_tiles = (C_PAD + TILE_C - 1) // TILE_C

        notes = []
        if num_tiles == 1:
            notes.append("single tile (no tiling)")
        else:
            notes.append(f"{num_tiles} tiles")

        if C_PAD % TILE_C != 0:
            notes.append("uneven split")

        if TILE_C > C:
            notes.append("tile > C")

        print(f"{C:>4} | {C_PAD:>6} | {TILE_C:>7} | {num_tiles:>7} | {', '.join(notes)}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Problem configuration")
    print("=" * 70)

    # The problem config
    C = 8
    K = 16
    T = 48
    checkpoint_interval = 27

    C_PAD = _next_power_of_2_local(C)
    TILE_C = _compute_tile_c_local(C)
    num_tiles = C_PAD // TILE_C

    print("\nConfiguration:")
    print(f"  T = {T}")
    print(f"  K = {K}")
    print(f"  C = {C}")
    print(f"  C_PAD = {C_PAD}")
    print(f"  TILE_C = {TILE_C}")
    print(f"  num_tiles = {num_tiles}")
    print(f"  checkpoint_interval = {checkpoint_interval}")
    print(f"  num_checkpoints = {(T + checkpoint_interval - 1) // checkpoint_interval}")

    print("\nSegment analysis:")
    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval
    for ckpt_idx in range(num_checkpoints - 1, -1, -1):
        seg_start = ckpt_idx * checkpoint_interval
        seg_end = min((ckpt_idx + 1) * checkpoint_interval, T)
        print(
            f"  Checkpoint {ckpt_idx}: positions {seg_start}-{seg_end-1} "
            f"(processed {'first' if ckpt_idx == num_checkpoints - 1 else 'second'} in backward)"
        )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Tile iteration analysis for marginal computation")
    print("=" * 70)

    print(f"\nFor c_dst_tile_start in static_range(0, {C_PAD}, {TILE_C}):")
    for c_dst_tile_start in range(0, C_PAD, TILE_C):
        tile_indices = list(range(c_dst_tile_start, c_dst_tile_start + TILE_C))
        valid_indices = [i for i in tile_indices if i < C]
        invalid_indices = [i for i in tile_indices if i >= C]

        print(f"  Tile at {c_dst_tile_start}: c_dst = {tile_indices}")
        print(f"    Valid (c_dst < C={C}): {valid_indices}")
        if invalid_indices:
            print(f"    Masked out: {invalid_indices}")


def test_with_fixed_tile_c():
    """Test if forcing TILE_C=C_PAD (single tile) fixes the issue."""
    if not HAS_TRITON or not torch.cuda.is_available():
        print("\nSkipping Triton tests (no CUDA/Triton)")
        return

    print("\n" + "=" * 70)
    print("TEST: Compare adaptive vs fixed TILE_C")
    print("=" * 70)

    # Configuration that triggers the bug
    batch, T, C, K = 1, 48, 8, 16
    device = torch.device("cuda")

    # Generate test data
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    scores = torch.randn(batch, T, C, device=device)
    scores = scores - scores.mean(dim=1, keepdim=True)
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
    cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

    transition = torch.randn(C, C, device=device)
    duration_bias = torch.randn(K, C, device=device)
    lengths = torch.full((batch,), T, device=device, dtype=torch.long)

    # PyTorch reference
    pytorch_marginals, log_Z_pytorch = semi_crf_streaming_marginals_pytorch(
        cum_scores.cpu(), transition.cpu(), duration_bias.cpu(), lengths.cpu(), K
    )

    # Triton with adaptive TILE_C
    log_Z_triton, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
        cum_scores, transition, duration_bias, lengths, K
    )
    triton_marginals_adaptive = launch_streaming_triton_marginals(
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
    diff_adaptive = (triton_marginals_adaptive.cpu() - pytorch_marginals).abs()[0]

    print(f"\nPyTorch log_Z: {log_Z_pytorch[0].item():.6f}")
    print(f"Triton log_Z:  {log_Z_triton[0].item():.6f}")

    print("\nAdaptive TILE_C results:")
    print(f"  TILE_C = {_compute_tile_c_local(C)} (computed)")
    print(f"  Max diff: {diff_adaptive.max().item():.2e}")
    print(f"  Mean diff: {diff_adaptive.mean().item():.2e}")

    # Find where errors are
    threshold = 1e-4
    error_positions = torch.where(diff_adaptive > threshold)[0].tolist()
    print(
        f"\n  Positions with error > {threshold}: {error_positions if error_positions else 'None'}"
    )

    if error_positions:
        checkpoint_interval = interval
        print(f"\n  Checkpoint interval: {checkpoint_interval}")
        print(f"  Segment 0: positions 0-{checkpoint_interval-1}")
        print(f"  Segment 1: positions {checkpoint_interval}-{T-1}")

        seg0_errors = [p for p in error_positions if p < checkpoint_interval]
        seg1_errors = [p for p in error_positions if p >= checkpoint_interval]
        print(f"\n  Errors in segment 0: {len(seg0_errors)} positions")
        print(f"  Errors in segment 1: {len(seg1_errors)} positions")


if __name__ == "__main__":
    diagnose_tiling()
    test_with_fixed_tile_c()
