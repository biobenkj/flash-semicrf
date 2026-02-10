#!/usr/bin/env python3
"""Trace intermediate values in the marginal computation.

This script computes the marginal at a specific position step-by-step,
matching the Triton kernel's tiled computation, to identify where divergence occurs.
"""

import numpy as np
import torch

# Configuration matching the test
batch, T, C, K = 1, 48, 8, 16
C_PAD = 8  # _next_power_of_2(8) = 8
TILE_C = 4  # _compute_tile_c(8) = 4 since C_PAD <= 8
checkpoint_interval = 27

# Generate same test data as the test script
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scores = torch.randn(batch, T, C, device=device)
scores = scores - scores.mean(dim=1, keepdim=True)
cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

transition = torch.randn(C, C, device=device)
duration_bias = torch.randn(K, C, device=device)
lengths = torch.full((batch,), T, device=device, dtype=torch.long)

NEG_INF = -1e9


def compute_edge_block(cum_scores, transition, duration_bias, start_pos, k, C):
    """Compute edge scores for segment starting at start_pos with duration k."""
    end_pos = start_pos + k
    content_score = cum_scores[0, end_pos, :C] - cum_scores[0, start_pos, :C]
    dur_idx = k - 1
    dur_bias = duration_bias[dur_idx, :C]
    segment_score = content_score + dur_bias  # (C,)
    edge = segment_score[:, None] + transition[:C, :C]  # (C_dst, C_src)
    return edge


def simulate_forward_pass():
    """Run the forward pass to get alpha, log_Z, and log_norm_checkpoints."""
    # Initialize
    alpha_ring = torch.full((K, C), NEG_INF, device=device, dtype=torch.float64)
    alpha_ring[0, :] = 0.0  # Initial state

    accum_log_norm = 0.0
    log_norm_checkpoints = torch.zeros(2, device=device, dtype=torch.float64)
    ring_checkpoints = torch.full((2, K, C), NEG_INF, device=device, dtype=torch.float64)

    # Checkpoint 0 initialization
    ring_checkpoints[0, 0, :] = 0.0
    log_norm_checkpoints[0] = 0.0

    all_alpha = torch.full((T + 1, C), NEG_INF, device=device, dtype=torch.float64)
    all_alpha[0, :] = 0.0

    for t in range(1, T + 1):
        alpha_t = torch.full((C,), NEG_INF, device=device, dtype=torch.float64)

        k_eff = min(K, t)
        for k in range(1, k_eff + 1):
            start_pos = t - k
            ring_idx = start_pos % K
            alpha_prev = alpha_ring[ring_idx, :]

            edge = compute_edge_block(
                cum_scores, transition, duration_bias, start_pos, k, C
            ).double()

            # score[c_dst, c_src] = alpha_prev[c_src] + edge[c_dst, c_src]
            scores_k = alpha_prev[None, :] + edge  # (C, C)

            # logsumexp over c_src
            scores_over_src = torch.logsumexp(scores_k, dim=-1)  # (C,)

            # logsumexp accumulation over k
            alpha_t = torch.logsumexp(torch.stack([alpha_t, scores_over_src]), dim=0)

        # Normalization at checkpoint boundary
        is_checkpoint = t % checkpoint_interval == 0
        if is_checkpoint and t > 0:
            shift = alpha_t.max().item()
            alpha_t = alpha_t - shift
            accum_log_norm = accum_log_norm + shift

            ckpt_idx = t // checkpoint_interval
            if ckpt_idx < 2:
                ring_checkpoints[ckpt_idx, :, :] = alpha_ring.clone()
                log_norm_checkpoints[ckpt_idx] = accum_log_norm

        # Store in ring
        alpha_ring[t % K, :] = alpha_t
        all_alpha[t, :] = alpha_t + accum_log_norm  # Unnormalized for storage

    # Final log_Z
    final_alpha = alpha_ring[T % K, :]
    log_Z = torch.logsumexp(final_alpha, dim=0) + accum_log_norm

    return log_Z, all_alpha, ring_checkpoints, log_norm_checkpoints


def simulate_backward_segment0_marginal_at_t(t, all_alpha, beta, log_Z, log_norm_at_ckpt):
    """Compute the marginal at position t in segment 0 using the tiled approach."""
    marginal_sum = 0.0

    # Alpha at position t (unnormalized for segment 0)
    alpha_t = all_alpha[t, :].clone()

    print(f"\n=== Position t={t} ===")
    print(f"  alpha_t (unnormalized): min={alpha_t.min():.4f}, max={alpha_t.max():.4f}")
    print(f"  log_norm_at_ckpt: {log_norm_at_ckpt:.6f}")
    print(f"  log_Z: {log_Z:.6f}")

    max_k = min(K, T - t)

    for k in range(1, max_k + 1):
        end_pos = t + k
        if end_pos > T:
            continue

        beta_next = beta[end_pos, :].clone()
        edge = compute_edge_block(cum_scores, transition, duration_bias, t, k, C).double()

        # Simulate tiled computation with TILE_C=4
        tile_marginals = []
        tile_local_refs = []

        for c_dst_tile_start in range(0, C_PAD, TILE_C):
            c_dst_end = min(c_dst_tile_start + TILE_C, C)
            c_dst_slice = slice(c_dst_tile_start, c_dst_end)

            # Load tile values
            beta_tile = beta_next[c_dst_slice]  # (TILE_C,)
            edge_tile = edge[c_dst_slice, :]  # (TILE_C, C)

            # Clamp for safety
            alpha_t_clamped = torch.clamp(alpha_t, -1e6, 1e6)
            beta_tile_clamped = torch.clamp(beta_tile, -1e6, 1e6)
            edge_tile_clamped = torch.clamp(edge_tile, -1e6, 1e6)

            # log_joint = alpha + edge + beta
            # (1, C) + (TILE_C, C) + (TILE_C, 1)
            log_joint_tile = (
                alpha_t_clamped[None, :] + edge_tile_clamped + beta_tile_clamped[:, None]
            )

            # local_ref = max over tile
            local_ref = log_joint_tile.max().item()
            tile_local_refs.append(local_ref)

            # marginal_unnorm = exp(log_joint - local_ref)
            log_marginal_rel = log_joint_tile - local_ref
            marginal_unnorm = torch.exp(log_marginal_rel)

            # log_scale = local_ref + log_norm_at_ckpt - log_Z
            log_scale = local_ref + log_norm_at_ckpt - log_Z

            # Clamp scale to [exp(-700), 1]
            log_scale_clamped = max(min(log_scale, 0.0), -700.0)
            scale = np.exp(log_scale_clamped)

            # Final marginal
            marginal_tile = marginal_unnorm * scale
            tile_marginals.append(marginal_tile.sum().item())

            if k == 1:  # Only print for k=1 to reduce output
                print(f"    Tile [{c_dst_tile_start}:{c_dst_end}], k={k}:")
                print(f"      local_ref: {local_ref:.4f}")
                print(f"      log_scale: {log_scale:.6f}")
                print(f"      scale: {scale:.6f}")
                print(f"      tile_marginal_sum: {marginal_tile.sum().item():.6f}")

        # Sum across tiles for this k
        k_marginal = sum(tile_marginals)
        marginal_sum += k_marginal

    print(f"  Total marginal: {marginal_sum:.6f}")
    return marginal_sum


def simulate_backward_full():
    """Compute beta values and marginals for all positions."""
    # Initialize beta at final position
    beta = torch.full((T + 1, C), NEG_INF, device=device, dtype=torch.float64)
    beta[T, :] = 0.0

    # Backward pass to compute beta
    for t in range(T - 1, -1, -1):
        new_beta = torch.full((C,), NEG_INF, device=device, dtype=torch.float64)

        max_k = min(K, T - t)
        for k in range(1, max_k + 1):
            end_pos = t + k
            if end_pos > T:
                continue

            beta_next = beta[end_pos, :]
            edge = compute_edge_block(cum_scores, transition, duration_bias, t, k, C).double()

            # beta_k[c_src] = logsumexp over c_dst of (edge[c_dst, c_src] + beta_next[c_dst])
            scores_for_beta = edge + beta_next[:, None]  # (C_dst, C_src)
            beta_k = torch.logsumexp(scores_for_beta, dim=0)  # (C_src,)

            # Accumulate
            new_beta = torch.logsumexp(torch.stack([new_beta, beta_k]), dim=0)

        beta[t, :] = new_beta

    return beta


def main():
    print("=" * 70)
    print("MARGINAL COMPUTATION TRACE")
    print("=" * 70)
    print(f"Configuration: T={T}, C={C}, K={K}, C_PAD={C_PAD}, TILE_C={TILE_C}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print()

    # Forward pass
    print("Running forward pass...")
    log_Z, all_alpha, ring_checkpoints, log_norm_checkpoints = simulate_forward_pass()
    print(f"log_Z: {log_Z.item():.6f}")
    print(f"log_norm_checkpoints: {log_norm_checkpoints.tolist()}")
    print()

    # Backward pass for beta
    print("Running backward pass for beta...")
    beta = simulate_backward_full()
    print(f"beta[0] range: [{beta[0].min():.4f}, {beta[0].max():.4f}]")
    print(f"beta[27] range: [{beta[27].min():.4f}, {beta[27].max():.4f}]")
    print()

    # Compute marginals for specific positions
    print("=" * 70)
    print("SEGMENT 0 MARGINALS (positions 0-26)")
    print("log_norm_at_ckpt = 0.0")
    print("=" * 70)

    # For segment 0, alpha values are unnormalized
    # But wait - the recomputed alpha in the backward pass uses NORMALIZED checkpoints
    # Let me check this...

    # Actually, for segment 0 with ckpt_idx=0:
    # - log_norm_checkpoints[0] = 0 (no normalization accumulated yet)
    # - The checkpoint stores the initial state (alpha=0)
    # - Recomputation starts from this initial state
    # - So recomputed alpha for segment 0 is UNNORMALIZED

    marginals_seg0 = []
    for t in [0, 1, 2, 26]:
        m = simulate_backward_segment0_marginal_at_t(
            t, all_alpha, beta, log_Z.item(), log_norm_checkpoints[0].item()
        )
        marginals_seg0.append((t, m))

    print("\n" + "=" * 70)
    print("SEGMENT 1 MARGINALS (positions 27-47)")
    print(f"log_norm_at_ckpt = {log_norm_checkpoints[1].item():.6f}")
    print("=" * 70)

    # For segment 1, alpha values are normalized
    # The stored all_alpha contains unnormalized values, but for segment 1
    # we need to use normalized values (subtract log_norm_checkpoints[1])

    # Actually, let me recompute alpha for segment 1 properly
    print("\nRecomputing alpha for segment 1 from checkpoint...")

    # Load checkpoint 1 (the ring buffer state at t=27)
    alpha_ring_seg1 = ring_checkpoints[1].clone()
    print("  Checkpoint 1 ring buffer:")
    for k_slot in range(K):
        alpha_slot = alpha_ring_seg1[k_slot, :]
        if alpha_slot.max() > NEG_INF + 1:
            print(f"    slot {k_slot}: min={alpha_slot.min():.4f}, max={alpha_slot.max():.4f}")

    # The checkpoint stores NORMALIZED alpha values (after shift at t=27)
    # So for segment 1, log_norm_at_ckpt[1] is the accumulated shift

    for t in [27, 28, 47]:
        m = simulate_backward_segment0_marginal_at_t(
            t, all_alpha, beta, log_Z.item(), log_norm_checkpoints[1].item()
        )

    # Now run the actual Triton implementation and compare
    print("\n" + "=" * 70)
    print("COMPARING WITH TRITON IMPLEMENTATION")
    print("=" * 70)

    try:
        from flash_semicrf.streaming import (
            launch_streaming_triton_marginals,
            semi_crf_streaming_marginals_pytorch,
        )
        from flash_semicrf.streaming.triton_forward import launch_streaming_triton_kernel

        # PyTorch reference
        pytorch_marginals, log_Z_pytorch = semi_crf_streaming_marginals_pytorch(
            cum_scores.cpu(), transition.cpu(), duration_bias.cpu(), lengths.cpu(), K
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

        print(f"\nPyTorch log_Z: {log_Z_pytorch[0].item():.6f}")
        print(f"Triton log_Z: {log_Z_triton[0].item():.6f}")
        print(f"Simulated log_Z: {log_Z.item():.6f}")

        print(f"\nlog_norm_checkpoints from Triton: {log_norm_ckpts[0].tolist()}")

        print("\nMarginals comparison at key positions:")
        print(f"  {'t':>3} | {'PyTorch':>10} | {'Triton':>10} | {'Ratio':>8}")
        for t in [0, 1, 2, 26, 27, 28, 47]:
            pt_val = pytorch_marginals[0, t].item()
            tr_val = triton_marginals[0, t].item()
            ratio = tr_val / pt_val if pt_val != 0 else 0
            print(f"  {t:3d} | {pt_val:10.6f} | {tr_val:10.6f} | {ratio:8.4f}")

    except ImportError as e:
        print(f"Could not import Triton implementation: {e}")


if __name__ == "__main__":
    main()
