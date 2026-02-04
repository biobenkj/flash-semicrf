#!/usr/bin/env python3
"""Debug script for backward pass non-determinism.

Runs the two failing configurations multiple times and compares results
to identify the source of non-determinism.

Failing configs from find_determinism.py:
  1. T=20, C=32, K=10 - 286.82% relative diff
  2. T=100, C=8, K=5 - 656.11% relative diff

Usage (capture kernel debug prints):
    python scripts/debug_backward_nondeterminism.py 2>&1 | tee debug_backward.log
    bash scripts/parse_debug_output.sh debug_backward.log

The kernel has debug prints at t=9, k=1 which will show:
  - global_max and global_sum_exp (Pass 1 statistics)
  - log_scale and scale (normalization values)
  - grad_cs_t_local_sum and grad_db_k_local_sum (local accumulators)

If these values differ between runs, it indicates where non-determinism enters.
"""

import torch

from torch_semimarkov.streaming.triton_backward import (
    launch_streaming_triton_backward,
)
from torch_semimarkov.streaming.triton_forward import (
    _compute_checkpoint_interval,
    _next_power_of_2,
    launch_streaming_triton_kernel,
)
from torch_semimarkov.streaming.pytorch_reference import (
    semi_crf_streaming_backward_pytorch,
    semi_crf_streaming_forward_pytorch,
)


def test_config_detailed(batch, T, C, K, num_runs=5, device="cuda"):
    """Test a specific config with detailed output for debugging."""
    print(f"\n{'='*70}")
    print(f"TESTING: batch={batch}, T={T}, C={C}, K={K}")
    print(f"{'='*70}")

    C_PAD = _next_power_of_2(C)
    interval = _compute_checkpoint_interval(T, K)
    num_ckpts = (T + interval - 1) // interval

    print(f"C_PAD={C_PAD}, checkpoint_interval={interval}, num_ckpts={num_ckpts}")
    print(f"TILE_C estimate: {min(C_PAD, max(4, C_PAD // 4))}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    cum_scores = torch.randn(batch, T + 1, C, device=device, dtype=torch.float32)
    cum_scores[:, 0] = 0.0
    cum_scores[:, 1:] = torch.cumsum(cum_scores[:, 1:], dim=1)

    transition = torch.randn(C, C, device=device, dtype=torch.float32) * 0.1
    duration_bias = torch.randn(K, C, device=device, dtype=torch.float32) * 0.1
    lengths = torch.full((batch,), T, device=device, dtype=torch.int64)
    grad_output = torch.ones(batch, device=device, dtype=torch.float32)

    results = []

    # Get PyTorch reference for comparison
    print("\nComputing PyTorch reference...")
    # First run PyTorch forward to get checkpoints
    pytorch_log_Z, pytorch_ring_ckpts, pytorch_interval, pytorch_log_norm_ckpts = (
        semi_crf_streaming_forward_pytorch(
            cum_scores.cpu(),
            transition.cpu(),
            duration_bias.cpu(),
            lengths.cpu(),
            K,
        )
    )
    # Then run PyTorch backward with those checkpoints
    pytorch_grads = semi_crf_streaming_backward_pytorch(
        cum_scores.cpu(),
        transition.cpu(),
        duration_bias.cpu(),
        lengths.cpu(),
        K,
        pytorch_log_Z,
        pytorch_ring_ckpts,
        pytorch_log_norm_ckpts,
        pytorch_interval,
    )
    pytorch_grad_cs, pytorch_grad_tr, pytorch_grad_db = pytorch_grads[:3]
    # IMPORTANT: PyTorch returns per-batch gradients (batch, C, C) and (batch, K, C)
    # Triton returns reduced gradients (C, C) and (K, C) via einsum with grad_output
    # Since grad_output = ones, the einsum is equivalent to sum over batch
    pytorch_grad_tr_reduced = pytorch_grad_tr.sum(dim=0)  # (batch, C, C) -> (C, C)
    pytorch_grad_db_reduced = pytorch_grad_db.sum(dim=0)  # (batch, K, C) -> (K, C)
    print(f"  PyTorch log_Z = {pytorch_log_Z[0].item():.6f}")
    print(f"  PyTorch grad_tr shape: {pytorch_grad_tr.shape} -> reduced: {pytorch_grad_tr_reduced.shape}")
    print(f"  PyTorch grad_db shape: {pytorch_grad_db.shape} -> reduced: {pytorch_grad_db_reduced.shape}")
    # Use valid index based on C dimension
    c_idx = min(C - 1, 6)  # Use index 6 or last valid index
    print(f"  PyTorch grad_cs[0,9,{c_idx}] = {pytorch_grad_cs[0, 9, c_idx].item():.6f}")

    print(f"\nRunning {num_runs} Triton iterations...")

    with torch.no_grad():
        for run_idx in range(num_runs):
            # Forward pass
            partition, ring_ckpts, ckpt_interval, log_norm_ckpts = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K, "log"
            )

            if run_idx == 0:
                print(f"\nForward pass:")
                print(f"  partition[0] = {partition[0].item():.6f}")
                print(f"  log_norm_ckpts[0] = {log_norm_ckpts[0].tolist()}")

            # Backward pass
            grads = launch_streaming_triton_backward(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                partition,
                ring_ckpts,
                log_norm_ckpts,
                ckpt_interval,
                grad_output,
            )
            grad_cum_scores, grad_transition, grad_duration_bias, _, _, _ = grads

            # Check for NaN
            has_nan = (
                torch.isnan(grad_cum_scores).any()
                or torch.isnan(grad_transition).any()
                or torch.isnan(grad_duration_bias).any()
            )

            if has_nan:
                print(f"  Run {run_idx}: NaN detected!")

            results.append({
                "grad_cs": grad_cum_scores.clone(),
                "grad_tr": grad_transition.clone(),
                "grad_db": grad_duration_bias.clone(),
            })

    # Compare each run against PyTorch reference
    print(f"\n--- Comparison vs PyTorch Reference (grad_cs[0,9,{c_idx}]) ---")
    for run_idx in range(num_runs):
        triton_val = results[run_idx]["grad_cs"][0, 9, c_idx].item()
        pytorch_val = pytorch_grad_cs[0, 9, c_idx].item()
        diff = abs(triton_val - pytorch_val)
        rel_diff = diff / (abs(pytorch_val) + 1e-8)
        match = "[MATCH]" if rel_diff < 0.01 else "[DIFF]"
        print(f"  Run {run_idx}: Triton={triton_val:.6f}, PyTorch={pytorch_val:.6f}, "
              f"diff={diff:.6e}, rel={rel_diff:.2%} {match}")

    # Also compare full tensors
    # Use reduced PyTorch gradients for grad_tr and grad_db comparison
    print(f"\n--- Full Tensor Comparison vs PyTorch ---")
    for run_idx in range(num_runs):
        diff_cs = (results[run_idx]["grad_cs"].cpu() - pytorch_grad_cs).abs()
        diff_tr = (results[run_idx]["grad_tr"].cpu() - pytorch_grad_tr_reduced).abs()
        diff_db = (results[run_idx]["grad_db"].cpu() - pytorch_grad_db_reduced).abs()
        print(f"  Run {run_idx}: max_diff_cs={diff_cs.max():.6e}, "
              f"max_diff_tr={diff_tr.max():.6e}, max_diff_db={diff_db.max():.6e}")

    # Analyze differences between runs
    print(f"\n--- Difference Analysis (Run-to-Run) ---")

    ref = results[0]
    for run_idx in range(1, num_runs):
        curr = results[run_idx]

        diff_cs = (ref["grad_cs"] - curr["grad_cs"]).abs()
        diff_tr = (ref["grad_tr"] - curr["grad_tr"]).abs()
        diff_db = (ref["grad_db"] - curr["grad_db"]).abs()

        max_diff_cs = diff_cs.max().item()
        max_diff_tr = diff_tr.max().item()
        max_diff_db = diff_db.max().item()

        if max_diff_cs > 0 or max_diff_tr > 0 or max_diff_db > 0:
            print(f"\nRun 0 vs Run {run_idx}:")
            print(f"  grad_cum_scores: max_diff = {max_diff_cs:.6e}")
            print(f"  grad_transition: max_diff = {max_diff_tr:.6e}")
            print(f"  grad_duration_bias: max_diff = {max_diff_db:.6e}")

            # Find where the differences are
            if max_diff_cs > 0:
                # Find position of max diff in grad_cum_scores
                flat_idx = diff_cs.flatten().argmax().item()
                b_idx = flat_idx // (diff_cs.shape[1] * diff_cs.shape[2])
                t_idx = (flat_idx // diff_cs.shape[2]) % diff_cs.shape[1]
                c_idx = flat_idx % diff_cs.shape[2]
                print(f"    grad_cs max diff at: batch={b_idx}, t={t_idx}, c={c_idx}")
                print(f"      Run 0: {ref['grad_cs'][b_idx, t_idx, c_idx].item():.6f}")
                print(f"      Run {run_idx}: {curr['grad_cs'][b_idx, t_idx, c_idx].item():.6f}")

            if max_diff_tr > 0:
                # Find position of max diff in grad_transition
                flat_idx = diff_tr.flatten().argmax().item()
                if len(diff_tr.shape) == 3:  # (K, C, C)
                    k_idx = flat_idx // (diff_tr.shape[1] * diff_tr.shape[2])
                    src_idx = (flat_idx // diff_tr.shape[2]) % diff_tr.shape[1]
                    dst_idx = flat_idx % diff_tr.shape[2]
                    print(f"    grad_tr max diff at: k={k_idx}, src={src_idx}, dst={dst_idx}")
                    print(f"      Run 0: {ref['grad_tr'][k_idx, src_idx, dst_idx].item():.6f}")
                    print(f"      Run {run_idx}: {curr['grad_tr'][k_idx, src_idx, dst_idx].item():.6f}")
                else:  # (C, C)
                    src_idx = flat_idx // diff_tr.shape[1]
                    dst_idx = flat_idx % diff_tr.shape[1]
                    print(f"    grad_tr max diff at: src={src_idx}, dst={dst_idx}")
                    print(f"      Run 0: {ref['grad_tr'][src_idx, dst_idx].item():.6f}")
                    print(f"      Run {run_idx}: {curr['grad_tr'][src_idx, dst_idx].item():.6f}")

            if max_diff_db > 0:
                # Find position of max diff in grad_duration_bias
                flat_idx = diff_db.flatten().argmax().item()
                k_idx = flat_idx // diff_db.shape[1]
                c_idx = flat_idx % diff_db.shape[1]
                print(f"    grad_db max diff at: k={k_idx}, c={c_idx}")
                print(f"      Run 0: {ref['grad_db'][k_idx, c_idx].item():.6f}")
                print(f"      Run {run_idx}: {curr['grad_db'][k_idx, c_idx].item():.6f}")

    # Print gradient statistics from first run
    print(f"\n--- Gradient Statistics (Run 0) ---")
    print(f"grad_cum_scores:")
    print(f"  shape: {ref['grad_cs'].shape}")
    print(f"  range: [{ref['grad_cs'].min().item():.6f}, {ref['grad_cs'].max().item():.6f}]")
    print(f"  mean: {ref['grad_cs'].mean().item():.6f}")
    print(f"  nonzero: {(ref['grad_cs'].abs() > 1e-8).sum().item()}")

    print(f"grad_transition:")
    print(f"  shape: {ref['grad_tr'].shape}")
    print(f"  range: [{ref['grad_tr'].min().item():.6f}, {ref['grad_tr'].max().item():.6f}]")
    print(f"  mean: {ref['grad_tr'].mean().item():.6f}")

    print(f"grad_duration_bias:")
    print(f"  shape: {ref['grad_db'].shape}")
    print(f"  range: [{ref['grad_db'].min().item():.6f}, {ref['grad_db'].max().item():.6f}]")
    print(f"  mean: {ref['grad_db'].mean().item():.6f}")

    # Check if all runs are identical
    all_identical = True
    for run_idx in range(1, num_runs):
        if not torch.allclose(ref["grad_cs"], results[run_idx]["grad_cs"]):
            all_identical = False
        if not torch.allclose(ref["grad_tr"], results[run_idx]["grad_tr"]):
            all_identical = False
        if not torch.allclose(ref["grad_db"], results[run_idx]["grad_db"]):
            all_identical = False

    if all_identical:
        print(f"\n[PASS] All {num_runs} runs produced identical results")
    else:
        print(f"\n[FAIL] Non-determinism detected across {num_runs} runs")

    return all_identical


def main():
    print("=" * 70)
    print("BACKWARD PASS NON-DETERMINISM DEBUG")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = "cuda"

    # Test failing configurations from find_determinism.py (with proper seeding)
    # Debug prints are conditional on t=9, k=1
    configs = [
        # (batch, T, C, K)
        (2, 20, 8, 10),    # Config 1: T=20, C=8, K=10 - 348.26% rel diff (FAILS)
        # (2, 50, 8, 5),   # Config 2: T=50, C=8, K=5 - 335.92% rel diff
        # (2, 100, 8, 5),  # Config 3: T=100, C=8, K=5 - 356.67% rel diff
    ]

    print("\n--- Testing FAILING configuration (with kernel debug prints at t=9) ---")
    print("NOTE: Kernel will print debug values for t=9, k=1 on each run.")
    print("      Multiple unique values = non-determinism source identified.")
    failing_results = []
    for batch, T, C, K in configs:
        is_deterministic = test_config_detailed(batch, T, C, K, num_runs=5, device=device)
        failing_results.append((batch, T, C, K, is_deterministic))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nFailing config result:")
    for batch, T, C, K, is_det in failing_results:
        status = "[PASS]" if is_det else "[FAIL]"
        print(f"  {status} batch={batch}, T={T}, C={C}, K={K}")

    print("\nNext steps:")
    print("  1. Run: bash scripts/parse_debug_output.sh <log_file>")
    print("  2. Check 'NON-DETERMINISM DEBUG' section for value differences")
    print("  3. If global_max differs -> tl.static_range issue")
    print("  4. If scale differs -> normalization issue")
    print("  5. If local accumulators differ -> Pass 2 accumulation issue")


if __name__ == "__main__":
    main()
