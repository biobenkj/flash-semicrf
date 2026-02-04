#!/usr/bin/env python3
"""Find which parameter combinations trigger non-deterministic forward/backward.

Usage:
    python src/torch_semimarkov/streaming/find_determinism.py

To test with different num_warps values:
    1. Edit triton_backward.py line ~888: change `num_warps=2` to desired value
    2. Edit triton_forward.py line ~957 and ~1115: change `num_warps=2` to desired value
    3. Run this script and check for NaN or determinism failures

Expected behavior after loop tiling fix:
    - num_warps=2: Should pass (baseline)
    - num_warps=4: Should now pass (was failing before tiling)
    - num_warps=8: May pass depending on register pressure
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


def test_config(batch, T, C, K, num_runs=10, device="cuda"):
    """Test if a specific config is deterministic."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    cum_scores = torch.randn(batch, T + 1, C, device=device, dtype=torch.float32)
    cum_scores[:, 0] = 0.0
    cum_scores[:, 1:] = torch.cumsum(cum_scores[:, 1:], dim=1)

    transition = torch.randn(C, C, device=device, dtype=torch.float32) * 0.1
    duration_bias = torch.randn(K, C, device=device, dtype=torch.float32) * 0.1
    lengths = torch.full((batch,), T, device=device, dtype=torch.int64)

    results = []
    with torch.no_grad():
        for _ in range(num_runs):
            p, _, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K, "log"
            )
            results.append(p.clone())

    # Check all pairs
    max_diff = 0.0
    for i in range(1, num_runs):
        diff = (results[0] - results[i]).abs().max().item()
        max_diff = max(max_diff, diff)

    return max_diff


def test_batch_pattern(batch, T, C, K, num_runs=20, device="cuda"):
    """Test which batch indices are affected by non-determinism."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    cum_scores = torch.randn(batch, T + 1, C, device=device, dtype=torch.float32)
    cum_scores[:, 0] = 0.0
    cum_scores[:, 1:] = torch.cumsum(cum_scores[:, 1:], dim=1)

    transition = torch.randn(C, C, device=device, dtype=torch.float32) * 0.1
    duration_bias = torch.randn(K, C, device=device, dtype=torch.float32) * 0.1
    lengths = torch.full((batch,), T, device=device, dtype=torch.int64)

    results = []
    with torch.no_grad():
        for _ in range(num_runs):
            p, _, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K, "log"
            )
            results.append(p.clone())

    # Check which batch indices are affected
    affected_batches = set()
    for i in range(1, num_runs):
        for b in range(batch):
            if results[0][b] != results[i][b]:
                affected_batches.add(b)

    return affected_batches


def test_backward_config(batch, T, C, K, num_runs=10, device="cuda"):
    """Test if backward pass is deterministic and free of NaN."""
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
    nan_count = 0

    with torch.no_grad():
        for _ in range(num_runs):
            # Forward pass to get checkpoints
            partition, ring_ckpts, ckpt_interval, log_norm_ckpts = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K, "log"
            )

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
            if (
                torch.isnan(grad_cum_scores).any()
                or torch.isnan(grad_transition).any()
                or torch.isnan(grad_duration_bias).any()
            ):
                nan_count += 1

            results.append(
                {
                    "grad_cs": grad_cum_scores.clone(),
                    "grad_tr": grad_transition.clone(),
                    "grad_db": grad_duration_bias.clone(),
                }
            )

    # Check determinism across runs
    max_diff_cs = 0.0
    max_diff_tr = 0.0
    max_diff_db = 0.0
    for i in range(1, num_runs):
        max_diff_cs = max(
            max_diff_cs, (results[0]["grad_cs"] - results[i]["grad_cs"]).abs().max().item()
        )
        max_diff_tr = max(
            max_diff_tr, (results[0]["grad_tr"] - results[i]["grad_tr"]).abs().max().item()
        )
        max_diff_db = max(
            max_diff_db, (results[0]["grad_db"] - results[i]["grad_db"]).abs().max().item()
        )

    # Compute gradient magnitudes for relative error calculation
    # Use mean of absolute values across all runs as reference magnitude
    eps = 1e-8
    mag_cs = torch.stack([r["grad_cs"].abs() for r in results]).mean().item() + eps
    mag_tr = torch.stack([r["grad_tr"].abs() for r in results]).mean().item() + eps
    mag_db = torch.stack([r["grad_db"].abs() for r in results]).mean().item() + eps

    return {
        "max_diff_cs": max_diff_cs,
        "max_diff_tr": max_diff_tr,
        "max_diff_db": max_diff_db,
        "rel_diff_cs": max_diff_cs / mag_cs,
        "rel_diff_tr": max_diff_tr / mag_tr,
        "rel_diff_db": max_diff_db / mag_db,
        "mag_cs": mag_cs,
        "mag_tr": mag_tr,
        "mag_db": mag_db,
        "nan_count": nan_count,
    }


def main():
    print("=" * 70)
    print("SEARCHING FOR NON-DETERMINISM BOUNDARY")
    print("=" * 70)

    # First, confirm the known bad case and check batch pattern
    print("\n=== Known bad configuration: T=50, C=8, K=5 ===")
    for batch in [2, 4, 8, 16]:
        affected = test_batch_pattern(batch, 50, 8, 5, num_runs=30)
        print(f"  batch={batch}: affected batches = {sorted(affected) if affected else 'none'}")

    # Check if the issue is specific to certain batch indices
    print("\n=== Testing batch_size=1 ===")
    max_diff = test_config(1, 50, 8, 5, num_runs=30)
    print(f"  batch=1: max_diff = {max_diff}")

    # Now search for the boundary
    print("\n=== Searching parameter space ===")
    print("Testing T in [20, 30, 40, 50, 60], C in [4, 8, 10, 16], K in [3, 5, 8]")
    print()

    bad_configs = []

    for T in [20, 30, 40, 50, 60, 80, 100]:
        for C in [4, 8, 10, 16, 32]:
            for K in [3, 5, 8, 10]:
                if K > T:
                    continue
                max_diff = test_config(2, T, C, K, num_runs=15)
                C_PAD = _next_power_of_2(C)
                interval = _compute_checkpoint_interval(T, K)
                num_ckpts = (T + interval - 1) // interval

                if max_diff > 0:
                    bad_configs.append((T, C, K, C_PAD, interval, num_ckpts, max_diff))
                    print(
                        f"  [FAIL] T={T:3d}, C={C:2d}, K={K:2d}, C_PAD={C_PAD:2d}, interval={interval:2d}, ckpts={num_ckpts} : diff={max_diff:.6f}"
                    )

    print("\n" + "=" * 70)
    print("FORWARD SUMMARY")
    print("=" * 70)

    if bad_configs:
        print(f"\nFound {len(bad_configs)} non-deterministic FORWARD configurations:")
        for T, C, K, C_PAD, interval, num_ckpts, _diff in bad_configs:
            print(f"  T={T}, C={C}, K={K}, C_PAD={C_PAD}, interval={interval}, ckpts={num_ckpts}")

        # Look for patterns
        print("\n=== Pattern analysis ===")

        # Check if C == C_PAD is the issue (no padding)
        no_pad = [c for c in bad_configs if c[1] == c[3]]
        with_pad = [c for c in bad_configs if c[1] != c[3]]
        print(f"  Configs where C == C_PAD (no padding): {len(no_pad)}")
        print(f"  Configs where C != C_PAD (with padding): {len(with_pad)}")

        # Check checkpoint counts
        ckpt_counts = {c[5] for c in bad_configs}
        print(f"  Checkpoint counts in bad configs: {sorted(ckpt_counts)}")

        # Check intervals
        intervals = {c[4] for c in bad_configs}
        print(f"  Intervals in bad configs: {sorted(intervals)}")
    else:
        print("\n[PASS] All tested FORWARD configurations are deterministic!")

    # ========== BACKWARD PASS TESTING ==========
    print("\n" + "=" * 70)
    print("BACKWARD PASS DETERMINISM TEST")
    print("=" * 70)
    print("\nTesting backward pass for determinism and NaN issues...")

    bad_backward_configs = []

    for T in [20, 50, 100]:
        for C in [8, 16, 32]:
            for K in [5, 10]:
                if K > T:
                    continue
                result = test_backward_config(2, T, C, K, num_runs=15)
                C_PAD = _next_power_of_2(C)

                is_bad = (
                    result["max_diff_cs"] > 0
                    or result["max_diff_tr"] > 0
                    or result["max_diff_db"] > 0
                    or result["nan_count"] > 0
                )

                if is_bad:
                    bad_backward_configs.append((T, C, K, C_PAD, result))
                    print(
                        f"  [FAIL] T={T:3d}, C={C:2d}, K={K:2d}, C_PAD={C_PAD:2d} : "
                        f"NaN={result['nan_count']}"
                    )
                    print(
                        f"         abs_diff: cs={result['max_diff_cs']:.6f}, "
                        f"tr={result['max_diff_tr']:.6f}, db={result['max_diff_db']:.6f}"
                    )
                    print(
                        f"         rel_diff: cs={result['rel_diff_cs']:.2%}, "
                        f"tr={result['rel_diff_tr']:.2%}, db={result['rel_diff_db']:.2%}"
                    )
                    print(
                        f"         magnitude: cs={result['mag_cs']:.4f}, "
                        f"tr={result['mag_tr']:.4f}, db={result['mag_db']:.4f}"
                    )

    print("\n" + "=" * 70)
    print("BACKWARD SUMMARY")
    print("=" * 70)

    if bad_backward_configs:
        print(f"\nFound {len(bad_backward_configs)} configurations with non-zero BACKWARD diff:")

        # Categorize by severity
        severe = []  # > 1% relative diff
        moderate = []  # 0.1% - 1% relative diff
        minor = []  # < 0.1% relative diff (likely acceptable GPU noise)

        for T, C, K, C_PAD, result in bad_backward_configs:
            max_rel = max(result["rel_diff_cs"], result["rel_diff_tr"], result["rel_diff_db"])
            entry = (T, C, K, C_PAD, result, max_rel)
            if max_rel > 0.01:
                severe.append(entry)
            elif max_rel > 0.001:
                moderate.append(entry)
            else:
                minor.append(entry)

        if severe:
            print(f"\n  [SEVERE] {len(severe)} configs with >1% relative diff (likely bug):")
            for T, C, K, _C_PAD, _result, max_rel in severe:
                print(f"    T={T}, C={C}, K={K}, max_rel={max_rel:.2%}")

        if moderate:
            print(
                f"\n  [MODERATE] {len(moderate)} configs with 0.1%-1% relative diff (worth investigating):"
            )
            for T, C, K, _C_PAD, _result, max_rel in moderate:
                print(f"    T={T}, C={C}, K={K}, max_rel={max_rel:.2%}")

        if minor:
            print(
                f"\n  [MINOR] {len(minor)} configs with <0.1% relative diff (acceptable GPU noise):"
            )
            for T, C, K, _C_PAD, _result, max_rel in minor:
                print(f"    T={T}, C={C}, K={K}, max_rel={max_rel:.4%}")

        # Overall assessment
        print("\n" + "-" * 50)
        if severe:
            print("[ASSESSMENT] FAIL - Severe non-determinism detected")
        elif moderate:
            print("[ASSESSMENT] WARNING - Moderate non-determinism, may be acceptable for training")
        else:
            print("[ASSESSMENT] PASS - Only minor GPU floating-point noise detected")
    else:
        print("\n[PASS] All tested BACKWARD configurations are deterministic and NaN-free!")


if __name__ == "__main__":
    main()
