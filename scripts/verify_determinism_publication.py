#!/usr/bin/env python3
"""Publication-quality verification of Triton kernel determinism and correctness.

This script demonstrates that the Triton Semi-CRF implementation is:
1. **Deterministic**: Identical inputs produce identical outputs across runs
2. **Correct**: Results match the PyTorch reference implementation
3. **Numerically stable**: Works across small to extreme sequence lengths

Designed for inclusion in publications to validate the implementation.

Usage:
    python scripts/verify_determinism_publication.py

    # Quick mode (faster, fewer runs):
    python scripts/verify_determinism_publication.py --quick

    # Verbose mode (show per-run details):
    python scripts/verify_determinism_publication.py --verbose

    # Include genome-scale test (T=50k, K=1000, C=32):
    python scripts/verify_determinism_publication.py --genome

Output includes:
    - Scale configurations tested (small, medium, large, extreme)
    - Run-to-run determinism verification
    - PyTorch reference comparison
    - Timing and memory statistics
    - Publication-ready summary table

Note on numerical stability:
    The Triton kernel uses incremental log-normalization to handle extreme
    sequence lengths (T > 10,000) where naive log-space computation would
    overflow. This is validated by the "extreme" scale test. The optional
    "genome" scale (--genome flag) pushes this to T=50,000 with K=1000,
    demonstrating stability at scales relevant for genomic segmentation.
"""

import argparse
import gc
import time
from dataclasses import dataclass

import torch

from torch_semimarkov.streaming.pytorch_reference import (
    semi_crf_streaming_backward_pytorch,
    semi_crf_streaming_forward_pytorch,
)
from torch_semimarkov.streaming.triton_backward import (
    launch_streaming_triton_backward,
)
from torch_semimarkov.streaming.triton_forward import (
    _compute_checkpoint_interval,
    _next_power_of_2,
    launch_streaming_triton_kernel,
)


@dataclass
class ScaleConfig:
    """Configuration for a test scale."""

    name: str
    T: int  # Sequence length
    C: int  # Number of classes
    K: int  # Max segment duration
    batch: int
    description: str


# Define test scales from small to extreme
SCALE_CONFIGS = [
    ScaleConfig(
        name="small",
        T=50,
        C=8,
        K=10,
        batch=4,
        description="Quick validation, subsecond runtime",
    ),
    ScaleConfig(
        name="medium",
        T=500,
        C=32,
        K=20,
        batch=8,
        description="Realistic use case (speech/NER)",
    ),
    ScaleConfig(
        name="large",
        T=2000,
        C=64,
        K=50,
        batch=4,
        description="Stress test for register pressure",
    ),
    ScaleConfig(
        name="extreme",
        T=10000,
        C=16,
        K=100,
        batch=2,
        description="Numerical stability at extreme T",
    ),
]

# Optional genome-scale config (enabled with --genome flag)
GENOME_SCALE_CONFIG = ScaleConfig(
    name="genome",
    T=50000,
    C=32,
    K=1000,
    batch=1,
    description="Genome-scale: 50k positions, 1k max segment length",
)


@dataclass
class TestResult:
    """Results from testing a single scale configuration."""

    scale_name: str
    config: ScaleConfig

    # Determinism metrics (run-to-run)
    is_deterministic: bool
    max_diff_cs: float  # max |run_i - run_0| for grad_cum_scores
    max_diff_tr: float  # max |run_i - run_0| for grad_transition
    max_diff_db: float  # max |run_i - run_0| for grad_duration_bias

    # Correctness metrics (vs PyTorch reference)
    is_correct: bool
    pytorch_diff_cs: float
    pytorch_diff_tr: float
    pytorch_diff_db: float
    pytorch_rel_diff_cs: float
    pytorch_rel_diff_tr: float
    pytorch_rel_diff_db: float

    # Performance metrics
    forward_time_ms: float
    backward_time_ms: float
    peak_memory_mb: float

    # Additional info
    num_runs: int
    checkpoint_interval: int
    num_checkpoints: int
    C_PAD: int


def run_triton_forward_backward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
) -> tuple[dict, float, float]:
    """Run Triton forward and backward passes, returning gradients and timing."""
    grad_output = torch.ones(cum_scores.shape[0], device=cum_scores.device, dtype=cum_scores.dtype)

    # Forward pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    partition, ring_ckpts, ckpt_interval, log_norm_ckpts = launch_streaming_triton_kernel(
        cum_scores, transition, duration_bias, lengths, K, "log"
    )

    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - t0) * 1000

    # Backward pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()

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

    torch.cuda.synchronize()
    backward_ms = (time.perf_counter() - t0) * 1000

    grad_cum_scores, grad_transition, grad_duration_bias, _, _, _ = grads

    return (
        {
            "grad_cs": grad_cum_scores.clone(),
            "grad_tr": grad_transition.clone(),
            "grad_db": grad_duration_bias.clone(),
            "partition": partition.clone(),
            "ckpt_interval": ckpt_interval,
        },
        forward_ms,
        backward_ms,
    )


def run_pytorch_reference(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
) -> dict:
    """Run PyTorch reference implementation for comparison."""
    # Move to CPU for PyTorch reference
    cum_scores_cpu = cum_scores.cpu()
    transition_cpu = transition.cpu()
    duration_bias_cpu = duration_bias.cpu()
    lengths_cpu = lengths.cpu()

    # Forward
    log_Z, ring_ckpts, interval, log_norm_ckpts = semi_crf_streaming_forward_pytorch(
        cum_scores_cpu, transition_cpu, duration_bias_cpu, lengths_cpu, K
    )

    # Backward
    grads = semi_crf_streaming_backward_pytorch(
        cum_scores_cpu,
        transition_cpu,
        duration_bias_cpu,
        lengths_cpu,
        K,
        log_Z,
        ring_ckpts,
        log_norm_ckpts,
        interval,
    )

    grad_cs, grad_tr, grad_db = grads[:3]

    # PyTorch returns per-batch gradients, Triton returns reduced
    # Reduce PyTorch gradients for fair comparison
    grad_tr_reduced = grad_tr.sum(dim=0)  # (batch, C, C) -> (C, C)
    grad_db_reduced = grad_db.sum(dim=0)  # (batch, K, C) -> (K, C)

    return {
        "grad_cs": grad_cs,
        "grad_tr": grad_tr_reduced,
        "grad_db": grad_db_reduced,
        "partition": log_Z,
    }


def test_scale(
    config: ScaleConfig,
    num_runs: int = 5,
    device: str = "cuda",
    verbose: bool = False,
) -> TestResult:
    """Test determinism and correctness at a specific scale."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Create test data
    cum_scores = torch.randn(
        config.batch, config.T + 1, config.C, device=device, dtype=torch.float32
    )
    cum_scores[:, 0] = 0.0
    cum_scores[:, 1:] = torch.cumsum(cum_scores[:, 1:], dim=1)

    transition = torch.randn(config.C, config.C, device=device, dtype=torch.float32) * 0.1
    duration_bias = torch.randn(config.K, config.C, device=device, dtype=torch.float32) * 0.1
    lengths = torch.full((config.batch,), config.T, device=device, dtype=torch.int64)

    C_PAD = _next_power_of_2(config.C)
    ckpt_interval = _compute_checkpoint_interval(config.T, config.K)
    num_ckpts = (config.T + ckpt_interval - 1) // ckpt_interval

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Run multiple times for determinism check
    results = []
    forward_times = []
    backward_times = []

    with torch.no_grad():
        for run_idx in range(num_runs):
            result, fwd_ms, bwd_ms = run_triton_forward_backward(
                cum_scores, transition, duration_bias, lengths, config.K
            )
            results.append(result)
            forward_times.append(fwd_ms)
            backward_times.append(bwd_ms)

            if verbose:
                print(f"    Run {run_idx}: fwd={fwd_ms:.2f}ms, bwd={bwd_ms:.2f}ms")

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

    # Check determinism (all runs should match run 0)
    max_diff_cs = 0.0
    max_diff_tr = 0.0
    max_diff_db = 0.0

    for run_idx in range(1, num_runs):
        max_diff_cs = max(
            max_diff_cs,
            (results[0]["grad_cs"] - results[run_idx]["grad_cs"]).abs().max().item(),
        )
        max_diff_tr = max(
            max_diff_tr,
            (results[0]["grad_tr"] - results[run_idx]["grad_tr"]).abs().max().item(),
        )
        max_diff_db = max(
            max_diff_db,
            (results[0]["grad_db"] - results[run_idx]["grad_db"]).abs().max().item(),
        )

    is_deterministic = max_diff_cs == 0.0 and max_diff_tr == 0.0 and max_diff_db == 0.0

    # Compare against PyTorch reference
    pytorch_result = run_pytorch_reference(cum_scores, transition, duration_bias, lengths, config.K)

    triton_cs = results[0]["grad_cs"].cpu()
    triton_tr = results[0]["grad_tr"].cpu()
    triton_db = results[0]["grad_db"].cpu()

    pytorch_diff_cs = (triton_cs - pytorch_result["grad_cs"]).abs().max().item()
    pytorch_diff_tr = (triton_tr - pytorch_result["grad_tr"]).abs().max().item()
    pytorch_diff_db = (triton_db - pytorch_result["grad_db"]).abs().max().item()

    # Relative differences
    eps = 1e-8
    mag_cs = pytorch_result["grad_cs"].abs().mean().item() + eps
    mag_tr = pytorch_result["grad_tr"].abs().mean().item() + eps
    mag_db = pytorch_result["grad_db"].abs().mean().item() + eps

    pytorch_rel_diff_cs = pytorch_diff_cs / mag_cs
    pytorch_rel_diff_tr = pytorch_diff_tr / mag_tr
    pytorch_rel_diff_db = pytorch_diff_db / mag_db

    # Correctness threshold: relative error < 1% or absolute error < 1e-4
    is_correct = (
        (pytorch_rel_diff_cs < 0.01 or pytorch_diff_cs < 1e-4)
        and (pytorch_rel_diff_tr < 0.01 or pytorch_diff_tr < 1e-4)
        and (pytorch_rel_diff_db < 0.01 or pytorch_diff_db < 1e-4)
    )

    return TestResult(
        scale_name=config.name,
        config=config,
        is_deterministic=is_deterministic,
        max_diff_cs=max_diff_cs,
        max_diff_tr=max_diff_tr,
        max_diff_db=max_diff_db,
        is_correct=is_correct,
        pytorch_diff_cs=pytorch_diff_cs,
        pytorch_diff_tr=pytorch_diff_tr,
        pytorch_diff_db=pytorch_diff_db,
        pytorch_rel_diff_cs=pytorch_rel_diff_cs,
        pytorch_rel_diff_tr=pytorch_rel_diff_tr,
        pytorch_rel_diff_db=pytorch_rel_diff_db,
        forward_time_ms=sum(forward_times) / len(forward_times),
        backward_time_ms=sum(backward_times) / len(backward_times),
        peak_memory_mb=peak_memory,
        num_runs=num_runs,
        checkpoint_interval=results[0]["ckpt_interval"],
        num_checkpoints=num_ckpts,
        C_PAD=C_PAD,
    )


def print_result_details(result: TestResult) -> None:
    """Print detailed results for a single scale."""
    cfg = result.config
    det_status = "PASS" if result.is_deterministic else "FAIL"
    cor_status = "PASS" if result.is_correct else "FAIL"

    print(f"\n  [{result.scale_name.upper()}] T={cfg.T}, C={cfg.C}, K={cfg.K}, batch={cfg.batch}")
    print(f"  {cfg.description}")
    print(
        f"  C_PAD={result.C_PAD}, checkpoint_interval={result.checkpoint_interval}, num_ckpts={result.num_checkpoints}"
    )
    print()
    print(f"  Determinism ({result.num_runs} runs): [{det_status}]")
    print(
        f"    max_diff: grad_cs={result.max_diff_cs:.2e}, grad_tr={result.max_diff_tr:.2e}, grad_db={result.max_diff_db:.2e}"
    )
    print()
    print(f"  Correctness (vs PyTorch): [{cor_status}]")
    print(
        f"    abs_diff: grad_cs={result.pytorch_diff_cs:.2e}, grad_tr={result.pytorch_diff_tr:.2e}, grad_db={result.pytorch_diff_db:.2e}"
    )
    print(
        f"    rel_diff: grad_cs={result.pytorch_rel_diff_cs:.2%}, grad_tr={result.pytorch_rel_diff_tr:.2%}, grad_db={result.pytorch_rel_diff_db:.2%}"
    )
    print()
    print("  Performance:")
    print(f"    forward: {result.forward_time_ms:.2f}ms, backward: {result.backward_time_ms:.2f}ms")
    print(f"    peak memory: {result.peak_memory_mb:.1f}MB")


def print_summary_table(results: list[TestResult]) -> None:
    """Print publication-ready summary table."""
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (Publication Ready)")
    print("=" * 90)
    print()
    print(
        "| Scale   | T      | C  | K   | Batch | Deterministic | Correct | Fwd (ms) | Bwd (ms) | Mem (MB) |"
    )
    print(
        "|---------|--------|----|----|-------|---------------|---------|----------|----------|----------|"
    )

    for r in results:
        det = "Yes" if r.is_deterministic else "NO"
        cor = "Yes" if r.is_correct else "NO"
        print(
            f"| {r.scale_name:<7} | {r.config.T:>6} | {r.config.C:>2} | {r.config.K:>3} | "
            f"{r.config.batch:>5} | {det:>13} | {cor:>7} | {r.forward_time_ms:>8.2f} | "
            f"{r.backward_time_ms:>8.2f} | {r.peak_memory_mb:>8.1f} |"
        )

    print()

    # Correctness details
    print("Correctness Details (max relative error vs PyTorch reference):")
    print("| Scale   | grad_cum_scores | grad_transition | grad_duration_bias |")
    print("|---------|-----------------|-----------------|---------------------|")
    for r in results:
        print(
            f"| {r.scale_name:<7} | {r.pytorch_rel_diff_cs:>15.2e} | {r.pytorch_rel_diff_tr:>15.2e} | "
            f"{r.pytorch_rel_diff_db:>19.2e} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Verify Triton Semi-CRF determinism and correctness across scales"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer runs, skip extreme scale",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-run timing details",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs for determinism check (default: 5)",
    )
    parser.add_argument(
        "--genome",
        action="store_true",
        help="Include genome-scale test: T=50k, K=1000, C=32 (requires significant GPU memory and time)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TRITON SEMI-CRF DETERMINISM AND CORRECTNESS VERIFICATION")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available. This script requires a GPU.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"\nDevice: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    num_runs = 3 if args.quick else args.num_runs
    scales = SCALE_CONFIGS[:-1] if args.quick else SCALE_CONFIGS.copy()

    if args.genome:
        scales.append(GENOME_SCALE_CONFIG)
        print("\n*** GENOME SCALE ENABLED: T=50k, K=1000, C=32 ***")
        print("    This will require significant GPU memory (~10-20GB) and time (~minutes).")

    print(f"\nTesting {len(scales)} scale configurations with {num_runs} runs each...")

    results = []
    all_pass = True

    for config in scales:
        print(f"\nTesting {config.name} scale...", end="", flush=True)

        # Clean up memory between tests
        gc.collect()
        torch.cuda.empty_cache()

        try:
            result = test_scale(config, num_runs=num_runs, verbose=args.verbose)
            results.append(result)

            if result.is_deterministic and result.is_correct:
                print(" PASS")
            else:
                print(" FAIL")
                all_pass = False

            print_result_details(result)

        except Exception as e:
            print(f" ERROR: {e}")
            all_pass = False

    # Print summary
    print_summary_table(results)

    # Final verdict
    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: ALL TESTS PASSED")
        print()
        max_T = max(r.config.T for r in results)
        max_K = max(r.config.K for r in results)
        print("The Triton Semi-CRF implementation is:")
        print("  - Deterministic: Identical results across multiple runs")
        print("  - Correct: Matches PyTorch reference within numerical tolerance")
        print(f"  - Numerically stable: Works from T=50 to T={max_T:,}, K up to {max_K:,}")
    else:
        print("OVERALL: SOME TESTS FAILED")
        print()
        print("Review the detailed results above for failures.")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
