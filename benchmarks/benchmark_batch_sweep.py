#!/usr/bin/env python3
"""Batch-sweep benchmark: throughput, wall time, and memory vs batch size.

Holds T, K, C fixed, sweeps batch size B to demonstrate GPU utilization
scaling and the transition from SM-underutilized to compute-bound regime.

Key claims this benchmark supports:
  - Flash-SemiCRF converts semi-CRF inference from memory-bound to compute-bound
  - Throughput scales linearly with B until SM saturation, then plateaus
  - Wall time remains constant while B ≤ SM count (all programs fit in one wave)
  - At T=100K, linear scan streaming OOMs at B≥3 while Triton scales to B=256+

Generates publication figures:
  - Fig A: Throughput vs batch size (with SM count annotation)
  - Fig B: Peak GPU memory vs batch size (with linear scan OOM reference)
  - Fig C: Wall time vs batch size (showing wave structure)
  - Fig D: Per-element time vs batch size (showing amortization)

Place this script in the ``benchmarks/`` directory alongside
``benchmark_t_sweep.py`` so that ``from lib import ...`` resolves.

Usage:
    # Default: T=100K, K=200, C=6, Triton streaming
    python benchmarks/benchmark_batch_sweep.py

    # Custom T
    python benchmarks/benchmark_batch_sweep.py --T 50000

    # Include linear scan streaming for OOM comparison
    python benchmarks/benchmark_batch_sweep.py --include-linear-scan

    # Custom batch sizes
    python benchmarks/benchmark_batch_sweep.py --B 1,2,4,8,16,32,64,128,256

    # Quick mode
    python benchmarks/benchmark_batch_sweep.py --quick

    # Forward-only (isolate compute-bound claim)
    python benchmarks/benchmark_batch_sweep.py --phase forward
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from lib import (
    BenchmarkResult,
    run_single_benchmark,
)
from lib.runner import (
    LOG_MAX_BACKENDS,
    LOG_SEMIRING_ONLY_BACKENDS,
)

try:
    from flash_semicrf.streaming import HAS_TRITON
except ImportError:
    HAS_TRITON = False


# ======================================================================
# Default configurations
# ======================================================================

DEFAULT_B_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
QUICK_B_VALUES = [1, 4, 16, 64, 256]

# Extended sweep to capture wave boundaries on common GPUs
# L40S: 142 SMs, A100: 108 SMs, H100: 132 SMs
FINE_B_VALUES = [1, 2, 4, 8, 16, 32, 48, 64, 96, 108, 128, 142, 160, 192, 256, 284, 320, 512]

DEFAULT_BACKENDS = ["triton_streaming"]

REGIMES = {
    "genomics": {
        "description": "Genomics-scale: T=100K, K=200, C=6",
        "T": 100000,
        "K": 200,
        "C": 6,
        "B": DEFAULT_B_VALUES,
    },
    "genomics_fine": {
        "description": "Genomics-scale with fine SM-boundary resolution",
        "T": 100000,
        "K": 200,
        "C": 6,
        "B": FINE_B_VALUES,
    },
    "ner": {
        "description": "NER-scale: T=2048, K=16, C=24",
        "T": 2048,
        "K": 16,
        "C": 24,
        "B": DEFAULT_B_VALUES,
    },
}

# Plotting colors per backend
BACKEND_STYLE = {
    "triton_streaming": {"color": "#2563eb", "marker": "o", "ls": "-", "label": "Triton streaming"},
    "linear_scan_streaming": {
        "color": "#dc2626",
        "marker": "s",
        "ls": "--",
        "label": "Streaming scan",
    },
}


def _format_throughput(tp: float) -> str:
    """Format throughput (positions/second) with appropriate unit."""
    if tp <= 0:
        return "---"
    elif tp >= 1e9:
        return f"{tp/1e9:.1f}G pos/s"
    elif tp >= 1e6:
        return f"{tp/1e6:.1f}M pos/s"
    elif tp >= 1e3:
        return f"{tp/1e3:.1f}K pos/s"
    elif tp >= 1:
        return f"{tp:.1f} pos/s"
    else:
        return f"{tp:.2f} pos/s"


def _get_sm_count(device: torch.device) -> int:
    """Get the number of SMs on the current GPU."""
    return torch.cuda.get_device_properties(device).multi_processor_count


# ======================================================================
# Sweep runner
# ======================================================================


def run_sweep(
    B_values: list[int],
    T: int,
    K: int,
    C: int,
    backends: list[str],
    device: torch.device,
    repeats: int,
    phase: str,
    semiring: str,
    max_memory_gb: float,
) -> list[BenchmarkResult]:
    """Run all (backend x B) combinations, skipping after OOM."""

    results: list[BenchmarkResult] = []

    for backend in backends:
        # Check semiring compatibility up front
        if backend in LOG_SEMIRING_ONLY_BACKENDS and semiring != "Log":
            print(f"\n  SKIP {backend}: only supports Log semiring (got {semiring})")
            for B in B_values:
                results.append(
                    _skip_result(
                        T,
                        K,
                        C,
                        B,
                        backend,
                        semiring,
                        phase,
                        f"{backend} only supports Log semiring",
                    )
                )
            continue
        if backend in LOG_MAX_BACKENDS and semiring not in {"Log", "Max"}:
            print(f"\n  SKIP {backend}: only supports Log/Max semirings (got {semiring})")
            for B in B_values:
                results.append(
                    _skip_result(
                        T,
                        K,
                        C,
                        B,
                        backend,
                        semiring,
                        phase,
                        f"{backend} only supports Log/Max semirings",
                    )
                )
            continue

        style = BACKEND_STYLE.get(backend, {})
        label = style.get("label", backend)
        print(f"\n  {label.upper()}")
        print("  " + "-" * 100)
        print(
            f"  {'B':>6} | {'Time':>10} | {'Std':>8} | {'Mem Alloc':>10} | {'Mem Rsrv':>10} | "
            f"{'Throughput':>14} | {'Wall/elem':>12} | Status"
        )
        print("  " + "-" * 100)

        backend_oom = False

        for B in B_values:
            # Skip after OOM
            if backend_oom:
                r = _skip_result(T, K, C, B, backend, semiring, phase, "Previous B OOM'd")
                results.append(r)
                print(
                    f"  {B:>6} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | {'':>14} | {'':>12} | skip (prev OOM)"
                )
                continue

            # Predict memory and skip if too large
            # Conservative estimate: Triton uses ~O(B*K*C) for ring buffers + checkpoints
            # Linear scan uses ~O(B*T*K*C^2) for edge tensor
            if backend == "linear_scan_streaming":
                predicted_gb = B * T * K * C * C * 4 / (1024**3)
            else:
                # Triton: very conservative upper bound
                predicted_gb = B * K * C * 8 * 100 / (1024**3)  # generous overestimate

            if predicted_gb > max_memory_gb:
                r = _skip_result(
                    T,
                    K,
                    C,
                    B,
                    backend,
                    semiring,
                    phase,
                    f"predicted {predicted_gb:.1f}GB > {max_memory_gb:.1f}GB limit",
                )
                results.append(r)
                print(
                    f"  {B:>6} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | "
                    f"{'':>14} | {'':>12} | skip (predicted {predicted_gb:.1f}GB > {max_memory_gb:.1f}GB limit)"
                )
                backend_oom = True  # no point trying larger B
                continue

            gc.collect()
            torch.cuda.empty_cache()

            r = run_single_benchmark(
                T,
                K,
                C,
                B,
                backend,
                device,
                repeats,
                semiring_name=semiring,
                phase=phase,
            )
            results.append(r)

            if r.status == "success":
                tp = (B * T) / (r.time_ms_median / 1000) if r.time_ms_median > 0 else 0
                tp_str = _format_throughput(tp)
                wall_per_elem = r.time_ms_median / B if B > 0 else 0
                iqr = r.time_ms_iqr_high - r.time_ms_iqr_low
                std_str = f"{iqr:.1f}ms" if iqr < 1000 else f"{iqr/1000:.2f}s"
                wall_str = (
                    f"{wall_per_elem:.1f}ms"
                    if wall_per_elem < 1000
                    else f"{wall_per_elem/1000:.2f}s"
                )
                print(
                    f"  {B:>6} | {r.time_ms_median:>8.1f}ms | {std_str:>8} | {r.peak_allocated_gb:>8.3f}GB | "
                    f"{r.peak_reserved_gb:>8.3f}GB | {tp_str:>14} | {wall_str:>12} | ok"
                )
            elif r.status == "oom":
                backend_oom = True
                print(
                    f"  {B:>6} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | {'':>14} | {'':>12} | OOM"
                )
            else:
                print(
                    f"  {B:>6} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | {'':>14} | {'':>12} | {r.status}"
                )

    return results


def _skip_result(T, K, C, B, backend, semiring, phase, reason) -> BenchmarkResult:
    """Create a skip/not_tested BenchmarkResult."""
    return BenchmarkResult(
        T=T,
        K=K,
        C=C,
        B=B,
        KC=K * C,
        backend=backend,
        semiring=semiring,
        phase=phase,
        time_ms_median=float("nan"),
        time_ms_iqr_low=float("nan"),
        time_ms_iqr_high=float("nan"),
        time_per_position_ms=float("nan"),
        peak_allocated_gb=float("nan"),
        peak_reserved_gb=float("nan"),
        status="not_tested",
        error_msg=reason,
    )


# ======================================================================
# Summary
# ======================================================================


def print_sweep_summary(
    results: list[BenchmarkResult],
    backends: list[str],
    T: int,
    K: int,
    C: int,
    sm_count: int,
):
    """Print comparison summary with compute-bound analysis."""

    ok = {b: [r for r in results if r.backend == b and r.status == "success"] for b in backends}

    # Memory scaling per backend
    print(f"\n  MEMORY SCALING (T={T:,}, K={K}, C={C})")
    print("  " + "-" * 60)
    for b in backends:
        pts = ok.get(b, [])
        if len(pts) < 2:
            continue
        label = BACKEND_STYLE.get(b, {}).get("label", b)
        min_r = min(pts, key=lambda r: r.B)
        max_r = max(pts, key=lambda r: r.B)
        B_ratio = max_r.B / min_r.B
        mem_ratio = max_r.peak_allocated_gb / max(min_r.peak_allocated_gb, 1e-6)
        print(
            f"  {label:30s}: B grew {B_ratio:>6.0f}x, memory grew {mem_ratio:>5.1f}x "
            f"({min_r.peak_allocated_gb:.3f} -> {max_r.peak_allocated_gb:.3f} GB)"
        )

    # Throughput scaling analysis
    print(f"\n  THROUGHPUT SCALING (SM count: {sm_count})")
    print("  " + "-" * 60)
    for b in backends:
        pts = sorted(ok.get(b, []), key=lambda r: r.B)
        if len(pts) < 2:
            continue
        label = BACKEND_STYLE.get(b, {}).get("label", b)

        # Compute throughput for each B
        tp_data = []
        for r in pts:
            tp = (r.B * T) / (r.time_ms_median / 1000) if r.time_ms_median > 0 else 0
            tp_data.append((r.B, tp, r.time_ms_median))

        if not tp_data:
            continue

        # Find where throughput plateaus (< 10% gain from doubling B)
        plateau_B = None
        for i in range(1, len(tp_data)):
            B_prev, tp_prev, _ = tp_data[i - 1]
            B_curr, tp_curr, _ = tp_data[i]
            if B_prev > 0 and tp_prev > 0:
                B_scale = B_curr / B_prev
                tp_scale = tp_curr / tp_prev
                efficiency = tp_scale / B_scale  # 1.0 = perfect scaling
                if efficiency < 0.15 and plateau_B is None:
                    plateau_B = B_prev

        peak_B, peak_tp, _ = max(tp_data, key=lambda x: x[1])
        min_B, min_tp, _ = min(tp_data, key=lambda x: x[0])
        total_scaling = peak_tp / min_tp if min_tp > 0 else 0

        print(f"  {label}:")
        print(f"    B=1 throughput:      {_format_throughput(min_tp)}")
        print(f"    Peak throughput:     {_format_throughput(peak_tp)} (at B={peak_B})")
        print(f"    Total scaling:       {total_scaling:.1f}x over B={min_B}→{peak_B}")
        if plateau_B:
            print(f"    Compute-bound from:  B≈{plateau_B} (≈{plateau_B/sm_count:.1f}× SM count)")
        else:
            print(f"    Compute-bound from:  not reached in sweep (max B={tp_data[-1][0]})")

    # Wall time analysis (wave structure)
    print(f"\n  WAVE ANALYSIS (SM count: {sm_count})")
    print("  " + "-" * 60)
    for b in backends:
        pts = sorted(ok.get(b, []), key=lambda r: r.B)
        if len(pts) < 3:
            continue
        label = BACKEND_STYLE.get(b, {}).get("label", b)

        # Detect wave boundaries: wall time jumps > 50% between adjacent B values
        print(f"  {label}:")
        for i in range(1, len(pts)):
            r_prev, r_curr = pts[i - 1], pts[i]
            time_ratio = (
                r_curr.time_ms_median / r_prev.time_ms_median if r_prev.time_ms_median > 0 else 0
            )
            if time_ratio > 1.5:
                print(
                    f"    Wave boundary: B={r_prev.B}→{r_curr.B} "
                    f"(time {r_prev.time_ms_median:.0f}ms → {r_curr.time_ms_median:.0f}ms, "
                    f"{time_ratio:.2f}x jump)"
                )

    # OOM boundaries
    oom = {
        b: [r for r in results if r.backend == b and r.status in ("oom", "not_tested")]
        for b in backends
    }
    oom_backends = {b: v for b, v in oom.items() if v}
    if oom_backends:
        print("\n  OOM BOUNDARIES")
        print("  " + "-" * 60)
        for b, oom_pts in oom_backends.items():
            label = BACKEND_STYLE.get(b, {}).get("label", b)
            first_oom_B = min(r.B for r in oom_pts)
            print(f"  {label:30s}: OOM/skipped at B={first_oom_B}")


# ======================================================================
# Plotting
# ======================================================================


def generate_figures(
    results: list[BenchmarkResult],
    outdir: Path,
    backends: list[str],
    T: int,
    K: int,
    C: int,
    sm_count: int,
    linear_scan_ref: BenchmarkResult | None = None,
):
    """Generate publication figures for batch sweep."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available -- skipping figures")
        return

    ok = {
        b: sorted(
            [r for r in results if r.backend == b and r.status == "success"], key=lambda r: r.B
        )
        for b in backends
    }

    active = [b for b in backends if ok.get(b)]
    if not active:
        print("  No successful results -- skipping figures")
        return

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )

    # -- Fig 1: Throughput vs B ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for b in active:
        s = BACKEND_STYLE.get(b, {"color": "gray", "marker": "o", "ls": "-", "label": b})
        Bs = [r.B for r in ok[b]]
        throughput = [(r.B * T) / (r.time_ms_median / 1000) for r in ok[b]]
        ax.plot(
            Bs,
            throughput,
            marker=s["marker"],
            linestyle=s["ls"],
            color=s["color"],
            label=s["label"],
            markersize=5,
            linewidth=1.5,
        )
        # Annotate peak throughput
        if throughput:
            peak_idx = throughput.index(max(throughput))
            val = throughput[peak_idx]
            if val >= 1e6:
                label_text = f"{val / 1e6:.1f}M"
            elif val >= 1e3:
                label_text = f"{val / 1e3:.1f}K"
            else:
                label_text = f"{val:.0f}"
            ax.annotate(
                f"{label_text} pos/s",
                (Bs[peak_idx], val),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
                color=s["color"],
                fontweight="bold",
            )

    # Add linear scan reference point if available
    if linear_scan_ref and linear_scan_ref.status == "success":
        ls_tp = (linear_scan_ref.B * T) / (linear_scan_ref.time_ms_median / 1000)
        ax.scatter(
            [linear_scan_ref.B],
            [ls_tp],
            marker="x",
            color="#dc2626",
            s=100,
            zorder=10,
            linewidths=2,
            label=f"Streaming scan (OOMs at B≥{linear_scan_ref.B + 1})",
        )
        ax.annotate(
            f"{_format_throughput(ls_tp)}\n(OOMs at B≥{linear_scan_ref.B + 1})",
            (linear_scan_ref.B, ls_tp),
            textcoords="offset points",
            xytext=(12, -15),
            fontsize=8,
            color="#dc2626",
            fontstyle="italic",
        )

    # SM count vertical line
    ax.axvline(
        sm_count,
        color="#9ca3af",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"SM count ({sm_count})",
    )
    ax.annotate(
        f"{sm_count} SMs",
        (sm_count, ax.get_ylim()[0]),
        textcoords="offset points",
        xytext=(5, 10),
        fontsize=8,
        color="#9ca3af",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Throughput (positions / second)")
    ax.set_title(f"Throughput vs Batch Size  (T={T:,}, K={K}, C={C})")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fig.savefig(outdir / "fig_throughput_vs_B.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_throughput_vs_B.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_throughput_vs_B.pdf/png")

    # -- Fig 2: Memory vs B -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for b in active:
        s = BACKEND_STYLE.get(b, {"color": "gray", "marker": "o", "ls": "-", "label": b})
        Bs = [r.B for r in ok[b]]
        mems_mb = [r.peak_allocated_gb * 1024 for r in ok[b]]
        ax.plot(
            Bs,
            mems_mb,
            marker=s["marker"],
            linestyle=s["ls"],
            color=s["color"],
            label=s["label"],
            markersize=5,
            linewidth=1.5,
        )
        # Annotate last point
        if mems_mb:
            raw_mb = mems_mb[-1]
            if raw_mb >= 1024:
                label_text = f"{raw_mb / 1024:.2f} GB"
            else:
                label_text = f"{raw_mb:.0f} MB"
            ax.annotate(
                label_text,
                (Bs[-1], mems_mb[-1]),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=8,
                color=s["color"],
                fontweight="bold",
            )

    # Add linear scan reference point with OOM annotation
    if linear_scan_ref and linear_scan_ref.status == "success":
        ls_mem_mb = linear_scan_ref.peak_allocated_gb * 1024
        ax.scatter(
            [linear_scan_ref.B],
            [ls_mem_mb],
            marker="x",
            color="#dc2626",
            s=100,
            zorder=10,
            linewidths=2,
            label=f"Streaming scan B={linear_scan_ref.B}",
        )
        # Show projected OOM region
        max_gpu_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        ax.axhline(
            max_gpu_mb,
            color="#dc2626",
            linestyle=":",
            linewidth=1,
            alpha=0.4,
        )
        ax.annotate(
            f"GPU limit ({max_gpu_mb / 1024:.0f} GB)",
            (ax.get_xlim()[0], max_gpu_mb),
            textcoords="offset points",
            xytext=(5, -12),
            fontsize=8,
            color="#dc2626",
            alpha=0.6,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Peak allocated GPU memory (MB)")
    ax.set_title(f"Memory Scaling vs Batch Size  (T={T:,}, K={K}, C={C})")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fig.savefig(outdir / "fig_memory_vs_B.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_memory_vs_B.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_memory_vs_B.pdf/png")

    # -- Fig 3: Wall time vs B ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for b in active:
        s = BACKEND_STYLE.get(b, {"color": "gray", "marker": "o", "ls": "-", "label": b})
        Bs = [r.B for r in ok[b]]
        times = [r.time_ms_median for r in ok[b]]
        lo = [r.time_ms_iqr_low for r in ok[b]]
        hi = [r.time_ms_iqr_high for r in ok[b]]
        ax.plot(
            Bs,
            times,
            marker=s["marker"],
            linestyle=s["ls"],
            color=s["color"],
            label=s["label"],
            markersize=5,
            linewidth=1.5,
        )
        ax.fill_between(Bs, lo, hi, color=s["color"], alpha=0.12)

    # SM count vertical line
    ax.axvline(
        sm_count,
        color="#9ca3af",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"SM count ({sm_count})",
    )

    # Annotation for the wave structure
    ax.annotate(
        "Single wave\n(all programs concurrent)",
        xy=(sm_count * 0.5, ax.get_ylim()[0]),
        xytext=(sm_count * 0.15, ax.get_ylim()[1] * 0.3),
        fontsize=8,
        color="#6b7280",
        arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8},
        ha="center",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Wall time (ms)")
    ax.set_title(f"Wall Time vs Batch Size  (T={T:,}, K={K}, C={C})")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fig.savefig(outdir / "fig_time_vs_B.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_time_vs_B.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_time_vs_B.pdf/png")

    # -- Fig 4: Per-element time vs B ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for b in active:
        s = BACKEND_STYLE.get(b, {"color": "gray", "marker": "o", "ls": "-", "label": b})
        Bs = [r.B for r in ok[b]]
        per_elem = [r.time_ms_median / r.B for r in ok[b]]
        ax.plot(
            Bs,
            per_elem,
            marker=s["marker"],
            linestyle=s["ls"],
            color=s["color"],
            label=s["label"],
            markersize=5,
            linewidth=1.5,
        )
        # Annotate first and last
        if per_elem:
            for idx, _lbl in [(0, "B=1"), (-1, f"B={Bs[-1]}")]:
                val = per_elem[idx]
                if val >= 1000:
                    text = f"{val/1000:.1f}s"
                else:
                    text = f"{val:.0f}ms"
                ax.annotate(
                    f"{text}/elem",
                    (Bs[idx], per_elem[idx]),
                    textcoords="offset points",
                    xytext=(8, 5 if idx == 0 else -10),
                    fontsize=8,
                    color=s["color"],
                    fontweight="bold",
                )

    # SM count vertical line
    ax.axvline(
        sm_count,
        color="#9ca3af",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"SM count ({sm_count})",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Time per element (ms)")
    ax.set_title(f"Per-Element Amortization  (T={T:,}, K={K}, C={C})")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fig.savefig(outdir / "fig_per_element_vs_B.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_per_element_vs_B.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_per_element_vs_B.pdf/png")


# ======================================================================
# Linear scan reference
# ======================================================================


def run_linear_scan_reference(
    T: int,
    K: int,
    C: int,
    device: torch.device,
    repeats: int,
    phase: str,
    semiring: str,
    max_memory_gb: float,
) -> BenchmarkResult | None:
    """Run linear scan streaming at B=2 for OOM reference point.

    Returns the result if successful, None if OOM or skipped.
    """
    B_ref = 2
    backend = "linear_scan_streaming"

    # Check predicted memory
    predicted_gb = B_ref * T * K * C * C * 4 / (1024**3)
    if predicted_gb > max_memory_gb:
        print(
            f"\n  Linear scan reference: skip (predicted {predicted_gb:.1f}GB > {max_memory_gb:.1f}GB)"
        )
        return None

    print(f"\n  LINEAR SCAN REFERENCE (B={B_ref}, for OOM comparison)")
    print("  " + "-" * 80)

    gc.collect()
    torch.cuda.empty_cache()

    r = run_single_benchmark(
        T,
        K,
        C,
        B_ref,
        backend,
        device,
        repeats,
        semiring_name=semiring,
        phase=phase,
    )

    if r.status == "success":
        tp = (B_ref * T) / (r.time_ms_median / 1000) if r.time_ms_median > 0 else 0
        tp_str = _format_throughput(tp)
        print(
            f"  B={B_ref}: {r.time_ms_median:.1f}ms, {r.peak_allocated_gb:.3f}GB allocated, {tp_str}"
        )

        # Estimate max B before OOM
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        mem_per_elem_gb = r.peak_allocated_gb / B_ref
        max_B_estimate = int(gpu_gb / mem_per_elem_gb) if mem_per_elem_gb > 0 else 0
        print(
            f"  Estimated memory per element: {mem_per_elem_gb * 1024:.0f} MB "
            f"→ OOM at B≈{max_B_estimate} on {gpu_gb:.0f}GB GPU"
        )
    elif r.status == "oom":
        print(f"  B={B_ref}: OOM (cannot run even at B={B_ref})")
    else:
        print(f"  B={B_ref}: {r.status}")

    return r if r.status == "success" else None


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch-sweep: throughput and memory vs batch size (GPU utilization scaling)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--T", type=int, default=100000, help="Sequence length (fixed). Default: 100000"
    )
    parser.add_argument("--K", type=int, default=200, help="Max duration (fixed). Default: 200")
    parser.add_argument("--C", type=int, default=6, help="Num classes (fixed). Default: 6")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations. Default: 5")
    parser.add_argument(
        "--B",
        type=str,
        default=None,
        help="Comma-separated batch sizes. Default: 1,2,4,...,256",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="forward",
        choices=["forward", "backward", "both"],
        help="Which phase to time. Default: forward",
    )
    parser.add_argument(
        "--semiring",
        type=str,
        default="Log",
        help="Semiring to use. Default: Log",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=40.0,
        help="Skip configs predicted to exceed this memory. Default: 40",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        choices=list(REGIMES.keys()),
        help="Benchmark regime preset.",
    )
    parser.add_argument(
        "--include-linear-scan",
        action="store_true",
        help="Run linear scan streaming at B=2 for OOM reference point",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: fewer B values, fewer repeats"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    parser.add_argument(
        "--outdir", type=Path, default=Path("results/batch_sweep"), help="Output directory"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1

    device = torch.device("cuda")
    sm_count = _get_sm_count(device)

    # Apply regime preset (explicit CLI flags override)
    regime_B = None
    if args.regime:
        regime = REGIMES[args.regime]
        if "--T" not in sys.argv:
            args.T = regime["T"]
        if "--K" not in sys.argv:
            args.K = regime["K"]
        if "--C" not in sys.argv:
            args.C = regime["C"]
        regime_B = regime["B"]
        args.outdir = args.outdir / args.regime

    # Parse B values (priority: --B > --quick > regime > default)
    if args.B:
        B_values = sorted([int(x.strip()) for x in args.B.split(",") if x.strip()])
    elif args.quick:
        B_values = QUICK_B_VALUES
        args.repeats = 3
    elif regime_B:
        B_values = regime_B
    else:
        B_values = DEFAULT_B_VALUES

    T, K, C = args.T, args.K, args.C
    backends = DEFAULT_BACKENDS
    args.outdir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print("=" * 72)
    print("  BATCH-SWEEP BENCHMARK: GPU Utilization Scaling")
    print("=" * 72)
    if args.regime:
        print(f"  Regime: {args.regime} — {REGIMES[args.regime]['description']}")
    print(f"  GPU: {gpu_name} ({total_gb:.0f} GB, {sm_count} SMs)")
    print(f"  Fixed: T={T:,}, K={K}, C={C}")
    print(f"  B values: {B_values}")
    print(f"  Backends: {backends}")
    print(f"  Phase: {args.phase}, Semiring: {args.semiring}")
    print(f"  Repeats: {args.repeats}")
    print(f"  Triton available: {HAS_TRITON}")
    print("=" * 72)

    # -- Optional: linear scan reference --------------------------------
    linear_scan_ref = None
    if args.include_linear_scan:
        linear_scan_ref = run_linear_scan_reference(
            T,
            K,
            C,
            device,
            args.repeats,
            args.phase,
            args.semiring,
            args.max_memory_gb,
        )

    # -- Run sweep -----------------------------------------------------
    results = run_sweep(
        B_values=B_values,
        T=T,
        K=K,
        C=C,
        backends=backends,
        device=device,
        repeats=args.repeats,
        phase=args.phase,
        semiring=args.semiring,
        max_memory_gb=args.max_memory_gb,
    )

    # -- Summary -------------------------------------------------------
    print_sweep_summary(results, backends, T, K, C, sm_count)

    # -- Save JSON -----------------------------------------------------
    json_path = args.outdir / "batch_sweep_results.json"
    json_data = {
        "config": {
            "regime": args.regime,
            "T": T,
            "K": K,
            "C": C,
            "phase": args.phase,
            "semiring": args.semiring,
            "gpu": gpu_name,
            "sm_count": sm_count,
            "repeats": args.repeats,
        },
        "results": [asdict(r) for r in results],
    }
    if linear_scan_ref:
        json_data["linear_scan_reference"] = asdict(linear_scan_ref)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    # -- Figures -------------------------------------------------------
    if not args.no_plot:
        print("\n  Generating figures...")
        generate_figures(results, args.outdir, backends, T, K, C, sm_count, linear_scan_ref)

    print("\n" + "=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
