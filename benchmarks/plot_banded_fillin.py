#!/usr/bin/env python3
"""Generate fill-in-under-composition figures for the banded-is-not-viable supplement.

Produces three figure sets supporting the structural claims in
``docs/manuscript/banded_is_not_viable.md``:

1. **Boolean reachability fill-in** (Lemma 2, Section 3).
   Matrix model: boundaries 0..T, size (T+1)x(T+1), B[i,j]=1 iff 1<=j-i<=K.
   Theory: bw(B^m) = min(T, mK).

2. **Tree-level adjacency patterns** (Proposition 1, Sections 2 & 4).
   Matrix model: duration-label states, size (K-1)*C, constraint k1+k2<=S.
   Shows anti-diagonal triangular sparsity and effect of permutations.

3. **Bandwidth ratio summary** (Section 4, empirical bandwidth ratios).
   Best achievable bandwidth ratio vs normalized span S/K across configs.

Recommended figures for the supplement
--------------------------------------
The full grid produces 16 figures (redundant across (K,C) configs).
For the supplement, use ``--representative`` to emit only the 5 essential
figures:

  fig1a  Spy panel (B, B^2, B^4, B^8)       — visual intuition for fill-in
  fig1b  Bandwidth growth curve               — quantitative match to Lemma 2
  fig2a  Tree adjacency at S=K/2, K, 2(K-1)  — anti-diagonal triangular pattern
  fig2b  Permutation comparison at S=K        — identity/snake/RCM all near-dense
  fig3   Summary ratio vs S/K                 — universal collapse across configs

Representative config for fig2a/2b: K=16, C=3 (large enough K to show
triangular structure clearly; small enough C that block structure is visible).
Override with ``--repr-K`` and ``--repr-C``.

Example::

    # Full grid (16 figures, useful for exploration):
    python3 benchmarks/plot_banded_fillin.py

    # Supplement-ready (5 figures + caption guidance):
    python3 benchmarks/plot_banded_fillin.py --representative
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    print("ERROR: scipy is required for this script (boolean sparse matrix powers).")
    print("Install with: pip install scipy")
    sys.exit(1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from flash_semicrf import (  # noqa: E402
    SemiMarkov,
    apply_permutation,
    measure_effective_bandwidth,
    rcm_ordering_from_adjacency,
    snake_ordering,
)
from flash_semicrf.semirings import LogSemiring  # noqa: E402

# ---------------------------------------------------------------------------
# Style constants (consistent with benchmarks/plot_figures.py)
# ---------------------------------------------------------------------------
PERM_COLORS = {
    "identity": "#1f77b4",
    "snake": "#ff7f0e",
    "rcm": "#2ca02c",
}
THEORY_COLOR = "#d62728"
BINARY_CMAP = ListedColormap(["white", "#1f3d73"])


# ---------------------------------------------------------------------------
# Caption templates — printed in --representative mode to help write legends.
# Each key matches a figure function name.
# ---------------------------------------------------------------------------
CAPTIONS = {
    "fig1a": (
        "Sparsity pattern of the boolean boundary reachability matrix $B$ and "
        "its powers $B^m$ under boolean (OR/AND) multiplication, for $T={T}$ "
        "boundaries and maximum step size $K={K}$. Entry $(i,j)$ is dark iff "
        "boundary $j$ is reachable from $i$ in exactly $m$ segments. "
        "Bandwidth doubles with each power, filling the upper triangle by "
        "$m = T/K = {ToverK}$. This illustrates Lemma~2: $\\mathrm{{bw}}(B^m) "
        "= \\min(T, mK)$."
    ),
    "fig1b": (
        "Effective bandwidth of $B^m$ (blue circles) versus the theoretical "
        "prediction $\\min(T, mK)$ (red crosses) from Lemma~2, for $T={T}$ "
        "and $K={K}$. The measured bandwidth matches the theory exactly, "
        "confirming linear growth at rate $K$ per composed step until "
        "saturation at the dense width $T$. In a balanced binary tree, $m$ "
        "doubles per level, so saturation occurs after "
        "$O(\\log_2(T/K))$ levels."
    ),
    "fig2a": (
        "Adjacency pattern of the $(K\\!-\\!1) \\cdot C = {n}$ dimensional "
        "state space at a single tree node, for $K={K}$, $C={C}$. Each panel "
        "shows a different span (the number of sequence positions covered by "
        "the node): {spans}. Dark entries indicate feasible duration pairs "
        "satisfying $k_1 + k_2 \\le \\text{{span}}$. The pattern is "
        "anti-diagonal triangular (not diagonal-banded). At span $= K$, "
        "bandwidth is already {bw_at_K}/{denom} = {ratio_at_K:.2f} of the "
        "dense width, and at span $= 2(K\\!-\\!1)$ the matrix is fully dense."
    ),
    "fig2b": (
        "Effect of state-space permutations on the adjacency bandwidth at "
        "span $= K = {K}$ for $C = {C}$ ($n = {n}$). Identity ordering "
        "(bw = {bw_id}/{denom}), snake interleaving (bw = {bw_snake}/{denom}), "
        "and Reverse Cuthill--McKee (bw = {bw_rcm}/{denom}). No permutation "
        "reduces bandwidth below {best_ratio:.0%} of the dense width, "
        "confirming the clique lower bound of Proposition~1."
    ),
    "fig3": (
        "Best achievable bandwidth ratio (across identity, snake, and RCM "
        "orderings) versus normalized span (span/$K$), for several $(K, C)$ "
        "configurations. All curves collapse onto essentially the same shape: "
        "ratios rise steeply and exceed 0.95 by span/$K \\approx 0.8$, "
        "reaching 1.0 (fully dense) by span/$K \\approx 1.5$. Since higher "
        "tree levels have larger spans that dominate compute and memory, "
        "banded storage offers negligible benefit where it matters most."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_fig(fig: plt.Figure, stem: Path, dpi: int = 300, fmt: str = "both") -> None:
    if fmt in ("pdf", "both"):
        fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    if fmt in ("png", "both"):
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem.name}")


def build_boolean_adjacency(T: int, K: int) -> sp.csr_matrix:
    """Build sparse boolean adjacency B where B[i,j]=1 iff 1 <= j-i <= K.

    Boundary indices 0..T, matrix size (T+1)x(T+1).
    """
    n = T + 1
    rows, cols = [], []
    for i in range(n):
        for d in range(1, K + 1):
            j = i + d
            if j < n:
                rows.append(i)
                cols.append(j)
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.int8)
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix((data, (np.array(rows), np.array(cols))), shape=(n, n))


def boolean_matmul(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    """Boolean (OR/AND) sparse matrix multiply: clamp entries to {0, 1}."""
    C = A @ B
    if C.nnz:
        C.data[:] = 1
        C.eliminate_zeros()
    return C


def sparse_bandwidth(M: sp.spmatrix) -> int:
    """Max |i - j| over nonzero entries of a scipy sparse matrix."""
    if M.nnz == 0:
        return 0
    coo = M.tocoo()
    return int(np.max(np.abs(coo.row - coo.col)))


def upper_triangle_density(M: sp.spmatrix) -> float:
    """Fraction of strict upper-triangle entries that are nonzero."""
    n = M.shape[0]
    possible = n * (n - 1) // 2
    if possible == 0:
        return 0.0
    coo = M.tocoo()
    cnt = int(np.sum(coo.col > coo.row))
    return cnt / possible


def compute_bw_ratios(struct: SemiMarkov, K: int, C: int, span: int) -> dict[str, Optional[float]]:
    """Compute bandwidth ratios under identity, snake, and RCM orderings.

    Denominator is (K-1)*C - 1  (max bandwidth of a dense matrix of that size).
    Returns dict with keys: identity, snake, rcm (None if unavailable), best.
    Also includes raw bandwidths as identity_bw, snake_bw, rcm_bw.
    """
    K_1 = K - 1
    n = K_1 * C
    denom = n - 1
    if denom <= 0:
        return {
            "identity": 1.0,
            "snake": 1.0,
            "rcm": 1.0,
            "best": 1.0,
            "identity_bw": 0,
            "snake_bw": 0,
            "rcm_bw": 0,
            "denom": 0,
        }

    adj = struct._build_adjacency(span, K, C, device="cpu")  # noqa: SLF001
    adj_float = adj.float()

    # Identity
    bw_id = measure_effective_bandwidth(adj_float, fill_value=0.0)
    ratio_id = bw_id / denom

    # Snake
    perm_snake = snake_ordering(K_1, C)
    adj_snake = apply_permutation(adj_float.unsqueeze(0), perm_snake).squeeze(0)
    bw_snake = measure_effective_bandwidth(adj_snake, fill_value=0.0)
    ratio_snake = bw_snake / denom

    # RCM
    perm_rcm, used_scipy = rcm_ordering_from_adjacency(adj_float)
    if used_scipy:
        adj_rcm = apply_permutation(adj_float.unsqueeze(0), perm_rcm).squeeze(0)
        bw_rcm = measure_effective_bandwidth(adj_rcm, fill_value=0.0)
        ratio_rcm = bw_rcm / denom
    else:
        bw_rcm = None
        ratio_rcm = None

    candidates = [ratio_id, ratio_snake]
    if ratio_rcm is not None:
        candidates.append(ratio_rcm)
    best = min(candidates)

    return {
        "identity": ratio_id,
        "snake": ratio_snake,
        "rcm": ratio_rcm,
        "best": best,
        "identity_bw": bw_id,
        "snake_bw": bw_snake,
        "rcm_bw": bw_rcm,
        "denom": denom,
    }


# ---------------------------------------------------------------------------
# Figure Set 1: Boolean reachability fill-in
# ---------------------------------------------------------------------------
def plot_spy_panel(
    T: int,
    K: int,
    powers_dict: dict[int, sp.csr_matrix],
    output_dir: Path,
    dpi: int,
    fmt: str,
    print_caption: bool = False,
) -> None:
    """fig1a — Spy panel of B and its boolean powers.

    Visual intuition for Lemma 2.  Each panel shows the sparsity pattern
    of B^m over boundary indices 0..T.  As m increases, the band widens
    at rate K per power and fills the entire upper triangle by m = T/K.
    """
    display_powers = [m for m in [1, 2, 4, 8] if m in powers_dict]
    ncols = len(display_powers)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.2))
    if ncols == 1:
        axes = [axes]

    for ax, m in zip(axes, display_powers, strict=True):
        M = powers_dict[m]
        dense = M.toarray().astype(float)
        bw = sparse_bandwidth(M)
        ax.imshow(dense, cmap=BINARY_CMAP, interpolation="nearest", aspect="equal", vmin=0, vmax=1)
        label = "$B$" if m == 1 else f"$B^{{{m}}}$"
        ax.set_title(f"{label}\nbw = {bw}", fontsize=10)
        ax.set_xlabel("boundary $j$", fontsize=8)
        ax.set_ylabel("boundary $i$", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"Boolean reachability fill-in ($T={T}$, $K={K}$)", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir / f"fig1a_spy_T{T}_K{K}", dpi, fmt)

    if print_caption:
        caption = CAPTIONS["fig1a"].format(T=T, K=K, ToverK=T // K)
        print(f"\n    Caption (fig1a):\n    {caption}\n")


def plot_bandwidth_growth(
    T: int,
    K: int,
    powers_dict: dict[int, sp.csr_matrix],
    output_dir: Path,
    dpi: int,
    fmt: str,
    print_caption: bool = False,
) -> None:
    """fig1b — Measured bandwidth vs Lemma 2 prediction.

    Plots bw(B^m) alongside the closed-form min(T, mK).  The two curves
    overlap exactly, validating the theory.  A horizontal reference line
    marks the dense width T.
    """
    ms = sorted(powers_dict.keys())
    bws = [sparse_bandwidth(powers_dict[m]) for m in ms]
    ref = [min(T, m * K) for m in ms]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ms, bws, "o-", color=PERM_COLORS["identity"], label="measured bw($B^m$)")
    ax.plot(ms, ref, "x--", color=THEORY_COLOR, label="min($T$, $mK$)")
    ax.axhline(T, color="gray", linestyle=":", alpha=0.5, label=f"dense width $T={T}$")
    ax.set_xlabel("power $m$ (number of composed segments)")
    ax.set_ylabel("effective bandwidth")
    ax.set_title(f"Bandwidth growth under composition ($T={T}$, $K={K}$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir / f"fig1b_bandwidth_T{T}_K{K}", dpi, fmt)

    if print_caption:
        caption = CAPTIONS["fig1b"].format(T=T, K=K)
        print(f"\n    Caption (fig1b):\n    {caption}\n")


def plot_density_growth(
    T: int,
    K: int,
    powers_dict: dict[int, sp.csr_matrix],
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> None:
    """fig1c — Upper-triangle density vs boolean power.

    Supplementary to fig1a/1b.  Shows the fraction of reachable (i,j) pairs
    in the strict upper triangle as a function of composed segments m.
    Not recommended for the main supplement (redundant with spy panel).
    """
    ms = sorted(powers_dict.keys())
    dens = [upper_triangle_density(powers_dict[m]) for m in ms]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ms, dens, "o-", color=PERM_COLORS["identity"])
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="fully dense")
    ax.set_xlabel("power $m$")
    ax.set_ylabel("upper-triangle density")
    ax.set_title(f"Fill-in: density of reachable pairs ($T={T}$, $K={K}$)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir / f"fig1c_density_T{T}_K{K}", dpi, fmt)


# ---------------------------------------------------------------------------
# Figure Set 2: Tree-level adjacency patterns
# ---------------------------------------------------------------------------
def plot_tree_adjacency(
    struct: SemiMarkov,
    K: int,
    C: int,
    output_dir: Path,
    dpi: int,
    fmt: str,
    print_caption: bool = False,
) -> None:
    """fig2a — Adjacency at three representative spans for one (K, C) config.

    Demonstrates Proposition 1: the feasibility constraint k1 + k2 <= S
    produces an anti-diagonal triangular pattern (not diagonal-banded).
    Three panels show the progression:
      - S = max(1, K//2): sub-K span, moderate sparsity.
      - S = K:            critical transition, nearly dense.
      - S = 2*(K-1):      fully saturated, structurally dense.
    """
    spans = [max(1, K // 2), K, 2 * (K - 1)]
    n = (K - 1) * C
    denom = n - 1

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    bw_at_K = None
    ratio_at_K = None
    for ax, S in zip(axes, spans, strict=True):
        adj = struct._build_adjacency(S, K, C, device="cpu")  # noqa: SLF001
        bw = measure_effective_bandwidth(adj.float(), fill_value=0.0)
        ratio = bw / denom if denom > 0 else 1.0

        if S == K:
            bw_at_K = bw
            ratio_at_K = ratio

        ax.imshow(
            adj.float().numpy(),
            cmap=BINARY_CMAP,
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"span = {S}\nbw = {bw}/{denom} ({ratio:.2f})", fontsize=9)
        ax.set_xlabel("state index", fontsize=8)
        ax.set_ylabel("state index", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Tree-node adjacency ($K={K}$, $C={C}$, $n={n}$)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, output_dir / f"fig2a_adjacency_K{K}_C{C}", dpi, fmt)

    if print_caption:
        span_str = ", ".join(str(s) for s in spans)
        caption = CAPTIONS["fig2a"].format(
            K=K,
            C=C,
            n=n,
            spans=span_str,
            denom=denom,
            bw_at_K=bw_at_K,
            ratio_at_K=ratio_at_K,
        )
        print(f"\n    Caption (fig2a):\n    {caption}\n")


def plot_permutation_compare(
    struct: SemiMarkov,
    K: int,
    C: int,
    output_dir: Path,
    dpi: int,
    fmt: str,
    print_caption: bool = False,
) -> None:
    """fig2b — Permutation comparison at S=K.

    Demonstrates Section 4: even the best permutation (identity, snake
    interleaving, or Reverse Cuthill-McKee) cannot reduce bandwidth to a
    useful fraction of the dense width when S >= K.  The clique lower bound
    of Proposition 1 forces bandwidth >= C * floor(S/2) - 1.
    """
    S = K
    K_1 = K - 1
    n = K_1 * C
    denom = n - 1
    adj = struct._build_adjacency(S, K, C, device="cpu")  # noqa: SLF001
    adj_float = adj.float()

    # Build orderings
    orderings = []

    # Identity
    bw_id = measure_effective_bandwidth(adj_float, fill_value=0.0)
    orderings.append(("Identity", adj_float.numpy(), bw_id))

    # Snake
    perm_snake = snake_ordering(K_1, C)
    adj_snake = apply_permutation(adj_float.unsqueeze(0), perm_snake).squeeze(0)
    bw_snake = measure_effective_bandwidth(adj_snake, fill_value=0.0)
    orderings.append(("Snake", adj_snake.numpy(), bw_snake))

    # RCM
    perm_rcm, used_scipy = rcm_ordering_from_adjacency(adj_float)
    bw_rcm = None
    if used_scipy:
        adj_rcm = apply_permutation(adj_float.unsqueeze(0), perm_rcm).squeeze(0)
        bw_rcm = measure_effective_bandwidth(adj_rcm, fill_value=0.0)
        orderings.append(("RCM", adj_rcm.numpy(), bw_rcm))

    ncols = len(orderings)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.5))
    if ncols == 1:
        axes = [axes]

    for ax, (name, mat, bw) in zip(axes, orderings, strict=True):
        ratio = bw / denom if denom > 0 else 1.0
        color = PERM_COLORS.get(name.lower(), "#333333")
        ax.imshow(mat, cmap=BINARY_CMAP, interpolation="nearest", aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"{name}\nbw = {bw}/{denom} ({ratio:.2f})", fontsize=9, color=color)
        ax.set_xlabel("state index", fontsize=8)
        ax.set_ylabel("state index", fontsize=8)
        ax.tick_params(labelsize=7)

    if not used_scipy:
        fig.text(0.98, 0.02, "RCM: requires scipy", ha="right", fontsize=7, style="italic")

    fig.suptitle(
        f"Permutation comparison ($K={K}$, $C={C}$, span$={S}$)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, output_dir / f"fig2b_permutation_K{K}_C{C}_S{S}", dpi, fmt)

    if print_caption:
        best_bw = min(o[2] for o in orderings)
        best_ratio = best_bw / denom if denom > 0 else 1.0
        caption = CAPTIONS["fig2b"].format(
            K=K,
            C=C,
            n=n,
            denom=denom,
            bw_id=bw_id,
            bw_snake=bw_snake,
            bw_rcm=bw_rcm if bw_rcm is not None else "N/A",
            best_ratio=best_ratio,
        )
        print(f"\n    Caption (fig2b):\n    {caption}\n")


def plot_bw_vs_span(
    struct: SemiMarkov,
    K: int,
    C: int,
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> None:
    """fig2c — Bandwidth ratio vs span for identity, snake, RCM.

    Supplementary per-(K,C) detail.  Not recommended for the main supplement
    (subsumed by fig3 which overlays all configs on a normalized axis).
    """
    max_span = 3 * (K - 1)
    spans = list(range(1, max_span + 1))

    id_ratios, snake_ratios, rcm_ratios = [], [], []
    rcm_available = None

    for S in spans:
        ratios = compute_bw_ratios(struct, K, C, S)
        id_ratios.append(ratios["identity"])
        snake_ratios.append(ratios["snake"])
        if ratios["rcm"] is not None:
            rcm_ratios.append(ratios["rcm"])
            rcm_available = True
        elif rcm_available is None:
            rcm_available = False

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(spans, id_ratios, "-", color=PERM_COLORS["identity"], label="Identity", alpha=0.8)
    ax.plot(spans, snake_ratios, "-", color=PERM_COLORS["snake"], label="Snake", alpha=0.8)
    if rcm_available:
        ax.plot(spans, rcm_ratios, "-", color=PERM_COLORS["rcm"], label="RCM", alpha=0.8)
    else:
        ax.plot([], [], "-", color=PERM_COLORS["rcm"], label="RCM (unavailable)", alpha=0.3)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(K, color="gray", linestyle="--", alpha=0.4, label=f"span $= K = {K}$")
    ax.set_xlabel("span")
    ax.set_ylabel("bw / ($n - 1$)")
    ax.set_title(f"Bandwidth ratio vs span ($K={K}$, $C={C}$, $n={(K - 1) * C}$)")
    ax.set_ylim(-0.02, 1.1)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_fig(fig, output_dir / f"fig2c_bw_vs_span_K{K}_C{C}", dpi, fmt)


# ---------------------------------------------------------------------------
# Figure Set 3: Summary bandwidth ratio
# ---------------------------------------------------------------------------
def plot_bandwidth_summary(
    struct: SemiMarkov,
    K_values: list[int],
    C_values: list[int],
    output_dir: Path,
    dpi: int,
    fmt: str,
    print_caption: bool = False,
) -> None:
    """fig3 — Best bandwidth ratio vs S/K for all (K, C) configs overlaid.

    The key takeaway figure.  All configs collapse onto the same universal
    curve: bandwidth ratio rises steeply through S/K in [0, 1] and saturates
    at 1.0 (fully dense) by S/K ~ 1.5.  Since higher tree levels have
    S >> K, banded storage provides no benefit where it matters most.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    markers = ["o", "s", "D", "^", "v", "<", ">", "h"]
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    idx = 0
    for K in K_values:
        for C in C_values:
            max_span = 3 * (K - 1)
            spans = list(range(1, max_span + 1))
            s_over_k = [s / K for s in spans]
            best_ratios = []
            for S in spans:
                ratios = compute_bw_ratios(struct, K, C, S)
                best_ratios.append(ratios["best"])

            ax.plot(
                s_over_k,
                best_ratios,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                markersize=3,
                linewidth=1,
                label=f"$K={K}$, $C={C}$",
                alpha=0.8,
            )
            idx += 1

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.4, label="span$/K = 1$")
    ax.set_xlabel("span / $K$ (normalized)")
    ax.set_ylabel("best bw / ($n - 1$)")
    ax.set_title("Best bandwidth ratio vs normalized span")
    ax.set_ylim(-0.02, 1.1)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save_fig(fig, output_dir / "fig3_bandwidth_summary", dpi, fmt)

    if print_caption:
        caption = CAPTIONS["fig3"]
        print(f"\n    Caption (fig3):\n    {caption}\n")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/banded_fillin"),
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=32,
        help="Boundary count for boolean model (matrix is (T+1)x(T+1)).",
    )
    parser.add_argument(
        "--K-bool",
        type=int,
        default=4,
        help="Step size K for boolean reachability B.",
    )
    parser.add_argument(
        "--K-values",
        type=str,
        default="8,16",
        help="Comma-separated K values for tree adjacency analysis.",
    )
    parser.add_argument(
        "--C-values",
        type=str,
        default="3,6",
        help="Comma-separated C values for tree adjacency analysis.",
    )
    parser.add_argument(
        "--max-power",
        type=int,
        default=None,
        help="Max boolean power m (default: max(1, T // K_bool)).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["pdf", "png", "both"],
        help="Output format.",
    )
    parser.add_argument(
        "--representative",
        action="store_true",
        help="Emit only the 5 recommended supplement figures (fig1a, fig1b, "
        "fig2a, fig2b, fig3) with caption guidance printed to stdout.",
    )
    parser.add_argument(
        "--repr-K",
        type=int,
        default=16,
        help="K value for representative fig2a/2b (default: 16).",
    )
    parser.add_argument(
        "--repr-C",
        type=int,
        default=3,
        help="C value for representative fig2a/2b (default: 3).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    T = args.T
    K_bool = args.K_bool
    max_power = args.max_power or max(1, T // K_bool)
    K_values = parse_int_list(args.K_values)
    C_values = parse_int_list(args.C_values)
    representative = args.representative

    struct = SemiMarkov(LogSemiring)

    # === Figure Set 1: Boolean Reachability ===
    print("Figure Set 1: Boolean reachability fill-in")
    B = build_boolean_adjacency(T, K_bool)
    powers_dict: dict[int, sp.csr_matrix] = {1: B.copy()}
    current = B.copy()
    for m in range(2, max_power + 1):
        current = boolean_matmul(B, current)
        powers_dict[m] = current.copy()

    plot_spy_panel(
        T, K_bool, powers_dict, output_dir, args.dpi, args.format, print_caption=representative
    )
    plot_bandwidth_growth(
        T, K_bool, powers_dict, output_dir, args.dpi, args.format, print_caption=representative
    )
    if not representative:
        plot_density_growth(T, K_bool, powers_dict, output_dir, args.dpi, args.format)

    # === Figure Set 2: Tree-Level Adjacency ===
    print("Figure Set 2: Tree-level adjacency patterns")
    if representative:
        # Single representative config
        K_rep, C_rep = args.repr_K, args.repr_C
        print(f"  Representative config: K={K_rep}, C={C_rep}")
        plot_tree_adjacency(
            struct, K_rep, C_rep, output_dir, args.dpi, args.format, print_caption=True
        )
        plot_permutation_compare(
            struct, K_rep, C_rep, output_dir, args.dpi, args.format, print_caption=True
        )
    else:
        for K in K_values:
            for C in C_values:
                print(f"  K={K}, C={C}")
                plot_tree_adjacency(struct, K, C, output_dir, args.dpi, args.format)
                plot_permutation_compare(struct, K, C, output_dir, args.dpi, args.format)
                plot_bw_vs_span(struct, K, C, output_dir, args.dpi, args.format)

    # === Figure Set 3: Summary ===
    print("Figure Set 3: Summary bandwidth ratio")
    plot_bandwidth_summary(
        struct, K_values, C_values, output_dir, args.dpi, args.format, print_caption=representative
    )

    n_figs = 5 if representative else 3 + 3 * len(K_values) * len(C_values) + 1
    print(f"\n{n_figs} figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
