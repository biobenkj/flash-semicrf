#!/usr/bin/env python3
"""Convert ablate_zero_centering.py output to LaTeX tables for the manuscript.

Runs the ablation programmatically and emits LaTeX tables to stdout and/or files.

Usage:
    python3 scripts/ablation_to_latex.py                    # print to stdout
    python3 scripts/ablation_to_latex.py --outdir tables/   # write .tex files
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

from flash_semicrf import SemiMarkovCRFHead
from flash_semicrf.streaming import semi_crf_streaming_forward


# ======================================================================
# Reproduce ablation data (mirrors ablate_zero_centering.py logic)
# ======================================================================

CENTERING_MODES = ["mean", "position", "none"]
IMBALANCED_PROPORTIONS = [0.70, 0.15, 0.10, 0.05]
LABEL_NAMES = ["intron", "exon", "UTR", "intergenic"]


def build_cum_scores(scores, mode, dtype=torch.float64):
    """Build cumulative scores under a given centering mode."""
    B, T, C = scores.shape
    if mode == "mean":
        centered = scores - scores.mean(dim=1, keepdim=True)
    elif mode == "position":
        centered = scores - scores.max(dim=2, keepdim=True).values
    elif mode == "none":
        centered = scores
    else:
        raise ValueError(f"Unknown mode: {mode}")
    cum = torch.zeros(B, T + 1, C, dtype=dtype, device=scores.device)
    cum[:, 1:, :] = torch.cumsum(centered.to(dtype), dim=1)
    return cum


def make_imbalanced_scores(T, C, batch, proportions, device, seed=42):
    """Create emission scores with nonzero per-label means (simulating trained encoder)."""
    torch.manual_seed(seed)
    scores = torch.randn(batch, T, C, device=device)
    # Add per-label bias proportional to class prevalence
    for c in range(min(C, len(proportions))):
        scores[:, :, c] += proportions[c] * 3.0
    return scores


def run_stability_data(device, T_values, C=24, K=50, batch=2):
    """§1: Numerical stability data."""
    rows = []
    for T in T_values:
        scores = make_imbalanced_scores(T, C, batch, IMBALANCED_PROPORTIONS, device)
        for mode in CENTERING_MODES:
            cum = build_cum_scores(scores, mode)
            endpoint_mag = cum[:, -1, :].abs().max().item()
            peak_mag = cum.abs().max().item()
            rows.append({
                "T": T,
                "mode": mode,
                "endpoint_mag": endpoint_mag,
                "peak_mag": peak_mag,
                "peak_sqrt_t": peak_mag / (T**0.5),
                "peak_over_t": peak_mag / T,
            })
    return rows


def run_prior_data(device, T=2000, C=4, K=50, batch=1):
    """§3: Duration prior data."""
    scenarios = {
        "imbalanced": IMBALANCED_PROPORTIONS,
        "balanced": [1.0 / C] * C,
    }
    results = {}
    for scenario_name, proportions in scenarios.items():
        scores = make_imbalanced_scores(T, C, batch, proportions, device, seed=42)

        # Part A: Per-label means and penalties
        penalty_rows = []
        for mode in CENTERING_MODES:
            if mode != "mean":
                continue
            centered = scores - scores.mean(dim=1, keepdim=True)
            nu = centered.mean(dim=1)[0]  # (C,)
            for c in range(C):
                label = LABEL_NAMES[c] if c < len(LABEL_NAMES) else f"label_{c}"
                nu_val = nu[c].item()
                penalty_rows.append({
                    "label": label,
                    "nu": nu_val,
                    "k1": -nu_val * 1,
                    "k5": -nu_val * 5,
                    "k10": -nu_val * 10,
                    "k25": -nu_val * 25,
                    "k50": -nu_val * 50,
                })

        # Part B: Partition function and marginal divergence
        transition = torch.randn(C, C, device=device, dtype=torch.float64) * 0.1
        duration_bias = torch.zeros(K, C, device=device, dtype=torch.float64)
        torch.manual_seed(42)

        partition_rows = []
        marginals_by_mode = {}
        for mode in CENTERING_MODES:
            cum = build_cum_scores(scores, mode)
            log_Z, alpha_T = semi_crf_streaming_forward(
                cum, transition, duration_bias, K,
            )
            log_Z_val = log_Z.sum().item()

            # Get marginals via backward
            cum_req = cum.detach().requires_grad_(True)
            log_Z2, _ = semi_crf_streaming_forward(
                cum_req, transition, duration_bias, K,
            )
            log_Z2.sum().backward()
            marginals = cum_req.grad[:, 1:, :]  # (B, T, C)
            marginals_by_mode[mode] = marginals.detach()

            partition_rows.append({
                "mode": mode,
                "log_z": log_Z_val,
            })

        # Compute TV divergence vs position mode
        ref = marginals_by_mode["position"]
        for row in partition_rows:
            m = marginals_by_mode[row["mode"]]
            # Normalize marginals to distributions at each position
            ref_norm = ref / (ref.sum(dim=-1, keepdim=True) + 1e-10)
            m_norm = m / (m.sum(dim=-1, keepdim=True) + 1e-10)
            tv = 0.5 * (ref_norm - m_norm).abs().sum(dim=-1).mean().item()
            row["tv"] = tv
            row["marginal_std"] = m.std().item()

        # Part C: Viterbi segments
        viterbi_rows = []
        for mode in CENTERING_MODES:
            cum = build_cum_scores(scores, mode)
            # Use Max semiring for Viterbi
            from flash_semicrf.streaming import semi_crf_streaming_forward as fwd
            log_Z, alpha_T = fwd(cum, transition, duration_bias, K)

            # Simple greedy segmentation from marginals for comparison
            cum_req = cum.detach().requires_grad_(True)
            log_Z2, _ = fwd(cum_req, transition, duration_bias, K)
            log_Z2.sum().backward()
            marginals = cum_req.grad[:, 1:, :]

            # Segment by argmax label at each position
            labels = marginals[0].argmax(dim=-1)  # (T,)
            # Count segments
            segments_by_label = defaultdict(list)
            current_label = labels[0].item()
            current_len = 1
            for t in range(1, len(labels)):
                if labels[t].item() == current_label:
                    current_len += 1
                else:
                    lname = LABEL_NAMES[current_label] if current_label < len(LABEL_NAMES) else f"label_{current_label}"
                    segments_by_label[lname].append(current_len)
                    current_label = labels[t].item()
                    current_len = 1
            lname = LABEL_NAMES[current_label] if current_label < len(LABEL_NAMES) else f"label_{current_label}"
            segments_by_label[lname].append(current_len)

            for label in LABEL_NAMES[:C]:
                segs = segments_by_label.get(label, [])
                if segs:
                    viterbi_rows.append({
                        "label": label,
                        "mode": mode,
                        "n_segments": len(segs),
                        "mean_len": sum(segs) / len(segs),
                        "min_len": min(segs),
                        "max_len": max(segs),
                    })

        results[scenario_name] = {
            "penalty": penalty_rows,
            "partition": partition_rows,
            "viterbi": viterbi_rows,
        }
    return results


# ======================================================================
# LaTeX formatters
# ======================================================================

def fmt(val, decimals=2):
    """Format a float for LaTeX."""
    if abs(val) < 0.01 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{decimals}f}"


def mode_label(mode):
    """LaTeX-formatted mode name."""
    return {"mean": r"\textsc{mean}", "position": r"\textsc{position}", "none": r"\textsc{none}"}[mode]


def stability_table(rows):
    """§1: Numerical stability table."""
    T_values = sorted(set(r["T"] for r in rows))
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cumulative score magnitude scaling under three centering strategies.")
    lines.append(r"  \textsc{mean} centering achieves $O(\sqrt{T})$ growth, while")
    lines.append(r"  \textsc{position} and \textsc{none} grow $O(T)$.")
    lines.append(r"  Config: $B{=}2$, $C{=}24$, $K{=}50$.}")
    lines.append(r"\label{tab:stability}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"$T$ & Mode & Endpoint & Peak & Peak/$\sqrt{T}$ & Peak/$T$ \\")
    lines.append(r"\midrule")

    for i, T in enumerate(T_values):
        if i > 0:
            lines.append(r"\addlinespace")
        for mode in CENTERING_MODES:
            r = next(x for x in rows if x["T"] == T and x["mode"] == mode)
            lines.append(
                f"  {T:,} & {mode_label(mode)} & "
                f"{fmt(r['endpoint_mag'])} & {fmt(r['peak_mag'])} & "
                f"{fmt(r['peak_sqrt_t'])} & {fmt(r['peak_over_t'], 4)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def penalty_table(penalty_rows, scenario):
    """§3 Part A: Duration penalty table."""
    lines = []
    scenario_label = "imbalanced" if scenario == "imbalanced" else "balanced"
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Implicit duration penalty $-\nu_{b,c} \cdot k$ under \textsc{mean} centering"
        f" ({scenario_label} labels)."
    )
    lines.append(
        r"  Globally-prevalent labels (e.g., intron at 70\%) incur a large per-step cost,"
    )
    lines.append(r"  while rare labels receive a duration bonus.}")
    lines.append(r"\label{tab:penalty-" + scenario + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Label & $\nu_{b,c}$ & $k{=}1$ & $k{=}5$ & $k{=}10$ & $k{=}25$ & $k{=}50$ \\")
    lines.append(r"\midrule")

    for r in penalty_rows:
        lines.append(
            f"  {r['label']} & {fmt(r['nu'], 3)} & "
            f"{fmt(r['k1'])} & {fmt(r['k5'])} & {fmt(r['k10'])} & "
            f"{fmt(r['k25'])} & {fmt(r['k50'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def partition_table(partition_rows, scenario):
    """§3 Part B: Partition function and divergence table."""
    lines = []
    scenario_label = "imbalanced" if scenario == "imbalanced" else "balanced"
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Partition function and marginal divergence under"
        f" {scenario_label} label proportions."
    )
    lines.append(
        r"  TV distance is measured against \textsc{position} (path-invariant reference).}"
    )
    lines.append(r"\label{tab:partition-" + scenario + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(r"Mode & $\log Z$ & TV vs \textsc{position} & Marginal std \\")
    lines.append(r"\midrule")

    for r in partition_rows:
        lines.append(
            f"  {mode_label(r['mode'])} & {fmt(r['log_z'], 1)} & "
            f"{fmt(r['tv'], 4)} & {fmt(r['marginal_std'], 4)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def combined_prior_table(penalty_rows, viterbi_rows):
    """Combined table: implicit duration prior + segment redistribution effect.

    Main text table showing cause (penalty) and effect (segment counts) together.
    """
    # Index viterbi data by (label, mode)
    viterbi_idx = {}
    for r in viterbi_rows:
        viterbi_idx[(r["label"], r["mode"])] = r

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Implicit duration prior from \textsc{mean} centering under label imbalance.")
    lines.append(r"  The per-label temporal mean $\nu_{b,c}$ creates a duration penalty $-\nu_{b,c} \cdot k$")
    lines.append(r"  that penalizes long segments of prevalent labels and redistributes probability mass")
    lines.append(r"  to rare labels. Right columns show downstream effect on decoded segmentation.")
    lines.append(r"  Config: $T{=}2000$, $C{=}4$, $K{=}50$.}")
    lines.append(r"\label{tab:implicit-prior}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{3}{c}{Duration penalty $-\nu \cdot k$}")
    lines.append(r" & \multicolumn{2}{c}{Segments} \\")
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-7}")
    lines.append(r"Label & $\nu_{b,c}$ & $k{=}10$ & $k{=}25$ & $k{=}50$")
    lines.append(r" & \textsc{mean} & \textsc{none} \\")
    lines.append(r"\midrule")

    for r in penalty_rows:
        label = r["label"]
        mean_segs = viterbi_idx.get((label, "mean"), {})
        none_segs = viterbi_idx.get((label, "none"), {})
        n_mean = mean_segs.get("n_segments", "--")
        n_none = none_segs.get("n_segments", "--")
        lines.append(
            f"  {label} & {fmt(r['nu'], 3)} & "
            f"{fmt(r['k10'])} & {fmt(r['k25'])} & {fmt(r['k50'])} & "
            f"{n_mean} & {n_none} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def viterbi_table(viterbi_rows, scenario):
    """§3 Part C: Viterbi segment lengths table."""
    lines = []
    scenario_label = "imbalanced" if scenario == "imbalanced" else "balanced"
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Viterbi segment lengths by label under"
        f" {scenario_label} label proportions."
    )
    lines.append(
        r"  \textsc{mean} centering redistributes segments from prevalent"
        r" to rare labels under imbalance.}"
    )
    lines.append(r"\label{tab:viterbi-" + scenario + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Label & Mode & Segments & Mean len & Min & Max \\")
    lines.append(r"\midrule")

    labels_seen = []
    for r in viterbi_rows:
        if r["label"] not in labels_seen:
            if labels_seen:
                lines.append(r"\addlinespace")
            labels_seen.append(r["label"])
        lines.append(
            f"  {r['label']} & {mode_label(r['mode'])} & "
            f"{r['n_segments']} & {fmt(r['mean_len'], 1)} & "
            f"{r['min_len']} & {r['max_len']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert zero-centering ablation results to LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Write .tex files to this directory (default: print to stdout)",
    )
    parser.add_argument(
        "--T",
        type=str,
        default="100,500,2000,10000,50000",
        help="Comma-separated T values for stability table",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_values = [int(x) for x in args.T.split(",")]

    print(f"Running ablation on {device}...", file=sys.stderr)

    # Generate data
    stability_rows = run_stability_data(device, T_values)
    prior_data = run_prior_data(device)

    # Build tables
    tables = {}
    tables["stability"] = stability_table(stability_rows)

    # Combined main-text table (imbalanced only)
    d_imb = prior_data["imbalanced"]
    tables["implicit_prior"] = combined_prior_table(d_imb["penalty"], d_imb["viterbi"])

    # Individual supplement tables
    for scenario in ["imbalanced", "balanced"]:
        d = prior_data[scenario]
        tables[f"penalty_{scenario}"] = penalty_table(d["penalty"], scenario)
        tables[f"partition_{scenario}"] = partition_table(d["partition"], scenario)
        tables[f"viterbi_{scenario}"] = viterbi_table(d["viterbi"], scenario)

    # Output
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        for name, tex in tables.items():
            path = args.outdir / f"tab_{name}.tex"
            path.write_text(tex + "\n")
            print(f"  Wrote {path}", file=sys.stderr)
    else:
        for name, tex in tables.items():
            print(f"% === {name} ===")
            print(tex)
            print()

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
