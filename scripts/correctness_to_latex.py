#!/usr/bin/env python3
"""Convert validate_correctness.py JSON output to LaTeX tables for the manuscript.

Reads the JSON exported by ``validate_correctness.py --json`` and emits
publication-ready LaTeX tables (booktabs format).

Usage:
    # Run validation and export JSON
    python3 scripts/validate_correctness.py --scale large --json results/correctness.json

    # Convert to LaTeX (stdout)
    python3 scripts/correctness_to_latex.py results/correctness.json

    # Write individual .tex files
    python3 scripts/correctness_to_latex.py results/correctness.json --outdir docs/manuscript/tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ======================================================================
# LaTeX formatters
# ======================================================================


def fmt_sci(val, sig=2):
    """Format a small float in LaTeX scientific notation."""
    if val == 0:
        return "0"
    import math

    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / (10**exp)
    if abs(exp) <= 1:
        return f"{val:.{sig + abs(exp)}f}"
    return rf"{mantissa:.{sig - 1}f}\times 10^{{{exp}}}"


def fmt(val, decimals=2):
    """Format a float for LaTeX."""
    if abs(val) < 0.01 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{decimals}f}"


def backend_label(name):
    """LaTeX-formatted backend name."""
    labels = {
        "pytorch": r"\textsc{pytorch}",
        "triton": r"\textsc{triton}",
        "dp_standard": r"\textsc{dp\_standard}",
    }
    return labels.get(name, rf"\textsc{{{name}}}")


# ======================================================================
# Table generators
# ======================================================================


def finite_diff_table(data):
    """Finite-difference gradient validation table."""
    config = data["config"]
    params = data["parameters"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Finite-difference gradient validation (\S\ref{sec:correctness}).")
    lines.append(
        r"  Central differences ($\varepsilon{=}" + fmt_sci(config["eps"]) + r"$)"
        r" vs.\ autograd for each learnable parameter."
    )
    lines.append(
        r"  Config: $B{=}" + str(config["batch"]) + r"$, "
        r"$T{=}" + str(config["T"]) + r"$, "
        r"$C{=}" + str(config["C"]) + r"$, "
        r"$K{=}" + str(config["K"]) + r"$.}"
    )
    lines.append(r"\label{tab:finite-diff}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Parameter & Elements & Cosine sim. & Norm.\ max err & Max $|\Delta|$ \\")
    lines.append(r"\midrule")

    for p in params:
        name_tex = p["name"].replace("_", r"\_")
        lines.append(
            f"  {name_tex} & {p['elements']:,} & "
            f"{p['cosine_similarity']:.6f} & "
            f"{fmt_sci(p['normalized_max_err'])} & "
            f"{fmt_sci(p['max_abs_diff'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def convergence_table(data):
    """Training convergence across backends table."""
    config = data["config"]
    backends = data["backends"]
    comparisons = data["comparisons"]

    # Index comparisons by name
    comp_idx = {c["name"]: c for c in comparisons}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Training convergence across backends (\S\ref{sec:correctness}).")
    lines.append(
        r"  All backends trained for "
        + str(config["epochs"])
        + r" epochs on identical synthetic data."
    )
    lines.append(
        r"  Config: $B{=}" + str(config["batch"]) + r"$, "
        r"$T{=}" + str(config["T"]) + r"$, "
        r"$C{=}" + str(config["C"]) + r"$, "
        r"$K{=}" + str(config["K"]) + r"$.}"
    )
    lines.append(r"\label{tab:convergence}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Backend & Final NLL & Rel.\ diff & Curve cos. & Time (s) \\")
    lines.append(r"\midrule")

    for b in backends:
        name = b["name"]
        comp = comp_idx.get(name, {})
        if b["is_reference"]:
            rel_str = "--"
            cos_str = "--"
        else:
            rel_str = f"{comp.get('rel_diff', 0):.4%}"
            cos_str = f"{comp.get('loss_curve_cosine', 0):.6f}"

        lines.append(
            f"  {backend_label(name)} & {b['final_loss']:.2f} & "
            f"{rel_str} & {cos_str} & {b['elapsed_s']:.0f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def self_consistency_table(data):
    """Self-consistency checks table."""
    configs = data["configs"]

    # Use the main config for the table
    main_cfg = next((c for c in configs if c["label"] == "main"), configs[0])

    check_labels = {
        "marginals_non_negative": (r"Boundary marginals $\geq 0$", "min"),
        "marginal_sum_in_range": (r"Marginal sum $\in [1, T]$", "mean"),
        "emission_marginals_sum_to_1": (r"$\sum_c P(c \mid t) = 1$", "max_deviation"),
        "total_emission_marginals": (r"$\sum_{t,c} P(c \mid t) \approx T$", "max_deviation"),
        "emission_marginals_non_negative": (r"Emission marginals $\geq 0$", "min"),
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Self-consistency checks on boundary and emission marginals")
    lines.append(r"  (\S\ref{sec:correctness}).")
    lines.append(
        r"  Config: $B{=}" + str(main_cfg["batch"]) + r"$, "
        r"$T{=}" + str(main_cfg["T"]) + r"$, "
        r"$C{=}" + str(main_cfg["C"]) + r"$, "
        r"$K{=}" + str(main_cfg["K"]) + r"$.}"
    )
    lines.append(r"\label{tab:self-consistency}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r l}")
    lines.append(r"\toprule")
    lines.append(r"Invariant & Value & Status \\")
    lines.append(r"\midrule")

    for check in main_cfg["checks"]:
        name = check["name"]
        if name not in check_labels:
            continue
        label, val_key = check_labels[name]
        val = check.get(val_key, 0)
        status = r"\cmark" if check["passed"] else r"\xmark"
        lines.append(f"  {label} & {fmt_sci(val)} & {status} \\\\")

    # Variable-length masking
    vl_status = r"\cmark" if data.get("variable_length_passed", True) else r"\xmark"
    lines.append(f"  Padding marginals $= 0$ & -- & {vl_status} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def combined_correctness_table(json_data):
    """Combined correctness table for main text.

    Shows finite-diff results and convergence in a single table.
    """
    fd = json_data.get("finite_diff", {})
    conv = json_data.get("convergence", {})

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Correctness validation of the Triton Semi-CRF implementation.")
    lines.append(
        r"  \emph{Top}: autograd vs.\ central finite differences"
        r" ($\varepsilon{=}" + fmt_sci(fd.get("config", {}).get("eps", 1e-3)) + r"$)."
    )
    lines.append(
        r"  \emph{Bottom}: training convergence across backends"
        r" (" + str(conv.get("config", {}).get("epochs", 0)) + r" epochs).}"
    )
    lines.append(r"\label{tab:correctness}")
    lines.append(r"\small")

    # --- Panel A: Finite differences ---
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{4}{l}{\textbf{(a) Finite-difference gradient validation}} \\")
    lines.append(r"\midrule")
    lines.append(r"Parameter & Elements & Cosine sim. & Norm.\ max err \\")
    lines.append(r"\midrule")

    for p in fd.get("parameters", []):
        name_tex = p["name"].replace("_", r"\_")
        lines.append(
            f"  {name_tex} & {p['elements']:,} & "
            f"{p['cosine_similarity']:.6f} & "
            f"{fmt_sci(p['normalized_max_err'])} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textbf{(b) Training convergence}} \\")
    lines.append(r"\midrule")

    # Re-use columns: Backend & (Final NLL) & Rel. diff & Curve cos.
    lines.append(r"Backend & Final NLL & Rel.\ diff & Curve cos. \\")
    lines.append(r"\midrule")

    comp_idx = {c["name"]: c for c in conv.get("comparisons", [])}
    for b in conv.get("backends", []):
        name = b["name"]
        comp = comp_idx.get(name, {})
        if b["is_reference"]:
            rel_str = "--"
            cos_str = "--"
        else:
            rel_str = f"{comp.get('rel_diff', 0):.4\\%}"
            cos_str = f"{comp.get('loss_curve_cosine', 0):.6f}"

        lines.append(
            f"  {backend_label(name)} & {b['final_loss']:.2f} & " f"{rel_str} & {cos_str} \\\\"
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
        description="Convert correctness validation JSON to LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="JSON file from validate_correctness.py --json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Write .tex files to this directory (default: print to stdout)",
    )
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    tables = {}

    # Combined main-text table (if both finite-diff and convergence present)
    if "finite_diff" in data and "convergence" in data:
        tables["correctness"] = combined_correctness_table(data)

    # Individual tables
    if "finite_diff" in data:
        tables["finite_diff"] = finite_diff_table(data["finite_diff"])

    if "convergence" in data:
        tables["convergence"] = convergence_table(data["convergence"])

    if "self_consistency" in data:
        tables["self_consistency"] = self_consistency_table(data["self_consistency"])

    if not tables:
        print("No data found in JSON. Run validation with --test all.", file=sys.stderr)
        return 1

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
