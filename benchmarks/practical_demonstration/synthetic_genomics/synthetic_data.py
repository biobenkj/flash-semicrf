#!/usr/bin/env python3
"""
Synthetic Genomics Data Generator

Generates synthetic gene sequences with known structure:
- Genes with exons (~150bp log-normal) and introns (1-10kb heavy-tailed)
- Intergenic regions between genes
- Optional splice site motifs (GT...AG)
- Simple label scheme: C=3-5 labels

This provides ground truth for evaluating Semi-CRF duration learning.
The important observation: if Semi-CRF learns duration distributions that match
the known generating distributions, it validates the duration modeling
capability that linear CRFs cannot provide.

Usage:
    from synthetic_data import SyntheticGenomicsGenerator, SyntheticGeneConfig

    config = SyntheticGeneConfig()
    generator = SyntheticGenomicsGenerator(config, seed=42)

    # Generate a single sequence
    seq = generator.generate_sequence(target_length=10000)
    print(f"Sequence length: {len(seq.sequence)}")
    print(f"Exon lengths: {seq.true_durations[1]}")  # Should be ~150bp median
    print(f"Intron lengths: {seq.true_durations[2]}")  # Should be ~3kb median

    # Generate a dataset
    train_seqs = generator.generate_dataset(num_sequences=1000, target_length=10000)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Label Schemes
# =============================================================================

# Simple 3-class scheme (recommended for synthetic benchmark)
SIMPLE_LABELS = {
    "intergenic": 0,
    "exon": 1,
    "intron": 2,
}

# Extended 5-class scheme (optional, for more detailed experiments)
EXTENDED_LABELS = {
    "intergenic": 0,
    "exon": 1,
    "intron": 2,
    "5UTR": 3,
    "3UTR": 4,
}

# Reverse mappings
SIMPLE_LABEL_NAMES = {v: k for k, v in SIMPLE_LABELS.items()}
EXTENDED_LABEL_NAMES = {v: k for k, v in EXTENDED_LABELS.items()}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SyntheticGeneConfig:
    """
    Configuration for synthetic gene generation.

    The default parameters are biologically motivated:
    - Exon lengths: log-normal with median ~150bp (typical mammalian exon)
    - Intron lengths: heavy-tailed log-normal, median ~3kb (typical mammalian intron)
    - Intergenic: log-normal with median ~5kb

    These distributions create a benchmark where:
    1. Duration information is highly informative (exons vs introns differ dramatically)
    2. A linear CRF cannot encode duration preferences
    3. A Semi-CRF should learn the true generating distributions
    """

    # Exon length distribution: log-normal with median ~150bp
    # P(X < median) = 0.5 for log-normal when mu = log(median)
    exon_mu: float = field(default_factory=lambda: float(np.log(150)))
    exon_sigma: float = 0.5  # Controls spread; ~0.5 gives reasonable variance
    exon_min: int = 30  # Minimum biological exon length
    exon_max: int = 500  # Cap for K constraint

    # Intron length distribution: heavy-tailed log-normal
    # Larger sigma creates heavier tail
    intron_mu: float = field(default_factory=lambda: float(np.log(3000)))
    intron_sigma: float = 1.0  # Heavy tail: sigma=1.0 gives ~10x spread at 95%ile
    intron_min: int = 100  # Minimum intron length
    intron_max: int = 10000  # Maximum for K constraint

    # Gene structure
    min_exons_per_gene: int = 2  # At least 2 exons (1 intron)
    max_exons_per_gene: int = 8  # Typical gene complexity

    # Intergenic spacing between genes
    intergenic_mu: float = field(default_factory=lambda: float(np.log(5000)))
    intergenic_sigma: float = 0.8
    intergenic_min: int = 500
    intergenic_max: int = 20000

    # Sequence content
    add_splice_motifs: bool = True  # Add GT...AG at intron boundaries
    gc_content: float = 0.42  # Typical mammalian GC content

    # Label scheme
    num_classes: int = 3  # 3 for simple, 5 for extended
    label_scheme: Literal["simple", "extended"] = "simple"

    def __post_init__(self):
        # Ensure mu values are properly computed if defaults were used
        if self.exon_mu == 0:
            self.exon_mu = float(np.log(150))
        if self.intron_mu == 0:
            self.intron_mu = float(np.log(3000))
        if self.intergenic_mu == 0:
            self.intergenic_mu = float(np.log(5000))

    @property
    def labels(self) -> dict[str, int]:
        """Get the label mapping for this configuration."""
        return SIMPLE_LABELS if self.label_scheme == "simple" else EXTENDED_LABELS

    @property
    def label_names(self) -> dict[int, str]:
        """Get the reverse label mapping."""
        return SIMPLE_LABEL_NAMES if self.label_scheme == "simple" else EXTENDED_LABEL_NAMES


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Segment:
    """A single segment in a sequence."""

    start: int  # Inclusive
    end: int  # Exclusive
    label: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SyntheticSequence:
    """
    A single synthetic genomic sequence with labels.

    Attributes:
        sequence: DNA sequence string (A, C, G, T)
        labels: Per-position labels as numpy array
        segments: List of Segment objects
        gene_boundaries: List of (start, end) tuples for each gene
        true_durations: Dict mapping label -> list of segment lengths
    """

    sequence: str
    labels: np.ndarray
    segments: list[Segment]
    gene_boundaries: list[tuple[int, int]]
    true_durations: dict[int, list[int]]

    def __len__(self) -> int:
        return len(self.sequence)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "sequence": self.sequence,
            "labels": self.labels.tolist(),
            "segments": [{"start": s.start, "end": s.end, "label": s.label} for s in self.segments],
            "gene_boundaries": self.gene_boundaries,
            "true_durations": {str(k): v for k, v in self.true_durations.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> SyntheticSequence:
        """Create from JSON dictionary."""
        return cls(
            sequence=data["sequence"],
            labels=np.array(data["labels"], dtype=np.int32),
            segments=[
                Segment(start=s["start"], end=s["end"], label=s["label"]) for s in data["segments"]
            ],
            gene_boundaries=data["gene_boundaries"],
            true_durations={int(k): v for k, v in data["true_durations"].items()},
        )


# =============================================================================
# Generator
# =============================================================================


class SyntheticGenomicsGenerator:
    """
    Generate synthetic genomic sequences with known gene structure.

    The generator creates sequences with:
    1. Multiple genes separated by intergenic regions
    2. Each gene has 2-8 exons separated by introns
    3. Exon lengths: log-normal ~150bp (median)
    4. Intron lengths: heavy-tailed log-normal 1-10kb
    5. Optional splice site motifs (GT...AG)

    Example:
        config = SyntheticGeneConfig()
        generator = SyntheticGenomicsGenerator(config, seed=42)

        # Generate sequences
        train_seqs = generator.generate_dataset(num_sequences=1000, target_length=10000)

        # Inspect duration distributions
        exon_lengths = [d for seq in train_seqs for d in seq.true_durations[1]]
        print(f"Exon median: {np.median(exon_lengths)}")  # Should be ~150
    """

    def __init__(self, config: SyntheticGeneConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Nucleotide alphabet
        self.nucleotides = ["A", "C", "G", "T"]

        # Splice site consensus sequences
        self.donor_site = "GT"  # 5' splice site (start of intron)
        self.acceptor_site = "AG"  # 3' splice site (end of intron)

    def _sample_exon_length(self) -> int:
        """
        Sample exon length from log-normal distribution.

        Log-normal with mu=log(150), sigma=0.5 gives:
        - Median: ~150bp
        - Mean: ~170bp
        - 95% range: ~60-400bp
        """
        length = int(self.rng.lognormal(self.config.exon_mu, self.config.exon_sigma))
        return max(self.config.exon_min, min(length, self.config.exon_max))

    def _sample_intron_length(self) -> int:
        """
        Sample intron length from heavy-tailed log-normal.

        Log-normal with mu=log(3000), sigma=1.0 gives:
        - Median: ~3kb
        - Mean: ~5kb (heavy right tail)
        - 95% range: ~400bp-20kb
        """
        length = int(self.rng.lognormal(self.config.intron_mu, self.config.intron_sigma))
        return max(self.config.intron_min, min(length, self.config.intron_max))

    def _sample_intergenic_length(self) -> int:
        """Sample intergenic region length from log-normal."""
        length = int(self.rng.lognormal(self.config.intergenic_mu, self.config.intergenic_sigma))
        return max(self.config.intergenic_min, min(length, self.config.intergenic_max))

    def _generate_random_sequence(self, length: int) -> str:
        """
        Generate random DNA sequence with specified GC content.

        GC content of 0.42 means:
        - P(G) = P(C) = 0.21
        - P(A) = P(T) = 0.29
        """
        gc = self.config.gc_content
        at = 1 - gc
        probs = [at / 2, gc / 2, gc / 2, at / 2]  # A, C, G, T
        return "".join(self.rng.choice(self.nucleotides, size=length, p=probs))

    def _generate_gene(self) -> tuple[str, list[Segment], dict[int, list[int]]]:
        """
        Generate a single gene with exons and introns.

        Returns:
            tuple: (sequence, segments, durations)
                - sequence: DNA string for the gene
                - segments: list of Segment objects (positions relative to gene start)
                - durations: dict mapping label -> list of segment lengths
        """
        labels = self.config.labels
        num_exons = self.rng.integers(
            self.config.min_exons_per_gene, self.config.max_exons_per_gene + 1
        )

        sequence_parts: list[str] = []
        segments: list[Segment] = []
        durations: dict[int, list[int]] = {c: [] for c in range(self.config.num_classes)}
        position = 0

        for i in range(num_exons):
            # Generate exon
            exon_len = self._sample_exon_length()
            exon_seq = self._generate_random_sequence(exon_len)

            # Record segment
            exon_label = labels["exon"]
            segments.append(Segment(start=position, end=position + exon_len, label=exon_label))
            durations[exon_label].append(exon_len)

            sequence_parts.append(exon_seq)
            position += exon_len

            # Generate intron (except after last exon)
            if i < num_exons - 1:
                intron_len = self._sample_intron_length()
                intron_seq = self._generate_random_sequence(intron_len)

                # Add splice site motifs at intron boundaries
                if self.config.add_splice_motifs and intron_len >= 10:
                    # GT at start (donor), AG at end (acceptor)
                    intron_seq = self.donor_site + intron_seq[2:-2] + self.acceptor_site

                intron_label = labels["intron"]
                segments.append(
                    Segment(start=position, end=position + intron_len, label=intron_label)
                )
                durations[intron_label].append(intron_len)

                sequence_parts.append(intron_seq)
                position += intron_len

        return "".join(sequence_parts), segments, durations

    def generate_sequence(self, target_length: int) -> SyntheticSequence:
        """
        Generate a synthetic sequence of approximately target_length.

        Generates genes interspersed with intergenic regions until
        reaching the target length. The actual length may differ slightly
        due to gene boundary alignment.

        Args:
            target_length: Approximate sequence length (will be close but not exact)

        Returns:
            SyntheticSequence with sequence, labels, and ground truth metadata
        """
        labels = self.config.labels
        full_sequence: list[str] = []
        all_segments: list[Segment] = []
        all_durations: dict[int, list[int]] = {c: [] for c in range(self.config.num_classes)}
        gene_boundaries: list[tuple[int, int]] = []
        position = 0

        # Optionally start with intergenic region
        if self.rng.random() > 0.3:  # 70% chance of starting with intergenic
            intergenic_len = self._sample_intergenic_length()
            intergenic_len = min(intergenic_len, target_length // 4)
            intergenic_seq = self._generate_random_sequence(intergenic_len)
            full_sequence.append(intergenic_seq)

            intergenic_label = labels["intergenic"]
            all_segments.append(
                Segment(start=position, end=position + intergenic_len, label=intergenic_label)
            )
            all_durations[intergenic_label].append(intergenic_len)
            position += intergenic_len

        # Generate genes with intergenic spacers
        while position < target_length:
            # Generate a gene
            gene_start = position
            gene_seq, gene_segments, gene_durations = self._generate_gene()

            # Adjust segment positions to absolute coordinates
            adjusted_segments = [
                Segment(start=s.start + position, end=s.end + position, label=s.label)
                for s in gene_segments
            ]

            full_sequence.append(gene_seq)
            all_segments.extend(adjusted_segments)
            for c, lengths in gene_durations.items():
                all_durations[c].extend(lengths)

            position += len(gene_seq)
            gene_boundaries.append((gene_start, position))

            if position >= target_length:
                break

            # Add intergenic region between genes
            intergenic_len = self._sample_intergenic_length()
            remaining = target_length - position
            intergenic_len = min(intergenic_len, remaining)

            if intergenic_len > 100:  # Only add if substantial
                intergenic_seq = self._generate_random_sequence(intergenic_len)
                full_sequence.append(intergenic_seq)

                intergenic_label = labels["intergenic"]
                all_segments.append(
                    Segment(start=position, end=position + intergenic_len, label=intergenic_label)
                )
                all_durations[intergenic_label].append(intergenic_len)
                position += intergenic_len

        # Build final sequence and labels
        final_sequence = "".join(full_sequence)
        labels_array = np.zeros(len(final_sequence), dtype=np.int32)

        for seg in all_segments:
            if seg.end <= len(final_sequence):
                labels_array[seg.start : seg.end] = seg.label

        return SyntheticSequence(
            sequence=final_sequence,
            labels=labels_array,
            segments=all_segments,
            gene_boundaries=gene_boundaries,
            true_durations=all_durations,
        )

    def generate_dataset(
        self,
        num_sequences: int,
        target_length: int,
        length_variance: float = 0.1,
    ) -> list[SyntheticSequence]:
        """
        Generate a dataset of synthetic sequences.

        Args:
            num_sequences: Number of sequences to generate
            target_length: Target length for each sequence
            length_variance: Fraction of length variance (0.1 = +/- 10%)

        Returns:
            List of SyntheticSequence objects
        """
        sequences = []
        for i in range(num_sequences):
            # Vary length slightly to avoid artificial patterns
            actual_length = int(
                target_length * (1 + self.rng.uniform(-length_variance, length_variance))
            )
            seq = self.generate_sequence(actual_length)
            sequences.append(seq)

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_sequences} sequences")

        return sequences


# =============================================================================
# Utility Functions
# =============================================================================


def get_ground_truth_distributions(
    sequences: list[SyntheticSequence],
    num_classes: int,
    max_duration: int,
) -> torch.Tensor:
    """
    Compute ground truth duration distributions from synthetic data.

    This is used to compare against learned duration distributions.
    The "money plot" shows how well Semi-CRF recovers these true distributions.

    Args:
        sequences: List of SyntheticSequence objects
        num_classes: Number of label classes
        max_duration: Maximum duration K

    Returns:
        Tensor of shape (K, C) with normalized probabilities (not log)
    """
    # Collect all durations per class
    all_durations: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for seq in sequences:
        for c, lengths in seq.true_durations.items():
            all_durations[c].extend(lengths)

    # Build histogram-based distribution
    probs = torch.zeros(max_duration, num_classes)

    for c in range(num_classes):
        if len(all_durations[c]) == 0:
            # Uniform fallback for classes with no samples
            probs[:, c] = 1.0 / max_duration
            continue

        durations = np.array(all_durations[c])
        # Clip to valid range [1, max_duration]
        durations = np.clip(durations, 1, max_duration)

        # Build histogram with bins [1, 2, ..., max_duration+1)
        hist, _ = np.histogram(durations, bins=np.arange(1, max_duration + 2))
        hist = hist.astype(np.float32) + 1e-8  # Add smoothing
        hist = hist / hist.sum()  # Normalize

        probs[:, c] = torch.from_numpy(hist)

    return probs


def compute_duration_statistics(sequences: list[SyntheticSequence]) -> dict:
    """
    Compute summary statistics of duration distributions.

    Returns dict with per-class statistics (mean, median, std, min, max).
    """
    all_durations: dict[int, list[int]] = {}
    for seq in sequences:
        for c, lengths in seq.true_durations.items():
            if c not in all_durations:
                all_durations[c] = []
            all_durations[c].extend(lengths)

    stats = {}
    for c, lengths in all_durations.items():
        if len(lengths) > 0:
            arr = np.array(lengths)
            stats[c] = {
                "count": len(lengths),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            }
    return stats


def save_dataset(
    sequences: list[SyntheticSequence],
    output_path: Path,
    config: SyntheticGeneConfig,
) -> None:
    """
    Save dataset to JSONL file.

    Each line is a JSON object representing one sequence.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for seq in sequences:
            f.write(json.dumps(seq.to_dict()) + "\n")

    # Also save config and statistics
    stats = compute_duration_statistics(sequences)
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "config": {
                    "exon_mu": config.exon_mu,
                    "exon_sigma": config.exon_sigma,
                    "intron_mu": config.intron_mu,
                    "intron_sigma": config.intron_sigma,
                    "num_classes": config.num_classes,
                },
                "num_sequences": len(sequences),
                "total_length": sum(len(seq) for seq in sequences),
                "duration_statistics": stats,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved {len(sequences)} sequences to {output_path}")


def load_dataset(input_path: Path) -> list[SyntheticSequence]:
    """Load dataset from JSONL file."""
    sequences = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line.strip())
            sequences.append(SyntheticSequence.from_dict(data))
    return sequences


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic genomics data")
    parser.add_argument("--num-sequences", type=int, default=10, help="Number of sequences")
    parser.add_argument("--target-length", type=int, default=10000, help="Target sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, help="Output JSONL file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Generate data
    config = SyntheticGeneConfig()
    generator = SyntheticGenomicsGenerator(config, seed=args.seed)

    print(f"Generating {args.num_sequences} sequences of ~{args.target_length}bp...")
    sequences = generator.generate_dataset(
        num_sequences=args.num_sequences,
        target_length=args.target_length,
    )

    # Print statistics
    stats = compute_duration_statistics(sequences)
    print("\nDuration Statistics:")
    label_names = config.label_names
    for c, s in stats.items():
        print(
            f"  {label_names.get(c, c)}: median={s['median']:.0f}, mean={s['mean']:.0f}, "
            f"range=[{s['min']}, {s['max']}], 5-95%=[{s['p5']:.0f}, {s['p95']:.0f}]"
        )

    # Save if output specified
    if args.output:
        save_dataset(sequences, args.output, config)

    # Print example
    print("\nExample sequence (first 200bp):")
    seq = sequences[0]
    print(f"  DNA: {seq.sequence[:200]}...")
    print(f"  Labels: {seq.labels[:200].tolist()}...")
    print(f"  Total segments: {len(seq.segments)}")
    print(f"  Genes: {len(seq.gene_boundaries)}")
