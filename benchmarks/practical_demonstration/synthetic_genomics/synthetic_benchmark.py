#!/usr/bin/env python3
"""
Synthetic Genomics Benchmark

A benchmark demonstrating that Semi-CRF learns biologically meaningful
duration distributions on synthetic genomic data with known ground truth.

This addresses reviewer feedback by showing:
1. Boundary F1 improvement over softmax and linear CRF
2. Duration distribution recovery (the "money plot")
3. Training stability at genomic scales (T=10k-50k)
4. Memory/throughput validation

Four-way model comparison:
1. Mamba → softmax: Per-position classifier (baseline)
2. Mamba → linear CRF (K=1): No duration modeling
3. Mamba → Semi-CRF (K=1000, learned): Full duration modeling
4. Mamba → Semi-CRF (K=1000, uniform): Ablation isolating duration contribution

Usage:
    # Generate synthetic data
    python synthetic_benchmark.py generate \\
        --output-dir data/synthetic \\
        --num-train 1000 --num-val 200 --num-test 200 \\
        --seq-length 20000

    # Run full benchmark
    python synthetic_benchmark.py run \\
        --data-dir data/synthetic \\
        --output-dir results/synthetic \\
        --encoder mamba

    # Compare models on existing checkpoints
    python synthetic_benchmark.py compare \\
        --results-dir results/synthetic
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import data generation
from synthetic_data import (
    SIMPLE_LABEL_NAMES,
    SIMPLE_LABELS,
    Segment,
    SyntheticGeneConfig,
    SyntheticGenomicsGenerator,
    SyntheticSequence,
    compute_duration_statistics,
    get_ground_truth_distributions,
    load_dataset,
    save_dataset,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Mamba SSM encoder (optional)
try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None  # type: ignore

# pytorch-crf baseline (optional)
try:
    from torchcrf import CRF as TorchCRF

    HAS_TORCHCRF = True
except ImportError:
    HAS_TORCHCRF = False
    TorchCRF = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

NUM_CLASSES = 3  # intergenic, exon, intron
DNA_DIM = 5  # A, C, G, T, N (one-hot)
KMER_K = 4  # 4-mer for k-mer encoding
KMER_DIM = 4**KMER_K  # 256 for 4-mers


# =============================================================================
# Feature Encoding
# =============================================================================


class OneHotEncoder:
    """One-hot encode DNA sequences."""

    DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    def __init__(self):
        self.dim = DNA_DIM

    def encode(self, sequence: str) -> torch.Tensor:
        """Encode DNA string to one-hot tensor (T, 5)."""
        encoded = torch.zeros(len(sequence), self.dim)
        for i, base in enumerate(sequence):
            encoded[i, self.DNA_VOCAB.get(base.upper(), 4)] = 1.0
        return encoded


class KmerEncoder:
    """
    K-mer frequency encoder with sliding window.

    Encodes local k-mer composition around each position.
    Output dimension is 4^k (256 for k=4).
    """

    def __init__(self, k: int = KMER_K, window: int = 20):
        self.k = k
        self.window = window
        self.dim = 4**k

        # Build k-mer to index mapping
        self.kmer_to_idx = {}
        bases = "ACGT"
        idx = 0
        self._build_kmers(bases, "", k, idx)

    def _build_kmers(self, bases: str, prefix: str, k: int, idx: int) -> int:
        if k == 0:
            self.kmer_to_idx[prefix] = idx
            return idx + 1
        for b in bases:
            idx = self._build_kmers(bases, prefix + b, k - 1, idx)
        return idx

    def encode(self, sequence: str) -> torch.Tensor:
        """
        Encode DNA string to k-mer frequency tensor (T, 4^k).

        Each position gets a frequency vector of k-mers in its local window.
        """
        T = len(sequence)
        encoded = torch.zeros(T, self.dim)
        seq = sequence.upper()

        for i in range(T):
            # Get window bounds
            start = max(0, i - self.window // 2)
            end = min(T, i + self.window // 2)

            # Count k-mers in window
            counts = torch.zeros(self.dim)
            for j in range(start, end - self.k + 1):
                kmer = seq[j : j + self.k]
                if kmer in self.kmer_to_idx:
                    counts[self.kmer_to_idx[kmer]] += 1

            # Normalize
            total = counts.sum()
            if total > 0:
                counts = counts / total

            encoded[i] = counts

        return encoded


# =============================================================================
# Dataset
# =============================================================================


class SyntheticGenomicsDataset(Dataset):
    """
    PyTorch Dataset for synthetic genomics benchmark.

    Supports one-hot and k-mer encoding options.
    """

    def __init__(
        self,
        sequences: list[SyntheticSequence],
        features: Literal["onehot", "kmer"] = "onehot",
        max_length: int | None = None,
    ):
        """
        Args:
            sequences: List of SyntheticSequence objects
            features: Feature encoding type ("onehot" or "kmer")
            max_length: Optional maximum sequence length (for truncation)
        """
        self.sequences = sequences
        self.max_length = max_length

        if features == "onehot":
            self.encoder = OneHotEncoder()
        else:
            self.encoder = KmerEncoder()

        self.feature_dim = self.encoder.dim

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        seq_data = self.sequences[idx]

        sequence = seq_data.sequence
        labels = seq_data.labels

        # Truncate if needed
        if self.max_length and len(sequence) > self.max_length:
            sequence = sequence[: self.max_length]
            labels = labels[: self.max_length]

        # Encode features
        seq_encoded = self.encoder.encode(sequence)

        return {
            "sequence": seq_encoded,
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": torch.tensor(len(sequence), dtype=torch.long),
        }


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate function with padding."""
    max_len = max(b["length"].item() for b in batch)

    sequences = []
    labels_list = []
    lengths = []

    for b in batch:
        seq = b["sequence"]
        lab = b["labels"]
        length = b["length"].item()

        # Pad sequence
        if seq.shape[0] < max_len:
            padding = torch.zeros(max_len - seq.shape[0], seq.shape[1])
            seq = torch.cat([seq, padding], dim=0)

        # Pad labels with -100 (ignore index)
        if lab.shape[0] < max_len:
            padding = torch.full((max_len - lab.shape[0],), -100, dtype=torch.long)
            lab = torch.cat([lab, padding])

        sequences.append(seq)
        labels_list.append(lab)
        lengths.append(length)

    return {
        "sequence": torch.stack(sequences),
        "labels": torch.stack(labels_list),
        "lengths": torch.tensor(lengths),
    }


# =============================================================================
# Encoders (BiLSTM, Mamba, MambaStub)
# =============================================================================


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            (batch, T, hidden_dim)
        """
        output, _ = self.lstm(x)
        return output


class MambaBlockStub(nn.Module):
    """
    CPU-compatible Mamba approximation using GRU.

    For development/testing without mamba-ssm GPU dependency.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        d_inner = d_model * expand
        self.conv = nn.Conv1d(d_model, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=1)
        self.gru = nn.GRU(d_inner, d_inner, batch_first=True)
        self.proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        # Conv
        y = self.conv(x.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        y = F.silu(y)
        # GRU (SSM approximation)
        y, _ = self.gru(y)
        # Project back
        y = self.proj(y)
        return self.norm(y + residual)


class MambaEncoderStub(nn.Module):
    """CPU-compatible bidirectional Mamba approximation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim // 2)
        self.norm = nn.LayerNorm(hidden_dim // 2)

        self.forward_layers = nn.ModuleList(
            [MambaBlockStub(hidden_dim // 2, d_state, d_conv, expand) for _ in range(num_layers)]
        )
        self.backward_layers = nn.ModuleList(
            [MambaBlockStub(hidden_dim // 2, d_state, d_conv, expand) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Embed
        h = self.embed(x)
        h = self.norm(h)

        h_fwd = h
        h_bwd = h.flip(dims=[1])

        # Forward direction
        for layer in self.forward_layers:
            h_fwd = layer(h_fwd)
            h_fwd = self.dropout(h_fwd)

        # Backward direction
        for layer in self.backward_layers:
            h_bwd = layer(h_bwd)
            h_bwd = self.dropout(h_bwd)

        h_bwd = h_bwd.flip(dims=[1])

        # Concatenate
        out = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.output_norm(out)


class MambaEncoder(nn.Module):
    """
    Bidirectional Mamba SSM encoder.

    Requires mamba-ssm package (GPU only).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba-ssm required. Install with: pip install mamba-ssm")

        self.embed = nn.Linear(input_dim, hidden_dim // 2)
        self.norm = nn.LayerNorm(hidden_dim // 2)

        self.forward_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=hidden_dim // 2,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )
        self.backward_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=hidden_dim // 2,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Embed
        h = self.embed(x)
        h = self.norm(h)

        h_fwd = h
        h_bwd = h.flip(dims=[1])

        # Forward direction with residual connections
        for layer in self.forward_layers:
            h_fwd = layer(h_fwd) + h_fwd
            h_fwd = self.dropout(h_fwd)

        # Backward direction with residual connections
        for layer in self.backward_layers:
            h_bwd = layer(h_bwd) + h_bwd
            h_bwd = self.dropout(h_bwd)

        h_bwd = h_bwd.flip(dims=[1])

        # Concatenate
        out = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.output_norm(out)


def create_encoder(
    encoder_type: Literal["bilstm", "mamba", "mamba_stub"],
    input_dim: int,
    hidden_dim: int,
    num_layers: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Create encoder based on type."""
    if encoder_type == "bilstm":
        return BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "mamba":
        return MambaEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "mamba_stub":
        return MambaEncoderStub(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


# =============================================================================
# Models
# =============================================================================


class SoftmaxModel(nn.Module):
    """
    Mamba/BiLSTM → per-position softmax baseline.

    No CRF structure - just position-wise classification.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, sequence: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            sequence: (batch, T, input_dim)
            lengths: (batch,)
        Returns:
            (batch, T, num_classes) logits
        """
        hidden = self.encoder(sequence)
        return self.classifier(hidden)

    def compute_loss(
        self,
        sequence: Tensor,
        lengths: Tensor,
        labels: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute cross-entropy loss with length masking."""
        logits = self.forward(sequence, lengths)
        batch, T, C = logits.shape

        # Create mask
        mask = torch.arange(T, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Cross-entropy
        loss = F.cross_entropy(
            logits.view(-1, C), labels.view(-1), ignore_index=-100, reduction="none"
        )
        loss = loss.view(batch, T)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def decode(self, sequence: Tensor, lengths: Tensor, **kwargs):
        """Decode via argmax."""
        logits = self.forward(sequence, lengths)
        return logits.argmax(dim=-1)  # (batch, T)


class PytorchCRFModel(nn.Module):
    """
    Mamba/BiLSTM → external pytorch-crf baseline.

    Uses the torchcrf library (pip install pytorch-crf) for comparison.
    This is a linear CRF without duration modeling.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if not HAS_TORCHCRF:
            raise ImportError(
                "pytorch-crf required for this model. Install with: pip install pytorch-crf"
            )
        self.encoder = encoder
        self.emission_proj = nn.Linear(hidden_dim, num_classes)
        self.crf = TorchCRF(num_classes, batch_first=True)

    def forward(self, sequence: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass returning emission scores."""
        hidden = self.encoder(sequence)
        return self.emission_proj(hidden)

    def compute_loss(
        self,
        sequence: Tensor,
        lengths: Tensor,
        labels: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute NLL loss using pytorch-crf."""
        hidden = self.encoder(sequence)
        emissions = self.emission_proj(hidden)

        _, seq_len = sequence.shape[:2]
        mask = torch.arange(seq_len, device=sequence.device).unsqueeze(0) < lengths.unsqueeze(1)

        # pytorch-crf requires valid labels (no -100), replace padding with 0
        labels_clean = labels.clone()
        labels_clean[labels == -100] = 0

        # pytorch-crf.forward() returns log-likelihood, we want NLL
        log_likelihood = self.crf(emissions, labels_clean, mask=mask, reduction="mean")
        return -log_likelihood

    def decode(self, sequence: Tensor, lengths: Tensor, **kwargs) -> list[list[int]]:
        """Viterbi decode to get best label sequences."""
        hidden = self.encoder(sequence)
        emissions = self.emission_proj(hidden)

        _, seq_len = sequence.shape[:2]
        mask = torch.arange(seq_len, device=sequence.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Returns list of lists (batch_size x variable_length)
        return self.crf.decode(emissions, mask=mask)


class SemiCRFModel(nn.Module):
    """
    Mamba/BiLSTM → Semi-CRF model.

    Supports both linear CRF (K=1) and semi-CRF (K>1) with learned
    or uniform duration distributions.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_CLASSES,
        max_duration: int = 1,
        hidden_dim: int = 256,
        duration_distribution: str = "learned",
        backend: str = "streaming",
        use_triton: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.max_duration = max_duration
        self.backend = backend
        self.use_triton = use_triton

        from torch_semimarkov import SemiMarkovCRFHead

        self.crf = SemiMarkovCRFHead(
            num_classes=num_classes,
            max_duration=max_duration,
            hidden_dim=hidden_dim,
            duration_distribution=duration_distribution,
        )

    def forward(self, sequence: Tensor, lengths: Tensor) -> dict:
        hidden = self.encoder(sequence)
        return self.crf(hidden, lengths)

    def compute_loss(
        self,
        sequence: Tensor,
        lengths: Tensor,
        labels: Tensor,
        backend: str | None = None,
        use_triton: bool | None = None,
    ) -> Tensor:
        hidden = self.encoder(sequence)
        return self.crf.compute_loss(
            hidden,
            lengths,
            labels,
            backend=backend if backend is not None else self.backend,
            use_triton=use_triton if use_triton is not None else self.use_triton,
        )

    def decode(
        self,
        sequence: Tensor,
        lengths: Tensor,
        backend: str | None = None,
        use_triton: bool | None = None,
    ):
        hidden = self.encoder(sequence)
        return self.crf.decode_with_traceback(
            hidden,
            lengths,
            backend=backend if backend is not None else self.backend,
            use_triton=use_triton if use_triton is not None else self.use_triton,
        )


def create_model(
    model_type: Literal["softmax", "pytorch-crf", "linear", "semicrf", "semicrf_uniform"],
    encoder_type: Literal["bilstm", "mamba", "mamba_stub"],
    input_dim: int,
    num_classes: int = NUM_CLASSES,
    max_duration: int = 1000,
    hidden_dim: int = 256,
    num_layers: int = 4,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    backend: str = "streaming",
    use_triton: bool = True,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Create a model based on type.

    Model types:
    - softmax: Per-position softmax (baseline)
    - pytorch-crf: External pytorch-crf library (linear CRF baseline)
    - linear: torch-semimarkov K=1 (linear CRF, same codebase as semi-CRF)
    - semicrf: Semi-CRF with learned duration
    - semicrf_uniform: Semi-CRF with uniform duration (ablation)
    """
    encoder = create_encoder(
        encoder_type=encoder_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
    )

    if model_type == "softmax":
        return SoftmaxModel(
            encoder=encoder,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )
    elif model_type == "pytorch-crf":
        return PytorchCRFModel(
            encoder=encoder,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )
    elif model_type == "linear":
        return SemiCRFModel(
            encoder=encoder,
            num_classes=num_classes,
            max_duration=1,
            hidden_dim=hidden_dim,
            duration_distribution="learned",
            backend=backend,
            use_triton=use_triton,
        )
    elif model_type == "semicrf":
        return SemiCRFModel(
            encoder=encoder,
            num_classes=num_classes,
            max_duration=max_duration,
            hidden_dim=hidden_dim,
            duration_distribution="learned",
            backend=backend,
            use_triton=use_triton,
        )
    elif model_type == "semicrf_uniform":
        return SemiCRFModel(
            encoder=encoder,
            num_classes=num_classes,
            max_duration=max_duration,
            hidden_dim=hidden_dim,
            duration_distribution="uniform",
            backend=backend,
            use_triton=use_triton,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class BenchmarkMetrics:
    """Metrics for synthetic genomics benchmark."""

    # Standard metrics
    position_f1: dict[str, float]
    position_f1_macro: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    boundary_f1_tolerance: dict[int, float]
    segment_precision: float
    segment_recall: float
    segment_f1: float

    # Duration calibration
    duration_kl: dict[str, float]
    duration_kl_mean: float

    # Training info
    final_train_loss: float
    loss_curve: list[float] = field(default_factory=list)

    # Performance
    training_time_seconds: float = 0.0
    inference_throughput_seqs_per_sec: float = 0.0
    peak_memory_gb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "position_f1": self.position_f1,
            "position_f1_macro": self.position_f1_macro,
            "boundary_precision": self.boundary_precision,
            "boundary_recall": self.boundary_recall,
            "boundary_f1": self.boundary_f1,
            "boundary_f1_tolerance": {str(k): v for k, v in self.boundary_f1_tolerance.items()},
            "segment_precision": self.segment_precision,
            "segment_recall": self.segment_recall,
            "segment_f1": self.segment_f1,
            "duration_kl": self.duration_kl,
            "duration_kl_mean": self.duration_kl_mean,
            "final_train_loss": self.final_train_loss,
            "loss_curve": self.loss_curve,
            "training_time_seconds": self.training_time_seconds,
            "inference_throughput_seqs_per_sec": self.inference_throughput_seqs_per_sec,
            "peak_memory_gb": self.peak_memory_gb,
        }


def extract_boundaries(labels: np.ndarray) -> set[int]:
    """Extract boundary positions from label sequence."""
    boundaries = set()
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1] and labels[i] != -100 and labels[i - 1] != -100:
            boundaries.add(i)
    return boundaries


def extract_segments(labels: np.ndarray) -> list[Segment]:
    """Extract segments from label sequence."""
    segments = []
    if len(labels) == 0:
        return segments

    start = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            if current_label != -100:
                segments.append(Segment(start=start, end=i, label=int(current_label)))
            start = i
            current_label = labels[i]

    if current_label != -100:
        segments.append(Segment(start=start, end=len(labels), label=int(current_label)))

    return segments


def compute_position_metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Compute position-level F1 scores."""
    all_preds = np.concatenate(predictions)
    all_targets = np.concatenate(targets)

    # Mask padding
    mask = all_targets != -100
    all_preds = all_preds[mask]
    all_targets = all_targets[mask]

    f1_scores = {}
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores[SIMPLE_LABEL_NAMES[c]] = f1

    f1_scores["macro"] = np.mean([v for k, v in f1_scores.items() if k != "macro"])
    return f1_scores


def compute_boundary_metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    tolerances: list[int] | None = None,
) -> dict[str, float]:
    """Compute boundary detection metrics."""
    if tolerances is None:
        tolerances = [0, 1, 2, 5, 10]

    results = {"tolerance_" + str(t): {"tp": 0, "fp": 0, "fn": 0} for t in tolerances}

    for pred, target in zip(predictions, targets, strict=False):
        mask = target != -100
        pred = pred[mask]
        target = target[mask]

        pred_bounds = extract_boundaries(pred)
        true_bounds = extract_boundaries(target)

        for tol in tolerances:
            key = f"tolerance_{tol}"
            matched_true = set()

            for pb in pred_bounds:
                for tb in true_bounds:
                    if abs(pb - tb) <= tol and tb not in matched_true:
                        results[key]["tp"] += 1
                        matched_true.add(tb)
                        break
                else:
                    results[key]["fp"] += 1

            results[key]["fn"] += len(true_bounds) - len(matched_true)

    metrics = {}
    for tol in tolerances:
        key = f"tolerance_{tol}"
        tp = results[key]["tp"]
        fp = results[key]["fp"]
        fn = results[key]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"boundary_f1_tol{tol}"] = f1
        if tol == 0:
            metrics["boundary_precision"] = precision
            metrics["boundary_recall"] = recall
            metrics["boundary_f1"] = f1

    return metrics


def compute_segment_metrics(
    pred_segments: list[list[Segment]],
    true_segments: list[list[Segment]],
) -> dict[str, float]:
    """Compute segment-level metrics."""
    tp = 0
    total_pred = 0
    total_true = 0

    for pred_segs, true_segs in zip(pred_segments, true_segments, strict=False):
        pred_set = {(s.start, s.end, s.label) for s in pred_segs}
        true_set = {(s.start, s.end, s.label) for s in true_segs}

        tp += len(pred_set & true_set)
        total_pred += len(pred_set)
        total_true += len(true_set)

    precision = tp / total_pred if total_pred > 0 else 0
    recall = tp / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
    }


def compute_duration_kl(
    pred_segments: list[list[Segment]],
    true_durations: dict[int, list[int]],
    num_classes: int = NUM_CLASSES,
    max_duration: int = 1000,
) -> tuple[dict[str, float], torch.Tensor]:
    """
    Compute KL divergence between predicted and true duration distributions.

    Returns both KL divergence per class and the predicted distribution tensor
    for visualization.
    """
    # Build predicted duration histogram
    pred_durations = {c: [] for c in range(num_classes)}
    for segments in pred_segments:
        for seg in segments:
            dur = min(seg.end - seg.start, max_duration)
            pred_durations[seg.label].append(dur)

    pred_dist = torch.zeros(max_duration, num_classes)
    kl_per_class = {}

    for c in range(num_classes):
        # Predicted distribution
        if pred_durations[c]:
            hist_pred, _ = np.histogram(pred_durations[c], bins=np.arange(1, max_duration + 2))
            hist_pred = hist_pred.astype(np.float32) + 1e-8
            hist_pred = hist_pred / hist_pred.sum()
            pred_dist[:, c] = torch.from_numpy(hist_pred)

        # True distribution
        if true_durations.get(c):
            true_lengths = [min(d, max_duration) for d in true_durations[c]]
            hist_true, _ = np.histogram(true_lengths, bins=np.arange(1, max_duration + 2))
            hist_true = hist_true.astype(np.float32) + 1e-8
            hist_true = hist_true / hist_true.sum()
            true_dist_c = torch.from_numpy(hist_true)

            # KL divergence: KL(true || pred)
            kl = (true_dist_c * torch.log(true_dist_c / (pred_dist[:, c] + 1e-8))).sum()
            kl_per_class[SIMPLE_LABEL_NAMES[c]] = float(kl)
        else:
            kl_per_class[SIMPLE_LABEL_NAMES[c]] = float("nan")

    return kl_per_class, pred_dist


# =============================================================================
# Training
# =============================================================================


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    crf_reg: float = 0.0,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        crf_reg: L2 regularization coefficient for CRF parameters (Semi-Markov only)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        sequence = batch["sequence"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        # Replace -100 padding with 0 for validation (ignored anyway via lengths)
        labels_clean = labels.clone()
        labels_clean[labels_clean == -100] = 0

        optimizer.zero_grad()
        loss = model.compute_loss(sequence, lengths, labels_clean)

        # Add CRF parameter regularization for Semi-Markov models
        if crf_reg > 0 and isinstance(model, SemiCRFModel):
            loss = loss + crf_reg * model.crf.parameter_penalty()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log first batch diagnostics
        if epoch == 0 and batch_idx == 0:
            with torch.no_grad():
                if hasattr(model, "encoder"):
                    hidden = model.encoder(sequence)
                    logger.info(
                        f"Encoder output: mean={hidden.mean():.4f}, "
                        f"std={hidden.std():.4f}, "
                        f"min={hidden.min():.4f}, max={hidden.max():.4f}"
                    )

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    true_durations: dict[int, list[int]],
    max_duration: int = 1000,
) -> BenchmarkMetrics:
    """Evaluate model and compute all metrics."""
    model.eval()

    all_predictions = []
    all_targets = []
    all_pred_segments = []
    all_true_segments = []

    # Determine model type for decode handling
    is_softmax = isinstance(model, SoftmaxModel)
    is_pytorch_crf = isinstance(model, PytorchCRFModel)

    start_time = time.time()
    num_seqs = 0

    for batch in dataloader:
        sequence = batch["sequence"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        # Decode
        result = model.decode(sequence, lengths)

        for i in range(len(lengths)):
            seq_len = lengths[i].item()
            true_labels = labels[i, :seq_len].cpu().numpy()

            if is_softmax:
                # Softmax returns (batch, T) tensor
                pred_labels = result[i, :seq_len].cpu().numpy()
            elif is_pytorch_crf:
                # pytorch-crf returns list[list[int]]
                pred_labels = np.array(result[i][:seq_len], dtype=np.int64)
            else:
                # torch-semimarkov returns DecodeResult with segments
                # Segment.end is INCLUSIVE, convert to exclusive
                pred_labels = np.zeros(seq_len, dtype=np.int64)
                for seg in result.segments[i]:
                    pred_labels[seg.start : seg.end + 1] = seg.label

            pred_segs = extract_segments(pred_labels)
            true_segs = extract_segments(true_labels)

            all_predictions.append(pred_labels)
            all_targets.append(true_labels)
            all_pred_segments.append(pred_segs)
            all_true_segments.append(true_segs)
            num_seqs += 1

    inference_time = time.time() - start_time

    # Compute all metrics
    position_metrics = compute_position_metrics(all_predictions, all_targets)
    boundary_metrics = compute_boundary_metrics(all_predictions, all_targets)
    segment_metrics = compute_segment_metrics(all_pred_segments, all_true_segments)
    duration_kl, pred_dist = compute_duration_kl(
        all_pred_segments, true_durations, max_duration=max_duration
    )

    kl_values = [v for v in duration_kl.values() if not np.isnan(v)]
    duration_kl_mean = np.mean(kl_values) if kl_values else float("nan")

    return BenchmarkMetrics(
        position_f1={k: v for k, v in position_metrics.items() if k != "macro"},
        position_f1_macro=position_metrics["macro"],
        boundary_precision=boundary_metrics["boundary_precision"],
        boundary_recall=boundary_metrics["boundary_recall"],
        boundary_f1=boundary_metrics["boundary_f1"],
        boundary_f1_tolerance={
            int(k.split("tol")[1]): v
            for k, v in boundary_metrics.items()
            if k.startswith("boundary_f1_tol")
        },
        segment_precision=segment_metrics["segment_precision"],
        segment_recall=segment_metrics["segment_recall"],
        segment_f1=segment_metrics["segment_f1"],
        duration_kl=duration_kl,
        duration_kl_mean=duration_kl_mean,
        final_train_loss=0.0,  # Set by caller
        inference_throughput_seqs_per_sec=num_seqs / inference_time if inference_time > 0 else 0,
    )


def train_model(
    model_type: Literal["softmax", "pytorch-crf", "linear", "semicrf", "semicrf_uniform"],
    train_sequences: list[SyntheticSequence],
    val_sequences: list[SyntheticSequence],
    test_sequences: list[SyntheticSequence],
    encoder_type: str = "mamba",
    features: str = "onehot",
    max_duration: int = 1000,
    hidden_dim: int = 256,
    num_layers: int = 4,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_every: int = 10,
    backend: str = "streaming",
    use_triton: bool = True,
    weight_decay: float = 1e-5,
    crf_reg: float = 0.0,
    dropout: float = 0.1,
) -> tuple[nn.Module, BenchmarkMetrics]:
    """Train a model and return it with test metrics.

    Args:
        model_type: Type of model to train
        train_sequences: Training sequences
        val_sequences: Validation sequences
        test_sequences: Test sequences
        encoder_type: Type of encoder (bilstm, mamba, mamba_stub)
        features: Feature encoding (onehot, kmer)
        max_duration: Maximum segment duration K
        hidden_dim: Hidden dimension
        num_layers: Number of encoder layers
        batch_size: Batch size
        learning_rate: Learning rate
        epochs: Number of epochs
        device: Device to train on
        log_every: Log every N epochs
        backend: Semi-CRF backend
        use_triton: Whether to use Triton kernels
        weight_decay: AdamW weight decay
        crf_reg: L2 regularization for CRF parameters
        dropout: Dropout rate for encoder
    """
    device_obj = torch.device(device)

    # Create datasets
    train_dataset = SyntheticGenomicsDataset(train_sequences, features=features)
    val_dataset = SyntheticGenomicsDataset(val_sequences, features=features)
    test_dataset = SyntheticGenomicsDataset(test_sequences, features=features)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Collect true durations for KL computation
    true_durations = {c: [] for c in range(NUM_CLASSES)}
    for seq in train_sequences + val_sequences + test_sequences:
        for c, lengths in seq.true_durations.items():
            true_durations[c].extend(lengths)

    # Create model
    input_dim = train_dataset.feature_dim
    model = create_model(
        model_type=model_type,
        encoder_type=encoder_type,
        input_dim=input_dim,
        max_duration=max_duration,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        backend=backend,
        use_triton=use_triton,
        dropout=dropout,
    ).to(device_obj)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model: {model_type} + {encoder_type}, "
        f"K={max_duration if model_type.startswith('semicrf') else 1}, "
        f"params={num_params:,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_curve = []
    best_val_f1 = 0.0
    best_state = None

    start_time = time.time()
    peak_memory = 0.0

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device_obj, epoch=epoch, crf_reg=crf_reg
        )
        scheduler.step()
        loss_curve.append(train_loss)

        # Track memory
        if torch.cuda.is_available():
            current_mem = torch.cuda.max_memory_allocated() / 1e9
            peak_memory = max(peak_memory, current_mem)

        # Evaluate periodically
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            val_metrics = evaluate(model, val_loader, device_obj, true_durations, max_duration)
            logger.info(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                f"Val Boundary F1: {val_metrics.boundary_f1:.4f} | "
                f"Val Duration KL: {val_metrics.duration_kl_mean:.4f}"
            )

            if val_metrics.boundary_f1 > best_val_f1:
                best_val_f1 = val_metrics.boundary_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    training_time = time.time() - start_time

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, device_obj, true_durations, max_duration)
    test_metrics.final_train_loss = loss_curve[-1]
    test_metrics.loss_curve = loss_curve
    test_metrics.training_time_seconds = training_time
    test_metrics.peak_memory_gb = peak_memory

    return model, test_metrics


# =============================================================================
# Visualization
# =============================================================================


def plot_duration_distributions(
    results: dict[str, BenchmarkMetrics],
    ground_truth: torch.Tensor,
    pred_distributions: dict[str, torch.Tensor],
    output_path: Path,
    max_display_duration: int = 500,
):
    """
    Create the "money plot" showing learned vs true duration distributions.

    This is the key visualization proving Semi-CRF learns correct durations.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping duration plot")
        return

    class_names = list(SIMPLE_LABELS.keys())
    model_names = list(results.keys())

    fig, axes = plt.subplots(
        len(class_names),
        len(model_names) + 1,
        figsize=(3.5 * (len(model_names) + 1), 3 * len(class_names)),
    )

    # Ensure 2D axes array
    if len(class_names) == 1:
        axes = axes.reshape(1, -1)

    for row, class_name in enumerate(class_names):
        c = SIMPLE_LABELS[class_name]

        # Ground truth column
        ax = axes[row, 0]
        gt = ground_truth[:max_display_duration, c].numpy()
        ax.bar(range(1, len(gt) + 1), gt, alpha=0.7, color="green")
        ax.set_title(f"{class_name}\n(Ground Truth)")
        ax.set_xlabel("Duration")
        ax.set_ylabel("Probability")
        ax.set_xlim(0, max_display_duration)

        # Each model
        for col, model_name in enumerate(model_names, 1):
            ax = axes[row, col]

            if model_name in pred_distributions:
                pred = pred_distributions[model_name][:max_display_duration, c].numpy()
                ax.bar(range(1, len(pred) + 1), pred, alpha=0.7, color="blue", label="Predicted")
                ax.plot(range(1, len(gt) + 1), gt, "r-", linewidth=2, alpha=0.8, label="True")

            kl = results[model_name].duration_kl.get(class_name, float("nan"))
            ax.set_title(f"{model_name}\nKL={kl:.3f}")
            ax.set_xlabel("Duration")
            ax.set_xlim(0, max_display_duration)

            if row == 0 and col == 1:
                ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved duration comparison plot to {output_path}")


def plot_training_curves(
    results: dict[str, BenchmarkMetrics],
    output_path: Path,
):
    """Plot training loss curves for all models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping training curves")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (model_name, metrics) in enumerate(results.items()):
        if metrics.loss_curve:
            epochs = range(1, len(metrics.loss_curve) + 1)
            ax.plot(
                epochs,
                metrics.loss_curve,
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Stability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def print_comparison_table(results: dict[str, BenchmarkMetrics]):
    """Print comparison table to console."""
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)

    print(
        f"{'Model':<20} {'Boundary F1':>12} {'Segment F1':>12} {'Position F1':>12} "
        f"{'Duration KL':>12} {'Time (s)':>10} {'Mem (GB)':>10}"
    )
    print("-" * 100)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<20} "
            f"{metrics.boundary_f1:>12.4f} "
            f"{metrics.segment_f1:>12.4f} "
            f"{metrics.position_f1_macro:>12.4f} "
            f"{metrics.duration_kl_mean:>12.4f} "
            f"{metrics.training_time_seconds:>10.1f} "
            f"{metrics.peak_memory_gb:>10.2f}"
        )

    print("=" * 100)


# =============================================================================
# CLI Commands
# =============================================================================


def cmd_generate(args):
    """Generate synthetic data."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SyntheticGeneConfig(num_classes=args.num_classes)
    generator = SyntheticGenomicsGenerator(config, seed=args.seed)

    logger.info(f"Generating {args.num_train} training sequences of ~{args.seq_length}bp...")
    train_seqs = generator.generate_dataset(args.num_train, args.seq_length)
    save_dataset(train_seqs, output_dir / "train.jsonl", config)

    logger.info(f"Generating {args.num_val} validation sequences...")
    val_seqs = generator.generate_dataset(args.num_val, args.seq_length)
    save_dataset(val_seqs, output_dir / "val.jsonl", config)

    logger.info(f"Generating {args.num_test} test sequences...")
    test_seqs = generator.generate_dataset(args.num_test, args.seq_length)
    save_dataset(test_seqs, output_dir / "test.jsonl", config)

    # Print statistics
    stats = compute_duration_statistics(train_seqs)
    print("\nDuration Statistics (training set):")
    for c, s in stats.items():
        name = config.label_names.get(c, str(c))
        print(
            f"  {name}: median={s['median']:.0f}, mean={s['mean']:.0f}, "
            f"range=[{s['min']}, {s['max']}]"
        )

    print(f"\nData saved to {output_dir}")


def cmd_run(args):
    """Run full benchmark."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {data_dir}...")
    train_seqs = load_dataset(data_dir / "train.jsonl")
    val_seqs = load_dataset(data_dir / "val.jsonl")
    test_seqs = load_dataset(data_dir / "test.jsonl")

    logger.info(
        f"Loaded {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences"
    )

    # Compute ground truth distribution
    all_seqs = train_seqs + val_seqs + test_seqs
    ground_truth = get_ground_truth_distributions(all_seqs, NUM_CLASSES, args.max_duration)

    # Train all models (or subset if --models specified)
    # 5-way comparison:
    # - softmax: per-position baseline
    # - pytorch-crf: external library linear CRF baseline (if available)
    # - linear: torch-semimarkov K=1 (same codebase as semi-CRF)
    # - semicrf: Semi-CRF with learned duration
    # - semicrf_uniform: Semi-CRF with uniform duration (ablation)
    all_model_types = ["softmax", "pytorch-crf", "linear", "semicrf", "semicrf_uniform"]
    model_types = args.models if args.models else all_model_types
    results = {}
    pred_distributions = {}

    for model_type in model_types:
        # Skip pytorch-crf if not installed
        if model_type == "pytorch-crf" and not HAS_TORCHCRF:
            logger.info(f"Skipping {model_type} (pytorch-crf not installed)")
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type}...")
        logger.info(f"{'='*60}")

        model, metrics = train_model(
            model_type=model_type,
            train_sequences=train_seqs,
            val_sequences=val_seqs,
            test_sequences=test_seqs,
            encoder_type=args.encoder,
            features=args.features,
            max_duration=args.max_duration,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            device=args.device,
            log_every=args.log_every,
            backend=args.backend,
            use_triton=not args.no_triton,
            weight_decay=args.weight_decay,
            crf_reg=args.crf_reg,
            dropout=args.dropout,
        )

        results[model_type] = metrics

        # Save checkpoint
        checkpoint_path = output_dir / "checkpoints" / f"{model_type}_best.pt"
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        # Compute predicted distribution for visualization
        # Re-evaluate to get pred_dist
        device_obj = torch.device(args.device)
        test_dataset = SyntheticGenomicsDataset(test_seqs, features=args.features)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

        true_durations = {c: [] for c in range(NUM_CLASSES)}
        for seq in all_seqs:
            for c, lengths in seq.true_durations.items():
                true_durations[c].extend(lengths)

        model.eval()
        all_pred_segments = []
        is_softmax = isinstance(model, SoftmaxModel)
        is_pytorch_crf = isinstance(model, PytorchCRFModel)
        with torch.no_grad():
            for batch in test_loader:
                sequence = batch["sequence"].to(device_obj)
                lengths = batch["lengths"].to(device_obj)
                result = model.decode(sequence, lengths)

                for i in range(len(lengths)):
                    seq_len = lengths[i].item()
                    if is_softmax:
                        pred_labels = result[i, :seq_len].cpu().numpy()
                    elif is_pytorch_crf:
                        pred_labels = np.array(result[i][:seq_len], dtype=np.int64)
                    else:
                        pred_labels = np.zeros(seq_len, dtype=np.int64)
                        for seg in result.segments[i]:
                            pred_labels[seg.start : seg.end + 1] = seg.label
                    all_pred_segments.append(extract_segments(pred_labels))

        _, pred_dist = compute_duration_kl(
            all_pred_segments, true_durations, max_duration=args.max_duration
        )
        pred_distributions[model_type] = pred_dist

        # Clear memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print results
    print_comparison_table(results)

    # Save results
    results_path = output_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
    logger.info(f"Saved metrics to {results_path}")

    # Generate plots
    plot_duration_distributions(
        results,
        ground_truth,
        pred_distributions,
        output_dir / "duration_comparison.png",
        max_display_duration=min(500, args.max_duration),
    )

    plot_training_curves(results, output_dir / "training_curves.png")

    print(f"\nResults saved to {output_dir}")


def cmd_compare(args):
    """Compare results from existing benchmark run."""
    results_dir = Path(args.results_dir)

    # Load metrics
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        logger.error(f"No metrics found at {metrics_path}")
        return

    with open(metrics_path) as f:
        results_dict = json.load(f)

    # Reconstruct BenchmarkMetrics objects
    results = {}
    for model_name, data in results_dict.items():
        results[model_name] = BenchmarkMetrics(
            position_f1=data["position_f1"],
            position_f1_macro=data["position_f1_macro"],
            boundary_precision=data["boundary_precision"],
            boundary_recall=data["boundary_recall"],
            boundary_f1=data["boundary_f1"],
            boundary_f1_tolerance={int(k): v for k, v in data["boundary_f1_tolerance"].items()},
            segment_precision=data["segment_precision"],
            segment_recall=data["segment_recall"],
            segment_f1=data["segment_f1"],
            duration_kl=data["duration_kl"],
            duration_kl_mean=data["duration_kl_mean"],
            final_train_loss=data.get("final_train_loss", 0),
            loss_curve=data.get("loss_curve", []),
            training_time_seconds=data.get("training_time_seconds", 0),
            inference_throughput_seqs_per_sec=data.get("inference_throughput_seqs_per_sec", 0),
            peak_memory_gb=data.get("peak_memory_gb", 0),
        )

    print_comparison_table(results)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Genomics Benchmark for Semi-CRF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    gen_parser.add_argument(
        "--num-train", type=int, default=1000, help="Number of training sequences"
    )
    gen_parser.add_argument(
        "--num-val", type=int, default=200, help="Number of validation sequences"
    )
    gen_parser.add_argument("--num-test", type=int, default=200, help="Number of test sequences")
    gen_parser.add_argument("--seq-length", type=int, default=20000, help="Target sequence length")
    gen_parser.add_argument(
        "--num-classes", type=int, choices=[3, 5], default=3, help="Number of classes"
    )
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run full benchmark")
    run_parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    run_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    run_parser.add_argument(
        "--encoder", choices=["bilstm", "mamba", "mamba_stub"], default="mamba", help="Encoder type"
    )
    run_parser.add_argument(
        "--features", choices=["onehot", "kmer"], default="onehot", help="Feature encoding"
    )
    run_parser.add_argument("--max-duration", type=int, default=1000, help="Max segment duration K")
    run_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    run_parser.add_argument("--num-layers", type=int, default=4, help="Number of encoder layers")
    run_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    run_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    run_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    run_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run_parser.add_argument("--log-every", type=int, default=10, help="Log every N epochs")
    run_parser.add_argument("--backend", default="streaming", help="Semi-CRF backend")
    run_parser.add_argument("--no-triton", action="store_true", help="Disable Triton kernels")
    run_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="AdamW weight decay (default: 1e-5)",
    )
    run_parser.add_argument(
        "--crf-reg",
        type=float,
        default=0.0,
        help="L2 regularization for CRF parameters (transition, duration_bias). "
        "Helps prevent gradient explosion. Try 1e-4 to 1e-3 if training unstable.",
    )
    run_parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for encoder (default: 0.1)",
    )
    run_parser.add_argument(
        "--models",
        nargs="+",
        choices=["softmax", "pytorch-crf", "linear", "semicrf", "semicrf_uniform"],
        default=None,
        help="Models to run (default: all). Example: --models semicrf semicrf_uniform",
    )

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare results")
    cmp_parser.add_argument("--results-dir", type=Path, required=True, help="Results directory")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
