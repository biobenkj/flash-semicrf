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
1. Mamba -> softmax: Per-position classifier (baseline)
2. Mamba -> linear CRF (K=1): No duration modeling
3. Mamba -> Semi-CRF (K=1000, learned): Full duration modeling
4. Mamba -> Semi-CRF (K=1000, uniform): Ablation isolating duration contribution

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
import math
import os
import tempfile
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
from torch.nn.utils.rnn import pad_sequence as rnn_pad_sequence
from torch.utils.data import DataLoader, Dataset

# Mamba SSM encoder (optional)
try:
    from mamba_ssm import Mamba

    try:
        from mamba_ssm import Mamba2  # mamba-ssm >= 2.0 top-level export

        HAS_MAMBA2 = True
    except ImportError:
        try:
            from mamba_ssm.modules.mamba2 import Mamba2  # fallback submodule path

            HAS_MAMBA2 = True
        except ImportError:
            HAS_MAMBA2 = False
            Mamba2 = None  # type: ignore

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    HAS_MAMBA2 = False
    Mamba = None  # type: ignore
    Mamba2 = None  # type: ignore

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
        # Pre-build a 256-entry byte→index lookup (unknown bytes map to 4 = N)
        self._byte_to_idx = np.full(256, 4, dtype=np.int64)
        for base, idx in self.DNA_VOCAB.items():
            self._byte_to_idx[ord(base.upper())] = idx
            self._byte_to_idx[ord(base.lower())] = idx

    def encode(self, sequence: str) -> torch.Tensor:
        """Encode DNA string to one-hot tensor (T, 5) — fully vectorized."""
        arr = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
        indices = self._byte_to_idx[arr]  # (T,)
        encoded = np.eye(self.dim, dtype=np.float32)[indices]  # (T, 5)
        return torch.from_numpy(encoded)


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

        Each position gets a normalized frequency vector of k-mers in its
        local window.  Vectorized: builds a (T, dim) accumulator via
        ``np.add.at`` over all k-mer positions, then applies sliding-window
        normalisation without any Python loop over T.
        """
        T = len(sequence)
        seq = sequence.upper()
        k = self.k
        half_w = self.window // 2

        # Build integer index array for each k-mer start position
        kmer_indices = np.array(
            [self.kmer_to_idx.get(seq[j : j + k], -1) for j in range(T - k + 1)],
            dtype=np.int64,
        )
        valid_mask = kmer_indices >= 0
        valid_positions = np.where(valid_mask)[0]  # k-mer start indices
        valid_kmer_ids = kmer_indices[valid_mask]  # corresponding dim index

        # For each k-mer at position j, it contributes to windows centred at
        # positions max(0, j - half_w + k - 1) … min(T-1, j + half_w).
        encoded = np.zeros((T, self.dim), dtype=np.float32)

        # Vectorised: accumulate k-mer counts, then smooth with a uniform window
        # using cumulative sums — O(T * dim) instead of O(T * W * k).
        kmer_counts = np.zeros((T - k + 1, self.dim), dtype=np.float32)
        if len(valid_positions) > 0:
            np.add.at(kmer_counts, (valid_positions, valid_kmer_ids), 1.0)

        # Prefix sum along position axis for O(1) window queries
        prefix = np.zeros((T - k + 2, self.dim), dtype=np.float32)
        prefix[1:] = np.cumsum(kmer_counts, axis=0)

        for i in range(T):
            j_start = max(0, i - half_w)
            j_end = min(T - k + 1, i + half_w + 1)
            if j_start < j_end:
                counts = prefix[j_end] - prefix[j_start]
                total = counts.sum()
                if total > 0:
                    encoded[i] = counts / total

        return torch.from_numpy(encoded)


# =============================================================================
# Dataset
# =============================================================================


class SyntheticGenomicsDataset(Dataset):
    """
    PyTorch Dataset for synthetic genomics benchmark.

    Supports one-hot and k-mer encoding options.

    Cache modes
    -----------
    ``ram``    : pre-encode all sequences once into a list of numpy arrays in
                 ``__init__``.  Best for onehot (≈1.4 GiB for 1,400×50k) or
                 when running with num_workers=0 / Linux fork workers.
    ``memmap`` : write the encoded arrays to per-sequence .npy files in
                 *cache_dir* (a temp directory by default), then load them
                 with ``mmap_mode='r'``.  Workers share the OS page-cache so
                 the data is never duplicated regardless of the spawn method.
                 Recommended for kmer mode (≈66.8 GiB) or multi-process runs.
    ``none``   : original behaviour — encode on every ``__getitem__`` call.
    """

    def __init__(
        self,
        sequences: list[SyntheticSequence],
        features: Literal["onehot", "kmer"] = "onehot",
        max_length: int | None = None,
        cache_mode: Literal["ram", "memmap", "none"] = "ram",
        cache_dir: Path | None = None,
    ):
        """
        Args:
            sequences: List of SyntheticSequence objects
            features: Feature encoding type ("onehot" or "kmer")
            max_length: Optional maximum sequence length (for truncation)
            cache_mode: "ram" | "memmap" | "none"
            cache_dir: Directory for memmap files (temp dir if None)
        """
        self.sequences = sequences
        self.max_length = max_length
        self.cache_mode = cache_mode

        if features == "onehot":
            self.encoder = OneHotEncoder()
        else:
            self.encoder = KmerEncoder()

        self.feature_dim = self.encoder.dim

        # Pre-compute encoded sequences
        self._ram_cache: list[np.ndarray] | None = None
        self._memmap_files: list[Path] | None = None

        if cache_mode == "ram":
            logger.info(
                f"Pre-encoding {len(sequences)} sequences into RAM cache " f"(features={features})…"
            )
            self._ram_cache = [self._encode_sequence(s.sequence, s.labels)[0] for s in sequences]
        elif cache_mode == "memmap":
            _cache_dir = Path(cache_dir or tempfile.mkdtemp(prefix="genomics_cache_"))
            _cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Pre-encoding {len(sequences)} sequences to memmap cache "
                f"at {_cache_dir} (features={features})…"
            )
            self._memmap_files = []
            for i, s in enumerate(sequences):
                fpath = _cache_dir / f"{i}.npy"
                if not fpath.exists():
                    arr, _ = self._encode_sequence(s.sequence, s.labels)
                    np.save(str(fpath), arr)
                self._memmap_files.append(fpath)

    def _encode_sequence(self, sequence: str, labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Truncate, encode, and return (encoded_np, labels_np)."""
        if self.max_length and len(sequence) > self.max_length:
            sequence = sequence[: self.max_length]
            labels = labels[: self.max_length]
        encoded = self.encoder.encode(sequence)
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.numpy()
        return encoded, np.array(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        seq_data = self.sequences[idx]

        if self.cache_mode == "ram" and self._ram_cache is not None:
            seq_encoded = torch.from_numpy(self._ram_cache[idx])
            sequence = seq_data.sequence
            labels = seq_data.labels
            if self.max_length and len(sequence) > self.max_length:
                labels = labels[: self.max_length]
        elif self.cache_mode == "memmap" and self._memmap_files is not None:
            seq_encoded = torch.from_numpy(
                np.load(str(self._memmap_files[idx]), mmap_mode="r").copy()
            )
            sequence = seq_data.sequence
            labels = seq_data.labels
            if self.max_length and len(sequence) > self.max_length:
                labels = labels[: self.max_length]
        else:
            sequence = seq_data.sequence
            labels = seq_data.labels
            if self.max_length and len(sequence) > self.max_length:
                sequence = sequence[: self.max_length]
                labels = labels[: self.max_length]
            seq_encoded = self.encoder.encode(sequence)

        seq_len = seq_encoded.shape[0]
        return {
            "sequence": seq_encoded,
            "labels": torch.tensor(labels[:seq_len], dtype=torch.long),
            "length": torch.tensor(seq_len, dtype=torch.long),
        }


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate function with padding (uses pad_sequence for single-pass allocation)."""
    sequences = [b["sequence"] for b in batch]
    labels_list = [b["labels"] for b in batch]
    lengths = torch.stack([b["length"] for b in batch])

    seq_padded = rnn_pad_sequence(sequences, batch_first=True)  # (B, T_max, D)
    lab_padded = rnn_pad_sequence(labels_list, batch_first=True, padding_value=-100)  # (B, T_max)

    return {
        "sequence": seq_padded,
        "labels": lab_padded,
        "lengths": lengths,
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


def _pick_headdim(d_inner: int) -> int:
    """Return the largest value in [128, 64, 32, 16, 8] that evenly divides d_inner.

    Mamba2 requires headdim to divide d_model * expand (d_inner). Using min() is
    insufficient because the result may not divide d_inner (e.g. d_inner=96, min=64,
    but 96 % 64 != 0).
    """
    for hd in (128, 64, 32, 16, 8):
        if d_inner % hd == 0:
            return hd
    raise ValueError(
        f"No valid headdim found for d_inner={d_inner}. "
        "Ensure (hidden_dim // 2) * expand is divisible by 8."
    )


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
        use_checkpoint: bool = False,
        mamba_version: int = 2,
        chunk_size: int = 256,
    ):
        """
        Args:
            use_checkpoint: If True, wrap each Mamba layer with
                torch.utils.checkpoint (Python-level recomputation).  Default False.
                Mamba1 already recomputes conv1d_out and delta during its backward
                CUDA kernel (checkpoint_lvl=1); Mamba2 uses a chunked SSD scan
                (use_mem_eff_path=True) that amortizes activation memory without any
                Python-level recomputation.  Adding a Python-level wrapper on top of
                either causes double recomputation with no additional memory benefit.
                Enable only if you are OOM after exhausting batch-size / grad-accum
                options and the built-in kernel recomputation is insufficient.
            mamba_version: 1 = Mamba (selective scan, checkpoint_lvl=1 internally).
                2 = Mamba2 (SSD chunked scan, use_mem_eff_path=True; default).
                Requires mamba-ssm >= 2.0 for version 2.
            chunk_size: Mamba2 SSD chunk size (ignored for mamba_version=1).
                Smaller values reduce peak activation memory at the cost of more
                recomputation. Default 256; try 64–128 if OOM at large T.
        """
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba-ssm required. Install with: pip install mamba-ssm")

        self.use_checkpoint = use_checkpoint
        self.mamba_version = mamba_version

        self.embed = nn.Linear(input_dim, hidden_dim // 2)
        self.norm = nn.LayerNorm(hidden_dim // 2)

        if mamba_version == 2:
            if not HAS_MAMBA2:
                raise ImportError(
                    "Mamba2 requires mamba-ssm >= 2.0. "
                    "Install a newer release or pass mamba_version=1."
                )
            d_inner = (hidden_dim // 2) * expand
            _kw = {
                "d_model": hidden_dim // 2,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
                "headdim": _pick_headdim(d_inner),
                "chunk_size": chunk_size,
            }
            _Cls = Mamba2
        else:
            _kw = {
                "d_model": hidden_dim // 2,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
            }
            _Cls = Mamba

        self.forward_layers = nn.ModuleList([_Cls(**_kw) for _ in range(num_layers)])
        self.backward_layers = nn.ModuleList([_Cls(**_kw) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _apply_layer(self, layer: nn.Module, h: Tensor) -> Tensor:
        """Apply one Mamba layer with optional gradient checkpointing + residual."""
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint as ckpt

            return ckpt(layer, h, use_reentrant=False) + h
        return layer(h) + h

    def forward(self, x: Tensor) -> Tensor:
        # Embed
        h = self.embed(x)
        h = self.norm(h)

        h_fwd = h
        h_bwd = h.flip(dims=[1])

        # Forward direction with residual connections
        for layer in self.forward_layers:
            h_fwd = self._apply_layer(layer, h_fwd)
            h_fwd = self.dropout(h_fwd)

        # Backward direction with residual connections
        for layer in self.backward_layers:
            h_bwd = self._apply_layer(layer, h_bwd)
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
    use_checkpoint: bool = False,
    mamba_version: int = 2,
    chunk_size: int = 256,
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
            use_checkpoint=use_checkpoint,
            mamba_version=mamba_version,
            chunk_size=chunk_size,
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
    Mamba/BiLSTM -> per-position softmax baseline.

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
    Mamba/BiLSTM -> external pytorch-crf baseline.

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
    Mamba/BiLSTM -> Semi-CRF model.

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

        from flash_semicrf import SemiMarkovCRFHead

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
    use_checkpoint: bool = False,
    mamba_version: int = 2,
    chunk_size: int = 256,
) -> nn.Module:
    """
    Create a model based on type.

    Model types:
    - softmax: Per-position softmax (baseline)
    - pytorch-crf: External pytorch-crf library (linear CRF baseline)
    - linear: flash-semicrf K=1 (linear CRF, same codebase as semi-CRF)
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
        use_checkpoint=use_checkpoint,
        mamba_version=mamba_version,
        chunk_size=chunk_size,
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

    # Encoder configuration — serialized to JSON for reproducibility.
    mamba_version: int = 2
    chunk_size: int = 256
    d_state: int = 16

    # Predicted duration distribution — populated from the final test evaluation.
    # Shape: (max_duration, num_classes). Not serialized to JSON.
    pred_dist: torch.Tensor | None = None

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
            "mamba_version": self.mamba_version,
            "chunk_size": self.chunk_size,
            "d_state": self.d_state,
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

        pred_sorted = sorted(pred_bounds)
        true_sorted = sorted(true_bounds)

        for tol in tolerances:
            key = f"tolerance_{tol}"
            # O(n+m) two-pointer greedy match — preserves one-to-one semantics and
            # avoids O(n*m) temporary allocations from broadcasting.
            matched_true_count = 0
            tp = 0
            fp = 0
            j = 0  # pointer into true_sorted
            for pb in pred_sorted:
                # Advance j past true boundaries that are too far left
                while j < len(true_sorted) and true_sorted[j] < pb - tol:
                    j += 1
                if j < len(true_sorted) and abs(true_sorted[j] - pb) <= tol:
                    tp += 1
                    matched_true_count += 1
                    j += 1  # consume this true boundary (one-to-one)
                else:
                    fp += 1
            fn = len(true_sorted) - matched_true_count
            results[key]["tp"] += tp
            results[key]["fp"] += fp
            results[key]["fn"] += fn

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
    # Build predicted duration histogram — flatten all segments in one pass
    if pred_segments and any(segs for segs in pred_segments):
        all_durs = np.array(
            [min(seg.end - seg.start, max_duration) for segs in pred_segments for seg in segs],
            dtype=np.int64,
        )
        all_seg_labels = np.array(
            [seg.label for segs in pred_segments for seg in segs], dtype=np.int64
        )
        pred_durations = {c: all_durs[all_seg_labels == c] for c in range(num_classes)}
    else:
        pred_durations = {c: np.array([], dtype=np.int64) for c in range(num_classes)}

    pred_dist = torch.zeros(max_duration, num_classes)
    kl_per_class = {}

    for c in range(num_classes):
        # Predicted distribution
        if len(pred_durations[c]) > 0:
            hist_pred, _ = np.histogram(pred_durations[c], bins=np.arange(1, max_duration + 2))
            hist_pred = hist_pred.astype(np.float32) + 1e-8
            hist_pred = hist_pred / hist_pred.sum()
            pred_dist[:, c] = torch.from_numpy(hist_pred)

        # True distribution
        if true_durations.get(c):
            true_lengths = np.minimum(true_durations[c], max_duration)
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
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    debug_timing: bool = False,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        crf_reg: L2 regularization coefficient for CRF parameters (Semi-Markov only)
        use_amp: Wrap forward/backward with BF16 autocast (GPU only)
        grad_accum_steps: Accumulate gradients over this many microbatches before
            each optimizer step. Effective batch = physical_batch × grad_accum_steps.
            The last partial window (when len(dataloader) % grad_accum_steps != 0) is
            scaled by its actual size so every optimizer step is correctly normalized.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    _amp = use_amp and device.type == "cuda"

    n = len(dataloader)
    remainder = n % grad_accum_steps  # 0 → all windows are full

    # zero_grad once before the loop; subsequent resets happen after each step
    optimizer.zero_grad(set_to_none=True)

    t_prev = time.time()

    for batch_idx, batch in enumerate(dataloader):
        t_loaded = time.time()

        sequence = batch["sequence"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)

        t_h2d = time.time()

        # Replace -100 padding with 0 (ignored anyway via lengths)
        labels_clean = labels.clone()
        labels_clean[labels_clean == -100] = 0

        is_last_batch = batch_idx == n - 1
        is_update_step = ((batch_idx + 1) % grad_accum_steps == 0) or is_last_batch

        # Last partial window has fewer microbatches than grad_accum_steps.
        # Use its actual size so the gradient magnitude is correctly normalized.
        in_last_partial = (remainder != 0) and (batch_idx >= n - remainder)
        ws = remainder if in_last_partial else grad_accum_steps

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_amp):
            loss = model.compute_loss(sequence, lengths, labels_clean) / ws
            # CRF regularization uses the same window denominator as the data loss
            if crf_reg > 0 and isinstance(model, SemiCRFModel):
                loss = loss + crf_reg * model.crf.parameter_penalty() / ws

        loss.backward()  # accumulates into .grad

        total_loss += loss.item() * ws  # unscale for logging
        num_batches += 1

        if is_update_step:
            # Clip and step once per accumulation window, not per microbatch
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        t_step = time.time()
        if debug_timing and (batch_idx % 50 == 0):
            logger.info(
                f"[timing] batch={batch_idx} load={t_loaded - t_prev:.3f}s "
                f"h2d={t_h2d - t_loaded:.3f}s step={t_step - t_h2d:.3f}s"
            )
        t_prev = t_loaded

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


@torch.inference_mode()
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
        # Labels are only used for metric computation (all CPU numpy) — keep on CPU.
        sequence = batch["sequence"].to(device, non_blocking=True)
        labels_cpu = batch["labels"].numpy()  # already CPU
        lengths = batch["lengths"].to(device, non_blocking=True)

        # Decode
        result = model.decode(sequence, lengths)

        # Single batch transfer before the inner loop (one CUDA sync total)
        lengths_np = lengths.cpu().numpy()
        if is_softmax:
            result_np = result.cpu().numpy()

        for i in range(len(lengths_np)):
            seq_len = int(lengths_np[i])
            true_labels = labels_cpu[i, :seq_len]

            if is_softmax:
                pred_labels = result_np[i, :seq_len]
            elif is_pytorch_crf:
                # pytorch-crf returns list[list[int]]
                pred_labels = np.array(result[i][:seq_len], dtype=np.int64)
            else:
                # flash-semicrf returns DecodeResult with segments
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
        pred_dist=pred_dist,
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
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_every: int = 10,
    backend: str = "streaming",
    use_triton: bool = True,
    weight_decay: float = 1e-5,
    crf_reg: float = 0.0,
    dropout: float = 0.1,
    num_workers: int = 8,
    cache_mode: Literal["ram", "memmap", "none"] = "ram",
    use_amp: bool = True,
    use_checkpoint: bool = False,
    grad_accum_steps: int = 1,
    mamba_version: int = 2,
    chunk_size: int = 256,
    d_state: int = 16,
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
        num_workers: DataLoader worker processes (0 = main process only)
        cache_mode: Feature-encoding cache strategy ("ram" | "memmap" | "none")
        use_amp: Enable BF16 autocast for encoder forward/backward (halves Mamba memory)
        use_checkpoint: Enable Python-level gradient checkpointing on Mamba layers
            (default False — mamba-ssm already recomputes internally via checkpoint_lvl=1)
        grad_accum_steps: Accumulate gradients over this many microbatches before each
            optimizer step. Effective batch = batch_size × grad_accum_steps.
            Use when physical batch size is limited by VRAM at large T.
        mamba_version: 1 = Mamba (selective scan), 2 = Mamba2 (SSD chunked scan; default).
        chunk_size: Mamba2 SSD chunk size (ignored for mamba_version=1).
        d_state: SSM state dimension (Mamba1 default 16; Mamba2 natural default 128).
    """
    device_obj = torch.device(device)

    # Create datasets (encoding is pre-computed once according to cache_mode)
    train_dataset = SyntheticGenomicsDataset(
        train_sequences, features=features, cache_mode=cache_mode
    )
    val_dataset = SyntheticGenomicsDataset(val_sequences, features=features, cache_mode=cache_mode)
    test_dataset = SyntheticGenomicsDataset(
        test_sequences, features=features, cache_mode=cache_mode
    )

    _pin = torch.cuda.is_available()
    _nw = num_workers
    _pf = 4 if _nw > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_nw > 0,
        prefetch_factor=_pf,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_nw > 0,
        prefetch_factor=_pf,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_nw > 0,
        prefetch_factor=_pf,
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
        d_state=d_state,
        backend=backend,
        use_triton=use_triton,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
        mamba_version=mamba_version,
        chunk_size=chunk_size,
    ).to(device_obj)

    num_params = sum(p.numel() for p in model.parameters())
    expected_steps = math.ceil(len(train_loader) / grad_accum_steps)
    logger.info(
        f"Model: {model_type} + {encoder_type}, "
        f"K={max_duration if model_type.startswith('semicrf') else 1}, "
        f"params={num_params:,}"
    )
    _enc_desc = (
        f"mamba{mamba_version}, d_state={d_state}"
        + (f", chunk_size={chunk_size}" if mamba_version == 2 else "")
        if encoder_type == "mamba"
        else encoder_type
    )
    logger.info(
        f"Encoder: {_enc_desc} | "
        f"Batch: physical={batch_size}, accum={grad_accum_steps}, "
        f"effective={batch_size * grad_accum_steps} | "
        f"steps/epoch={expected_steps} | "
        f"SM saturation target: ~142 for L40S"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_curve = []
    best_val_f1 = 0.0
    best_state = None

    start_time = time.time()
    peak_memory = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # isolate this model's peak from prior runs

    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device_obj,
            epoch=epoch,
            crf_reg=crf_reg,
            use_amp=use_amp,
            grad_accum_steps=grad_accum_steps,
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
            mem_info = ""
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                mem_info = f" | mem alloc={alloc:.2f}GB reserved={reserved:.2f}GB"
                # Return unused cached blocks to CUDA — prevents the caching allocator
                # from slowly ratcheting up nvidia-smi reserved memory over long runs.
                torch.cuda.empty_cache()
            logger.info(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                f"Val Boundary F1: {val_metrics.boundary_f1:.4f} | "
                f"Val Duration KL: {val_metrics.duration_kl_mean:.4f}"
                f"{mem_info}"
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
    test_metrics.mamba_version = mamba_version
    test_metrics.chunk_size = chunk_size
    test_metrics.d_state = d_state

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
    if args.chunk_size <= 0:
        logger.error(f"--chunk-size must be > 0, got {args.chunk_size}")
        import sys

        sys.exit(1)
    if args.mamba_version == 1 and args.chunk_size != 256:
        logger.warning("--chunk-size is ignored when --mamba-version 1")

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
    # - linear: flash-semicrf K=1 (same codebase as semi-CRF)
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
            num_workers=args.num_workers,
            cache_mode=args.cache_mode,
            use_amp=not args.no_amp,
            use_checkpoint=args.checkpoint,
            grad_accum_steps=args.grad_accum,
            mamba_version=args.mamba_version,
            chunk_size=args.chunk_size,
            d_state=args.d_state,
        )

        results[model_type] = metrics

        # Save checkpoint (filename encodes the encoder config for traceability)
        _mv = args.mamba_version if args.encoder == "mamba" else ""
        _mv_tag = f"_mamba{_mv}" if _mv else ""
        checkpoint_path = output_dir / "checkpoints" / f"{model_type}{_mv_tag}_best.pt"
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        # pred_dist is already computed inside train_model's final evaluate() call
        pred_distributions[model_type] = metrics.pred_dist

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


def _smoke_test_mamba(device: torch.device, required_version: int | None = None) -> None:
    """Instantiate Mamba1 and Mamba2 encoders, run one fwd/bwd to catch config errors early.

    Args:
        device: Device to run the test on.
        required_version: If set and the smoke test for that version fails, raise a
            RuntimeError with install instructions rather than just logging a warning.
    """
    if not HAS_MAMBA:
        return
    versions = [1]
    if HAS_MAMBA2:
        versions.append(2)
    passed = []
    for version in versions:
        try:
            enc = MambaEncoder(
                input_dim=4,
                hidden_dim=64,
                num_layers=1,
                mamba_version=version,
            ).to(device)
            x = torch.randn(2, 16, 4, device=device)
            loss = enc(x).sum()
            loss.backward()
            del enc, x, loss
            passed.append(version)
        except Exception as e:
            if version == required_version:
                raise RuntimeError(
                    f"Mamba{version} smoke test failed: {e}\n"
                    "Mamba2 requires the 'causal_conv1d' CUDA extension in addition to "
                    "mamba-ssm. Install it with:\n"
                    "  pip install causal-conv1d\n"
                    "Or fall back to Mamba1 with --mamba-version 1."
                ) from e
            logger.warning(f"Mamba{version} smoke test failed (skipping version {version}): {e}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"Mamba smoke test passed (versions={passed}).")


def cmd_profile(args):
    """
    Quick profiling mode: one model, 2 epochs, fixed seed, per-phase timers.

    Loads (or generates on the fly) a tiny dataset, runs ``args.model`` for
    ``args.profile_epochs`` epochs, and prints a per-phase wall-clock
    breakdown so you can see which bottleneck dominates after each change.
    """
    if args.chunk_size <= 0:
        logger.error(f"--chunk-size must be > 0, got {args.chunk_size}")
        import sys

        sys.exit(1)
    if args.mamba_version == 1 and args.chunk_size != 256:
        logger.warning("--chunk-size is ignored when --mamba-version 1")

    import random

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Load a small slice of data (first args.profile_seqs sequences)
    data_dir = Path(args.data_dir)
    from synthetic_data import load_dataset as _load

    train_seqs = _load(data_dir / "train.jsonl")[: args.profile_seqs]
    val_seqs = _load(data_dir / "val.jsonl")[: max(4, args.profile_seqs // 5)]
    logger.info(
        f"[profile] {len(train_seqs)} train / {len(val_seqs)} val seqs, "
        f"model={args.model}, device={device}"
    )

    true_durations: dict[int, list[int]] = {c: [] for c in range(NUM_CLASSES)}
    for s in train_seqs + val_seqs:
        for c, lens in s.true_durations.items():
            true_durations[c].extend(lens)

    # ---- dataset + dataloader ------------------------------------------------
    t0 = time.perf_counter()
    train_ds = SyntheticGenomicsDataset(
        train_seqs,
        features=args.features,
        cache_mode=args.cache_mode,
    )
    val_ds = SyntheticGenomicsDataset(
        val_seqs,
        features=args.features,
        cache_mode=args.cache_mode,
    )
    t_encode = time.perf_counter() - t0

    _pin = device.type == "cuda"
    _nw = args.num_workers
    _pf = 4 if _nw > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_nw > 0,
        prefetch_factor=_pf,
    )

    # ---- model ---------------------------------------------------------------
    _smoke_test_mamba(
        device, required_version=args.mamba_version if args.encoder == "mamba" else None
    )
    model = create_model(
        model_type=args.model,
        encoder_type=args.encoder,
        input_dim=train_ds.feature_dim,
        max_duration=args.max_duration,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        d_state=args.d_state,
        use_checkpoint=args.checkpoint,
        mamba_version=args.mamba_version,
        chunk_size=args.chunk_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _amp = not args.no_amp and device.type == "cuda"

    phase_times: dict[str, float] = {
        "encode_cache_s": t_encode,
        "dataloader_wait_s": 0.0,
        "h2d_copy_s": 0.0,
        "forward_s": 0.0,
        "backward_s": 0.0,
        "decode_s": 0.0,
        "metric_compute_s": 0.0,
    }

    for _epoch in range(args.profile_epochs):
        model.train()
        _iter = iter(train_loader)
        for _ in range(len(train_loader)):
            _t = time.perf_counter()
            batch = next(_iter)
            phase_times["dataloader_wait_s"] += time.perf_counter() - _t

            _t = time.perf_counter()
            seq = batch["sequence"].to(device, non_blocking=True)
            labs = batch["labels"].to(device, non_blocking=True)
            lens = batch["lengths"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            phase_times["h2d_copy_s"] += time.perf_counter() - _t

            labs_clean = labs.clone()
            labs_clean[labs_clean == -100] = 0
            optimizer.zero_grad(set_to_none=True)

            _t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_amp):
                loss = model.compute_loss(seq, lens, labs_clean)
            if device.type == "cuda":
                torch.cuda.synchronize()
            phase_times["forward_s"] += time.perf_counter() - _t

            _t = time.perf_counter()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            phase_times["backward_s"] += time.perf_counter() - _t

            optimizer.step()

        # One val decode pass to time decode + metrics
        val_loader_p = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=_nw,
            pin_memory=_pin,
            persistent_workers=False,
            prefetch_factor=_pf,
        )
        model.eval()
        all_preds, all_tgts, all_pred_segs, all_true_segs = [], [], [], []
        with torch.inference_mode():
            for batch in val_loader_p:
                seq = batch["sequence"].to(device, non_blocking=True)
                lens = batch["lengths"].to(device, non_blocking=True)
                labels_cpu = batch["labels"].numpy()

                _t = time.perf_counter()
                result = model.decode(seq, lens)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                phase_times["decode_s"] += time.perf_counter() - _t

                lengths_np = lens.cpu().numpy()
                is_softmax = isinstance(model, SoftmaxModel)
                result_np = result.cpu().numpy() if is_softmax else None
                for i in range(len(lengths_np)):
                    sl = int(lengths_np[i])
                    tl = labels_cpu[i, :sl]
                    if is_softmax:
                        pl = result_np[i, :sl]
                    else:
                        pl = np.zeros(sl, dtype=np.int64)
                        for seg in result.segments[i]:
                            pl[seg.start : seg.end + 1] = seg.label
                    all_preds.append(pl)
                    all_tgts.append(tl)
                    all_pred_segs.append(extract_segments(pl))
                    all_true_segs.append(extract_segments(tl))

        _t = time.perf_counter()
        compute_position_metrics(all_preds, all_tgts)
        compute_boundary_metrics(all_preds, all_tgts)
        compute_segment_metrics(all_pred_segs, all_true_segs)
        compute_duration_kl(all_pred_segs, true_durations, max_duration=args.max_duration)
        phase_times["metric_compute_s"] += time.perf_counter() - _t

    # ---- report --------------------------------------------------------------
    total = sum(v for k, v in phase_times.items() if k != "encode_cache_s")
    _enc_tag = (
        f"mamba{args.mamba_version}, d_state={args.d_state}"
        + (f", chunk_size={args.chunk_size}" if args.mamba_version == 2 else "")
        if args.encoder == "mamba"
        else args.encoder
    )
    print(f"\n{'─'*55}")
    print(
        f"  Profile: model={args.model}  encoder={_enc_tag}  "
        f"epochs={args.profile_epochs}  seqs={len(train_seqs)}"
    )
    print(f"{'─'*55}")
    print(f"  {'Phase':<28} {'Time (s)':>10}  {'%':>6}")
    print(f"{'─'*55}")
    print(f"  {'encode_cache (one-time)':<28} {phase_times['encode_cache_s']:>10.3f}  {'n/a':>6}")
    for phase in [
        "dataloader_wait_s",
        "h2d_copy_s",
        "forward_s",
        "backward_s",
        "decode_s",
        "metric_compute_s",
    ]:
        t = phase_times[phase]
        label = phase.removesuffix("_s")
        pct = 100 * t / total if total > 0 else 0
        print(f"  {label:<28} {t:>10.3f}  {pct:>5.1f}%")
    print(f"{'─'*55}")
    print(f"  {'total (excl. cache)':<28} {total:>10.3f}")
    print(f"{'─'*55}\n")


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
    run_parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
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
    run_parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="DataLoader worker processes per loader (default: min(8, cpu_count)). "
        "Set 0 to disable async loading.",
    )
    run_parser.add_argument(
        "--cache-mode",
        choices=["ram", "memmap", "none"],
        default="ram",
        help="Feature-encoding cache strategy (default: ram). "
        "Use 'memmap' for kmer features or when RAM is limited (avoids per-worker duplication). "
        "Use 'none' to encode on-the-fly (original behaviour).",
    )
    run_parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable BF16 autocast (default: AMP enabled on CUDA). "
        "Use when debugging NaNs or if the GPU does not support BF16.",
    )
    run_parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Enable Python-level gradient checkpointing on Mamba layers (default: off). "
        "mamba-ssm already performs kernel-level recomputation internally "
        "(checkpoint_lvl=1), so this flag causes double recomputation. "
        "Only use if you are OOM after exhausting --batch-size / --grad-accum options.",
    )
    run_parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1 = no accumulation). "
        "Effective batch = batch_size × grad_accum_steps. "
        "Use when physical batch is limited by VRAM at large T. "
        "Example: --batch-size 8 --grad-accum 16 → effective B=128.",
    )
    run_parser.add_argument(
        "--mamba-version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Mamba version (default: 2). "
        "2 = Mamba2 (SSD chunked scan, better memory at large T; requires mamba-ssm >= 2.0). "
        "1 = Mamba (selective scan, checkpoint_lvl=1 kernel recomputation).",
    )
    run_parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Mamba2 SSD chunk size (ignored for --mamba-version 1; default: 256). "
        "Smaller values reduce peak activation memory at the cost of more recomputation. "
        "Try 64–128 if OOM at large T.",
    )
    run_parser.add_argument(
        "--d-state",
        type=int,
        default=16,
        help="SSM state dimension (default: 16). "
        "Mamba2's natural default is 128 (better long-range modeling, ~8× more SSM memory). "
        "Keep at 16 for direct Mamba1 comparison.",
    )

    # Profile command
    prof_parser = subparsers.add_parser(
        "profile",
        help="Quick per-phase profiling (one model, few epochs, fixed seed)",
    )
    prof_parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    prof_parser.add_argument(
        "--model",
        choices=["softmax", "pytorch-crf", "linear", "semicrf", "semicrf_uniform"],
        default="semicrf",
        help="Model to profile",
    )
    prof_parser.add_argument(
        "--encoder", choices=["bilstm", "mamba", "mamba_stub"], default="mamba"
    )
    prof_parser.add_argument("--features", choices=["onehot", "kmer"], default="onehot")
    prof_parser.add_argument("--max-duration", type=int, default=1000)
    prof_parser.add_argument("--hidden-dim", type=int, default=256)
    prof_parser.add_argument("--num-layers", type=int, default=4)
    prof_parser.add_argument("--batch-size", type=int, default=32)
    prof_parser.add_argument("--profile-epochs", type=int, default=2, help="Epochs to profile")
    prof_parser.add_argument(
        "--profile-seqs", type=int, default=20, help="Training sequences to use"
    )
    prof_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    prof_parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    prof_parser.add_argument("--cache-mode", choices=["ram", "memmap", "none"], default="ram")
    prof_parser.add_argument("--no-amp", action="store_true")
    prof_parser.add_argument("--checkpoint", action="store_true")
    prof_parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1).",
    )
    prof_parser.add_argument(
        "--mamba-version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Mamba version (default: 2). 2 = Mamba2, 1 = Mamba.",
    )
    prof_parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Mamba2 SSD chunk size (ignored for --mamba-version 1; default: 256).",
    )
    prof_parser.add_argument(
        "--d-state",
        type=int,
        default=16,
        help="SSM state dimension (default: 16).",
    )
    prof_parser.add_argument("--seed", type=int, default=42)

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare results")
    cmp_parser.add_argument("--results-dir", type=Path, required=True, help="Results directory")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()


