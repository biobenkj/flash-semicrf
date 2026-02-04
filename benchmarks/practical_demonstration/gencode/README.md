# Gencode Exon/Intron Segmentation Benchmark

This benchmark demonstrates where **Semi-Markov CRFs outperform linear CRFs**: segmenting genomic sequences into exons, introns, UTRs, and codons where segment durations carry strong biological signal.

## Why Semi-CRF for Genomics?

Genomic features have characteristic length distributions that a linear CRF cannot model:

| Feature | Typical Length | Why Duration Matters |
|---------|---------------|---------------------|
| Start/stop codons | **exactly 3bp** | Perfect for Semi-CRF |
| First exons | ~100bp | Shorter than internal |
| Internal exons | ~150bp | Standard length |
| Last exons | ~200bp | Often longer |
| First introns | ~2kb | Longest introns |
| Internal introns | ~1kb | Standard length |
| Last introns | ~500bp | Shortest introns |

A **linear CRF (K=1)** treats each position independently - it cannot encode "this state tends to last N positions."

A **Semi-CRF (K>1)** explicitly models segment durations, penalizing implausible lengths (e.g., a 1000bp codon or a 2bp intron).

---

## Quick Start

```bash
# 1. Preprocess Gencode annotations
python gencode_exon_intron.py preprocess \
    --gtf gencode.v44.annotation.gtf.gz \
    --fasta GRCh38.primary_assembly.genome.fa \
    --output-dir data/gencode/

# 2. Train Semi-CRF with Mamba encoder
python gencode_exon_intron.py train \
    --data-dir data/gencode/ \
    --model semicrf \
    --encoder mamba \
    --max-duration 500

# 3. Compare all CRF types
python gencode_exon_intron.py compare \
    --data-dir data/gencode/ \
    --encoder mamba
```

---

## Label Scheme (11 classes)

```
0: intergenic       - regions between genes
1: first_exon       - first coding exon (~100bp)
2: internal_exon    - middle coding exons (~150bp)
3: last_exon        - final coding exon (~200bp)
4: first_intron     - first intron (~2kb, longest)
5: internal_intron  - middle introns (~1kb)
6: last_intron      - final intron (~500bp, shortest)
7: 5UTR             - 5' untranslated region (~150bp)
8: 3UTR             - 3' untranslated region (~500bp)
9: start_codon      - ATG start codon (exactly 3bp)
10: stop_codon      - TAA/TAG/TGA stop codon (exactly 3bp)
```

The position-based exon/intron labels (first/internal/last) highlight duration differences that Semi-CRF can exploit.

---

## CLI Reference

### Preprocessing

```bash
python gencode_exon_intron.py preprocess \
    --gtf <path>              # Gencode GTF file (required)
    --fasta <path>            # Reference genome FASTA (required)
    --output-dir <path>       # Output directory (required)
    --chunk-size 10000        # Chunk size in bp (default: 10000)
    --overlap 500             # Overlap between chunks (default: 500)
    --train-chroms chr1,chr2  # Training chromosomes (default: chr1-18)
    --val-chroms chr19        # Validation chromosomes (default: chr19-20)
    --test-chroms chr21       # Test chromosomes (default: chr21-22)
```

**Tip**: For quick debugging, use a single chromosome:
```bash
python gencode_exon_intron.py preprocess \
    --gtf ... --fasta ... --output-dir data/debug/ \
    --train-chroms chr1 --val-chroms chr19 --test-chroms chr21
```

### Training

```bash
python gencode_exon_intron.py train \
    --data-dir <path>         # Preprocessed data directory (required)
    --model <type>            # Model type (see below)
    --encoder <type>          # Encoder type (see below)
    --max-duration 500        # Max segment duration K (default: 500)
    --hidden-dim 256          # Hidden dimension (default: 256)
    --num-layers 2            # Encoder layers (default: 2, use 4+ for Mamba)
    --epochs 50               # Training epochs (default: 50)
    --batch-size 32           # Batch size (default: 32)
    --lr 1e-3                 # Learning rate (default: 1e-3)
    --weight-decay 1e-5       # AdamW weight decay (default: 1e-5)
    --crf-reg 0.0             # CRF parameter regularization (default: 0)
    --emission-clamp 0.0      # Clamp encoder outputs (default: 0, disabled)
    --backend streaming       # Backend: streaming, exact, auto (default: streaming)
    --no-triton               # Disable Triton kernels (use PyTorch fallback)
    --log-every 1             # Evaluate every N epochs (default: 1)
    --device cuda             # Device (default: cuda if available)
```

**Model types (`--model`):**
| Type | Description |
|------|-------------|
| `pytorch-crf` | External pytorch-crf baseline (requires `pip install pytorch-crf`) |
| `linear` | torch-semimarkov with K=1 (linear CRF, no duration modeling) |
| `semicrf` | torch-semimarkov with K>1 (full Semi-CRF with duration modeling) |

**Encoder types (`--encoder`):**
| Type | Description |
|------|-------------|
| `bilstm` | Bidirectional LSTM (default, works on CPU/GPU) |
| `mamba` | Mamba SSM encoder (requires `pip install mamba-ssm`, GPU only) |
| `mamba_stub` | CPU-compatible Mamba approximation (for development/testing) |

**Mamba-specific options:**
```bash
--d-state 16    # SSM state dimension (default: 16)
--d-conv 4      # Local convolution width (default: 4)
--expand 2      # Expansion factor (default: 2)
```

**Distributed training:**
```bash
python gencode_exon_intron.py train \
    --data-dir data/gencode/ \
    --distributed \
    --world-size 4 \
    --nccl-ifname ib0  # Network interface for NCCL (optional)
```

### Comparing Models

```bash
python gencode_exon_intron.py compare \
    --data-dir <path>         # Preprocessed data directory (required)
    --encoder <type>          # Encoder for all models
    --max-duration 500        # K for Semi-CRF
    --output-json results.json  # Save results to JSON (optional)
```

This runs a 3-way comparison:
1. **pytorch-crf**: External linear CRF baseline
2. **linear (K=1)**: torch-semimarkov linear CRF
3. **semicrf (K>1)**: torch-semimarkov Semi-CRF

---

## Diagnostics and Intuition

### First-Batch Diagnostics

On the first batch of epoch 0, the training loop logs diagnostic information:

```
Encoder output stats (first batch): mean=0.0000, std=1.0000, min=-3.49, max=3.33, NaN=0, Inf=0
Emission scores (after CRF projection): mean=0.0168, std=0.6018, min=-2.51, max=2.73
CRF transition: mean=0.0054, std=0.0942, range=[-0.24, 0.22]
CRF duration_bias: mean=0.0004, std=0.0999, range=[-0.41, 0.43]
```

#### Understanding the Output

**Encoder output stats**: Raw output from the encoder (Mamba/BiLSTM).
- Mamba uses LayerNorm, so you'll see mean≈0, std≈1
- Check for NaN/Inf which indicates initialization problems
- Large values (|max| > 100) suggest using `--emission-clamp`

**Emission scores**: After projecting encoder output to label space via `crf.projection`.
- These are the "content scores" - how well does this sequence match each label?

**CRF transition/duration_bias**: The structured prediction parameters.
- `transition[i,j]` = log-score for transitioning from label i to label j
- `duration_bias[k,c]` = log-score for a segment of length k with label c

### Scale Balance: Why It Matters

The Semi-CRF computes segment scores as:

```
segment_score = emission + transition + duration_bias
```

**If emissions >> CRF parameters:**
```
emission = 50.0     # "this looks like an exon"
transition = 0.2    # "exon → intron is valid"
duration = 0.3      # "100bp exon is typical"
```
The emission dominates. The model ignores transitions and durations, becoming a position-wise classifier. You lose the benefit of Semi-CRF.

**If CRF parameters >> emissions:**
```
emission = 0.1      # "this looks like an exon"
transition = 15.0   # "exon → intron is valid"
duration = 20.0     # "100bp exon is typical"
```
The encoder's learned features barely matter. The model just uses prior transition/duration statistics.

**Balanced (ideal):**
```
emission = 2.5      # "this looks like an exon"
transition = 0.2    # "exon → intron is valid"
duration = 0.4      # "100bp exon is typical"
```
Both components contribute meaningfully.

The diagnostic will warn you if the scale ratio exceeds 10x:
```
WARNING: Emission scale (15.32) >> CRF scale (0.21). CRF transitions/durations may have minimal effect.
```

### What Scales Should I Expect?

**At initialization**: Controlled by design.
- CRF uses `init_scale=0.1`, so transition/duration start at ~0.1-0.4
- Emissions depend on projection layer init, typically ~1-3

**After training**: Dataset-dependent.

| Data characteristic | Effect |
|---------------------|--------|
| Strong duration signal (3bp codons) | duration_bias grows |
| Clear transition rules (exon→intron valid, exon→stop_codon rare) | transition learns large positive/negative values |
| Distinctive sequence content (splice site motifs) | emissions grow |
| Ambiguous content | emissions stay moderate |

For genomics, expect:
- **duration_bias** to grow significantly (durations are very informative)
- **transition** to learn forbidden paths (large negative values)
- **emissions** to be moderate (DNA is somewhat but not fully informative)

---

## Troubleshooting

### NaN in Backward Pass

If you see NaN during training at long sequences (T > 50k):

1. **Try emission clamping**:
   ```bash
   --emission-clamp 50
   ```

2. **Add CRF regularization**:
   ```bash
   --crf-reg 1e-4
   ```

3. **Check the float64 fix**: The Triton backward kernel uses `float64` for gradient accumulation. If you modified `triton_backward.py`, ensure `grad_cum_scores` is `torch.float64`.

### Out of Memory

The streaming backend is memory-efficient (O(TC) vs O(TKC²) for exact):

```bash
--backend streaming  # Default, use this for long sequences
```

Memory estimates are printed at startup:
```
Memory estimate (B=32, T=10000, K=500, C=11):
  Exact backend:     72.6GB (edge tensor)
  Streaming backend: 0.01GB (cumsum tensor)
```

### Triton Not Available

Use PyTorch fallback:
```bash
--no-triton
```

Or for Mamba without GPU:
```bash
--encoder mamba_stub
```

---

## Metrics

The benchmark reports:

| Metric | Description |
|--------|-------------|
| **Position F1** | Per-position label accuracy (macro-averaged) |
| **Boundary F1** | Exact boundary detection (where labels change) |
| **Boundary F1 @ tol=k** | Boundary within k positions |
| **Segment F1** | Entire segment must match (start, end, label) |
| **Duration KL** | KL divergence between predicted and true duration distributions (lower is better) |

Semi-CRF should significantly outperform linear CRF on:
- Segment F1 (correctly predicting whole segments)
- Duration KL (matching true duration distributions)
- Boundary F1 (precise boundary placement)

---

## Files

After preprocessing:
```
data/gencode/
├── train.jsonl              # Training chunks
├── val.jsonl                # Validation chunks
├── test.jsonl               # Test chunks
├── train_segment_stats.json # Duration statistics per label
├── val_segment_stats.json
└── test_segment_stats.json
```

Each JSONL line contains:
```json
{
  "chrom": "chr1",
  "start": 10000,
  "end": 20000,
  "sequence": "ATCG...",
  "labels": [0, 0, 1, 1, 1, ...]
}
```
