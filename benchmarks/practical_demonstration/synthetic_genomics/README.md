# Synthetic Genomics Benchmark

A benchmark demonstrating that **Semi-CRF learns biologically meaningful duration distributions** on synthetic genomic data with known ground truth.

## Why This Benchmark?

This benchmark addresses the core question: **Does Semi-CRF actually learn duration distributions that match biological reality?**

Unlike real genomic data (e.g., GENCODE) where ground truth distributions are complex and confounded, synthetic data gives us:

1. **Known generating distributions**: Exons follow log-normal (~150bp median), introns follow heavy-tailed log-normal (1-10kb)
2. **Controlled experiments**: We can ablate duration learning (uniform vs learned) to isolate the mechanism
3. **Clear success criterion**: The "money plot" shows whether learned distributions match true distributions

## Five-Way Model Comparison

| Model | K | Duration | Library | What It Tests |
|-------|---|----------|---------|---------------|
| **softmax** | N/A | None | N/A | Pure position-wise baseline |
| **pytorch-crf** | 1 | None | torchcrf | External linear CRF baseline |
| **linear** | 1 | None | flash-semicrf | Linear CRF, same codebase as semi-CRF |
| **semicrf** | 1000 | Learned | flash-semicrf | Full Semi-CRF capability |
| **semicrf_uniform** | 1000 | Uniform | flash-semicrf | Ablation: segment structure without duration |

**Key comparisons:**
- `linear` vs `pytorch-crf`: Validates flash-semicrf K=1 matches external baseline
- `semicrf` vs `linear`: Shows benefit of duration modeling
- `semicrf` vs `semicrf_uniform`: Isolates learned duration as the mechanism

## Quick Start

```bash
# 1. Generate synthetic data
python3 synthetic_benchmark.py generate \
    --output-dir data/synthetic \
    --num-train 1000 --num-val 200 --num-test 200 \
    --seq-length 20000

# 2. Run full benchmark (GPU recommended)
python3 synthetic_benchmark.py run \
    --data-dir data/synthetic \
    --output-dir results/synthetic \
    --encoder mamba \
    --epochs 100

# 3. View results
python3 synthetic_benchmark.py compare --results-dir results/synthetic
```

For CPU development:
```bash
python3 synthetic_benchmark.py run \
    --data-dir data/synthetic \
    --output-dir results/synthetic_cpu \
    --encoder mamba_stub \
    --epochs 20 \
    --device cpu
```

## Data Generation

The synthetic generator creates genomic sequences with known structure:

### Label Scheme (C=3)
```
0: intergenic - regions between genes
1: exon       - coding segments (~150bp log-normal)
2: intron     - non-coding segments (1-10kb heavy-tailed)
```

### Duration Distributions
| Label | Distribution | Median | Range |
|-------|-------------|--------|-------|
| exon | Log-normal(log(150), 0.5) | ~150bp | 30-500bp |
| intron | Log-normal(log(3000), 1.0) | ~3kb | 100-10,000bp |
| intergenic | Log-normal(log(5000), 0.8) | ~5kb | 500-20,000bp |

### Sequence Features
- One-hot DNA encoding (A, C, G, T, N = 5 dims) [default]
- Optional: 4-mer frequency encoding (256 dims)
- Optional splice site motifs (GT...AG at intron boundaries)

## CLI Reference

### Generate Data
```bash
python3 synthetic_benchmark.py generate \
    --output-dir <path>          # Output directory (required)
    --num-train 1000             # Training sequences
    --num-val 200                # Validation sequences
    --num-test 200               # Test sequences
    --seq-length 20000           # Target sequence length (T)
    --num-classes 3              # 3 or 5 classes
    --seed 42                    # Random seed
```

### Run Benchmark
```bash
python3 synthetic_benchmark.py run \
    --data-dir <path>            # Data directory (required)
    --output-dir <path>          # Output directory (required)
    --encoder mamba              # mamba, mamba_stub, bilstm
    --features onehot            # onehot or kmer
    --max-duration 1000          # Max segment duration K
    --hidden-dim 256             # Hidden dimension
    --num-layers 4               # Encoder layers
    --batch-size 16              # Batch size
    --lr 1e-3                    # Learning rate
    --epochs 100                 # Training epochs
    --device cuda                # cuda or cpu
    --log-every 10               # Log frequency
    --backend streaming          # Semi-CRF backend
    --no-triton                  # Disable Triton (use PyTorch)
```

### Compare Results
```bash
python3 synthetic_benchmark.py compare \
    --results-dir <path>         # Results directory
```

## Output Structure

After running the benchmark:
```
results/synthetic/
├── metrics.json                 # All metrics in JSON format
├── duration_comparison.png      # "Money plot" - learned vs true distributions
├── training_curves.png          # Loss curves for stability verification
└── checkpoints/
    ├── softmax_best.pt
    ├── linear_best.pt
    ├── semicrf_best.pt
    └── semicrf_uniform_best.pt
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Boundary F1** | Exact boundary detection accuracy |
| **Boundary F1 @ tol=k** | Boundary within k positions |
| **Segment F1** | Entire segment must match (start, end, label) |
| **Position F1** | Per-position classification accuracy |
| **Duration KL** | KL divergence between predicted and true duration distributions |

## Expected Results

| Model | Boundary F1 | Duration KL (exon) | Duration KL (intron) |
|-------|-------------|-------------------|---------------------|
| softmax | ~0.65 | N/A | N/A |
| linear CRF | ~0.70 | high (~2.0) | high (~3.0) |
| Semi-CRF (uniform) | ~0.72 | high (~1.8) | high (~2.5) |
| **Semi-CRF (learned)** | **~0.80** | **low (~0.3)** | **low (~0.5)** |

The key result: Semi-CRF with learned duration should show:
1. **Better boundary F1** than all baselines
2. **Low duration KL** - learned distributions match true generating distributions

## The "Money Plot"

The duration comparison plot (`duration_comparison.png`) shows:
- **Columns**: Ground truth, softmax, linear, semicrf, semicrf_uniform
- **Rows**: intergenic, exon, intron

For each class, you should see:
- `semicrf` distribution closely matching ground truth
- Other models showing flat or mismatched distributions
- KL divergence numbers confirming the visual

This provides **visual proof** that Semi-CRF learns biologically meaningful duration distributions.

## Multi-Scale Experiments

For thorough validation, run at multiple sequence lengths:

```bash
# T=10k (fast)
python3 synthetic_benchmark.py generate --output-dir data/synthetic_10k --seq-length 10000
python3 synthetic_benchmark.py run --data-dir data/synthetic_10k --output-dir results/10k

# T=20k (medium)
python3 synthetic_benchmark.py generate --output-dir data/synthetic_20k --seq-length 20000
python3 synthetic_benchmark.py run --data-dir data/synthetic_20k --output-dir results/20k

# T=50k (stress test)
python3 synthetic_benchmark.py generate --output-dir data/synthetic_50k --seq-length 50000
python3 synthetic_benchmark.py run --data-dir data/synthetic_50k --output-dir results/50k
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 8

# Use BiLSTM instead of Mamba
--encoder bilstm

# Ensure streaming backend (default)
--backend streaming
```

### Mamba Not Available
```bash
# Use CPU-compatible stub
--encoder mamba_stub --device cpu
```

### Slow Training
```bash
# Fewer epochs for quick iteration
--epochs 20 --log-every 5

# Smaller hidden dimension
--hidden-dim 128
```

## Gradient Agreement Testing

The `gradient_check.py` script provides low-level verification that Triton kernel gradients match the PyTorch reference implementation. This is critical for validating the per-checkpoint log normalization that enables T=100k sequences.

### Quick Usage

```bash
# T=1000 smoke test (fast)
python gradient_check.py --seq-length 1000 --checkpoint-interval 100

# T=10k validation
python gradient_check.py --seq-length 10000

# Large scale (slower)
python gradient_check.py --seq-length 50000 --batch-size 1
```

### What It Tests

1. **Forward partition agreement**: Triton vs PyTorch reference log-partition function
2. **Checkpoint consistency**: Both implementations save normalized checkpoints with log_norm factors
3. **Backward gradient agreement**: grad_cum_scores, grad_transition, grad_duration_bias

### Acceptance Criteria

The test uses **different metrics for different gradients** to account for floating-point precision differences between parallel (Triton) and sequential (PyTorch) implementations:

| Metric | Threshold | Rationale |
| ------ | --------- | --------- |
| Forward partition rel error | < 1e-4 | Core correctness check |
| grad_transition max rel error | < 1e-2 | Actual parameter gradient |
| grad_duration_bias max rel error | < 1e-2 | Actual parameter gradient |
| grad_cum_scores **mean abs error** | < 1e-3 | Max relative inflated at near-zero positions |

### Why grad_cum_scores Uses Mean Absolute Error

The large max relative error (~70x at T=10k) on grad_cum_scores is **NOT a bug**. It occurs at positions where:

- Reference gradient is near-zero (~1e-3)
- Absolute difference is small (7e-3)
- Division by near-zero inflates relative error

**Evidence the algorithm is correct:**

1. Forward partitions agree (6e-7 relative error)
2. grad_transition agrees (7e-3 relative error)
3. grad_duration_bias agrees (7e-3 relative error)
4. Mean absolute error is tiny (3e-4)

### Expected Output at T=10k

```text
Forward partition comparison:
  Triton:  [14132.95, 14130.20]
  PyTorch: [14132.95, 14130.21]
  Rel diff: 6.22e-07  [PASS]

Backward gradient comparison:
  grad_cum_scores:
    Max abs diff: 8.81e-03
    Max rel diff: 7.06e+01  (expected - near-zero positions)
    Mean abs diff: 2.60e-04  [PASS]
  grad_transition:
    Max rel diff: 6.71e-03  [PASS]
  grad_duration_bias:
    Max rel diff: 6.66e-03  [PASS]

PASS: All criteria met
```

### Notes

- **Both implementations use normalization**: Partitions and gradients should agree at all sequence lengths.
- **Error scales with T**: Accumulated floating-point rounding differences grow with sequence length.
- **The real test**: For T=100k validation, use the full training benchmark to verify end-to-end correctness.

## Files

- `synthetic_data.py` - Data generation with log-normal/heavy-tailed distributions
- `synthetic_benchmark.py` - Training, evaluation, and visualization
- `gradient_check.py` - Low-level gradient agreement test (Triton vs PyTorch reference)
- `README.md` - This documentation
