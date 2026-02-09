# FAQ Part 2: Using the Library

## Parameters

### Q: What do T, K, and C mean?

**T** is the sequence length in positions (base pairs, time steps, tokens, etc.). **K** is the maximum segment duration — the longest single segment the model can produce without splitting. **C** is the number of segment labels (your annotation vocabulary). These three parameters control everything about cost and memory.

| Parameter | Genomics Example | Practical Advice |
|-----------|-----------------|------------------|
| T (sequence length) | 10,000–100,000+ bp per chunk | Larger T = more context but more compute. Use chunked processing for genomes. |
| K (max duration) | 500–3,000 (exon/intron lengths) | Pick from the 95th–99th percentile of your observed segment lengths, not the absolute max. |
| C (label count) | 3–30 (exon, intron, UTR, etc.) | More labels = finer annotation but quadratic cost in transitions (C²). |

### Q: How do I choose K?

K should cover the vast majority of segments in your data without being wasteful. If 99% of exons are shorter than 500 bp, setting K=500 is reasonable. Setting K=50,000 because one intron is that long wastes memory and compute. Instead, cap long segments and let the model produce multiple consecutive segments of the same label. The library's duration bias parameters can encode that splitting behavior.

### Q: Which duration distribution should I use?

Start with `"learned"` (the default). It allocates K × C free parameters and can represent any duration pattern. Switch to a parametric distribution (`"geometric"`, `"poisson"`, `"negative_binomial"`) if you have limited training data relative to K × C, or if you have domain knowledge about how segment lengths behave. Important observation: when K is large (say K=3000 for introns), `"learned"` has thousands of parameters per class for rare long durations that may get little gradient signal — a parametric form acts as a regularizer.

| Scenario | Recommended distribution |
| -------- | ----------------------- |
| Plenty of data, no strong prior | `"learned"` (default) |
| Short segments dominate, exponential decay | `"geometric"` |
| Each class has a characteristic length | `"poisson"` |
| Peaked lengths with varying spread per class | `"negative_binomial"` |
| Ablation baseline (duration shouldn't matter) | `"uniform"` |
| External histograms or domain priors | `CallableDuration` |

See the [parameter guide](../guides/parameter_guide.md#choosing-a-duration-distribution) for a full decision flowchart and code examples.

### Q: What are semirings and why are there so many?

A semiring is a way to swap out the arithmetic in the dynamic program without rewriting the algorithm. The **LogSemiring** computes the partition function (for training). The **MaxSemiring** finds the best path (for prediction). The **EntropySemiring** measures uncertainty. The **KMaxSemiring** gives you the top-k segmentations. Same code, different questions answered.

| Semiring | What It Computes | When to Use |
|----------|-----------------|-------------|
| `LogSemiring` | Partition function (log Z), marginals | Training (NLL loss), getting marginal probabilities |
| `MaxSemiring` | Best path score (Viterbi) | Inference / prediction time |
| `KMaxSemiring(k=N)` | Top-N path scores | Multiple hypotheses, N-best lists |
| `EntropySemiring` | Entropy H(P) of the segmentation distribution | Uncertainty quantification, active learning |
| `CrossEntropySemiring` | Cross-entropy H(P,Q) between two distributions | Model distillation, comparing two models |

## Backends

### Q: What is a "backend" and which one should I use?

A backend is the algorithm that runs the dynamic program. The library offers several, but the two that matter in practice are: (1) the **streaming Triton kernel** for sequences longer than about 10,000 positions, and (2) the **vectorized linear scan** for shorter sequences or CPU-only environments. The `backend="auto"` setting picks the right one automatically.

| Backend | Memory | Best For |
|---------|--------|----------|
| Streaming (Triton) | O(KC) — independent of T | Genome-scale sequences (T > 10K), GPU required |
| Vectorized linear scan | O(TKC²) — pre-materialized edges | Short–moderate sequences, CPU or GPU |
| Binary tree (parallel scan) | O(T·(KC)²) | Small state spaces only; mostly historical |

### Q: What does "streaming" mean in this context?

In the standard approach, you first build a giant tensor of all edge potentials (shape: batch × T × K × C × C) and then run the DP over it. For genomic-scale sequences, this tensor alone can exceed 300 GB. The streaming approach never builds that tensor. Instead, it computes each edge potential on-the-fly from compact cumulative scores as the DP scan walks through the sequence, using a ring buffer to store only the K most recent forward messages. The result is that memory is independent of sequence length.

### Q: Why is K=1 handled separately?

When K=1, every segment has duration 1, and the semi-CRF reduces to a standard linear-chain CRF. The library detects this and dispatches to an optimized path that skips the ring buffer, duration loop, and checkpoint machinery. Similarly, K=2 gets its own fast path. For K≥3, the full streaming architecture kicks in.

## Workflows

### Q: How does torch-semimarkov fit into my model?

Think of it as a structured prediction layer that sits on top of your encoder (transformer, Mamba, CNN, BiLSTM, etc.). Your encoder produces hidden states for each position; a small projection head converts those into semi-CRF parameters; and torch-semimarkov runs the DP to compute losses and predictions. The library handles the inference; you handle the encoder and the data.

```
Input sequence (batch, T, input_dim)
        │
        ▼
┌────────────────────┐
│      Encoder       │   BERT, Mamba, CNN, BiLSTM, etc.
│ (any architecture) │
└────────────────────┘
        │
        ▼
Hidden states (batch, T, hidden_dim)
        │
        ▼
┌───────────────────┐
│  Projection head  │   Linear layers → semi-CRF parameters
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    SemiMarkov     │   Structured inference (this library)
└───────────────────┘
        │
        ▼
Log partition / Viterbi path / Marginals
```

### Q: How do I train a model with this?

The training loss is the standard CRF negative log-likelihood: `loss = log_Z - gold_score`. You compute `log_Z` with the LogSemiring, score the gold segmentation by summing the edge potentials along the labeled path, and backpropagate. The library provides autograd-compatible functions for all of this, including through the Triton kernel.

### Q: How do I decode (get predictions)?

Use the MaxSemiring. The "marginals" under MaxSemiring become hard 0/1 indicators for the best path. Call `SemiMarkov.from_parts(hard_marginals)` to extract the predicted label sequence and segment boundaries.

---

**See also:** [The Basics](01-basics.md) · [Engineering Decisions](04-engineering-decisions.md) · [Parameter guide](../parameter_guide.md) · [Semirings guide](../semirings.md) · [Workflow integration](../workflow_integration.md)
