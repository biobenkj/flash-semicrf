# FAQ Part 4: Engineering Decisions

### Q: Why not just use sparse matrices?

This is a natural thought: the segment duration constraint K means most entries in the full transition matrix are zero, so sparse representations should help. In practice, two problems defeat this:

1. **The sparsity pattern is wrong.** The feasible combinations of left/right partial durations satisfy a sum constraint (d₁ + d₂ ≤ S), which produces anti-diagonal triangular sparsity — not the diagonal-banded pattern that standard sparse libraries exploit.

2. **Sparsity fills in under composition.** Even if the one-step operator is sparse, repeated composition in the binary tree algorithm rapidly expands the support. After just O(log(T/K)) levels, the matrices are nearly dense.

The supplementary note *"Why Banded Matrices Do Not Provide a Viable Exact Backend"* proves this formally with bandwidth lower bounds and empirical measurements showing bandwidth ratios > 0.90 for spans S > K/2.

### Q: Why Triton instead of CUDA?

Triton provides Python-level syntax with auto-tuned memory access patterns, which dramatically reduces development time compared to raw CUDA while achieving comparable performance. For the semi-CRF use case, where the kernel logic is complex (nested loops, conditional loads, online reductions), Triton's productivity advantage outweighs any small performance gap. The Triton kernel also compiles to PTX, so it runs on standard NVIDIA GPUs without additional dependencies.

### Q: Why is the backward pass non-deterministic?

The Triton backward kernel uses `atomic_add` to accumulate gradients for shared parameters (transition matrix, duration bias) from multiple positions simultaneously. Because floating-point addition is not associative, different thread execution orders produce slightly different results. The magnitude is small:

| Gradient | Typical Variance Across Runs |
|----------|------------------------------|
| `grad_transition` | ~1–2% relative difference |
| `grad_duration_bias` | ~7–14% relative difference |
| `grad_cum_scores` | Position-dependent, lower variance |

This is within normal SGD noise, so training converges correctly. If exact reproducibility is required, a two-phase kernel approach is documented but not yet implemented.

To reduce this variance, the library allocates **per-segment workspace buffers** so that gradient accumulation across checkpoint segments is deterministic. Only within-segment atomic contention remains.

### Q: Why does the backward pass use float64?

Gradient accumulation for shared parameters sums contributions from every (position, duration, label) combination in the sequence. For T=100,000, K=1,000, C=24, that's billions of small floating-point additions. In float32, the accumulated rounding error can reach ~10⁻³, which is noticeable. Float64 reduces this to ~10⁻¹⁰, which is negligible. The forward pass remains in float32 because checkpoint normalization handles the stability there.

### Q: How does checkpoint normalization work?

Inspired by FlashAttention, the forward pass periodically extracts a normalization factor from the forward messages (the maximum log-value) and subtracts it, preventing the messages from growing without bound. The extracted factors are saved alongside the ring buffer checkpoints. During the backward pass, these factors are added back when computing marginals, restoring the correct scale. This is what enables stable inference at T > 100,000.

### Q: Why is the edge tensor decomposed into cumulative scores + transition + duration bias?

The full edge tensor has shape (batch, T, K, C, C) and stores one score for every possible segment at every position. For genomic sequences, this tensor alone exceeds available GPU memory. The decomposition factors it into three much smaller pieces:

| Component | Shape | Size (T=50K, K=100, C=8) |
|-----------|-------|--------------------------|
| Cumulative scores | (batch, T+1, C) | 1.6 MB |
| Transition matrix | (C, C) | 256 bytes |
| Duration bias | (K, C) | 3.2 KB |
| **Total** | | **~1.6 MB** |
| Full edge tensor | (batch, T, K, C, C) | **320 GB** |

The emission score for any segment can be recovered in O(1) by subtracting two cumulative-score entries. This is the mathematical foundation of the streaming approach.

### Q: What is the memory profile for realistic genomics use cases?

| Component | Standard Approach | Streaming Approach |
|-----------|-------------------|--------------------|
| Edge tensor | 320 GB (not feasible) | Not materialized |
| Cumulative scores | 1.6 MB | 1.6 MB |
| Ring buffer (forward) | — | 6.4 KB |
| Ring buffer (backward) | — | 12.8 KB |
| Checkpoints | — | ~50 KB |
| **Total working memory** | **320+ GB** | **< 2 MB** |

*(Example: T=50,000, K=100, C=8, batch=1)*

### Q: Why does K=1 use a different code path?

The ring buffer and checkpoint machinery are designed for K≥3, where there are enough history slots to justify the complexity. At K=1, the ring buffer degenerates to a single slot, and at K=2, modular arithmetic with two slots is fragile. The specialized K=1 and K=2 paths use simple variables instead, which is both faster and less error-prone. The results are numerically identical to the general path.

| K | Backend | Ring Buffer | Checkpointing |
|---|---------|-------------|---------------|
| 1 | PyTorch (LinearCRFStreaming) | No | No |
| 2 | PyTorch (SemiCRFK2Streaming) | No | No |
| ≥3 | Triton kernel or PyTorch fallback | Yes (size K / 2K) | Yes |

### Q: How does this compare to pytorch-crf for linear CRF tasks?

At K=1, torch-semimarkov is functionally equivalent to a linear-chain CRF like pytorch-crf. The two implementations produce the same accuracy on identical data. They define slightly different probability models (different handling of the first position), so their NLL values are not directly comparable, but this difference has no practical impact on prediction quality.

The advantage of using torch-semimarkov even for K=1 is a clean upgrade path: when you later need K>1, the API stays the same. See the [Linear CRF Equivalence](../linear_crf_equivalence.md) document for the formal comparison and empirical validation.

### Q: What are the complexity characteristics of each backend?

| Backend | Time | Memory | Depth | Notes |
|---------|------|--------|-------|-------|
| Binary tree (dense) | O(T·n³) | O(T·n²) | O(log T) | n = (K−1)·C; only viable for n < 100 |
| Binary tree (sharded) | O(T·n³) | O(√T·n²) | O(√T) | n < 150 |
| Linear scan (vectorized) | O(TKC²) | O(TKC²)* | O(T) | *Dominated by edge tensor |
| Streaming (Triton) | O(TKC²) | **O(KC)** | O(T) | Genome-scale; edges on-the-fly |

The streaming kernel achieves the same O(TKC²) compute as the linear scan but with memory that is independent of sequence length — the key property enabling genome-scale inference.

---

**See also:** [Jargon Decoder](03-jargon-decoder.md) · [Troubleshooting](05-troubleshooting.md) · [Backend algorithm supplement](../backend_algorithm_supplement.pdf) · [Streaming algorithm supplement](../streaming_algorithm_supplement.pdf)
