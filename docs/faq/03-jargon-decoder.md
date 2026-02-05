# FAQ Part 3: Jargon Decoder

This section explains systems and GPU terminology that appears throughout the documentation. You don't need to understand all of this to use the library, but it helps when reading the supplements or debugging performance.

## GPU Concepts

### Q: What is a warp?

A warp is a group of 32 threads on an NVIDIA GPU that execute the same instruction at the same time (SIMT: Single Instruction, Multiple Threads). When the documentation mentions "warp-level reduction," it means 32 threads are cooperating to compute a sum or maximum by sharing partial results. This is fast because it uses dedicated hardware shuffle instructions rather than going through shared memory.

**Why it matters:** when multiple warps write to the same memory location, the order of writes is non-deterministic. This is why the backward pass can produce slightly different gradients across runs (see the [non-determinism FAQ](04-engineering-decisions.md#q-why-is-the-backward-pass-non-deterministic)).

### Q: What is an atomic operation?

An atomic operation is a memory write that is guaranteed to complete without interference from other threads. When two GPU threads both try to add a value to the same memory location, a normal add might lose one update (a race condition). An `atomic_add` prevents this by serializing the writes. The tradeoff: atomics are slow (they force threads to wait) and introduce non-determinism because the order of additions is unpredictable, and floating-point addition is not associative — `(a+b)+c ≠ a+(b+c)` in floating point.

### Q: What is a Triton kernel?

Triton is a compiler and programming language by OpenAI for writing GPU kernels in Python-like syntax. A "kernel" is a function that runs on the GPU. torch-semimarkov uses Triton to write custom kernels for the forward and backward passes of the semi-CRF DP, which is more efficient than relying on PyTorch's general-purpose operations because the kernel fuses many small operations into one GPU launch and controls memory access patterns precisely.

### Q: What is tiling?

Tiling means processing data in small blocks (tiles) that fit in the GPU's fast on-chip memory (registers and shared memory) rather than reading from slow global memory repeatedly. In torch-semimarkov, the label dimension C is processed in tiles when C is large enough that the full C×C transition matrix wouldn't fit in registers. The library automatically selects tile sizes based on C:

| Padded C (C_PAD) | Tile Size (τ) | Iterations | Rationale |
|-------------------|---------------|------------|-----------|
| ≤ 8 | 4 | 2 | Minimal iteration count |
| ≤ 16 | 8 | 2 | Minimal iteration count |
| 32 | 16 | 2 | Balanced |
| 64 | 16 | 4 | Moderate register pressure |
| ≥ 128 | 32 | ≤ 8 | Bounded compile time |

### Q: What does "register pressure" or "register spilling" mean?

Each GPU thread has a limited number of fast local variables called registers. If a kernel needs more registers than are available, the excess "spills" to slower local memory, which can dramatically reduce performance. The tiling strategy in torch-semimarkov is specifically designed to keep register usage under the spilling threshold (~120 registers per thread, enabling 4–8 warps without spilling).

## Algorithmic Concepts

### Q: What is a ring buffer?

A ring buffer is a fixed-size array that wraps around: when you reach the end, you start overwriting from the beginning. In torch-semimarkov, the forward pass only needs the K most recent forward messages (because a segment can be at most K positions long). Instead of storing all T messages, a ring buffer of size K stores them at index `t mod K`. Position 0 goes to slot 0, position K goes to slot 0 again (overwriting), and so on. This is what makes memory O(KC) instead of O(TC).

```
Time:     0   1   2   3   4   5   6   7   8   ...
Ring idx: 0   1   2   0   1   2   0   1   2   ...  (K=3)
          ↑           ↑           ↑
          overwritten by t=3, 6, 9, ...
```

The backward pass uses a ring buffer of size 2K (not K) to avoid a subtle aliasing problem: when computing the backward message at position t, you need to read forward from positions t+1 through t+K. With a buffer of size K, the write slot for t and the read slot for t+K would collide (`t mod K == (t+K) mod K`). Doubling the buffer eliminates this conflict.

### Q: What is checkpointing (gradient checkpointing)?

During the forward pass, you normally save all intermediate results so the backward pass can use them to compute gradients. For long sequences, this uses too much memory. Gradient checkpointing saves only occasional snapshots (every Δ ≈ √(T·K) steps). During the backward pass, the forward computation is re-run from the nearest snapshot to regenerate the needed intermediates. This trades compute (roughly 2×) for memory (roughly √T× reduction).

### Q: What is the prefix-sum decomposition?

The key trick that makes streaming possible. Instead of storing a score for every (position, duration, label) triple, you store cumulative scores: `S[t,c] = sum of emission scores from position 0 to t−1 for label c`. Then the emission score for a segment spanning positions [a, b] is just `S[b+1,c] − S[a,c]`, computed in O(1) time. This is why the library only needs O(TC) storage for cumulative scores rather than O(TKC) for all possible segments.

### Q: What is logsumexp and why is it everywhere?

LogSumExp computes `log(sum(exp(x)))` in a numerically stable way. In the semi-CRF, we work in log-space because probabilities can be astronomically small (the product of millions of factors). Adding in log-space corresponds to multiplying probabilities; logsumexp corresponds to summing probabilities. The stable version subtracts the maximum before exponentiating to prevent overflow:

```
logsumexp(x) = max(x) + log(sum(exp(x − max(x))))
```

### Q: What is zero-centering and why does it matter?

When computing cumulative sums over 100,000+ positions, the running total can drift to very large values, causing floating-point precision loss (large numbers swallow small corrections). Zero-centering subtracts the mean of the per-position scores before taking the cumulative sum, keeping the running total near zero throughout. This is critical for numerical stability at genomic scale.

## Software Concepts

### Q: What is a semiring (in code)?

In torch-semimarkov, a semiring is a Python object that defines two operations: `combine` (how to merge candidates at a position, e.g., logsumexp or max) and `extend` (how to add a new segment's score, typically addition in log-space). The DP loop calls these operations without knowing which semiring it's using. Swapping the semiring object changes the question the DP answers, with no changes to the DP code itself.

### Q: What is NEG_INF and why isn't it negative infinity?

The library uses −1×10⁹ (a very large negative number) instead of true −∞ for masking invalid states. True −∞ causes problems in gradient computation: −∞ − (−∞) = NaN, which propagates and corrupts training. Using a finite sentinel avoids this while being negative enough that it never contributes to logsumexp results.

### Q: What does "on-the-fly" mean for edge computation?

Instead of pre-computing and storing every possible edge potential in a giant tensor (shape: batch × T × K × C × C), the streaming kernel computes each edge potential from three small components at the moment it's needed:

```
edge[c_dst, c_src] = (cum_scores[t, c_dst] − cum_scores[t−k, c_dst])   # content
                   + duration_bias[k, c_dst]                             # duration
                   + transition[c_src, c_dst]                            # transition
```

The content term comes from a subtraction of two entries in the cumulative score array (O(1)). The duration bias and transition matrix are small shared parameters. After the edge is used in the DP update, it's discarded — never stored. This is how memory stays at O(KC) regardless of T.

---

**See also:** [Engineering Decisions](04-engineering-decisions.md) · [Streaming internals](../streaming_internals.md) · [Backend algorithm supplement](../backend_algorithm_supplement.pdf)
