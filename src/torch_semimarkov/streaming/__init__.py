r"""Streaming API for memory-efficient Semi-CRF inference.

This module implements on-the-fly edge computation using prefix-sum decomposition:
edge potentials are computed on-the-fly from pre-projected cumulative scores,
eliminating the need to materialize the full (batch, T-1, K, C, C) edge tensor.

.. important::
    **When to use this module vs. pre-computed edges:**

    Use ``streaming`` (this module) for:
        - **Training and inference** (default) - hand-written Triton forward and backward kernels
        - **Very long sequences** (T = 10K - 400K+) - edge tensor cannot fit in memory
        - **Semirings**: log and max only

    Use ``SemiMarkov.logpartition`` with pre-computed edge tensors when:
        - Edge tensor fits in GPU memory (O(T x K x C²))
        - You need semirings beyond log/max (entropy, KL divergence, cross-entropy, counting, K-best)
        - You have pre-computed edges from an external source

    **Memory comparison:**

    +-----------------------+------------------+-------------------+
    | Scenario              | edge tensor size | cum_scores size   |
    +=======================+==================+===================+
    | T=1K, K=32, C=24      | 18 MB            | 192 KB            |
    +-----------------------+------------------+-------------------+
    | T=10K, K=100, C=24    | 5.5 GB           | 1.9 MB            |
    +-----------------------+------------------+-------------------+
    | T=400K, K=3K, C=24    | **2.76 TB**      | 76 MB             |
    +-----------------------+------------------+-------------------+

    For the T=400K case, the edge tensor cannot fit in memory. This module
    computes edges on-the-fly from O(TxC) cumulative scores instead.

    Why streaming is faster than pre-computed edges:

    - **Memory bandwidth**: Loading O(TxKxC²) edges from memory is slower than
      computing O(TxC) edge blocks on-the-fly from cumulative scores
    - **Cache efficiency**: Streaming keeps working set in L1/L2 cache
    - **Linear batch scaling**: Memory grows as O(batchxTxC), not O(batchxTxKxC²)

    **Training advantages:**

    - Hand-written Triton forward and backward kernels
    - No compilation latency (torch.compile takes 20+ minutes for T=1000)
    - No RecursionError from deep computational graphs
    - No OOM from compiled gradient buffers

API Comparison
--------------
``SemiMarkov.logpartition`` takes a **pre-computed edge tensor**::

    edge = model(x)  # shape: (batch, T-1, K, C, C) - must fit in GPU memory!
    crf = SemiMarkov(LogSemiring)
    log_Z, _ = crf.logpartition(edge, lengths=lengths)

This module takes **cumulative scores** and computes edges on-the-fly::

    cum_scores = cumsum(projected, dim=1)  # shape: (batch, T+1, C) - much smaller!
    partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)

Memory Complexity
-----------------
- Pre-computed edge API (``SemiMarkov``): O(T x K x C²) - 2.76 TB for T=400K, K=3K, C=24
- Streaming API (this module): O(T x C + K x C + C²) - ~50 MB for same dimensions

Streaming Edge Computation
--------------------------
Instead of pre-computing edges, we pre-project encoder features to label space
BEFORE the kernel (loop-invariant projection), then compute edges on-the-fly inside:

    # Outside kernel (parallel, efficient)
    projected = h @ W_content                    # (batch, T, C)
    projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center!
    cum_scores = cumsum(projected.double(), dim=1)  # (batch, T+1, C) in float64

    # Inside kernel (just vector ops, no matmuls)
    content_score = cum_scores[:, t+k, :] - cum_scores[:, t, :]  # (batch, C)
    segment_score = content_score + duration_bias[k]
    edge_block = segment_score.unsqueeze(-1) + transition        # (batch, C, C)

The edge potential for segment [t, t+k) with label c_dest from c_src is::

    edge[t, k, c_dest, c_src] = (cum_scores[t+k, c_dest] - cum_scores[t, c_dest])
                              + duration_bias[k, c_dest]
                              + transition[c_src, c_dest]

This structure means you **never need to materialize the full edge tensor**.

Numerical Stability
-------------------
Two critical requirements for T=400K+ sequences:

1. **Float64 cumsum**: Cumsum must be float64 for numerical stability.
   Float16 loses all precision at T=400K magnitudes. Float32 is acceptable
   but float64 matches the kernel's internal precision.
   All Triton kernels (forward and backward) compute internally in float64.

2. **Zero-centering**: Without centering, cumsum drifts to ~T magnitude.
   At T=400K, float32 epsilon at that magnitude is ~0.04 - any signal
   smaller than that is completely erased. Zero-centering keeps magnitude
   at √T (~632 for T=400K), preserving signals down to ~10⁻⁴.

Transition Matrix Convention
----------------------------
The transition matrix follows ``transition[source, destination]`` convention::

    transition[i, j] = log P(label_j | previous_label_i)
                     = score for transitioning FROM i TO j

When computing edge potentials, the transition is transposed to match
the edge tensor orientation ``(C_dest, C_src)``::

    edge[c_dest, c_src] = segment_score[c_dest] + transition.T[c_dest, c_src]
                        = segment_score[c_dest] + transition[c_src, c_dest]

This transposition ensures efficient memory access during the forward pass,
where the reduction is over ``c_src`` (the source/previous label).

Usage
-----
>>> import torch
>>> from torch_semimarkov.streaming import semi_crf_streaming_forward
>>>
>>> # Pre-project features (outside kernel)
>>> h = encoder(x)  # (batch, T, hidden_dim)
>>> projected = h @ W_content
>>> projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center!
>>> cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float64)
>>> cum_scores[:, 1:, :] = torch.cumsum(projected.double(), dim=1)
>>>
>>> # Streaming forward (edges computed on-the-fly)
>>> partition = semi_crf_streaming_forward(
...     cum_scores, transition, duration_bias, lengths, K
... )

See Also
--------
:class:`torch_semimarkov.SemiMarkov` : Pre-computed edge tensor API with all 7 semirings
"""

from .autograd import (
    SemiCRFStreaming,
    SemiCRFStreamingTriton,
    semi_crf_streaming_forward,
)
from .constants import NEG_INF
from .pytorch_reference import (
    _compute_checkpoint_interval,
    compute_edge_block_streaming,
    semi_crf_streaming_backward_pytorch,
    semi_crf_streaming_forward_pytorch,
    semi_crf_streaming_marginals_pytorch,
    semi_crf_streaming_viterbi_with_backpointers,
)

# Re-export HAS_TRITON for external checks
try:
    from .triton_forward import HAS_TRITON
except ImportError:
    HAS_TRITON = False

# Conditionally export Triton launchers
if HAS_TRITON:
    from .triton_backward import (
        launch_streaming_triton_backward,
        launch_streaming_triton_marginals,
    )
    from .triton_forward import (
        launch_streaming_triton_kernel,
        launch_streaming_triton_kernel_max_bp,
        semi_crf_streaming_viterbi_triton,
    )

__all__ = [
    # Main API
    "semi_crf_streaming_forward",
    # Autograd Functions
    "SemiCRFStreaming",
    "SemiCRFStreamingTriton",
    # PyTorch reference implementations
    "semi_crf_streaming_forward_pytorch",
    "semi_crf_streaming_backward_pytorch",
    "semi_crf_streaming_marginals_pytorch",
    "semi_crf_streaming_viterbi_with_backpointers",
    "compute_edge_block_streaming",
    # Utilities
    "_compute_checkpoint_interval",
    "NEG_INF",
    "HAS_TRITON",
    # Triton launchers (conditionally available)
    "launch_streaming_triton_backward",
    "launch_streaming_triton_marginals",
    "launch_streaming_triton_kernel",
    "launch_streaming_triton_kernel_max_bp",
    "semi_crf_streaming_viterbi_triton",
]
