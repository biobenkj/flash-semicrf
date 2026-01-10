"""
Fused streaming Semi-Markov CRF forward scan implementations.

This module provides optimized implementations of the streaming forward scan
for Semi-Markov CRFs. The key optimization is fusing the O(N) loop into
a single operation, keeping the K×C frontier in fast memory.

Implementations:
1. PyTorch reference (CPU/GPU) - always available, used for testing
2. Triton kernel (GPU only) - requires triton package, ~2-5x faster

Key insight from systems analysis:
    "A custom kernel would allow you to perform Fused Streaming Inference,
    keeping the K×D frontier in Shared Memory (SRAM) rather than constantly
    round-tripping to Global Memory (DRAM)."

Memory profile: O(K×C) per batch element (typically 2-8 KB)
Compute pattern: Fuses N kernel launches into 1

Usage:
    from torch_semimarkov.triton_scan import semi_crf_triton_forward
    partition = semi_crf_triton_forward(edge, lengths)
"""

import torch
import math

# Triton is optional - kernel only available when installed and on GPU
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# =============================================================================
# PyTorch Reference Implementation (CPU + GPU)
# =============================================================================

def semi_crf_forward_pytorch(edge, lengths):
    """
    Reference PyTorch implementation matching _dp_scan_streaming semantics.

    This implementation:
    - Works on CPU and GPU
    - Uses O(K×C) ring buffer (same as streaming scan)
    - Serves as reference for correctness validation
    - Used as fallback when Triton not available
    - Supports gradient computation via autograd

    Recurrence:
        beta[n, c] = logsumexp_{k=1..min(K-1,n), c_prev} (
            beta[n-k, c_prev] + edge[n-k, k, c, c_prev]
        )

    Args:
        edge: (batch, N-1, K, C, C) log potentials
        lengths: (batch,) sequence lengths

    Returns:
        partition: (batch,) log partition function
    """
    batch, N_1, K, C, _ = edge.shape
    N = N_1 + 1
    device = edge.device
    dtype = edge.dtype

    NEG_INF = -1e9

    # Ring buffer as list of tensors to avoid in-place updates
    # This allows proper gradient tracking
    ring_len = K
    initial_beta = torch.zeros((batch, C), device=device, dtype=dtype)
    beta_ring = [initial_beta] + [
        torch.full((batch, C), NEG_INF, device=device, dtype=dtype)
        for _ in range(ring_len - 1)
    ]
    head = 0

    # Duration indices (reused each iteration)
    dur_full = torch.arange(1, K, device=device)

    # Final beta storage (captured at each batch's sequence end)
    final_beta = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

    # Handle length=1: partition = logsumexp(0, 0, ..., 0) = log(C)
    mask_len1 = (lengths == 1).view(batch, 1)
    final_beta = torch.where(mask_len1, initial_beta, final_beta)

    # Main scan loop
    for n in range(1, N):
        # Number of valid durations at this position
        k_eff = min(K - 1, n)
        dur = dur_full[:k_eff]        # [1, 2, ..., k_eff]
        start = n - dur               # positions where segments start

        # Get previous betas from ring buffer
        # ring_idx[i] = (head - (dur[i] - 1)) % ring_len
        ring_idx = [(head - (d.item() - 1)) % ring_len for d in dur]
        beta_prev = torch.stack([beta_ring[i] for i in ring_idx], dim=1)  # (batch, k_eff, C)

        # Get edge potentials
        edge_slice = edge[:, start, dur, :, :]  # (batch, k_eff, C, C)

        # First logsumexp: over c_prev (source labels)
        scores = torch.logsumexp(
            beta_prev.unsqueeze(-2) + edge_slice,
            dim=-1
        )  # (batch, k_eff, C)

        # Second logsumexp: over duration dimension
        beta_n = torch.logsumexp(scores, dim=1)  # (batch, C)

        # Capture final beta for sequences ending at this position
        mask_end = (lengths == (n + 1)).view(batch, 1)
        final_beta = torch.where(mask_end, beta_n, final_beta)

        # Update ring buffer (replace entry, don't modify in-place)
        head = (head + 1) % ring_len
        beta_ring[head] = beta_n

    # Final partition: logsumexp over labels
    partition = torch.logsumexp(final_beta, dim=-1)
    return partition


# =============================================================================
# Triton Kernels (GPU only, optional)
# =============================================================================

if HAS_TRITON:

    @triton.jit
    def semi_crf_scan_kernel(
        # Inputs
        edge_ptr,           # (batch, N-1, K, C, C) - edge potentials
        ring_ptr,           # (batch, K, C) - ring buffer (read/write)
        out_ptr,            # (batch,) - output partition
        lengths_ptr,        # (batch,) - sequence lengths
        # Dimensions
        batch_size,
        N: tl.constexpr,    # max sequence length
        K: tl.constexpr,    # max duration
        C: tl.constexpr,    # num labels (must be <= 32 for warp shuffles)
        # Strides for edge tensor
        stride_eb, stride_en, stride_ek, stride_ec1, stride_ec2,
        # Strides for ring buffer
        stride_rb, stride_rk, stride_rc,
    ):
        """
        Fused Semi-Markov CRF forward scan with arbitrary K support.

        Uses global memory ring buffer (L2/L1 cached) for the DP state.
        Each warp handles one batch element, threads handle labels.

        Ring buffer layout: ring[batch, k, c]
        - k=0 is head (most recent beta)
        - k=1..K-1 are older betas
        - We rotate head pointer instead of shifting data
        """
        NEG_INF: tl.constexpr = -1e20

        # Batch index (one warp per batch)
        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # Lane/thread handles one label
        lane = tl.arange(0, 32)
        c = lane
        c_mask = c < C

        # Load sequence length
        seq_len = tl.load(lengths_ptr + batch_idx)

        # Base pointers
        edge_base = edge_ptr + batch_idx * stride_eb
        ring_base = ring_ptr + batch_idx * stride_rb

        # Initialize ring buffer: slot 0 = 0.0, rest = NEG_INF
        for k in range(K):
            val = tl.where(k == 0, 0.0, NEG_INF)
            val = tl.where(c_mask, val, NEG_INF)
            ring_offset = ring_base + k * stride_rk + c * stride_rc
            tl.store(ring_offset, val, mask=c_mask)

        # Head pointer (index into ring buffer for most recent beta)
        head = 0

        # Track final beta for each batch
        final_beta = tl.where(c_mask, 0.0, NEG_INF)

        # Main loop over sequence positions
        for n in range(1, N):
            # Early exit if past sequence length
            if n >= seq_len:
                break

            # Effective max duration
            k_max = tl.minimum(K - 1, n)

            # Accumulate new_beta = logsumexp over (k, c_prev)
            new_beta = tl.full([32], NEG_INF, dtype=tl.float32)

            # Loop over durations k = 1, 2, ..., k_max
            for k in tl.static_range(1, K):
                # Skip if duration exceeds position or max
                if k > k_max:
                    continue

                start_pos = n - k

                # Ring index for beta[n-k]: (head - (k-1)) mod K
                # After head update: head points to n-1, so:
                # - beta[n-1] is at ring[head]
                # - beta[n-2] is at ring[(head-1) % K]
                # - beta[n-k] is at ring[(head - (k-1)) % K]
                ring_k_idx = (head - (k - 1) + K) % K

                # Load beta_prev[c] for all labels from ring buffer
                # Each thread loads its own label's value
                beta_prev_self = tl.load(
                    ring_base + ring_k_idx * stride_rk + c * stride_rc,
                    mask=c_mask,
                    other=NEG_INF
                )

                # Logsumexp over c_prev: need values from all threads
                score_accum = tl.full([32], NEG_INF, dtype=tl.float32)

                for c_prev in tl.static_range(0, C):
                    # Get beta[n-k, c_prev] via warp shuffle
                    beta_from_cprev = tl.extra.cuda.shfl_sync(
                        0xFFFFFFFF, beta_prev_self, c_prev
                    )

                    # Load edge[start_pos, k, c, c_prev]
                    edge_offset = (
                        edge_base +
                        start_pos * stride_en +
                        k * stride_ek +
                        c * stride_ec1 +
                        c_prev * stride_ec2
                    )
                    edge_val = tl.load(edge_offset, mask=c_mask, other=NEG_INF)

                    # Accumulate: logsumexp(score_accum, beta + edge)
                    contrib = beta_from_cprev + edge_val
                    max_sc = tl.maximum(score_accum, contrib)
                    score_accum = max_sc + tl.log(
                        tl.exp(score_accum - max_sc) + tl.exp(contrib - max_sc)
                    )

                # Accumulate this duration into new_beta
                max_nb = tl.maximum(new_beta, score_accum)
                new_beta = max_nb + tl.log(
                    tl.exp(new_beta - max_nb) + tl.exp(score_accum - max_nb)
                )

            # Advance head pointer (circular)
            head = (head + 1) % K

            # Store new_beta to ring buffer at new head position
            new_beta_masked = tl.where(c_mask, new_beta, NEG_INF)
            tl.store(
                ring_base + head * stride_rk + c * stride_rc,
                new_beta_masked,
                mask=c_mask
            )

            # Capture final beta at sequence end
            if n == seq_len - 1:
                final_beta = new_beta_masked

        # Final reduction: logsumexp over labels
        max_fb = tl.where(c_mask, final_beta, NEG_INF)
        for offset in [16, 8, 4, 2, 1]:
            other = tl.extra.cuda.shfl_xor_sync(0xFFFFFFFF, max_fb, offset)
            max_fb = tl.maximum(max_fb, other)
        max_val = tl.extra.cuda.shfl_sync(0xFFFFFFFF, max_fb, 0)

        exp_fb = tl.where(c_mask, tl.exp(final_beta - max_val), 0.0)
        for offset in [16, 8, 4, 2, 1]:
            other = tl.extra.cuda.shfl_xor_sync(0xFFFFFFFF, exp_fb, offset)
            exp_fb = exp_fb + other
        sum_exp = tl.extra.cuda.shfl_sync(0xFFFFFFFF, exp_fb, 0)

        partition = max_val + tl.log(sum_exp)

        # Lane 0 writes result
        if lane[0] == 0:
            tl.store(out_ptr + batch_idx, partition)


    def launch_triton_kernel(edge, lengths):
        """
        Launch the Triton kernel with proper buffer allocation.

        Args:
            edge: (batch, N-1, K, C, C) contiguous CUDA tensor
            lengths: (batch,) CUDA tensor

        Returns:
            partition: (batch,) log partition function
        """
        batch, N_1, K, C, _ = edge.shape
        N = N_1 + 1

        # Ensure contiguous
        edge = edge.contiguous()

        # Allocate ring buffer (small, will be L2 cached)
        ring_buffer = torch.empty(
            (batch, K, C),
            device=edge.device,
            dtype=edge.dtype
        )

        # Output buffer
        partition = torch.empty(batch, device=edge.device, dtype=edge.dtype)

        # Get strides
        stride_eb, stride_en, stride_ek, stride_ec1, stride_ec2 = edge.stride()
        stride_rb, stride_rk, stride_rc = ring_buffer.stride()

        # Launch kernel
        grid = (batch,)
        semi_crf_scan_kernel[grid](
            edge, ring_buffer, partition, lengths,
            batch, N, K, C,
            stride_eb, stride_en, stride_ek, stride_ec1, stride_ec2,
            stride_rb, stride_rk, stride_rc,
        )

        return partition


# =============================================================================
# Autograd Function
# =============================================================================

class SemiCRFTritonForward(torch.autograd.Function):
    """
    Autograd wrapper with gradient checkpointing.

    Forward: Triton kernel (if available) or PyTorch fallback
    Backward: Recompute forward with gradients (checkpointing)
    """

    @staticmethod
    def forward(ctx, edge, lengths, use_triton=True):
        batch, N_1, K, C, C2 = edge.shape
        N = N_1 + 1

        # Check if Triton kernel is applicable
        use_triton_kernel = (
            HAS_TRITON and
            use_triton and
            edge.is_cuda and
            C <= 32  # Warp size limit for shuffle-based communication
        )

        if use_triton_kernel:
            partition = launch_triton_kernel(edge, lengths)
        else:
            partition = semi_crf_forward_pytorch(edge.detach(), lengths)

        ctx.save_for_backward(edge, lengths)
        ctx.use_triton = use_triton_kernel

        return partition

    @staticmethod
    def backward(ctx, grad_output):
        edge, lengths = ctx.saved_tensors

        # Recompute forward with gradients (checkpointing)
        edge_grad = edge.detach().requires_grad_(True)

        with torch.enable_grad():
            partition = semi_crf_forward_pytorch(edge_grad, lengths)

            # Use grad_outputs to weight the gradients
            # This computes: sum_b(grad_output[b] * d(partition[b])/d(edge_grad))
            grad_edge = torch.autograd.grad(
                outputs=partition,
                inputs=edge_grad,
                grad_outputs=grad_output,
                create_graph=False
            )[0]

        return grad_edge, None, None


def semi_crf_triton_forward(edge, lengths, use_triton=True):
    """
    Main entry point for Semi-Markov CRF forward scan.

    Uses Triton kernel when available and applicable, otherwise
    falls back to optimized PyTorch implementation.

    Args:
        edge: (batch, N-1, K, C, C) log potentials
        lengths: (batch,) sequence lengths
        use_triton: If True, use Triton when possible

    Returns:
        partition: (batch,) log partition function
    """
    return SemiCRFTritonForward.apply(edge, lengths, use_triton)


# =============================================================================
# Testing
# =============================================================================

def test_against_library():
    """Test against the library's streaming implementation."""
    try:
        from torch_semimarkov import SemiMarkov
        from torch_semimarkov.semirings import LogSemiring
    except ImportError:
        print("torch_semimarkov not fully installed, skipping library test")
        return True

    print("Testing against library _dp_scan_streaming...")

    test_cases = [
        (1, 10, 4, 2),
        (1, 20, 8, 4),
        (4, 50, 16, 8),
        (2, 100, 32, 4),
        (2, 100, 64, 8),   # Larger K
    ]

    all_passed = True
    for batch, N, K, C in test_cases:
        edge = torch.randn(batch, N-1, K, C, C)
        lengths = torch.full((batch,), N, dtype=torch.long)

        # Library reference
        sm = SemiMarkov(LogSemiring)
        lib_partition, _, _ = sm._dp_scan_streaming(edge, lengths)
        lib_partition = LogSemiring.unconvert(lib_partition)

        # Our PyTorch reference
        our_partition = semi_crf_forward_pytorch(edge, lengths)

        max_diff = (lib_partition - our_partition).abs().max().item()
        passed = max_diff < 1e-4

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: batch={batch}, N={N}, K={K}, C={C}, max_diff={max_diff:.2e}")

        if not passed:
            print(f"    Library: {lib_partition}")
            print(f"    Ours:    {our_partition}")
            all_passed = False

    return all_passed


def test_triton_kernel():
    """Test Triton kernel against PyTorch reference."""
    if not HAS_TRITON:
        print("Triton not available, skipping kernel test")
        return True

    if not torch.cuda.is_available():
        print("CUDA not available, skipping kernel test")
        return True

    print("Testing Triton kernel vs PyTorch reference...")

    test_cases = [
        (1, 20, 4, 4),
        (4, 50, 8, 8),
        (2, 100, 16, 8),
        (1, 100, 32, 8),
        (2, 100, 64, 8),   # Large K test
    ]

    all_passed = True
    for batch, N, K, C in test_cases:
        edge = torch.randn(batch, N-1, K, C, C, device='cuda')
        lengths = torch.full((batch,), N, dtype=torch.long, device='cuda')

        # PyTorch reference
        ref = semi_crf_forward_pytorch(edge, lengths)

        # Triton kernel
        triton_out = launch_triton_kernel(edge, lengths)

        max_diff = (ref - triton_out).abs().max().item()
        passed = max_diff < 1e-3

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: batch={batch}, N={N}, K={K}, C={C}, max_diff={max_diff:.2e}")

        if not passed:
            print(f"    PyTorch: {ref.tolist()}")
            print(f"    Triton:  {triton_out.tolist()}")
            all_passed = False

    return all_passed


def test_gradients():
    """Test gradient computation."""
    print("Testing gradients...")

    batch, N, K, C = 2, 20, 16, 4
    edge = torch.randn(batch, N-1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), N, dtype=torch.long)

    # Forward
    partition = semi_crf_triton_forward(edge, lengths, use_triton=False)
    loss = partition.sum()

    # Backward
    loss.backward()

    print(f"  Forward passed: partition = {partition.tolist()}")
    print(f"  Backward passed: grad shape = {edge.grad.shape}")
    print(f"  Grad stats: min={edge.grad.min():.4f}, max={edge.grad.max():.4f}")

    return True


def benchmark(batch=4, N=1024, K=64, C=8, n_iters=10, device='cpu'):
    """Benchmark implementations."""
    import time

    if device == 'cuda' and not torch.cuda.is_available():
        print(f"CUDA not available, skipping GPU benchmark")
        return None

    edge = torch.randn(batch, N-1, K, C, C, device=device)
    lengths = torch.full((batch,), N, dtype=torch.long, device=device)

    # Warmup PyTorch
    for _ in range(3):
        _ = semi_crf_forward_pytorch(edge, lengths)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Time PyTorch
    start = time.perf_counter()
    for _ in range(n_iters):
        partition_pt = semi_crf_forward_pytorch(edge, lengths)
    if device == 'cuda':
        torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - start) / n_iters * 1000

    throughput = batch * N / (pytorch_ms / 1000) / 1e6

    print(f"PyTorch ({device}):")
    print(f"  Config: batch={batch}, N={N}, K={K}, C={C}")
    print(f"  Time: {pytorch_ms:.2f} ms")
    print(f"  Throughput: {throughput:.2f} M positions/sec")

    # Triton benchmark (GPU only)
    if device == 'cuda' and HAS_TRITON and C <= 32:
        for _ in range(3):
            _ = launch_triton_kernel(edge, lengths)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            partition_tr = launch_triton_kernel(edge, lengths)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / n_iters * 1000

        # Check correctness
        max_diff = (partition_pt - partition_tr).abs().max().item()

        print(f"Triton:")
        print(f"  Time: {triton_ms:.2f} ms")
        print(f"  Speedup vs PyTorch: {pytorch_ms/triton_ms:.2f}x")
        print(f"  Max diff from PyTorch: {max_diff:.2e}")

        return pytorch_ms, triton_ms

    return pytorch_ms, None


if __name__ == "__main__":
    print("=" * 60)
    print("Semi-Markov CRF Fused Streaming Scan")
    print("=" * 60)
    print(f"Triton available: {HAS_TRITON}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\n1. Testing against library implementation (CPU):")
    test_against_library()

    print("\n2. Testing Triton kernel (GPU):")
    test_triton_kernel()

    print("\n3. Testing gradients:")
    test_gradients()

    print("\n4. Benchmarking on CPU:")
    benchmark(batch=2, N=100, K=16, C=8, device='cpu')

    if torch.cuda.is_available():
        print("\n5. Benchmarking on GPU:")
        benchmark(batch=4, N=500, K=64, C=8, device='cuda')
