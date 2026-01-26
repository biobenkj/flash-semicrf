r"""Triton forward kernels for streaming Semi-CRF.

Edge potentials computed on-the-fly via prefix-sum:

.. math::
    \text{edge} = (\text{cum}[t+k] - \text{cum}[t]) + \text{dur\_bias}[k] + \text{transition}
"""

import torch

from .constants import NEG_INF
from .pytorch_reference import _compute_checkpoint_interval

# Triton is optional
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n. Returns 1 for n <= 0."""
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    p = 1
    while p < n:
        p *= 2
    return p


if HAS_TRITON:

    @triton.jit
    def semi_crf_streaming_scan_kernel(
        # Inputs
        cum_scores_ptr,  # (batch, T+1, C) - cumulative projected scores
        transition_ptr,  # (C, C) or (K, C, C) - transition matrix
        duration_bias_ptr,  # (K, C) - duration-specific bias
        lengths_ptr,  # (batch,) - sequence lengths
        # Boundary projections (optional, may be null if HAS_BOUNDARIES=False)
        proj_start_ptr,  # (batch, T, C) - start boundary scores
        proj_end_ptr,  # (batch, T, C) - end boundary scores
        # Outputs
        out_ptr,  # (batch,) - partition function
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer (read/write)
        ring_ckpt_ptr,  # (batch, num_ckpts, K, C_PAD) - checkpoints for backward
        # Dimensions
        batch_size,
        T: tl.constexpr,  # max sequence length (T, not T+1)
        K: tl.constexpr,  # max segment duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        CHECKPOINT_INTERVAL: tl.constexpr,  # interval for saving ring buffer
        NUM_CKPTS: tl.constexpr,  # number of checkpoints
        HAS_BOUNDARIES: tl.constexpr,  # whether boundary projections are provided
        HAS_DURATION_TRANSITIONS: tl.constexpr,  # whether transitions are (K, C, C)
        # Strides for cum_scores (batch, T+1, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        # Strides for transition (C, C) or (K, C, C)
        stride_tr_k,  # Only used if HAS_DURATION_TRANSITIONS
        stride_tr_src,
        stride_tr_dst,
        # Strides for duration_bias (K, C)
        stride_db_k,
        stride_db_c,
        # Strides for proj_start/proj_end (batch, T, C) - only used if HAS_BOUNDARIES
        stride_ps_b,
        stride_ps_t,
        stride_ps_c,
        # Strides for ring buffer (batch, K, C_PAD)
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        # Strides for ring checkpoints (batch, num_ckpts, K, C_PAD)
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
    ):
        """Forward scan with on-the-fly edge computation (log semiring).

        Memory: O(KC) ring buffer. One program per batch element.
        """
        NEG_INF: tl.constexpr = -1e9

        # Batch index (one program per batch element)
        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # 1D indices for labels (padded to power of 2)
        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        # 2D indices for (C_dst, C_src) operations
        c_dst_idx = tl.arange(0, C_PAD)[:, None]  # (C_PAD, 1)
        c_src_idx = tl.arange(0, C_PAD)[None, :]  # (1, C_PAD)
        c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)

        # Load sequence length
        seq_len = tl.load(lengths_ptr + batch_idx)

        # Base pointers
        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_base = ring_ptr + batch_idx * stride_ring_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b

        # Boundary projection base pointers (only used if HAS_BOUNDARIES)
        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b

        # Load transition matrix into registers: (C_PAD, C_PAD)
        # For static transitions (C, C): load once here
        # For duration-dependent (K, C, C): load inside k-loop
        # transition[c_src, c_dst] -> we need transition.T for edge computation
        # So we load transition_ptr[c_dst, c_src] effectively
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )  # (C_PAD, C_PAD) - this is transition.T

        # Ring buffer and checkpoint 0 are pre-initialized by the launcher:
        # - ring_buffer[:, 0, :C] = 0.0, rest = NEG_INF
        # - ring_checkpoints[:, 0, 0, :C] = 0.0, rest = NEG_INF
        # This avoids K iterations of conditional writes per batch element.

        # Track final alpha for each batch element
        final_alpha = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

        # Main forward loop: t = 1, 2, ..., T
        for t in tl.range(1, T + 1):
            # Include t == seq_len to compute alpha at final position
            active = t <= seq_len

            # Accumulate alpha[t] = logsumexp over (k, c_src)
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            # Loop over valid segment durations k = 1, 2, ..., min(K-1, t)
            # tl.maximum ensures K=1 processes at least one duration
            for k in tl.range(1, tl.maximum(K, 2)):
                # For K=1: k=1 is valid (maps to duration_bias[0])
                # For K>1: k=1..K-1 are valid
                k_valid = (k <= t) & (k <= tl.maximum(K - 1, 1))
                start_pos = t - k

                # Ring index for alpha[start_pos]
                ring_k_idx = start_pos % K

                # Load alpha_prev from live ring buffer
                alpha_prev = tl.load(
                    ring_base + ring_k_idx * stride_ring_k + c_idx * stride_ring_c,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )  # (C_PAD,) - alpha[start_pos, c_src]

                # Compute edge block on-the-fly (prefix-sum)
                cum_end = tl.load(
                    cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                cum_start = tl.load(
                    cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                content_score = cum_end - cum_start

                # Load duration bias
                # Use min(k, K-1) to handle K=1 case: k=1 maps to index 0
                dur_idx = tl.minimum(k, K - 1)
                dur_bias = tl.load(
                    duration_bias_ptr + dur_idx * stride_db_k + c_idx * stride_db_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                # Segment score = content_score + duration_bias
                segment_score = content_score + dur_bias  # (C_PAD,)

                # Add boundary scores if provided
                # Segment starts at start_pos, ends at t-1 (inclusive)
                if HAS_BOUNDARIES:
                    # proj_start[start_pos, :] - start boundary score
                    start_score = tl.load(
                        proj_start_base + start_pos * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    # proj_end[t-1, :] - end boundary score (t-1 is last position in segment)
                    end_pos_boundary = t - 1
                    end_score = tl.load(
                        proj_end_base + end_pos_boundary * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    segment_score = segment_score + start_score + end_score

                # Edge block: edge[c_dst, c_src] = segment_score[c_dst] + transition[c_src, c_dst]
                # segment_score is (C_PAD,), expand to (C_PAD, 1) for c_dst
                # transition_block is (C_PAD, C_PAD) as transition.T

                # For duration-dependent transitions, load transition[k] inside the loop
                if HAS_DURATION_TRANSITIONS:
                    transition_block = tl.load(
                        transition_ptr
                        + k * stride_tr_k
                        + c_dst_idx * stride_tr_dst
                        + c_src_idx * stride_tr_src,
                        mask=c_mask_2d,
                        other=0.0,
                    )  # (C_PAD, C_PAD) - transition[k].T

                edge_block = segment_score[:, None] + transition_block
                scores = alpha_prev[None, :] + edge_block

                # Mask out invalid entries
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Logsumexp over c_src (axis=1) -> (C_PAD,)
                # Guard against all-NEG_INF case to prevent undefined arithmetic
                # When max_scores == NEG_INF, scores - max_scores would be 0 (incorrect)
                max_scores = tl.max(scores, axis=1)
                is_all_neginf = max_scores < (NEG_INF + 1.0)
                max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
                log_sum_exp = tl.log(
                    tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1) + 1e-10
                )
                score_for_k = tl.where(is_all_neginf, NEG_INF, max_scores + log_sum_exp)

                # Mask invalid durations and labels
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Accumulate into alpha_t via logsumexp
                # Guard against both inputs being NEG_INF to prevent undefined arithmetic
                max_alpha = tl.maximum(alpha_t, score_for_k)
                is_both_neginf = (alpha_t < (NEG_INF + 1.0)) & (score_for_k < (NEG_INF + 1.0))
                max_alpha_safe = tl.where(is_both_neginf, 0.0, max_alpha)
                log_sum_exp_acc = tl.log(
                    tl.exp(alpha_t - max_alpha_safe) + tl.exp(score_for_k - max_alpha_safe) + 1e-10
                )
                alpha_t = tl.where(is_both_neginf, NEG_INF, max_alpha + log_sum_exp_acc)

            # Mask inactive sequences
            alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # Store to live ring buffer
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Save checkpoint at interval boundaries
            # Checkpoint i stores the ring buffer state at position i * CHECKPOINT_INTERVAL
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
                # Save entire ring buffer to checkpoint
                for k_save in tl.range(0, K):
                    ring_val = tl.load(
                        ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    # Only save if checkpoint index is valid
                    save_mask = (ckpt_idx < NUM_CKPTS) & c_mask
                    tl.store(
                        ring_ckpt_base
                        + ckpt_idx * stride_ckpt_n
                        + k_save * stride_ckpt_k
                        + c_idx * stride_ckpt_c,
                        ring_val,
                        mask=save_mask,
                    )

            # Capture final alpha at sequence end (t == seq_len)
            # At iteration t, alpha_t represents segments ending at position t-1
            # For sequence of length L, we need alpha at t=L (segments ending at L-1)
            is_final = t == seq_len
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # Final reduction: logsumexp over labels
        # Guard against all-NEG_INF case to prevent undefined arithmetic
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        max_val = tl.max(final_alpha_masked, axis=0)
        is_final_neginf = max_val < (NEG_INF + 1.0)
        max_val_safe = tl.where(is_final_neginf, 0.0, max_val)
        exp_fa = tl.where(c_mask, tl.exp(final_alpha - max_val_safe), 0.0)
        sum_exp = tl.sum(exp_fa, axis=0)
        partition = tl.where(is_final_neginf, NEG_INF, max_val + tl.log(sum_exp + 1e-10))

        # Store result
        tl.store(out_ptr + batch_idx, partition)

    @triton.jit
    def semi_crf_streaming_scan_kernel_max(
        # Same signature as log kernel
        cum_scores_ptr,
        transition_ptr,  # (C, C) or (K, C, C)
        duration_bias_ptr,
        lengths_ptr,
        # Boundary projections (optional)
        proj_start_ptr,
        proj_end_ptr,
        out_ptr,
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer
        ring_ckpt_ptr,
        batch_size,
        T: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        CHECKPOINT_INTERVAL: tl.constexpr,
        NUM_CKPTS: tl.constexpr,
        HAS_BOUNDARIES: tl.constexpr,
        HAS_DURATION_TRANSITIONS: tl.constexpr,  # whether transitions are (K, C, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        stride_tr_k,  # Only used if HAS_DURATION_TRANSITIONS
        stride_tr_src,
        stride_tr_dst,
        stride_db_k,
        stride_db_c,
        stride_ps_b,
        stride_ps_t,
        stride_ps_c,
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
    ):
        """Forward scan with max semiring (Viterbi). Same structure as log kernel."""
        NEG_INF: tl.constexpr = -1e9

        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        c_dst_idx = tl.arange(0, C_PAD)[:, None]
        c_src_idx = tl.arange(0, C_PAD)[None, :]
        c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)

        seq_len = tl.load(lengths_ptr + batch_idx)

        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_base = ring_ptr + batch_idx * stride_ring_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b

        # Boundary projection base pointers (only used if HAS_BOUNDARIES)
        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b

        # Load static transitions once (duration-dependent loaded inside k-loop)
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )

        # Ring buffer and checkpoint 0 are pre-initialized by the launcher:
        # - ring_buffer[:, 0, :C] = 0.0, rest = NEG_INF
        # - ring_checkpoints[:, 0, 0, :C] = 0.0, rest = NEG_INF
        # This avoids K iterations of conditional writes per batch element.

        final_alpha = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

        for t in tl.range(1, T + 1):
            # Include t == seq_len to compute alpha at final position
            active = t <= seq_len
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            # tl.maximum ensures K=1 processes at least one duration
            for k in tl.range(1, tl.maximum(K, 2)):
                # For K=1: k=1 is valid (maps to duration_bias[0])
                # For K>1: k=1..K-1 are valid
                k_valid = (k <= t) & (k <= tl.maximum(K - 1, 1))
                start_pos = t - k
                ring_k_idx = start_pos % K

                # Load from live ring buffer
                alpha_prev = tl.load(
                    ring_base + ring_k_idx * stride_ring_k + c_idx * stride_ring_c,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )

                cum_end = tl.load(
                    cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                cum_start = tl.load(
                    cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                content_score = cum_end - cum_start
                # Use min(k, K-1) to handle K=1 case: k=1 maps to index 0
                dur_idx = tl.minimum(k, K - 1)
                dur_bias = tl.load(
                    duration_bias_ptr + dur_idx * stride_db_k + c_idx * stride_db_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )
                segment_score = content_score + dur_bias

                # Add boundary scores if provided
                # Segment starts at start_pos, ends at t-1 (inclusive)
                if HAS_BOUNDARIES:
                    # proj_start[start_pos, :] - start boundary score
                    start_score = tl.load(
                        proj_start_base + start_pos * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    # proj_end[t-1, :] - end boundary score (t-1 is last position in segment)
                    end_pos_boundary = t - 1
                    end_score = tl.load(
                        proj_end_base + end_pos_boundary * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    segment_score = segment_score + start_score + end_score

                # For duration-dependent transitions, load transition[k] inside the loop
                if HAS_DURATION_TRANSITIONS:
                    transition_block = tl.load(
                        transition_ptr
                        + k * stride_tr_k
                        + c_dst_idx * stride_tr_dst
                        + c_src_idx * stride_tr_src,
                        mask=c_mask_2d,
                        other=0.0,
                    )  # (C_PAD, C_PAD) - transition[k].T

                edge_block = segment_score[:, None] + transition_block

                scores = alpha_prev[None, :] + edge_block
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Max semiring: max over c_src
                score_for_k = tl.max(scores, axis=1)
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Max semiring: max over k
                alpha_t = tl.maximum(alpha_t, score_for_k)

            alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # Store to live ring buffer
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Save checkpoint at interval boundaries
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
                for k_save in tl.range(0, K):
                    ring_val = tl.load(
                        ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    save_mask = (ckpt_idx < NUM_CKPTS) & c_mask
                    tl.store(
                        ring_ckpt_base
                        + ckpt_idx * stride_ckpt_n
                        + k_save * stride_ckpt_k
                        + c_idx * stride_ckpt_c,
                        ring_val,
                        mask=save_mask,
                    )

            # Capture final alpha at sequence end (t == seq_len)
            # At iteration t, alpha_t represents segments ending at position t-1
            # For sequence of length L, we need alpha at t=L (segments ending at L-1)
            is_final = t == seq_len
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # Max semiring: max over labels
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        partition = tl.max(final_alpha_masked, axis=0)

        tl.store(out_ptr + batch_idx, partition)

    @triton.jit
    def semi_crf_streaming_scan_kernel_max_bp(
        # Same signature as max kernel, plus backpointer outputs
        cum_scores_ptr,
        transition_ptr,  # (C, C) or (K, C, C)
        duration_bias_ptr,
        lengths_ptr,
        # Boundary projections (optional)
        proj_start_ptr,
        proj_end_ptr,
        out_ptr,
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer
        ring_ckpt_ptr,
        # Backpointer outputs
        bp_k_ptr,  # (batch, T, C) - best duration for each (t, c_dest)
        bp_c_ptr,  # (batch, T, C) - best source label for each (t, c_dest)
        final_labels_ptr,  # (batch,) - best final label
        batch_size,
        T: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        CHECKPOINT_INTERVAL: tl.constexpr,
        NUM_CKPTS: tl.constexpr,
        HAS_BOUNDARIES: tl.constexpr,
        HAS_DURATION_TRANSITIONS: tl.constexpr,
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        stride_tr_k,
        stride_tr_src,
        stride_tr_dst,
        stride_db_k,
        stride_db_c,
        stride_ps_b,
        stride_ps_t,
        stride_ps_c,
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
        stride_bp_b,
        stride_bp_t,
        stride_bp_c,
    ):
        """Max semiring with backpointer tracking for O(T) traceback."""
        NEG_INF: tl.constexpr = -1e9

        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        c_dst_idx = tl.arange(0, C_PAD)[:, None]
        c_src_idx = tl.arange(0, C_PAD)[None, :]
        c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)

        seq_len = tl.load(lengths_ptr + batch_idx)

        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_base = ring_ptr + batch_idx * stride_ring_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b
        bp_base = batch_idx * stride_bp_b

        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b

        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx * stride_tr_dst + c_src_idx * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )

        final_alpha = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

        for t in tl.range(1, T + 1):
            active = t <= seq_len
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=tl.float32)

            # Initialize backpointer tracking for this timestep
            best_k_t = tl.zeros([C_PAD], dtype=tl.int32)
            best_c_src_t = tl.zeros([C_PAD], dtype=tl.int32)

            for k in tl.range(1, tl.maximum(K, 2)):
                k_valid = (k <= t) & (k <= tl.maximum(K - 1, 1))
                start_pos = t - k
                ring_k_idx = start_pos % K

                alpha_prev = tl.load(
                    ring_base + ring_k_idx * stride_ring_k + c_idx * stride_ring_c,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )

                cum_end = tl.load(
                    cum_scores_base + t * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                cum_start = tl.load(
                    cum_scores_base + start_pos * stride_cs_t + c_idx * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                content_score = cum_end - cum_start
                dur_idx = tl.minimum(k, K - 1)
                dur_bias = tl.load(
                    duration_bias_ptr + dur_idx * stride_db_k + c_idx * stride_db_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )
                segment_score = content_score + dur_bias

                if HAS_BOUNDARIES:
                    start_score = tl.load(
                        proj_start_base + start_pos * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    end_pos_boundary = t - 1
                    end_score = tl.load(
                        proj_end_base + end_pos_boundary * stride_ps_t + c_idx * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    segment_score = segment_score + start_score + end_score

                if HAS_DURATION_TRANSITIONS:
                    transition_block = tl.load(
                        transition_ptr
                        + k * stride_tr_k
                        + c_dst_idx * stride_tr_dst
                        + c_src_idx * stride_tr_src,
                        mask=c_mask_2d,
                        other=0.0,
                    )

                edge_block = segment_score[:, None] + transition_block

                scores = alpha_prev[None, :] + edge_block
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Max over c_src WITH argmax tracking
                score_for_k = tl.max(scores, axis=1)
                argmax_c_src = tl.argmax(scores, axis=1)
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Track which k wins for each c_dest
                better_mask = score_for_k > alpha_t
                best_k_t = tl.where(better_mask, k, best_k_t)
                best_c_src_t = tl.where(better_mask, argmax_c_src, best_c_src_t)

                # Max over k
                alpha_t = tl.maximum(alpha_t, score_for_k)

            alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # Store backpointers at position t-1 (0-indexed)
            bp_pos = t - 1
            tl.store(
                bp_k_ptr + bp_base + bp_pos * stride_bp_t + c_idx * stride_bp_c,
                best_k_t,
                mask=active & c_mask,
            )
            tl.store(
                bp_c_ptr + bp_base + bp_pos * stride_bp_t + c_idx * stride_bp_c,
                best_c_src_t,
                mask=active & c_mask,
            )

            # Store to live ring buffer
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Save checkpoint at interval boundaries
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
                for k_save in tl.range(0, K):
                    ring_val = tl.load(
                        ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    save_mask = (ckpt_idx < NUM_CKPTS) & c_mask
                    tl.store(
                        ring_ckpt_base
                        + ckpt_idx * stride_ckpt_n
                        + k_save * stride_ckpt_k
                        + c_idx * stride_ckpt_c,
                        ring_val,
                        mask=save_mask,
                    )

            is_final = t == seq_len
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # Max over labels with argmax for final label
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        partition = tl.max(final_alpha_masked, axis=0)
        final_label = tl.argmax(final_alpha_masked, axis=0)

        tl.store(out_ptr + batch_idx, partition)
        tl.store(final_labels_ptr + batch_idx, final_label)

    def launch_streaming_triton_kernel(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        checkpoint_interval: int = None,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
        num_warps: int = 4,
        validate_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Launch Triton forward kernel with buffer allocation.

        Args:
            cum_scores: Shape (batch, T+1, C).
            transition: Shape (C, C) or (K, C, C).
            duration_bias: Shape (K, C).
            lengths: Shape (batch,).
            K: Max segment duration.
            semiring: "log" or "max".
            checkpoint_interval: Ring buffer save interval. Default: sqrt(T*K).
            proj_start, proj_end: Optional boundary scores (batch, T, C).
            num_warps: Warps per block (2-8).
            validate_cache: Validate Triton cache.

        Returns:
            (partition, ring_checkpoints, checkpoint_interval)
        """
        from .triton_cache import TritonConfig, update_cache_sentinel, validate_triton_cache

        # Validate cache if requested
        if validate_cache:
            config = TritonConfig(num_warps=num_warps)
            validate_triton_cache(config)
            update_cache_sentinel(config)
        batch, T_plus_1, C = cum_scores.shape
        T = T_plus_1 - 1
        device = cum_scores.device
        dtype = cum_scores.dtype

        # Compute checkpoint interval if not provided
        if checkpoint_interval is None:
            checkpoint_interval = _compute_checkpoint_interval(T, K)
        else:
            checkpoint_interval = max(checkpoint_interval, K)

        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval

        # Pad C to next power of 2
        C_PAD = _next_power_of_2(C)

        # Determine if boundaries are provided
        has_boundaries = proj_start is not None and proj_end is not None

        has_duration_transitions = transition.ndim == 3

        # Ensure inputs are contiguous
        cum_scores = cum_scores.contiguous()
        transition = transition.contiguous()
        duration_bias = duration_bias.contiguous()
        lengths = lengths.contiguous()

        # Handle boundary projections
        if has_boundaries:
            proj_start = proj_start.contiguous()
            proj_end = proj_end.contiguous()
            stride_ps_b, stride_ps_t, stride_ps_c = proj_start.stride()
        else:
            # Create dummy tensor for stride calculation (won't be accessed)
            proj_start = cum_scores[:, :T, :]  # Reuse cum_scores memory, won't be accessed
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0  # Strides don't matter when not used

        # Allocate outputs
        partition = torch.empty(batch, device=device, dtype=dtype)

        # Live ring buffer (will be L1/L2 cached for small K*C)
        # Initialize to NEG_INF, then set k=0 to 0.0 (initial alpha state)
        ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)
        ring_buffer[:, 0, :C] = 0.0  # alpha[0, c] = 0.0 for all valid labels

        # Checkpoint storage for backward pass
        # Initialize to NEG_INF, then set checkpoint 0, k=0 to 0.0
        ring_checkpoints = torch.full(
            (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=dtype
        )
        ring_checkpoints[:, 0, 0, :C] = 0.0  # Initial state at checkpoint 0

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()

        # Handle transition strides for both (C, C) and (K, C, C)
        if has_duration_transitions:
            stride_tr_k, stride_tr_src, stride_tr_dst = transition.stride()
        else:
            stride_tr_k = 0  # Not used for static transitions
            stride_tr_src, stride_tr_dst = transition.stride()

        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ring_b, stride_ring_k, stride_ring_c = ring_buffer.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_checkpoints.stride()

        # Launch kernel with device context for multi-GPU support
        grid = (batch,)
        kernel = (
            semi_crf_streaming_scan_kernel
            if semiring == "log"
            else semi_crf_streaming_scan_kernel_max
        )
        with torch.cuda.device(device):
            kernel[grid](
                cum_scores,
                transition,
                duration_bias,
                lengths,
                proj_start,
                proj_end,
                partition,
                ring_buffer,
                ring_checkpoints,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                num_checkpoints,
                has_boundaries,  # HAS_BOUNDARIES constexpr
                has_duration_transitions,  # HAS_DURATION_TRANSITIONS constexpr
                stride_cs_b,
                stride_cs_t,
                stride_cs_c,
                stride_tr_k,
                stride_tr_src,
                stride_tr_dst,
                stride_db_k,
                stride_db_c,
                stride_ps_b,
                stride_ps_t,
                stride_ps_c,
                stride_ring_b,
                stride_ring_k,
                stride_ring_c,
                stride_ckpt_b,
                stride_ckpt_n,
                stride_ckpt_k,
                stride_ckpt_c,
                num_warps=num_warps,
            )

        # Trim padding from checkpoints for return
        ring_checkpoints = ring_checkpoints[:, :, :, :C]

        return partition, ring_checkpoints, checkpoint_interval

    def launch_streaming_triton_kernel_max_bp(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        checkpoint_interval: int = None,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
        num_warps: int = 4,
        validate_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Launch max kernel with backpointer tracking for O(T) traceback.

        Returns:
            (viterbi_scores, bp_k, bp_c, final_labels)
        """
        from .triton_cache import TritonConfig, update_cache_sentinel, validate_triton_cache

        # Validate cache if requested
        if validate_cache:
            config = TritonConfig(num_warps=num_warps)
            validate_triton_cache(config)
            update_cache_sentinel(config)
        batch, T_plus_1, C = cum_scores.shape
        T = T_plus_1 - 1
        device = cum_scores.device
        dtype = cum_scores.dtype

        # Compute checkpoint interval if not provided
        if checkpoint_interval is None:
            checkpoint_interval = _compute_checkpoint_interval(T, K)
        else:
            checkpoint_interval = max(checkpoint_interval, K)

        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval

        # Pad C to next power of 2
        C_PAD = _next_power_of_2(C)

        # Determine if boundaries are provided
        has_boundaries = proj_start is not None and proj_end is not None

        # Determine if duration-dependent transitions
        has_duration_transitions = transition.ndim == 3

        # Ensure inputs are contiguous
        cum_scores = cum_scores.contiguous()
        transition = transition.contiguous()
        duration_bias = duration_bias.contiguous()
        lengths = lengths.contiguous()

        # Handle boundary projections
        if has_boundaries:
            proj_start = proj_start.contiguous()
            proj_end = proj_end.contiguous()
            stride_ps_b, stride_ps_t, stride_ps_c = proj_start.stride()
        else:
            proj_start = cum_scores[:, :T, :]
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0

        # Allocate outputs
        partition = torch.empty(batch, device=device, dtype=dtype)
        bp_k = torch.zeros((batch, T, C), device=device, dtype=torch.int32)
        bp_c = torch.zeros((batch, T, C), device=device, dtype=torch.int32)
        final_labels = torch.zeros(batch, device=device, dtype=torch.int64)

        # Ring buffer
        ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=dtype)
        ring_buffer[:, 0, :C] = 0.0

        # Checkpoint storage
        ring_checkpoints = torch.full(
            (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=dtype
        )
        ring_checkpoints[:, 0, 0, :C] = 0.0

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()

        if has_duration_transitions:
            stride_tr_k, stride_tr_src, stride_tr_dst = transition.stride()
        else:
            stride_tr_k = 0
            stride_tr_src, stride_tr_dst = transition.stride()

        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ring_b, stride_ring_k, stride_ring_c = ring_buffer.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_checkpoints.stride()
        stride_bp_b, stride_bp_t, stride_bp_c = bp_k.stride()

        # Launch kernel
        grid = (batch,)
        with torch.cuda.device(device):
            semi_crf_streaming_scan_kernel_max_bp[grid](
                cum_scores,
                transition,
                duration_bias,
                lengths,
                proj_start,
                proj_end,
                partition,
                ring_buffer,
                ring_checkpoints,
                bp_k,
                bp_c,
                final_labels,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                num_checkpoints,
                has_boundaries,
                has_duration_transitions,
                stride_cs_b,
                stride_cs_t,
                stride_cs_c,
                stride_tr_k,
                stride_tr_src,
                stride_tr_dst,
                stride_db_k,
                stride_db_c,
                stride_ps_b,
                stride_ps_t,
                stride_ps_c,
                stride_ring_b,
                stride_ring_k,
                stride_ring_c,
                stride_ckpt_b,
                stride_ckpt_n,
                stride_ckpt_k,
                stride_ckpt_c,
                stride_bp_b,
                stride_bp_t,
                stride_bp_c,
                num_warps=num_warps,
            )

        return partition, bp_k, bp_c, final_labels

    def semi_crf_streaming_viterbi_triton(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Triton Viterbi with backpointers.

        Returns: (viterbi_scores, bp_k, bp_c, final_labels)
        """
        return launch_streaming_triton_kernel_max_bp(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            checkpoint_interval=None,
            proj_start=proj_start,
            proj_end=proj_end,
        )
