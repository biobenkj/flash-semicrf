"""Tests for input validation utilities."""

import pytest
import torch

from flash_semicrf.validation import (
    validate_cum_scores,
    validate_device_consistency,
    validate_hidden_states,
    validate_labels,
    validate_lengths,
    validate_streaming_shapes,
)


class TestValidateHiddenStates:
    """Tests for validate_hidden_states."""

    def test_valid_3d_tensor(self):
        """Valid 3D tensor should pass."""
        hidden = torch.randn(2, 100, 64)
        validate_hidden_states(hidden)  # Should not raise

    def test_2d_tensor_raises(self):
        """2D tensor should raise ValueError."""
        hidden = torch.randn(100, 64)
        with pytest.raises(ValueError, match="must be 3D"):
            validate_hidden_states(hidden)

    def test_4d_tensor_raises(self):
        """4D tensor should raise ValueError."""
        hidden = torch.randn(2, 100, 64, 32)
        with pytest.raises(ValueError, match="must be 3D"):
            validate_hidden_states(hidden)

    def test_nan_raises(self):
        """Tensor with NaN should raise ValueError."""
        hidden = torch.randn(2, 100, 64)
        hidden[0, 50, 32] = float("nan")
        with pytest.raises(ValueError, match="contains NaN"):
            validate_hidden_states(hidden)

    def test_inf_raises(self):
        """Tensor with Inf should raise ValueError."""
        hidden = torch.randn(2, 100, 64)
        hidden[1, 25, 16] = float("inf")
        with pytest.raises(ValueError, match="contains Inf"):
            validate_hidden_states(hidden)

    def test_nan_check_disabled(self):
        """NaN check can be disabled."""
        hidden = torch.randn(2, 100, 64)
        hidden[0, 50, 32] = float("nan")
        validate_hidden_states(hidden, check_nan=False)  # Should not raise

    def test_inf_check_disabled(self):
        """Inf check can be disabled."""
        hidden = torch.randn(2, 100, 64)
        hidden[1, 25, 16] = float("inf")
        validate_hidden_states(hidden, check_inf=False)  # Should not raise

    def test_custom_name_in_error(self):
        """Custom name should appear in error message."""
        hidden = torch.randn(100, 64)
        with pytest.raises(ValueError, match="encoder_output"):
            validate_hidden_states(hidden, name="encoder_output")


class TestValidateLengths:
    """Tests for validate_lengths."""

    def test_valid_lengths(self):
        """Valid lengths should pass."""
        lengths = torch.tensor([100, 100])
        validate_lengths(lengths, max_length=100)  # Should not raise

    def test_2d_tensor_raises(self):
        """2D tensor should raise ValueError."""
        lengths = torch.tensor([[100, 100]])
        with pytest.raises(ValueError, match="must be 1D"):
            validate_lengths(lengths, max_length=100)

    def test_batch_size_mismatch_raises(self):
        """Wrong batch size should raise ValueError."""
        lengths = torch.tensor([100, 100])
        with pytest.raises(ValueError, match="batch size"):
            validate_lengths(lengths, max_length=100, batch_size=3)

    def test_zero_length_raises(self):
        """Zero length should raise ValueError."""
        lengths = torch.tensor([100, 0])
        with pytest.raises(ValueError, match="must be positive"):
            validate_lengths(lengths, max_length=100)

    def test_negative_length_raises(self):
        """Negative length should raise ValueError."""
        lengths = torch.tensor([100, -5])
        with pytest.raises(ValueError, match="must be positive"):
            validate_lengths(lengths, max_length=100)

    def test_exceeds_max_raises(self):
        """Length exceeding max should raise ValueError."""
        lengths = torch.tensor([100, 200])
        with pytest.raises(ValueError, match="cannot exceed T=100"):
            validate_lengths(lengths, max_length=100)

    def test_non_integer_dtype_with_integral_values_warns(self):
        """Non-integer dtype is accepted with warning when values are integral."""
        lengths = torch.tensor([100.0, 75.0], dtype=torch.float32)
        with pytest.warns(UserWarning, match="non-integer dtype"):
            validate_lengths(lengths, max_length=100)

    def test_non_integer_dtype_with_non_integral_values_raises(self):
        """Non-integer dtype must raise if any length value is non-integral."""
        lengths = torch.tensor([100.0, 75.5], dtype=torch.float32)
        with pytest.raises(ValueError, match="must contain integral values"):
            validate_lengths(lengths, max_length=100)

    def test_bool_dtype_raises(self):
        """Bool dtype must raise even though it's technically integral."""
        lengths = torch.tensor([True, True])
        with pytest.raises(ValueError, match="torch.bool"):
            validate_lengths(lengths, max_length=100)

    def test_complex_dtype_raises(self):
        """Complex dtype must raise."""
        lengths = torch.tensor([50.0 + 0j, 75.0 + 0j], dtype=torch.complex64)
        with pytest.raises(ValueError, match="complex dtype"):
            validate_lengths(lengths, max_length=100)

    def test_float_with_inf_raises(self):
        """Float lengths containing inf must raise."""
        lengths = torch.tensor([50.0, float("inf")], dtype=torch.float32)
        with pytest.raises(ValueError, match="non-finite"):
            validate_lengths(lengths, max_length=100)

    def test_float_with_nan_raises(self):
        """Float lengths containing NaN must raise."""
        lengths = torch.tensor([50.0, float("nan")], dtype=torch.float32)
        with pytest.raises(ValueError, match="non-finite"):
            validate_lengths(lengths, max_length=100)

    def test_float_integral_exceeding_max_raises(self):
        """Float lengths with integral values must still be bounds-checked."""
        lengths = torch.tensor([50.0, 200.0], dtype=torch.float32)
        with pytest.warns(UserWarning, match="non-integer dtype"):
            with pytest.raises(ValueError, match="cannot exceed T=100"):
                validate_lengths(lengths, max_length=100)


class TestValidateLabels:
    """Tests for validate_labels."""

    def test_valid_labels(self):
        """Valid labels should pass."""
        labels = torch.randint(0, 4, (2, 100))
        validate_labels(labels, num_classes=4)  # Should not raise

    def test_1d_tensor_raises(self):
        """1D tensor should raise ValueError."""
        labels = torch.randint(0, 4, (100,))
        with pytest.raises(ValueError, match="must be 2D"):
            validate_labels(labels, num_classes=4)

    def test_batch_size_mismatch_raises(self):
        """Wrong batch size should raise ValueError."""
        labels = torch.randint(0, 4, (2, 100))
        with pytest.raises(ValueError, match="batch size"):
            validate_labels(labels, num_classes=4, batch_size=3)

    def test_seq_length_mismatch_raises(self):
        """Wrong sequence length should raise ValueError."""
        labels = torch.randint(0, 4, (2, 100))
        with pytest.raises(ValueError, match="sequence length"):
            validate_labels(labels, num_classes=4, seq_length=50)

    def test_negative_label_raises(self):
        """Negative label should raise ValueError."""
        labels = torch.randint(0, 4, (2, 100))
        labels[0, 50] = -1
        with pytest.raises(ValueError, match="must be in"):
            validate_labels(labels, num_classes=4)

    def test_label_out_of_range_raises(self):
        """Label >= num_classes should raise ValueError."""
        labels = torch.randint(0, 4, (2, 100))
        labels[1, 25] = 4  # num_classes = 4, so 4 is out of range
        with pytest.raises(ValueError, match="must be in"):
            validate_labels(labels, num_classes=4)


class TestValidateCumScores:
    """Tests for validate_cum_scores."""

    def test_valid_cum_scores(self):
        """Valid cum_scores should pass."""
        cum_scores = torch.zeros(2, 101, 4)
        validate_cum_scores(cum_scores)  # Should not raise

    def test_2d_tensor_raises(self):
        """2D tensor should raise ValueError."""
        cum_scores = torch.zeros(101, 4)
        with pytest.raises(ValueError, match="must be 3D"):
            validate_cum_scores(cum_scores)

    def test_t_plus_1_too_small_raises(self):
        """T+1 < 2 should raise ValueError."""
        cum_scores = torch.zeros(2, 1, 4)  # T=0
        with pytest.raises(ValueError, match="T\\+1 dimension must be >= 2"):
            validate_cum_scores(cum_scores)

    def test_dtype_warning(self):
        """Non-float32 dtype should warn."""
        cum_scores = torch.zeros(2, 101, 4, dtype=torch.float16)
        with pytest.warns(UserWarning, match="should be float32"):
            validate_cum_scores(cum_scores)

    def test_dtype_warning_disabled(self):
        """dtype warning can be disabled."""
        cum_scores = torch.zeros(2, 101, 4, dtype=torch.float16)
        validate_cum_scores(cum_scores, warn_dtype=False)  # Should not warn


class TestValidateDeviceConsistency:
    """Tests for validate_device_consistency."""

    def test_same_device_passes(self):
        """Tensors on same device should pass."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 3)
        validate_device_consistency(t1, t2)  # Should not raise

    def test_single_tensor_passes(self):
        """Single tensor should pass."""
        t1 = torch.randn(2, 3)
        validate_device_consistency(t1)  # Should not raise

    def test_none_values_skipped(self):
        """None values should be skipped."""
        t1 = torch.randn(2, 3)
        validate_device_consistency(t1, None, t1)  # Should not raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_mismatch_raises(self):
        """Different devices should raise ValueError."""
        t_cpu = torch.randn(2, 3)
        t_cuda = torch.randn(2, 3, device="cuda")
        with pytest.raises(ValueError, match="Device mismatch"):
            validate_device_consistency(t_cpu, t_cuda)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_mismatch_with_names(self):
        """Device mismatch should include names in error."""
        t_cpu = torch.randn(2, 3)
        t_cuda = torch.randn(2, 3, device="cuda")
        with pytest.raises(ValueError, match="hidden_states"):
            validate_device_consistency(t_cpu, t_cuda, names=["hidden_states", "lengths"])


class TestCRFHeadValidation:
    """Integration tests for validation in CRF heads."""

    def test_forward_validates_hidden_states_shape(self):
        """forward() should reject wrong hidden_states shape."""
        from flash_semicrf import SemiMarkovCRFHead

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=64)
        hidden = torch.randn(100, 64)  # Missing batch dim
        lengths = torch.tensor([100])

        with pytest.raises(ValueError, match="must be 3D"):
            crf.forward(hidden, lengths)

    def test_forward_validates_lengths_bounds(self):
        """forward() should reject lengths > T."""
        from flash_semicrf import SemiMarkovCRFHead

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=64)
        hidden = torch.randn(2, 100, 64)
        lengths = torch.tensor([100, 200])  # 200 > T=100

        with pytest.raises(ValueError, match="cannot exceed"):
            crf.forward(hidden, lengths)

    def test_compute_loss_validates_labels(self):
        """compute_loss() should reject out-of-range labels."""
        from flash_semicrf import SemiMarkovCRFHead

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=64)
        hidden = torch.randn(2, 100, 64)
        lengths = torch.tensor([100, 100])
        labels = torch.randint(0, 10, (2, 100))  # Labels go up to 9, but num_classes=4

        with pytest.raises(ValueError, match="must be in"):
            crf.compute_loss(hidden, lengths, labels)

    def test_decode_validates_inputs(self):
        """decode() should validate inputs."""
        from flash_semicrf import SemiMarkovCRFHead

        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=64)
        hidden = torch.randn(100, 64)  # Missing batch dim
        lengths = torch.tensor([100])

        with pytest.raises(ValueError, match="must be 3D"):
            crf.decode(hidden, lengths)


class TestStreamingAPIValidation:
    """Tests for validation in streaming API."""

    def test_streaming_forward_validates_cum_scores(self):
        """semi_crf_streaming_forward() should validate cum_scores."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        cum_scores = torch.zeros(2, 1, 4)  # T=0, invalid
        transition = torch.randn(4, 4)
        duration_bias = torch.randn(8, 4)
        lengths = torch.tensor([1, 1])

        with pytest.raises(ValueError, match="T\\+1 dimension must be >= 2"):
            semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K=8)

    def test_streaming_forward_validates_lengths(self):
        """semi_crf_streaming_forward() should validate lengths."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        cum_scores = torch.zeros(2, 101, 4)  # T=100
        transition = torch.randn(4, 4)
        duration_bias = torch.randn(8, 4)
        lengths = torch.tensor([100, 200])  # 200 > T=100

        with pytest.raises(ValueError, match="cannot exceed"):
            semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K=8)

    def test_streaming_forward_validates_duration_bias_k(self):
        """semi_crf_streaming_forward() should reject K/duration_bias mismatch."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        cum_scores = torch.zeros(2, 101, 4)
        transition = torch.randn(4, 4)
        duration_bias = torch.randn(4, 4)  # K=4, but passing K=8
        lengths = torch.tensor([100, 100])

        with pytest.raises(ValueError, match="duration_bias"):
            semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K=8)

    def test_streaming_forward_validates_transition_3d_k(self):
        """semi_crf_streaming_forward() should reject K/transition mismatch."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        cum_scores = torch.zeros(2, 101, 4)
        transition = torch.randn(4, 4, 4)  # K=4, but passing K=8
        duration_bias = torch.randn(8, 4)
        lengths = torch.tensor([100, 100])

        with pytest.raises(ValueError, match="transition"):
            semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K=8)

    def test_streaming_forward_validates_transition_c(self):
        """semi_crf_streaming_forward() should reject C mismatch in transition."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        cum_scores = torch.zeros(2, 101, 4)  # C=4
        transition = torch.randn(6, 6)  # C=6
        duration_bias = torch.randn(8, 4)
        lengths = torch.tensor([100, 100])

        with pytest.raises(ValueError, match="transition"):
            semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K=8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_streaming_forward_triton_path_validates_shapes(self):
        """Triton path should reject mismatched shapes before kernel launch."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        cum_scores = torch.zeros(2, 101, 4, device="cuda", dtype=torch.float64)
        transition = torch.randn(4, 4, device="cuda", dtype=torch.float64)
        duration_bias = torch.randn(4, 4, device="cuda", dtype=torch.float64)  # K=4, but passing K=8
        lengths = torch.tensor([100, 100], device="cuda")

        with pytest.raises(ValueError, match="duration_bias"):
            semi_crf_streaming_forward(
                cum_scores, transition, duration_bias, lengths, K=8, use_triton=True
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_forward_launcher_validates_shapes(self):
        """Direct launcher call should reject mismatched shapes."""
        from flash_semicrf.streaming.triton_forward import launch_streaming_triton_kernel

        cum_scores = torch.zeros(2, 101, 4, device="cuda", dtype=torch.float64)
        transition = torch.randn(4, 4, device="cuda", dtype=torch.float64)
        duration_bias = torch.randn(4, 4, device="cuda", dtype=torch.float64)  # K=4, but passing K=8
        lengths = torch.tensor([100, 100], device="cuda")

        with pytest.raises(ValueError, match="duration_bias"):
            launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K=8
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_backward_launcher_validates_checkpoint_count(self):
        """Backward launcher should reject mismatched checkpoint counts."""
        from flash_semicrf.streaming.triton_backward import launch_streaming_triton_backward

        batch, T, C, K = 2, 100, 4, 8
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda", dtype=torch.float64)
        transition = torch.randn(C, C, device="cuda", dtype=torch.float64)
        duration_bias = torch.randn(K, C, device="cuda", dtype=torch.float64)
        lengths = torch.tensor([T, T], device="cuda")
        log_Z = torch.zeros(batch, device="cuda", dtype=torch.float64)
        grad_output = torch.ones(batch, device="cuda", dtype=torch.float64)
        checkpoint_interval = 16
        num_ckpts = (T + checkpoint_interval - 1) // checkpoint_interval

        ring_checkpoints = torch.zeros(batch, num_ckpts, K, C, device="cuda", dtype=torch.float64)
        log_norm_checkpoints = torch.zeros(batch, num_ckpts + 1, device="cuda", dtype=torch.float64)

        with pytest.raises(ValueError, match="checkpoint count must match"):
            launch_streaming_triton_backward(
                cum_scores, transition, duration_bias, lengths,
                log_Z, ring_checkpoints, log_norm_checkpoints,
                checkpoint_interval, grad_output,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_backward_launcher_validates_checkpoint_class_dim(self):
        """Backward launcher should reject mismatched checkpoint class dim."""
        from flash_semicrf.streaming.triton_backward import launch_streaming_triton_backward

        batch, T, C, K = 2, 100, 4, 8
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda", dtype=torch.float64)
        transition = torch.randn(C, C, device="cuda", dtype=torch.float64)
        duration_bias = torch.randn(K, C, device="cuda", dtype=torch.float64)
        lengths = torch.tensor([T, T], device="cuda")
        log_Z = torch.zeros(batch, device="cuda", dtype=torch.float64)
        grad_output = torch.ones(batch, device="cuda", dtype=torch.float64)
        checkpoint_interval = 16
        num_ckpts = (T + checkpoint_interval - 1) // checkpoint_interval

        ring_checkpoints = torch.zeros(batch, num_ckpts, K, C + 1, device="cuda", dtype=torch.float64)
        log_norm_checkpoints = torch.zeros(batch, num_ckpts, device="cuda", dtype=torch.float64)

        with pytest.raises(ValueError, match="class dim must equal"):
            launch_streaming_triton_backward(
                cum_scores, transition, duration_bias, lengths,
                log_Z, ring_checkpoints, log_norm_checkpoints,
                checkpoint_interval, grad_output,
            )


class TestValidateStreamingShapes:
    """Tests for validate_streaming_shapes."""

    def test_valid_static_transition(self):
        """Valid shapes with static transition should pass."""
        validate_streaming_shapes(
            K=8, C=4, batch=2, T=100,
            transition=torch.randn(4, 4),
            duration_bias=torch.randn(8, 4),
        )

    def test_valid_duration_dependent_transition(self):
        """Valid shapes with duration-dependent transition should pass."""
        validate_streaming_shapes(
            K=8, C=4, batch=2, T=100,
            transition=torch.randn(8, 4, 4),
            duration_bias=torch.randn(8, 4),
        )

    def test_valid_with_boundaries(self):
        """Valid shapes with boundary projections should pass."""
        validate_streaming_shapes(
            K=8, C=4, batch=2, T=100,
            transition=torch.randn(4, 4),
            duration_bias=torch.randn(8, 4),
            proj_start=torch.randn(2, 100, 4),
            proj_end=torch.randn(2, 100, 4),
        )

    # --- K validation ---

    def test_k_zero_raises(self):
        """K=0 should raise ValueError."""
        with pytest.raises(ValueError, match="K must be a positive integer"):
            validate_streaming_shapes(
                K=0, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(1, 4),
            )

    def test_k_negative_raises(self):
        """Negative K should raise ValueError."""
        with pytest.raises(ValueError, match="K must be a positive integer"):
            validate_streaming_shapes(
                K=-1, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(1, 4),
            )

    def test_k_float_raises(self):
        """Float K should raise ValueError."""
        with pytest.raises(ValueError, match="K must be a positive integer"):
            validate_streaming_shapes(
                K=2.5, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(2, 4),
            )

    def test_k_bool_raises(self):
        """Bool K should raise ValueError (bool is subclass of int)."""
        with pytest.raises(ValueError, match="K must be a positive integer"):
            validate_streaming_shapes(
                K=True, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(1, 4),
            )

    def test_k_numpy_int_passes(self):
        """numpy.int64 should be accepted as integral."""
        np = pytest.importorskip("numpy")
        validate_streaming_shapes(
            K=np.int64(8), C=4, batch=2, T=100,
            transition=torch.randn(4, 4),
            duration_bias=torch.randn(8, 4),
        )

    # --- duration_bias validation ---

    def test_duration_bias_wrong_ndim_raises(self):
        """1D duration_bias should raise."""
        with pytest.raises(ValueError, match="duration_bias must be 2D"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(32),
            )

    def test_duration_bias_k_mismatch_raises(self):
        """duration_bias.shape[0] != K should raise."""
        with pytest.raises(ValueError, match=r"duration_bias.shape\[0\] must equal K=8"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(4, 4),
            )

    def test_duration_bias_c_mismatch_raises(self):
        """duration_bias.shape[1] != C should raise."""
        with pytest.raises(ValueError, match=r"duration_bias.shape\[1\] must equal C=4"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(8, 6),
            )

    # --- transition validation ---

    def test_transition_1d_raises(self):
        """1D transition should raise."""
        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(16),
                duration_bias=torch.randn(8, 4),
            )

    def test_transition_4d_raises(self):
        """4D transition should raise."""
        with pytest.raises(ValueError, match="must be 2D.*or 3D"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(8, 4, 4, 4),
                duration_bias=torch.randn(8, 4),
            )

    def test_transition_2d_wrong_shape_raises(self):
        """(C, C') with C'!=C should raise."""
        with pytest.raises(ValueError, match=r"transition must be \(C, C\)"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 6),
                duration_bias=torch.randn(8, 4),
            )

    def test_transition_3d_k_mismatch_raises(self):
        """transition.shape[0] != K for 3D transition should raise."""
        with pytest.raises(ValueError, match="must equal K=8"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 4, 4),
                duration_bias=torch.randn(8, 4),
            )

    def test_transition_3d_c_mismatch_raises(self):
        """transition (K, C, C') with wrong C dims should raise."""
        with pytest.raises(ValueError, match=r"must be \(K, C, C\)"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(8, 4, 6),
                duration_bias=torch.randn(8, 4),
            )

    # --- proj_start / proj_end validation ---

    def test_proj_start_wrong_ndim_raises(self):
        """2D proj_start should raise."""
        with pytest.raises(ValueError, match="proj_start must be 3D"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(8, 4),
                proj_start=torch.randn(200, 4),
            )

    def test_proj_start_wrong_shape_raises(self):
        """proj_start with wrong T should raise."""
        with pytest.raises(ValueError, match="proj_start shape must be"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(8, 4),
                proj_start=torch.randn(2, 50, 4),
            )

    def test_proj_end_wrong_ndim_raises(self):
        """2D proj_end should raise."""
        with pytest.raises(ValueError, match="proj_end must be 3D"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(8, 4),
                proj_end=torch.randn(200, 4),
            )

    def test_proj_end_wrong_shape_raises(self):
        """proj_end with wrong C should raise."""
        with pytest.raises(ValueError, match="proj_end shape must be"):
            validate_streaming_shapes(
                K=8, C=4, batch=2, T=100,
                transition=torch.randn(4, 4),
                duration_bias=torch.randn(8, 4),
                proj_end=torch.randn(2, 100, 6),
            )
