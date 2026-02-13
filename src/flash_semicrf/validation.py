r"""Input validation utilities for Semi-Markov CRF."""

import numbers
import warnings
from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "validate_hidden_states",
    "validate_lengths",
    "validate_labels",
    "validate_cum_scores",
    "validate_device_consistency",
    "validate_streaming_shapes",
]


def validate_hidden_states(
    hidden_states: Tensor,
    name: str = "hidden_states",
    check_nan: bool = True,
    check_inf: bool = True,
) -> None:
    r"""Validate hidden_states tensor shape and values.

    Args:
        hidden_states (Tensor): Input tensor, expected shape :math:`(\text{batch}, T, \text{features})`.
        name (str, optional): Name for error messages. Default: ``"hidden_states"``
        check_nan (bool, optional): Check for NaN values. Default: ``True``
        check_inf (bool, optional): Check for Inf values. Default: ``True``

    Raises:
        ValueError: If shape is not 3D or contains NaN/Inf when checked.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"{name} must be 3D (batch, T, features), got {hidden_states.ndim}D")

    if check_nan and torch.isnan(hidden_states).any():
        raise ValueError(f"{name} contains NaN values")

    if check_inf and torch.isinf(hidden_states).any():
        raise ValueError(f"{name} contains Inf values")


def validate_lengths(
    lengths: Tensor,
    max_length: int,
    batch_size: Optional[int] = None,
    name: str = "lengths",
) -> None:
    r"""Validate sequence lengths tensor.

    Args:
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        max_length (int): Maximum allowed length (T).
        batch_size (int, optional): Expected batch size. Default: ``None``
        name (str, optional): Name for error messages. Default: ``"lengths"``

    Raises:
        ValueError: If not 1D, non-positive, exceeds max_length, or wrong batch size.
    """
    if lengths.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {lengths.ndim}D")

    if batch_size is not None and lengths.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size {lengths.shape[0]} doesn't match expected {batch_size}"
        )

    # Check for non-positive lengths
    if (lengths <= 0).any():
        raise ValueError(f"{name} must be positive, got min={lengths.min().item()}")

    # Check for lengths exceeding max
    if (lengths > max_length).any():
        raise ValueError(f"{name} cannot exceed T={max_length}, got max={lengths.max().item()}")


def validate_labels(
    labels: Tensor,
    num_classes: int,
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    name: str = "labels",
) -> None:
    r"""Validate label tensor shape and value range.

    Args:
        labels (Tensor): Labels of shape :math:`(\text{batch}, T)`.
        num_classes (int): Number of valid classes (C).
        batch_size (int, optional): Expected batch size. Default: ``None``
        seq_length (int, optional): Expected sequence length. Default: ``None``
        name (str, optional): Name for error messages. Default: ``"labels"``

    Raises:
        ValueError: If not 2D, wrong shape, or values outside ``[0, C)``.
    """
    if labels.ndim != 2:
        raise ValueError(f"{name} must be 2D (batch, T), got {labels.ndim}D")

    if batch_size is not None and labels.shape[0] != batch_size:
        raise ValueError(f"{name} batch size {labels.shape[0]} doesn't match expected {batch_size}")

    if seq_length is not None and labels.shape[1] != seq_length:
        raise ValueError(
            f"{name} sequence length {labels.shape[1]} doesn't match expected {seq_length}"
        )

    # Check value range
    min_val = labels.min().item()
    max_val = labels.max().item()
    if min_val < 0 or max_val >= num_classes:
        raise ValueError(f"{name} must be in [0, {num_classes}), got range [{min_val}, {max_val}]")


def validate_cum_scores(
    cum_scores: Tensor,
    name: str = "cum_scores",
    warn_dtype: bool = True,
    check_leading_zeros: bool = False,
) -> None:
    r"""Validate cumulative scores tensor.

    Args:
        cum_scores (Tensor): Cumulative scores of shape :math:`(\text{batch}, T+1, C)`.
        name (str, optional): Name for error messages. Default: ``"cum_scores"``
        warn_dtype (bool, optional): Warn if not float32/float64. Default: ``True``
        check_leading_zeros (bool, optional): Warn if ``[:, 0, :]`` not zero. Default: ``False``

    Raises:
        ValueError: If not 3D or T+1 dimension < 2.
    """
    if cum_scores.ndim != 3:
        raise ValueError(f"{name} must be 3D (batch, T+1, C), got {cum_scores.ndim}D")

    batch, T_plus_1, C = cum_scores.shape
    if T_plus_1 < 2:
        raise ValueError(f"{name} T+1 dimension must be >= 2 (need at least T=1), got {T_plus_1}")

    if warn_dtype and cum_scores.dtype not in (torch.float32, torch.float64):
        warnings.warn(
            f"{name} should be float32 or float64 for numerical stability at long sequences, "
            f"got {cum_scores.dtype}. Float64 is recommended; all Triton kernels compute "
            f"internally in float64.",
            UserWarning,
            stacklevel=3,
        )

    if check_leading_zeros:
        leading = cum_scores[:, 0, :]
        if not torch.allclose(leading, torch.zeros_like(leading)):
            warnings.warn(
                f"{name}[:, 0, :] should be zeros (cumsum convention), "
                f"got max abs value {leading.abs().max().item():.2e}",
                UserWarning,
                stacklevel=3,
            )


def validate_device_consistency(
    *tensors: Tensor,
    names: Optional[list[str]] = None,
) -> None:
    r"""Validate all tensors are on the same device.

    Args:
        *tensors (Tensor): Tensors to check. ``None`` values are skipped.
        names (list[str], optional): Names for error messages. Default: ``None``

    Raises:
        ValueError: If tensors are on different devices.
    """
    # Filter out None values
    valid_tensors = [t for t in tensors if t is not None]
    if len(valid_tensors) <= 1:
        return  # Nothing to compare

    # Get devices
    devices = [t.device for t in valid_tensors]

    # Check if all devices are the same
    if len({str(d) for d in devices}) > 1:
        if names is not None:
            valid_names = [n for n, t in zip(names, tensors, strict=False) if t is not None]
            device_map = dict(zip(valid_names, devices, strict=True))
        else:
            device_map = {f"tensor_{i}": d for i, d in enumerate(devices)}
        raise ValueError(f"Device mismatch: {device_map}")


def validate_streaming_shapes(
    K: int,
    C: int,
    batch: int,
    T: int,
    transition: Tensor,
    duration_bias: Tensor,
    proj_start: Optional[Tensor] = None,
    proj_end: Optional[Tensor] = None,
) -> None:
    r"""Validate shape consistency of streaming Semi-CRF inputs.

    Ensures that ``K``, ``duration_bias``, ``transition``, and optional
    boundary projections are mutually consistent with the ``(batch, T+1, C)``
    shape of ``cum_scores``. This prevents undefined Triton memory access
    on shape-mismatched inputs.

    Args:
        K (int): Maximum segment duration. Must be a positive integer
            (not ``bool``).
        C (int): Number of classes (from ``cum_scores.shape[2]``).
        batch (int): Batch size (from ``cum_scores.shape[0]``).
        T (int): Sequence length (``cum_scores.shape[1] - 1``).
        transition (Tensor): Transition scores of shape :math:`(C, C)` for
            static transitions, or :math:`(K, C, C)` for duration-dependent.
        duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``

    Raises:
        ValueError: If any shape contract is violated.
    """
    # K: integral, not bool, >= 1
    if isinstance(K, bool) or not isinstance(K, numbers.Integral):
        raise ValueError(f"K must be a positive integer, got {type(K).__name__}({K!r})")
    if K < 1:
        raise ValueError(f"K must be a positive integer, got {K}")

    # duration_bias: (K, C)
    if duration_bias.ndim != 2:
        raise ValueError(
            f"duration_bias must be 2D (K, C), got {duration_bias.ndim}D"
        )
    if duration_bias.shape[0] != K:
        raise ValueError(
            f"duration_bias.shape[0] must equal K={K}, got {duration_bias.shape[0]}"
        )
    if duration_bias.shape[1] != C:
        raise ValueError(
            f"duration_bias.shape[1] must equal C={C}, got {duration_bias.shape[1]}"
        )

    # transition: (C, C) or (K, C, C)
    if transition.ndim == 2:
        if transition.shape != (C, C):
            raise ValueError(
                f"transition must be (C, C) = ({C}, {C}), got {tuple(transition.shape)}"
            )
    elif transition.ndim == 3:
        if transition.shape[0] != K:
            raise ValueError(
                f"duration-dependent transition.shape[0] must equal K={K}, "
                f"got {transition.shape[0]}"
            )
        if transition.shape[1] != C or transition.shape[2] != C:
            raise ValueError(
                f"transition must be (K, C, C) = ({K}, {C}, {C}), "
                f"got {tuple(transition.shape)}"
            )
    else:
        raise ValueError(
            f"transition must be 2D (C, C) or 3D (K, C, C), got {transition.ndim}D"
        )

    # proj_start: (batch, T, C) if provided
    if proj_start is not None:
        if proj_start.ndim != 3:
            raise ValueError(
                f"proj_start must be 3D (batch, T, C), got {proj_start.ndim}D"
            )
        if proj_start.shape != (batch, T, C):
            raise ValueError(
                f"proj_start shape must be (batch, T, C) = ({batch}, {T}, {C}), "
                f"got {tuple(proj_start.shape)}"
            )

    # proj_end: (batch, T, C) if provided
    if proj_end is not None:
        if proj_end.ndim != 3:
            raise ValueError(
                f"proj_end must be 3D (batch, T, C), got {proj_end.ndim}D"
            )
        if proj_end.shape != (batch, T, C):
            raise ValueError(
                f"proj_end shape must be (batch, T, C) = ({batch}, {T}, {C}), "
                f"got {tuple(proj_end.shape)}"
            )
