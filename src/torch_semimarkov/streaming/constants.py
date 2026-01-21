r"""Shared constants for streaming Semi-CRF.

This module defines constants used across the streaming Semi-CRF implementation.

Attributes:
    NEG_INF (float): Negative infinity approximation used for log-space computations.
        Set to ``-1e9`` to avoid numerical issues with ``float('-inf')``.
"""

NEG_INF = -1e9
