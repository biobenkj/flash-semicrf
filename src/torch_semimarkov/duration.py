r"""Duration distribution classes for Semi-Markov CRF."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor


class DurationDistribution(nn.Module, ABC):
    r"""Base class for duration distributions.

    Subclasses produce a :math:`(K, C)` log-space bias tensor where
    ``bias[k, c]`` is the log-probability (up to normalization) of
    duration ``k`` for class ``c``.

    Args:
        max_duration (int): Maximum segment duration (K).
        num_classes (int): Number of label classes (C).
    """

    def __init__(self, max_duration: int, num_classes: int):
        super().__init__()
        self.max_duration = max_duration
        self.num_classes = num_classes

    @abstractmethod
    def forward(self) -> Tensor:
        r"""Compute duration bias tensor.

        Returns:
            Tensor: Duration bias of shape :math:`(K, C)`.
        """
        raise NotImplementedError


class LearnedDuration(DurationDistribution):
    r"""Fully learned (K, C) bias. Default behavior."""

    def __init__(self, max_duration: int, num_classes: int, init_std: float = 0.1):
        super().__init__(max_duration, num_classes)
        self.duration_bias = nn.Parameter(torch.randn(max_duration, num_classes) * init_std)

    def forward(self) -> Tensor:
        return self.duration_bias


class GeometricDuration(DurationDistribution):
    r"""Geometric: :math:`P(k) \propto p(1-p)^{k-1}`. Numerically stable."""

    def __init__(
        self,
        max_duration: int,
        num_classes: int,
        init_logit: float = 0.0,
        learn_rate: bool = True,
    ):
        super().__init__(max_duration, num_classes)
        if learn_rate:
            self.logit_p = nn.Parameter(torch.full((num_classes,), init_logit))
        else:
            self.register_buffer("logit_p", torch.full((num_classes,), init_logit))

    def forward(self) -> Tensor:
        p = torch.sigmoid(self.logit_p)
        k = torch.arange(1, self.max_duration + 1, device=p.device, dtype=p.dtype)
        log_p = torch.log(p + 1e-8)
        log_1_minus_p = torch.log(1 - p + 1e-8)
        return log_p.unsqueeze(0) + (k - 1).unsqueeze(1) * log_1_minus_p.unsqueeze(0)


class NegativeBinomialDuration(DurationDistribution):
    r"""Negative binomial: :math:`P(k) \propto \binom{k+r-2}{k-1} p^r (1-p)^{k-1}`.

    When r=1, reduces to geometric. Very small r (init_log_r < -10) may cause
    numerical instability via lgamma overflow.
    """

    def __init__(
        self,
        max_duration: int,
        num_classes: int,
        init_logit: float = 0.0,
        init_log_r: float = 0.0,
        learn_rate: bool = True,
        learn_shape: bool = True,
    ):
        super().__init__(max_duration, num_classes)
        if learn_rate:
            self.logit_p = nn.Parameter(torch.full((num_classes,), init_logit))
        else:
            self.register_buffer("logit_p", torch.full((num_classes,), init_logit))

        if learn_shape:
            self.log_r = nn.Parameter(torch.full((num_classes,), init_log_r))
        else:
            self.register_buffer("log_r", torch.full((num_classes,), init_log_r))

    def forward(self) -> Tensor:
        p = torch.sigmoid(self.logit_p)
        r = torch.exp(self.log_r) + 1e-8
        k = torch.arange(1, self.max_duration + 1, device=p.device, dtype=p.dtype)

        k_expanded = k.unsqueeze(1)
        r_expanded = r.unsqueeze(0)
        log_binom = (
            torch.lgamma(k_expanded + r_expanded - 1)
            - torch.lgamma(r_expanded)
            - torch.lgamma(k_expanded)
        )
        log_p = torch.log(p + 1e-8)
        log_1_minus_p = torch.log(1 - p + 1e-8)
        log_prob = (
            log_binom
            + r_expanded * log_p.unsqueeze(0)
            + (k_expanded - 1) * log_1_minus_p.unsqueeze(0)
        )

        # Warn if numerical instability detected
        if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
            non_finite_count = (~torch.isfinite(log_prob)).sum().item()
            if non_finite_count > 0:
                r_min = r.min().item()
                warnings.warn(
                    f"NegativeBinomialDuration: {non_finite_count} non-finite values "
                    f"(min r={r_min:.2e}). Use larger init_log_r or GeometricDuration.",
                    stacklevel=2,
                )
        return log_prob


class PoissonDuration(DurationDistribution):
    r"""Poisson: :math:`P(k) \propto \lambda^k / k!`. Shifted (k starts at 1)."""

    def __init__(
        self,
        max_duration: int,
        num_classes: int,
        init_log_lambda: float = 1.0,
        learn_rate: bool = True,
    ):
        super().__init__(max_duration, num_classes)
        if learn_rate:
            self.log_lambda = nn.Parameter(torch.full((num_classes,), init_log_lambda))
        else:
            self.register_buffer("log_lambda", torch.full((num_classes,), init_log_lambda))

    def forward(self) -> Tensor:
        lam = torch.exp(self.log_lambda)
        k = torch.arange(1, self.max_duration + 1, device=lam.device, dtype=lam.dtype)
        k_expanded = k.unsqueeze(1)
        lam_expanded = lam.unsqueeze(0)
        return (
            k_expanded * torch.log(lam_expanded + 1e-8)
            - torch.lgamma(k_expanded + 1)
            - lam_expanded
        )


class CallableDuration(DurationDistribution):
    r"""User-provided callable: func(K, C, device) -> (K, C) log-probs."""

    def __init__(
        self,
        max_duration: int,
        num_classes: int,
        func: Callable[[int, int, torch.device], Tensor],
    ):
        super().__init__(max_duration, num_classes)
        self._func = func
        self._device_tracker = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self) -> Tensor:
        return self._func(self.max_duration, self.num_classes, self._device_tracker.device)


class UniformDuration(DurationDistribution):
    r"""Uniform: log P(k) = 0 for all k. No duration preference."""

    def __init__(self, max_duration: int, num_classes: int):
        super().__init__(max_duration, num_classes)
        self.register_buffer("_zeros", torch.zeros(max_duration, num_classes))

    def forward(self) -> Tensor:
        return self._zeros


def create_duration_distribution(
    distribution: Union[str, DurationDistribution, None],
    max_duration: int,
    num_classes: int,
    **kwargs,
) -> DurationDistribution:
    r"""Factory function for duration distributions.

    Args:
        distribution (str, DurationDistribution, None): Distribution type or instance.
            Options: ``"learned"``, ``"geometric"``, ``"negative_binomial"``/``"negbin"``,
            ``"poisson"``, ``"uniform"``, or a :class:`DurationDistribution` instance.
        max_duration (int): Maximum segment duration (K).
        num_classes (int): Number of label classes (C).
        **kwargs: Passed to distribution constructor.

    Returns:
        DurationDistribution: The requested distribution instance.
    """
    if distribution is None or distribution == "learned":
        return LearnedDuration(max_duration, num_classes, **kwargs)
    elif isinstance(distribution, DurationDistribution):
        return distribution
    elif distribution == "geometric":
        return GeometricDuration(max_duration, num_classes, **kwargs)
    elif distribution in ("negative_binomial", "negbin"):
        return NegativeBinomialDuration(max_duration, num_classes, **kwargs)
    elif distribution == "poisson":
        return PoissonDuration(max_duration, num_classes, **kwargs)
    elif distribution == "uniform":
        return UniformDuration(max_duration, num_classes)
    else:
        raise ValueError(
            f"Unknown duration distribution: {distribution}. "
            f"Options: learned, geometric, negative_binomial, poisson, uniform"
        )
