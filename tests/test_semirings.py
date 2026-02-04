"""
Tests for semiring implementations.

Verifies semiring operations produce correct results and maintain
invariants like associativity and identity elements.
"""

import pytest
import torch

from torch_semimarkov.semirings import (
    CrossEntropySemiring,
    EntropySemiring,
    KLDivergenceSemiring,
    KMaxSemiring,
    LogSemiring,
    MaxSemiring,
    StdSemiring,
)
from torch_semimarkov.semirings.checkpoint import CheckpointShardSemiring

# =============================================================================
# Semiring Algebraic Property Tests
# =============================================================================


class TestSemiringAxioms:
    """Test fundamental semiring axioms for all scalar semirings."""

    @pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring, StdSemiring])
    def test_sum_associativity(self, semiring):
        """(a(+)b)(+)c == a(+)(b(+)c) - associativity of addition."""
        torch.manual_seed(42)
        a = torch.randn(3, 4)
        b = torch.randn(3, 4)
        c = torch.randn(3, 4)

        # (a(+)b)(+)c
        ab = semiring.plus(a, b)
        ab_c = semiring.plus(ab, c)

        # a(+)(b(+)c)
        bc = semiring.plus(b, c)
        a_bc = semiring.plus(a, bc)

        assert torch.allclose(
            ab_c, a_bc, atol=1e-5
        ), f"{semiring.__name__}: associativity failed, max diff = {(ab_c - a_bc).abs().max()}"

    @pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring, StdSemiring])
    def test_sum_commutativity(self, semiring):
        """a(+)b == b(+)a - commutativity of addition."""
        torch.manual_seed(43)
        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        ab = semiring.plus(a, b)
        ba = semiring.plus(b, a)

        assert torch.allclose(ab, ba, atol=1e-6), f"{semiring.__name__}: commutativity failed"

    @pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring, StdSemiring])
    def test_additive_identity(self, semiring):
        """a(+)zero == a - zero is additive identity."""
        torch.manual_seed(44)
        a = torch.randn(3, 4)
        zero = torch.full_like(a, semiring.zero)

        result = semiring.plus(a, zero)

        assert torch.allclose(
            result, a, atol=1e-5
        ), f"{semiring.__name__}: additive identity failed"

    @pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring, StdSemiring])
    def test_multiplicative_identity(self, semiring):
        """a(*)one == a - one is multiplicative identity."""
        torch.manual_seed(45)
        a = torch.randn(3, 4)
        one = torch.full_like(a, semiring.one)

        result = semiring.mul(a, one)

        assert torch.allclose(
            result, a, atol=1e-6
        ), f"{semiring.__name__}: multiplicative identity failed"

    @pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring, StdSemiring])
    def test_distributivity(self, semiring):
        """a(*)(b(+)c) == (a(*)b)(+)(a(*)c) - multiplication distributes over addition."""
        torch.manual_seed(46)
        a = torch.randn(3, 4)
        b = torch.randn(3, 4)
        c = torch.randn(3, 4)

        # a(*)(b(+)c)
        bc_sum = semiring.plus(b, c)
        left = semiring.mul(a, bc_sum)

        # (a(*)b)(+)(a(*)c)
        ab = semiring.mul(a, b)
        ac = semiring.mul(a, c)
        right = semiring.plus(ab, ac)

        assert torch.allclose(
            left, right, atol=1e-5
        ), f"{semiring.__name__}: distributivity failed, max diff = {(left - right).abs().max()}"


class TestMaxSemiringProperties:
    """Test properties specific to MaxSemiring."""

    def test_idempotence(self):
        """max(a, a) == a - max is idempotent."""
        torch.manual_seed(50)
        a = torch.randn(3, 4)

        result = MaxSemiring.plus(a, a)

        assert torch.allclose(result, a, atol=1e-6), "MaxSemiring: idempotence failed"

    def test_finds_actual_maximum(self):
        """MaxSemiring.sum returns the actual maximum value."""
        x = torch.tensor([1.0, 5.0, 3.0, 2.0])
        result = MaxSemiring.sum(x, dim=-1)
        assert result.item() == 5.0, f"Expected 5.0, got {result.item()}"

    def test_max_vs_logsumexp_ordering(self):
        """max(x) <= logsumexp(x) always (max is tighter bound)."""
        torch.manual_seed(51)
        x = torch.randn(10, 20)

        max_result = MaxSemiring.sum(x, dim=-1)
        log_result = LogSemiring.sum(x, dim=-1)

        assert (max_result <= log_result + 1e-6).all(), "Max should always be <= logsumexp"

    def test_gradient_is_sparse_indicator(self):
        """MaxSemiring gradients should be 0/1 sparse indicators (argmax)."""
        torch.manual_seed(52)
        x = torch.randn(5, 10, requires_grad=True)

        result = MaxSemiring.sum(x, dim=-1)
        result.sum().backward()

        # Gradients should be 0 or 1
        grad = x.grad
        is_zero_or_one = (grad.abs() < 1e-6) | ((grad - 1.0).abs() < 1e-6)
        assert is_zero_or_one.all(), f"Expected 0/1 gradients, got values: {grad[~is_zero_or_one]}"

        # Each row should have exactly one 1 (the argmax)
        assert torch.allclose(
            grad.sum(dim=-1), torch.ones(5), atol=1e-6
        ), "Each row should have exactly one gradient of 1"


class TestEntropySemiringProperties:
    """Test properties specific to EntropySemiring."""

    def test_entropy_non_negative(self):
        """Shannon entropy H(P) >= 0 always."""
        torch.manual_seed(60)
        # Create log-probabilities (negative values)
        logp = torch.randn(5, 10) - 2.0  # Shift to make more negative

        converted = EntropySemiring.convert(logp)
        summed = EntropySemiring.sum(converted, dim=-1)
        entropy = EntropySemiring.unconvert(summed)

        assert (
            entropy >= -1e-5
        ).all(), f"Entropy should be non-negative, got min = {entropy.min().item()}"

    def test_entropy_zero_for_delta_distribution(self):
        """H(delta) ≈ 0 for deterministic (delta) distribution."""
        # Delta distribution: all mass on one outcome
        # In log space: one value is 0 (log(1)), rest are -inf
        n = 10
        logp = torch.full((n,), -1e5)  # Approximating -inf
        logp[3] = 0.0  # All mass on index 3

        converted = EntropySemiring.convert(logp)
        summed = EntropySemiring.sum(converted, dim=-1)
        entropy = EntropySemiring.unconvert(summed)

        assert (
            entropy.abs().item() < 0.01
        ), f"Delta distribution entropy should be ~0, got {entropy.item()}"

    def test_entropy_maximum_for_uniform(self):
        """H(uniform) ≈ log(N) for uniform distribution over N outcomes."""
        n = 8
        # Uniform distribution: each outcome has probability 1/N
        # In log space: each value is log(1/N) = -log(N)
        logp = torch.full((n,), -torch.log(torch.tensor(float(n))))

        converted = EntropySemiring.convert(logp)
        summed = EntropySemiring.sum(converted, dim=-1)
        entropy = EntropySemiring.unconvert(summed)

        expected = torch.log(torch.tensor(float(n)))
        assert torch.isclose(
            entropy, expected, atol=0.01
        ), f"Uniform entropy should be log({n})={expected.item():.3f}, got {entropy.item():.3f}"


class TestKLDivergenceProperties:
    """Test properties specific to KLDivergenceSemiring."""

    def test_kl_non_negative(self):
        """KL(P||Q) >= 0 (Gibbs' inequality)."""
        torch.manual_seed(70)
        # Create two different distributions
        logp = torch.log_softmax(torch.randn(5, 10), dim=-1)
        logq = torch.log_softmax(torch.randn(5, 10), dim=-1)

        converted = KLDivergenceSemiring.convert([logp, logq])
        summed = KLDivergenceSemiring.sum(converted, dim=-1)
        kl = KLDivergenceSemiring.unconvert(summed)

        assert (
            kl >= -1e-4
        ).all(), f"KL divergence should be non-negative, got min = {kl.min().item()}"

    def test_kl_asymmetry(self):
        """KL(P||Q) != KL(Q||P) in general (KL is asymmetric)."""
        torch.manual_seed(71)
        logp = torch.log_softmax(torch.randn(10), dim=-1)
        logq = torch.log_softmax(torch.randn(10), dim=-1)

        # KL(P||Q)
        converted_pq = KLDivergenceSemiring.convert([logp, logq])
        summed_pq = KLDivergenceSemiring.sum(converted_pq, dim=-1)
        kl_pq = KLDivergenceSemiring.unconvert(summed_pq)

        # KL(Q||P)
        converted_qp = KLDivergenceSemiring.convert([logq, logp])
        summed_qp = KLDivergenceSemiring.sum(converted_qp, dim=-1)
        kl_qp = KLDivergenceSemiring.unconvert(summed_qp)

        assert not torch.isclose(
            kl_pq, kl_qp, atol=0.01
        ), f"KL should be asymmetric: KL(P||Q)={kl_pq.item():.4f}, KL(Q||P)={kl_qp.item():.4f}"


class TestCrossEntropyProperties:
    """Test properties specific to CrossEntropySemiring."""

    def test_cross_entropy_equals_entropy_when_equal(self):
        """H(P, P) = H(P) when P=Q."""
        torch.manual_seed(80)
        logp = torch.log_softmax(torch.randn(10), dim=-1)

        # Cross entropy H(P, P)
        converted_ce = CrossEntropySemiring.convert([logp, logp])
        summed_ce = CrossEntropySemiring.sum(converted_ce, dim=-1)
        ce = CrossEntropySemiring.unconvert(summed_ce)

        # Entropy H(P)
        converted_e = EntropySemiring.convert(logp)
        summed_e = EntropySemiring.sum(converted_e, dim=-1)
        entropy = EntropySemiring.unconvert(summed_e)

        assert torch.isclose(
            ce, entropy, atol=0.01
        ), f"H(P,P) should equal H(P): H(P,P)={ce.item():.4f}, H(P)={entropy.item():.4f}"


class TestKMaxSemiringProperties:
    """Test properties specific to KMaxSemiring."""

    def test_maintains_descending_order(self):
        """KMax results should always be in descending order."""
        torch.manual_seed(90)
        k = 5
        KMax = KMaxSemiring(k)

        x = torch.randn(20)
        converted = KMax.convert(x)
        result = KMax.sum(converted, dim=-1)

        # Check descending order
        for i in range(k - 1):
            assert (
                result[i] >= result[i + 1]
            ), f"KMax results not descending at {i}: {result[i].item()} < {result[i+1].item()}"

    def test_k1_matches_max_semiring(self):
        """KMaxSemiring(1) should match MaxSemiring."""
        torch.manual_seed(91)
        KMax1 = KMaxSemiring(1)

        x = torch.randn(3, 10)

        # KMax(1) result
        converted = KMax1.convert(x)
        kmax_result = KMax1.sum(converted, dim=-1)
        kmax_top1 = KMax1.unconvert(kmax_result)

        # MaxSemiring result
        max_result = MaxSemiring.sum(x, dim=-1)

        assert torch.allclose(
            kmax_top1, max_result, atol=1e-6
        ), "KMaxSemiring(1) should match MaxSemiring"

    def test_output_size_is_k(self):
        """KMax output should always have exactly k values."""
        k = 4
        KMax = KMaxSemiring(k)

        x = torch.randn(15)
        converted = KMax.convert(x)
        result = KMax.sum(converted, dim=-1)

        assert result.shape[0] == k, f"Expected {k} values, got {result.shape[0]}"

    def test_top_k_values_correct(self):
        """KMax should return the actual top-k values."""
        k = 3
        KMax = KMaxSemiring(k)

        x = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        converted = KMax.convert(x)
        result = KMax.sum(converted, dim=-1)

        expected = torch.tensor([5.0, 4.0, 3.0])
        assert torch.allclose(
            result, expected, atol=1e-6
        ), f"Expected top-3 = [5, 4, 3], got {result.tolist()}"


# =============================================================================
# Original Tests (preserved)
# =============================================================================


def test_kmax_semiring_roundtrip():
    torch.manual_seed(0)
    k = 3
    semiring = KMaxSemiring(k)
    x = torch.randn(2, 4)

    converted = semiring.convert(x)
    assert converted.shape == (k, 2, 4)

    roundtrip = semiring.unconvert(converted)
    assert torch.allclose(roundtrip, x)


def test_entropy_semiring_sum_shape():
    torch.manual_seed(1)
    x = torch.randn(3, 5)

    converted = EntropySemiring.convert(x)
    summed = EntropySemiring.sum(converted, dim=-1)
    entropy = EntropySemiring.unconvert(summed)

    assert summed.shape == (2, 3)
    assert entropy.shape == (3,)
    assert torch.isfinite(entropy).all()


def test_kl_divergence_semiring_zero_for_equal():
    torch.manual_seed(2)
    logp = torch.randn(2, 6)
    converted = KLDivergenceSemiring.convert([logp, logp])
    summed = KLDivergenceSemiring.sum(converted, dim=-1)
    kl = KLDivergenceSemiring.unconvert(summed)

    assert torch.isfinite(kl).all()
    assert kl.abs().max().item() < 1e-3


def test_cross_entropy_semiring_non_negative():
    torch.manual_seed(3)
    logp = torch.randn(2, 6)
    converted = CrossEntropySemiring.convert([logp, logp])
    summed = CrossEntropySemiring.sum(converted, dim=-1)
    ce = CrossEntropySemiring.unconvert(summed)

    assert torch.isfinite(ce).all()
    assert ce.min().item() > -1e-4


def test_logsemiring_matmul():
    """Test LogSemiring matrix multiplication."""
    torch.manual_seed(4)
    batch, n = 2, 4
    a = torch.randn(batch, n, n)
    b = torch.randn(batch, n, n)

    result = LogSemiring.matmul(a, b)

    assert result.shape == (batch, n, n)
    assert torch.isfinite(result).all()


def test_stdsemiring_matmul():
    """Test StdSemiring matrix multiplication."""
    torch.manual_seed(5)
    batch, n = 2, 4
    a = torch.randn(batch, n, n)
    b = torch.randn(batch, n, n)

    result = StdSemiring.matmul(a, b)

    assert result.shape == (batch, n, n)
    assert torch.isfinite(result).all()


def test_checkpoint_shard_semiring_matches_logsemiring():
    torch.manual_seed(6)
    batch, n = 2, 3
    a = torch.randn(batch, n, n)
    b = torch.randn(batch, n, n)

    ShardedLog = CheckpointShardSemiring(LogSemiring, max_size=128)
    expected = LogSemiring.matmul(a, b)
    actual = ShardedLog.matmul(a, b)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_logsemiring_sum():
    """Test LogSemiring sum (logsumexp)."""
    torch.manual_seed(7)
    x = torch.randn(3, 4, 5)

    result = LogSemiring.sum(x, dim=-1)
    expected = torch.logsumexp(x, dim=-1)

    assert torch.allclose(result, expected)


def test_logsemiring_identity():
    """Test LogSemiring identity elements."""
    # zero: additive identity (should be -inf or very negative)
    # one: multiplicative identity (should be 0 in log space)
    assert LogSemiring.zero < -1e4  # -100000.0
    assert LogSemiring.one == 0.0
