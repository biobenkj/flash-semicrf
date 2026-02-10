import torch

from flash_semicrf.semirings import LogSemiring
from flash_semicrf.semirings.checkpoint import CheckpointSemiring, CheckpointShardSemiring


def test_checkpoint_semiring_matmul_matches_logsemiring():
    torch.manual_seed(0)
    a = torch.randn(1, 2, 2, requires_grad=True)
    b = torch.randn(1, 2, 2, requires_grad=True)

    CheckLog = CheckpointSemiring(LogSemiring, min_size=0)
    out = CheckLog.matmul(a, b)
    ref = LogSemiring.matmul(a, b)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    out.sum().backward()
    assert a.grad is not None
    assert b.grad is not None


def test_checkpoint_semiring_no_checkpoint_branch():
    torch.manual_seed(1)
    a = torch.randn(1, 2, 2)
    b = torch.randn(1, 2, 2)

    CheckLog = CheckpointSemiring(LogSemiring, min_size=10_000)
    out = CheckLog.matmul(a, b)
    ref = LogSemiring.matmul(a, b)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_checkpoint_shard_semiring_matmul_matches_logsemiring():
    torch.manual_seed(2)
    a = torch.randn(1, 2, 2, requires_grad=True)
    b = torch.randn(1, 2, 2, requires_grad=True)

    ShardedLog = CheckpointShardSemiring(LogSemiring, max_size=1, min_size=0)
    out = ShardedLog.matmul(a, b)
    ref = LogSemiring.matmul(a, b)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    out.sum().backward()
    assert a.grad is not None
    assert b.grad is not None


def test_checkpoint_shard_semiring_no_checkpoint_branch():
    torch.manual_seed(3)
    a = torch.randn(1, 2, 2)
    b = torch.randn(1, 2, 2)

    ShardedLog = CheckpointShardSemiring(LogSemiring, max_size=10_000, min_size=10_000)
    out = ShardedLog.matmul(a, b)
    ref = LogSemiring.matmul(a, b)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
