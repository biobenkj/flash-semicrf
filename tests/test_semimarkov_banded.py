import torch

from torch_semimarkov import SemiMarkov
from torch_semimarkov.semirings import LogSemiring


def test_banded_logpartition_matches_linear_scan():
    torch.manual_seed(0)
    batch, T, K, C = 1, 6, 4, 2
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)
    struct = SemiMarkov(LogSemiring)

    v_banded, _, _ = struct.logpartition(
        edge,
        lengths=lengths,
        use_banded=True,
        banded_perm="none",
        banded_bw_ratio=1.1,
    )
    v_linear, _, _ = struct.logpartition(edge, lengths=lengths, use_linear_scan=True)

    assert torch.allclose(v_banded, v_linear, atol=1e-5, rtol=1e-5)


def test_banded_helpers_shapes():
    struct = SemiMarkov(LogSemiring)
    device = torch.device("cpu")
    lu, ld = struct._compute_bandwidth(span_length=4, K=4, C=2)
    assert lu == ld

    adj = struct._build_adjacency(span_length=3, K=3, C=2, device=device)
    assert adj.shape == (4, 4)
    assert adj.dtype == torch.bool

    use_banded, perm, best_bw, threshold = struct._choose_banded_permutation(
        span_length=4, K=4, C=2, perm_mode="rcm", bw_ratio=1.0, device=device
    )
    assert isinstance(use_banded, bool)
    assert isinstance(best_bw, int)
    assert isinstance(threshold, float)
    if perm is not None:
        size = (4 - 1) * 2
        assert perm.numel() == size
        assert torch.equal(torch.sort(perm).values, torch.arange(size))
