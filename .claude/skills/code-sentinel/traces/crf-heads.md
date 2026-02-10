# Sentinel: CRF Heads (nn.py)

**Verified against:** `src/flash_semicrf/nn.py` @ commit `8b8e3ed` (+ staged changes)

**Linked tests:** `tests/test_semimarkov.py`, `tests/test_streaming_triton.py::TestTritonBasic`

## Summary

The `SemiMarkovCRFHead` class provides the user-facing API for semi-Markov CRF
sequence labeling. It wraps streaming/exact backends with:

1. **Automatic backend selection** based on memory heuristics
2. **Numerical stability** via zero-centering and float64 conversion
3. **NaN validation** at projection and cumsum stages
4. **Unified interface** for training (`compute_loss`) and inference (`decode`)
5. **Dual edge construction** — in-place (`_build_edge_tensor`) for streaming/scan, differentiable (`_build_differentiable_edge`) for `_dp_standard`

## Active Assumptions

### Mechanically Verified

These are verified automatically via `python3 verify-assumptions.py crf-heads`.

| ID | Assumption | Verification |
|----|------------|--------------|
| N1 | Zero-centering applied before cumsum | anchor: ZERO_CENTER |
| N2 | Float64 conversion for numerical stability | anchor: FLOAT64_CONVERT |
| N3 | NaN check after projection exists | anchor: NAN_CHECK_PROJECTION |
| N4 | NaN check after cumsum exists | anchor: NAN_CHECK_CUMSUM |
| N5 | Streaming forward called for partition | anchor: STREAMING_FORWARD_CALL |

### Agent-Verified (on trace load)

These require human/agent judgment when loading the trace.

| ID | Assumption | Verification Guidance |
|----|------------|----------------------|
| N6 | Backend selection matches dispatch-overview.md | Compare `_select_backend` logic with dispatch-overview decision tree |
| N7 | T=1 skip for zero-centering documented | Check `if T > 1:` guard before zero-centering (~line 407) |
| N8 | Duration bias indexing uses k-1 | Verify `dur_idx = k - 1` in `_build_edge_tensor` (~line 215) |
| N9 | `_build_differentiable_edge` uses finite sentinel `-1e18` not `-inf` | Verify `_NEGINF_SAFE = -1e18` at ~line 259 |
| N10 | `_forward_exact` routes `algorithm="standard"` to `_build_differentiable_edge` | Check `if algorithm == "standard":` at ~line 298 |

## Edge Construction Methods

Two edge construction methods exist, chosen based on the DP algorithm:

### `_build_edge_tensor` (lines 189-224)

- **Uses:** In-place assignment (`edge[:, n, dur_idx] = ...`)
- **Invalid positions:** `float("-inf")` (true negative infinity)
- **Autograd:** Breaks graph for in-place writes — fine for `_dp_scan_streaming` which uses its own autograd
- **Used by:** `_forward_exact(algorithm="scan")`, `_forward_binary_tree_sharded`

### `_build_differentiable_edge` (lines 226-277)

- **Uses:** `torch.stack` / `torch.cat` (preserves autograd graph)
- **Invalid positions:** `-1e18` (finite sentinel)
- **Autograd:** Full gradient flow from edge back to `self.transition` and `self.duration_bias`
- **Used by:** `_forward_exact(algorithm="standard")` via `_dp_standard`

**Why the finite sentinel?** `_dp_standard` computes `logsumexp` over ALL K durations at each position, including invalid ones. With true `-inf`, backward encounters `softmax([-inf,...,-inf])` = `0/0` = NaN. With `-1e18`, softmax gives uniform `1/C` but the upstream gradient is ~0 (since `exp(-1e18) = 0` in the outer logsumexp), so the gradient is clean.

**Why two methods?** `_dp_scan_streaming` only iterates over valid durations and has its own custom autograd, so in-place + true `-inf` is fine. `_dp_standard` (pytorch-struct reference) needs standard autograd through the edge tensor and processes all K slots including padding.

## Algorithm Flow: forward()

1. **Input validation** (lines 363-366)
   - `validate_hidden_states()`, `validate_lengths()`, `validate_device_consistency()`

2. **Projection** (lines 370-374)
   ```python
   if self.projection is not None:
       scores = self.projection(hidden_states)  # (batch, T, C)
   ```

3. **NaN check after projection** (lines 376-383)
   - Detects gradient corruption from corrupted model parameters

4. **Backend selection** (lines 385-400)
   - `auto` -> streaming vs exact based on edge tensor size
   - `streaming` -> force streaming backend
   - `exact` -> force exact backend via `SemiMarkov.logpartition` (`_dp_scan_streaming`)
   - `dp_standard` -> exact backend via `SemiMarkov._dp_standard` (pytorch-struct reference)
   - `binary_tree_sharded` -> memory-efficient reference implementation

5. **Cumulative scores** (lines 402-412)
   ```python
   scores_float = scores.double()  # Float64 for stability (matches Triton kernel precision)
   if T > 1:
       scores_float = scores_float - scores_float.mean(dim=1, keepdim=True)  # Zero-center
   cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)
   ```

6. **NaN check after cumsum** (lines 414-419)
   - Detects extreme input values causing overflow

7. **Partition computation** (lines 421-442)
   - streaming -> `semi_crf_streaming_forward()`
   - binary_tree_sharded -> `_forward_binary_tree_sharded()`
   - dp_standard -> `_forward_exact(scores, lengths, "log", algorithm="standard")` (uses `_build_differentiable_edge`)
   - exact -> `_forward_exact(scores, lengths, "log")` (uses `_build_edge_tensor`)

## `_forward_exact()` Flow (lines 279-307)

Routes edge construction and DP algorithm based on `algorithm` parameter:

| `algorithm` | Edge Construction | DP Method | Autograd |
|-------------|------------------|-----------|----------|
| `"standard"` | `_build_differentiable_edge` | `_dp_standard` with `force_grad=False` | Through edge tensor |
| `"scan"` (default) | `_build_edge_tensor` | `logpartition` → `_dp_scan_streaming` | Custom streaming autograd |

## Semiring Restriction

The `SemiMarkovCRFHead` public API does **not** expose semiring selection. It is hardcoded per method:

| Method | Semiring | Purpose |
|--------|----------|---------|
| `forward()` | "log" | Partition function for training loss |
| `compute_loss()` | "log" | NLL = log(Z) - score(y*) |
| `decode()` | "max" | Viterbi best-path score |
| `decode_with_traceback()` | "max" | Viterbi score + segment reconstruction |

For other semirings (Entropy, KL Divergence, Cross-Entropy, StdSemiring, KMaxSemiring), use the `SemiMarkov` class directly with pre-computed edge tensors. See non-streaming-backends.md for the full semiring list.

## Backend Selection (_select_backend)

**Location:** lines 170-187

| Condition | Backend | Triton |
|-----------|---------|--------|
| Edge tensor > 8GB | streaming | use_triton param |
| Edge tensor <= 8GB | exact | False |
| Semiring not log/max | exact | False (error if OOM) |

**Edge tensor size formula:** `T * K * C * C * 8` bytes (float64)

```python
def _should_use_streaming(self, T: int) -> bool:
    edge_tensor_bytes = T * K * C * C * 8
    return edge_tensor_bytes > self.edge_memory_threshold  # default 8GB
```

## compute_loss() Flow

**Location:** lines 445-499

1. Validate labels via `validate_labels()`
2. Call `forward()` to get partition and cum_scores
3. Score gold segmentation via `_score_gold()` -> `score_gold_vectorized()`
4. Return `partition - gold_score` with optional reduction

## decode() vs decode_with_traceback()

| Method | Line | Returns | Memory |
|--------|------|---------|--------|
| `decode()` | 530 | Viterbi score only | O(KC) |
| `decode_with_traceback()` | 608 | Score + segment list | O(TC) for backpointers |

**decode_with_traceback** uses:
- `semi_crf_streaming_viterbi_with_backpointers()` for backpointer computation
- `_traceback_from_backpointers()` for O(T) segment reconstruction

## Critical Invariants

- [ ] Zero-centering MUST be skipped for T=1 (single position has no variance)
- [ ] Float64 used for cumulative scores (matches Triton kernel internal precision)
- [ ] NaN checks MUST come after projection and cumsum (early detection)
- [ ] Duration bias shape: (K, C) where index 0 = duration 1
- [ ] Transition matrix convention: `transition[i, j]` = FROM label i TO label j
- [ ] `_build_differentiable_edge` MUST use finite sentinel (`-1e18`), NOT `-inf`, to avoid NaN gradient in `_dp_standard` backward pass
- [ ] `_forward_exact(algorithm="standard")` MUST use `_build_differentiable_edge` (not `_build_edge_tensor`) for autograd correctness

## Numerical Guards

| Location | Guard | Purpose |
|----------|-------|---------|
| line 378 | `torch.isnan(scores).any()` | Detect corrupted projection weights |
| line 415 | `torch.isnan(cum_scores).any()` | Detect extreme input values |
| line 259 | `_NEGINF_SAFE = -1e18` | Prevent NaN in `logsumexp` backward for invalid durations |

## Entry Points

| Method | Line | Purpose | Returns |
|--------|------|---------|---------|
| `forward()` | 336 | Partition function | dict with `partition`, `cum_scores` |
| `compute_loss()` | 445 | NLL training loss | Tensor |
| `decode()` | 530 | Viterbi score | Tensor |
| `decode_with_traceback()` | 608 | Viterbi + segments | `ViterbiResult` |
| `_select_backend()` | 170 | Backend selection | `(backend_type, use_triton)` |
| `_build_edge_tensor()` | 189 | In-place edge construction | Tensor (batch, T, K, C, C) |
| `_build_differentiable_edge()` | 226 | Autograd-safe edge construction | Tensor (batch, T, K, C, C) |
| `_forward_exact()` | 279 | Exact DP dispatch | Tensor |
| `_score_gold()` | 514 | Gold segmentation score | Tensor |

## Known Issues

| Issue | Severity | Resolution |
|-------|----------|------------|
| T=1 mean zeros content | Medium | Skip zero-centering with `if T > 1:` guard |
| Triton backpointer disabled | Low | Uses PyTorch reference due to memory corruption issue |
| `_build_differentiable_edge` is O(T*K) Python loops | Medium | Acceptable for validation; streaming backend avoids this for production |

## Version History

- **2026-02-10**: Major update — documented new `_build_differentiable_edge()` method (torch.stack, finite sentinel -1e18), split edge construction in `_forward_exact()`, new `dp_standard` backend option, updated all line numbers, added invariants N9/N10 and critical invariant for finite sentinel
- **2026-02-09**: Updated float32→float64 throughout: cum_scores construction, edge tensor sizing (4→8 bytes), invariants. Renamed anchor FLOAT32_CONVERT→FLOAT64_CONVERT. Updated assumption N2.
- **2026-02-09**: Added Semiring Restriction section documenting that nn.py only exposes log/max semirings
- **2026-01-28**: Initial trace @ commit `6d6c535`
