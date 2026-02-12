# Sentinel: Autograd-Kernel Interface

**Verified against:** `src/flash_semicrf/streaming/autograd.py` @ commit `e45c7f1`
**Linked tests:** `tests/test_streaming_triton.py::TestTritonGradients`

## Summary

Documents the contract between autograd functions (`SemiCRFStreamingTriton`, `SemiCRFStreaming`) and the underlying kernels. This is where the nastiest bugs often live - at the boundary between PyTorch's autograd and custom Triton/PyTorch implementations.

## Shape Legend

- `B` = Batch size
- `T` = Sequence length (time steps)
- `K` = Maximum segment length
- `C` = Number of classes/labels
- `C_PAD` = Padded class count (power of 2 for Triton)

## ctx.save_for_backward() Semantics

### SemiCRFStreamingTriton (lines 205-219)

```python
ctx.save_for_backward(
    cum_scores,           # (B, T+1, C) - with gradients
    transition,           # (C, C) or (K, C, C) - with gradients
    duration_bias,        # (K, C) - with gradients
    lengths,              # (B,) - no gradients
    ring_checkpoints,     # (B, num_ckpts, K, C_PAD) - from forward kernel
    log_norm_checkpoints, # (B, num_ckpts) - cumulative normalization factors
    partition,            # (B,) - forward output, validated before backward
    proj_start,           # (B, T, C) or None - optional, with gradients
    proj_end,             # (B, T, C) or None - optional, with gradients
)
ctx.K = K                           # int constant
ctx.semiring = semiring             # str constant
ctx.checkpoint_interval = interval  # int constant
ctx.num_warps = num_warps           # int constant (passed to backward kernel)
```

### SemiCRFStreaming (PyTorch, lines 65-78)

Same tensors saved, minus `num_warps`.

## What's Saved vs Recomputed

| Tensor | Saved? | Why |
|--------|--------|-----|
| `cum_scores` | Yes | Needed to recompute edges on-the-fly |
| `transition` | Yes | Needed for edge computation |
| `duration_bias` | Yes | Needed for edge computation |
| `lengths` | Yes | Needed for masking |
| `ring_checkpoints` | Yes | Checkpointed alpha states (normalized) |
| `log_norm_checkpoints` | Yes | Cumulative normalization factors for T=100k+ stability |
| `partition` | Yes | Validated before backward, used for normalization |
| `proj_start/end` | Yes | Needed if boundaries used |
| **Alpha values** | Partial | Only checkpoints saved; recompute forward between checkpoints |
| **Edge potentials** | No | Recomputed on-the-fly in backward |
| **LogSumExp intermediates** | No | Recomputed during segment-wise backward |

## Gradient Tensor Conventions

### Input Requirements

All input tensors must be:
1. **Contiguous** - Kernel calls `.contiguous()` on all inputs
2. **Same device** - All on CPU or all on CUDA
3. **Same dtype** - Float32 or float64 for inputs (float64 recommended; float16 causes overflow)

### Dtype Handling (Triton Only)

**IMPORTANT**: The Triton autograd function performs dtype conversion (autograd.py:209-221):

```python
# Forward computes partition in float64
partition, ring_checkpoints, actual_checkpoint_interval, log_norm_checkpoints = (
    launch_streaming_triton_kernel(...)  # Returns float64
)

# Cast partition back to input dtype for return, but keep float64 for backward
partition_f64 = partition  # Keep float64 for backward pass
partition_return = partition.to(cum_scores.dtype)  # Return in input dtype to user

# Save float64 version for backward (line 214)
ctx.save_for_backward(..., partition_f64, ...)  # <-- float64

return partition_return  # <-- in input dtype for user
```

**Why**: Kernels compute in float64 for numerical stability at extreme T. The user sees output matching their input dtype, but backward uses float64 internally for precision. With float64 cum_scores (now recommended), there is no dtype conversion on the return path.

### Output Guarantees

Backward kernels produce:

| Gradient | Shape | Notes |
|----------|-------|-------|
| `grad_cum_scores` | (B, T+1, C) | Per-batch; scaled by grad_output |
| `grad_transition` | (B, C, C) or (B, K, C, C) | Per-batch; reduced via einsum |
| `grad_duration_bias` | (B, K, C) | Per-batch; reduced via einsum |
| `grad_proj_start` | (B, T, C) or None | Per-batch if boundaries used |
| `grad_proj_end` | (B, T, C) or None | Per-batch if boundaries used |

### Shared Parameter Gradient Reduction

Both PyTorch and Triton backward return **per-batch** gradients for shared parameters. Autograd function reduces them:

```python
# autograd.py lines 149-154 (PyTorch path)
# Shared parameters: weighted sum via einsum (memory-efficient)
if grad_transition.ndim == 3:  # (batch, C, C) - static transitions
    grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
else:  # (batch, K, C, C) - duration-dependent
    grad_transition = torch.einsum("bkij, b -> kij", grad_transition, grad_output)

grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)
```

Triton backward (lines 245-260) scales internally via `grad_output` parameter.

## Error Handling at Boundary

### Partition Validation (Before Backward)

Both autograd functions validate partition before backward:

```python
# autograd.py lines 98-105 (PyTorch) and 239-246 (Triton)
if not torch.isfinite(partition).all():
    nan_count = torch.isnan(partition).sum().item()
    inf_count = torch.isinf(partition).sum().item()
    raise RuntimeError(
        f"Non-finite partition from forward pass: "
        f"{nan_count} NaN, {inf_count} Inf. "
        f"Check forward pass numerical stability."
    )
```

**Why**: If forward produced NaN/Inf, backward will produce garbage. Fail early.

### Gradient Validation (After Backward)

```python
# autograd.py lines 126-132 (PyTorch) and 270-281 (Triton)
if not torch.isfinite(grad_cum_scores).all():
    nan_count = torch.isnan(grad_cum_scores).sum().item()
    inf_count = torch.isinf(grad_cum_scores).sum().item()
    raise RuntimeError(
        f"Non-finite values in CRF backward: "
        f"grad_cum_scores has {nan_count} NaN, {inf_count} Inf"
    )
```

**Why**: Catch gradient corruption before it propagates to optimizer.

### Max Semiring Guard (Forward Only)

All four autograd classes now reject `semiring='max'` in their `.forward()` methods:

```python
# In SemiCRFStreaming.forward(), SemiCRFStreamingTriton.forward(),
# LinearCRFStreaming.forward(), SemiCRFK2Streaming.forward():
if semiring == "max":
    raise ValueError(
        "semiring='max' does not support autograd backward. "
        "Use semi_crf_streaming_forward() with torch.no_grad() instead."
    )
```

Additionally, the dispatch function (`semi_crf_streaming_forward`) has a top-level guard:

```python
# autograd.py:601-616
needs_grad = torch.is_grad_enabled() and (
    cum_scores.requires_grad or transition.requires_grad or ...
)

if semiring == "max" and needs_grad:
    raise ValueError("semiring='max' (Viterbi) does not support gradients. ...")
```

**Why `torch.is_grad_enabled()` gating**: Without this, `model.decode()` during training (where params have `requires_grad=True`) would take the autograd path even inside `torch.no_grad()`. The gate ensures `torch.no_grad()` contexts correctly bypass autograd and reach the direct inference kernels.

## Kernel Launch Interface

### Triton Forward: `launch_streaming_triton_kernel()`

```python
partition, ring_checkpoints, checkpoint_interval, log_norm_checkpoints = (
    launch_streaming_triton_kernel(
        cum_scores.detach(),      # Detached - kernel doesn't need gradients
        transition.detach(),
        duration_bias.detach(),
        lengths,
        K,
        semiring,
        proj_start=proj_start.detach() if proj_start is not None else None,
        proj_end=proj_end.detach() if proj_end is not None else None,
        num_warps=num_warps,
    )
)
```

Returns:
- `partition`: (B,) log partition values
- `ring_checkpoints`: (B, num_ckpts, K, C_PAD) for backward (normalized)
- `checkpoint_interval`: int actual interval used
- `log_norm_checkpoints`: (B, num_ckpts) cumulative normalization factors

### Triton Backward: `launch_streaming_triton_backward()`

```python
grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end, _ = (
    launch_streaming_triton_backward(
        cum_scores,           # Not detached - need original tensor
        transition,
        duration_bias,
        lengths,
        partition,            # Forward output
        ring_checkpoints,     # Checkpoints from forward (normalized)
        log_norm_checkpoints, # Cumulative normalization factors
        checkpoint_interval,
        grad_output,          # Upstream gradient
        proj_start=proj_start,
        proj_end=proj_end,
        num_warps=num_warps,
    )
)
```

Note: Returns 6 values; 6th (`boundary_marginals`) is unused in autograd path.

## Critical Invariants

- [ ] All inputs detached in forward kernel call (lines 184-195)
- [ ] Partition validated before backward (lines 92, 233)
- [ ] All backward outputs validated (lines 120, 264)
- [ ] Shared params reduced via einsum, not expanded then summed (memory)
- [ ] `checkpoint_interval >= K` (required for correct recomputation)
- [ ] `num_warps` passed from forward to backward (consistency)
- [ ] Max semiring rejected in all autograd forward methods (lines 45, 193, 342, 430)
- [ ] `needs_grad` gates on `torch.is_grad_enabled()` (line 601)

## Known Issues

| Issue | Severity | Symptom | Resolution |
|-------|----------|---------|------------|
| Missing `.detach()` in forward | Critical | Double backward errors | Always detach inputs to kernel |
| Wrong checkpoint_interval | Critical | Incorrect gradients | Use same interval in forward/backward |
| float16 overflow | High | NaN in long sequences | Use float64 (recommended) or float32 for cum_scores |
| Non-contiguous input | Medium | Kernel crash or wrong results | Always call `.contiguous()` |
| Max semiring in autograd | Critical | semiring="max" with grad | Raise ValueError; use torch.no_grad() for decode (`e45c7f1`) |

## Debugging: Interface Violations

When gradients are wrong but partition is correct, check:

```python
# Insert in SemiCRFStreamingTriton.backward(), line 242
print(f"checkpoint_interval: forward={ctx.checkpoint_interval}")
print(f"ring_checkpoints shape: {ring_checkpoints.shape}")
print(f"expected checkpoints: {(T + ctx.checkpoint_interval - 1) // ctx.checkpoint_interval}")

# After backward kernel returns
print(f"grad_cum_scores finite: {torch.isfinite(grad_cum_scores).all()}")
print(f"grad_transition finite: {torch.isfinite(grad_transition).all()}")
print(f"grad_duration_bias finite: {torch.isfinite(grad_duration_bias).all()}")
```

## Version History

- **2026-02-12**: Documented max semiring guard in all four autograd classes; added `torch.is_grad_enabled()` gating for `needs_grad`; updated line numbers throughout; updated to `e45c7f1`
- **2026-02-09**: Updated dtype docs: float32→float64 recommended for cum_scores; docstring now says "float32 or float64"; return comment updated to "input dtype" instead of "float32"
- **2026-02-05**: Added dtype handling documentation for Triton path (partition computed in float64, returned as input dtype); added checkpoint_interval parameter to forward methods; updated to commit `6c463c3`
- **2026-02-02**: Updated line numbers throughout; clarified per-batch gradient convention
- **2026-02-01**: Added log_norm_checkpoints to save_for_backward and kernel interfaces for T=100k+ numerical stability
- **2026-01-27**: Initial trace @ commit `40fe66b`
