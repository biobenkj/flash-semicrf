# Sentinel: Autograd-Kernel Interface

**Verified against:** `src/torch_semimarkov/streaming/autograd.py` @ commit `871c352`
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

### SemiCRFStreamingTriton (lines 199-213)

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

### SemiCRFStreaming (PyTorch, lines 59-72)

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
3. **Same dtype** - Typically float32 (float16 causes overflow)

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
# autograd.py lines 143-148 (PyTorch path)
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
# autograd.py lines 92-99 (PyTorch) and 233-240 (Triton)
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
# autograd.py lines 120-126 (PyTorch) and 264-275 (Triton)
if not torch.isfinite(grad_cum_scores).all():
    nan_count = torch.isnan(grad_cum_scores).sum().item()
    inf_count = torch.isinf(grad_cum_scores).sum().item()
    raise RuntimeError(
        f"Non-finite values in CRF backward: "
        f"grad_cum_scores has {nan_count} NaN, {inf_count} Inf"
    )
```

**Why**: Catch gradient corruption before it propagates to optimizer.

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

## Known Issues

| Issue | Severity | Symptom | Resolution |
|-------|----------|---------|------------|
| Missing `.detach()` in forward | Critical | Double backward errors | Always detach inputs to kernel |
| Wrong checkpoint_interval | Critical | Incorrect gradients | Use same interval in forward/backward |
| float16 overflow | High | NaN in long sequences | Use float32 for cum_scores |
| Non-contiguous input | Medium | Kernel crash or wrong results | Always call `.contiguous()` |

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

- **2026-02-02**: Updated line numbers throughout; clarified per-batch gradient convention
- **2026-02-01**: Added log_norm_checkpoints to save_for_backward and kernel interfaces for T=100k+ numerical stability
- **2026-01-27**: Initial trace @ commit `40fe66b`
