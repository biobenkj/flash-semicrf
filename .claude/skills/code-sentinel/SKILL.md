---
name: code-sentinel
description: >
  Expert system for torch-semimarkov. Manages persistent execution traces ("Sentinels")
  for Triton/PyTorch backends.

  MANDATORY: Invoke this skill BEFORE debugging NaN/Inf, shape mismatches,
  or modifying autograd logic.

  Actions:
  1. Locate active backend via `dispatch-overview.md`.
  2. Load specific trace (e.g., `triton-forward-k3plus.md`) into context.
  3. Compare trace "Verified Commit" against current git HEAD.

  Triggers: "debug semi-markov", "trace execution", "explain backend dispatch",
  "Triton kernel debug", "NaN", "shape mismatch", "gradient flow", "sentinel"
allowed-tools: Read, Grep, Glob, Bash
---

# Code Sentinel

Persistent execution traces ("Sentinels") for torch-semimarkov backends. Prevents hallucinations about code execution paths by maintaining verified baseline documentation.

## Quick Start

1. **Identify the backend**: Check [dispatch-overview.md](traces/dispatch-overview.md)
2. **Load the trace**: Read the relevant trace document
3. **Check staleness**: Verify commit anchor matches current code

## Staleness Check (ALWAYS DO FIRST)

Before providing debugging advice, check if the sentinel is stale:

```bash
# Get current commit for traced file
git log -1 --format=%h -- src/torch_semimarkov/streaming/triton_forward.py

# Check for uncommitted changes
git diff --name-only src/torch_semimarkov/streaming/triton_forward.py
git diff --cached --name-only src/torch_semimarkov/streaming/triton_forward.py
```

Compare against the "Verified against" commit in the trace header.

**If stale, output this EXACT format:**

```
SENTINEL STALE
Source: triton_forward.py
Verified: <old_hash> | Current: <new_hash>
Status: [COMMITTED_CHANGES | UNCOMMITTED_CHANGES | BOTH]

I am now running `git diff` to synchronize my map before providing advice.
```

Then run `git diff <old_hash>..HEAD -- <file>` and update understanding before advising.

## Backend Selection Tree

```
semi_crf_streaming_forward() [autograd.py:474]
|
+-- K == 1 (no boundary projections)
|   +-- needs_grad: LinearCRFStreaming.apply() [line 590]
|   +-- inference: linear_crf_forward_pytorch() [line 605]
|   --> Trace: k1-linear-crf.md
|
+-- K == 2 (no boundary projections)
|   +-- needs_grad: SemiCRFK2Streaming.apply() [line 616]
|   +-- inference: semi_crf_k2_forward_pytorch() [line 631]
|   --> Trace: k2-fast-path.md
|
+-- K >= 3
    +-- GPU + HAS_TRITON + use_triton
    |   +-- needs_grad: SemiCRFStreamingTriton.apply() [line 642]
    |   +-- inference: launch_streaming_triton_kernel() [line 669]
    |   --> Trace: triton-forward-k3plus.md
    |
    +-- CPU or no Triton
        +-- needs_grad: SemiCRFStreaming.apply() [line 655]
        +-- inference: semi_crf_streaming_forward_pytorch() [line 683]
        --> Trace: pytorch-forward-k3plus.md
```

## Failure Mode Routing

| Symptom | Primary Trace | Secondary Trace | Check First |
|---------|---------------|-----------------|-------------|
| **NaN in loss** | triton-forward-k3plus.md | dispatch-overview.md | NEG_INF guards |
| **NaN in backward** | triton-backward-k3plus.md | pytorch-backward-k3plus.md | Partition validation |
| **Wrong gradients** | triton-backward-k3plus.md | pytorch-backward-k3plus.md | Cross-reference outputs |
| **OOM on GPU** | triton-backward-k3plus.md | - | Recomputation Logic section |
| **K=1/K=2 mismatch** | k1-linear-crf.md / k2-fast-path.md | dispatch-overview.md | Dispatch conditions |
| **Triton vs PyTorch diff** | triton-forward-k3plus.md | pytorch-forward-k3plus.md | Ring buffer indexing |

## Available Traces

| Trace | Purpose | Source File |
|-------|---------|-------------|
| [dispatch-overview.md](traces/dispatch-overview.md) | Backend selection decision tree | autograd.py |
| [autograd-kernel-interface.md](traces/autograd-kernel-interface.md) | Contract between autograd and Triton | autograd.py |
| [triton-forward-k3plus.md](traces/triton-forward-k3plus.md) | Triton forward kernel (K>=3, GPU) | triton_forward.py |
| [triton-backward-k3plus.md](traces/triton-backward-k3plus.md) | Triton backward kernel | triton_backward.py |
| [pytorch-forward-k3plus.md](traces/pytorch-forward-k3plus.md) | PyTorch reference forward | pytorch_reference.py |
| [pytorch-backward-k3plus.md](traces/pytorch-backward-k3plus.md) | PyTorch reference backward | pytorch_reference.py |
| [k1-linear-crf.md](traces/k1-linear-crf.md) | K=1 linear CRF fast path | pytorch_reference.py |
| [k2-fast-path.md](traces/k2-fast-path.md) | K=2 specialized path | pytorch_reference.py |

## Update Sentinel Action

When a sentinel is stale, update it using this template:

**Preserve these sections VERBATIM** (unless explicitly changed):
- Critical Invariants
- Known Issues
- Linked Tests

**Regenerate from source:**
- Algorithm Steps (re-trace from code)
- Data Flow (verify shapes)
- Numerical Guards (check for new guards)

**Always update:**
- `Verified against: <new_commit_hash>`
- Version History: add entry with date + change summary

## Symbolic Shape Legend

Used across all traces:
- `B` = Batch size
- `T` = Sequence length (time steps)
- `K` = Maximum segment length
- `C` = Number of classes/labels
- `C_PAD` = Padded class count (power of 2 for Triton)

## Domain-Specific Invariants

These must hold across all backends:

| Invariant | Math | Python Check |
|-----------|------|--------------|
| Log-partition bounds | Z >= max_y s(x,y) | `assert partition >= viterbi_score` |
| Probability simplex | sum_c exp(alpha_t^c) <= 1 | `assert alpha.exp().sum(dim=-1) <= 1 + eps` |
| Padding sentinel | alpha_t^c = -inf for t > T | `assert (alpha[mask] == NEG_INF).all()` |
| Prefix-sum init | cum_scores[t=0] = 0 | `assert (cum_scores[:, 0, :] == 0).all()` |
| Ring buffer aliasing | alpha[t] overwrites alpha[t-K] | Ring index: `t % K`, never read after overwrite |

## Example Invocations

```
/code-sentinel                                    # Show this overview
Trace the Triton forward kernel                   # Load triton-forward-k3plus.md
Why is K=1 using a different code path?           # Load k1-linear-crf.md
Debug: NaN in backward pass with K=8              # Load triton-backward trace + debugging workflow
Update the triton-forward sentinel                # Run Update Sentinel action
```

## Files Reference

| File | Purpose |
|------|---------|
| `streaming/autograd.py` | Backend dispatch, autograd functions |
| `streaming/triton_forward.py` | Triton forward kernels |
| `streaming/triton_backward.py` | Triton backward kernel |
| `streaming/pytorch_reference.py` | PyTorch reference implementations |
| `streaming/constants.py` | Shared constants (NEG_INF) |
