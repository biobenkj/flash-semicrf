---
name: code-sentinel
description: "Anti-hallucination guard. Calls .sentinel/sentinel.py to verify execution traces before providing code path advice. MANDATORY: Invoke BEFORE debugging NaN/Inf, shape mismatches, modifying autograd logic, or making commits. Triggers: debug semi-markov, trace execution, explain backend dispatch, Triton kernel debug, NaN, shape mismatch, gradient flow, sentinel, verify code, check consistency, pre-commit, retrace"
allowed-tools: Read, Grep, Glob, Bash
---

# Code Sentinel (Claude Adapter)

All sentinel operations delegate to the core CLI at `.sentinel/sentinel.py`.

## Verification Protocol (ALWAYS EXECUTE FIRST)

```bash
# Step 1: Verify trace
python3 .sentinel/sentinel.py verify --trace <trace-name>

# Step 2: If failed, remediate
python3 .sentinel/sentinel.py retrace <trace-name> --auto --apply
```

On PASS: proceed with advice.
On FAIL: do NOT provide path-level advice until sentinel is updated.

## Before Committing

```bash
python3 .sentinel/sentinel.py pipeline
```

## Failure Mode Routing

| Symptom | Primary Trace | Check First |
|---------|---------------|-------------|
| NaN in loss | triton-forward-k3plus | NEG_INF guards |
| NaN in backward | triton-backward-k3plus | Partition validation |
| Wrong gradients | triton-backward-k3plus | Cross-reference outputs |
| OOM on GPU | triton-backward-k3plus | Recomputation Logic section |
| K=1/K=2 mismatch | k1-linear-crf | Dispatch conditions |
| Triton vs PyTorch diff | triton-forward-k3plus | Ring buffer indexing |

## Backend Selection Tree

```
semi_crf_streaming_forward() [autograd.py:474]
|
+-- K == 1 --> k1-linear-crf.md
+-- K == 2 --> k2-fast-path.md
+-- K >= 3
    +-- GPU + Triton --> triton-forward-k3plus.md
    +-- CPU / no Triton --> pytorch-forward-k3plus.md
```

## Reference

- `.sentinel/README.md` -- full CLI docs and configuration
- `.sentinel/traces/` -- all trace documents
- `.sentinel/.sentinel-meta.yaml` -- machine-readable sentinel state
