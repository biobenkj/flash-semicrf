---
name: code-sentinel
description: "Mechanically-anchored code execution traces. Verifies that documentation of code paths, invariants, and algorithm flow is current before providing advice. MANDATORY: Invoke BEFORE debugging NaN/Inf, shape mismatches, modifying autograd logic, or making commits. Triggers: debug semi-markov, trace execution, explain backend dispatch, Triton kernel debug, NaN, shape mismatch, gradient flow, sentinel, verify code, check consistency, pre-commit, retrace"
allowed-tools: Read, Grep, Glob, Bash
---

# Code Sentinel

Maintains verified execution traces anchored to git commits. Traces document
algorithm flow, critical invariants, and numerical stability patterns -- knowledge
that source code alone does not express. When source changes, sentinel detects
drift mechanically and gates advice on verification.

All operations delegate to `.sentinel/sentinel.py`.

## Verification Protocol (ALWAYS EXECUTE FIRST)

```bash
# Step 1: Verify trace is current
python3 .sentinel/sentinel.py verify --trace <trace-name>

# Step 2: If failed, update anchors
python3 .sentinel/sentinel.py retrace <trace-name> --auto --apply
```

On PASS: trace is anchored to current source. Proceed with advice.
On FAIL: trace is stale. Do NOT provide path-level advice until updated.

## Before Committing

```bash
python3 .sentinel/sentinel.py pipeline
```

## Symptom Routing

```bash
python3 .sentinel/sentinel.py route "<symptom>"
```

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
- `.sentinel/spec.md` -- sentinel specification
- `.sentinel/traces/` -- all trace documents
