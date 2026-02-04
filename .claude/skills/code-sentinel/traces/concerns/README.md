# Future: Concern-Based Traces

This directory is a stub for future concern-based traces that will reduce duplication across backend traces.

## Planned Traces

When duplication becomes burdensome, extract these cross-cutting concerns:

### ring-buffer-semantics.md

Common ring buffer behavior shared by all K>=3 traces:
- `t % K` indexing pattern
- Aliasing invariant: alpha[t] overwrites alpha[t-K]
- Variable length masking
- Checkpoint save/restore

### logsumexp-numerics.md

Numerical stability patterns shared by forward traces:
- NEG_INF sentinel handling
- Guard against all-NEG_INF inputs
- Two-way logsumexp accumulation
- Epsilon for numerical stability (1e-10)

### gradient-checkpointing.md

Memory-efficient backward shared by backward traces:
- Checkpoint interval computation (sqrt(T*K))
- Segment-wise alpha recomputation
- Beta ring buffer
- Log marginal clamping (-80, 80)

## When to Extract

Extract a concern when:
1. The same explanation appears in 3+ traces
2. Bug fixes need to be propagated across multiple traces
3. Invariants are getting out of sync

## How to Reference

Backend traces will reference concern traces like:

```markdown
## Ring Buffer Mechanics

See [ring-buffer-semantics.md](../concerns/ring-buffer-semantics.md) for details.
```
