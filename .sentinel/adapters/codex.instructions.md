# Code Sentinel

Mechanically-anchored code execution traces for this codebase. Traces document
algorithm flow, critical invariants, and domain knowledge that source code alone
does not express. Each trace is verified against a specific git commit -- when
source changes, sentinel detects the drift and flags stale documentation.

Before providing code path advice, modifying traced source files, or committing
changes, you MUST call the sentinel CLI to verify traces are current.

## Required: Before debugging or advising on code paths

```bash
python3 .sentinel/sentinel.py verify --trace <trace-name> --format json
```

If `exit_code` != 0 in the JSON output, the trace is stale or missing.
Do NOT provide path-level advice based on stale trace documentation.

## Required: Before committing

```bash
python3 .sentinel/sentinel.py pipeline
```

## On test failures or symptoms

```bash
python3 .sentinel/sentinel.py route "<symptom>" --format json
```

Follow the `trace` and `guidance` fields in the response.

## Quick status check

```bash
python3 .sentinel/sentinel.py status --verify --format json
```

## Updating stale traces

```bash
python3 .sentinel/sentinel.py update --trace <name>
```

## Available traces

Run `python3 .sentinel/sentinel.py status` to list all traces and their states.

## Reference

- `.sentinel/README.md` -- full CLI documentation
- `.sentinel/spec.md` -- sentinel specification
- `.sentinel/sentinel.yaml` -- project configuration
