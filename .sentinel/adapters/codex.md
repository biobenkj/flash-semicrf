# Code Sentinel - Codex / General Agent Adapter

Integration guide for OpenAI Codex, Cursor, and other coding agents.

## How it works

The instructions file is a **generated artifact** installed by `sentinel install-adapter codex`.
The source of truth is `.sentinel/adapters/codex.instructions.md`.

For the codex adapter, `managed_block` mode is used -- sentinel owns a marked block
within the target file (AGENTS.md), leaving any other content untouched.

To update the installed adapter after editing the template:

```bash
python3 .sentinel/sentinel.py install-adapter codex --force
```

## Protocol

All agents MUST call the sentinel CLI before providing code path advice.
The CLI is at `.sentinel/sentinel.py` and requires Python 3.10+ and pyyaml.

1. Before starting a task:

   ```bash
   python3 .sentinel/sentinel.py status --verify --format json
   ```

2. Before modifying traced files:

   ```bash
   python3 .sentinel/sentinel.py verify --trace <name> --format json
   ```

3. On test failures:

   ```bash
   python3 .sentinel/sentinel.py route "<symptom>" --format json
   ```

4. Before committing:

   ```bash
   python3 .sentinel/sentinel.py pipeline
   ```

## JSON output contract

All JSON responses include:
- `schema_version` -- currently "1.0"
- `command`, `timestamp`, `exit_code`, `strict_mode_active`

New keys may be added; existing keys are stable within a major schema_version.

## Strictness

By default, stale traces produce warnings but exit 0 (advisory mode).
For CI or strict enforcement:

```bash
python3 .sentinel/sentinel.py verify --all --strict
```

## Reference

- `.sentinel/README.md` -- full CLI documentation
- `.sentinel/spec.md` -- sentinel specification
- `.sentinel/sentinel.yaml` -- project configuration
