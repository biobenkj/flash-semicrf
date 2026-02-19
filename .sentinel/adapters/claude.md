# Code Sentinel - Claude Code Adapter

Integration guide for Claude Code via `.claude/skills/code-sentinel/SKILL.md`.

## How it works

The SKILL.md is a **generated artifact** installed by `sentinel install-adapter claude`.
The source of truth is `.sentinel/adapters/claude.skill.md`.

To update the installed adapter after editing the template:
```bash
python3 .sentinel/sentinel.py install-adapter claude --force
```

## What the adapter provides

- **Trigger keywords** for automatic skill invocation (NaN, gradient, sentinel, etc.)
- **Verification protocol** that gates advice on trace freshness
- **Symptom routing table** for quick failure diagnosis
- **Backend selection tree** for dispatch understanding

## Protocol

1. Before debugging or advising on code paths:
   ```bash
   python3 .sentinel/sentinel.py verify --trace <name>
   ```

2. On test failures or symptoms:
   ```bash
   python3 .sentinel/sentinel.py route "<symptom>"
   ```

3. Before committing:
   ```bash
   python3 .sentinel/sentinel.py pipeline
   ```

## JSON consumption

For programmatic use, add `--format json` to any command.
All JSON output includes `schema_version`, `command`, `timestamp`, `exit_code`, `strict_mode_active`.

## Reference

- `.sentinel/README.md` -- full CLI documentation
- `.sentinel/spec.md` -- sentinel specification
