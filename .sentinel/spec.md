# Sentinel Specification v1.0

Agent-neutral specification for Code Sentinel implementations.

## Overview

Code Sentinel maintains persistent documentation of code execution paths
("traces") that are mechanically anchored to git commits and verified against
source code. Traces capture algorithm flow, critical invariants, and domain
knowledge that source code alone does not express. When source code changes,
sentinel detects the drift and flags stale documentation before any agent
or developer relies on it.

Any conforming implementation must satisfy this specification.

## Trace Schema

A trace is a Markdown document that captures verified knowledge about a code path.

### Required header

```markdown
# Sentinel: <trace-name>

**Verified against:** `<file-path>` @ commit `<short-hash>`
**Linked tests:** `<test-reference>`
```

### Required sections

- **Summary** -- purpose and key functionality
- **Active Assumptions** -- mechanically verified and agent-verified tables
- **Algorithm Flow** -- line-referenced execution steps
- **Critical Invariants** -- properties that must hold for correctness

### Optional sections

- Known Issues, Version History, Data Flow, Shape Legend

## Anchor Schema

Anchors are pattern-based references to specific code locations in `anchors.yaml`.

```yaml
<trace-name>:
  <ANCHOR_NAME>:
    file: <relative-path>          # Required. Relative to repo root.
    pattern: "<string-pattern>"    # Required. Literal string to search for.
    expected_line: <integer>       # Required. Baseline line number.
    drift_tolerance: <integer>     # Required. Max acceptable line drift.
    after: "<context-pattern>"     # Optional. Search only after this pattern.
    content_hash: "<sha256>"       # Optional in v1, required in v2. Hash of matched line.
```

### Anchor verification states

- `ANCHOR_VERIFIED` -- pattern found within drift tolerance
- `ANCHOR_MISSING` -- pattern not found in file
- `ANCHOR_DRIFT` -- pattern found but beyond tolerance
- `ANCHOR_AMBIGUOUS` -- pattern matches multiple lines

## State Model

### States

| State | Meaning |
|-------|---------|
| `VERIFIED` | Anchors pass, commit matches verified hash |
| `STALE_COMMIT` | Source file has newer commits than verified hash |
| `STALE_CONTENT` | Source file has uncommitted changes |
| `DEGRADED` | Anchors or assumptions failed but trace exists |
| `MISSING` | Trace not found in metadata |

### Transitions

```
VERIFIED --> STALE_COMMIT    (new commits on source files)
VERIFIED --> STALE_CONTENT   (uncommitted changes on source files)
VERIFIED --> DEGRADED        (anchor drift or assumption failure)
STALE_*  --> VERIFIED        (after update + re-verify)
DEGRADED --> VERIFIED        (after retrace + re-verify)
MISSING  --> VERIFIED        (after init + populate + verify)
```

### Anti-hallucination guarantee

An agent MUST NOT provide code-path-level advice for a trace in any state
other than VERIFIED. The `verify` command enforces this:
- MISSING always causes non-zero exit
- STALE_COMMIT / STALE_CONTENT cause non-zero exit in strict mode
- DEGRADED always causes non-zero exit

## CLI Contract

### Required commands

Every conforming implementation must provide:

| Command | Description |
|---------|-------------|
| `init <source>` | Scaffold a new trace from a source file |
| `status` | Show sentinel health (metadata-based by default) |
| `verify --trace <name>` | Verify a specific trace against current source |
| `verify --all` | Verify all traces |
| `route <symptom>` | Route a symptom to the appropriate trace |
| `update --trace <name>` | Auto-update anchor line numbers |

### Optional extensions

These commands are valuable but not required for spec compliance:

| Command | Description |
|---------|-------------|
| `pipeline` | Full pre-commit verification pipeline |
| `report` | Generate machine-readable verification report |
| `install-adapter` | Install/update agent registration from templates |
| `graph` | Generate dependency graph visualization |
| `coverage` | Trace coverage metrics |
| `context` | Generate LLM-optimized context |
| `sync-docs` | Regenerate markdown tables from structured data |

`install-adapter` implementations SHOULD support `--dry-run` to preview file changes
without writing.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All verification passed |
| 1 | One or more anchors missing |
| 2 | One or more anchors drifted (or stale in strict mode) |
| 3 | One or more anchor patterns ambiguous |
| 4 | Consistency check failed |
| 5 | Assumption verification failed |
| 10 | General error (conflicting flags, missing config, etc.) |

### Strictness

Implementations must support three layers of strictness control:

1. If both `--strict` and `--no-strict` are passed: exit 10
2. Else if `--strict`: strict mode
3. Else if `--no-strict`: advisory mode
4. Else if `SENTINEL_STRICT=1` env var: strict mode
5. Else use config file `ci.strict_mode`
6. Else default: advisory

In strict mode, STALE_COMMIT and STALE_CONTENT cause exit code 2.
In advisory mode (default), they produce warnings but exit 0.

## JSON Output Contract

All `--format json` output must include a stable envelope:

```json
{
  "schema_version": "1.0",
  "command": "<command-name>",
  "timestamp": "<ISO-8601>",
  "exit_code": <integer>,
  "strict_mode_active": <boolean>
}
```

### Per-command payload

**status:**
- `traces` -- object mapping trace name to status string
- `last_global_verification` -- ISO 8601 or null
- `status_source` -- `"metadata"` or `"live"`

**verify:**
- `traces` -- object mapping trace name to verification detail
  - `status`, `commit_status`, `verified_commit`, `current_commit`, `uncommitted_changes`
  - `anchors` -- `{verified, total, details: [{name, status, expected, actual}]}`
  - `assumptions` -- `{passed, total, details: [{id, passed, message}]}`
- `consistency_errors` -- array of strings

**route:**
- `symptom` -- the input symptom string
- `matched_key` -- the routing entry that matched
- `trace` -- the suggested trace name
- `guidance` -- what to check first
- `match_type` -- `"exact"`, `"case_insensitive"`, or `"substring"`
- `verification` -- compact verify summary:
  - `trace`, `status`, `commit_status`, `anchors_verified`, `anchors_total`

### Evolution policy

New keys may be added in future versions. Existing keys will not change
semantics within a major `schema_version`. Implementations should ignore
unknown keys when consuming JSON output.

## Agent Adapter Requirements

An adapter translates agent-native conventions into sentinel CLI calls.

### Mandatory behavior

1. Before providing code-path-level advice: call `verify --trace <name>`
2. If exit code != 0: refuse to advise on that code path
3. Before committing: call `pipeline` (or `verify --all`)
4. On symptoms/failures: call `route "<symptom>"` to identify the right trace

### Adapter location

- Claude Code: `.claude/skills/code-sentinel/SKILL.md`
- Codex / others: `.sentinel/adapters/<agent>.md`
- All adapters delegate to the same `.sentinel/sentinel.py` CLI

## Configuration

Project config lives at `.sentinel/sentinel.yaml`:

```yaml
version: "1.0"

project:
  name: <project-name>
  src_root: <source-directory>

init_patterns:
  critical: [<regex-patterns>]
  high: [<regex-patterns>]

routing:
  - symptom: "<description>"
    trace: <trace-name>
    guidance: "<what-to-check>"

ci:
  strict_mode: <boolean>

adapters:
  <adapter-name>:
    template: <path-under-.sentinel>
    target: <path-under-repo-root>
    mode: copy | managed_block   # optional, default copy
    block_start: "<marker>"      # optional, managed_block only
    block_end: "<marker>"        # optional, managed_block only
```

The `routing` field is an ordered list (not a map) to ensure deterministic
match priority across YAML implementations.

For safety, implementations MUST reject adapter template/target paths that
escape their allowed roots (`.sentinel/` for templates, repo root for targets).
