# Sentinel v2: Design Spec

## Context for Implementor

This document describes the evolution of Code Sentinel from a CLI-first verification
tool into a seamless, always-on code understanding layer. The existing implementation
lives at `.sentinel/sentinel.py` (~2400 lines) and is currently piloted in the
`flash-semicrf` repo. The v2 changes will be developed in-place in flash-semicrf, then
extracted to a standalone repo (`code-sentinel`, pip-installable).

**Read these files before implementing:**
- `.sentinel/sentinel.py` — current CLI (all 14 subcommands)
- `.sentinel/spec.md` — formal specification (state model, CLI contract, JSON schema)
- `.sentinel/sentinel.yaml` — project config
- `.sentinel/.sentinel-meta.yaml` — trace metadata
- `.sentinel/anchors/anchors.yaml` — anchor definitions

**Do NOT change:**
- The verification model (anchors, assumptions, state machine, anti-hallucination
  guarantee)
- The trace schema or anchor schema
- The JSON output envelope contract
- The `init` command's intentional 17%/83% scaffolding philosophy
- Exit codes or strictness resolution logic

---

## The Felt Pain

Every developer using an AI coding agent has experienced some version of this:

> You're debugging a function with Claude Code. It confidently explains the control
> flow, references line numbers, tells you where the bug is. You follow the advice,
> make the change — and the tests still fail. You look more carefully and realize:
> the function was refactored two commits ago. The line numbers Claude referenced
> don't correspond to what's there anymore. The invariant it described was true
> *last week* but isn't true now. Claude wasn't lying — it was reasoning from a
> stale understanding of your code, and neither of you noticed.

This isn't a rare edge case. It's the default behavior of every agentic coding tool —
Claude Code, Cursor, Copilot Chat, Codex. They build mental models of codebases from
whatever context is available — documentation, code comments, session memory, or the
source code itself. But code changes constantly, and *nothing tells the agent its
understanding has drifted*. There's no freshness check, no expiration date, no
mechanism to detect that the premises behind the agent's reasoning have moved.

The result is confidently wrong advice. Not hallucination in the usual sense — the
agent isn't inventing facts from nothing. It's reasoning correctly from premises that
are no longer true. This is harder to catch than pure hallucination because the advice
*sounds right* and *was right* at some point.

This happens whether the agent has perfect session memory or none at all. An agent can
read a source file fresh, build an understanding, and have that understanding go stale
two hours later in the same session when someone pushes a refactor. Memory tools solve
recall. **Sentinel solves a different problem: code understanding integrity.** It
detects when any tool's understanding of your code — agent, documentation, architecture
diagram, onboarding guide — has fallen out of sync with reality.

**Code changes. Your AI coding assistant doesn't notice. Sentinel does.**

---

## Why v1 Doesn't Reach People

Sentinel v1 has the right verification model — pattern-based anchors, content hashes,
mechanical staleness detection, anti-hallucination gating — but the wrong interaction
model. The current experience:

1. 14 CLI subcommands that the developer must compose manually
2. The developer is the orchestrator: `verify` → interpret → `retrace --auto --apply`
   → `verify` again
3. The pre-commit hook reports failures but doesn't fix them
4. The agent adapter tells the agent *how to call* Sentinel, but Sentinel never *pushes*
   state to the agent
5. Setup requires manually editing three YAML files after `init` scaffolds a trace

Compare this to the best developer tools: two commands to install, zero commands to
use day-to-day. That's the UX bar. Think of ESLint, Prettier, or even git hooks
themselves — you set them up once and they work invisibly from then on.

The 14 subcommands are the right *plumbing* — they're granular, composable, and
machine-readable. What's missing is the *porcelain*: an experience where Sentinel is
invisible when things are fine and unmissable when they aren't.

---

## Design Principles

### 1. Invisible when healthy, loud when stale

Sentinel should feel like a type checker, not a manual audit. When all traces are
current, you never think about it. When something drifts, it's surfaced immediately —
not at commit time, not when you remember to run `verify`, but *as it happens*.

### 2. Fix what's fixable, flag what's not

Most drift is mechanical: lines shifted but content is unchanged. Sentinel should fix
this silently and automatically. Only genuine semantic drift — where code meaning has
changed — should require human or agent attention. The developer should never be
interrupted for a line number update.

### 3. Two commands for humans, structured events for machines

The human-facing surface is: `sentinel init` (one time) and then Sentinel runs
automatically. The machine-facing surface is a structured event stream (JSON Lines)
that any consumer — VSCode extension, MCP server, tmux status bar, CI pipeline — can
read. The 14 plumbing commands remain for power users and scripts.

### 4. Zero-config defaults, progressive disclosure of control

Everything works out of the box with sensible defaults. Configuration exists for people
who want it, but the first experience should be: install, bootstrap, forget about it
until it saves you from bad advice.

### 5. Meet developers where they already are

The primary integration is not the CLI — it's the agent. For Claude Code users, that
means an MCP server or plugin. For VSCode users, an extension. For CI, a pre-commit
hook. The CLI is the implementation layer; the integrations are the product.

---

## The First Five Minutes

This is the experience we're designing toward. Not all of this ships in Phase 1, but
every Phase 1 decision should move toward this:

```bash
# Install (pip)
pip install code-sentinel

# Bootstrap in an existing project (one-time)
cd my-project
sentinel init --scan

# That's it. Sentinel is now running.
```

`sentinel init --scan` does the following:
1. Creates `.sentinel/` directory structure
2. Scans the repo for source files, detecting language and framework
3. Uses built-in pattern presets (Python/PyTorch, TypeScript/React, Rust, etc.) to
   identify critical functions, entry points, and invariant-bearing code
4. Scaffolds trace documents and anchors for discovered code — same 17%/83% philosophy,
   but the 17% is generated automatically instead of requiring manual YAML editing
5. Computes initial content hashes for all anchors
6. Populates `.sentinel-meta.yaml` with current commit hashes
7. Installs a git pre-commit hook (`sentinel gate`)
8. Writes a `.sentinel/status.json` that integrations can read

After bootstrap, the developer has traces that are VERIFIED against current HEAD. As
they (or their agent) modify code, Sentinel detects drift in real-time (if `watch` is
running) or at commit time (via the pre-commit hook). Mechanical drift is auto-fixed.
Semantic drift surfaces as a clear, actionable message.

**For Claude Code specifically**, the experience should be:

```bash
# In Claude Code
/plugin install code-sentinel
# or
sentinel init --scan --adapter claude
```

The Claude Code adapter (MCP server or plugin hook, depending on the integration path)
registers Sentinel's verification as a tool. When Claude Code is about to give advice
about a code path that has a trace, it automatically verifies the trace first. If the
trace is stale, it tells the developer instead of giving stale advice. This happens
without the developer asking — the same way a type checker surfaces errors without
being invoked explicitly.

---

## Architecture: Two Temperatures

Sentinel v2 operates at two temperatures:

**Hot path (mid-session):** Sentinel monitors source files continuously and pushes
drift events as they occur. The developer or agent doesn't poll — they receive
structured notifications when traces go stale. Trivial mechanical drift (line shifts
where content hash matches) is auto-repaired silently. This is `sentinel watch`, which
runs as a background daemon or is spawned by an editor extension.

**Cold path (pre-commit):** Sentinel runs the full verification pipeline, attempts
auto-repair of mechanical drift, and only blocks the commit if there's genuine semantic
drift requiring human/agent judgment. This is `sentinel gate`, replacing the current
`pipeline` command. The shift is from "report-then-block" to "repair-then-gate."

---

## New Porcelain Commands

### 1. `sentinel watch`

Continuous file monitoring that detects drift as it happens. This is the core of the
"invisible when healthy" experience.

```
sentinel watch [--json-lines] [--auto-fix] [--no-auto-fix] [--traces TRACE1,TRACE2] [--daemon]
```

**Behavior:**

1. On startup: run `verify --all`, emit `status_snapshot` event
2. Monitor all files referenced in `anchors.yaml` and `.sentinel-meta.yaml`
   source_files entries via filesystem events
3. On file change (debounced 500ms per file):
   a. Identify which traces are affected (file → trace mapping)
   b. Re-verify affected anchors only (not full `verify`)
   c. If drift is purely mechanical (content hash matches): auto-fix silently, emit
      `fix_applied` event
   d. If drift is semantic (content hash mismatch or pattern missing): emit
      `semantic_drift` event
4. On git events (commit, checkout, rebase): re-verify all traces
5. Write current state to `.sentinel/status.json` after every state change

**`--daemon` mode:** Fork to background, write PID to `.sentinel/watch.pid`. This is
what editor extensions and the MCP server use to spawn watch automatically.

**`--auto-fix` is ON by default.** The v1 design had this as opt-in. The v2 design
inverts it: auto-fix is the default because the whole point is to be invisible for
mechanical drift. Use `--no-auto-fix` to disable (e.g. in CI where you want to detect
rather than repair).

**`.sentinel/status.json`** — always-current state file:

```json
{
  "timestamp": "2026-03-08T12:00:01Z",
  "traces": {
    "triton-forward-k3plus": {
      "status": "VERIFIED",
      "verified_commit": "abc1234",
      "anchors": {"verified": 7, "total": 7},
      "last_checked": "2026-03-08T12:00:01Z"
    },
    "triton-backward-k3plus": {
      "status": "DEGRADED",
      "verified_commit": "40fe66b",
      "anchors": {"verified": 5, "total": 6},
      "last_checked": "2026-03-08T12:00:01Z",
      "issues": [
        {"anchor": "GRAD_ACCUMULATE", "status": "MISSING", "message": "Pattern deleted"}
      ]
    }
  },
  "summary": "9/10 verified, 1 degraded"
}
```

This file is what makes Sentinel "always-on" without requiring consumers to run CLI
commands. An MCP server reads it. A VSCode extension reads it. A Claude Code skill
reads it. A tmux status bar script reads it. The file is the universal integration
point — cheap to produce, cheap to consume, no IPC needed.

**Event schema** (JSON Lines to stdout when `--json-lines`):

```json
{"event": "status_snapshot", "timestamp": "...", "traces": {...}, "summary": "..."}
{"event": "drift_detected", "timestamp": "...", "trace": "...", "file": "...", "anchors_affected": [...], "auto_fixable": true}
{"event": "fix_applied", "timestamp": "...", "trace": "...", "anchors_updated": [...], "new_status": "VERIFIED"}
{"event": "semantic_drift", "timestamp": "...", "trace": "...", "anchors_affected": [...], "message": "..."}
{"event": "trace_verified", "timestamp": "...", "trace": "...", "status": "VERIFIED"}
{"event": "shutdown", "timestamp": "...", "summary": "..."}
```

**Implementation notes:**

- Use `watchdog` if available, fall back to `stat()`-based polling (2s interval)
- Build file→trace mapping from meta at startup; rebuild on meta file changes
- Debounce 500ms per file (editors write multiple times on save)
- Handle SIGINT/SIGTERM gracefully with summary on exit
- `--auto-fix` fixes ONLY: pattern found + (no content_hash OR content_hash matches)
- Write `.sentinel/status.json` atomically (write-to-temp + rename)


### 2. `sentinel fix`

One-shot "make everything current." The composed verify→retrace→apply→re-verify loop
that v1 required the developer to do manually.

```
sentinel fix [--trace NAME] [--all] [--dry-run] [--format text|json]
```

**Behavior:**

1. Verify all (or specified) traces
2. For each non-VERIFIED trace:
   a. Analyze anchor impacts (`analyze_anchor_impacts`)
   b. Apply all safe updates (shifted anchors where content hash matches)
   c. Update `.sentinel-meta.yaml` verified_commit for traces where all anchors now
      pass and no semantic drift was detected
   d. Update trace markdown headers (`**Verified against:** ... @ commit \`<hash>\``)
   e. Re-verify to confirm
3. Report what was fixed, what still needs attention

**This is the key workflow gap `fix` closes:** Currently `retrace --auto --apply`
updates anchors.yaml but does NOT bump `verified_commit` in meta or trace headers.
So even after a successful auto-fix, the trace still reads as STALE_COMMIT on the
next verify. `fix` completes the loop.

**Output (text mode):**

```
Fixing 3 stale traces...

  triton-forward-k3plus:
    ✓ RING_BUFFER_WRITE: 320→325 (fixed)
    ✓ CHECKPOINT_SAVE: 329→334 (fixed)
    ✓ verified_commit updated: 40fe66b→abc1234
    → VERIFIED

  triton-backward-k3plus:
    ✓ RECOMP_ENTRY: 180→185 (fixed)
    ✗ GRAD_ACCUMULATE: pattern deleted
    → DEGRADED (manual review required)

  k1-linear-crf:
    (already VERIFIED)

Summary: 2 fixed, 1 needs attention
```

**JSON output** follows the standard envelope with a `fixes` payload:

```json
{
  "schema_version": "1.0",
  "command": "fix",
  "timestamp": "...",
  "exit_code": 2,
  "strict_mode_active": false,
  "fixes": {
    "triton-forward-k3plus": {
      "previous_status": "STALE_COMMIT",
      "new_status": "VERIFIED",
      "anchors_fixed": [
        {"name": "RING_BUFFER_WRITE", "old_line": 320, "new_line": 325}
      ],
      "anchors_failed": [],
      "commit_updated": true,
      "new_verified_commit": "abc1234"
    }
  },
  "summary": {"fixed": 2, "still_broken": 1, "already_verified": 1}
}
```

**Commit update logic:** When all anchors pass after applying safe updates:
1. Get current HEAD commit for each file in the trace's `source_files`
2. Update `sentinels.<trace>.verified_commit` in `.sentinel-meta.yaml`
3. Update `**Verified against:** ... @ commit \`<hash>\`` in trace markdown
4. Update `last_global_verification` timestamp
5. Opportunistically populate missing `content_hash` fields


### 3. `sentinel gate`

The pre-commit entry point. Repair, then gate.

```
sentinel gate [--strict] [--no-strict] [--no-fix] [--format text|json]
```

**Behavior:**

1. Unless `--no-fix`: run `fix --all` (attempt auto-repair)
2. Run `verify --all --check-consistency`
3. Apply strictness rules (same precedence as existing `_resolve_strict`)
4. Run test advisory (same as current pipeline step 3)
5. Exit 0 if all traces pass, non-zero otherwise

**Replaces `pipeline` as the pre-commit hook entry point.** `pipeline` becomes an alias
for `gate` with a deprecation notice on stderr. Update `hooks/pre-commit-anchors.sh`:

```bash
python3 "$SENTINEL_DIR/sentinel.py" gate --ci
```


### 4. `sentinel init --scan` (enhanced init)

The current `init` requires a specific source file argument and produces output that
the developer must manually copy into three YAML files. The enhanced version does the
full bootstrap:

```
sentinel init --scan [--preset PRESET] [--adapter claude|codex]
```

**When called without `--scan` (backward compatible):** behaves exactly as today —
takes a source file, scaffolds one trace with suggestions.

**When called with `--scan`:**

1. If `.sentinel/` doesn't exist, create the full directory structure
2. Walk the repo's source tree (respecting `.gitignore`)
3. Apply pattern presets to identify trace-worthy files:
   - Default preset: common patterns (entry points, API handlers, test files)
   - `--preset pytorch`: Triton kernels, autograd, forward/backward
   - `--preset typescript`: route handlers, middleware, hooks
   - Custom presets loadable from `.sentinel/presets/`
4. For each discovered file, scaffold:
   - A trace document in `traces/`
   - Anchors in `anchors/anchors.yaml`
   - Meta entry in `.sentinel-meta.yaml` (with current commit hash)
   - Content hashes for all anchors
5. If `--adapter claude`: also run `install-adapter claude`
6. Install the pre-commit hook (`gate`)
7. Write initial `.sentinel/status.json`

**The output is a working Sentinel installation with VERIFIED traces.** Not drafts —
actual VERIFIED traces with populated commit hashes and content hashes, ready for
`watch` or `gate` to monitor. The traces still contain TODO sections for the 83%
domain knowledge, but the mechanical verification pipeline is operational from minute
one.

**`sentinel init --scan --dry-run`**: Shows what would be discovered and scaffolded
without writing anything. Useful for evaluating whether the preset is detecting the
right files.

---

## Changes to Existing Commands

### `verify` — add `--affected-by FILE` filter

```
sentinel verify --affected-by FILE [FILE ...]
```

Resolves which traces cover the given files (via `.sentinel-meta.yaml` `source_files`)
and verifies only those. Used internally by `watch` and useful for editor integrations.

### `status` — add `--watch-summary`

```
sentinel status --watch-summary
```

Single-line output for status bars: `sentinel: 8/10 verified, 2 stale`

### `pipeline` — deprecated alias for `gate`

Keep for backward compatibility. Print deprecation notice on stderr.

---

## Content Hash (Anchor Schema Addition)

Distinguishing mechanical from semantic drift is the foundation of safe auto-fix.

### Schema change:

```yaml
triton-forward-k3plus:
  RING_BUFFER_WRITE:
    file: src/flash_semicrf/streaming/triton_forward.py
    pattern: "ring_buf[t_idx % K]"
    expected_line: 320
    drift_tolerance: 20
    content_hash: "sha256:<hash-of-stripped-matched-line>"  # NEW
```

### Behavior:

- `content_hash` = SHA-256 hex of the stripped matched line
- `verify`: if content_hash present and anchor found, compare hashes
- Pattern found + hash matches → mechanical drift (auto-fixable)
- Pattern found + hash differs → semantic drift (needs review)
- Pattern not found → structural change (needs review)

### Migration:

**Opportunistic:** `fix` populates missing content_hash fields after successful anchor
verification. No explicit migration command needed — hashes fill in gradually as you
use `fix` or `watch`. This avoids a migration step — the system self-improves with use.

---

## Integration Architecture

### The status.json pattern

The key integration insight is: don't make consumers call your CLI. Give them a file
they can read. `.sentinel/status.json` is that file — the equivalent of a lockfile or
a `.eslintcache`, but for code understanding state.

`watch` keeps it current. `fix` updates it. `gate` updates it. Every state change
writes it atomically. Any consumer that can read a JSON file can know the state of
every trace without importing Sentinel, spawning a process, or understanding the CLI.

### MCP Server (`sentinel-mcp`)

Thin wrapper that exposes Sentinel as MCP tools for Claude Code:

**Tools:**
- `sentinel/status` — returns current status.json content
- `sentinel/verify` — verify a specific trace (calls plumbing)
- `sentinel/fix` — run fix on a trace or all traces
- `sentinel/route` — route a symptom to a trace
- `sentinel/context` — get LLM-optimized context for a trace

**Notifications (from watch):**
- `sentinel/drift` — pushed when a trace goes stale
- `sentinel/fixed` — pushed when auto-fix resolves drift

**The agent experience:** Claude Code's tool surface includes Sentinel tools. When the
agent is about to give code-path-level advice, it calls `sentinel/verify` first (the
adapter instructions tell it to). If the trace is stale, the agent says so. If watch
has already auto-fixed it, the verify returns clean. The developer never has to think
about it — the verification is invisible infrastructure, not a manual step.

### Claude Code Plugin (alternative to MCP)

If the integration path is a Claude Code plugin rather than an MCP server:

**Hooks:**
- `SessionStart`: spawn `sentinel watch --daemon` if not running, inject status summary
- `PostToolUse`: if the tool modified a monitored file, check status.json for drift
- `SessionEnd`: (optional) run `sentinel fix --all` to clean up

**Skills:**
- `sentinel-verify` skill (replaces current adapter): reads status.json, if any trace
  is stale, warns before providing advice

This follows the standard Claude Code plugin architecture: hooks for lifecycle, skills
for behavior, background worker for continuous processing.

### VSCode Extension (Phase 3)

- Spawns `sentinel watch --json-lines --daemon`
- Reads events, maps to VS Code diagnostics API
- Gutter decorations on anchored lines (green/yellow/red)
- Problems panel entries for stale traces
- Command palette: "Sentinel: Fix All", "Sentinel: Show Trace"

---

## Implementation Plan

The phasing is ordered by impact on the felt pain, not by engineering complexity.

### Phase 1: Make it work invisibly (implement now, in flash-semicrf)

The goal: after Phase 1, a developer can bootstrap Sentinel in a project and have
auto-repair and pre-commit gating work without ever manually composing subcommands.

**1a. Content hash support**
- Add `_compute_content_hash(line: str) -> str` to `CodeSentinel`
- Extend `verify_anchor` to check content_hash when present
- Extend `AnchorResult` with `content_hash_match: bool | None`
- Opportunistic population in existing `retrace --auto --apply`

**1b. `sentinel fix`**
- `cmd_fix(args, sentinel) -> int`
- Composes: verify → analyze → safe-update → commit-bump → header-update → re-verify
- Writes updated `.sentinel/status.json`

**1c. `sentinel gate`**
- `cmd_gate(args, sentinel) -> int`
- Composes: fix → verify --all --check-consistency → test advisory
- `pipeline` becomes alias with deprecation notice
- Update `hooks/pre-commit-anchors.sh`

**1d. `status.json` output**
- Written by `fix`, `gate`, `verify`, and (later) `watch`
- Atomic write (temp + rename)
- Schema as defined above

**1e. `sentinel verify --affected-by FILE`**
- File → trace resolution from meta source_files

**1f. `sentinel status --watch-summary`**
- Single-line output

### Phase 2: Make it continuous (implement next)

The goal: after Phase 2, Sentinel runs in the background and auto-fixes drift in
real-time. The MCP server / plugin makes it available to Claude Code without any
agent-side configuration beyond installation.

**2a. `sentinel watch`**
- File monitoring (watchdog or stat-based fallback)
- Auto-fix on by default
- JSON Lines event stream
- `--daemon` mode with PID file
- Debouncing (500ms)
- Continuous status.json updates

**2b. Enhanced `sentinel init --scan`**
- Repo scanning with pattern presets
- Full bootstrap (traces + anchors + meta + hashes + hook)
- `--preset` system with built-in presets for Python/PyTorch, TypeScript, Rust
- `--adapter` flag to also install agent adapter
- `--dry-run` preview

**2c. MCP server or Claude Code plugin**
- Choose integration path based on what Claude Code supports best at the time
- Expose status/verify/fix/route/context as tools
- Push drift notifications from watch
- Agent automatically verifies before advising

### Phase 3: Make it a product (implement for standalone release)

The goal: Sentinel is pip-installable, discoverable, and the README speaks to the
felt pain rather than the architecture.

**3a. Standalone packaging**
- `pip install code-sentinel`
- CLI entry point: `sentinel`
- Domain-specific patterns as installable presets: `pip install sentinel-preset-pytorch`
- Configuration autodiscovery (walk up to find `.sentinel/`)

**3b. VSCode extension**
- `sentinel watch --json-lines` as child process
- Diagnostics, gutter decorations, command palette
- Marketplace listing

**3c. Preset ecosystem**
- Community-contributed pattern presets
- `sentinel init --scan --preset <n>` pulls from registry or local file
- Presets define: init_patterns, common anchors, routing tables

**3d. Documentation and positioning**
- README leads with the felt pain, not the architecture
- "Getting started" is the five-minute experience described above
- Architecture docs are separate from user-facing docs
- Blog post / announcement: "Code changes. Your AI coding assistant doesn't notice."

---

## Positioning: Code Understanding Integrity

### The independent problem

Sentinel addresses a problem that exists regardless of what memory, context, or RAG
system is in play: **code understanding drifts from reality, and nothing detects it.**

This is true for:
- An AI agent in a single session with no memory system at all — it reads a file, you
  refactor it an hour later, and its understanding is now wrong
- A human developer's own mental model — you remember how the auth flow works, but
  someone restructured it last sprint
- A CLAUDE.md, AGENTS.md, or ARCHITECTURE.md that was accurate when written and is
  now three months stale
- A team wiki, onboarding guide, or code review checklist that references APIs that
  have been renamed
- Session memory tools that faithfully recall *what the agent did last time* — but
  "last time" the code was different, so the recalled understanding is outdated

The common failure mode is always the same: something that was true about the code at
time T is assumed to still be true at time T+N, and nobody checks. Sentinel is the
check.

This makes Sentinel's problem space fundamentally different from agent memory. Memory
tools answer "what did we do before?" Sentinel answers "is what we believe about this
code still true right now?" These are orthogonal concerns. You can have perfect recall
of stale understanding (memory without verification) or fresh understanding with no
history (verification without memory). The best outcome is both, but neither depends on
the other.

### The felt pain

The pain statement is: **"Code changes. Your AI coding assistant doesn't notice."**

This is universally true across every agentic coding tool — Claude Code, Cursor,
Copilot Chat, Codex. None of them have a mechanism to detect when their understanding
of your codebase has drifted from reality. The longer a project runs and the more
contributors it has, the worse this gets.

But there's an adoption challenge: this pain is felt *intermittently*. Unlike "I have
to re-explain my project every session" (which happens every session and is immediately
annoying), stale-understanding failures happen unpredictably. You don't always realize
why the agent's advice was wrong — you just think it made a mistake. The cause
(stale understanding) is invisible; only the symptom (bad advice) is visible.

This means:

1. **The pitch needs to make the latent pain vivid.** Concrete scenarios: "Has an AI
   agent ever told you to call a function that was renamed? Referenced a config option
   that was removed? Explained control flow through a branch that was deleted? Suggested
   a fix based on line numbers that don't match the current file?" These are all the
   same failure mode — stale code understanding — and most developers have hit at least
   one without recognizing the pattern.

2. **The tool needs to surface its value visibly.** When Sentinel auto-fixes a drifted
   anchor, it should briefly note it (in watch output, in status.json, in the MCP
   notification). Not noisily — just enough that the developer registers "Sentinel just
   kept my agent from giving me wrong advice" rather than never knowing it was there.
   A status summary at session start ("Sentinel: 10 traces verified, 2 auto-fixed since
   last session") makes the invisible work visible.

3. **The on-ramp needs to be effortless.** `sentinel init --scan` should produce useful
   results on *any* Python project without configuration. The presets make it great for
   specific domains (PyTorch, TypeScript, etc.), but the default preset should be good
   enough to catch function renames, deleted files, and signature changes in any
   language with pattern-based function detection. Start broad, specialize later.

4. **Integration with memory tools is a feature, not the identity.** Sentinel can and
   should integrate with tools like claude-mem, context management systems, and
   documentation generators — any tool that produces or consumes understanding of code
   benefits from a freshness check. But the integration is symmetric: Sentinel verifies
   the understanding that memory tools recall, and memory tools can record Sentinel's
   verification state as part of the project's history. Neither is an extension of the
   other.

### From niche to felt pain

There's a pattern worth naming. Semi-CRFs, flash-semicrf, Sentinel — these all start
as solutions to real problems experienced acutely by a small number of people and
latently by a large number. The challenge is always: bridging from "niche tool that
solves my problem" to "tool that solves a problem everyone has but hasn't named yet."

The bridge for Sentinel is naming the failure mode. "Stale code understanding" isn't
a term people use, but they immediately recognize the experience when you describe it.
The goal is to make "stale understanding" as recognizable a category as "merge
conflict" or "dependency hell" — a named problem that has a named solution.

### Naming and language

The README and docs should avoid:
- "Mechanically-anchored execution traces" (accurate but opaque)
- "Anti-hallucination guarantee" (too academic)
- "Anchor verification states" (implementation detail)

And lean toward:
- "Code changes. Your AI coding assistant doesn't notice. Sentinel does."
- "Catches stale advice before it wastes your time"
- "A freshness check for code understanding"
- "Keeps your AI honest about what it actually knows"

The technical depth should be available (it's a real differentiator for credibility)
but not front-and-center. The README should lead with the problem and the install
command. The architecture lives in dedicated docs.

---

## Config Additions to `sentinel.yaml`

```yaml
# v2 additions
watch:
  auto_fix: true               # auto-fix mechanical drift (default ON)
  debounce_ms: 500
  poll_interval_s: 2           # fallback when watchdog unavailable
  daemon: false                # run as background daemon by default
  status_file: .sentinel/status.json
  ignored_patterns:
    - "*.pyc"
    - "__pycache__/*"

fix:
  update_verified_commit: true  # auto-bump commit hash on successful fix
  update_trace_headers: true    # auto-update "Verified against" in trace .md
  opportunistic_hash: true      # populate content_hash when missing

init:
  default_preset: auto          # auto-detect from repo contents
  presets_dir: .sentinel/presets/
```

---

## Testing Strategy

**`fix`**: Set up a repo with a trace, commit a line-shift change, run `fix`, assert
anchors.yaml updated + meta updated + trace header updated. Repeat with semantic
change (pattern deleted), assert reported as unfixable.

**`gate`**: Same setup, verify exit codes match strictness rules. Verify `--no-fix`
skips auto-repair.

**`watch`**: Test event generation in isolation: given file change + anchor state →
assert correct event payload. Test debouncing with rapid-fire synthetic events. Test
daemon mode (spawn, verify PID file, send SIGTERM, verify clean shutdown).

**`content_hash`**: Unit test `_compute_content_hash`. Integration: set up anchor with
hash, shift line (same content), verify mechanical. Change content, verify semantic.

**`init --scan`**: Run on a small test repo with known structure, verify discovered
traces and anchors match expectations. Test with different presets.

**`status.json`**: Verify atomic write behavior. Verify schema stability across
fix/gate/watch.

---

## Summary

| Aspect | v1 | v2 |
|--------|----|----|
| Felt pain | (implicit) | "Code changes. Your AI coding assistant doesn't notice." |
| Interaction | Developer as orchestrator | Invisible when healthy, loud when stale |
| Default mode | Manual CLI invocation | Background watch + auto-fix |
| Pre-commit | Report then block | Repair then gate |
| Agent integration | Agent shells out to CLI | MCP server / plugin pushes state |
| Bootstrap | Manual YAML editing | `sentinel init --scan` |
| Integration surface | CLI stdout | status.json + JSON Lines events |
| Drift response | Report all equally | Auto-fix mechanical, flag semantic |
| Positioning | Technical infrastructure | "Code changes. Your AI doesn't notice. Sentinel does." |

The verification model, trace schema, anchor schema, and anti-hallucination guarantee
are unchanged. The evolution is: from a tool you operate to a tool that operates
for you.
