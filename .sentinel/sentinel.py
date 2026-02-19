#!/usr/bin/env python3
"""
Code Sentinel - Main CLI orchestrator for verification pipeline.

Unified interface for all sentinel operations in flash-semicrf.

Usage:
    ./sentinel.py status                          # Show sentinel health
    ./sentinel.py verify --trace NAME             # Verify specific trace
    ./sentinel.py verify --all                    # Verify all traces
    ./sentinel.py pipeline                        # Full pre-commit pipeline
    ./sentinel.py retrace NAME                    # Interactive sentinel update
    ./sentinel.py install-hooks                   # Install git pre-commit hooks
    ./sentinel.py report --format json            # Generate verification report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# Exit codes
EXIT_SUCCESS = 0
EXIT_ANCHOR_MISSING = 1
EXIT_ANCHOR_DRIFT = 2
EXIT_ANCHOR_AMBIGUOUS = 3
EXIT_CONSISTENCY_FAILED = 4
EXIT_ASSUMPTION_FAILED = 5
EXIT_GENERAL_ERROR = 10

# Importance patterns for init command (domain-specific for flash-semicrf)
CRITICAL_PATTERNS = [
    r"def forward\(",
    r"def backward\(",
    r"\.apply\(",
    r"@triton\.jit",
    r"def semi_crf",
    r"def launch_",
]
HIGH_PATTERNS = [
    r"def __init__\(",
    r"logsumexp",
    r"NEG_INF",
    r"torch\.isfinite",
    r"torch\.isnan",
    r"torch\.clamp",
    r"def _score",
    r"def decode",
]


class Status(Enum):
    VERIFIED = "VERIFIED"
    STALE_COMMIT = "STALE_COMMIT"
    STALE_CONTENT = "STALE_CONTENT"
    DEGRADED = "DEGRADED"
    MISSING = "MISSING"


@dataclass
class AnchorResult:
    name: str
    status: str  # VERIFIED, MISSING, DRIFT, AMBIGUOUS
    expected_line: int
    actual_line: int | None
    drift: int | None
    message: str


@dataclass
class AssumptionResult:
    id: str
    description: str
    passed: bool
    message: str


@dataclass
class TraceVerification:
    name: str
    commit_status: Status
    verified_commit: str
    current_commit: str | None
    uncommitted_changes: bool
    anchors: list[AnchorResult] = field(default_factory=list)
    assumptions: list[AssumptionResult] = field(default_factory=list)

    @property
    def overall_status(self) -> Status:
        if self.commit_status in (Status.STALE_COMMIT, Status.STALE_CONTENT, Status.MISSING):
            return self.commit_status
        if any(a.status != "VERIFIED" for a in self.anchors):
            return Status.DEGRADED
        if any(not a.passed for a in self.assumptions):
            return Status.DEGRADED
        return Status.VERIFIED


@dataclass
class AnchorImpact:
    """Impact assessment for an anchor after code changes."""

    anchor_name: str
    status: str  # unchanged, shifted, modified, deleted
    old_line: int
    new_line: int | None = None
    suggestion: str = ""


@dataclass
class FunctionInfo:
    """Extracted function/method information for init command."""

    name: str
    line: int
    importance: str  # critical, high, medium


class CodeSentinel:
    def __init__(self, sentinel_dir: Path | None = None):
        self.sentinel_dir = sentinel_dir or self._find_sentinel_dir()
        self.repo_root = self._find_repo_root()
        self.meta_path = self.sentinel_dir / ".sentinel-meta.yaml"
        self.anchors_path = self.sentinel_dir / "anchors" / "anchors.yaml"
        self.traces_dir = self.sentinel_dir / "traces"
        self.config = self.load_config()

    def _find_sentinel_dir(self) -> Path:
        """Find the sentinel directory."""
        # Check common locations (agent-neutral first, then legacy)
        candidates = [
            Path(__file__).parent,
            Path(".sentinel"),
            Path.cwd() / ".sentinel",
            Path(".claude/skills/code-sentinel"),
            Path.cwd() / ".claude/skills/code-sentinel",
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / ".sentinel-meta.yaml").exists():
                return candidate.resolve()
        raise FileNotFoundError("Could not find .sentinel directory")

    def _find_repo_root(self) -> Path:
        """Find the git repository root."""
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=self.sentinel_dir,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        # Fallback: walk up from sentinel_dir
        current = self.sentinel_dir
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return self.sentinel_dir.parent

    def load_meta(self) -> dict:
        """Load .sentinel-meta.yaml"""
        if not self.meta_path.exists():
            return {"version": "2.0", "sentinels": {}}
        return yaml.safe_load(self.meta_path.read_text())

    def load_anchors(self) -> dict:
        """Load anchors.yaml"""
        if not self.anchors_path.exists():
            return {}
        return yaml.safe_load(self.anchors_path.read_text())

    def get_current_commit(self, file_path: str) -> str | None:
        """Get the current commit hash for a file."""
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", file_path],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    def has_uncommitted_changes(self, file_path: str) -> bool:
        """Check if file has uncommitted changes."""
        # Check staged changes
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--", file_path],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        # Check unstaged changes
        unstaged = subprocess.run(
            ["git", "diff", "--name-only", "--", file_path],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        return bool(staged.stdout.strip() or unstaged.stdout.strip())

    def verify_anchor(
        self,
        pattern: str,
        expected_line: int,
        file_path: str,
        tolerance: int = 20,
        after: str | None = None,
    ) -> AnchorResult:
        """Verify a single anchor using verify-anchor.sh."""
        script = self.sentinel_dir / "anchors" / "verify-anchor.sh"
        cmd = [str(script), pattern, str(expected_line), file_path, str(tolerance)]
        if after:
            cmd.append(after)

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
        output = result.stdout.strip()

        # Parse output
        actual_line = None
        drift = None
        if "Line" in output:
            match = re.search(r"Line (\d+)", output)
            if match:
                actual_line = int(match.group(1))
        if "drift" in output:
            match = re.search(r"drift (\d+)", output)
            if match:
                drift = int(match.group(1))

        status_map = {0: "VERIFIED", 1: "MISSING", 2: "DRIFT", 3: "AMBIGUOUS"}
        return AnchorResult(
            name="",  # Will be set by caller
            status=status_map.get(result.returncode, "UNKNOWN"),
            expected_line=expected_line,
            actual_line=actual_line,
            drift=drift,
            message=output,
        )

    def verify_trace(self, trace_name: str, check_assumptions: bool = True) -> TraceVerification:
        """Run full verification for a trace."""
        meta = self.load_meta()
        anchors = self.load_anchors()

        trace_meta = meta.get("sentinels", {}).get(trace_name, {})
        if not trace_meta:
            return TraceVerification(
                name=trace_name,
                commit_status=Status.MISSING,
                verified_commit="",
                current_commit=None,
                uncommitted_changes=False,
            )

        verified_commit = trace_meta.get("verified_commit", "")
        source_files = trace_meta.get("source_files", [])

        # Check commit status
        current_commit = None
        uncommitted = False
        commit_status = Status.VERIFIED

        for source_file in source_files:
            file_commit = self.get_current_commit(source_file)
            if file_commit:
                current_commit = file_commit
            if self.has_uncommitted_changes(source_file):
                uncommitted = True

        if current_commit and current_commit != verified_commit:
            commit_status = Status.STALE_COMMIT
        elif uncommitted:
            commit_status = Status.STALE_CONTENT

        # Verify anchors
        anchor_results = []
        trace_anchors = anchors.get(trace_name, {})
        for anchor_name, spec in trace_anchors.items():
            result = self.verify_anchor(
                pattern=spec["pattern"],
                expected_line=spec["expected_line"],
                file_path=str(self.repo_root / spec["file"]),
                tolerance=spec.get("drift_tolerance", 20),
                after=spec.get("after"),
            )
            result.name = anchor_name
            anchor_results.append(result)

        # Verify assumptions
        assumption_results = []
        if check_assumptions:
            assumption_results = self._verify_assumptions(trace_name, anchors)

        return TraceVerification(
            name=trace_name,
            commit_status=commit_status,
            verified_commit=verified_commit,
            current_commit=current_commit,
            uncommitted_changes=uncommitted,
            anchors=anchor_results,
            assumptions=assumption_results,
        )

    def _verify_assumptions(self, trace_name: str, anchors: dict) -> list[AssumptionResult]:
        """Verify mechanically-checkable assumptions for a trace."""
        trace_path = self.traces_dir / f"{trace_name}.md"
        if not trace_path.exists():
            return []

        content = trace_path.read_text()

        # Parse assumptions table (same regex as verify-assumptions.py)
        match = re.search(
            r"### Mechanically Verified\n\n"
            r".*?\n\n"
            r"\| ID \| Assumption \| Verification \|\n"
            r"\|[-|]+\|\n"
            r"((?:\| [A-Z][0-9]+ \|[^\n]+\|\n)+)",
            content,
            re.DOTALL,
        )
        if not match:
            return []

        results = []
        for line in match.group(1).strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3 and re.match(r"[A-Z]\d+", parts[0]):
                assumption_id = parts[0]
                description = parts[1]
                verification = parts[2]

                passed = True
                message = "Verified"

                if verification.startswith("anchor:"):
                    anchor_name = verification.split(":")[1].strip()
                    trace_anchors = anchors.get(trace_name, {})
                    if anchor_name in trace_anchors:
                        spec = trace_anchors[anchor_name]
                        anchor_result = self.verify_anchor(
                            pattern=spec["pattern"],
                            expected_line=spec["expected_line"],
                            file_path=str(self.repo_root / spec["file"]),
                            tolerance=spec.get("drift_tolerance", 20),
                            after=spec.get("after"),
                        )
                        passed = anchor_result.status == "VERIFIED"
                        message = anchor_result.message
                    else:
                        passed = False
                        message = f"Anchor {anchor_name} not defined"
                elif verification.startswith("`") and verification.endswith("`"):
                    # Shell command verification
                    cmd = verification[1:-1]
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, cwd=self.repo_root
                    )
                    passed = result.returncode == 0
                    message = "Command passed" if passed else "Command failed"

                results.append(
                    AssumptionResult(
                        id=assumption_id, description=description, passed=passed, message=message
                    )
                )

        return results

    def check_consistency(self) -> list[str]:
        """Check consistency between meta, anchors, and trace headers."""
        errors = []
        meta = self.load_meta()
        anchors = self.load_anchors()

        for trace_name, trace_meta in meta.get("sentinels", {}).items():
            # Check anchors exist in anchors.yaml
            for anchor_id in trace_meta.get("anchors", []):
                if trace_name not in anchors or anchor_id not in anchors.get(trace_name, {}):
                    errors.append(
                        f"{trace_name}: Anchor '{anchor_id}' in meta but not in anchors.yaml"
                    )

            # Check verified_commit matches trace header
            trace_path = self.traces_dir / f"{trace_name}.md"
            if trace_path.exists():
                trace_content = trace_path.read_text()
                match = re.search(
                    r"\*\*Verified against:\*\*.*?@ commit `([a-f0-9]+)`", trace_content
                )
                if match:
                    trace_commit = match.group(1)
                    meta_commit = trace_meta.get("verified_commit", "")
                    if trace_commit != meta_commit:
                        errors.append(
                            f"{trace_name}: Commit mismatch - meta='{meta_commit}', trace='{trace_commit}'"
                        )

        return errors

    def get_suggested_tests(self, modified_files: list[str]) -> list[str]:
        """Get suggested tests for modified files."""
        meta = self.load_meta()
        bindings = meta.get("test_bindings", {})
        tests = set()
        for f in modified_files:
            for pattern, test_list in bindings.items():
                if pattern in f:
                    tests.update(test_list)
        return sorted(tests)

    def get_modified_files(self) -> list[str]:
        """Get list of staged files."""
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
        return []

    def extract_functions(self, source_path: Path) -> list[FunctionInfo]:
        """Extract function names and line numbers from a Python source file."""
        content = source_path.read_text()
        functions = []

        # Match function definitions with varying indentation
        pattern = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)

        for i, line in enumerate(content.splitlines(), 1):
            match = pattern.match(line)
            if match:
                indent = match.group(1)
                name = match.group(3)
                # Only include top-level or class methods (indent <= 4 spaces)
                if len(indent) <= 4:
                    # Determine importance
                    critical_pats = self.config.get("init_patterns", {}).get(
                        "critical", CRITICAL_PATTERNS
                    )
                    high_pats = self.config.get("init_patterns", {}).get("high", HIGH_PATTERNS)
                    importance = "medium"
                    for pat in critical_pats:
                        if re.search(pat, line):
                            importance = "critical"
                            break
                    if importance == "medium":
                        for pat in high_pats:
                            if re.search(pat, line):
                                importance = "high"
                                break
                    functions.append(FunctionInfo(name=name, line=i, importance=importance))

        return functions

    def extract_classes(self, source_path: Path) -> list[tuple[str, int]]:
        """Extract class names and line numbers from a Python source file."""
        content = source_path.read_text()
        classes = []

        pattern = re.compile(r"^class\s+(\w+)", re.MULTILINE)

        for i, line in enumerate(content.splitlines(), 1):
            match = pattern.match(line)
            if match:
                classes.append((match.group(1), i))

        return classes

    def get_files_changed_since(self, commit: str) -> set[str]:
        """Get set of files changed between commit and HEAD."""
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{commit}..HEAD"],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )
        if result.returncode == 0:
            return {f for f in result.stdout.strip().split("\n") if f}
        return set()

    def find_pattern_line(
        self, file_path: Path, pattern: str, after: str | None = None
    ) -> int | None:
        """Find the line number where a pattern occurs."""
        try:
            content = file_path.read_text()
            lines = content.splitlines()

            start_idx = 0
            if after:
                for i, line in enumerate(lines):
                    if after in line:
                        start_idx = i + 1
                        break

            for i, line in enumerate(lines[start_idx:], start_idx + 1):
                if pattern in line:
                    return i
            return None
        except Exception:
            return None

    def analyze_anchor_impacts(self, trace_name: str) -> list[AnchorImpact]:
        """Analyze how current HEAD affects trace anchors."""
        meta = self.load_meta()
        anchors = self.load_anchors()

        trace_meta = meta.get("sentinels", {}).get(trace_name, {})
        if not trace_meta:
            return []

        verified_commit = trace_meta.get("verified_commit", "")
        trace_anchors = anchors.get(trace_name, {})

        # Get changed files
        changed_files = self.get_files_changed_since(verified_commit)

        impacts = []
        for anchor_name, spec in trace_anchors.items():
            anchor_file = spec["file"]
            old_line = spec["expected_line"]

            # Check if the file was changed
            if anchor_file not in changed_files:
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="unchanged",
                        old_line=old_line,
                        new_line=old_line,
                    )
                )
                continue

            # Re-verify anchor at HEAD
            result = self.verify_anchor(
                pattern=spec["pattern"],
                expected_line=old_line,
                file_path=str(self.repo_root / anchor_file),
                tolerance=spec.get("drift_tolerance", 20),
                after=spec.get("after"),
            )

            if result.status == "MISSING":
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="deleted",
                        old_line=old_line,
                        new_line=None,
                        suggestion="Pattern no longer found - manual review required",
                    )
                )
            elif result.status == "VERIFIED":
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="unchanged",
                        old_line=old_line,
                        new_line=result.actual_line,
                    )
                )
            elif result.status == "DRIFT":
                # Check if this is just a line shift or semantic change
                # For now, treat all drifts as shifts (safe to auto-update)
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="shifted",
                        old_line=old_line,
                        new_line=result.actual_line,
                        suggestion=f"Update expected_line: {old_line} -> {result.actual_line}",
                    )
                )
            elif result.status == "AMBIGUOUS":
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="modified",
                        old_line=old_line,
                        new_line=None,
                        suggestion="Pattern matches multiple lines - add 'after' disambiguator",
                    )
                )

        return impacts

    def save_anchors(self, anchors: dict) -> None:
        """Save anchors to anchors.yaml."""
        with open(self.anchors_path, "w") as f:
            yaml.dump(anchors, f, default_flow_style=False, sort_keys=False)

    def load_config(self) -> dict:
        """Load sentinel.yaml project config."""
        config_path = self.sentinel_dir / "sentinel.yaml"
        if not config_path.exists():
            return {}
        return yaml.safe_load(config_path.read_text()) or {}


def _json_envelope(command: str, exit_code: int, strict_mode: bool = False, **payload) -> str:
    """Build JSON output with stable schema envelope."""
    envelope = {
        "schema_version": "1.0",
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "exit_code": exit_code,
        "strict_mode_active": strict_mode,
    }
    envelope.update(payload)
    return json.dumps(envelope, indent=2)


def _resolve_strict(args: argparse.Namespace, sentinel: CodeSentinel) -> tuple[bool, str | None]:
    """Resolve strictness mode with precedence: CLI flag > env > config > default.

    Returns `(strict_mode, error_message)`. If `error_message` is not None,
    strictness flags were invalid and callers should return EXIT_GENERAL_ERROR.
    """
    has_strict = getattr(args, "strict", False)
    has_no_strict = getattr(args, "no_strict", False)

    if has_strict and has_no_strict:
        return False, "conflicting strictness flags (--strict and --no-strict)"

    if has_strict:
        return True, None
    if has_no_strict:
        return False, None
    if os.environ.get("SENTINEL_STRICT") == "1":
        return True, None
    return sentinel.config.get("ci", {}).get("strict_mode", False), None


def cmd_status(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Show overall sentinel health."""
    meta = sentinel.load_meta()
    use_json = getattr(args, "format", None) == "json"
    live = getattr(args, "verify", False)

    traces_data = {}
    for trace_name, trace_meta in meta.get("sentinels", {}).items():
        if live:
            result = sentinel.verify_trace(trace_name)
            traces_data[trace_name] = result.overall_status.value
        else:
            traces_data[trace_name] = trace_meta.get("status", "UNKNOWN")

    if use_json:
        last_ver = meta.get("last_global_verification")
        if hasattr(last_ver, "isoformat"):
            last_ver = last_ver.isoformat()
        print(
            _json_envelope(
                "status",
                exit_code=EXIT_SUCCESS,
                traces=traces_data,
                last_global_verification=str(last_ver) if last_ver else None,
                status_source="live" if live else "metadata",
            )
        )
    else:
        print("Code Sentinel Status")
        print("=" * 40)
        print(f"Last global verification: {meta.get('last_global_verification', 'Unknown')}")
        if live:
            print("(live verification)")
        print()
        print("Sentinels:")
        for trace_name, status in traces_data.items():
            symbol = (
                "[PASS]" if status == "VERIFIED" else "[WARN]" if "STALE" in status else "[FAIL]"
            )
            print(f"  {symbol} {trace_name}: {status}")

    return EXIT_SUCCESS


def cmd_init(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Scaffold a new trace from a source file."""
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = sentinel.repo_root / source_path

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}")
        return EXIT_GENERAL_ERROR

    # Determine trace name
    trace_name = args.name or source_path.stem

    # Check if trace already exists
    trace_path = sentinel.traces_dir / f"{trace_name}.md"
    if trace_path.exists() and not args.force:
        print(f"Error: Trace already exists: {trace_path}")
        print("Use --force to overwrite")
        return EXIT_GENERAL_ERROR

    # Extract functions and classes
    print(f"Analyzing {source_path}...")
    functions = sentinel.extract_functions(source_path)
    classes = sentinel.extract_classes(source_path)

    # Categorize by importance
    critical = [f for f in functions if f.importance == "critical"]
    high = [f for f in functions if f.importance == "high"]

    print(f"  Found {len(functions)} functions, {len(classes)} classes")
    print(f"  Critical: {len(critical)}, High: {len(high)}")

    # Compute relative path for the source file
    try:
        rel_source = source_path.relative_to(sentinel.repo_root)
    except ValueError:
        rel_source = source_path

    # Generate trace markdown
    trace_content = f"""# Sentinel: {trace_name}

**Verified against:** `{rel_source}` @ commit `TODO`

**Status:** DRAFT

**Linked tests:** TODO

## Summary

TODO: Describe the purpose and key functionality of this module.

## Active Assumptions

### Mechanically Verified

| ID | Assumption | Verification |
|----|------------|--------------|
"""

    # Add suggested assumptions for critical/high functions
    assumption_id = 1
    for func in critical[:5]:  # Limit to 5 critical functions
        trace_content += f"| A{assumption_id} | {func.name} exists at expected location | anchor: {func.name.upper()} |\n"
        assumption_id += 1

    trace_content += """
### Agent-Verified (on trace load)

| ID | Assumption | Verification Guidance |
|----|------------|----------------------|
| TODO | TODO | TODO |

## Algorithm Flow

TODO: Document the key algorithm steps with line references.

"""

    # Add key functions section
    if critical or high:
        trace_content += "## Key Functions\n\n"
        trace_content += "| Function | Line | Importance |\n"
        trace_content += "|----------|------|------------|\n"
        for func in critical + high:
            trace_content += f"| `{func.name}` | {func.line} | {func.importance} |\n"
        trace_content += "\n"

    trace_content += """## Critical Invariants

- [ ] TODO: Document critical invariants

## Known Issues

| Issue | Severity | Resolution |
|-------|----------|------------|
| None documented | - | - |

## Version History

- **TODO**: Initial trace (DRAFT)
"""

    # Generate anchor suggestions
    anchors_content = f"\n# Anchors for {trace_name} (generated scaffold)\n{trace_name}:\n"
    for func in (critical + high)[:10]:  # Limit to 10 anchors
        anchor_name = func.name.upper()
        anchors_content += f"""  {anchor_name}:
    file: {rel_source}
    pattern: "def {func.name}("
    expected_line: {func.line}
    drift_tolerance: 30
"""

    # Generate meta entry suggestion
    meta_entry = f"""
# Add to .sentinel-meta.yaml under sentinels:
  {trace_name}:
    verified_commit: TODO
    source_files:
      - {rel_source}
    assumptions_mechanical: []
    assumptions_agent: []
    anchors: [{', '.join(f.name.upper() for f in (critical + high)[:10])}]
    linked_tests: []
    status: DRAFT
    depends_on: []
"""

    # Write trace file
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(trace_content)
    print(f"\nCreated trace: {trace_path}")

    # Show anchor suggestions
    print("\n=== Suggested Anchors ===")
    print("Add to anchors/anchors.yaml:")
    print(anchors_content)

    # Show meta entry suggestion
    print("\n=== Suggested Meta Entry ===")
    print(meta_entry)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Edit the trace file to complete TODO sections")
    print("  2. Add anchors to anchors/anchors.yaml")
    print("  3. Add entry to .sentinel-meta.yaml")
    print("  4. Update verified_commit and status when ready")
    print("  5. Run: ./sentinel.py verify --trace " + trace_name)
    print("=" * 60)

    return EXIT_SUCCESS


def cmd_verify(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Verify specific trace or all traces."""
    meta = sentinel.load_meta()
    exit_code = EXIT_SUCCESS
    use_json = getattr(args, "format", None) == "json"
    strict, strict_err = _resolve_strict(args, sentinel)
    if strict_err:
        if use_json:
            print(
                _json_envelope(
                    "verify",
                    exit_code=EXIT_GENERAL_ERROR,
                    strict_mode=strict,
                    traces={},
                    consistency_errors=[],
                    error=strict_err,
                )
            )
        else:
            print(f"Error: {strict_err}", file=sys.stderr)
        return EXIT_GENERAL_ERROR

    traces_to_verify = []
    if args.all:
        traces_to_verify = list(meta.get("sentinels", {}).keys())
    elif args.trace:
        traces_to_verify = [args.trace]
    else:
        msg = "Specify --trace NAME or --all"
        if use_json:
            print(
                _json_envelope(
                    "verify",
                    exit_code=EXIT_GENERAL_ERROR,
                    strict_mode=strict,
                    traces={},
                    consistency_errors=[],
                    error=msg,
                )
            )
        else:
            print(f"Error: {msg}")
        return EXIT_GENERAL_ERROR

    consistency_errors = []
    # Consistency check if requested
    if args.check_consistency:
        consistency_errors = sentinel.check_consistency()
        if not use_json:
            print("=== Consistency Check ===\n")
            if consistency_errors:
                for err in consistency_errors:
                    print(f"  [FAIL] {err}")
                print(f"\nConsistency: {len(consistency_errors)} errors found\n")
                exit_code = EXIT_CONSISTENCY_FAILED
            else:
                print("  [PASS] All metadata consistent\n")
        elif consistency_errors:
            exit_code = EXIT_CONSISTENCY_FAILED

    # Verify each trace
    json_traces = {}
    for trace_name in traces_to_verify:
        result = sentinel.verify_trace(trace_name)

        if use_json:
            json_traces[trace_name] = {
                "status": result.overall_status.value,
                "commit_status": result.commit_status.value,
                "verified_commit": result.verified_commit,
                "current_commit": result.current_commit,
                "uncommitted_changes": result.uncommitted_changes,
                "anchors": {
                    "verified": sum(1 for a in result.anchors if a.status == "VERIFIED"),
                    "total": len(result.anchors),
                    "details": [
                        {
                            "name": a.name,
                            "status": a.status,
                            "expected": a.expected_line,
                            "actual": a.actual_line,
                        }
                        for a in result.anchors
                    ],
                },
                "assumptions": {
                    "passed": sum(1 for a in result.assumptions if a.passed),
                    "total": len(result.assumptions),
                    "details": [
                        {"id": a.id, "passed": a.passed, "message": a.message}
                        for a in result.assumptions
                    ],
                },
            }
        else:
            _print_verification_result(result)

        if result.overall_status != Status.VERIFIED:
            if result.commit_status == Status.MISSING:
                exit_code = max(exit_code, EXIT_GENERAL_ERROR)
            if result.commit_status in (Status.STALE_COMMIT, Status.STALE_CONTENT):
                if strict:
                    exit_code = max(exit_code, EXIT_ANCHOR_DRIFT)
                # advisory mode: stale is warning only, no exit code change
            if any(a.status == "MISSING" for a in result.anchors):
                exit_code = max(exit_code, EXIT_ANCHOR_MISSING)
            elif any(a.status == "DRIFT" for a in result.anchors):
                exit_code = max(exit_code, EXIT_ANCHOR_DRIFT)
            elif any(a.status == "AMBIGUOUS" for a in result.anchors):
                exit_code = max(exit_code, EXIT_ANCHOR_AMBIGUOUS)
            elif any(not a.passed for a in result.assumptions):
                exit_code = max(exit_code, EXIT_ASSUMPTION_FAILED)

    if use_json:
        print(
            _json_envelope(
                "verify",
                exit_code=exit_code,
                strict_mode=strict,
                traces=json_traces,
                consistency_errors=consistency_errors,
            )
        )

    return exit_code


def _print_verification_result(result: TraceVerification) -> None:
    """Print formatted verification result."""
    if result.overall_status == Status.VERIFIED:
        print(f"Grounded: {result.name} @ {result.verified_commit} [PASS]")
    else:
        print(f"Grounded: {result.name} @ {result.verified_commit}")

        if result.commit_status == Status.MISSING:
            print("  Trace: MISSING (not found in .sentinel-meta.yaml)")

        elif result.commit_status in (Status.STALE_COMMIT, Status.STALE_CONTENT):
            print(f"  Commit: {result.commit_status.value}")
            print(
                f"    Verified: {result.verified_commit} | Current: {result.current_commit or 'unknown'}"
            )

        # Anchors summary
        verified = sum(1 for a in result.anchors if a.status == "VERIFIED")
        total = len(result.anchors)
        drifted = [a for a in result.anchors if a.status != "VERIFIED"]
        if drifted:
            print(f"  Anchors: {verified}/{total} verified, {len(drifted)} failed")
            for a in drifted:
                print(f"    [FAIL] {a.name}: {a.message}")

        # Assumptions summary
        if result.assumptions:
            failed = [a for a in result.assumptions if not a.passed]
            summary = ", ".join(
                f"{a.id} {'[PASS]' if a.passed else '[FAIL]'}" for a in result.assumptions
            )
            print(f"  Assumptions: {summary}")
            if failed:
                for a in failed:
                    print(f"    [FAIL] {a.id}: {a.message}")

        print()
        print("[WARN] Cannot provide advice until sentinel is updated.")
    print()


def cmd_pipeline(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Run full pre-commit pipeline."""
    strict, strict_err = _resolve_strict(args, sentinel)
    if strict_err:
        print(f"Error: {strict_err}", file=sys.stderr)
        return EXIT_GENERAL_ERROR

    print("=" * 60)
    print(f"Code Sentinel Pipeline{' (strict)' if strict else ''}")
    print("=" * 60)
    print()

    exit_code = EXIT_SUCCESS
    meta = sentinel.load_meta()

    # Step 1: Get modified files
    if args.files:
        modified_files = args.files
    else:
        modified_files = sentinel.get_modified_files()

    if modified_files:
        print("Modified files:")
        for f in modified_files:
            print(f"  {f}")
        print()

    # Step 2: Consistency check
    print("=== Step 1: Consistency Check ===\n")
    errors = sentinel.check_consistency()
    if errors:
        for err in errors:
            print(f"  [FAIL] {err}")
        exit_code = EXIT_CONSISTENCY_FAILED
    else:
        print("  [PASS] All metadata consistent")
    print()

    # Step 3: Verify all traces
    print("=== Step 2: Trace Verification ===\n")
    for trace_name in meta.get("sentinels", {}).keys():
        result = sentinel.verify_trace(trace_name)

        status_symbol = "[PASS]" if result.overall_status == Status.VERIFIED else "[FAIL]"
        anchor_status = (
            f"{sum(1 for a in result.anchors if a.status == 'VERIFIED')}/{len(result.anchors)}"
        )
        assumption_status = (
            f"{sum(1 for a in result.assumptions if a.passed)}/{len(result.assumptions)}"
            if result.assumptions
            else "n/a"
        )

        print(f"  {status_symbol} {trace_name}")
        print(f"      Commit: {result.commit_status.value}")
        print(f"      Anchors: {anchor_status}")
        print(f"      Assumptions: {assumption_status}")

        if result.overall_status != Status.VERIFIED:
            if result.commit_status == Status.MISSING:
                exit_code = max(exit_code, EXIT_GENERAL_ERROR)
            if result.commit_status in (Status.STALE_COMMIT, Status.STALE_CONTENT) and strict:
                exit_code = max(exit_code, EXIT_ANCHOR_DRIFT)
            if any(a.status != "VERIFIED" for a in result.anchors):
                exit_code = max(exit_code, EXIT_ANCHOR_DRIFT)
            if any(not a.passed for a in result.assumptions):
                exit_code = max(exit_code, EXIT_ASSUMPTION_FAILED)
    print()

    # Step 4: Test advisory
    print("=== Step 3: Test Advisory ===\n")
    if modified_files:
        suggested_tests = sentinel.get_suggested_tests(modified_files)
        if suggested_tests:
            print("  Suggested tests for modified files:")
            print(f"    pytest {' '.join(suggested_tests)}")
            print()
            if os.environ.get("SENTINEL_RUN_TESTS") == "1":
                print("  Running tests...")
                test_result = subprocess.run(["pytest"] + suggested_tests, cwd=sentinel.repo_root)
                if test_result.returncode != 0:
                    exit_code = max(exit_code, EXIT_GENERAL_ERROR)
            else:
                print("  Run with: SENTINEL_RUN_TESTS=1 ./sentinel.py pipeline")
        else:
            print("  No sentinel-bound tests for modified files")
    else:
        print("  No staged files to check")
    print()

    # Summary
    print("=" * 60)
    if exit_code == EXIT_SUCCESS:
        print("Pipeline: PASSED [PASS]")
    else:
        print(f"Pipeline: FAILED (exit code {exit_code})")
    print("=" * 60)

    return exit_code


def cmd_retrace(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Interactive sentinel update."""
    trace_name = args.trace_name
    trace_path = sentinel.traces_dir / f"{trace_name}.md"

    if not trace_path.exists():
        print(f"Error: Trace not found: {trace_path}")
        return EXIT_GENERAL_ERROR

    # Get current state
    result = sentinel.verify_trace(trace_name)
    _print_verification_result(result)

    if args.diff_only:
        # Show git diff for source files
        meta = sentinel.load_meta()
        trace_meta = meta.get("sentinels", {}).get(trace_name, {})
        verified_commit = trace_meta.get("verified_commit", "")

        for source_file in trace_meta.get("source_files", []):
            print(f"\n=== Diff: {source_file} ===")
            print(f"From {verified_commit} to HEAD:\n")
            subprocess.run(
                ["git", "diff", f"{verified_commit}..HEAD", "--", source_file],
                cwd=sentinel.repo_root,
            )
        return EXIT_SUCCESS

    if args.auto:
        # Auto-detect anchor changes
        print("\n=== Auto-Retrace Analysis ===\n")
        impacts = sentinel.analyze_anchor_impacts(trace_name)

        if not impacts:
            print("No anchors found for this trace")
            return EXIT_SUCCESS

        # Group by status
        unchanged = [i for i in impacts if i.status == "unchanged"]
        shifted = [i for i in impacts if i.status == "shifted"]
        modified = [i for i in impacts if i.status == "modified"]
        deleted = [i for i in impacts if i.status == "deleted"]

        print("Anchor Impact Summary:")
        print(f"  Unchanged: {len(unchanged)}")
        print(f"  Shifted (auto-fixable): {len(shifted)}")
        print(f"  Modified (needs review): {len(modified)}")
        print(f"  Deleted (manual fix): {len(deleted)}")
        print()

        if shifted:
            print("Shifted anchors (safe to auto-update):")
            for impact in shifted:
                print(f"  {impact.anchor_name}: {impact.old_line} -> {impact.new_line}")

        if modified:
            print("\nModified anchors (manual review required):")
            for impact in modified:
                print(f"  [FAIL] {impact.anchor_name}: {impact.suggestion}")

        if deleted:
            print("\nDeleted anchors (pattern not found):")
            for impact in deleted:
                print(f"  [FAIL] {impact.anchor_name}: {impact.suggestion}")

        # Apply if requested
        if args.apply and shifted:
            print("\n=== Applying Safe Updates ===\n")
            anchors = sentinel.load_anchors()
            trace_anchors = anchors.get(trace_name, {})
            updated = 0

            for impact in shifted:
                if impact.anchor_name in trace_anchors and impact.new_line:
                    trace_anchors[impact.anchor_name]["expected_line"] = impact.new_line
                    print(f"  Updated {impact.anchor_name}: {impact.old_line} -> {impact.new_line}")
                    updated += 1

            if updated:
                sentinel.save_anchors(anchors)
                print(f"\nSaved {updated} anchor update(s) to anchors.yaml")
            else:
                print("\nNo updates applied")
        elif args.apply and not shifted:
            print("\nNo safe updates to apply")
        elif shifted:
            print(f"\nRun with --apply to update {len(shifted)} shifted anchor(s)")

        if modified or deleted:
            print("\n[WARN] Some anchors require manual attention")
            return EXIT_ANCHOR_DRIFT

        return EXIT_SUCCESS

    if args.anchors_only:
        # Auto-update anchor line numbers
        print("\nAuto-updating anchor line numbers...")
        anchors = sentinel.load_anchors()
        trace_anchors = anchors.get(trace_name, {})
        updated = 0

        for anchor_name, spec in trace_anchors.items():
            file_path = str(sentinel.repo_root / spec["file"])
            pattern = spec["pattern"]
            after = spec.get("after")

            # Find current line
            if after:
                cmd = f'grep -n "{after}" "{file_path}" | head -1 | cut -d: -f1'
                after_result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, cwd=sentinel.repo_root
                )
                if after_result.stdout.strip():
                    after_line = int(after_result.stdout.strip())
                    cmd = f'tail -n "+{after_line}" "{file_path}" | grep -n "{pattern}" | head -1 | cut -d: -f1'
                    grep_result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, cwd=sentinel.repo_root
                    )
                    if grep_result.stdout.strip():
                        actual_line = after_line + int(grep_result.stdout.strip()) - 1
                    else:
                        actual_line = None
                else:
                    actual_line = None
            else:
                cmd = f'grep -n "{pattern}" "{file_path}" | head -1 | cut -d: -f1'
                grep_result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, cwd=sentinel.repo_root
                )
                actual_line = (
                    int(grep_result.stdout.strip()) if grep_result.stdout.strip() else None
                )

            if actual_line and actual_line != spec["expected_line"]:
                print(f"  {anchor_name}: {spec['expected_line']} -> {actual_line}")
                spec["expected_line"] = actual_line
                updated += 1

        if updated:
            # Save updated anchors
            sentinel.save_anchors(anchors)
            print(f"\nUpdated {updated} anchor(s) in anchors.yaml")
        else:
            print("\nNo anchor updates needed")

        return EXIT_SUCCESS

    # Interactive mode
    print("\nInteractive retrace not yet implemented.")
    print("Please manually update:")
    print(f"  1. {trace_path}")
    print(f"  2. {sentinel.anchors_path}")
    print(f"  3. {sentinel.meta_path}")
    print("\nOr use:")
    print("  --auto         Analyze anchor impacts")
    print("  --auto --apply Apply safe updates")
    print("  --anchors-only Force update all anchor line numbers")

    return EXIT_SUCCESS


def cmd_install_hooks(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Install git pre-commit hooks via pre-commit framework."""
    # Check if pre-commit is installed
    result = subprocess.run(["which", "pre-commit"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: pre-commit not found. Install with: pip install pre-commit")
        return EXIT_GENERAL_ERROR

    # Check if .pre-commit-config.yaml exists
    config_path = sentinel.repo_root / ".pre-commit-config.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        return EXIT_GENERAL_ERROR

    # Run pre-commit install
    result = subprocess.run(["pre-commit", "install"], cwd=sentinel.repo_root)
    if result.returncode == 0:
        print("Pre-commit hooks installed successfully")
        return EXIT_SUCCESS
    else:
        print("Failed to install pre-commit hooks")
        return EXIT_GENERAL_ERROR


def _normalize_text_for_compare(content: str) -> str:
    """Normalize text for stable comparisons across platforms."""
    return content.replace("\r\n", "\n").rstrip("\n")


def _resolve_path_within(
    base_dir: Path, configured_path: str, field_name: str
) -> tuple[Path | None, str | None]:
    """Resolve a configured path and ensure it stays under base_dir."""
    if not configured_path:
        return None, f"{field_name} is empty"

    raw = Path(configured_path)
    if raw.is_absolute():
        return None, f"{field_name} must be relative, got absolute path: {configured_path}"

    resolved = (base_dir / raw).resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError:
        return None, f"{field_name} escapes base directory: {configured_path}"

    return resolved, None


def _managed_block_markers(spec: dict, adapter_name: str) -> tuple[str, str]:
    """Get managed block markers for an adapter."""
    start = spec.get("block_start", f"<!-- sentinel:{adapter_name}:start -->")
    end = spec.get("block_end", f"<!-- sentinel:{adapter_name}:end -->")
    return start, end


def _extract_managed_block_content(content: str, block_start: str, block_end: str) -> str | None:
    """Extract managed block body, or None if markers are missing."""
    start_idx = content.find(block_start)
    if start_idx < 0:
        return None
    end_idx = content.find(block_end, start_idx + len(block_start))
    if end_idx < 0:
        return None
    return content[start_idx + len(block_start) : end_idx].strip("\r\n")


def _render_managed_block(template_content: str, block_start: str, block_end: str) -> str:
    """Render a managed block wrapper around template content."""
    return f"{block_start}\n{template_content.rstrip()}\n{block_end}\n"


def _prepare_adapter_install(
    sentinel: CodeSentinel, adapter_name: str, spec: dict
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate adapter config and resolve safe paths."""
    template_rel = spec.get("template", "")
    target_rel = spec.get("target", "")
    mode = spec.get("mode", "copy")

    if mode not in {"copy", "managed_block"}:
        return None, f"adapter '{adapter_name}' has unsupported mode: {mode}"

    template_path, template_err = _resolve_path_within(
        sentinel.sentinel_dir, template_rel, f"adapter '{adapter_name}' template"
    )
    if template_err:
        return None, template_err

    target_path, target_err = _resolve_path_within(
        sentinel.repo_root, target_rel, f"adapter '{adapter_name}' target"
    )
    if target_err:
        return None, target_err

    block_start, block_end = _managed_block_markers(spec, adapter_name)
    return {
        "name": adapter_name,
        "mode": mode,
        "template_rel": template_rel,
        "target_rel": target_rel,
        "template_path": template_path,
        "target_path": target_path,
        "block_start": block_start,
        "block_end": block_end,
    }, None


def _adapter_install_status(adapter_cfg: dict[str, Any]) -> tuple[str, str]:
    """Return adapter status: up_to_date, drifted, missing, invalid."""
    template_path: Path = adapter_cfg["template_path"]
    target_path: Path = adapter_cfg["target_path"]

    if not template_path.exists():
        return "invalid", f"template missing: {adapter_cfg['template_rel']}"
    if not template_path.is_file():
        return "invalid", f"template is not a file: {adapter_cfg['template_rel']}"

    template_content = template_path.read_text()
    if not target_path.exists():
        return "missing", "target file does not exist"

    target_content = target_path.read_text()
    mode = adapter_cfg["mode"]
    if mode == "copy":
        if _normalize_text_for_compare(target_content) == _normalize_text_for_compare(
            template_content
        ):
            return "up_to_date", "target matches template"
        return "drifted", "target differs from template"

    managed_content = _extract_managed_block_content(
        target_content, adapter_cfg["block_start"], adapter_cfg["block_end"]
    )
    if managed_content is None:
        return "drifted", "managed block markers not found in target"
    if _normalize_text_for_compare(managed_content) == _normalize_text_for_compare(
        template_content
    ):
        return "up_to_date", "managed block matches template"
    return "drifted", "managed block differs from template"


def cmd_install_adapter(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Install an agent adapter from template."""
    adapters = sentinel.config.get("adapters", {})
    dry_run = getattr(args, "dry_run", False)

    if args.list:
        if not adapters:
            print("No adapters configured in sentinel.yaml")
            return EXIT_SUCCESS
        print("Available adapters:")
        status_labels = {
            "up_to_date": "[up-to-date]",
            "drifted": "[drifted]",
            "missing": "[missing]",
            "invalid": "[invalid]",
        }
        for name, spec in adapters.items():
            target = spec.get("target", "?")
            desc = spec.get("description", "")
            adapter_cfg, prep_err = _prepare_adapter_install(sentinel, name, spec)
            if prep_err:
                status = status_labels["invalid"]
                detail = prep_err
            else:
                status_key, detail = _adapter_install_status(adapter_cfg)
                status = status_labels[status_key]
            print(f"  {name:<12} {status}  -> {target}")
            if desc:
                print(f"               {desc}")
            if detail:
                print(f"               {detail}")
        return EXIT_SUCCESS

    adapter_name = args.adapter
    if not adapter_name:
        print("Error: specify an adapter name or --list")
        return EXIT_GENERAL_ERROR

    if adapter_name not in adapters:
        print(f"Error: unknown adapter '{adapter_name}'")
        print(f"Available: {', '.join(adapters.keys())}")
        return EXIT_GENERAL_ERROR

    adapter_cfg, prep_err = _prepare_adapter_install(sentinel, adapter_name, adapters[adapter_name])
    if prep_err:
        print(f"Error: {prep_err}")
        return EXIT_GENERAL_ERROR

    status_key, status_detail = _adapter_install_status(adapter_cfg)
    if status_key == "invalid":
        print(f"Error: {status_detail}")
        return EXIT_GENERAL_ERROR

    template_path: Path = adapter_cfg["template_path"]
    target_path: Path = adapter_cfg["target_path"]
    template_rel = adapter_cfg["template_rel"]
    target_rel = adapter_cfg["target_rel"]
    content = template_path.read_text()
    mode = adapter_cfg["mode"]

    if mode == "copy":
        # Check if target already exists and differs
        if target_path.exists() and not args.force:
            existing = target_path.read_text()
            if _normalize_text_for_compare(existing) == _normalize_text_for_compare(content):
                print(f"Already up to date: {target_rel}")
                return EXIT_SUCCESS
            print(f"Target exists and differs: {target_rel}")
            print("Use --force to overwrite")
            return EXIT_GENERAL_ERROR

        # Install
        action = "overwrite existing target" if target_path.exists() else "create target"
        if dry_run:
            print(f"Dry run: would install {adapter_name} adapter: {target_rel}")
            print(f"  Source: {template_rel}")
            print(f"  Mode: {mode} ({action})")
            return EXIT_SUCCESS

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content)
        print(f"Installed {adapter_name} adapter: {target_rel}")
        print(f"  Source: {template_rel}")
        return EXIT_SUCCESS

    # managed_block mode: only update sentinel-managed block content.
    block_start = adapter_cfg["block_start"]
    block_end = adapter_cfg["block_end"]
    managed_block = _render_managed_block(content, block_start, block_end)
    target_existed = target_path.exists()

    if target_existed:
        existing = target_path.read_text()
        managed_content = _extract_managed_block_content(existing, block_start, block_end)

        if managed_content is not None:
            if _normalize_text_for_compare(managed_content) == _normalize_text_for_compare(content):
                print(f"Already up to date: {target_rel}")
                return EXIT_SUCCESS

            start_idx = existing.find(block_start)
            end_idx = existing.find(block_end, start_idx + len(block_start))
            suffix_idx = end_idx + len(block_end)
            if existing.startswith("\r\n", suffix_idx):
                suffix_idx += 2
            elif existing.startswith("\n", suffix_idx):
                suffix_idx += 1
            new_content = existing[:start_idx] + managed_block + existing[suffix_idx:]
            action = "updated managed block"
        elif _normalize_text_for_compare(existing) == _normalize_text_for_compare(content):
            # Legacy full-file adapter: migrate in-place to managed block.
            new_content = managed_block
            action = "migrated legacy adapter file to managed block"
        else:
            # Preserve non-sentinel content and append sentinel-owned block.
            prefix = existing.rstrip("\n")
            if prefix:
                new_content = f"{prefix}\n\n{managed_block}"
            else:
                new_content = managed_block
            action = "appended managed block to existing file"
            print("Warning: target has no managed block markers; appending sentinel block.")
    else:
        new_content = managed_block
        action = "created managed block target"

    if dry_run:
        print(f"Dry run: would install {adapter_name} adapter: {target_rel}")
        print(f"  Source: {template_rel}")
        print(f"  Mode: {mode} ({action})")
        if target_existed and action in {
            "updated managed block",
            "appended managed block to existing file",
        }:
            print("  Note: only sentinel-managed block content would be updated.")
        return EXIT_SUCCESS

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(new_content)
    print(f"Installed {adapter_name} adapter: {target_rel}")
    print(f"  Source: {template_rel}")
    print(f"  Mode: {mode} ({action})")
    if (
        not args.force
        and target_existed
        and action
        in {
            "updated managed block",
            "appended managed block to existing file",
        }
    ):
        print("  Note: only sentinel-managed block content was updated.")

    return EXIT_SUCCESS


def cmd_report(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Generate verification report."""
    meta = sentinel.load_meta()

    last_verification = meta.get("last_global_verification")
    if hasattr(last_verification, "isoformat"):
        last_verification = last_verification.isoformat()

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "version": meta.get("version", "unknown"),
        "last_global_verification": str(last_verification) if last_verification else None,
        "traces": {},
        "consistency_errors": sentinel.check_consistency(),
    }

    for trace_name in meta.get("sentinels", {}).keys():
        result = sentinel.verify_trace(trace_name)
        report["traces"][trace_name] = {
            "status": result.overall_status.value,
            "commit_status": result.commit_status.value,
            "verified_commit": result.verified_commit,
            "current_commit": result.current_commit,
            "uncommitted_changes": result.uncommitted_changes,
            "anchors": {
                "verified": sum(1 for a in result.anchors if a.status == "VERIFIED"),
                "total": len(result.anchors),
                "details": [
                    {
                        "name": a.name,
                        "status": a.status,
                        "expected": a.expected_line,
                        "actual": a.actual_line,
                    }
                    for a in result.anchors
                ],
            },
            "assumptions": {
                "passed": sum(1 for a in result.assumptions if a.passed),
                "total": len(result.assumptions),
                "details": [
                    {"id": a.id, "passed": a.passed, "message": a.message}
                    for a in result.assumptions
                ],
            },
        }

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:  # markdown
        print("# Code Sentinel Report")
        print()
        print(f"Generated: {report['timestamp']}")
        print()
        print("## Summary")
        print()
        all_verified = all(t["status"] == "VERIFIED" for t in report["traces"].values())
        print(f"Overall Status: {'[PASS] VERIFIED' if all_verified else '[FAIL] DEGRADED'}")
        print()
        print("## Traces")
        print()
        print("| Trace | Status | Commit | Anchors | Assumptions |")
        print("|-------|--------|--------|---------|-------------|")
        for name, data in report["traces"].items():
            status = "[PASS]" if data["status"] == "VERIFIED" else "[FAIL]"
            anchors = f"{data['anchors']['verified']}/{data['anchors']['total']}"
            assumptions = f"{data['assumptions']['passed']}/{data['assumptions']['total']}"
            print(
                f"| {name} | {status} {data['status']} | {data['commit_status']} | {anchors} | {assumptions} |"
            )

        if report["consistency_errors"]:
            print()
            print("## Consistency Errors")
            print()
            for err in report["consistency_errors"]:
                print(f"- {err}")

    return EXIT_SUCCESS


def cmd_graph(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Generate dependency graph visualization."""
    meta = sentinel.load_meta()
    sentinels = meta.get("sentinels", {})

    if not sentinels:
        print("No sentinels found")
        return EXIT_SUCCESS

    # Collect edges
    edges: list[tuple[str, str]] = []
    nodes: set[str] = set()

    if args.trace:
        # Single trace + its dependencies
        if args.trace not in sentinels:
            print(f"Error: Trace '{args.trace}' not found")
            return EXIT_GENERAL_ERROR
        nodes.add(args.trace)
        deps = sentinels[args.trace].get("depends_on", [])
        for dep in deps:
            edges.append((args.trace, dep))
            nodes.add(dep)
    else:
        # All traces
        for trace_name, trace_meta in sentinels.items():
            nodes.add(trace_name)
            for dep in trace_meta.get("depends_on", []):
                edges.append((trace_name, dep))
                nodes.add(dep)

    # Generate output
    if args.format == "dot":
        output = _generate_dot_graph(nodes, edges)
    else:  # mermaid (default)
        output = _generate_mermaid_graph(nodes, edges)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Graph written to {args.output}")
    else:
        print(output)

    return EXIT_SUCCESS


def _generate_mermaid_graph(nodes: set[str], edges: list[tuple[str, str]]) -> str:
    """Generate Mermaid diagram."""
    lines = ["```mermaid", "graph TD"]

    # Add edges
    if edges:
        for src, dst in sorted(edges):
            # Sanitize names for Mermaid (replace hyphens with underscores in IDs)
            src_id = src.replace("-", "_")
            dst_id = dst.replace("-", "_")
            lines.append(f"    {src_id}[{src}] --> {dst_id}[{dst}]")
    else:
        # No dependencies, just list nodes
        for node in sorted(nodes):
            node_id = node.replace("-", "_")
            lines.append(f"    {node_id}[{node}]")

    lines.append("```")
    return "\n".join(lines)


def _generate_dot_graph(nodes: set[str], edges: list[tuple[str, str]]) -> str:
    """Generate DOT format for Graphviz."""
    lines = ["digraph sentinels {", "    rankdir=TB;", "    node [shape=box];", ""]

    # Add nodes
    for node in sorted(nodes):
        node_id = node.replace("-", "_")
        lines.append(f'    {node_id} [label="{node}"];')

    lines.append("")

    # Add edges
    for src, dst in sorted(edges):
        src_id = src.replace("-", "_")
        dst_id = dst.replace("-", "_")
        lines.append(f"    {src_id} -> {dst_id};")

    lines.append("}")
    return "\n".join(lines)


def cmd_coverage(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Generate trace coverage metrics."""
    anchors = sentinel.load_anchors()
    src_root = sentinel.config.get("project", {}).get("src_root", "src")
    src_dir = Path(src_root)
    if not src_dir.is_absolute():
        src_dir = sentinel.repo_root / src_dir

    if not src_dir.exists():
        print(f"Error: source root not found at {src_dir} (project.src_root={src_root!r})")
        return EXIT_GENERAL_ERROR

    # Build map of file -> anchored lines
    anchored_files: dict[str, set[int]] = {}
    for _trace_name, trace_anchors in anchors.items():
        for _anchor_name, spec in trace_anchors.items():
            file_path = spec["file"]
            line = spec["expected_line"]
            if file_path not in anchored_files:
                anchored_files[file_path] = set()
            anchored_files[file_path].add(line)

    # Analyze each Python file
    coverage_data: dict[str, dict[str, Any]] = {}
    total_functions = 0
    total_anchored = 0
    total_weighted = 0.0
    total_weighted_anchored = 0.0

    importance_weights = {"critical": 3.0, "high": 2.0, "medium": 1.0}

    for py_file in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        try:
            rel_path = str(py_file.relative_to(sentinel.repo_root))
        except ValueError:
            rel_path = str(py_file)

        functions = sentinel.extract_functions(py_file)
        if not functions:
            continue

        anchored_lines = anchored_files.get(rel_path, set())

        # Count anchored functions (within tolerance of anchor line)
        anchored_count = 0
        file_weighted = 0.0
        file_weighted_anchored = 0.0

        for func in functions:
            weight = importance_weights.get(func.importance, 1.0)
            file_weighted += weight
            total_weighted += weight

            # Check if any anchor is near this function (within 5 lines)
            is_anchored = any(abs(func.line - anchor_line) <= 5 for anchor_line in anchored_lines)
            if is_anchored:
                anchored_count += 1
                file_weighted_anchored += weight
                total_weighted_anchored += weight

        total_functions += len(functions)
        total_anchored += anchored_count

        coverage_data[rel_path] = {
            "total": len(functions),
            "anchored": anchored_count,
            "percentage": (anchored_count / len(functions) * 100) if functions else 0,
            "weighted_total": file_weighted,
            "weighted_anchored": file_weighted_anchored,
        }

    # Calculate overall metrics
    overall_pct = (total_anchored / total_functions * 100) if total_functions else 0
    weighted_pct = (total_weighted_anchored / total_weighted * 100) if total_weighted else 0

    # Output
    if args.format == "json":
        report = {
            "files": coverage_data,
            "summary": {
                "total_functions": total_functions,
                "anchored_functions": total_anchored,
                "coverage_percent": round(overall_pct, 1),
                "weighted_coverage_percent": round(weighted_pct, 1),
            },
        }
        print(json.dumps(report, indent=2))
    else:
        # Text format with progress bars
        print("Code Sentinel Coverage Report")
        print("=" * 60)
        print()

        # Group by directory
        by_dir: dict[str, list[str]] = {}
        for file_path in coverage_data:
            dir_path = str(Path(file_path).parent)
            if dir_path not in by_dir:
                by_dir[dir_path] = []
            by_dir[dir_path].append(file_path)

        for dir_path in sorted(by_dir.keys()):
            print(f"{dir_path}/")
            for file_path in sorted(by_dir[dir_path]):
                data = coverage_data[file_path]
                filename = Path(file_path).name
                pct = data["percentage"]
                bar = _progress_bar(pct)
                print(f"  {filename:<30} {bar} {pct:5.1f}% ({data['anchored']}/{data['total']})")
            print()

        print("=" * 60)
        print(
            f"Overall: {overall_pct:.1f}% coverage ({total_anchored}/{total_functions} functions)"
        )
        print(f"Weighted: {weighted_pct:.1f}% (critical=3x, high=2x, medium=1x)")

    # Check threshold
    if args.threshold:
        if overall_pct < args.threshold:
            print(f"\n[FAIL] Coverage {overall_pct:.1f}% below threshold {args.threshold}%")
            return EXIT_GENERAL_ERROR
        else:
            print(f"\n[PASS] Coverage {overall_pct:.1f}% meets threshold {args.threshold}%")

    return EXIT_SUCCESS


def _progress_bar(percentage: float, width: int = 10) -> str:
    """Generate a text progress bar."""
    filled = int(percentage / 100 * width)
    empty = width - filled
    return "#" * filled + "-" * empty


def cmd_sync_docs(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Regenerate markdown assumption tables from assumptions.yaml."""
    assumptions_path = sentinel.sentinel_dir / "anchors" / "assumptions.yaml"
    if not assumptions_path.exists():
        print(f"Error: assumptions.yaml not found at {assumptions_path}")
        return EXIT_GENERAL_ERROR

    assumptions_data = yaml.safe_load(assumptions_path.read_text())
    if not assumptions_data or "assumptions" not in assumptions_data:
        print("Error: Invalid assumptions.yaml format")
        return EXIT_GENERAL_ERROR

    updated_count = 0

    for trace_name, assumptions in assumptions_data["assumptions"].items():
        trace_path = sentinel.traces_dir / f"{trace_name}.md"
        if not trace_path.exists():
            print(f"  Skip {trace_name}: trace file not found")
            continue

        content = trace_path.read_text()

        # Generate new tables
        mechanical = [a for a in assumptions if a.get("mechanical", True)]
        manual = [a for a in assumptions if not a.get("mechanical", True)]

        # Build mechanical table
        mech_table = "| ID | Assumption | Verification |\n"
        mech_table += "|----|------------|--------------|"
        for a in mechanical:
            v = a["verification"]
            if v["type"] == "anchor":
                verification_str = f"anchor: {v['ref']}"
            elif v["type"] == "shell":
                verification_str = f"`{v['command']}`"
            elif v["type"] == "pattern":
                verification_str = f"pattern: {v['regex'][:30]}..."
            else:
                verification_str = v.get("guidance", "manual")[:50]
            mech_table += f"\n| {a['id']} | {a['description']} | {verification_str} |"

        # Build manual table
        manual_table = "| ID | Assumption | Verification Guidance |\n"
        manual_table += "|----|------------|----------------------|"
        for a in manual:
            guidance = a["verification"].get("guidance", "No guidance")
            manual_table += f"\n| {a['id']} | {a['description']} | {guidance} |"

        # Try to find and replace the Mechanically Verified table
        # Pattern matches from "### Mechanically Verified" through the table
        mech_pattern = (
            r"(### Mechanically Verified\n\n)"
            r"(?:These are verified automatically[^\n]*\n\n)?"
            r"\| ID \| Assumption \| Verification \|\n"
            r"\|[-|]+\|\n"
            r"(?:\| [A-Z][0-9]+ \|[^\n]+\|\n)+"
        )

        mech_replacement = (
            r"\1"
            f"These are verified automatically via `python3 verify-assumptions.py {trace_name}`.\n\n"
            f"{mech_table}\n"
        )

        new_content, mech_count = re.subn(mech_pattern, mech_replacement, content)

        # Try to find and replace the Agent-Verified table
        agent_pattern = (
            r"(### Agent-Verified \(on trace load\)\n\n)"
            r"(?:These require human/agent judgment[^\n]*\n\n)?"
            r"\| ID \| Assumption \| Verification Guidance \|\n"
            r"\|[-|]+\|\n"
            r"(?:\| [A-Z][0-9]+ \|[^\n]+\|\n)+"
        )

        agent_replacement = (
            r"\1"
            "These require human/agent judgment when loading the trace.\n\n"
            f"{manual_table}\n"
        )

        new_content, agent_count = re.subn(agent_pattern, agent_replacement, new_content)

        if mech_count > 0 or agent_count > 0:
            if not args.dry_run:
                trace_path.write_text(new_content)
            updated_count += 1
            print(f"  {'Would update' if args.dry_run else 'Updated'} {trace_name}")
            if mech_count > 0:
                print(f"      Mechanical assumptions: {len(mechanical)}")
            if agent_count > 0:
                print(f"      Agent-verified: {len(manual)}")
        else:
            print(f"  Skip {trace_name}: no matching table structure found")

    print()
    if args.dry_run:
        print(f"Dry run: would update {updated_count} trace(s)")
    else:
        print(f"Updated {updated_count} trace(s) from assumptions.yaml")

    return EXIT_SUCCESS


def cmd_route(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Route a symptom to the appropriate trace."""
    symptom = args.symptom
    use_json = getattr(args, "format", None) == "json"
    routing = sentinel.config.get("routing", [])

    if not routing:
        msg = "No routing entries configured in sentinel.yaml"
        if use_json:
            print(_json_envelope("route", exit_code=EXIT_GENERAL_ERROR, error=msg))
        else:
            print(f"Error: {msg}")
        return EXIT_GENERAL_ERROR

    # Deterministic match precedence: exact > case-insensitive > substring
    match = None
    match_type = None

    # 1. Exact match
    for entry in routing:
        if entry["symptom"] == symptom:
            match = entry
            match_type = "exact"
            break

    # 2. Case-insensitive exact match
    if not match:
        symptom_lower = symptom.lower()
        for entry in routing:
            if entry["symptom"].lower() == symptom_lower:
                match = entry
                match_type = "case_insensitive"
                break

    # 3. Substring containment
    if not match:
        for entry in routing:
            if (
                symptom_lower in entry["symptom"].lower()
                or entry["symptom"].lower() in symptom_lower
            ):
                match = entry
                match_type = "substring"
                break

    if not match:
        msg = f"No routing match for symptom: {symptom}"
        if use_json:
            print(_json_envelope("route", exit_code=EXIT_GENERAL_ERROR, error=msg))
        else:
            print(f"Error: {msg}")
            print("\nAvailable symptoms:")
            for entry in routing:
                print(f"  - {entry['symptom']}")
        return EXIT_GENERAL_ERROR

    trace_name = match["trace"]
    guidance = match["guidance"]

    # Auto-verify the suggested trace
    verification_result = sentinel.verify_trace(trace_name)
    route_exit_code = EXIT_SUCCESS
    if verification_result.overall_status != Status.VERIFIED:
        if verification_result.commit_status == Status.MISSING:
            route_exit_code = EXIT_GENERAL_ERROR
        elif verification_result.commit_status in (Status.STALE_COMMIT, Status.STALE_CONTENT):
            route_exit_code = EXIT_ANCHOR_DRIFT
        elif any(a.status == "MISSING" for a in verification_result.anchors):
            route_exit_code = EXIT_ANCHOR_MISSING
        elif any(a.status == "DRIFT" for a in verification_result.anchors):
            route_exit_code = EXIT_ANCHOR_DRIFT
        elif any(a.status == "AMBIGUOUS" for a in verification_result.anchors):
            route_exit_code = EXIT_ANCHOR_AMBIGUOUS
        elif any(not a.passed for a in verification_result.assumptions):
            route_exit_code = EXIT_ASSUMPTION_FAILED
        else:
            route_exit_code = EXIT_GENERAL_ERROR

    verification_summary = {
        "trace": trace_name,
        "status": verification_result.overall_status.value,
        "commit_status": verification_result.commit_status.value,
        "anchors_verified": sum(1 for a in verification_result.anchors if a.status == "VERIFIED"),
        "anchors_total": len(verification_result.anchors),
    }

    if use_json:
        print(
            _json_envelope(
                "route",
                exit_code=route_exit_code,
                symptom=symptom,
                matched_key=match["symptom"],
                trace=trace_name,
                guidance=guidance,
                match_type=match_type,
                verification=verification_summary,
            )
        )
    else:
        print(f"Symptom: {symptom}")
        print(f"  Matched: {match['symptom']} ({match_type})")
        print(f"  Primary trace: {trace_name}")
        print(f"  Check first: {guidance}")
        print()
        print("  Verifying trace...")
        status_sym = "[PASS]" if verification_result.overall_status == Status.VERIFIED else "[FAIL]"
        print(f"  Grounded: {trace_name} @ {verification_result.verified_commit} {status_sym}")
        print()
        print(f"  Load trace: .sentinel/traces/{trace_name}.md")
        if route_exit_code != EXIT_SUCCESS:
            print(
                "  [WARN] Suggested trace is not grounded; run verify/update before relying on it."
            )

    return route_exit_code


def cmd_update(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Update a sentinel (alias for retrace --auto --apply)."""
    # Delegate to retrace with auto+apply
    args.auto = True
    args.apply = True
    args.anchors_only = False
    args.diff_only = False
    return cmd_retrace(args, sentinel)


def cmd_context(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Generate context for LLM consumption."""
    trace_name = args.trace
    meta = sentinel.load_meta()
    anchors = sentinel.load_anchors()

    trace_meta = meta.get("sentinels", {}).get(trace_name, {})
    if not trace_meta:
        print(f"Error: Trace '{trace_name}' not found")
        return EXIT_GENERAL_ERROR

    # Verify the trace
    result = sentinel.verify_trace(trace_name)

    # Extract critical invariants from trace markdown
    trace_path = sentinel.traces_dir / f"{trace_name}.md"
    critical_invariants = []
    if trace_path.exists():
        content = trace_path.read_text()
        # Find Critical Invariants section
        inv_match = re.search(
            r"## Critical Invariants\n\n((?:- \[[ x]\] .+\n)+)",
            content,
        )
        if inv_match:
            for line in inv_match.group(1).strip().split("\n"):
                # Remove checkbox prefix
                inv_text = re.sub(r"^- \[[ x]\] ", "", line.strip())
                if inv_text and inv_text != "TODO: Document critical invariants":
                    critical_invariants.append(inv_text)

    if args.format == "llm":
        # Generate LLM-optimized YAML output
        context = {
            "trace": trace_name,
            "status": result.overall_status.value,
            "commit": result.verified_commit,
            "staleness": "current" if result.commit_status == Status.VERIFIED else "stale",
            "anchors_verified": [],
            "assumptions_passed": [],
            "assumptions_failed": [],
            "assumptions_manual": [],
            "critical_invariants": critical_invariants,
            "load_with": trace_meta.get("depends_on", []),
        }

        # Add anchor details
        for anchor in result.anchors:
            trace_anchors = anchors.get(trace_name, {})
            anchor_spec = trace_anchors.get(anchor.name, {})
            context["anchors_verified"].append(
                {
                    "name": anchor.name,
                    "file": anchor_spec.get("file", "unknown"),
                    "line": anchor.actual_line or anchor.expected_line,
                    "status": anchor.status.lower(),
                }
            )

        # Categorize assumptions
        for assumption in result.assumptions:
            if assumption.passed:
                context["assumptions_passed"].append(assumption.id)
            else:
                context["assumptions_failed"].append(assumption.id)

        # Agent-verified assumptions are manual
        context["assumptions_manual"] = trace_meta.get("assumptions_agent", [])

        # Output as YAML
        print(yaml.dump(context, default_flow_style=False, sort_keys=False))

    else:
        # Human-readable format (default)
        print(f"=== Sentinel Context: {trace_name} ===")
        print()
        print(f"Status: {result.overall_status.value}")
        print(f"Commit: {result.verified_commit}")
        print(
            f"Staleness: {'current' if result.commit_status == Status.VERIFIED else result.commit_status.value}"
        )
        print()

        print("Anchors:")
        for anchor in result.anchors:
            status_sym = "[PASS]" if anchor.status == "VERIFIED" else "[FAIL]"
            print(
                f"  {status_sym} {anchor.name}: line {anchor.actual_line or anchor.expected_line}"
            )

        print()
        print("Assumptions:")
        for assumption in result.assumptions:
            status_sym = "[PASS]" if assumption.passed else "[FAIL]"
            print(f"  {status_sym} {assumption.id}: {assumption.description}")

        if critical_invariants:
            print()
            print("Critical Invariants:")
            for inv in critical_invariants:
                print(f"  - {inv}")

        deps = trace_meta.get("depends_on", [])
        if deps:
            print()
            print(f"Load with: {', '.join(deps)}")

    return EXIT_SUCCESS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Code Sentinel - Verification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./sentinel.py status                          Show sentinel health
  ./sentinel.py status --format json            JSON status output
  ./sentinel.py status --verify                 Live verification status
  ./sentinel.py init src/module.py              Scaffold a new trace
  ./sentinel.py verify --trace triton-forward-k3plus
  ./sentinel.py verify --all --format json      JSON verification output
  ./sentinel.py route "NaN in loss"             Route symptom to trace
  ./sentinel.py update --trace NAME             Auto-update a sentinel
  ./sentinel.py pipeline                        Full pre-commit pipeline
  ./sentinel.py retrace NAME --auto --apply     Analyze and apply safe updates
  ./sentinel.py install-adapter --list          List available adapters
  ./sentinel.py install-adapter claude          Install Claude Code adapter
  ./sentinel.py install-adapter codex --dry-run Preview AGENTS managed-block update
  ./sentinel.py report --format json
        """,
    )
    parser.add_argument("--sentinel-dir", type=Path, help="Path to code-sentinel directory")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    status_parser = subparsers.add_parser("status", help="Show sentinel health")
    status_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    status_parser.add_argument(
        "--verify", action="store_true", help="Run live verification (read-only, no state mutation)"
    )

    # init
    init_parser = subparsers.add_parser("init", help="Scaffold a new trace from source file")
    init_parser.add_argument("source", help="Source file to analyze")
    init_parser.add_argument("--name", help="Trace name (default: source filename stem)")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing trace")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify traces")
    verify_parser.add_argument("--trace", help="Specific trace to verify")
    verify_parser.add_argument("--all", action="store_true", help="Verify all traces")
    verify_parser.add_argument(
        "--check-consistency", action="store_true", help="Check meta/anchor/trace consistency"
    )
    verify_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    verify_parser.add_argument(
        "--strict", action="store_true", help="Stale traces cause non-zero exit"
    )
    verify_parser.add_argument(
        "--no-strict", action="store_true", help="Stale traces are warning-only (overrides env)"
    )

    # pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pre-commit pipeline")
    pipeline_parser.add_argument(
        "--files", nargs="*", help="Specific files to check (default: staged files)"
    )
    pipeline_parser.add_argument("--ci", action="store_true", help="CI mode (non-interactive)")
    pipeline_parser.add_argument(
        "--strict", action="store_true", help="Stale traces cause non-zero exit"
    )
    pipeline_parser.add_argument(
        "--no-strict", action="store_true", help="Stale traces are warning-only (overrides env)"
    )

    # retrace
    retrace_parser = subparsers.add_parser("retrace", help="Update a sentinel")
    retrace_parser.add_argument("trace_name", help="Name of trace to update")
    retrace_parser.add_argument("--auto", action="store_true", help="Auto-analyze anchor impacts")
    retrace_parser.add_argument(
        "--apply", action="store_true", help="Apply safe updates (use with --auto)"
    )
    retrace_parser.add_argument(
        "--anchors-only", action="store_true", help="Force update all anchor line numbers"
    )
    retrace_parser.add_argument(
        "--diff-only", action="store_true", help="Only show diff, don't update"
    )

    # install-hooks
    subparsers.add_parser("install-hooks", help="Install git pre-commit hooks")

    # install-adapter
    adapter_parser = subparsers.add_parser(
        "install-adapter", help="Install an agent adapter from template"
    )
    adapter_parser.add_argument("adapter", nargs="?", help="Adapter name (e.g. claude, codex)")
    adapter_parser.add_argument("--list", action="store_true", help="List available adapters")
    adapter_parser.add_argument("--force", action="store_true", help="Overwrite existing target")
    adapter_parser.add_argument(
        "--dry-run", action="store_true", help="Preview adapter install without writing files"
    )

    # report
    report_parser = subparsers.add_parser("report", help="Generate verification report")
    report_parser.add_argument(
        "--format", choices=["json", "markdown"], default="json", help="Output format"
    )

    # graph
    graph_parser = subparsers.add_parser("graph", help="Generate dependency graph")
    graph_parser.add_argument(
        "--trace", help="Generate graph for specific trace and its dependencies"
    )
    graph_parser.add_argument(
        "--format", choices=["mermaid", "dot"], default="mermaid", help="Output format"
    )
    graph_parser.add_argument("--output", help="Output file path")

    # coverage
    coverage_parser = subparsers.add_parser("coverage", help="Generate trace coverage metrics")
    coverage_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    coverage_parser.add_argument(
        "--threshold", type=float, help="Minimum coverage percentage (exit 1 if below)"
    )

    # context
    context_parser = subparsers.add_parser("context", help="Generate context for LLM consumption")
    context_parser.add_argument("trace", help="Trace name")
    context_parser.add_argument(
        "--format",
        choices=["text", "llm"],
        default="text",
        help="Output format (llm = YAML optimized for LLM context)",
    )

    # sync-docs
    sync_docs_parser = subparsers.add_parser(
        "sync-docs", help="Regenerate trace markdown tables from assumptions.yaml"
    )
    sync_docs_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without writing"
    )

    # route
    route_parser = subparsers.add_parser("route", help="Route a symptom to the appropriate trace")
    route_parser.add_argument("symptom", help="Symptom description (e.g. 'NaN in loss')")
    route_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # update
    update_parser = subparsers.add_parser(
        "update", help="Update a sentinel (auto-analyze and apply)"
    )
    update_parser.add_argument(
        "--trace", dest="trace_name", required=True, help="Name of trace to update"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_SUCCESS

    try:
        sentinel = CodeSentinel(args.sentinel_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return EXIT_GENERAL_ERROR

    commands = {
        "status": cmd_status,
        "init": cmd_init,
        "verify": cmd_verify,
        "pipeline": cmd_pipeline,
        "retrace": cmd_retrace,
        "update": cmd_update,
        "route": cmd_route,
        "install-hooks": cmd_install_hooks,
        "install-adapter": cmd_install_adapter,
        "report": cmd_report,
        "graph": cmd_graph,
        "coverage": cmd_coverage,
        "context": cmd_context,
        "sync-docs": cmd_sync_docs,
    }

    return commands[args.command](args, sentinel)


if __name__ == "__main__":
    sys.exit(main())
