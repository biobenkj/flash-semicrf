"""Core CodeSentinel class, bootstrap, and hash helpers."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from constants import (
    _SUPPORTED_LANGUAGES,
    _VERIFY_ANCHOR_SH,
    CRITICAL_PATTERNS,
    HIGH_PATTERNS,
)
from models import (
    BUILTIN_PRESETS,
    AnchorImpact,
    AnchorResult,
    AssumptionResult,
    FunctionInfo,
    ScanPreset,
    Status,
    TraceVerification,
)


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
        # Check common locations
        candidates = [
            Path(__file__).parent,
            Path(".sentinel"),
            Path.cwd() / ".sentinel",
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
        content_hash: str | None = None,
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
        anchor_status = status_map.get(result.returncode, "UNKNOWN")

        # Content hash check: read the actual line from the file and compare
        hash_match = None
        if content_hash and actual_line and anchor_status in ("VERIFIED", "DRIFT"):
            line_content = _read_line_from_file(file_path, actual_line)
            if line_content is not None:
                hash_match = _compute_content_hash(line_content) == content_hash

        return AnchorResult(
            name="",  # Will be set by caller
            status=anchor_status,
            expected_line=expected_line,
            actual_line=actual_line,
            drift=drift,
            message=output,
            content_hash_match=hash_match,
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
                content_hash=spec.get("content_hash"),
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

    # ── init --scan helpers ──────────────────────────────────────────────

    def load_scan_preset(self, name: str) -> ScanPreset:
        """Load a scan preset by name (built-in or custom from presets/).

        For the 'default' preset, overrides file_globs with project.src_root
        from config and critical/high patterns from init_patterns config.
        """
        if name in BUILTIN_PRESETS:
            preset = BUILTIN_PRESETS[name]
            # For default and pytorch presets, apply project config overrides
            if name in ("default", "pytorch"):
                src_root = self.config.get("project", {}).get("src_root", "src")
                cfg_critical = self.config.get("init_patterns", {}).get(
                    "critical", preset.critical_patterns
                )
                cfg_high = self.config.get("init_patterns", {}).get("high", preset.high_patterns)
                # For pytorch, merge config patterns with pytorch-specific extras
                if name == "pytorch":
                    pytorch_extra_critical = [
                        r"@torch\.no_grad",
                        r"class\s+\w+\(Function\)",
                        r"ctx\.save_for_backward",
                    ]
                    pytorch_extra_high = [
                        r"def training_step\(",
                        r"def validation_step\(",
                    ]
                    cfg_critical = list(cfg_critical) + [
                        p for p in pytorch_extra_critical if p not in cfg_critical
                    ]
                    cfg_high = list(cfg_high) + [p for p in pytorch_extra_high if p not in cfg_high]
                preset = ScanPreset(
                    name=preset.name,
                    description=preset.description,
                    language=preset.language,
                    file_globs=[f"{src_root}/**/*.py"],
                    exclude_globs=preset.exclude_globs,
                    critical_patterns=cfg_critical,
                    high_patterns=cfg_high,
                    min_functions=preset.min_functions,
                )
            return preset
        # Try custom preset from presets/ directory
        preset_path = self.sentinel_dir / "presets" / f"{name}.yaml"
        if preset_path.exists():
            data = yaml.safe_load(preset_path.read_text())
            return ScanPreset(**data)
        available = ", ".join(list(BUILTIN_PRESETS.keys()))
        raise ValueError(f"Unknown preset: {name}. Available: {available}")

    def discover_source_files(self, preset: ScanPreset) -> list[Path]:
        """Walk repo tree respecting .gitignore, filtered by preset globs.

        Only returns files whose language has a supported extractor.
        """
        import fnmatch

        if preset.language not in _SUPPORTED_LANGUAGES:
            print(
                f"  Skipping preset '{preset.name}': "
                f"no function extractor for {preset.language} (Phase 2b is Python-only)",
                file=sys.stderr,
            )
            return []

        cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
        for glob in preset.file_globs:
            # Prefix with :(glob) so git interprets ** as recursive match
            cmd.extend(["--", f":(glob){glob}"])
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
        if result.returncode != 0:
            return []

        files = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            excluded = any(fnmatch.fnmatch(line, pat) for pat in preset.exclude_globs)
            if not excluded:
                files.append(self.repo_root / line)
        return sorted(files)

    def derive_trace_name(self, source_path: Path, existing_names: set[str]) -> str:
        """Derive a trace name from a source file path, avoiding collisions.

        Converts file stem to kebab-case. On collision, prefixes with parent
        directory name. Last resort: numeric suffix. existing_names must include
        both existing trace names from meta AND names derived earlier in the
        current scan run.
        """
        stem = source_path.stem.replace("_", "-").replace(".", "-").strip("-")
        if stem not in existing_names:
            return stem
        # Add parent directory for disambiguation
        parent = source_path.parent.name.replace("_", "-").replace(".", "-").strip("-")
        qualified = f"{parent}-{stem}"
        if qualified not in existing_names:
            return qualified
        # Numeric suffix as last resort
        i = 2
        while f"{stem}-{i}" in existing_names:
            i += 1
        return f"{stem}-{i}"

    def scaffold_trace(
        self,
        source_path: Path,
        trace_name: str,
        head_commit: str,
        *,
        status: str = "VERIFIED",
        critical_patterns: list[str] | None = None,
        high_patterns: list[str] | None = None,
    ) -> tuple[str, dict, dict]:
        """Scaffold trace markdown, anchor entries, and meta entry for one source file.

        Args:
            source_path: Absolute path to source file.
            trace_name: Name for the trace.
            head_commit: Git short hash for verified_commit ("TODO" for draft mode).
            status: Trace status ("VERIFIED" for scan, "DRAFT" for single-file).
            critical_patterns: Override critical patterns for function classification.
            high_patterns: Override high patterns for function classification.

        Returns:
            (trace_markdown, anchor_dict, meta_entry)
        """
        # Temporarily override config patterns if provided
        orig_config = None
        if critical_patterns is not None or high_patterns is not None:
            orig_config = self.config.get("init_patterns", {}).copy()
            if critical_patterns is not None:
                self.config.setdefault("init_patterns", {})["critical"] = critical_patterns
            if high_patterns is not None:
                self.config.setdefault("init_patterns", {})["high"] = high_patterns

        try:
            functions = self.extract_functions(source_path)
            classes = self.extract_classes(source_path)
        finally:
            # Restore original config
            if orig_config is not None:
                self.config["init_patterns"] = orig_config

        try:
            rel_source = source_path.relative_to(self.repo_root)
        except ValueError:
            rel_source = source_path

        critical = [f for f in functions if f.importance == "critical"]
        high = [f for f in functions if f.importance == "high"]
        selected = (critical + high)[:10]

        # Read file lines for content hash computation
        try:
            source_lines = source_path.read_text().splitlines()
        except OSError:
            source_lines = []

        # Count how many times each function name appears (for ambiguity detection)
        name_counts: dict[str, int] = {}
        for func in functions:
            name_counts[func.name] = name_counts.get(func.name, 0) + 1

        # Build anchor dict with content hashes
        anchor_dict = {}
        for func in selected:
            anchor_name = func.name.upper()
            # Handle duplicate anchor names (e.g., forward in multiple classes)
            if anchor_name in anchor_dict:
                # Find enclosing class for disambiguation
                enclosing = None
                for cls_name, cls_line in reversed(classes):
                    if cls_line < func.line:
                        enclosing = cls_name
                        break
                if enclosing:
                    anchor_name = f"{enclosing.upper()}_{func.name.upper()}"
                else:
                    anchor_name = f"{func.name.upper()}_{func.line}"

            pattern = f"def {func.name}("
            content_hash = ""
            if func.line <= len(source_lines):
                content_hash = _compute_content_hash(source_lines[func.line - 1])

            entry: dict[str, Any] = {
                "file": str(rel_source),
                "pattern": pattern,
                "expected_line": func.line,
                "drift_tolerance": 30,
                "content_hash": content_hash,
            }

            # Add 'after' context for ambiguous patterns
            if name_counts.get(func.name, 0) > 1:
                enclosing = None
                for cls_name, cls_line in reversed(classes):
                    if cls_line < func.line:
                        enclosing = cls_name
                        break
                if enclosing:
                    entry["after"] = f"class {enclosing}"

            anchor_dict[anchor_name] = entry

        # Build trace markdown
        commit_display = f"`{head_commit}`" if head_commit != "TODO" else "TODO"
        trace_md = f"""# Sentinel: {trace_name}

**Verified against:** `{rel_source}` @ commit {commit_display}

**Status:** {status}

**Linked tests:** TODO

## Summary

TODO: Describe the purpose and key functionality of this module.

## Active Assumptions

### Mechanically Verified

| ID | Assumption | Verification |
|----|------------|--------------|
"""
        assumption_id = 1
        for func in critical[:5]:
            trace_md += (
                f"| A{assumption_id} | {func.name} exists at expected location "
                f"| anchor: {func.name.upper()} |\n"
            )
            assumption_id += 1

        trace_md += """
### Agent-Verified (on trace load)

| ID | Assumption | Verification Guidance |
|----|------------|----------------------|
| TODO | TODO | TODO |

## Algorithm Flow

TODO: Document the key algorithm steps with line references.

"""
        if critical or high:
            trace_md += "## Key Functions\n\n"
            trace_md += "| Function | Line | Importance |\n"
            trace_md += "|----------|------|------------|\n"
            for func in critical + high:
                trace_md += f"| `{func.name}` | {func.line} | {func.importance} |\n"
            trace_md += "\n"

        trace_md += """## Critical Invariants

- [ ] TODO: Document critical invariants

## Known Issues

| Issue | Severity | Resolution |
|-------|----------|------------|
| None documented | - | - |

## Version History

"""
        if status == "VERIFIED":
            trace_md += f"- **{head_commit}**: Initial trace (auto-scaffolded by `init --scan`)\n"
        else:
            trace_md += "- **TODO**: Initial trace (DRAFT)\n"

        # Build meta entry
        meta_entry = {
            "verified_commit": head_commit,
            "source_files": [str(rel_source)],
            "assumptions_mechanical": [],
            "assumptions_agent": [],
            "anchors": list(anchor_dict.keys()),
            "linked_tests": [],
            "status": status,
            "depends_on": [],
        }

        return trace_md, anchor_dict, meta_entry

    # ── end init --scan helpers ──────────────────────────────────────────

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
                content_hash=spec.get("content_hash"),
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
                        content_hash_match=result.content_hash_match,
                    )
                )
            elif result.status == "DRIFT":
                impacts.append(
                    AnchorImpact(
                        anchor_name=anchor_name,
                        status="shifted",
                        old_line=old_line,
                        new_line=result.actual_line,
                        suggestion=f"Update expected_line: {old_line} -> {result.actual_line}",
                        content_hash_match=result.content_hash_match,
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

    def save_meta(self, meta: dict) -> None:
        """Save .sentinel-meta.yaml."""
        with open(self.meta_path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    def build_file_trace_index(self) -> dict[str, list[str]]:
        """Invert meta source_files into {resolved_abs_path: [trace_names]}."""
        meta = self.load_meta()
        index: dict[str, list[str]] = {}
        for trace_name, trace_meta in meta.get("sentinels", {}).items():
            for src in trace_meta.get("source_files", []):
                abs_path = str((self.repo_root / src).resolve())
                index.setdefault(abs_path, []).append(trace_name)
        return index

    def get_monitored_dirs(self) -> set[str]:
        """Unique parent directories of all monitored source files."""
        meta = self.load_meta()
        dirs: set[str] = set()
        for trace_meta in meta.get("sentinels", {}).values():
            for src in trace_meta.get("source_files", []):
                dirs.add(str((self.repo_root / src).resolve().parent))
        return dirs

    def load_config(self) -> dict:
        """Load sentinel.yaml project config."""
        config_path = self.sentinel_dir / "sentinel.yaml"
        if not config_path.exists():
            return {}
        return yaml.safe_load(config_path.read_text()) or {}


def bootstrap_sentinel_dir(repo_root: Path) -> Path:
    """Create .sentinel/ directory structure if it doesn't exist.

    Creates all required subdirectories, empty config/meta/anchor files,
    and writes verify-anchor.sh with executable permissions. Safe to call
    on an existing .sentinel/ — only creates missing pieces.
    """
    sentinel_dir = repo_root / ".sentinel"
    for subdir in ["traces", "anchors", "adapters", "presets", "diffs", "hooks"]:
        (sentinel_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Empty meta if missing
    meta_path = sentinel_dir / ".sentinel-meta.yaml"
    if not meta_path.exists():
        meta_path.write_text('version: "2.0"\nsentinels: {}\n')

    # Empty anchors if missing
    anchors_path = sentinel_dir / "anchors" / "anchors.yaml"
    if not anchors_path.exists():
        anchors_path.write_text("{}\n")

    # Minimal config if missing
    config_path = sentinel_dir / "sentinel.yaml"
    if not config_path.exists():
        # Auto-detect project name from repo directory
        project_name = repo_root.name
        config_path.write_text(
            f'version: "1.0"\n\nproject:\n  name: {project_name}\n  src_root: src\n'
        )

    # verify-anchor.sh — always write to ensure it's current
    script_path = sentinel_dir / "anchors" / "verify-anchor.sh"
    script_path.write_text(_VERIFY_ANCHOR_SH)
    script_path.chmod(0o755)

    return sentinel_dir


def _compute_content_hash(line: str) -> str:
    """SHA-256 hex digest of a stripped source line."""
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()


def _read_line_from_file(file_path: str, line_number: int) -> str | None:
    """Read a specific line from a file by line number (1-based)."""
    try:
        with open(file_path) as f:
            for i, line in enumerate(f, 1):
                if i == line_number:
                    return line
    except OSError:
        pass
    return None
