"""Data models for Code Sentinel — dataclasses, enums, and presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from constants import CRITICAL_PATTERNS, HIGH_PATTERNS


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
    content_hash_match: bool | None = None  # None = no hash stored/not checked


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
    content_hash_match: bool | None = None  # None = no hash stored/not checked


@dataclass
class FunctionInfo:
    """Extracted function/method information for init command."""

    name: str
    line: int
    importance: str  # critical, high, medium


@dataclass
class WatchEvent:
    """Structured event emitted by sentinel watch."""

    event: str  # status_snapshot | drift_detected | fix_applied | semantic_drift | trace_verified | shutdown
    timestamp: str
    trace: str | None = None
    file: str | None = None
    traces: dict | None = None
    summary: str | None = None
    anchors_affected: list[str] | None = None
    anchors_updated: list[str] | None = None
    auto_fixable: bool | None = None
    new_status: str | None = None
    message: str | None = None


@dataclass
class ScanPreset:
    """Preset configuration for init --scan file discovery and function classification."""

    name: str
    description: str
    language: str  # "python", "typescript", "rust" — controls which extractor runs
    file_globs: list[str]
    exclude_globs: list[str]
    critical_patterns: list[str]
    high_patterns: list[str]
    min_functions: int = 2  # minimum critical+high functions to create a trace


BUILTIN_PRESETS: dict[str, ScanPreset] = {
    "default": ScanPreset(
        name="default",
        description="Python source files with project-specific patterns",
        language="python",
        file_globs=["src/**/*.py"],
        exclude_globs=[
            "**/test_*.py",
            "**/*_test.py",
            "**/__init__.py",
            "**/conftest.py",
            "**/setup.py",
        ],
        critical_patterns=CRITICAL_PATTERNS,
        high_patterns=HIGH_PATTERNS,
        min_functions=2,
    ),
    "pytorch": ScanPreset(
        name="pytorch",
        description="PyTorch/Triton projects (kernels, autograd, forward/backward)",
        language="python",
        file_globs=["src/**/*.py"],
        exclude_globs=[
            "**/test_*.py",
            "**/*_test.py",
            "**/__init__.py",
            "**/conftest.py",
            "**/setup.py",
        ],
        critical_patterns=CRITICAL_PATTERNS
        + [
            r"@torch\.no_grad",
            r"class\s+\w+\(Function\)",
            r"ctx\.save_for_backward",
        ],
        high_patterns=HIGH_PATTERNS
        + [
            r"def training_step\(",
            r"def validation_step\(",
        ],
        min_functions=2,
    ),
    "typescript": ScanPreset(
        name="typescript",
        description="TypeScript projects (route handlers, middleware, hooks)",
        language="typescript",
        file_globs=["src/**/*.ts", "src/**/*.tsx"],
        exclude_globs=["**/*.test.ts", "**/*.spec.ts", "**/*.d.ts"],
        critical_patterns=[
            r"export (async )?function",
            r"app\.(get|post|put|delete|patch)\(",
            r"router\.(get|post|put|delete|patch)\(",
        ],
        high_patterns=[
            r"export default",
            r"middleware",
            r"useEffect",
            r"useState",
        ],
        min_functions=2,
    ),
}
