"""Output formatting, status helpers, and shared pipeline utilities."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

from constants import EXIT_CONSISTENCY_FAILED, EXIT_SUCCESS
from models import Status, TraceVerification


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


def _resolve_strict(args: argparse.Namespace, sentinel) -> tuple[bool, str | None]:
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


def _write_status_json(sentinel, verifications: list[TraceVerification]) -> None:
    """Write .sentinel/status.json atomically from verification results."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    traces = {}
    for v in verifications:
        verified_count = sum(1 for a in v.anchors if a.status == "VERIFIED")
        total_count = len(v.anchors)
        entry: dict[str, Any] = {
            "status": v.overall_status.value,
            "verified_commit": v.verified_commit,
            "anchors": {"verified": verified_count, "total": total_count},
            "last_checked": now,
        }
        issues = []
        for a in v.anchors:
            if a.status != "VERIFIED":
                issues.append({"anchor": a.name, "status": a.status, "message": a.message})
        if issues:
            entry["issues"] = issues
        traces[v.name] = entry

    verified_total = sum(1 for v in verifications if v.overall_status == Status.VERIFIED)
    total = len(verifications)
    stale = total - verified_total
    summary = f"{verified_total}/{total} verified"
    if stale:
        summary += f", {stale} stale"

    status = {"timestamp": now, "traces": traces, "summary": summary}

    status_path = sentinel.sentinel_dir / "status.json"
    tmp_path = sentinel.sentinel_dir / "status.json.tmp"
    tmp_path.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp_path, status_path)


def _now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _progress_bar(percentage: float, width: int = 10) -> str:
    """Generate a text progress bar."""
    filled = int(percentage / 100 * width)
    empty = width - filled
    return "#" * filled + "-" * empty


def run_consistency_check(sentinel, use_json: bool = False) -> tuple[int, list[str]]:
    """Run consistency check and print results. Returns (exit_code_delta, errors)."""
    errors = sentinel.check_consistency()
    if not use_json:
        if errors:
            for err in errors:
                print(f"  [FAIL] {err}")
        else:
            print("  [PASS] All metadata consistent")
        print()
    return (EXIT_CONSISTENCY_FAILED if errors else EXIT_SUCCESS), errors


def status_badge(status_value: str) -> str:
    """Return [PASS], [WARN], or [FAIL] badge for a status string."""
    if status_value == "VERIFIED":
        return "[PASS]"
    if "STALE" in status_value:
        return "[WARN]"
    return "[FAIL]"
