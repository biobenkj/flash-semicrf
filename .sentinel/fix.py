"""Fix and retrace core logic — shared by cmd_fix, cmd_gate, and watch."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from constants import EXIT_ANCHOR_DRIFT, EXIT_GENERAL_ERROR, EXIT_SUCCESS
from core import CodeSentinel, _compute_content_hash, _read_line_from_file
from models import Status, TraceVerification


def _update_trace_header(sentinel: CodeSentinel, trace_name: str, new_commit: str) -> Path | None:
    """Update the 'Verified against' commit hash in a trace markdown header.

    Returns the trace path if modified, None otherwise.
    """
    trace_path = sentinel.traces_dir / f"{trace_name}.md"
    if not trace_path.exists():
        return None
    content = trace_path.read_text()
    updated = re.sub(
        r"(@ commit `)([a-f0-9]+)(`)",
        rf"\g<1>{new_commit}\g<3>",
        content,
    )
    if updated != content:
        trace_path.write_text(updated)
        return trace_path
    return None


def _get_latest_commit(sentinel: CodeSentinel, source_files: list[str]) -> str | None:
    """Get the most recent commit hash across a list of source files."""
    if not source_files:
        return None
    result = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--"] + source_files,
        capture_output=True,
        text=True,
        cwd=sentinel.repo_root,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def _fix_traces(
    sentinel: CodeSentinel,
    trace_names: list[str],
    dry_run: bool = False,
) -> tuple[int, list[Path], list[TraceVerification], dict]:
    """Core fix logic shared by cmd_fix and cmd_gate.

    Returns (exit_code, modified_files, final_verifications, fix_results).
    """
    meta = sentinel.load_meta()
    anchors = sentinel.load_anchors()
    modified_files: list[Path] = []
    fix_results: dict[str, dict] = {}
    exit_code = EXIT_SUCCESS
    anchors_dirty = False
    meta_dirty = False

    for trace_name in trace_names:
        verification = sentinel.verify_trace(trace_name)

        if verification.overall_status == Status.VERIFIED:
            fix_results[trace_name] = {
                "previous_status": "VERIFIED",
                "new_status": "VERIFIED",
                "anchors_fixed": [],
                "anchors_failed": [],
                "commit_updated": False,
            }
            continue

        if verification.overall_status == Status.MISSING:
            fix_results[trace_name] = {
                "previous_status": "MISSING",
                "new_status": "MISSING",
                "anchors_fixed": [],
                "anchors_failed": [],
                "commit_updated": False,
            }
            exit_code = max(exit_code, EXIT_GENERAL_ERROR)
            continue

        # Analyze impacts to classify anchors
        impacts = sentinel.analyze_anchor_impacts(trace_name)

        anchors_fixed = []
        anchors_failed = []
        trace_anchors = anchors.get(trace_name, {})

        for impact in impacts:
            if impact.status == "shifted":
                # Safe if content_hash_match is True or None (no hash stored)
                if impact.content_hash_match is False:
                    anchors_failed.append(
                        {
                            "name": impact.anchor_name,
                            "reason": "content_hash mismatch (semantic change)",
                        }
                    )
                elif not dry_run and impact.new_line and impact.anchor_name in trace_anchors:
                    anchor_spec = trace_anchors[impact.anchor_name]
                    anchor_spec["expected_line"] = impact.new_line
                    # Compute new content_hash
                    anchor_file = str(sentinel.repo_root / anchor_spec["file"])
                    line_content = _read_line_from_file(anchor_file, impact.new_line)
                    if line_content:
                        anchor_spec["content_hash"] = _compute_content_hash(line_content)
                    anchors_dirty = True
                    anchors_fixed.append(
                        {
                            "name": impact.anchor_name,
                            "old_line": impact.old_line,
                            "new_line": impact.new_line,
                        }
                    )
                elif dry_run:
                    anchors_fixed.append(
                        {
                            "name": impact.anchor_name,
                            "old_line": impact.old_line,
                            "new_line": impact.new_line,
                        }
                    )
            elif impact.status in ("deleted", "modified"):
                anchors_failed.append(
                    {
                        "name": impact.anchor_name,
                        "reason": impact.suggestion,
                    }
                )

        previous_status = verification.overall_status.value

        if anchors_failed:
            exit_code = max(exit_code, EXIT_ANCHOR_DRIFT)

        # Bump commit and update header if all anchors are now safe
        commit_updated = False
        if not anchors_failed and not dry_run:
            trace_meta = meta.get("sentinels", {}).get(trace_name, {})
            source_files = trace_meta.get("source_files", [])
            new_commit = _get_latest_commit(sentinel, source_files)
            if new_commit:
                meta.setdefault("sentinels", {}).setdefault(trace_name, {})
                meta["sentinels"][trace_name]["verified_commit"] = new_commit
                meta["sentinels"][trace_name]["status"] = "VERIFIED"
                meta_dirty = True
                commit_updated = True

                trace_path = _update_trace_header(sentinel, trace_name, new_commit)
                if trace_path:
                    modified_files.append(trace_path)

        new_status = "VERIFIED" if (not anchors_failed and not dry_run) else "DEGRADED"
        if anchors_failed:
            new_status = "DEGRADED"

        fix_results[trace_name] = {
            "previous_status": previous_status,
            "new_status": new_status,
            "anchors_fixed": anchors_fixed,
            "anchors_failed": anchors_failed,
            "commit_updated": commit_updated,
            "new_verified_commit": (
                meta.get("sentinels", {}).get(trace_name, {}).get("verified_commit")
                if commit_updated
                else None
            ),
        }

    # Opportunistic hash population: fill in missing content_hash for verified anchors
    if not dry_run:
        for trace_name in trace_names:
            trace_anchors = anchors.get(trace_name, {})
            for _anchor_name, spec in trace_anchors.items():
                if "content_hash" not in spec:
                    anchor_file = str(sentinel.repo_root / spec["file"])
                    # Verify the anchor to find its actual line
                    result = sentinel.verify_anchor(
                        pattern=spec["pattern"],
                        expected_line=spec["expected_line"],
                        file_path=anchor_file,
                        tolerance=spec.get("drift_tolerance", 20),
                        after=spec.get("after"),
                    )
                    if result.status == "VERIFIED" and result.actual_line:
                        line_content = _read_line_from_file(anchor_file, result.actual_line)
                        if line_content:
                            spec["content_hash"] = _compute_content_hash(line_content)
                            anchors_dirty = True

    # Save modified state
    if anchors_dirty and not dry_run:
        sentinel.save_anchors(anchors)
        modified_files.append(sentinel.anchors_path)

    if meta_dirty and not dry_run:
        meta["last_global_verification"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        sentinel.save_meta(meta)
        modified_files.append(sentinel.meta_path)

    # Re-verify all traces for final state
    final_verifications = []
    for trace_name in trace_names:
        final_verifications.append(sentinel.verify_trace(trace_name))

    return exit_code, modified_files, final_verifications, fix_results
