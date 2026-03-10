"""All cmd_* CLI handlers, graph helpers, and main() entry point."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from adapters import (
    _adapter_install_status,
    _extract_managed_block_content,
    _normalize_text_for_compare,
    _prepare_adapter_install,
    _render_managed_block,
)
from constants import (
    EXIT_ANCHOR_AMBIGUOUS,
    EXIT_ANCHOR_DRIFT,
    EXIT_ANCHOR_MISSING,
    EXIT_ASSUMPTION_FAILED,
    EXIT_CONSISTENCY_FAILED,
    EXIT_GENERAL_ERROR,
    EXIT_SUCCESS,
)
from core import CodeSentinel, bootstrap_sentinel_dir
from fix import _fix_traces
from models import Status, TraceVerification, WatchEvent
from output import (
    _json_envelope,
    _now_iso,
    _print_verification_result,
    _progress_bar,
    _resolve_strict,
    _write_status_json,
    run_consistency_check,
    status_badge,
)
from watch import (
    _check_stale_pidfile,
    _cleanup_pidfile,
    _daemonize,
    _emit_watch_event,
    _FileDebouncer,
    _watch_with_polling,
    _watch_with_watchdog,
)


def cmd_status(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Show overall sentinel health."""
    # --watch-summary: single-line output for status bars
    if getattr(args, "watch_summary", False):
        # Prefer status.json if it exists (more accurate)
        status_path = sentinel.sentinel_dir / "status.json"
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text())
                print(f"sentinel: {data.get('summary', 'unknown')}")
                return EXIT_SUCCESS
            except (json.JSONDecodeError, KeyError):
                pass  # Fall through to meta-based summary
        # Fallback: compute from meta
        meta = sentinel.load_meta()
        sentinels = meta.get("sentinels", {})
        total = len(sentinels)
        verified = sum(1 for t in sentinels.values() if t.get("status") == "VERIFIED")
        stale = total - verified
        summary = f"{verified}/{total} verified"
        if stale:
            summary += f", {stale} stale"
        print(f"sentinel: {summary}")
        return EXIT_SUCCESS

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
            symbol = status_badge(status)
            print(f"  {symbol} {trace_name}: {status}")

    return EXIT_SUCCESS


def cmd_init_scan(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Scan repo and scaffold traces for all discovered source files."""
    dry_run = getattr(args, "dry_run", False)
    preset_name = getattr(args, "preset", "default") or "default"
    force = getattr(args, "force", False)

    # Load preset
    try:
        preset = sentinel.load_scan_preset(preset_name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_GENERAL_ERROR

    # Get HEAD commit
    head_result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=sentinel.repo_root,
    )
    if head_result.returncode != 0:
        print("Error: could not determine HEAD commit", file=sys.stderr)
        return EXIT_GENERAL_ERROR
    head_commit = head_result.stdout.strip()

    # Discover source files
    source_files = sentinel.discover_source_files(preset)
    if not source_files:
        print(f"No source files found for preset '{preset_name}'")
        return EXIT_SUCCESS

    print(f"Discovered {len(source_files)} source files (preset: {preset_name})")

    # Check for uncommitted changes across discovered files
    dirty_result = subprocess.run(
        ["git", "diff", "--name-only"],
        capture_output=True,
        text=True,
        cwd=sentinel.repo_root,
    )
    staged_result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        cwd=sentinel.repo_root,
    )
    dirty_files = set()
    for output in (dirty_result.stdout, staged_result.stdout):
        for line in output.strip().split("\n"):
            if line:
                dirty_files.add(str((sentinel.repo_root / line).resolve()))
    dirty_discovered = [f for f in source_files if str(f.resolve()) in dirty_files]
    if dirty_discovered:
        print(
            f"\nNote: {len(dirty_discovered)} file(s) have uncommitted changes; "
            "traces will show STALE_CONTENT until committed",
            file=sys.stderr,
        )

    # Load existing state
    meta = sentinel.load_meta()
    anchors = sentinel.load_anchors()
    if anchors is None:
        anchors = {}

    # Build set of already-covered source files (by resolved path)
    existing_sentinels = meta.get("sentinels", {})
    covered_files: set[str] = set()
    for trace_meta in existing_sentinels.values():
        for sf in trace_meta.get("source_files", []):
            covered_files.add(str((sentinel.repo_root / sf).resolve()))

    # Build set of existing trace names for collision detection
    existing_names: set[str] = set(existing_sentinels.keys())

    # Scaffold each file
    new_traces: list[str] = []
    skipped_covered = 0
    skipped_threshold = 0
    for source_path in source_files:
        resolved = str(source_path.resolve())

        # Skip already-covered files unless --force
        if resolved in covered_files and not force:
            skipped_covered += 1
            continue

        # Derive trace name (checks existing + scan-derived names)
        trace_name = sentinel.derive_trace_name(source_path, existing_names)

        # Check min_functions threshold
        functions = sentinel.extract_functions(source_path)
        trace_worthy = [f for f in functions if f.importance in ("critical", "high")]
        if len(trace_worthy) < preset.min_functions:
            skipped_threshold += 1
            continue

        if dry_run:
            rel = source_path.relative_to(sentinel.repo_root)
            print(
                f"  [dry-run] Would create trace: {trace_name} "
                f"({rel}, {len(trace_worthy)} anchors)"
            )
            existing_names.add(trace_name)
            continue

        # Get per-file commit (what verify uses to check staleness)
        try:
            rel = source_path.relative_to(sentinel.repo_root)
        except ValueError:
            rel = source_path
        file_commit = sentinel.get_current_commit(str(rel)) or head_commit

        # Scaffold the trace
        trace_md, anchor_dict, meta_entry = sentinel.scaffold_trace(
            source_path,
            trace_name,
            file_commit,
            status="VERIFIED",
            critical_patterns=preset.critical_patterns,
            high_patterns=preset.high_patterns,
        )

        # Write trace markdown
        trace_path = sentinel.traces_dir / f"{trace_name}.md"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(trace_md)

        # Merge into anchors and meta
        anchors[trace_name] = anchor_dict
        meta.setdefault("sentinels", {})[trace_name] = meta_entry

        existing_names.add(trace_name)
        new_traces.append(trace_name)
        print(f"  Created trace: {trace_name} ({len(anchor_dict)} anchors)")

    if dry_run:
        created = len(source_files) - skipped_covered - skipped_threshold
        print(f"\n[dry-run] Would create {created} traces")
        if skipped_covered:
            print(f"  Skipped {skipped_covered} already-covered files")
        if skipped_threshold:
            print(
                f"  Skipped {skipped_threshold} files below "
                f"min_functions={preset.min_functions} threshold"
            )
        return EXIT_SUCCESS

    if not new_traces:
        print("No new traces created (all files already covered or below threshold)")
        return EXIT_SUCCESS

    # Save merged state
    sentinel.save_anchors(anchors)
    sentinel.save_meta(meta)

    # Verify ALL traces (new + existing) for complete status.json
    all_trace_names = list(meta.get("sentinels", {}).keys())
    verifications = []
    for tname in all_trace_names:
        v = sentinel.verify_trace(tname)
        verifications.append(v)

    _write_status_json(sentinel, verifications)

    # Install hooks if not already installed
    hook_path = sentinel.repo_root / ".git" / "hooks" / "pre-commit"
    if not hook_path.exists():
        print("\nInstalling pre-commit hooks...")
        hook_args = argparse.Namespace()
        cmd_install_hooks(hook_args, sentinel)

    # Install adapter if requested
    adapter = getattr(args, "adapter", None)
    if adapter:
        print(f"\nInstalling {adapter} adapter...")
        adapter_args = argparse.Namespace(adapter=adapter, list=False, force=False, dry_run=False)
        cmd_install_adapter(adapter_args, sentinel)

    # Summary with honest status reporting
    verified_count = sum(1 for v in verifications if v.overall_status == Status.VERIFIED)
    total = len(verifications)
    stale_count = total - verified_count

    summary = f"\nScan complete: {len(new_traces)} traces created"
    summary += f", {verified_count}/{total} verified"
    if stale_count:
        summary += f", {stale_count} stale"
        if dirty_discovered:
            summary += " (uncommitted changes)"
    print(summary)

    if skipped_covered:
        print(f"  Skipped {skipped_covered} already-covered files")
    if skipped_threshold:
        print(
            f"  Skipped {skipped_threshold} files below "
            f"min_functions={preset.min_functions} threshold"
        )

    return EXIT_SUCCESS


def cmd_init(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Scaffold a new trace from a source file, or scan repo with --scan."""
    if getattr(args, "scan", False):
        return cmd_init_scan(args, sentinel)

    if not args.source:
        print("Error: source file required (or use --scan for repo-wide scan)")
        return EXIT_GENERAL_ERROR

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

    print(f"Analyzing {source_path}...")
    functions = sentinel.extract_functions(source_path)
    classes = sentinel.extract_classes(source_path)
    critical = [f for f in functions if f.importance == "critical"]
    high = [f for f in functions if f.importance == "high"]
    print(f"  Found {len(functions)} functions, {len(classes)} classes")
    print(f"  Critical: {len(critical)}, High: {len(high)}")

    # Scaffold via shared helper (single-file mode: DRAFT status, TODO commit)
    trace_md, anchor_dict, meta_entry = sentinel.scaffold_trace(
        source_path, trace_name, "TODO", status="DRAFT"
    )

    # Write trace file
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(trace_md)
    print(f"\nCreated trace: {trace_path}")

    # Format anchor suggestions for display
    try:
        rel_source = source_path.relative_to(sentinel.repo_root)
    except ValueError:
        rel_source = source_path

    print("\n=== Suggested Anchors ===")
    print("Add to anchors/anchors.yaml:")
    print(f"\n# Anchors for {trace_name} (generated scaffold)\n{trace_name}:")
    for aname, adef in anchor_dict.items():
        print(f"  {aname}:")
        print(f"    file: {adef['file']}")
        print(f"    pattern: \"{adef['pattern']}\"")
        print(f"    expected_line: {adef['expected_line']}")
        print(f"    drift_tolerance: {adef['drift_tolerance']}")

    # Format meta entry suggestion for display
    anchor_list = ", ".join(anchor_dict.keys())
    print("\n=== Suggested Meta Entry ===")
    print(f"""
# Add to .sentinel-meta.yaml under sentinels:
  {trace_name}:
    verified_commit: TODO
    source_files:
      - {rel_source}
    assumptions_mechanical: []
    assumptions_agent: []
    anchors: [{anchor_list}]
    linked_tests: []
    status: DRAFT
    depends_on: []
""")

    print("=" * 60)
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
    elif getattr(args, "affected_by", None):
        # Normalize input paths to repo-relative form
        normalized_files = set()
        for fp in args.affected_by:
            p = Path(fp).resolve()
            try:
                rel = str(p.relative_to(sentinel.repo_root))
            except ValueError:
                rel = fp  # Already relative or outside repo
            normalized_files.add(rel)
        # Find traces whose source_files match
        for trace_name, trace_meta in meta.get("sentinels", {}).items():
            for sf in trace_meta.get("source_files", []):
                if sf in normalized_files:
                    traces_to_verify.append(trace_name)
                    break
        if not traces_to_verify:
            if use_json:
                print(
                    _json_envelope(
                        "verify",
                        exit_code=EXIT_SUCCESS,
                        strict_mode=strict,
                        traces={},
                        consistency_errors=[],
                        message=f"No traces bound to {', '.join(args.affected_by)}",
                    )
                )
            else:
                print(f"No traces bound to {', '.join(args.affected_by)}")
            return EXIT_SUCCESS
    else:
        msg = "Specify --trace NAME, --all, or --affected-by FILE"
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
    all_verifications: list[TraceVerification] = []
    for trace_name in traces_to_verify:
        result = sentinel.verify_trace(trace_name)
        all_verifications.append(result)

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

    # Write status.json when verifying all traces
    if args.all and all_verifications:
        _write_status_json(sentinel, all_verifications)

    return exit_code


def cmd_pipeline(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Run full pre-commit pipeline (deprecated: use 'gate' instead)."""
    print("Warning: 'pipeline' is deprecated, use 'gate' instead", file=sys.stderr)
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
    cc_exit, _ = run_consistency_check(sentinel)
    exit_code = max(exit_code, cc_exit)

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
        source_files = trace_meta.get("source_files", [])

        if not verified_commit:
            print("Error: No verified_commit found for this trace")
            return EXIT_GENERAL_ERROR

        for source_file in source_files:
            print(f"\n=== Changes to {source_file} since {verified_commit} ===\n")
            subprocess.run(
                ["git", "diff", verified_commit, "--", source_file],
                cwd=sentinel.repo_root,
            )

        return EXIT_SUCCESS

    if args.auto:
        # Analyze anchor impacts
        impacts = sentinel.analyze_anchor_impacts(trace_name)

        if not impacts:
            print("No anchor impacts detected (all anchors verified)")
            return EXIT_SUCCESS

        print("=== Anchor Impact Analysis ===\n")
        for impact in impacts:
            symbol = {
                "unchanged": "[PASS]",
                "shifted": "[~]",
                "modified": "[!]",
                "deleted": "[X]",
            }.get(impact.status, "[?]")
            hash_info = ""
            if impact.content_hash_match is True:
                hash_info = " (content unchanged)"
            elif impact.content_hash_match is False:
                hash_info = " (CONTENT CHANGED)"
            print(f"  {symbol} {impact.anchor_name}: {impact.status}{hash_info}")
            if impact.status == "shifted" and impact.new_line:
                print(f"      Line {impact.old_line} -> {impact.new_line}")
            if impact.suggestion:
                print(f"      {impact.suggestion}")

        if args.apply:
            print("\n=== Applying Safe Updates ===\n")
            exit_code, modified_files, final_verifications, fix_results = _fix_traces(
                sentinel, [trace_name]
            )

            trace_result = fix_results.get(trace_name, {})
            for af in trace_result.get("anchors_fixed", []):
                print(f"  \u2713 {af['name']}: {af['old_line']}\u2192{af['new_line']}")
            for af in trace_result.get("anchors_failed", []):
                print(f"  \u2717 {af['name']}: {af['reason']}")
            if trace_result.get("commit_updated"):
                print(
                    f"  \u2713 verified_commit updated: {trace_result.get('new_verified_commit')}"
                )

            # Re-verify and display
            final = sentinel.verify_trace(trace_name)
            print()
            _print_verification_result(final)

            # Update status.json
            _write_status_json(sentinel, final_verifications)

            return exit_code

        return EXIT_SUCCESS

    if args.anchors_only:
        # Force update all anchor line numbers
        print("=== Force-updating all anchor line numbers ===\n")
        exit_code, modified_files, final_verifications, fix_results = _fix_traces(
            sentinel, [trace_name]
        )

        trace_result = fix_results.get(trace_name, {})
        for af in trace_result.get("anchors_fixed", []):
            print(f"  \u2713 {af['name']}: {af['old_line']}\u2192{af['new_line']}")
        for af in trace_result.get("anchors_failed", []):
            print(f"  \u2717 {af['name']}: {af['reason']}")

        _write_status_json(sentinel, final_verifications)
        return exit_code

    # Default: show instructions
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


def cmd_fix(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Auto-fix shifted anchors and bump verification state."""
    # Resolve targets
    meta = sentinel.load_meta()
    if args.trace:
        trace_names = [args.trace]
    elif getattr(args, "all", False):
        trace_names = list(meta.get("sentinels", {}).keys())
    else:
        if getattr(args, "format", "text") == "json":
            print(_json_envelope("fix", EXIT_GENERAL_ERROR, error="Specify --trace NAME or --all"))
        else:
            print("Error: Specify --trace NAME or --all")
        return EXIT_GENERAL_ERROR

    if not trace_names:
        if getattr(args, "format", "text") == "json":
            print(
                _json_envelope(
                    "fix",
                    EXIT_SUCCESS,
                    fixes={},
                    summary={"fixed": 0, "still_broken": 0, "already_verified": 0},
                )
            )
        else:
            print("No traces found")
        return EXIT_SUCCESS

    dry_run = getattr(args, "dry_run", False)
    fmt = getattr(args, "format", "text")

    exit_code, modified_files, final_verifications, fix_results = _fix_traces(
        sentinel, trace_names, dry_run=dry_run
    )

    # Write status.json (not in dry-run)
    if not dry_run:
        _write_status_json(sentinel, final_verifications)

    # Output
    if fmt == "json":
        fixed_count = sum(
            1
            for r in fix_results.values()
            if r["new_status"] == "VERIFIED" and r["previous_status"] != "VERIFIED"
        )
        broken_count = sum(1 for r in fix_results.values() if r["anchors_failed"])
        already_count = sum(1 for r in fix_results.values() if r["previous_status"] == "VERIFIED")
        print(
            _json_envelope(
                "fix",
                exit_code,
                fixes=fix_results,
                summary={
                    "fixed": fixed_count,
                    "still_broken": broken_count,
                    "already_verified": already_count,
                },
            )
        )
    else:
        prefix = "Would fix" if dry_run else "Fixing"
        print(f"{prefix} {len(trace_names)} trace(s)...\n")
        for trace_name, result in fix_results.items():
            print(f"  {trace_name}:")
            for af in result["anchors_fixed"]:
                symbol = "~" if dry_run else "\u2713"
                print(
                    f"    {symbol} {af['name']}: {af['old_line']}\u2192{af['new_line']} ({'would fix' if dry_run else 'fixed'})"
                )
            for af in result["anchors_failed"]:
                print(f"    \u2717 {af['name']}: {af['reason']}")
            if result.get("commit_updated"):
                print(f"    \u2713 verified_commit updated: {result.get('new_verified_commit')}")
            print(f"    \u2192 {result['new_status']}")
            print()

        fixed_count = sum(
            1
            for r in fix_results.values()
            if r["new_status"] == "VERIFIED" and r["previous_status"] != "VERIFIED"
        )
        broken_count = sum(1 for r in fix_results.values() if r["anchors_failed"])
        already_count = sum(1 for r in fix_results.values() if r["previous_status"] == "VERIFIED")
        print(
            f"Summary: {fixed_count} fixed, {broken_count} need{'s' if broken_count == 1 else ''} attention, {already_count} already verified"
        )

    return exit_code


def cmd_gate(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Pre-commit gate: auto-fix + verify + test advisory."""
    strict, strict_err = _resolve_strict(args, sentinel)
    if strict_err:
        print(f"Error: {strict_err}", file=sys.stderr)
        return EXIT_GENERAL_ERROR

    print("=" * 60)
    print(f"Code Sentinel Gate{' (strict)' if strict else ''}")
    print("=" * 60)
    print()

    exit_code = EXIT_SUCCESS
    meta = sentinel.load_meta()
    trace_names = list(meta.get("sentinels", {}).keys())
    modified_files: list[Path] = []

    # Step 1: Auto-fix (unless --no-fix)
    if not getattr(args, "no_fix", False):
        print("=== Step 1: Auto-Fix ===\n")
        fix_exit, fix_modified, _, fix_results = _fix_traces(sentinel, trace_names, dry_run=False)
        modified_files.extend(fix_modified)
        if fix_exit != EXIT_SUCCESS:
            exit_code = max(exit_code, fix_exit)

        fixed_count = sum(
            1
            for r in fix_results.values()
            if r["new_status"] == "VERIFIED" and r["previous_status"] != "VERIFIED"
        )
        broken_count = sum(1 for r in fix_results.values() if r["anchors_failed"])
        if fixed_count or broken_count:
            print(f"  Auto-fix: {fixed_count} fixed, {broken_count} need attention")
        else:
            print("  No fixes needed")
        print()
    else:
        print("=== Step 1: Auto-Fix (skipped: --no-fix) ===\n")

    # Step 2: Consistency check
    print("=== Step 2: Consistency Check ===\n")
    cc_exit, _ = run_consistency_check(sentinel)
    exit_code = max(exit_code, cc_exit)

    # Step 3: Verify all traces
    print("=== Step 3: Trace Verification ===\n")
    verifications = []
    for trace_name in trace_names:
        result = sentinel.verify_trace(trace_name)
        verifications.append(result)

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
    print("=== Step 4: Test Advisory ===\n")
    if args.files:
        modified_source_files = args.files
    else:
        modified_source_files = sentinel.get_modified_files()

    if modified_source_files:
        suggested_tests = sentinel.get_suggested_tests(modified_source_files)
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
                print("  Run with: SENTINEL_RUN_TESTS=1 ./sentinel.py gate")
        else:
            print("  No sentinel-bound tests for modified files")
    else:
        print("  No staged files to check")
    print()

    # Write status.json
    _write_status_json(sentinel, verifications)

    # Summary
    print("=" * 60)
    if exit_code == EXIT_SUCCESS:
        print("Gate: PASSED [PASS]")
    else:
        print(f"Gate: FAILED (exit code {exit_code})")
    print("=" * 60)

    return exit_code


def cmd_watch(args: argparse.Namespace, sentinel: CodeSentinel) -> int:
    """Continuous file monitoring with auto-fix."""
    # Resolve auto_fix: CLI flag > config > default (True)
    watch_cfg = sentinel.config.get("watch", {})
    auto_fix = args.auto_fix
    if auto_fix is None:  # neither --auto-fix nor --no-auto-fix passed
        auto_fix = watch_cfg.get("auto_fix", True)
    json_lines = getattr(args, "json_lines", False)
    debounce_ms = getattr(args, "debounce_ms", None) or watch_cfg.get("debounce_ms", 500)
    poll_interval = getattr(args, "poll_interval", None) or watch_cfg.get("poll_interval_s", 2)
    trace_filter = getattr(args, "traces", None)

    # Check for stale PID file
    _check_stale_pidfile(sentinel)

    # Build file-to-trace index
    file_trace_index = sentinel.build_file_trace_index()
    if trace_filter:
        allowed = set(trace_filter.split(","))
        file_trace_index = {
            f: [t for t in ts if t in allowed] for f, ts in file_trace_index.items()
        }
        file_trace_index = {f: ts for f, ts in file_trace_index.items() if ts}

    if not file_trace_index:
        print("No monitored files found. Run 'sentinel init' first.", file=sys.stderr)
        return EXIT_GENERAL_ERROR

    # Daemon mode (Unix only)
    if getattr(args, "daemon", False):
        _daemonize(sentinel)

    # Signal handling
    _shutdown = [False]

    def _on_signal(sig, frame):
        _shutdown[0] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Initial verify all -> populate cache + emit status_snapshot
    verification_cache: dict[str, TraceVerification] = {}
    meta = sentinel.load_meta()
    for trace_name in meta.get("sentinels", {}):
        verification_cache[trace_name] = sentinel.verify_trace(trace_name)
    _write_status_json(sentinel, list(verification_cache.values()))

    verified = sum(1 for v in verification_cache.values() if v.overall_status == Status.VERIFIED)
    total = len(verification_cache)
    _emit_watch_event(
        WatchEvent(
            event="status_snapshot",
            timestamp=_now_iso(),
            traces={n: v.overall_status.value for n, v in verification_cache.items()},
            summary=f"{verified}/{total} verified",
        ),
        json_lines,
    )

    if not json_lines:
        print(f"Watching {len(file_trace_index)} files across {total} traces...", file=sys.stderr)

    # Enter watch loop
    debouncer = _FileDebouncer(debounce_ms)
    try:
        try:
            import watchdog  # noqa: F401

            _watch_with_watchdog(
                sentinel,
                file_trace_index,
                debouncer,
                lambda: _shutdown[0],
                auto_fix,
                json_lines,
                verification_cache,
            )
        except ImportError:
            if not json_lines:
                print(
                    "watchdog not installed, using stat-based polling",
                    file=sys.stderr,
                )
            _watch_with_polling(
                sentinel,
                file_trace_index,
                debouncer,
                lambda: _shutdown[0],
                auto_fix,
                json_lines,
                verification_cache,
                poll_interval,
            )
    finally:
        verified = sum(
            1 for v in verification_cache.values() if v.overall_status == Status.VERIFIED
        )
        total = len(verification_cache)
        _emit_watch_event(
            WatchEvent(
                event="shutdown",
                timestamp=_now_iso(),
                summary=f"{verified}/{total} verified at shutdown",
            ),
            json_lines,
        )
        _cleanup_pidfile(sentinel)

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
            src_id = src.replace("-", "_")
            dst_id = dst.replace("-", "_")
            lines.append(f"    {src_id}[{src}] --> {dst_id}[{dst}]")
    else:
        for node in sorted(nodes):
            node_id = node.replace("-", "_")
            lines.append(f"    {node_id}[{node}]")

    lines.append("```")
    return "\n".join(lines)


def _generate_dot_graph(nodes: set[str], edges: list[tuple[str, str]]) -> str:
    """Generate DOT format for Graphviz."""
    lines = ["digraph sentinels {", "    rankdir=TB;", "    node [shape=box];", ""]

    for node in sorted(nodes):
        node_id = node.replace("-", "_")
        lines.append(f'    {node_id} [label="{node}"];')

    lines.append("")

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
  ./sentinel.py status --watch-summary          Single-line summary for status bars
  ./sentinel.py verify --trace triton-forward-k3plus
  ./sentinel.py verify --all --format json      JSON verification output
  ./sentinel.py verify --affected-by src/file.py  Verify traces covering a file
  ./sentinel.py fix --all                       Auto-fix shifted anchors
  ./sentinel.py fix --all --dry-run             Preview fixes without writing
  ./sentinel.py gate                            Pre-commit: fix + verify + test advisory
  ./sentinel.py gate --no-fix                   Pre-commit: verify only (no auto-fix)
  ./sentinel.py watch                           Continuous file monitoring with auto-fix
  ./sentinel.py watch --json-lines --daemon     Background daemon with JSON event stream
  ./sentinel.py route "NaN in loss"             Route symptom to trace
  ./sentinel.py retrace NAME --auto --apply     Analyze and apply safe updates
  ./sentinel.py pipeline                        (deprecated: use 'gate')
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
    status_parser.add_argument(
        "--watch-summary", action="store_true", help="Single-line summary for status bars"
    )

    # init
    init_parser = subparsers.add_parser(
        "init",
        help="Scaffold traces from source file(s)",
        epilog=(
            "Examples:\n"
            "  %(prog)s src/module.py              # Scaffold one trace (interactive)\n"
            "  %(prog)s --scan                     # Scan repo, scaffold all traces\n"
            "  %(prog)s --scan --preset pytorch    # Use PyTorch-specific patterns\n"
            "  %(prog)s --scan --dry-run           # Preview without writing\n"
            "  %(prog)s --scan --adapter claude    # Also install Claude Code adapter\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    init_parser.add_argument("source", nargs="?", help="Source file to analyze (single-file mode)")
    init_parser.add_argument("--name", help="Trace name (default: source filename stem)")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing traces")
    init_parser.add_argument(
        "--scan", action="store_true", help="Scan repo and scaffold all trace-worthy files"
    )
    init_parser.add_argument(
        "--preset",
        default="default",
        help="Scan preset: default, pytorch, typescript, or custom name (default: default)",
    )
    init_parser.add_argument(
        "--adapter",
        choices=["claude", "codex"],
        help="Install agent adapter after scan",
    )
    init_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be created without writing"
    )

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
    verify_parser.add_argument("--affected-by", nargs="+", help="Filter to traces covering FILE(s)")

    # fix
    fix_parser = subparsers.add_parser(
        "fix", help="Auto-fix shifted anchors and bump verification state"
    )
    fix_parser.add_argument("--trace", help="Specific trace to fix")
    fix_parser.add_argument("--all", action="store_true", help="Fix all traces")
    fix_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be fixed without writing"
    )
    fix_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # gate
    gate_parser = subparsers.add_parser(
        "gate", help="Pre-commit gate: auto-fix + verify + test advisory"
    )
    gate_parser.add_argument("--no-fix", action="store_true", help="Skip auto-fix step")
    gate_parser.add_argument(
        "--files", nargs="*", help="Specific files to check (default: staged files)"
    )
    gate_parser.add_argument("--ci", action="store_true", help="CI mode (non-interactive)")
    gate_parser.add_argument(
        "--strict", action="store_true", help="Stale traces cause non-zero exit"
    )
    gate_parser.add_argument(
        "--no-strict", action="store_true", help="Stale traces are warning-only (overrides env)"
    )
    gate_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # pipeline (deprecated — use 'gate')
    pipeline_parser = subparsers.add_parser("pipeline", help="(deprecated: use 'gate')")
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

    # watch
    watch_parser = subparsers.add_parser("watch", help="Continuous file monitoring with auto-fix")
    watch_parser.add_argument(
        "--json-lines", action="store_true", help="JSON Lines event stream to stdout"
    )
    fix_group = watch_parser.add_mutually_exclusive_group()
    fix_group.add_argument(
        "--auto-fix",
        action="store_true",
        default=None,
        dest="auto_fix",
        help="Auto-fix mechanical drift (default)",
    )
    fix_group.add_argument(
        "--no-auto-fix",
        action="store_false",
        dest="auto_fix",
        help="Disable auto-fix",
    )
    watch_parser.add_argument("--traces", help="Comma-separated trace names to monitor")
    watch_parser.add_argument(
        "--daemon", action="store_true", help="Run as background daemon (Unix only)"
    )
    watch_parser.add_argument(
        "--poll-interval", type=float, help="Polling interval in seconds (fallback)"
    )
    watch_parser.add_argument("--debounce-ms", type=int, help="Debounce delay in milliseconds")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_SUCCESS

    # Bootstrap .sentinel/ for init --scan on fresh repos
    if args.command == "init" and getattr(args, "scan", False):
        repo_root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if repo_root_result.returncode == 0:
            repo_root = Path(repo_root_result.stdout.strip())
            sentinel_dir = args.sentinel_dir or (repo_root / ".sentinel")
            if not sentinel_dir.exists() or not (sentinel_dir / ".sentinel-meta.yaml").exists():
                print(f"Bootstrapping {sentinel_dir}/ ...")
                bootstrap_sentinel_dir(repo_root)

    try:
        sentinel = CodeSentinel(args.sentinel_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return EXIT_GENERAL_ERROR

    commands = {
        "status": cmd_status,
        "init": cmd_init,
        "verify": cmd_verify,
        "fix": cmd_fix,
        "gate": cmd_gate,
        "watch": cmd_watch,
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
