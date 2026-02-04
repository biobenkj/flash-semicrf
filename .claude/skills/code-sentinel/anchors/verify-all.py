#!/usr/bin/env python3
"""Verify all anchors in anchors.yaml"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
SENTINEL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent


def verify_anchor(
    pattern: str, expected_line: int, file_path: str, tolerance: int = 20, after: str | None = None
) -> tuple[int, str]:
    """Run verify-anchor.sh and return (exit_code, message)."""
    cmd = [
        str(SCRIPT_DIR / "verify-anchor.sh"),
        pattern,
        str(expected_line),
        file_path,
        str(tolerance),
    ]
    if after:
        cmd.append(after)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return result.returncode, result.stdout.strip()


def check_consistency() -> list[str]:
    """Validate consistency between .sentinel-meta.yaml, anchors.yaml, and trace headers."""
    errors = []

    meta_path = SENTINEL_DIR / ".sentinel-meta.yaml"
    anchors_path = SCRIPT_DIR / "anchors.yaml"
    traces_dir = SENTINEL_DIR / "traces"

    if not meta_path.exists():
        return ["Missing .sentinel-meta.yaml"]

    meta = yaml.safe_load(meta_path.read_text())
    anchors = yaml.safe_load(anchors_path.read_text())

    for trace_name, trace_meta in meta.get("sentinels", {}).items():
        # Check anchors exist in anchors.yaml
        for anchor_id in trace_meta.get("anchors", []):
            if trace_name not in anchors or anchor_id not in anchors.get(trace_name, {}):
                errors.append(f"{trace_name}: Anchor '{anchor_id}' in meta but not in anchors.yaml")

        # Check verified_commit matches trace header
        trace_path = traces_dir / f"{trace_name}.md"
        if trace_path.exists():
            trace_content = trace_path.read_text()
            match = re.search(r"\*\*Verified against:\*\*.*?@ commit `([a-f0-9]+)`", trace_content)
            if match:
                trace_commit = match.group(1)
                meta_commit = trace_meta.get("verified_commit", "")
                if trace_commit != meta_commit:
                    errors.append(
                        f"{trace_name}: Commit mismatch - meta has '{meta_commit}', trace has '{trace_commit}'"
                    )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify sentinel anchors")
    parser.add_argument("trace", nargs="?", help="Filter to specific trace")
    parser.add_argument(
        "--check-consistency", action="store_true", help="Validate meta/anchor/trace consistency"
    )
    args = parser.parse_args()

    # Consistency check if requested
    if args.check_consistency:
        print("=== Consistency Check ===\n")
        errors = check_consistency()
        if errors:
            for err in errors:
                print(f"  [FAIL] {err}")
            print(f"\nConsistency: {len(errors)} errors found")
        else:
            print("  [PASS] All metadata consistent")
        print()

    # Anchor verification
    anchors_path = SCRIPT_DIR / "anchors.yaml"
    if not anchors_path.exists():
        print(f"Error: anchors.yaml not found at {anchors_path}")
        return 1

    anchors = yaml.safe_load(anchors_path.read_text())

    failed = 0
    verified = 0
    total = 0

    print("=== Anchor Verification Report ===\n")

    for trace_name, trace_anchors in anchors.items():
        if args.trace and trace_name != args.trace:
            continue

        print(f"[{trace_name}]")
        for anchor_name, spec in trace_anchors.items():
            total += 1
            code, msg = verify_anchor(
                spec["pattern"],
                spec["expected_line"],
                str(REPO_ROOT / spec["file"]),
                spec.get("drift_tolerance", 20),
                spec.get("after"),
            )
            status = "[PASS]" if code == 0 else "[FAIL]"
            print(f"  {status} {anchor_name}: {msg}")
            if code == 0:
                verified += 1
            else:
                failed += 1
        print()

    print(f"Summary: {verified}/{total} verified, {failed} failed")

    if args.check_consistency and check_consistency():
        return 1
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
