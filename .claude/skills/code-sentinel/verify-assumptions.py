#!/usr/bin/env python3
"""Verify mechanically-checkable assumptions for a trace."""
import re
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
TRACES_DIR = SCRIPT_DIR / "traces"
ANCHORS_DIR = SCRIPT_DIR / "anchors"


def parse_assumptions(trace_path: Path) -> list[dict]:
    """Parse Mechanically Verified assumptions table from trace markdown."""
    content = trace_path.read_text()

    # Find the Mechanically Verified section - it starts with the header and ends at the next section
    # Look for "### Mechanically Verified" followed by the table, ending before the next "###" or "##"
    match = re.search(
        r"### Mechanically Verified\n\n"
        r".*?\n\n"  # Description text
        r"\| ID \| Assumption \| Verification \|\n"
        r"\|[-|]+\|\n"
        r"((?:\| [A-Z][0-9]+ \|[^\n]+\|\n)+)",  # Only match rows starting with | A1 | or | D1 | etc
        content,
        re.DOTALL,
    )
    if not match:
        return []

    assumptions = []
    for line in match.group(1).strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 3 and re.match(r"[A-Z]\d+", parts[0]):
            assumptions.append({"id": parts[0], "assumption": parts[1], "verification": parts[2]})

    return assumptions


def verify_assumption(assumption: dict, anchors: dict, trace_name: str) -> tuple[bool, str]:
    """Verify a single assumption. Returns (passed, message)."""
    verification = assumption["verification"]

    if verification.startswith("anchor:"):
        anchor_name = verification.split(":")[1].strip()
        trace_anchors = anchors.get(trace_name, {})
        if anchor_name not in trace_anchors:
            return False, f"Anchor {anchor_name} not defined in anchors.yaml"

        spec = trace_anchors[anchor_name]
        cmd = [
            str(ANCHORS_DIR / "verify-anchor.sh"),
            spec["pattern"],
            str(spec["expected_line"]),
            str(REPO_ROOT / spec["file"]),
            str(spec.get("drift_tolerance", 20)),
        ]
        if spec.get("after"):
            cmd.append(spec["after"])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        return result.returncode == 0, result.stdout.strip()

    elif verification.startswith("`") and verification.endswith("`"):
        # Shell command verification
        cmd = verification[1:-1]
        result = subprocess.run(cmd, shell=True, capture_output=True, cwd=REPO_ROOT)
        return result.returncode == 0, (
            "Command passed" if result.returncode == 0 else "Command failed"
        )

    return True, "Manual verification required"


def main(trace_name: str) -> int:
    trace_path = TRACES_DIR / f"{trace_name}.md"
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}")
        return 1

    anchors_path = ANCHORS_DIR / "anchors.yaml"
    if not anchors_path.exists():
        print(f"Anchors file not found: {anchors_path}")
        return 1

    anchors = yaml.safe_load(anchors_path.read_text())
    assumptions = parse_assumptions(trace_path)

    if not assumptions:
        print(f"No mechanically verified assumptions found in {trace_name}")
        return 0

    print(f"=== Assumption Verification: {trace_name} ===\n")

    failed = 0
    for assumption in assumptions:
        passed, msg = verify_assumption(assumption, anchors, trace_name)
        status = "✓" if passed else "✗"
        print(f"  {status} {assumption['id']}: {assumption['assumption']}")
        print(f"      {msg}")
        if not passed:
            failed += 1

    print(f"\nResult: {'VERIFIED' if failed == 0 else 'DEGRADED'}")
    if failed > 0:
        print(f"Failed assumptions: {failed}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify-assumptions.py <trace-name>")
        print("\nAvailable traces:")
        for trace in sorted(TRACES_DIR.glob("*.md")):
            print(f"  {trace.stem}")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
