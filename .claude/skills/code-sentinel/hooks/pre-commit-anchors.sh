#!/bin/bash
# Pre-commit hook: Run full sentinel pipeline
# This hook is called by pre-commit framework on commit
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SENTINEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run the sentinel pipeline
# --ci flag for non-interactive mode
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 "$SENTINEL_DIR/sentinel.py" pipeline --ci
