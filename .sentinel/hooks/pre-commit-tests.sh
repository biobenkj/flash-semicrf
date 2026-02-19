#!/bin/bash
# Standalone script: Suggest/run tests for modified files
# Advisory mode: prints suggested tests
# CI mode (SENTINEL_RUN_TESTS=1): actually runs them
#
# NOTE: The main pre-commit hook runs ./sentinel.py pipeline which includes
# test advisory. This script is kept for standalone use or granular control.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SENTINEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

modified_files="$@"

# Parse test_bindings from .sentinel-meta.yaml using Python
suggested_tests=$(python3 -c "
import yaml
from pathlib import Path
import sys

meta_path = Path('$SENTINEL_DIR/.sentinel-meta.yaml')
if not meta_path.exists():
    sys.exit(0)

meta = yaml.safe_load(meta_path.read_text())
bindings = meta.get('test_bindings', {})
modified = sys.argv[1:]
tests = set()
for f in modified:
    for pattern, test_list in bindings.items():
        if pattern in f:
            tests.update(test_list)
print(' '.join(sorted(tests)))
" $modified_files)

if [[ -z "$suggested_tests" ]]; then
  echo "No sentinel-bound tests for modified files."
  exit 0
fi

echo "=== Sentinel Test Advisory ==="
echo "The following tests cover modified files:"
echo "  pytest $suggested_tests"
echo ""

if [[ "$SENTINEL_RUN_TESTS" == "1" ]]; then
  echo "Running tests..."
  pytest $suggested_tests
else
  echo "Run with: SENTINEL_RUN_TESTS=1 git commit ..."
fi
