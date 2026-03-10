#!/usr/bin/env python3
"""Code Sentinel — commit-anchored verification framework.

Thin re-export facade. All logic lives in sibling modules under .sentinel/.
The CLI entry point and importlib-based test loading pattern are preserved:
    python3 .sentinel/sentinel.py [command]
    mod = importlib.util.spec_from_file_location(name, ".sentinel/sentinel.py")
"""

from __future__ import annotations

import os
import sys

# Phase 3 (pip install code-sentinel) replaces this with proper package
# imports via __init__.py + relative imports, eliminating the sys.path hack.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# --- Re-exports: every public name that tests or external callers use ---

# constants (leaf)
# adapters

# commands
from commands import (  # noqa: E402
    main,
)

# core

# fix

# models

# output

# watch

if __name__ == "__main__":
    sys.exit(main())
