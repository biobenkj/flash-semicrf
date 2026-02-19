#!/usr/bin/env python3
"""Backward-compat shim — delegates to .sentinel/sentinel.py

Sunset: Remove after v0.3.0 or 2026-04-01, whichever comes first.
"""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent.parent
sys.exit(subprocess.call([sys.executable, str(root / ".sentinel" / "sentinel.py")] + sys.argv[1:]))
