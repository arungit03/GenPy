#!/usr/bin/env python3
"""Resume Phase 6.3 continued pretraining from the configured checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genpy_llm.continued_training import run_phase63_cli


if __name__ == "__main__":
    raise SystemExit(run_phase63_cli())
