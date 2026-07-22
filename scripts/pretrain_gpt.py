#!/usr/bin/env python3
"""Run Phase 6 GenPy GPT pretraining."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genpy_llm.pretraining import run_phase6_pretraining


if __name__ == "__main__":
    raise SystemExit(run_phase6_pretraining())
