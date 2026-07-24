#!/usr/bin/env python3
"""Build Phase 6.2 Corpus V2 artifacts without starting training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genpy_llm.corpus_v2.pipeline import run_corpus_v2_cli


if __name__ == "__main__":
    raise SystemExit(run_corpus_v2_cli())
