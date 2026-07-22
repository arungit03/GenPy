#!/usr/bin/env python3
"""Build the final merged GenPy pretraining corpus."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genpy_llm.corpus_merger import run_pretraining_corpus_cli


if __name__ == "__main__":
    raise SystemExit(run_pretraining_corpus_cli())
