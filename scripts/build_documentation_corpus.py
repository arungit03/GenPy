#!/usr/bin/env python3
"""Build the local GenPy documentation corpus."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    from genpy_llm.documentation_corpus import run_documentation_corpus_cli

    raise SystemExit(run_documentation_corpus_cli())
