"""Build the Phase 5.5A GitHub Python corpus and binary token shards."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.github_corpus_builder import run_github_corpus_cli


if __name__ == "__main__":
    raise SystemExit(run_github_corpus_cli())
