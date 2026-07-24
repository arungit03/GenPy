#!/usr/bin/env python3
"""Benchmark the base vs continued-pretraining GenPy checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    from genpy_llm.benchmark_suite import run_benchmark_cli

    raise SystemExit(run_benchmark_cli())
