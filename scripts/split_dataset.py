"""Create deterministic source-grouped train, validation, and test splits."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.python_dataset_pipeline import run_stage_cli


if __name__ == "__main__":
    raise SystemExit(run_stage_cli("split"))
