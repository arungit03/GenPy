#!/usr/bin/env python3
"""Run the Phase 6.3 previous-vs-continued benchmark comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genpy_llm.benchmark_monitor import benchmark_phase63_checkpoints
from genpy_llm.continued_training import _phase6_config, load_phase63_config
from genpy_llm.device import select_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Phase 6.3 checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase6_3.yaml"))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--continued", type=Path)
    args = parser.parse_args()
    config = load_phase63_config(args.config)
    previous = args.previous or config.training.source_checkpoint
    if previous is None:
        previous = config.paths.checkpoint_search_dir / "last_checkpoint.pt"
    continued = args.continued or config.paths.checkpoint_output_dir / "last_checkpoint.pt"
    phase6 = _phase6_config(config, Path(previous))
    device = select_device(config.training.device)
    benchmark_phase63_checkpoints(
        config=phase6,
        previous_checkpoint=Path(previous),
        continued_checkpoint=Path(continued),
        output_dir=config.paths.report_dir,
        device=device,
        settings=config.benchmark,
    )
    print(f"Comparison JSON: {config.paths.report_dir / 'comparison_report.json'}")
    print(f"Comparison report: {config.paths.report_dir / 'comparison_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
