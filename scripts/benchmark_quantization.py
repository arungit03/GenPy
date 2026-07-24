#!/usr/bin/env python3
"""Run Phase 10 quantization benchmarks for GenPy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.device import select_device
from genpy_llm.evaluation_benchmark import resolve_evaluation_checkpoint
from genpy_llm.quantization_benchmark import (
    benchmark_quantization,
    load_phase10_config,
    write_quantization_artifacts,
)


def main() -> int:
    """Benchmark configured quantized checkpoints."""

    args = _parse_args()
    try:
        config = load_phase10_config(_resolve(args.config))
        checkpoint = resolve_evaluation_checkpoint(
            args.checkpoint or config.source_checkpoint,
            config=config.phase7,
        )
        device = select_device(args.device or config.device)
        summary = benchmark_quantization(config, checkpoint_path=checkpoint, device=device)
        output_dir = _resolve(args.output_dir) if args.output_dir else config.evaluation_output_dir
        artifacts = write_quantization_artifacts(summary, output_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("GenPy Phase 10 quantization benchmark complete")
    print(f"Source checkpoint: {summary.source_checkpoint}")
    print(f"Device: {summary.device}")
    for result in summary.results:
        speed = "N/A" if result.tokens_per_second is None else f"{result.tokens_per_second:.3f}"
        loss = "N/A" if result.validation_loss is None else f"{result.validation_loss:.6f}"
        print(
            f"{result.method}: {result.status}, "
            f"size={result.checkpoint_size_bytes:,} bytes, "
            f"speed={speed} tokens/sec, loss={loss}"
        )
    print(f"JSON: {artifacts.json_path}")
    print(f"CSV: {artifacts.csv_path}")
    print(f"Report: {artifacts.report_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GenPy quantization methods.")
    parser.add_argument("--config", type=Path, default=Path("configs/quantization.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
