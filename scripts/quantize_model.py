#!/usr/bin/env python3
"""Create Phase 10 quantized GenPy checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.evaluation_benchmark import resolve_evaluation_checkpoint
from genpy_llm.quantization_benchmark import (
    load_phase10_config,
    quantize_checkpoint_variants,
)


def main() -> int:
    """Create configured quantized checkpoints from a source checkpoint."""

    args = _parse_args()
    try:
        config = load_phase10_config(_resolve(args.config))
        checkpoint = resolve_evaluation_checkpoint(
            args.checkpoint or config.source_checkpoint,
            config=config.phase7,
        )
        methods = tuple(args.methods) if args.methods else config.methods
        outputs = quantize_checkpoint_variants(
            config,
            checkpoint_path=checkpoint,
            methods=methods,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("GenPy Phase 10 quantization complete")
    print(f"Source checkpoint: {checkpoint}")
    for method, path in outputs.items():
        print(f"{method}: {path} ({path.stat().st_size:,} bytes)")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GenPy quantized checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/quantization.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["fp16", "bf16", "dynamic_int8", "int8"],
        default=None,
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
