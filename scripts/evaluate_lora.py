#!/usr/bin/env python3
"""Compare a Phase 9 LoRA adapter with full fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.device import select_device
from genpy_llm.logging_utils import setup_structured_logging
from genpy_llm.lora_evaluation import evaluate_full_vs_lora
from genpy_llm.lora_training import load_phase9_config


def main() -> int:
    """Run an apples-to-apples full-fine-tuning versus LoRA comparison."""

    args = _parse_args()
    try:
        config = load_phase9_config(_resolve(args.config))
        setup_structured_logging(config.outputs.log_file, config.outputs.log_level)
        adapter_path = _adapter_path(config, args.adapter)
        output_dir = _resolve(args.output_dir) if args.output_dir is not None else None
        result = evaluate_full_vs_lora(
            config,
            adapter_path=adapter_path,
            device=select_device(args.device),
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            validation_batches=args.validation_batches,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("GenPy Phase 9 comparison complete")
    print(f"Full validation loss: {result.full_fine_tuning.validation_loss}")
    print(f"LoRA validation loss: {result.lora.validation_loss}")
    print(f"Full checkpoint bytes: {result.full_fine_tuning.checkpoint_size_bytes:,}")
    print(f"LoRA adapter bytes: {result.lora.checkpoint_size_bytes:,}")
    print(f"Report: {result.comparison_report}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GenPy full fine-tuning and LoRA.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora.yaml"))
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--validation-batches", type=int, default=None)
    return parser.parse_args()


def _adapter_path(config, configured: Path | None) -> Path:
    if configured is not None:
        path = _resolve(configured)
        if not path.is_file():
            raise FileNotFoundError(f"LoRA adapter not found: {path}")
        return path
    best = config.checkpoints.output_dir / config.checkpoints.best_filename
    if best.is_file():
        return best
    latest = config.checkpoints.output_dir / config.checkpoints.adapter_filename
    if not latest.is_file():
        raise FileNotFoundError(f"No LoRA adapter found in {config.checkpoints.output_dir}")
    return latest


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
