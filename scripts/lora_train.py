#!/usr/bin/env python3
"""Train Phase 9 LoRA adapters for GenPy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.logging_utils import setup_structured_logging
from genpy_llm.lora_training import (
    Phase9LoRATrainer,
    load_phase9_config,
    override_phase9_config,
)


def main() -> int:
    """Load Phase 9 configuration and run LoRA-only training."""

    args = _parse_args()
    try:
        config = load_phase9_config(_resolve(args.config))
        config = override_phase9_config(
            config,
            device=args.device,
            max_steps=args.max_steps,
            base_checkpoint=args.base_checkpoint,
            resume_from=args.resume_from,
        )
        setup_structured_logging(config.outputs.log_file, config.outputs.log_level)
        result = Phase9LoRATrainer(config).train()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    stats = result.parameter_stats
    print("GenPy Phase 9 LoRA training complete")
    print(f"Global step: {result.global_step}")
    print(f"Latest loss: {result.latest_loss}")
    print(f"Trainable adapter parameters: {stats.trainable_parameters:,}")
    print(f"Frozen parameters: {stats.frozen_parameters:,}")
    print(f"Trainable percentage: {stats.trainable_percentage:.4f}%")
    print(f"Adapters: {stats.adapter_count}")
    print(f"Last adapter: {result.last_adapter}")
    print(f"Best adapter: {result.best_adapter}")
    print(f"Metrics: {result.metrics_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GenPy LoRA attention adapters.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora.yaml"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--max-steps", type=_positive_int, default=None)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
