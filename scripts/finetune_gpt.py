#!/usr/bin/env python3
"""Run Phase 7 supervised instruction fine-tuning."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.fine_tuning import Phase7Trainer, load_phase7_config  # noqa: E402
from genpy_llm.logging_utils import setup_structured_logging  # noqa: E402


def main() -> int:
    args = _parse_args()
    try:
        config = load_phase7_config(args.config)
        config = _override_config(config, args)
        setup_structured_logging(config.outputs.log_file, config.log_level)
        result = Phase7Trainer(config).train()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Phase 7 fine-tuning complete")
    print(f"Global step: {result.global_step}")
    print(f"Latest loss: {result.latest_loss}")
    print(f"Last checkpoint: {result.last_checkpoint}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Latest sample: {result.latest_sample_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GenPy as a Python assistant.")
    parser.add_argument("--config", type=Path, default=Path("configs/finetuning.yaml"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def _override_config(config, args: argparse.Namespace):
    training = config.training
    checkpoint = config.checkpoint
    if args.device is not None:
        training = replace(training, device=args.device)
    if args.resume:
        training = replace(training, resume=True)
    if args.max_steps is not None:
        training = replace(training, max_steps=args.max_steps)
    if args.checkpoint is not None:
        base_checkpoint = (
            args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
        )
        checkpoint = replace(
            checkpoint,
            base_checkpoint=base_checkpoint,
        )
    return replace(config, training=training, checkpoint=checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
