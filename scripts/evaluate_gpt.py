#!/usr/bin/env python3
"""Run Phase 8 evaluation and benchmarking for a fine-tuned GenPy checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.device import select_device
from genpy_llm.evaluation_benchmark import (
    DEFAULT_EVALUATION_DATASET,
    build_evaluation_summary,
    calculate_validation_metrics,
    evaluate_prompts,
    load_evaluation_prompts,
    load_model_for_evaluation,
    resolve_evaluation_checkpoint,
    write_evaluation_artifacts,
)
from genpy_llm.fine_tuning import load_phase7_config
from genpy_llm.logging_utils import setup_structured_logging

LOGGER = logging.getLogger("genpy_llm.evaluate_gpt")


def main() -> int:
    """Evaluate the selected or latest Phase 7 checkpoint."""

    args = _parse_args()
    output_dir = _resolve(args.output_dir)
    try:
        config = load_phase7_config(_resolve(args.config))
        setup_structured_logging(output_dir / "evaluation.log", config.log_level)
        checkpoint_path = resolve_evaluation_checkpoint(args.checkpoint, config=config)
        device = select_device(args.device)
        prompts = load_evaluation_prompts(_resolve(args.dataset))
        LOGGER.info("Loading checkpoint %s on %s", checkpoint_path, device)
        model, tokenizer, loaded = load_model_for_evaluation(config, checkpoint_path, device)
        LOGGER.info("Calculating validation loss and perplexity")
        validation = calculate_validation_metrics(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            max_batches=args.validation_batches,
        )
        LOGGER.info("Running inference on %d prompts", len(prompts))
        results = evaluate_prompts(
            model=model,
            tokenizer=tokenizer,
            config=config,
            prompts=prompts,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        summary = build_evaluation_summary(
            checkpoint_path=checkpoint_path,
            loaded_checkpoint=loaded,
            device=device,
            validation=validation,
            results=results,
        )
        artifacts = write_evaluation_artifacts(summary, output_dir)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Evaluation failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    validation_loss = "N/A" if validation is None else f"{validation.loss:.6f}"
    perplexity = "N/A" if validation is None else f"{validation.perplexity:.6f}"
    passed = sum(result.passed is True for result in results)
    print("GenPy Phase 8 evaluation complete")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompts: {len(results)}")
    print(f"Validation loss: {validation_loss}")
    print(f"Perplexity: {perplexity}")
    print(f"Generation speed: {summary.tokens_per_second:.3f} tokens/sec")
    print(f"Automatic checks passed: {passed}/{len(results)}")
    print(f"JSON: {artifacts.json_path}")
    print(f"CSV: {artifacts.csv_path}")
    print(f"Report: {artifacts.report_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned GenPy checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint path or 'latest' (default: latest).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
        help="Directory for JSON, CSV, and Markdown results.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/finetuning.yaml"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_EVALUATION_DATASET)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the Phase 7 generation length.",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=None,
        help="Override the Phase 7 validation batch limit.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
