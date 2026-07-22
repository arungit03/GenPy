"""Evaluate a GenPy Code LLM checkpoint and produce benchmark artifacts."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.code_evaluation import (
    DEFAULT_CODE_PROMPTS,
    CodeCheckpointSummary,
    build_loss_history,
    discover_code_checkpoints,
    format_size,
    loss_history_from_training_metrics,
    perplexity_from_loss,
    read_training_metrics_csv,
    resolve_code_checkpoint,
    run_generation_benchmark,
    write_generation_examples,
    write_loss_curve_png,
    write_loss_history_csv,
)
from genpy_llm.code_generation import load_code_model_for_generation
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.code_training import (
    create_code_dataloader,
    evaluate_code_model,
    load_code_config,
    select_device,
    validate_code_training_artifacts,
)
from genpy_llm.losses import GPTCrossEntropyLoss
from genpy_llm.utils import set_seed

LOGGER = logging.getLogger("genpy_llm.evaluate_code_model")


def main() -> int:
    """Run checkpoint evaluation from the command line."""

    args = _parse_args()
    _configure_logging(args.debug)
    try:
        config = load_code_config(_resolve(args.config))
        if args.checkpoint_dir is not None:
            config = replace(
                config,
                checkpoint=replace(config.checkpoint, directory=_resolve(args.checkpoint_dir)),
            )
        validate_code_training_artifacts(config)
        set_seed(args.seed if args.seed is not None else config.seed)
        tokenizer = CodeTokenizer.from_file(config.tokenizer.path)
        device = select_device(args.device or config.training.device)
        if config.training.mixed_precision == "fp16" and device.type != "cuda":
            LOGGER.warning("fp16 requires CUDA; using full precision on this device.")
            config = replace(config, training=replace(config.training, mixed_precision="none"))
        checkpoint_path = resolve_code_checkpoint(
            args.checkpoint,
            checkpoint_directory=config.checkpoint.directory,
            filename_prefix=config.checkpoint.filename_prefix,
            best_filename=config.checkpoint.best_filename,
            project_root=PROJECT_ROOT,
        )
        model = load_code_model_for_generation(
            config=config,
            tokenizer=tokenizer,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        validation_loader = create_code_dataloader(
            config,
            tokenizer,
            split="validation",
            batch_size=args.batch_size or config.training.batch_size,
        )
        loss_fn = GPTCrossEntropyLoss(
            padding_idx=tokenizer.pad_token_id,
            ignore_padding=config.loss.ignore_padding,
            label_smoothing=config.loss.label_smoothing,
        )
        validation_loss = evaluate_code_model(
            model,
            validation_loader,
            loss_fn,
            device,
            config.training.mixed_precision,
            max_batches=(
                args.validation_batches
                if args.validation_batches is not None
                else config.training.validation_steps
            ),
            logger=LOGGER,
        )
        benchmark = run_generation_benchmark(
            model=model,
            tokenizer=tokenizer,
            prompts=DEFAULT_CODE_PROMPTS,
            device=device,
            max_new_tokens=(
                args.max_new_tokens
                if args.max_new_tokens is not None
                else config.generation.max_new_tokens
            ),
            temperature=(
                args.temperature
                if args.temperature is not None
                else config.generation.temperature
            ),
            top_k=args.top_k if args.top_k is not None else config.generation.top_k,
            top_p=args.top_p if args.top_p is not None else config.generation.top_p,
            repetition_penalty=(
                args.repetition_penalty
                if args.repetition_penalty is not None
                else config.generation.repetition_penalty
            ),
            do_sample=not args.greedy if args.greedy else config.generation.do_sample,
            stop_on_eos=config.generation.stop_on_eos,
            context_length=config.model.context_length,
        )
        evaluation_dir = _resolve(args.output_dir)
        generated_path = evaluation_dir / "generated_examples.txt"
        write_generation_examples(benchmark, generated_path, checkpoint_path=checkpoint_path)
        checkpoint_summary = discover_code_checkpoints(
            config.checkpoint.directory,
            filename_prefix=config.checkpoint.filename_prefix,
            best_filename=config.checkpoint.best_filename,
        )
        loss_rows = build_loss_history(checkpoint_summary)
        loss_history_path = evaluation_dir / "loss_history.csv"
        loss_curve_path = evaluation_dir / "loss_curve.png"
        write_loss_history_csv(loss_rows, loss_history_path)
        training_metric_rows = read_training_metrics_csv(evaluation_dir / "training_metrics.csv")
        curve_rows = (
            loss_history_from_training_metrics(training_metric_rows)
            if training_metric_rows
            else loss_rows
        )
        write_loss_curve_png(curve_rows, loss_curve_path)
    except Exception as exc:  # noqa: BLE001
        _report_error(exc, debug=args.debug)
        return 1

    print("GenPy Code Model Evaluation")
    print("===========================")
    print(
        _metrics_table(
            validation_loss,
            benchmark.average_generation_length,
            benchmark.tokens_per_second,
        )
    )
    print()
    print(_checkpoint_table(checkpoint_summary))
    print()
    print("Artifacts")
    print("---------")
    print(f"Generated examples: {generated_path}")
    print(f"Loss history: {loss_history_path}")
    print(f"Loss curve: {loss_curve_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GenPy code checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/code_small.yaml"))
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-batches", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _metrics_table(
    validation_loss: float,
    average_generation_length: float,
    generation_speed: float,
) -> str:
    perplexity = perplexity_from_loss(validation_loss)
    return _table(
        ("Metric", "Value"),
        (
            ("Validation loss", f"{validation_loss:.6f}"),
            (
                "Perplexity",
                "inf"
                if perplexity is None or math.isinf(perplexity)
                else f"{perplexity:.4f}",
            ),
            ("Average generation length", f"{average_generation_length:.2f} tokens"),
            ("Generation speed", f"{generation_speed:.2f} tokens/sec"),
        ),
    )


def _checkpoint_table(summary: CodeCheckpointSummary) -> str:
    rows = [
        (
            "Latest",
            "-" if summary.latest_checkpoint is None else summary.latest_checkpoint.path.name,
            "-"
            if summary.latest_checkpoint is None
            else _step(summary.latest_checkpoint.global_step),
            "-"
            if summary.latest_checkpoint is None
            else format_size(summary.latest_checkpoint.size_bytes),
        ),
        (
            "Best",
            "-" if summary.best_checkpoint is None else summary.best_checkpoint.path.name,
            "-" if summary.best_checkpoint is None else _step(summary.best_checkpoint.global_step),
            "-"
            if summary.best_checkpoint is None
            else format_size(summary.best_checkpoint.size_bytes),
        ),
        (
            "Total",
            f"{summary.total_checkpoints} checkpoint(s)",
            "-",
            format_size(summary.total_size_bytes),
        ),
    ]
    rows.extend(
        (
            "File",
            checkpoint.path.name,
            _step(checkpoint.global_step),
            format_size(checkpoint.size_bytes),
        )
        for checkpoint in summary.checkpoints
    )
    return _table(("Type", "Checkpoint", "Step", "Size"), tuple(rows))


def _table(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _step(step: int | None) -> str:
    return "-" if step is None else str(step)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _report_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        LOGGER.exception("Code model evaluation failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
