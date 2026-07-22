"""Supervised fine-tune a trained GenPy GPT checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.checkpointing import CheckpointError
from genpy_llm.config import ConfigError, load_config
from genpy_llm.device import select_device
from genpy_llm.fine_tuning import (
    FineTuningError,
    configure_trainable_parameters,
    load_base_model_for_fine_tuning,
    prepare_fine_tuning_dataset,
    run_fine_tuning,
)
from genpy_llm.gpt import GPTModelError
from genpy_llm.logging_utils import setup_logging
from genpy_llm.performance import PerformanceError
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.utils import set_seed
from genpy_llm.vocabulary import Vocabulary, VocabularyError


def main() -> int:
    """Parse arguments and run a bounded or configured fine-tuning job."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        app_config = load_config(_resolve_optional_path(args.config))
        setup_logging(
            log_dir=app_config.paths.logs_dir,
            log_file=app_config.logging.log_file,
            level="DEBUG" if args.debug else app_config.logging.level,
        )
        fine_tuning_config = _fine_tuning_config_from_args(args, app_config.fine_tuning)
        set_seed(args.seed if args.seed is not None else fine_tuning_config.seed)
        device = select_device(args.device or app_config.training.device)
        mixed_precision = args.mixed_precision or app_config.optimization.mixed_precision
        compile_enabled = args.compile or app_config.optimization.torch_compile
        compile_mode = args.compile_mode or app_config.optimization.compile_mode
        gradient_checkpointing = (
            args.gradient_checkpointing or app_config.optimization.gradient_checkpointing
        )
        base_checkpoint_path = _resolve_path(args.base_checkpoint)
        dataset_path = (
            _resolve_path(args.dataset)
            if args.dataset is not None
            else fine_tuning_config.dataset_file
        )
        output_directory = (
            _resolve_path(args.output_directory)
            if args.output_directory is not None
            else fine_tuning_config.output_directory
        )
        fine_tuning_config = replace(
            fine_tuning_config,
            dataset_file=dataset_path,
            output_directory=output_directory,
        )
        tokenizer = TextTokenizer(app_config.tokenization)
        vocabulary = Vocabulary.load(
            app_config.data.vocabulary_file,
            encoding=app_config.data.encoding,
        )
        train_dataset, validation_dataset, dataset_stats = prepare_fine_tuning_dataset(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            context_length=app_config.model.context_length,
            train_validation_ratio=fine_tuning_config.train_validation_ratio,
            seed=fine_tuning_config.seed,
        )
        model = load_base_model_for_fine_tuning(
            base_checkpoint_path=base_checkpoint_path,
            app_config=app_config,
            device=device,
        )
        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
        parameter_stats = configure_trainable_parameters(
            model,
            freeze_embeddings=fine_tuning_config.freeze_embeddings,
            freeze_first_n_layers=fine_tuning_config.freeze_first_n_layers,
        )
        resume_path = _resume_path(output_directory) if args.resume else None
        result = run_fine_tuning(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            vocabulary_path=app_config.data.vocabulary_file,
            app_config=app_config,
            fine_tuning_config=fine_tuning_config,
            output_directory=output_directory,
            base_checkpoint_path=base_checkpoint_path,
            dataset_path=dataset_path,
            device=device,
            max_batches=args.max_batches,
            resume_checkpoint_path=resume_path,
            parameter_stats=parameter_stats,
            mixed_precision=mixed_precision,
            torch_compile=compile_enabled,
            compile_mode=compile_mode,
        )
    except (
        CheckpointError,
        ConfigError,
        FileNotFoundError,
        FineTuningError,
        GPTModelError,
        IsADirectoryError,
        OSError,
        PerformanceError,
        RuntimeError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    latest = result.epochs[-1]
    print("GenPy LLM Fine-Tuning")
    print("=====================")
    print(f"Base checkpoint: {base_checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Source records: {dataset_stats.source_records}")
    print(f"Usable records: {dataset_stats.usable_records}")
    print(f"Train examples: {dataset_stats.train_examples}")
    print(f"Validation examples: {dataset_stats.validation_examples}")
    print(f"Trainable parameters: {parameter_stats.trainable_parameter_count}")
    print(f"Frozen parameters: {parameter_stats.frozen_parameter_count}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"torch.compile: {compile_enabled} ({compile_mode})")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"Training loss: {latest.training_loss:.6f}")
    if latest.validation_loss is not None:
        print(f"Validation loss: {latest.validation_loss:.6f}")
    print(f"Global step: {result.global_step}")
    print(f"Latest checkpoint: {result.latest_checkpoint_path}")
    print(f"Best checkpoint: {result.best_checkpoint_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GenPy GPT from a base checkpoint.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--learning-rate", type=_positive_float, default=None)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--freeze-first-n-layers", type=_non_negative_int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--max-batches", type=_positive_int, default=None)
    parser.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _fine_tuning_config_from_args(args: argparse.Namespace, config):
    return replace(
        config,
        epochs=args.epochs if args.epochs is not None else config.epochs,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        learning_rate=args.learning_rate
        if args.learning_rate is not None
        else config.learning_rate,
        freeze_embeddings=args.freeze_embeddings or config.freeze_embeddings,
        freeze_first_n_layers=args.freeze_first_n_layers
        if args.freeze_first_n_layers is not None
        else config.freeze_first_n_layers,
        seed=args.seed if args.seed is not None else config.seed,
    )


def _resume_path(output_directory: Path) -> Path:
    best_path = output_directory / "genpy_ft_best.pt"
    if best_path.exists():
        return best_path
    checkpoints = sorted(output_directory.glob("genpy_ft_epoch_*.pt"))
    if not checkpoints:
        raise FineTuningError(f"No fine-tuning checkpoint found in {output_directory}.")
    return checkpoints[-1]


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to zero.")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return _resolve_path(path)


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Fine-tuning failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
