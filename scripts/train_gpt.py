"""Train the GPT model with checkpoint saving and resume support."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.checkpointing import CheckpointError, find_latest_checkpoint, load_checkpoint
from genpy_llm.config import ConfigError, load_config
from genpy_llm.dataset import DatasetPreparationError, load_dataset_split
from genpy_llm.device import select_device
from genpy_llm.gpt import GPTModelError, create_gpt_model
from genpy_llm.logging_utils import setup_logging
from genpy_llm.losses import LossError, create_loss_function
from genpy_llm.optimizers import OptimizerError, create_optimizer
from genpy_llm.performance import PerformanceError, compile_model
from genpy_llm.training import GPTTrainer, TrainingError
from genpy_llm.utils import set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Train GPT for a small or configured run."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        app_config = load_config(_resolve_optional_path(args.config))
        logger = setup_logging(
            log_dir=app_config.paths.logs_dir,
            log_file=app_config.logging.log_file,
            level="DEBUG" if args.debug else app_config.logging.level,
        )
        set_seed(args.seed if args.seed is not None else app_config.training.seed)
        device = select_device(args.device or app_config.training.device)
        mixed_precision = args.mixed_precision or app_config.optimization.mixed_precision
        compile_enabled = args.compile or app_config.optimization.torch_compile
        compile_mode = args.compile_mode or app_config.optimization.compile_mode
        gradient_checkpointing = (
            args.gradient_checkpointing or app_config.optimization.gradient_checkpointing
        )

        model, metadata = create_gpt_model(app_config.data.vocabulary_file, app_config)
        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
        loss_fn = create_loss_function(app_config.data.vocabulary_file, app_config.loss)
        optimizer = create_optimizer(model, app_config.optimizer)
        model = compile_model(model, enabled=compile_enabled, mode=compile_mode)
        trainer = GPTTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else app_config.training.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm
            if args.max_grad_norm is not None
            else app_config.training.max_grad_norm,
            mixed_precision=mixed_precision,
        )

        checkpoint_directory = (
            _resolve_optional_path(args.checkpoint_dir)
            if args.checkpoint_dir is not None
            else app_config.checkpoint.directory
        )
        resume_path = _resolve_resume_path(args, checkpoint_directory)
        start_epoch = 1
        best_metric = None
        if resume_path is not None:
            loaded = load_checkpoint(
                resume_path,
                trainer.model,
                optimizer=trainer.optimizer,
                scaler=trainer.scaler,
            )
            trainer.total_optimizer_steps = loaded.global_step
            start_epoch = loaded.epoch + 1
            best_metric = loaded.best_metric
            logger.info(
                "Resumed checkpoint %s at epoch=%s global_step=%s.",
                loaded.checkpoint_path,
                loaded.epoch,
                loaded.global_step,
            )

        train_dataset = load_dataset_split(app_config.data.train_dataset_file)
        validation_dataset = load_dataset_split(app_config.data.validation_dataset_file)
        train_loader = _limited_loader(
            DataLoader(train_dataset, batch_size=app_config.dataset.batch_size, shuffle=False),
            args.max_batches,
        )
        validation_loader = _limited_loader(
            DataLoader(
                validation_dataset,
                batch_size=app_config.dataset.batch_size,
                shuffle=False,
            ),
            args.max_batches,
        )

        result = trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=args.epochs if args.epochs is not None else app_config.training.epochs,
            validate_every_epochs=app_config.training.validate_every_epochs,
            log_every_steps=app_config.training.log_every_steps,
            start_epoch=start_epoch,
            checkpoint_config=None if args.no_checkpoint else app_config.checkpoint,
            checkpoint_directory=None if args.no_checkpoint else checkpoint_directory,
            model_config=asdict(app_config.model),
            vocabulary_metadata={
                "vocabulary_size": metadata.vocab_size,
                "vocabulary_path": str(app_config.data.vocabulary_file),
            },
            best_metric=best_metric,
        )
    except (
        CheckpointError,
        ConfigError,
        DatasetPreparationError,
        FileNotFoundError,
        GPTModelError,
        IsADirectoryError,
        LossError,
        OSError,
        OptimizerError,
        PerformanceError,
        RuntimeError,
        TrainingError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    latest = result.epochs[-1]
    print("GenPy LLM Training")
    print("==================")
    print(f"Vocabulary size: {metadata.vocab_size}")
    print(f"Start epoch: {start_epoch}")
    print(f"Completed epochs: {result.completed_epochs}")
    print(f"Last epoch: {latest.epoch}")
    print(f"Training loss: {latest.training_loss:.6f}")
    if latest.validation_loss is not None:
        print(f"Validation loss: {latest.validation_loss:.6f}")
    print(f"Global step: {result.total_optimizer_steps}")
    print(f"Best metric: {result.best_metric}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"torch.compile: {compile_enabled} ({compile_mode})")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    if result.checkpoint_paths:
        print(f"Latest saved checkpoint: {result.checkpoint_paths[-1]}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT with checkpoint support.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=_positive_int, default=None)
    parser.add_argument("--max-batches", type=_positive_int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=None)
    parser.add_argument("--max-grad-norm", type=_positive_float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve_resume_path(args: argparse.Namespace, checkpoint_directory: Path) -> Path | None:
    if args.resume_from is not None:
        return _resolve_optional_path(args.resume_from)
    if not args.resume:
        return None
    latest = find_latest_checkpoint(checkpoint_directory)
    if latest is None:
        raise CheckpointError(f"No managed checkpoint found in {checkpoint_directory}.")
    return latest


def _limited_loader(
    loader: DataLoader,
    max_batches: int | None,
) -> DataLoader | list[dict[str, torch.Tensor]]:
    if max_batches is None:
        return loader
    return list(islice(loader, max_batches))


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("GPT training failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
