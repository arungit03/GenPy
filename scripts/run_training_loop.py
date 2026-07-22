"""Run a tiny GPT training-loop smoke test."""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import islice
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import ConfigError, load_config
from genpy_llm.dataset import DatasetPreparationError, load_dataset_split
from genpy_llm.device import select_device
from genpy_llm.gpt import GPTModelError, create_gpt_model
from genpy_llm.logging_utils import setup_logging
from genpy_llm.losses import LossError, create_loss_function
from genpy_llm.optimizers import OptimizerError, create_optimizer
from genpy_llm.training import GPTTrainer, TrainingError
from genpy_llm.utils import set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Parse arguments and run a small training-loop smoke test."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        app_config = load_config()
        logger = setup_logging(
            log_dir=app_config.paths.logs_dir,
            log_file=app_config.logging.log_file,
            level="DEBUG" if args.debug else app_config.logging.level,
        )
        set_seed(args.seed if args.seed is not None else app_config.training.seed)
        device = select_device(args.device or app_config.training.device)

        model, metadata = create_gpt_model(app_config.data.vocabulary_file, app_config)
        train_dataset = load_dataset_split(app_config.data.train_dataset_file)
        validation_dataset = load_dataset_split(app_config.data.validation_dataset_file)
        train_loader = _limited_loader(
            DataLoader(train_dataset, batch_size=app_config.dataset.batch_size, shuffle=False),
            max_batches=args.max_batches,
        )
        validation_loader = _limited_loader(
            DataLoader(validation_dataset, batch_size=app_config.dataset.batch_size, shuffle=False),
            max_batches=args.max_batches,
        )

        loss_fn = create_loss_function(app_config.data.vocabulary_file, app_config.loss)
        optimizer = create_optimizer(model, app_config.optimizer)
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
        )
        result = trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=args.epochs,
            validate_every_epochs=1,
            log_every_steps=app_config.training.log_every_steps,
        )
    except (
        ConfigError,
        DatasetPreparationError,
        FileNotFoundError,
        GPTModelError,
        IsADirectoryError,
        LossError,
        OSError,
        OptimizerError,
        RuntimeError,
        TrainingError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Training-loop smoke test completed successfully.")
    latest = result.epochs[-1]
    print("GenPy LLM Training Loop Smoke Test")
    print("==================================")
    print("This is a training-loop smoke test, not full model training.")
    print(f"Vocabulary size: {metadata.vocab_size}")
    print(f"Layer count: {metadata.num_layers}")
    print(f"Training loss: {latest.training_loss:.6f}")
    print(f"Validation loss: {latest.validation_loss:.6f}")
    print(f"Training tokens: {latest.training_tokens}")
    print(f"Validation tokens: {latest.validation_tokens}")
    print(f"Optimizer steps: {result.total_optimizer_steps}")
    print(f"Gradient accumulation steps: {trainer.gradient_accumulation_steps}")
    print(f"Max gradient norm: {trainer.max_grad_norm}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny GPT training-loop smoke test.")
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--max-batches", type=_positive_int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=None)
    parser.add_argument("--max-grad-norm", type=_positive_float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _limited_loader(loader: DataLoader, max_batches: int) -> list[dict[str, torch.Tensor]]:
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


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Training-loop smoke test failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
