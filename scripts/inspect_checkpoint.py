"""Inspect GPT checkpoint save/load behavior."""

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

from genpy_llm.checkpointing import CheckpointError, load_checkpoint, save_checkpoint
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
    """Run a tiny checkpoint save/load inspection."""

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
        set_seed(args.seed if args.seed is not None else app_config.training.seed)
        device = select_device(args.device or app_config.training.device)
        model, metadata = create_gpt_model(app_config.data.vocabulary_file, app_config)
        optimizer = create_optimizer(model, app_config.optimizer)
        loss_fn = create_loss_function(app_config.data.vocabulary_file, app_config.loss)
        trainer = GPTTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_accumulation_steps=app_config.training.gradient_accumulation_steps,
            max_grad_norm=app_config.training.max_grad_norm,
        )

        if args.load_path is not None:
            loaded = load_checkpoint(args.load_path, trainer.model, optimizer=trainer.optimizer)
            trainer.total_optimizer_steps = loaded.global_step
            print("GenPy LLM Checkpoint Inspection")
            print("==============================")
            print(f"Loaded checkpoint: {loaded.checkpoint_path}")
            print(f"Loaded epoch: {loaded.epoch}")
            print(f"Loaded global step: {loaded.global_step}")
            print(f"Loaded best metric: {loaded.best_metric}")
            return 0

        if args.save_path is None:
            raise ValueError("Provide --save-path or --load-path.")

        train_dataset = load_dataset_split(app_config.data.train_dataset_file)
        train_loader = list(
            islice(
                DataLoader(train_dataset, batch_size=app_config.dataset.batch_size, shuffle=False),
                1,
            )
        )
        epoch_metrics = trainer.train_epoch(
            train_loader,
            epoch=1,
            log_every_steps=app_config.training.log_every_steps,
        )
        expected_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in trainer.model.state_dict().items()
        }
        checkpoint_metadata = save_checkpoint(
            args.save_path,
            trainer.model,
            trainer.optimizer,
            epoch=epoch_metrics.epoch,
            global_step=trainer.total_optimizer_steps,
            training_loss=epoch_metrics.training_loss,
            validation_loss=None,
            best_metric=epoch_metrics.training_loss,
            model_config=asdict(app_config.model),
            vocabulary_metadata={
                "vocabulary_size": metadata.vocab_size,
                "vocabulary_path": str(app_config.data.vocabulary_file),
            },
        )
        _damage_model(trainer.model)
        loaded = load_checkpoint(args.save_path, trainer.model, optimizer=trainer.optimizer)
        trainer.total_optimizer_steps = loaded.global_step
        restore_ok = _state_matches(trainer.model.state_dict(), expected_state)

        if args.resume:
            trainer.train_epoch(train_loader, epoch=loaded.epoch + 1, log_every_steps=1)

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
        RuntimeError,
        TrainingError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    print("GenPy LLM Checkpoint Inspection")
    print("==============================")
    print(f"Saved checkpoint: {Path(args.save_path).resolve()}")
    print(f"Metadata epoch: {checkpoint_metadata.epoch}")
    print(f"Metadata global step: {checkpoint_metadata.global_step}")
    print(f"Loaded epoch: {loaded.epoch}")
    print(f"Loaded global step: {loaded.global_step}")
    print(f"Model restore test: {'passed' if restore_ok else 'failed'}")
    if args.resume:
        print(f"Resume global step: {trainer.total_optimizer_steps}")
    return 0 if restore_ok else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect checkpoint save/load behavior.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--load-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _damage_model(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)


def _state_matches(
    actual_state: dict[str, torch.Tensor],
    expected_state: dict[str, torch.Tensor],
) -> bool:
    if actual_state.keys() != expected_state.keys():
        return False
    return all(
        torch.equal(actual_state[name].detach().cpu(), expected_state[name])
        for name in expected_state
    )


def _resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Checkpoint inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
