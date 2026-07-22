"""Inspect GPT loss and optimizer setup without full training."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
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
from genpy_llm.optimizers import OptimizerError, create_optimizer_with_metadata
from genpy_llm.utils import set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Parse arguments and inspect configured loss/optimizer behavior."""

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
        loss_config = replace(
            app_config.loss,
            label_smoothing=args.label_smoothing
            if args.label_smoothing is not None
            else app_config.loss.label_smoothing,
        )
        optimizer_config = replace(
            app_config.optimizer,
            learning_rate=args.learning_rate
            if args.learning_rate is not None
            else app_config.optimizer.learning_rate,
            weight_decay=args.weight_decay
            if args.weight_decay is not None
            else app_config.optimizer.weight_decay,
        )

        model, _metadata = create_gpt_model(app_config.data.vocabulary_file, app_config)
        model = model.to(device)
        loss_fn = create_loss_function(app_config.data.vocabulary_file, loss_config)
        optimizer, optimizer_metadata = create_optimizer_with_metadata(model, optimizer_config)
        dataset = load_dataset_split(app_config.data.train_dataset_file)
        loader = DataLoader(dataset, batch_size=app_config.dataset.batch_size, shuffle=False)
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, padding_mask=attention_mask)
        loss = loss_fn(logits, targets)
        loss.backward()
        gradient_norm = _gradient_norm(model)
        if args.optimizer_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
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
        StopIteration,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Loss and optimizer inspection completed successfully.")
    print("GenPy LLM Loss and Optimizer")
    print("============================")
    print(f"Logits shape: {list(logits.shape)}")
    print(f"Targets shape: {list(targets.shape)}")
    print(f"Loss value: {loss.item():.6f}")
    print(f"Padding ID: {loss_fn.padding_idx}")
    print(f"Label smoothing: {loss_fn.label_smoothing}")
    print(f"Optimizer type: {optimizer_metadata.optimizer_type}")
    print(f"Learning rate: {optimizer_metadata.learning_rate}")
    print(f"Weight decay: {optimizer_metadata.weight_decay}")
    print(f"Decayed parameters: {optimizer_metadata.decayed_parameter_count}")
    print(f"Non-decayed parameters: {optimizer_metadata.non_decayed_parameter_count}")
    print(f"Trainable tensors: {optimizer_metadata.trainable_tensor_count}")
    print(f"Gradient norm: {gradient_norm:.6f}")
    print(f"Optimizer step performed: {args.optimizer_step}")
    if args.show_groups:
        for index, group in enumerate(optimizer.param_groups):
            parameters = group["params"]
            parameter_count = sum(parameter.numel() for parameter in parameters)
            print(
                f"Group {index}: tensors={len(parameters)}, "
                f"parameters={parameter_count}, weight_decay={group['weight_decay']}"
            )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GenPy LLM loss and optimizer.")
    parser.add_argument("--show-groups", action="store_true")
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--label-smoothing", type=_label_smoothing, default=None)
    parser.add_argument("--learning-rate", type=_positive_float, default=None)
    parser.add_argument("--weight-decay", type=_non_negative_float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _gradient_norm(model: torch.nn.Module) -> float:
    total = torch.tensor(0.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum().cpu()
    return float(total.sqrt().item())


def _label_smoothing(value: str) -> float:
    number = float(value)
    if not 0.0 <= number < 1.0:
        raise argparse.ArgumentTypeError("label smoothing must be at least 0.0 and less than 1.0.")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return number


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Loss and optimizer inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
