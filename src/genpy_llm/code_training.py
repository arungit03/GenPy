"""Configuration and training utilities for GenPy Code LLM."""

from __future__ import annotations

import glob
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.config import LossConfig, OptimizerConfig, get_project_root
from genpy_llm.device import select_device
from genpy_llm.gpt import GPTModel
from genpy_llm.losses import GPTCrossEntropyLoss
from genpy_llm.optimizers import create_optimizer
from genpy_llm.performance import (
    StepTimer,
    autocast_context,
    create_grad_scaler,
    peak_memory_mb,
    reset_peak_memory,
    validate_mixed_precision,
)
from genpy_llm.streaming_dataset import StreamingGPTDataset

LOGGER = logging.getLogger("genpy_llm.code_training")


class CodeTrainingError(RuntimeError):
    """Raised when code-model training cannot continue."""


@dataclass(frozen=True)
class CodeTokenizerConfig:
    path: Path
    metadata_path: Path
    type: str
    vocab_size: int
    pad_token: str
    unknown_token: str
    bos_token: str
    eos_token: str


@dataclass(frozen=True)
class CodeStreamingConfig:
    train_pattern: str
    validation_pattern: str
    text_field: str
    context_length: int
    stride: int
    append_eos: bool
    pack_across_files: bool
    shuffle_shards: bool
    shuffle_buffer_records: int
    seed: int
    incomplete_window_policy: str
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class CodeModelConfig:
    embedding_dim: int
    num_heads: int
    num_layers: int
    context_length: int
    dropout: float
    use_bias: bool
    tie_embeddings: bool
    initialization_std: float
    gradient_checkpointing: bool


@dataclass(frozen=True)
class CodeTrainingSettings:
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float | None
    device: str
    mixed_precision: str
    log_every_steps: int
    save_every_steps: int
    validate_every_steps: int
    validation_steps: int
    keep_last: int
    resume: bool


@dataclass(frozen=True)
class CodeSchedulerConfig:
    type: str
    warmup_steps: int
    minimum_learning_rate_ratio: float


@dataclass(frozen=True)
class CodeCheckpointConfig:
    directory: Path
    filename_prefix: str
    best_filename: str
    monitor: str
    mode: str
    save_best: bool


@dataclass(frozen=True)
class CodeGenerationConfig:
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    do_sample: bool
    repetition_penalty: float
    stop_on_eos: bool


@dataclass(frozen=True)
class CodeFineTuningConfig:
    dataset_file: Path
    output_directory: Path
    best_filename: str
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    validation_ratio: float
    response_only_loss: bool
    max_sequence_length: int
    save_every_steps: int
    validate_every_steps: int
    validation_steps: int


@dataclass(frozen=True)
class CodeConfig:
    project_name: str
    seed: int
    tokenizer: CodeTokenizerConfig
    streaming_dataset: CodeStreamingConfig
    model: CodeModelConfig
    training: CodeTrainingSettings
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: CodeSchedulerConfig
    checkpoint: CodeCheckpointConfig
    generation: CodeGenerationConfig
    fine_tuning: CodeFineTuningConfig
    project_root: Path


@dataclass(frozen=True)
class StepTrainingResult:
    global_step: int
    training_loss: float
    validation_loss: float | None
    tokens_processed: int
    tokens_per_second: float
    latest_checkpoint: Path | None
    best_checkpoint: Path | None


@dataclass(frozen=True)
class CodeArtifactValidation:
    tokenizer_path: Path
    tokenizer_metadata_path: Path
    train_shards: tuple[Path, ...]
    validation_shards: tuple[Path, ...]
    checkpoint_directory: Path
    mixed_precision: str


def load_code_config(config_path: Path | str = "configs/code_small.yaml") -> CodeConfig:
    """Load and validate the code-model YAML configuration."""

    root = get_project_root()
    path = Path(config_path)
    path = path if path.is_absolute() else root / path
    if not path.exists():
        raise FileNotFoundError(f"Code config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise CodeTrainingError("Code config must be a YAML mapping.")
    try:
        config = CodeConfig(
            project_name=str(raw["project"]["name"]),
            seed=int(raw["project"].get("seed", 42)),
            tokenizer=CodeTokenizerConfig(
                path=_resolve(root, raw["tokenizer"]["path"]),
                metadata_path=_resolve(root, raw["tokenizer"]["metadata_path"]),
                type=str(raw["tokenizer"]["type"]),
                vocab_size=int(raw["tokenizer"]["vocab_size"]),
                pad_token=str(raw["tokenizer"]["pad_token"]),
                unknown_token=str(raw["tokenizer"]["unknown_token"]),
                bos_token=str(raw["tokenizer"]["bos_token"]),
                eos_token=str(raw["tokenizer"]["eos_token"]),
            ),
            streaming_dataset=CodeStreamingConfig(**raw["streaming_dataset"]),
            model=CodeModelConfig(**raw["model"]),
            training=CodeTrainingSettings(**raw["training"]),
            loss=LossConfig(**raw["loss"]),
            optimizer=OptimizerConfig(**raw["optimizer"]),
            scheduler=CodeSchedulerConfig(**raw["scheduler"]),
            checkpoint=CodeCheckpointConfig(
                directory=_resolve(root, raw["checkpoint"]["directory"]),
                filename_prefix=str(raw["checkpoint"]["filename_prefix"]),
                best_filename=str(raw["checkpoint"]["best_filename"]),
                monitor=str(raw["checkpoint"]["monitor"]),
                mode=str(raw["checkpoint"]["mode"]),
                save_best=bool(raw["checkpoint"]["save_best"]),
            ),
            generation=CodeGenerationConfig(**raw["generation"]),
            fine_tuning=CodeFineTuningConfig(
                dataset_file=_resolve(root, raw["fine_tuning"]["dataset_file"]),
                output_directory=_resolve(root, raw["fine_tuning"]["output_directory"]),
                best_filename=str(raw["fine_tuning"]["best_filename"]),
                max_steps=int(raw["fine_tuning"]["max_steps"]),
                batch_size=int(raw["fine_tuning"]["batch_size"]),
                gradient_accumulation_steps=int(raw["fine_tuning"]["gradient_accumulation_steps"]),
                learning_rate=float(raw["fine_tuning"]["learning_rate"]),
                validation_ratio=float(raw["fine_tuning"]["validation_ratio"]),
                response_only_loss=bool(raw["fine_tuning"]["response_only_loss"]),
                max_sequence_length=int(raw["fine_tuning"]["max_sequence_length"]),
                save_every_steps=int(raw["fine_tuning"]["save_every_steps"]),
                validate_every_steps=int(raw["fine_tuning"]["validate_every_steps"]),
                validation_steps=int(raw["fine_tuning"]["validation_steps"]),
            ),
            project_root=root,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CodeTrainingError(f"Invalid code config values: {exc}") from exc
    _validate_code_config(config)
    return config


def validate_code_training_artifacts(config: CodeConfig) -> CodeArtifactValidation:
    """Validate files and directories needed before starting code training."""

    if not config.tokenizer.path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {config.tokenizer.path}")
    if not config.tokenizer.metadata_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer metadata file not found: {config.tokenizer.metadata_path}"
        )
    train_shards = _matched_shards(config, config.streaming_dataset.train_pattern)
    if not train_shards:
        raise FileNotFoundError(
            "No training shards match pattern: "
            f"{_resolve(config.project_root, config.streaming_dataset.train_pattern)}"
        )
    validation_shards = _matched_shards(config, config.streaming_dataset.validation_pattern)
    if not validation_shards:
        raise FileNotFoundError(
            "No validation shards match pattern: "
            f"{_resolve(config.project_root, config.streaming_dataset.validation_pattern)}"
        )
    if config.checkpoint.directory.exists() and not config.checkpoint.directory.is_dir():
        raise CodeTrainingError(
            f"Checkpoint path is not a directory: {config.checkpoint.directory}"
        )
    config.checkpoint.directory.mkdir(parents=True, exist_ok=True)
    return CodeArtifactValidation(
        tokenizer_path=config.tokenizer.path,
        tokenizer_metadata_path=config.tokenizer.metadata_path,
        train_shards=train_shards,
        validation_shards=validation_shards,
        checkpoint_directory=config.checkpoint.directory,
        mixed_precision=config.training.mixed_precision,
    )


def create_code_model(config: CodeConfig, vocab_size: int, padding_idx: int) -> GPTModel:
    """Create a GPTModel for code using the tokenizer vocabulary size."""

    hidden_dim = config.model.embedding_dim * 4
    model = GPTModel(
        vocab_size=vocab_size,
        embedding_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        context_length=config.model.context_length,
        feed_forward_hidden_dim=hidden_dim,
        padding_idx=padding_idx,
        dropout=config.model.dropout,
        use_bias=config.model.use_bias,
        tie_embeddings=config.model.tie_embeddings,
        initialization_std=config.model.initialization_std,
    )
    if config.model.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    return model


def create_code_dataloader(
    config: CodeConfig,
    tokenizer,
    *,
    split: str,
    batch_size: int | None = None,
) -> DataLoader:
    """Create a streaming DataLoader for train or validation."""

    pattern = (
        config.streaming_dataset.train_pattern
        if split == "train"
        else config.streaming_dataset.validation_pattern
    )
    dataset = StreamingGPTDataset(
        _resolve(config.project_root, pattern),
        tokenizer,
        text_field=config.streaming_dataset.text_field,
        context_length=config.streaming_dataset.context_length,
        stride=config.streaming_dataset.stride,
        append_eos=config.streaming_dataset.append_eos,
        pack_across_files=config.streaming_dataset.pack_across_files,
        shuffle_shards=config.streaming_dataset.shuffle_shards if split == "train" else False,
        shuffle_buffer_records=(
            config.streaming_dataset.shuffle_buffer_records if split == "train" else 0
        ),
        seed=config.streaming_dataset.seed,
        incomplete_window_policy=config.streaming_dataset.incomplete_window_policy,
        ignore_index=tokenizer.pad_token_id,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or config.training.batch_size,
        num_workers=config.streaming_dataset.num_workers,
        pin_memory=config.streaming_dataset.pin_memory,
    )


def train_code_steps(
    *,
    model: GPTModel,
    tokenizer,
    config: CodeConfig,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    device: torch.device,
    max_steps: int,
    max_batches: int | None = None,
    validation_batches: int | None = None,
    checkpoint_path: Path | None = None,
    evaluation_dir: Path | None = None,
    generation_prompts: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> StepTrainingResult:
    """Run small or full step-based code-model training."""

    log = logger or LOGGER
    log.debug(
        "Entered train_code_steps: device=%s max_steps=%s max_batches=%s "
        "validation_batches=%s mixed_precision=%s gradient_accumulation_steps=%s",
        device,
        max_steps,
        max_batches,
        validation_batches,
        config.training.mixed_precision,
        config.training.gradient_accumulation_steps,
    )
    validate_mixed_precision(config.training.mixed_precision, device)
    log.debug(
        "mixed precision validated: mode=%s device=%s",
        config.training.mixed_precision,
        device,
    )
    model.to(device)
    log.debug("model moved to device: %s", device)
    optimizer = create_optimizer(model, config.optimizer)
    log.debug(
        "optimizer created: type=%s learning_rate=%s weight_decay=%s",
        config.optimizer.type,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config.scheduler, max_steps)
    log.debug(
        "scheduler created: type=%s warmup_steps=%s min_lr_ratio=%s",
        config.scheduler.type,
        config.scheduler.warmup_steps,
        config.scheduler.minimum_learning_rate_ratio,
    )
    scaler = create_grad_scaler(config.training.mixed_precision, device)
    log.debug("gradient scaler created: enabled=%s", scaler is not None)
    loss_fn = GPTCrossEntropyLoss(
        padding_idx=tokenizer.pad_token_id,
        ignore_padding=config.loss.ignore_padding,
        label_smoothing=config.loss.label_smoothing,
    )
    global_step = 0
    tokens_processed = 0
    total_loss = 0.0
    best_validation = None
    if checkpoint_path is not None and checkpoint_path.exists():
        log.debug("loading checkpoint: %s", checkpoint_path)
        loaded = load_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        global_step = loaded.global_step
        best_validation = loaded.best_metric
        tokens_processed = _resume_tokens_processed(loaded.extra_state)
        if loaded.training_loss is not None and tokens_processed > 0:
            total_loss = loaded.training_loss * tokens_processed
        log.debug(
            "checkpoint loaded: path=%s global_step=%s best_metric=%s tokens_processed=%s",
            loaded.checkpoint_path,
            global_step,
            best_validation,
            tokens_processed,
        )
    start_global_step = global_step
    latest_checkpoint = None
    best_checkpoint = None
    accumulated = 0
    last_gradient_norm = None
    reset_peak_memory(device)
    log.debug("entering training loop")
    training_start = time.perf_counter()
    with StepTimer(device) as timer:
        optimizer.zero_grad(set_to_none=True)
        train_iterator = iter(train_loader)
        batch_index = 0
        while True:
            if max_batches is not None and batch_index >= max_batches:
                log.debug("stopping training loop at max_batches=%s", max_batches)
                break
            log.debug(
                "waiting for training batch: batch_index=%s global_step=%s accumulated=%s",
                batch_index,
                global_step,
                accumulated,
            )
            try:
                batch = next(train_iterator)
            except StopIteration:
                log.debug("training dataloader exhausted at batch_index=%s", batch_index)
                break
            log.debug("training batch received: batch_index=%s", batch_index)
            model.train()
            log.debug("moving training batch to device: batch_index=%s", batch_index)
            input_ids, target_ids, attention_mask = _move_batch(batch, device)
            log.debug(
                "training step start: batch_index=%s global_step=%s input_shape=%s "
                "target_shape=%s attention_mask=%s",
                batch_index,
                global_step,
                tuple(input_ids.shape),
                tuple(target_ids.shape),
                None if attention_mask is None else tuple(attention_mask.shape),
            )
            with autocast_context(config.training.mixed_precision, device):
                logits = model(input_ids, padding_mask=attention_mask)
                loss = loss_fn(logits, target_ids)
            log.debug(
                "training forward complete: batch_index=%s loss=%.6f logits_dtype=%s",
                batch_index,
                float(loss.detach().item()),
                logits.dtype,
            )
            if not torch.isfinite(loss):
                raise CodeTrainingError("NaN or infinite loss detected.")
            scaled_loss = loss / config.training.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            log.debug("training backward complete: batch_index=%s", batch_index)
            accumulated += 1
            token_count = (
                int(attention_mask.sum().item())
                if attention_mask is not None
                else int(input_ids.numel())
            )
            tokens_processed += token_count
            total_loss += float(loss.detach().item()) * token_count
            if accumulated >= config.training.gradient_accumulation_steps:
                log.debug(
                    "optimizer step start: next_global_step=%s accumulated_batches=%s",
                    global_step + 1,
                    accumulated,
                )
                last_gradient_norm = _optimizer_step(
                    model,
                    optimizer,
                    scaler,
                    config.training.max_grad_norm,
                )
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                global_step += 1
                log.debug(
                    "optimizer step complete: global_step=%s training_loss=%.6f tokens=%s",
                    global_step,
                    total_loss / max(tokens_processed, 1),
                    tokens_processed,
                )
                elapsed = time.perf_counter() - training_start
                training_loss = total_loss / max(tokens_processed, 1)
                if global_step % config.training.log_every_steps == 0:
                    _report_training_metrics(
                        global_step=global_step,
                        max_steps=max_steps,
                        start_global_step=start_global_step,
                        training_loss=training_loss,
                        learning_rate=_current_learning_rate(optimizer),
                        gradient_norm=last_gradient_norm,
                        tokens_processed=tokens_processed,
                        elapsed_seconds=elapsed,
                    )
                    _record_training_metrics(
                        evaluation_dir=evaluation_dir,
                        global_step=global_step,
                        max_steps=max_steps,
                        start_global_step=start_global_step,
                        training_loss=training_loss,
                        validation_loss=None,
                        learning_rate=_current_learning_rate(optimizer),
                        gradient_norm=last_gradient_norm,
                        tokens_processed=tokens_processed,
                        elapsed_seconds=elapsed,
                    )
                if global_step % config.training.validate_every_steps == 0 and validation_loader:
                    log.debug("validation step triggered: global_step=%s", global_step)
                    best_validation, validation_loss = _maybe_validate_and_save(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        tokenizer=tokenizer,
                        config=config,
                        validation_loader=validation_loader,
                        device=device,
                        global_step=global_step,
                        training_loss=total_loss / max(tokens_processed, 1),
                        best_validation=best_validation,
                        validation_batches=validation_batches,
                        tokens_processed=tokens_processed,
                        logger=log,
                    )
                    best_checkpoint = config.checkpoint.directory / config.checkpoint.best_filename
                    _report_validation_metrics(
                        global_step=global_step,
                        max_steps=max_steps,
                        start_global_step=start_global_step,
                        training_loss=total_loss / max(tokens_processed, 1),
                        validation_loss=validation_loss,
                        learning_rate=_current_learning_rate(optimizer),
                        gradient_norm=last_gradient_norm,
                        tokens_processed=tokens_processed,
                        elapsed_seconds=time.perf_counter() - training_start,
                    )
                    _record_training_metrics(
                        evaluation_dir=evaluation_dir,
                        global_step=global_step,
                        max_steps=max_steps,
                        start_global_step=start_global_step,
                        training_loss=total_loss / max(tokens_processed, 1),
                        validation_loss=validation_loss,
                        learning_rate=_current_learning_rate(optimizer),
                        gradient_norm=last_gradient_norm,
                        tokens_processed=tokens_processed,
                        elapsed_seconds=time.perf_counter() - training_start,
                    )
                    _maybe_write_generation_snapshot(
                        model=model,
                        tokenizer=tokenizer,
                        config=config,
                        device=device,
                        global_step=global_step,
                        evaluation_dir=evaluation_dir,
                        prompts=generation_prompts,
                        logger=log,
                    )
                if global_step % config.training.save_every_steps == 0:
                    log.debug("periodic checkpoint save triggered: global_step=%s", global_step)
                    latest_checkpoint = _save_code_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        tokenizer=tokenizer,
                        config=config,
                        global_step=global_step,
                        training_loss=total_loss / max(tokens_processed, 1),
                        validation_loss=None,
                        best_metric=best_validation,
                        tokens_processed=tokens_processed,
                        logger=log,
                    )
            if global_step >= max_steps:
                log.debug("stopping training loop at max_steps=%s", max_steps)
                break
            batch_index += 1
        if accumulated > 0 and global_step < max_steps:
            log.debug(
                "final partial optimizer step start: next_global_step=%s accumulated_batches=%s",
                global_step + 1,
                accumulated,
            )
            last_gradient_norm = _optimizer_step(
                model,
                optimizer,
                scaler,
                config.training.max_grad_norm,
            )
            scheduler.step()
            global_step += 1
            log.debug("final partial optimizer step complete: global_step=%s", global_step)
    if tokens_processed == 0:
        raise CodeTrainingError("Training dataloader produced no batches.")
    validation_loss = None
    if validation_loader is not None:
        log.debug("final validation step start")
        validation_loss = evaluate_code_model(
            model,
            validation_loader,
            loss_fn,
            device,
            config.training.mixed_precision,
            max_batches=_validation_batch_limit(validation_batches, config),
            logger=log,
        )
        log.debug("final validation step complete: validation_loss=%.6f", validation_loss)
        if best_validation is None or validation_loss < best_validation:
            best_validation = validation_loss
            best_checkpoint = config.checkpoint.directory / config.checkpoint.best_filename
            log.debug("best checkpoint save triggered: validation_loss=%.6f", validation_loss)
            _save_code_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                tokenizer=tokenizer,
                config=config,
                global_step=global_step,
                training_loss=total_loss / max(tokens_processed, 1),
                validation_loss=validation_loss,
                best_metric=best_validation,
                output_path=best_checkpoint,
                tokens_processed=tokens_processed,
                logger=log,
            )
        _report_validation_metrics(
            global_step=global_step,
            max_steps=max_steps,
            start_global_step=start_global_step,
            training_loss=total_loss / max(tokens_processed, 1),
            validation_loss=validation_loss,
            learning_rate=_current_learning_rate(optimizer),
            gradient_norm=last_gradient_norm,
            tokens_processed=tokens_processed,
            elapsed_seconds=time.perf_counter() - training_start,
        )
        _record_training_metrics(
            evaluation_dir=evaluation_dir,
            global_step=global_step,
            max_steps=max_steps,
            start_global_step=start_global_step,
            training_loss=total_loss / max(tokens_processed, 1),
            validation_loss=validation_loss,
            learning_rate=_current_learning_rate(optimizer),
            gradient_norm=last_gradient_norm,
            tokens_processed=tokens_processed,
            elapsed_seconds=time.perf_counter() - training_start,
        )
        _maybe_write_generation_snapshot(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            global_step=global_step,
            evaluation_dir=evaluation_dir,
            prompts=generation_prompts,
            logger=log,
        )
    log.debug("final checkpoint save triggered: global_step=%s", global_step)
    latest_checkpoint = _save_code_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        tokenizer=tokenizer,
        config=config,
        global_step=global_step,
        training_loss=total_loss / max(tokens_processed, 1),
        validation_loss=validation_loss,
        best_metric=best_validation,
        tokens_processed=tokens_processed,
        logger=log,
    )
    tokens_per_second = (
        tokens_processed / timer.elapsed_seconds if timer.elapsed_seconds > 0 else 0.0
    )
    _ = peak_memory_mb(device)
    return StepTrainingResult(
        global_step=global_step,
        training_loss=total_loss / max(tokens_processed, 1),
        validation_loss=validation_loss,
        tokens_processed=tokens_processed,
        tokens_per_second=tokens_per_second,
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=best_checkpoint,
    )


@torch.no_grad()
def evaluate_code_model(
    model: GPTModel,
    loader: DataLoader,
    loss_fn: GPTCrossEntropyLoss,
    device: torch.device,
    mixed_precision: str,
    *,
    max_batches: int,
    logger: logging.Logger | None = None,
) -> float:
    """Evaluate a code model for a bounded number of batches."""

    log = logger or LOGGER
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    iterator = iter(loader)
    batch_index = 0
    while True:
        if batch_index >= max_batches:
            break
        log.debug("waiting for validation batch: batch_index=%s", batch_index)
        try:
            batch = next(iterator)
        except StopIteration:
            log.debug("validation dataloader exhausted at batch_index=%s", batch_index)
            break
        log.debug("validation batch received: batch_index=%s", batch_index)
        input_ids, target_ids, attention_mask = _move_batch(batch, device)
        with autocast_context(mixed_precision, device):
            logits = model(input_ids, padding_mask=attention_mask)
            loss = loss_fn(logits, target_ids)
        log.debug(
            "validation step complete: batch_index=%s loss=%.6f logits_dtype=%s",
            batch_index,
            float(loss.detach().item()),
            logits.dtype,
        )
        tokens = (
            int(attention_mask.sum().item())
            if attention_mask is not None
            else int(input_ids.numel())
        )
        total_loss += float(loss.detach().item()) * tokens
        total_tokens += tokens
        batch_index += 1
    return total_loss / total_tokens if total_tokens else 0.0


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CodeSchedulerConfig,
    max_steps: int,
):
    """Build the configured learning-rate scheduler."""

    if config.type != "cosine":
        raise CodeTrainingError("Only cosine scheduler is supported.")

    def lr_lambda(step: int) -> float:
        if config.warmup_steps > 0 and step < config.warmup_steps:
            return max(step + 1, 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(max_steps - config.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        minimum = config.minimum_learning_rate_ratio
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_code_checkpoint(
    *,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object | None,
    tokenizer,
    config: CodeConfig,
    global_step: int,
    training_loss: float,
    validation_loss: float | None,
    best_metric: float | None,
    output_path: Path | None = None,
    tokens_processed: int | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    log = logger or LOGGER
    config.checkpoint.directory.mkdir(parents=True, exist_ok=True)
    path = output_path or (
        config.checkpoint.directory
        / f"{config.checkpoint.filename_prefix}_step_{global_step:08d}.pt"
    )
    log.debug("checkpoint save start: path=%s global_step=%s", path, global_step)
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=0,
        global_step=global_step,
        training_loss=training_loss,
        validation_loss=validation_loss,
        best_metric=best_metric,
        scheduler=scheduler,
        scaler=scaler,
        model_config=_model_config_dict(config),
        vocabulary_metadata={
            "tokenizer_type": "byte_bpe",
            "vocabulary_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "tokenizer_path": str(config.tokenizer.path),
        },
        extra_state={
            "checkpoint_type": "code_base",
            "tokens_processed": tokens_processed,
            "format_version": 1,
        },
    )
    _rotate_step_checkpoints(
        config.checkpoint.directory,
        config.checkpoint.filename_prefix,
        config.training.keep_last,
        logger=log,
    )
    log.debug("checkpoint save complete: path=%s", path)
    return path


def _maybe_validate_and_save(**kwargs) -> tuple[float, float]:
    log = kwargs.get("logger") or LOGGER
    log.debug("validation step start: global_step=%s", kwargs["global_step"])
    validation_loss = evaluate_code_model(
        kwargs["model"],
        kwargs["validation_loader"],
        GPTCrossEntropyLoss(
            padding_idx=kwargs["tokenizer"].pad_token_id,
            ignore_padding=kwargs["config"].loss.ignore_padding,
            label_smoothing=kwargs["config"].loss.label_smoothing,
        ),
        kwargs["device"],
        kwargs["config"].training.mixed_precision,
        max_batches=_validation_batch_limit(
            kwargs["validation_batches"],
            kwargs["config"],
        ),
        logger=log,
    )
    log.debug(
        "validation step complete: global_step=%s validation_loss=%.6f",
        kwargs["global_step"],
        validation_loss,
    )
    best_validation = kwargs["best_validation"]
    if best_validation is None or validation_loss < best_validation:
        log.debug("best checkpoint save triggered: global_step=%s", kwargs["global_step"])
        _save_code_checkpoint(
            model=kwargs["model"],
            optimizer=kwargs["optimizer"],
            scheduler=kwargs["scheduler"],
            scaler=kwargs["scaler"],
            tokenizer=kwargs["tokenizer"],
            config=kwargs["config"],
            global_step=kwargs["global_step"],
            training_loss=kwargs["training_loss"],
            validation_loss=validation_loss,
            best_metric=validation_loss,
            output_path=(
                kwargs["config"].checkpoint.directory / kwargs["config"].checkpoint.best_filename
            ),
            tokens_processed=kwargs.get("tokens_processed"),
            logger=log,
        )
        return validation_loss, validation_loss
    return best_validation, validation_loss


def _report_validation_metrics(
    *,
    global_step: int,
    max_steps: int,
    start_global_step: int,
    training_loss: float,
    validation_loss: float,
    learning_rate: float,
    gradient_norm: float | None,
    tokens_processed: int,
    elapsed_seconds: float,
) -> None:
    tokens_per_second = tokens_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    eta_seconds = _estimate_eta_seconds(
        global_step=global_step,
        max_steps=max_steps,
        start_global_step=start_global_step,
        elapsed_seconds=elapsed_seconds,
    )
    perplexity = math.exp(validation_loss) if validation_loss <= 80 else math.inf
    print()
    print("Validation")
    print("----------")
    print(f"Step: {global_step}")
    print(f"Train Loss: {training_loss:.6f}")
    print(f"Validation Loss: {validation_loss:.6f}")
    print(f"Perplexity: {'inf' if math.isinf(perplexity) else f'{perplexity:.4f}'}")
    print(f"Learning Rate: {learning_rate:.8f}")
    print(f"Gradient Norm: {_format_optional_float(gradient_norm)}")
    print(f"Tokens/sec: {tokens_per_second:.2f}")
    print(f"Elapsed Time: {_format_duration(elapsed_seconds)}")
    print(f"ETA: {_format_duration(eta_seconds)}")
    print()


def _report_training_metrics(
    *,
    global_step: int,
    max_steps: int,
    start_global_step: int,
    training_loss: float,
    learning_rate: float,
    gradient_norm: float | None,
    tokens_processed: int,
    elapsed_seconds: float,
) -> None:
    tokens_per_second = tokens_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    eta_seconds = _estimate_eta_seconds(
        global_step=global_step,
        max_steps=max_steps,
        start_global_step=start_global_step,
        elapsed_seconds=elapsed_seconds,
    )
    print(
        "Training "
        f"step={global_step} "
        f"loss={training_loss:.6f} "
        f"lr={learning_rate:.8f} "
        f"grad_norm={_format_optional_float(gradient_norm)} "
        f"tokens/sec={tokens_per_second:.2f} "
        f"eta={_format_duration(eta_seconds)}"
    )


def _record_training_metrics(
    *,
    evaluation_dir: Path | None,
    global_step: int,
    max_steps: int,
    start_global_step: int,
    training_loss: float,
    validation_loss: float | None,
    learning_rate: float,
    gradient_norm: float | None,
    tokens_processed: int,
    elapsed_seconds: float,
) -> Path | None:
    if evaluation_dir is None:
        return None
    from genpy_llm.code_evaluation import (
        TrainingMetricsRow,
        append_training_metrics_csv,
        loss_history_from_training_metrics,
        perplexity_from_loss,
        read_training_metrics_csv,
        write_loss_curve_png,
    )

    eta_seconds = _estimate_eta_seconds(
        global_step=global_step,
        max_steps=max_steps,
        start_global_step=start_global_step,
        elapsed_seconds=elapsed_seconds,
    )
    metrics_path = evaluation_dir / "training_metrics.csv"
    append_training_metrics_csv(
        TrainingMetricsRow(
            global_step=global_step,
            training_loss=training_loss,
            validation_loss=validation_loss,
            perplexity=perplexity_from_loss(validation_loss),
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            tokens_per_second=tokens_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            tokens_processed=tokens_processed,
            elapsed_seconds=elapsed_seconds,
            eta_seconds=eta_seconds,
        ),
        metrics_path,
    )
    rows = read_training_metrics_csv(metrics_path)
    write_loss_curve_png(
        loss_history_from_training_metrics(rows),
        evaluation_dir / "loss_curve.png",
    )
    return metrics_path


def _maybe_write_generation_snapshot(
    *,
    model: GPTModel,
    tokenizer,
    config: CodeConfig,
    device: torch.device,
    global_step: int,
    evaluation_dir: Path | None,
    prompts: Sequence[str] | None,
    logger: logging.Logger,
) -> Path | None:
    if evaluation_dir is None:
        return None
    from genpy_llm.code_evaluation import (
        DEFAULT_CODE_PROMPTS,
        run_generation_benchmark,
        write_generation_examples,
    )

    selected_prompts = tuple(prompts or DEFAULT_CODE_PROMPTS)[:5]
    output_path = evaluation_dir / f"step_{global_step:04d}_generation.txt"
    logger.debug("generation snapshot start: path=%s", output_path)
    benchmark = run_generation_benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=selected_prompts,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        repetition_penalty=config.generation.repetition_penalty,
        do_sample=config.generation.do_sample,
        stop_on_eos=config.generation.stop_on_eos,
        context_length=config.model.context_length,
    )
    write_generation_examples(benchmark, output_path, step=global_step)
    logger.debug("generation snapshot complete: path=%s", output_path)
    return output_path


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _validation_batch_limit(value: int | None, config: CodeConfig) -> int:
    return value if value is not None else config.training.validation_steps


def _resume_tokens_processed(extra_state: Mapping[str, Any]) -> int:
    value = extra_state.get("tokens_processed")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _estimate_eta_seconds(
    *,
    global_step: int,
    max_steps: int,
    start_global_step: int,
    elapsed_seconds: float,
) -> float:
    completed_steps = max(global_step - start_global_step, 0)
    if completed_steps <= 0:
        return 0.0
    remaining_steps = max(max_steps - global_step, 0)
    return remaining_steps * (elapsed_seconds / completed_steps)


def _format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_optional_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _optimizer_step(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scaler: object | None,
    max_grad_norm: float | None,
) -> float:
    if scaler is not None:
        scaler.unscale_(optimizer)
    gradient_norm = _compute_gradient_norm(model)
    if max_grad_norm is not None:
        clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        gradient_norm = float(clipped.detach().item())
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return gradient_norm


def _compute_gradient_norm(model: GPTModel) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameter_norm = parameter.grad.detach().float().norm(2)
        total += float(parameter_norm.item()) ** 2
    return math.sqrt(total)


def _move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, target_ids, attention_mask


def _model_config_dict(config: CodeConfig) -> dict[str, Any]:
    return {
        "vocab_size": config.tokenizer.vocab_size,
        "embedding_dim": config.model.embedding_dim,
        "num_heads": config.model.num_heads,
        "num_layers": config.model.num_layers,
        "context_length": config.model.context_length,
        "dropout": config.model.dropout,
        "tie_embeddings": config.model.tie_embeddings,
    }


def _rotate_step_checkpoints(
    directory: Path,
    prefix: str,
    keep_last: int,
    logger: logging.Logger | None = None,
) -> None:
    log = logger or LOGGER
    paths = sorted(directory.glob(f"{prefix}_step_*.pt"), key=lambda path: path.name, reverse=True)
    for path in paths[keep_last:]:
        log.debug("removing old checkpoint: %s", path)
        path.unlink(missing_ok=True)


def _validate_code_config(config: CodeConfig) -> None:
    if config.tokenizer.type != "byte_bpe":
        raise CodeTrainingError("tokenizer.type must be byte_bpe.")
    if config.model.embedding_dim % config.model.num_heads != 0:
        raise CodeTrainingError("model.embedding_dim must be divisible by model.num_heads.")
    if config.model.context_length != config.streaming_dataset.context_length:
        raise CodeTrainingError("model.context_length must match streaming_dataset.context_length.")
    for name, value in {
        "training.max_steps": config.training.max_steps,
        "training.batch_size": config.training.batch_size,
        "training.gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "fine_tuning.max_steps": config.fine_tuning.max_steps,
        "fine_tuning.batch_size": config.fine_tuning.batch_size,
    }.items():
        if value <= 0:
            raise CodeTrainingError(f"{name} must be greater than zero.")
    if config.training.mixed_precision not in {"none", "fp16", "bf16"}:
        raise CodeTrainingError("training.mixed_precision must be none, fp16, or bf16.")
    if config.scheduler.minimum_learning_rate_ratio < 0:
        raise CodeTrainingError("scheduler.minimum_learning_rate_ratio must be non-negative.")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _matched_shards(config: CodeConfig, pattern: str) -> tuple[Path, ...]:
    resolved_pattern = _resolve(config.project_root, pattern)
    return tuple(sorted(Path(path) for path in glob.glob(str(resolved_pattern))))


__all__ = [
    "CodeConfig",
    "CodeArtifactValidation",
    "CodeTrainingError",
    "StepTrainingResult",
    "build_scheduler",
    "create_code_dataloader",
    "create_code_model",
    "load_code_config",
    "select_device",
    "train_code_steps",
    "validate_code_training_artifacts",
]
