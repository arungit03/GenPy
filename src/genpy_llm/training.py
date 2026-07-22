"""Reusable training loop utilities for GenPy LLM."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from genpy_llm.checkpointing import CheckpointError, save_managed_checkpoint
from genpy_llm.performance import autocast_context, create_grad_scaler, validate_mixed_precision

LOGGER = logging.getLogger("genpy_llm")


class TrainingError(RuntimeError):
    """Raised when GPT training cannot continue safely."""


@dataclass(frozen=True)
class BatchMetrics:
    """Metrics for one batch or validation pass."""

    loss: float
    tokens: int
    batch_size: int


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics for one training epoch."""

    epoch: int
    training_loss: float
    validation_loss: float | None
    training_tokens: int
    validation_tokens: int
    optimizer_steps: int
    skipped_batches: int


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned by GPTTrainer.fit."""

    epochs: tuple[EpochMetrics, ...]
    total_optimizer_steps: int
    completed_epochs: int
    best_metric: float | None = None
    checkpoint_paths: tuple[Path, ...] = ()


class GPTTrainer:
    """Train a GPT-like model with externally provided loss and optimizer."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = 1.0,
        scheduler: object | None = None,
        mixed_precision: str = "none",
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TrainingError("model must be a torch.nn.Module.")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TrainingError("optimizer must be a torch.optim.Optimizer.")
        if not callable(loss_fn):
            raise TrainingError("loss_fn must be callable.")
        if not isinstance(device, torch.device):
            raise TrainingError("device must be a torch.device.")
        if (
            not isinstance(gradient_accumulation_steps, int)
            or isinstance(gradient_accumulation_steps, bool)
            or gradient_accumulation_steps <= 0
        ):
            raise TrainingError("gradient_accumulation_steps must be greater than zero.")
        if max_grad_norm is not None and (
            not isinstance(max_grad_norm, int | float)
            or isinstance(max_grad_norm, bool)
            or max_grad_norm <= 0
        ):
            raise TrainingError("max_grad_norm must be None or greater than zero.")
        try:
            validate_mixed_precision(mixed_precision, device)
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.scaler = create_grad_scaler(mixed_precision, device)
        self.total_optimizer_steps = 0
        self._pending_accumulation_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

    def train_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_index: int,
    ) -> BatchMetrics:
        """Run forward/backward for one mini-batch and step when accumulation is ready."""

        if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0:
            raise TrainingError("batch_index must be a non-negative integer.")
        self.model.train()
        input_ids, target_ids, attention_mask = self._prepare_batch(batch)
        with autocast_context(self.mixed_precision, self.device):
            logits = self.model(input_ids, padding_mask=attention_mask)
            loss = self._compute_loss(logits, target_ids)
        tokens = _count_tokens(target_ids, attention_mask)
        if tokens <= 0:
            raise TrainingError("Batch contains zero valid target tokens.")

        scaled_loss = loss / self.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self._pending_accumulation_batches += 1
        if self._pending_accumulation_batches >= self.gradient_accumulation_steps:
            self._optimizer_step()

        return BatchMetrics(
            loss=float(loss.detach().item()),
            tokens=tokens,
            batch_size=int(input_ids.shape[0]),
        )

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> BatchMetrics:
        """Evaluate without gradients and return token-weighted average loss."""

        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_batch_size = 0
        try:
            for batch in data_loader:
                input_ids, target_ids, attention_mask = self._prepare_batch(batch)
                with autocast_context(self.mixed_precision, self.device):
                    logits = self.model(input_ids, padding_mask=attention_mask)
                    loss = self._compute_loss(logits, target_ids)
                tokens = _count_tokens(target_ids, attention_mask)
                if tokens <= 0:
                    continue
                total_loss += float(loss.detach().item()) * tokens
                total_tokens += tokens
                total_batch_size += int(input_ids.shape[0])
        finally:
            if was_training:
                self.model.train()

        average_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        return BatchMetrics(
            loss=average_loss,
            tokens=total_tokens,
            batch_size=total_batch_size,
        )

    def train_epoch(
        self,
        data_loader: DataLoader,
        epoch: int,
        log_every_steps: int = 10,
    ) -> EpochMetrics:
        """Train for one epoch and flush any partial accumulation group."""

        _validate_positive_int("epoch", epoch)
        _validate_positive_int("log_every_steps", log_every_steps)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_accumulation_batches = 0

        total_loss = 0.0
        total_tokens = 0
        optimizer_steps_before = self.total_optimizer_steps
        batch_count = 0
        skipped_batches = 0
        for batch_index, batch in enumerate(data_loader):
            metrics = self.train_batch(batch, batch_index)
            batch_count += 1
            total_loss += metrics.loss * metrics.tokens
            total_tokens += metrics.tokens
            if (batch_index + 1) % log_every_steps == 0:
                LOGGER.info(
                    "epoch=%s batch=%s loss=%.6f tokens=%s",
                    epoch,
                    batch_index + 1,
                    metrics.loss,
                    metrics.tokens,
                )

        if batch_count == 0:
            raise TrainingError("Training data loader is empty.")
        if self._pending_accumulation_batches > 0:
            self._optimizer_step()

        return EpochMetrics(
            epoch=epoch,
            training_loss=total_loss / total_tokens,
            validation_loss=None,
            training_tokens=total_tokens,
            validation_tokens=0,
            optimizer_steps=self.total_optimizer_steps - optimizer_steps_before,
            skipped_batches=skipped_batches,
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader | None,
        epochs: int,
        validate_every_epochs: int = 1,
        log_every_steps: int = 10,
        start_epoch: int = 1,
        checkpoint_config: object | None = None,
        checkpoint_directory: Path | str | None = None,
        model_config: Mapping[str, Any] | None = None,
        vocabulary_metadata: Mapping[str, Any] | None = None,
        best_metric: float | None = None,
    ) -> TrainingResult:
        """Train for multiple epochs with optional validation."""

        _validate_positive_int("epochs", epochs)
        _validate_positive_int("validate_every_epochs", validate_every_epochs)
        _validate_positive_int("log_every_steps", log_every_steps)
        _validate_positive_int("start_epoch", start_epoch)
        results: list[EpochMetrics] = []
        checkpoint_paths: list[Path] = []
        active_best_metric = best_metric
        for epoch in range(start_epoch, start_epoch + epochs):
            training_metrics = self.train_epoch(
                train_loader,
                epoch=epoch,
                log_every_steps=log_every_steps,
            )
            validation_loss = None
            validation_tokens = 0
            if validation_loader is not None and epoch % validate_every_epochs == 0:
                validation_metrics = self.evaluate(validation_loader)
                validation_loss = validation_metrics.loss
                validation_tokens = validation_metrics.tokens
            results.append(
                EpochMetrics(
                    epoch=training_metrics.epoch,
                    training_loss=training_metrics.training_loss,
                    validation_loss=validation_loss,
                    training_tokens=training_metrics.training_tokens,
                    validation_tokens=validation_tokens,
                    optimizer_steps=training_metrics.optimizer_steps,
                    skipped_batches=training_metrics.skipped_batches,
                )
            )
            if _should_save_checkpoint(epoch, checkpoint_config, checkpoint_directory):
                latest_epoch = results[-1]
                try:
                    saved = save_managed_checkpoint(
                        _checkpoint_directory(checkpoint_config, checkpoint_directory),
                        self.model,
                        self.optimizer,
                        filename_prefix=_checkpoint_attr(
                            checkpoint_config,
                            "filename_prefix",
                        ),
                        epoch=latest_epoch.epoch,
                        global_step=self.total_optimizer_steps,
                        training_loss=latest_epoch.training_loss,
                        validation_loss=latest_epoch.validation_loss,
                        best_metric=active_best_metric,
                        keep_last=_checkpoint_attr(checkpoint_config, "keep_last"),
                        save_best=_checkpoint_attr(checkpoint_config, "save_best"),
                        monitor=_checkpoint_attr(checkpoint_config, "monitor"),
                        mode=_checkpoint_attr(checkpoint_config, "mode"),
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        model_config=model_config,
                        vocabulary_metadata=vocabulary_metadata,
                    )
                except CheckpointError as exc:
                    raise TrainingError(f"Checkpoint save failed: {exc}") from exc
                active_best_metric = saved.best_metric
                checkpoint_paths.append(saved.checkpoint_path)
                if saved.best_checkpoint_path is not None:
                    checkpoint_paths.append(saved.best_checkpoint_path)

        return TrainingResult(
            epochs=tuple(results),
            total_optimizer_steps=self.total_optimizer_steps,
            completed_epochs=len(results),
            best_metric=active_best_metric,
            checkpoint_paths=tuple(checkpoint_paths),
        )

    def _prepare_batch(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not isinstance(batch, Mapping):
            raise TrainingError("batch must be a mapping.")
        missing = {"input_ids", "target_ids"} - batch.keys()
        if missing:
            raise TrainingError(f"Batch is missing required key(s): {', '.join(sorted(missing))}.")
        input_ids = _move_tensor(batch["input_ids"], self.device, "input_ids")
        target_ids = _move_tensor(batch["target_ids"], self.device, "target_ids")
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = _move_tensor(attention_mask, self.device, "attention_mask")
        _validate_batch_tensors(input_ids, target_ids, attention_mask)
        return input_ids, target_ids, attention_mask

    def _compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise TrainingError("model must return logits as a torch.Tensor.")
        if logits.ndim != 3:
            raise TrainingError("logits must be a three-dimensional tensor.")
        if logits.shape[:2] != target_ids.shape:
            raise TrainingError("logits batch/sequence dimensions must match target_ids.")
        loss = self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )
        if not isinstance(loss, torch.Tensor):
            raise TrainingError("loss_fn must return a torch.Tensor.")
        if loss.ndim != 0:
            raise TrainingError("loss_fn must return a scalar tensor.")
        if not bool(torch.isfinite(loss).item()):
            raise TrainingError("Loss is NaN or infinite.")
        return loss

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            try:
                self.scheduler.step()
            except AttributeError as exc:
                raise TrainingError("scheduler must provide a step() method.") from exc
        self.optimizer.zero_grad(set_to_none=True)
        self.total_optimizer_steps += 1
        self._pending_accumulation_batches = 0


def _move_tensor(tensor: torch.Tensor, device: torch.device, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TrainingError(f"{name} must be a torch.Tensor.")
    try:
        return tensor.to(device)
    except RuntimeError as exc:
        raise TrainingError(f"Could not move {name} to device {device}: {exc}") from exc


def _validate_batch_tensors(
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> None:
    for name, tensor in {"input_ids": input_ids, "target_ids": target_ids}.items():
        if tensor.dtype != torch.long:
            raise TrainingError(f"{name} must use torch.long dtype.")
        if tensor.ndim != 2:
            raise TrainingError(f"{name} must be a two-dimensional tensor.")
    if input_ids.shape != target_ids.shape:
        raise TrainingError("input_ids and target_ids shapes must match.")
    if input_ids.shape[0] <= 0:
        raise TrainingError("Batch size must be greater than zero.")
    if input_ids.shape[1] <= 0:
        raise TrainingError("Sequence length must be greater than zero.")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise TrainingError("attention_mask shape must match input_ids.")
        if attention_mask.device != input_ids.device:
            raise TrainingError("attention_mask device must match input_ids.")
        if attention_mask.dtype == torch.bool:
            return
        if not attention_mask.dtype.is_floating_point and attention_mask.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TrainingError("attention_mask must be bool, integer, or floating zero/one.")
        if attention_mask.numel() > 0:
            is_zero_or_one = (attention_mask == 0) | (attention_mask == 1)
            if not bool(is_zero_or_one.all().item()):
                raise TrainingError("attention_mask values must be 0/1 or boolean.")


def _count_tokens(target_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> int:
    if attention_mask is None:
        return int(target_ids.numel())
    return int(attention_mask.to(dtype=torch.long).sum().item())


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingError(f"{name} must be an integer greater than zero.")


def _should_save_checkpoint(
    epoch: int,
    checkpoint_config: object | None,
    checkpoint_directory: Path | str | None,
) -> bool:
    if checkpoint_config is None and checkpoint_directory is None:
        return False
    if checkpoint_config is None:
        raise TrainingError("checkpoint_config is required when checkpoint_directory is provided.")
    save_every_epochs = _checkpoint_attr(checkpoint_config, "save_every_epochs")
    if (
        not isinstance(save_every_epochs, int)
        or isinstance(save_every_epochs, bool)
        or save_every_epochs <= 0
    ):
        raise TrainingError("checkpoint_config.save_every_epochs must be greater than zero.")
    return epoch % save_every_epochs == 0


def _checkpoint_directory(
    checkpoint_config: object | None,
    checkpoint_directory: Path | str | None,
) -> Path:
    if checkpoint_directory is not None:
        return Path(checkpoint_directory)
    return Path(_checkpoint_attr(checkpoint_config, "directory"))


def _checkpoint_attr(checkpoint_config: object | None, name: str) -> Any:
    if checkpoint_config is None or not hasattr(checkpoint_config, name):
        raise TrainingError(f"checkpoint_config must provide {name}.")
    return getattr(checkpoint_config, name)


__all__ = [
    "BatchMetrics",
    "EpochMetrics",
    "GPTTrainer",
    "TrainingError",
    "TrainingResult",
]
