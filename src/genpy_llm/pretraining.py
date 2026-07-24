"""Official Phase 6 GPT pretraining engine for GenPy."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash
from genpy_llm.compat import zip_strict
from genpy_llm.config import OptimizerConfig
from genpy_llm.device import select_device
from genpy_llm.gpt import GPTModel
from genpy_llm.logging_utils import setup_structured_logging
from genpy_llm.losses import GPTCrossEntropyLoss
from genpy_llm.optimizers import create_optimizer_with_metadata
from genpy_llm.performance import (
    StepTimer,
    autocast_context,
    compile_model,
    create_grad_scaler,
    normalize_mixed_precision,
    peak_memory_mb,
    reset_peak_memory,
    resolve_mixed_precision,
)
from genpy_llm.pretraining_dataset import DeterministicSequenceSampler, PackedSequenceDataset
from genpy_llm.pretraining_generation import CodeGenerationSettings, generate_code_sample

UTC = timezone.utc

LOGGER = logging.getLogger("genpy_llm.pretraining")
PRETRAINING_VERSION = 1


class PretrainingError(RuntimeError):
    """Raised when Phase 6 pretraining cannot continue."""


@dataclass(frozen=True)
class Phase6ModelConfig:
    """Decoder-only Transformer model settings."""

    vocabulary_size: int
    context_length: int
    hidden_size: int
    ffn_size: int
    decoder_layers: int
    attention_heads: int
    dropout: float
    attention_dropout: float
    residual_dropout: float
    activation: str
    layer_norm_epsilon: float
    positional_embedding: str
    tied_embedding_weights: bool
    use_bias: bool
    initialization_std: float
    gradient_checkpointing: bool
    torch_compile: bool
    compile_mode: str
    flash_attention: str


@dataclass(frozen=True)
class Phase6DataConfig:
    """Packed pretraining dataset settings."""

    shard_pattern: str
    shard_index: Path
    training_manifest: Path
    tokenizer: Path
    validation_fraction: float
    batch_size: int
    dataloader_workers: int
    pin_memory: bool
    prefetch_factor: int | None
    mmap: bool
    shuffle: bool
    seed: int
    distributed_ready: bool


@dataclass(frozen=True)
class Phase6TrainingConfig:
    """Training loop settings."""

    seed: int
    max_steps: int
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
    resume_from: Path | None


@dataclass(frozen=True)
class Phase6SchedulerConfig:
    """Cosine schedule settings."""

    warmup_steps: int
    minimum_learning_rate_ratio: float


@dataclass(frozen=True)
class Phase6CheckpointConfig:
    """Checkpoint artifact settings."""

    directory: Path
    step_prefix: str
    best_filename: str
    last_filename: str
    monitor: str
    mode: str


@dataclass(frozen=True)
class Phase6OutputConfig:
    """Output directories for metrics, samples, and TensorBoard."""

    metrics_directory: Path
    samples_directory: Path
    tensorboard_directory: Path
    log_file: Path


@dataclass(frozen=True)
class Phase6Config:
    """Complete Phase 6 pretraining configuration."""

    project_root: Path
    model: Phase6ModelConfig
    data: Phase6DataConfig
    training: Phase6TrainingConfig
    optimizer: OptimizerConfig
    scheduler: Phase6SchedulerConfig
    checkpoint: Phase6CheckpointConfig
    generation: CodeGenerationSettings
    outputs: Phase6OutputConfig
    log_level: str


@dataclass(frozen=True)
class PretrainingResult:
    """Summary returned after training."""

    global_step: int
    best_metric: float | None
    last_checkpoint: Path | None
    best_checkpoint: Path | None
    metrics_path: Path


def compute_scheduler_total_steps(
    *,
    dataset_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    max_steps: int | None = None,
) -> int:
    """Return the optimizer-step horizon for a cosine/warmup schedule.

    An explicit ``max_steps`` always wins. Otherwise the horizon is derived from
    how many optimizer steps a full training run actually performs: dataset
    examples per epoch, divided into micro-batches, divided into optimizer
    steps after gradient accumulation, multiplied by the number of epochs.
    Passing ``epochs`` alone (with no dataset size) previously stood in for
    this and silently produced a 1-step schedule for the common
    ``epochs=1, max_steps=None`` configuration, collapsing cosine decay to its
    floor almost immediately.
    """

    if max_steps is not None:
        if max_steps <= 0:
            raise PretrainingError("max_steps must be positive.")
        return int(max_steps)
    if dataset_size <= 0:
        raise PretrainingError("dataset_size must be positive.")
    if batch_size <= 0:
        raise PretrainingError("batch_size must be positive.")
    if gradient_accumulation_steps <= 0:
        raise PretrainingError("gradient_accumulation_steps must be positive.")
    if epochs <= 0:
        raise PretrainingError("epochs must be positive.")
    return max(1, epochs * dataset_size // batch_size // gradient_accumulation_steps)


class CosineWarmupScheduler:
    """AdamW learning-rate schedule with linear warmup and cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_steps: int,
        warmup_steps: int,
        minimum_learning_rate_ratio: float,
        last_step: int = 0,
    ) -> None:
        if max_steps <= 0:
            raise PretrainingError("max_steps must be positive.")
        if warmup_steps < 0:
            raise PretrainingError("warmup_steps must be non-negative.")
        if not 0 <= minimum_learning_rate_ratio <= 1:
            raise PretrainingError("minimum_learning_rate_ratio must be between 0 and 1.")
        self.optimizer = optimizer
        self.max_steps = int(max_steps)
        self.warmup_steps = int(warmup_steps)
        self.minimum_learning_rate_ratio = float(minimum_learning_rate_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_count = int(last_step)
        self._apply()

    def step(self) -> None:
        self.step_count += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "minimum_learning_rate_ratio": self.minimum_learning_rate_ratio,
            "base_lrs": self.base_lrs,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.step_count = int(state.get("step_count", 0))
        loaded_lrs = state.get("base_lrs")
        if isinstance(loaded_lrs, list) and len(loaded_lrs) == len(self.base_lrs):
            self.base_lrs = [float(value) for value in loaded_lrs]
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _apply(self) -> None:
        factor = self._factor(self.step_count)
        for base_lr, group in zip_strict(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def _factor(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return max(1e-12, step / self.warmup_steps)
        progress = min(
            1.0,
            max(0.0, (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.minimum_learning_rate_ratio
        return floor + (1.0 - floor) * cosine


class Phase6Trainer:
    """Production pretraining loop for packed GenPy binary shards."""

    def __init__(self, config: Phase6Config) -> None:
        self.config = config
        LOGGER.info("phase6_trainer_init_started")
        _seed_everything(config.training.seed)
        self.device = select_device(config.training.device)
        LOGGER.info(
            "phase6_device_selected requested=%s effective=%s",
            config.training.device,
            self.device,
        )
        self.mixed_precision = resolve_mixed_precision(
            config.training.mixed_precision,
            self.device,
            logger=LOGGER,
        )
        self.dataloader_workers = self._effective_dataloader_workers()
        self.dataloader_prefetch_factor = self._effective_prefetch_factor()
        self.dataloader_pin_memory = self._effective_pin_memory()
        LOGGER.info(
            "phase6_dataloader_effective_config workers=%d prefetch_factor=%s pin_memory=%s",
            self.dataloader_workers,
            self.dataloader_prefetch_factor,
            self.dataloader_pin_memory,
        )
        LOGGER.info("phase6_tokenizer_loading_started path=%s", config.data.tokenizer)
        self.tokenizer = CodeTokenizer.from_file(config.data.tokenizer)
        LOGGER.info(
            "phase6_tokenizer_loaded vocab_size=%d path=%s",
            self.tokenizer.vocab_size,
            config.data.tokenizer,
        )
        LOGGER.info("phase6_manifest_reading_started path=%s", config.data.training_manifest)
        self._validate_tokenizer_manifest()
        LOGGER.info("phase6_manifest_read")
        LOGGER.info("phase6_model_creating_started")
        self.model = create_phase6_model(config.model, self.tokenizer)
        LOGGER.info(
            "phase6_model_created layers=%d hidden_size=%d context_length=%d",
            config.model.decoder_layers,
            config.model.hidden_size,
            config.model.context_length,
        )
        if config.model.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        LOGGER.info("phase6_model_to_device_started device=%s", self.device)
        self.model.to(self.device)
        LOGGER.info("phase6_model_to_device_completed device=%s", self.device)
        if config.model.torch_compile:
            LOGGER.info("phase6_model_compile_started mode=%s", config.model.compile_mode)
            self.model = compile_model(self.model, enabled=True, mode=config.model.compile_mode)
            LOGGER.info("phase6_model_compile_completed")
        LOGGER.info("phase6_optimizer_creating_started")
        self.optimizer, self.optimizer_metadata = create_optimizer_with_metadata(
            self.model,
            config.optimizer,
        )
        LOGGER.info("phase6_optimizer_created type=%s", config.optimizer.type)
        LOGGER.info("phase6_scheduler_creating_started")
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            max_steps=config.training.max_steps,
            warmup_steps=config.scheduler.warmup_steps,
            minimum_learning_rate_ratio=config.scheduler.minimum_learning_rate_ratio,
        )
        LOGGER.info("phase6_scheduler_created")
        self.scaler = create_grad_scaler(self.mixed_precision, self.device)
        self.loss_fn = GPTCrossEntropyLoss(
            padding_idx=self.tokenizer.pad_token_id,
            ignore_padding=True,
        )
        self.global_step = 0
        self.epoch = 0
        self.best_metric: float | None = None
        self.latest_training_loss: float | None = None
        self.latest_validation_loss: float | None = None
        self.micro_step = 0
        self.metrics_path = config.outputs.metrics_directory / "training_metrics.csv"
        self.json_metrics_path = config.outputs.metrics_directory / "training_metrics.jsonl"
        self.tensorboard_writer = _tensorboard_writer(config.outputs.tensorboard_directory)
        LOGGER.info("phase6_dataset_construction_started")
        self.dataset, self.train_indices, self.validation_indices = self._datasets()
        LOGGER.info(
            "phase6_dataset_constructed total_sequences=%d train_sequences=%d "
            "validation_sequences=%d",
            len(self.dataset),
            len(self.train_indices),
            len(self.validation_indices),
        )
        self._first_batch_logged = False
        self._first_batch_request_logged = False
        self._first_device_transfer_logged = False
        self._first_forward_logged = False
        self._first_loss_logged = False
        self._first_backward_logged = False
        self._first_optimizer_step_logged = False
        self._first_log_output_logged = False
        if config.training.resume:
            self._resume()
        LOGGER.info("phase6_trainer_init_completed")

    def train(self) -> PretrainingResult:
        """Run pretraining until max_steps."""

        self._prepare_outputs()
        LOGGER.info(
            "phase6_training_started max_steps=%d device=%s mixed_precision=%s",
            self.config.training.max_steps,
            self.device,
            self.mixed_precision,
        )
        LOGGER.info("phase6_training_loop_entered")
        while self.global_step < self.config.training.max_steps:
            self.epoch += 1
            LOGGER.info("phase6_epoch_started epoch=%d", self.epoch)
            sampler = DeterministicSequenceSampler(
                Subset(self.dataset, self.train_indices),
                shuffle=self.config.data.shuffle,
                seed=self.config.data.seed,
            )
            sampler.set_epoch(self.epoch)
            LOGGER.info("phase6_dataloader_construction_started epoch=%d", self.epoch)
            loader = self._loader(Subset(self.dataset, self.train_indices), sampler=sampler)
            LOGGER.info(
                "phase6_dataloader_constructed epoch=%d batches=%d",
                self.epoch,
                len(loader),
            )
            loader_iter = iter(loader)
            while self.global_step < self.config.training.max_steps:
                if self.global_step >= self.config.training.max_steps:
                    break
                if not self._first_batch_request_logged:
                    LOGGER.info("phase6_first_batch_requesting")
                    self._first_batch_request_logged = True
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    LOGGER.info("phase6_dataloader_exhausted epoch=%d", self.epoch)
                    break
                if not self._first_batch_logged:
                    LOGGER.info(
                        "phase6_first_batch_received input_shape=%s target_shape=%s mask_shape=%s",
                        tuple(batch["input_ids"].shape),
                        tuple(batch["target_ids"].shape),
                        tuple(batch["attention_mask"].shape),
                    )
                    self._first_batch_logged = True
                metrics = self._train_micro_batch(batch)
                if metrics is None:
                    continue
                self.global_step += 1
                self.latest_training_loss = metrics["loss"]
                self._log_metrics(metrics)
                if not self._first_log_output_logged:
                    LOGGER.info(
                        "phase6_first_log_output step=%d loss=%.6f",
                        self.global_step,
                        metrics["loss"],
                    )
                    self._first_log_output_logged = True
                if self.global_step % self.config.training.log_every_steps == 0:
                    LOGGER.info(
                        "step=%d loss=%.6f lr=%.8f tokens_per_sec=%.2f",
                        self.global_step,
                        metrics["loss"],
                        metrics["learning_rate"],
                        metrics["tokens_per_second"],
                    )
                if self._should_validate():
                    self.latest_validation_loss = self.evaluate()
                    self._write_samples()
                if self.global_step % self.config.training.save_every_steps == 0:
                    self._save_checkpoint()
        last = self._save_checkpoint()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        return PretrainingResult(
            global_step=self.global_step,
            best_metric=self.best_metric,
            last_checkpoint=last,
            best_checkpoint=self.config.checkpoint.directory / self.config.checkpoint.best_filename
            if (self.config.checkpoint.directory / self.config.checkpoint.best_filename).is_file()
            else None,
            metrics_path=self.metrics_path,
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run bounded validation and return average loss."""

        if not self.validation_indices:
            return 0.0
        loader = self._loader(
            Subset(self.dataset, self.validation_indices),
            sampler=None,
            shuffle=False,
        )
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index >= self.config.training.validation_steps:
                break
            input_ids, target_ids, attention_mask = self._batch_to_device(batch)
            with autocast_context(self.mixed_precision, self.device):
                logits = self.model(input_ids, padding_mask=attention_mask)
                loss = self.loss_fn(logits, target_ids)
            tokens = int(attention_mask.sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
        if was_training:
            self.model.train()
        validation_loss = total_loss / total_tokens if total_tokens else 0.0
        perplexity = math.exp(min(20.0, validation_loss)) if validation_loss else 0.0
        elapsed = time.perf_counter() - started
        payload = {
            "step": self.global_step,
            "validation_loss": validation_loss,
            "perplexity": perplexity,
            "validation_tokens": total_tokens,
            "evaluation_seconds": elapsed,
        }
        self._append_json(payload | {"type": "validation"})
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar("validation/loss", validation_loss, self.global_step)
            self.tensorboard_writer.add_scalar(
                "validation/perplexity",
                perplexity,
                self.global_step,
            )
        LOGGER.info(
            "validation step=%d loss=%.6f perplexity=%.4f tokens=%d",
            self.global_step,
            validation_loss,
            perplexity,
            total_tokens,
        )
        return validation_loss

    def _train_micro_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any] | None:
        self.model.train()
        self.micro_step += 1
        if not self._first_optimizer_step_logged:
            LOGGER.info("phase6_micro_batch_started micro_step=%d", self.micro_step)
        input_ids, target_ids, attention_mask = self._batch_to_device(batch)
        reset_peak_memory(self.device)
        with StepTimer(self.device) as timer:
            with autocast_context(self.mixed_precision, self.device):
                if not self._first_forward_logged:
                    LOGGER.info("phase6_first_forward_pass_started")
                logits = self.model(input_ids, padding_mask=attention_mask)
                if not self._first_forward_logged:
                    LOGGER.info("phase6_first_forward_pass_completed")
                    self._first_forward_logged = True
                loss = self.loss_fn(logits, target_ids)
                if not self._first_loss_logged:
                    LOGGER.info("phase6_first_loss_calculated")
                    self._first_loss_logged = True
                scaled_loss = loss / self.config.training.gradient_accumulation_steps
            if self.scaler is not None:
                if not self._first_backward_logged:
                    LOGGER.info("phase6_first_backward_pass_started")
                self.scaler.scale(scaled_loss).backward()
            else:
                if not self._first_backward_logged:
                    LOGGER.info("phase6_first_backward_pass_started")
                scaled_loss.backward()
            if not self._first_backward_logged:
                LOGGER.info("phase6_first_backward_pass_completed")
                self._first_backward_logged = True
            if self.micro_step % self.config.training.gradient_accumulation_steps != 0:
                if not self._first_optimizer_step_logged:
                    LOGGER.info(
                        "phase6_gradient_accumulation_waiting micro_step=%d required=%d",
                        self.micro_step,
                        self.config.training.gradient_accumulation_steps,
                    )
                return None
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = self._clip_gradients()
            if self.scaler is not None:
                if not self._first_optimizer_step_logged:
                    LOGGER.info("phase6_first_optimizer_step_started")
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if not self._first_optimizer_step_logged:
                    LOGGER.info("phase6_first_optimizer_step_started")
                self.optimizer.step()
            if not self._first_optimizer_step_logged:
                LOGGER.info("phase6_first_optimizer_step_completed")
                self._first_optimizer_step_logged = True
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        tokens = int(attention_mask.sum().item())
        elapsed = max(timer.elapsed_seconds, 1e-12)
        return {
            "type": "train",
            "step": self.global_step + 1,
            "epoch": self.epoch,
            "loss": float(loss.detach().item()),
            "perplexity": math.exp(min(20.0, float(loss.detach().item()))),
            "tokens": tokens,
            "batch_size": int(input_ids.shape[0]),
            "tokens_per_second": tokens / elapsed,
            "examples_per_second": int(input_ids.shape[0]) / elapsed,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "gradient_norm": grad_norm,
            "gpu_memory_mb": peak_memory_mb(self.device),
            "elapsed_seconds": elapsed,
        }

    def _batch_to_device(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._first_device_transfer_logged:
            LOGGER.info("phase6_first_batch_to_device_started device=%s", self.device)
        moved = (
            batch["input_ids"].to(self.device, non_blocking=True),
            batch["target_ids"].to(self.device, non_blocking=True),
            batch["attention_mask"].to(self.device, non_blocking=True),
        )
        if not self._first_device_transfer_logged:
            LOGGER.info("phase6_first_batch_to_device_completed device=%s", self.device)
            self._first_device_transfer_logged = True
        return moved

    def _clip_gradients(self) -> float:
        if self.config.training.max_grad_norm is None:
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if not parameters:
                return 0.0
            norms = torch.stack([p.grad.detach().norm() for p in parameters])
            return float(torch.linalg.vector_norm(norms).item())
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.max_grad_norm,
        )
        return float(norm.item())

    def _save_checkpoint(self) -> Path:
        directory = self.config.checkpoint.directory
        directory.mkdir(parents=True, exist_ok=True)
        step_path = directory / f"{self.config.checkpoint.step_prefix}_{self.global_step:05d}.pt"
        extra_state = {
            "pretraining_version": PRETRAINING_VERSION,
            "manifest_hash": _file_hash(self.config.data.training_manifest),
            "shard_index_hash": _file_hash(self.config.data.shard_index),
            "tokenizer_hash": tokenizer_file_hash(self.config.data.tokenizer),
            "optimizer": asdict(self.optimizer_metadata),
            "training_statistics": {
                "global_step": self.global_step,
                "latest_training_loss": self.latest_training_loss,
                "latest_validation_loss": self.latest_validation_loss,
            },
        }
        save_checkpoint(
            step_path,
            self.model,
            self.optimizer,
            epoch=self.epoch,
            global_step=self.global_step,
            training_loss=self.latest_training_loss,
            validation_loss=self.latest_validation_loss,
            best_metric=self.best_metric,
            scheduler=self.scheduler,
            scaler=self.scaler,
            model_config=asdict(self.config.model),
            vocabulary_metadata={
                "tokenizer": str(self.config.data.tokenizer),
                "tokenizer_sha256": extra_state["tokenizer_hash"],
            },
            extra_state=extra_state,
        )
        last = directory / self.config.checkpoint.last_filename
        shutil.copy2(step_path, last)
        metric = (
            self.latest_training_loss
            if self.config.checkpoint.monitor == "training_loss"
            else self.latest_validation_loss
        )
        if metric is not None and _is_better(metric, self.best_metric, self.config.checkpoint.mode):
            self.best_metric = metric
            shutil.copy2(step_path, directory / self.config.checkpoint.best_filename)
        self._rotate_checkpoints()
        return last

    def _resume(self) -> None:
        path = self.config.training.resume_from
        if path is None:
            candidate = self.config.checkpoint.directory / self.config.checkpoint.last_filename
            path = candidate if candidate.is_file() else None
        if path is None:
            return
        loaded = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.epoch = loaded.epoch
        self.global_step = loaded.global_step
        self.micro_step = self.global_step * self.config.training.gradient_accumulation_steps
        self.best_metric = loaded.best_metric
        self.latest_training_loss = loaded.training_loss
        self.latest_validation_loss = loaded.validation_loss
        LOGGER.info("resumed_checkpoint=%s step=%d", loaded.checkpoint_path, self.global_step)

    def _datasets(self) -> tuple[PackedSequenceDataset, list[int], list[int]]:
        LOGGER.info("phase6_index_reading_started path=%s", self.config.data.shard_index)
        dataset = PackedSequenceDataset(
            self.config.data.shard_pattern,
            tokenizer=self.tokenizer,
            manifest_path=self.config.data.shard_index,
            sequence_length=self.config.model.context_length + 1,
            mmap=self.config.data.mmap,
        )
        LOGGER.info(
            "phase6_index_read shards=%d sequence_length=%d mmap=%s",
            len(dataset.shards),
            dataset.sequence_length,
            dataset.mmap,
        )
        indices = list(range(len(dataset)))
        random.Random(self.config.data.seed).shuffle(indices)
        validation_count = int(len(indices) * self.config.data.validation_fraction)
        validation = indices[:validation_count]
        train = indices[validation_count:] or indices
        return dataset, train, validation

    def _loader(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        sampler: torch.utils.data.Sampler[int] | None,
        shuffle: bool | None = None,
    ) -> DataLoader:
        kwargs: dict[str, Any] = {}
        if self.dataloader_workers > 0 and self.dataloader_prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            sampler=sampler,
            shuffle=bool(shuffle) if sampler is None else False,
            num_workers=self.dataloader_workers,
            pin_memory=self.dataloader_pin_memory,
            **kwargs,
        )

    def _effective_dataloader_workers(self) -> int:
        workers = self.config.data.dataloader_workers
        if self.device.type == "mps" and workers != 0:
            LOGGER.warning(
                "Apple MPS/macOS DataLoader multiprocessing can deadlock; "
                "forcing dataloader_workers=0."
            )
            return 0
        return workers

    def _effective_prefetch_factor(self) -> int | None:
        if self.dataloader_workers <= 0:
            if self.config.data.prefetch_factor is not None:
                LOGGER.warning("prefetch_factor requires dataloader_workers > 0; disabling it.")
            return None
        return self.config.data.prefetch_factor

    def _effective_pin_memory(self) -> bool:
        if self.device.type == "cuda":
            return self.config.data.pin_memory
        if self.config.data.pin_memory:
            LOGGER.warning("pin_memory is only useful for CUDA; disabling it on %s.", self.device)
        return False

    def _should_validate(self) -> bool:
        return (
            bool(self.validation_indices)
            and self.config.training.validate_every_steps > 0
            and self.global_step % self.config.training.validate_every_steps == 0
        )

    def _log_metrics(self, metrics: Mapping[str, Any]) -> None:
        self.config.outputs.metrics_directory.mkdir(parents=True, exist_ok=True)
        exists = self.metrics_path.is_file()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sorted(metrics))
            if not exists:
                writer.writeheader()
            writer.writerow({key: metrics.get(key) for key in sorted(metrics)})
        self._append_json(metrics)
        if self.tensorboard_writer is not None:
            step = int(metrics["step"])
            self.tensorboard_writer.add_scalar("train/loss", float(metrics["loss"]), step)
            self.tensorboard_writer.add_scalar(
                "train/perplexity",
                float(metrics["perplexity"]),
                step,
            )
            self.tensorboard_writer.add_scalar(
                "train/learning_rate",
                float(metrics["learning_rate"]),
                step,
            )
            self.tensorboard_writer.add_scalar(
                "train/tokens_per_second",
                float(metrics["tokens_per_second"]),
                step,
            )

    def _append_json(self, payload: Mapping[str, Any]) -> None:
        self.config.outputs.metrics_directory.mkdir(parents=True, exist_ok=True)
        with self.json_metrics_path.open("a", encoding="utf-8", newline="\n") as file:
            json.dump(
                {"timestamp": _timestamp(), **dict(payload)},
                file,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            file.write("\n")

    def _write_samples(self) -> None:
        self.config.outputs.samples_directory.mkdir(parents=True, exist_ok=True)
        path = self.config.outputs.samples_directory / f"step_{self.global_step:05d}.json"
        samples = [
            asdict(
                generate_code_sample(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    device=self.device,
                    context_length=self.config.model.context_length,
                    settings=self.config.generation,
                )
            )
            for prompt in self.config.generation.prompts
        ]
        _atomic_json(path, {"step": self.global_step, "samples": samples})

    def _prepare_outputs(self) -> None:
        for path in (
            self.config.checkpoint.directory,
            self.config.outputs.metrics_directory,
            self.config.outputs.samples_directory,
            self.config.outputs.tensorboard_directory,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _rotate_checkpoints(self) -> None:
        keep_last = self.config.training.keep_last
        if keep_last <= 0:
            return
        checkpoints = sorted(
            self.config.checkpoint.directory.glob(f"{self.config.checkpoint.step_prefix}_*.pt"),
            key=lambda path: path.name,
            reverse=True,
        )
        for path in checkpoints[keep_last:]:
            path.unlink(missing_ok=True)

    def _validate_tokenizer_manifest(self) -> None:
        manifest = json.loads(self.config.data.training_manifest.read_text(encoding="utf-8"))
        expected = tokenizer_file_hash(self.config.data.tokenizer)
        if manifest.get("tokenizer_hash") not in {None, expected}:
            raise PretrainingError("Training manifest tokenizer hash does not match tokenizer.")


def load_phase6_config(
    training_config: Path | str = "configs/training.yaml",
    *,
    model_config: Path | str = "configs/model.yaml",
    optimizer_config: Path | str = "configs/optimizer.yaml",
    generation_config: Path | str = "configs/generation.yaml",
) -> Phase6Config:
    """Load modular Phase 6 YAML configuration."""

    root = Path(__file__).resolve().parents[2]
    training_raw = _load_yaml(_resolve(root, training_config))
    model_raw = _load_yaml(_resolve(root, model_config))
    optimizer_raw = _load_yaml(_resolve(root, optimizer_config))
    generation_raw = _load_yaml(_resolve(root, generation_config))
    section = _mapping(training_raw.get("pretraining", {}), "pretraining")
    data_raw = _mapping(section.get("data", {}), "pretraining.data")
    train_raw = _mapping(section.get("training", {}), "pretraining.training")
    checkpoint_raw = _mapping(section.get("checkpoint", {}), "pretraining.checkpoint")
    outputs_raw = _mapping(section.get("outputs", {}), "pretraining.outputs")
    scheduler_raw = _mapping(section.get("scheduler", {}), "pretraining.scheduler")
    model_section = _mapping(model_raw.get("model", {}), "model")
    generation_section = _mapping(generation_raw.get("generation", {}), "generation")
    optimizer_section = _mapping(optimizer_raw.get("optimizer", {}), "optimizer")
    return Phase6Config(
        project_root=root,
        model=Phase6ModelConfig(
            vocabulary_size=int(model_section.get("vocabulary_size", 32000)),
            context_length=int(model_section.get("context_length", 1024)),
            hidden_size=int(model_section.get("hidden_size", 512)),
            ffn_size=int(model_section.get("ffn_size", 2048)),
            decoder_layers=int(model_section.get("decoder_layers", 6)),
            attention_heads=int(model_section.get("attention_heads", 8)),
            dropout=float(model_section.get("dropout", 0.1)),
            attention_dropout=float(model_section.get("attention_dropout", 0.1)),
            residual_dropout=float(model_section.get("residual_dropout", 0.1)),
            activation=str(model_section.get("activation", "gelu")),
            layer_norm_epsilon=float(model_section.get("layer_norm_epsilon", 1e-5)),
            positional_embedding=str(model_section.get("positional_embedding", "learned")),
            tied_embedding_weights=bool(model_section.get("tied_embedding_weights", True)),
            use_bias=bool(model_section.get("use_bias", True)),
            initialization_std=float(model_section.get("initialization_std", 0.02)),
            gradient_checkpointing=bool(model_section.get("gradient_checkpointing", False)),
            torch_compile=bool(model_section.get("torch_compile", False)),
            compile_mode=str(model_section.get("compile_mode", "default")),
            flash_attention=str(model_section.get("flash_attention", "auto")),
        ),
        data=Phase6DataConfig(
            shard_pattern=str(
                _resolve(root, data_raw.get("shard_pattern", "data/pretraining/shard_*.bin"))
            ),
            shard_index=_resolve(root, data_raw.get("shard_index", "data/pretraining/index.json")),
            training_manifest=_resolve(
                root,
                data_raw.get("manifest", "data/pretraining/manifest.json"),
            ),
            tokenizer=_resolve(root, data_raw.get("tokenizer", "data/tokenizer/tokenizer.json")),
            validation_fraction=float(data_raw.get("validation_fraction", 0.01)),
            batch_size=int(data_raw.get("batch_size", 4)),
            dataloader_workers=int(data_raw.get("dataloader_workers", 0)),
            pin_memory=bool(data_raw.get("pin_memory", True)),
            prefetch_factor=(
                int(data_raw["prefetch_factor"])
                if data_raw.get("prefetch_factor") is not None
                else None
            ),
            mmap=bool(data_raw.get("mmap", True)),
            shuffle=bool(data_raw.get("shuffle", True)),
            seed=int(data_raw.get("seed", 42)),
            distributed_ready=bool(data_raw.get("distributed_ready", True)),
        ),
        training=Phase6TrainingConfig(
            seed=int(train_raw.get("seed", 42)),
            max_steps=int(train_raw.get("max_steps", 100000)),
            gradient_accumulation_steps=int(train_raw.get("gradient_accumulation_steps", 1)),
            max_grad_norm=(
                float(train_raw["max_grad_norm"])
                if train_raw.get("max_grad_norm") is not None
                else None
            ),
            device=str(train_raw.get("device", "auto")),
            mixed_precision=_configured_mixed_precision(train_raw),
            log_every_steps=int(train_raw.get("log_every_steps", 10)),
            save_every_steps=int(train_raw.get("save_every_steps", 1000)),
            validate_every_steps=int(train_raw.get("validate_every_steps", 1000)),
            validation_steps=int(train_raw.get("validation_steps", 100)),
            keep_last=int(train_raw.get("keep_last", 3)),
            resume=bool(train_raw.get("resume", True)),
            resume_from=(
                _resolve(root, train_raw["resume_from"])
                if train_raw.get("resume_from") is not None
                else None
            ),
        ),
        optimizer=OptimizerConfig(
            type=str(optimizer_section.get("type", "adamw")),
            learning_rate=float(optimizer_section.get("learning_rate", 3e-4)),
            weight_decay=float(optimizer_section.get("weight_decay", 0.1)),
            beta1=float(optimizer_section.get("beta1", 0.9)),
            beta2=float(optimizer_section.get("beta2", 0.95)),
            epsilon=float(optimizer_section.get("epsilon", 1e-8)),
            separate_weight_decay=bool(optimizer_section.get("separate_weight_decay", True)),
        ),
        scheduler=Phase6SchedulerConfig(
            warmup_steps=int(scheduler_raw.get("warmup_steps", 1000)),
            minimum_learning_rate_ratio=float(
                scheduler_raw.get("minimum_learning_rate_ratio", 0.1)
            ),
        ),
        checkpoint=Phase6CheckpointConfig(
            directory=_resolve(root, checkpoint_raw.get("directory", "checkpoints")),
            step_prefix=str(checkpoint_raw.get("step_prefix", "step")),
            best_filename=str(checkpoint_raw.get("best_filename", "best_model.pt")),
            last_filename=str(checkpoint_raw.get("last_filename", "last_checkpoint.pt")),
            monitor=str(checkpoint_raw.get("monitor", "validation_loss")),
            mode=str(checkpoint_raw.get("mode", "min")),
        ),
        generation=CodeGenerationSettings(
            prompts=tuple(generation_section.get("prompts", ("def fibonacci(n):",))),
            max_new_tokens=int(generation_section.get("max_new_tokens", 128)),
            temperature=float(generation_section.get("temperature", 0.8)),
            top_k=(
                int(generation_section["top_k"])
                if generation_section.get("top_k") is not None
                else None
            ),
            top_p=(
                float(generation_section["top_p"])
                if generation_section.get("top_p") is not None
                else None
            ),
            do_sample=bool(generation_section.get("do_sample", True)),
            repetition_penalty=float(generation_section.get("repetition_penalty", 1.0)),
            stop_tokens=tuple(generation_section.get("stop_tokens", ("<eos>",))),
        ),
        outputs=Phase6OutputConfig(
            metrics_directory=_resolve(root, outputs_raw.get("metrics_directory", "metrics")),
            samples_directory=_resolve(
                root,
                outputs_raw.get("samples_directory", "generated_samples"),
            ),
            tensorboard_directory=_resolve(
                root,
                outputs_raw.get("tensorboard_directory", "tensorboard"),
            ),
            log_file=_resolve(root, outputs_raw.get("log_file", "logs/pretraining.jsonl")),
        ),
        log_level=str(
            _mapping(training_raw.get("logging", {}), "logging").get("level", "INFO")
        ).upper(),
    )


def create_phase6_model(config: Phase6ModelConfig, tokenizer: CodeTokenizer) -> GPTModel:
    """Create the configured decoder-only Transformer using existing GPTModel."""

    if config.positional_embedding not in {"learned", "sinusoidal", "rotary"}:
        raise PretrainingError("positional_embedding must be learned, sinusoidal, or rotary.")
    if config.flash_attention not in {"auto", "disabled"}:
        raise PretrainingError("flash_attention must be auto or disabled.")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.hidden_size,
        num_heads=config.attention_heads,
        num_layers=config.decoder_layers,
        context_length=config.context_length,
        feed_forward_hidden_dim=config.ffn_size,
        padding_idx=tokenizer.pad_token_id,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        feed_forward_dropout=config.dropout,
        residual_dropout=config.residual_dropout,
        normalization_epsilon=config.layer_norm_epsilon,
        activation=config.activation,
        use_bias=config.use_bias,
        positional_encoding_type=config.positional_embedding,
        tie_embeddings=config.tied_embedding_weights,
        initialization_std=config.initialization_std,
    )
    if config.vocabulary_size != tokenizer.vocab_size:
        LOGGER.warning(
            "configured vocabulary_size=%d but tokenizer vocab_size=%d; tokenizer wins",
            config.vocabulary_size,
            tokenizer.vocab_size,
        )
    return model


def run_phase6_pretraining(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Phase 6 pretraining."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train the GenPy base GPT model from packed shards."
    )
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--optimizer-config", type=Path, default=Path("configs/optimizer.yaml"))
    parser.add_argument("--generation-config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        LOGGER.info(
            "phase6_config_loading_started training_config=%s model_config=%s "
            "optimizer_config=%s generation_config=%s",
            args.training_config,
            args.model_config,
            args.optimizer_config,
            args.generation_config,
        )
        config = load_phase6_config(
            args.training_config,
            model_config=args.model_config,
            optimizer_config=args.optimizer_config,
            generation_config=args.generation_config,
        )
        LOGGER.info("phase6_config_loaded")
        has_overrides = (
            args.max_steps is not None
            or args.device is not None
            or args.resume_from is not None
            or args.no_resume
        )
        if has_overrides:
            config = _override_config(config, args)
            LOGGER.info("phase6_cli_overrides_applied")
        setup_structured_logging(config.outputs.log_file, config.log_level)
        LOGGER.info(
            "phase6_logging_configured log_file=%s level=%s",
            config.outputs.log_file,
            config.log_level,
        )
        result = Phase6Trainer(config).train()
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Phase 6 pretraining failed")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Phase 6 pretraining complete")
    print(f"Global step: {result.global_step}")
    print(f"Last checkpoint: {result.last_checkpoint}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Metrics: {result.metrics_path}")
    return 0


def _override_config(config: Phase6Config, args: argparse.Namespace) -> Phase6Config:
    from dataclasses import replace

    training = config.training
    if args.max_steps is not None:
        training = replace(training, max_steps=args.max_steps)
    if args.device is not None:
        training = replace(training, device=args.device)
    if args.resume_from is not None:
        training = replace(training, resume_from=args.resume_from, resume=True)
    if args.no_resume:
        training = replace(training, resume=False, resume_from=None)
    return replace(config, training=training)


def _configured_mixed_precision(training: Mapping[str, Any]) -> str:
    if "mixed_precision" in training:
        return normalize_mixed_precision(str(training["mixed_precision"]))
    if "precision" in training:
        return normalize_mixed_precision(
            str(training["precision"]),
            allow_fp32_alias=True,
        )
    return "none"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PretrainingError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PretrainingError(f"Config must be a YAML mapping: {path}")
    return payload


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PretrainingError(f"{name} must be a mapping.")
    return value


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise PretrainingError("Configured path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _is_better(metric: float, best_metric: float | None, mode: str) -> bool:
    if best_metric is None:
        return True
    if mode == "min":
        return metric < best_metric
    if mode == "max":
        return metric > best_metric
    raise PretrainingError("checkpoint mode must be min or max.")


def _tensorboard_writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CosineWarmupScheduler",
    "compute_scheduler_total_steps",
    "Phase6Config",
    "Phase6Trainer",
    "PretrainingError",
    "PretrainingResult",
    "create_phase6_model",
    "load_phase6_config",
    "run_phase6_pretraining",
]
