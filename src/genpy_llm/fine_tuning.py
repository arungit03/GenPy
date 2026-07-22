"""Supervised fine-tuning utilities for GenPy LLM."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash
from genpy_llm.config import AppConfig, FineTuningConfig, OptimizerConfig
from genpy_llm.conversation_formatter import ConversationTemplate
from genpy_llm.device import select_device
from genpy_llm.evaluation import EvaluationMetrics, evaluation_metrics
from genpy_llm.gpt import GPTModel, create_gpt_model
from genpy_llm.instruction_dataset import InstructionDataset
from genpy_llm.instruction_generation import format_generation_prompt, generation_prompts
from genpy_llm.losses import create_loss_function
from genpy_llm.optimizers import create_optimizer, create_optimizer_with_metadata
from genpy_llm.performance import (
    StepTimer,
    autocast_context,
    compile_model,
    create_grad_scaler,
    peak_memory_mb,
    reset_peak_memory,
    resolve_mixed_precision,
)
from genpy_llm.pretraining import CosineWarmupScheduler, Phase6ModelConfig, create_phase6_model
from genpy_llm.pretraining_generation import CodeGenerationSettings, generate_code_sample
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.training import EpochMetrics, GPTTrainer
from genpy_llm.vocabulary import Vocabulary

INSTRUCTION_MARKER = "<INSTRUCTION>"
RESPONSE_MARKER = "<RESPONSE>"


class FineTuningError(ValueError):
    """Raised when fine-tuning data or setup is invalid."""


@dataclass(frozen=True)
class FineTuningExample:
    """One shifted fine-tuning example."""

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class FineTuningStats:
    """Summary of prepared fine-tuning examples."""

    source_records: int
    usable_records: int
    skipped_records: int
    truncated_records: int
    train_examples: int
    validation_examples: int


@dataclass(frozen=True)
class FineTuningParameterStats:
    """Parameter freezing summary."""

    total_parameter_count: int
    trainable_parameter_count: int
    frozen_parameter_count: int
    frozen_tensor_count: int
    trainable_tensor_count: int


@dataclass(frozen=True)
class FineTuningResult:
    """Summary returned by the fine-tuning loop."""

    epochs: tuple[EpochMetrics, ...]
    best_validation_loss: float | None
    latest_checkpoint_path: Path | None
    best_checkpoint_path: Path | None
    global_step: int


class FineTuningDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for supervised fine-tuning examples."""

    def __init__(self, examples: Sequence[FineTuningExample]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "target_ids": example.target_ids,
            "attention_mask": example.attention_mask,
        }


def load_fine_tuning_records(
    path: Path,
    encoding: str = "utf-8",
) -> list[dict[str, str]]:
    """Load supported fine-tuning JSONL records."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Fine-tuning dataset not found: {input_path}")
    if not input_path.is_file():
        raise FineTuningError(f"Fine-tuning dataset path is not a file: {input_path}")

    records: list[dict[str, str]] = []
    with input_path.open("r", encoding=encoding) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                message = f"Malformed JSONL record at line {line_number}: {exc}"
                raise FineTuningError(message) from exc
            if not isinstance(record, dict):
                raise FineTuningError(
                    f"Malformed fine-tuning record at line {line_number}: expected object."
                )
            records.append(_parse_record(record, line_number))
    return records


def prepare_fine_tuning_dataset(
    dataset_path: Path,
    tokenizer: TextTokenizer,
    vocabulary: Vocabulary,
    context_length: int,
    train_validation_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset, FineTuningStats]:
    """Create train/validation datasets from a fine-tuning JSONL file."""

    _validate_prepare_inputs(tokenizer, vocabulary, context_length, train_validation_ratio)
    records = load_fine_tuning_records(dataset_path)
    examples: list[FineTuningExample] = []
    skipped_records = 0
    truncated_records = 0
    for record in records:
        text = _record_text(record)
        if not text.strip():
            skipped_records += 1
            continue
        tokens = tokenizer.tokenize(text)
        if tokens and tokens[-1] != vocabulary.config.eos_token:
            tokens.append(vocabulary.config.eos_token)
        if not tokens or tokens == [vocabulary.config.eos_token]:
            skipped_records += 1
            continue
        token_ids = vocabulary.encode(tokens)
        if len(token_ids) > context_length + 1:
            token_ids = token_ids[: context_length + 1]
            if token_ids[-1] != vocabulary.eos_id:
                token_ids[-1] = vocabulary.eos_id
            truncated_records += 1
        if len(token_ids) < 2:
            skipped_records += 1
            continue
        examples.append(_example_from_token_ids(token_ids, context_length, vocabulary.pad_id))

    if not examples:
        raise FineTuningError("Fine-tuning dataset produced no usable examples.")

    train_examples, validation_examples = _split_examples(examples, train_validation_ratio, seed)
    stats = FineTuningStats(
        source_records=len(records),
        usable_records=len(examples),
        skipped_records=skipped_records,
        truncated_records=truncated_records,
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
    )
    return FineTuningDataset(train_examples), FineTuningDataset(validation_examples), stats


def configure_trainable_parameters(
    model: GPTModel,
    freeze_embeddings: bool,
    freeze_first_n_layers: int,
) -> FineTuningParameterStats:
    """Freeze requested model regions and return parameter counts."""

    if not isinstance(model, GPTModel):
        raise FineTuningError("model must be a GPTModel.")
    if not isinstance(freeze_embeddings, bool):
        raise FineTuningError("freeze_embeddings must be true or false.")
    if (
        not isinstance(freeze_first_n_layers, int)
        or isinstance(freeze_first_n_layers, bool)
        or freeze_first_n_layers < 0
        or freeze_first_n_layers > model.num_layers
    ):
        raise FineTuningError("freeze_first_n_layers must be between 0 and model.num_layers.")

    for parameter in model.parameters():
        parameter.requires_grad = True
    if freeze_embeddings:
        for parameter in model.token_embedding.parameters():
            parameter.requires_grad = False
    for block in model.blocks[:freeze_first_n_layers]:
        for parameter in block.parameters():
            parameter.requires_grad = False

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = total - trainable
    trainable_tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    frozen_tensors = sum(1 for parameter in model.parameters() if not parameter.requires_grad)
    return FineTuningParameterStats(
        total_parameter_count=total,
        trainable_parameter_count=trainable,
        frozen_parameter_count=frozen,
        frozen_tensor_count=frozen_tensors,
        trainable_tensor_count=trainable_tensors,
    )


def load_base_model_for_fine_tuning(
    base_checkpoint_path: Path,
    app_config: AppConfig,
    device: torch.device,
) -> GPTModel:
    """Build a GPT model and load a compatible base checkpoint without optimizer state."""

    vocabulary = Vocabulary.load(app_config.data.vocabulary_file, encoding=app_config.data.encoding)
    model, metadata = create_gpt_model(app_config.data.vocabulary_file, app_config)
    loaded = load_checkpoint(
        base_checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
        restore_rng=False,
    )
    if metadata.vocab_size != len(vocabulary):
        raise FineTuningError("Model vocabulary size does not match the loaded vocabulary.")
    if model.vocab_size != len(vocabulary):
        raise FineTuningError("Checkpoint vocabulary size does not match the loaded vocabulary.")
    if model.context_length != app_config.model.context_length:
        raise FineTuningError("Checkpoint context length does not match configuration.")
    model.to(device)
    model.train()
    del loaded
    return model


def create_fine_tuning_optimizer(
    model: GPTModel,
    fine_tuning_config: FineTuningConfig,
) -> torch.optim.Optimizer:
    """Create a fresh AdamW optimizer for fine-tuning trainable parameters."""

    optimizer_config = OptimizerConfig(
        type="adamw",
        learning_rate=fine_tuning_config.learning_rate,
        weight_decay=fine_tuning_config.weight_decay,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        separate_weight_decay=True,
    )
    return create_optimizer(model, optimizer_config)


def run_fine_tuning(
    *,
    model: GPTModel,
    train_dataset: Dataset,
    validation_dataset: Dataset | None,
    vocabulary_path: Path,
    app_config: AppConfig,
    fine_tuning_config: FineTuningConfig,
    output_directory: Path,
    base_checkpoint_path: Path,
    dataset_path: Path,
    device: torch.device,
    max_batches: int | None = None,
    resume_checkpoint_path: Path | None = None,
    parameter_stats: FineTuningParameterStats | None = None,
    mixed_precision: str = "none",
    torch_compile: bool = False,
    compile_mode: str = "default",
) -> FineTuningResult:
    """Fine-tune a GPT model using GPTTrainer and save fine-tuning checkpoints."""

    if len(train_dataset) == 0:
        raise FineTuningError("Fine-tuning train dataset is empty.")
    output_directory.mkdir(parents=True, exist_ok=True)
    loss_fn = create_loss_function(vocabulary_path, app_config.loss)
    optimizer = create_fine_tuning_optimizer(model, fine_tuning_config)
    model = compile_model(model, enabled=torch_compile, mode=compile_mode)
    trainer = GPTTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        gradient_accumulation_steps=fine_tuning_config.gradient_accumulation_steps,
        max_grad_norm=fine_tuning_config.max_grad_norm,
        mixed_precision=mixed_precision,
    )
    start_epoch = 1
    best_validation_loss: float | None = None
    if resume_checkpoint_path is not None:
        loaded = load_checkpoint(
            resume_checkpoint_path,
            trainer.model,
            optimizer=trainer.optimizer,
            scaler=trainer.scaler,
            map_location=device,
            restore_rng=True,
        )
        trainer.total_optimizer_steps = loaded.global_step
        start_epoch = loaded.epoch + 1
        best_validation_loss = loaded.best_metric

    train_loader = _loader(train_dataset, fine_tuning_config.batch_size, max_batches=max_batches)
    validation_loader = (
        None
        if validation_dataset is None or len(validation_dataset) == 0
        else _loader(validation_dataset, fine_tuning_config.batch_size, max_batches=max_batches)
    )
    parameter_stats = parameter_stats or _parameter_stats(model)
    results: list[EpochMetrics] = []
    latest_checkpoint_path = None
    best_checkpoint_path = None
    for epoch in range(start_epoch, start_epoch + fine_tuning_config.epochs):
        training_metrics = trainer.train_epoch(
            train_loader,
            epoch=epoch,
            log_every_steps=app_config.training.log_every_steps,
        )
        validation_loss = None
        validation_tokens = 0
        if validation_loader is not None:
            validation_metrics = trainer.evaluate(validation_loader)
            validation_loss = validation_metrics.loss
            validation_tokens = validation_metrics.tokens
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            training_loss=training_metrics.training_loss,
            validation_loss=validation_loss,
            training_tokens=training_metrics.training_tokens,
            validation_tokens=validation_tokens,
            optimizer_steps=training_metrics.optimizer_steps,
            skipped_batches=training_metrics.skipped_batches,
        )
        results.append(epoch_metrics)
        latest_checkpoint_path = output_directory / f"genpy_ft_epoch_{epoch:04d}.pt"
        metric_for_best = (
            validation_loss if validation_loss is not None else training_metrics.training_loss
        )
        save_checkpoint(
            latest_checkpoint_path,
            trainer.model,
            trainer.optimizer,
            epoch=epoch,
            global_step=trainer.total_optimizer_steps,
            training_loss=training_metrics.training_loss,
            validation_loss=validation_loss,
            best_metric=best_validation_loss,
            model_config=asdict(app_config.model),
            vocabulary_metadata={
                "vocabulary_size": trainer.model.vocab_size,
                "vocabulary_path": str(vocabulary_path),
            },
            extra_state=_extra_state(base_checkpoint_path, dataset_path, parameter_stats),
            scaler=trainer.scaler,
        )
        if best_validation_loss is None or metric_for_best < best_validation_loss:
            best_validation_loss = metric_for_best
            best_checkpoint_path = output_directory / "genpy_ft_best.pt"
            save_checkpoint(
                best_checkpoint_path,
                trainer.model,
                trainer.optimizer,
                epoch=epoch,
                global_step=trainer.total_optimizer_steps,
                training_loss=training_metrics.training_loss,
                validation_loss=validation_loss,
                best_metric=best_validation_loss,
                model_config=asdict(app_config.model),
                vocabulary_metadata={
                    "vocabulary_size": trainer.model.vocab_size,
                    "vocabulary_path": str(vocabulary_path),
                },
                extra_state=_extra_state(base_checkpoint_path, dataset_path, parameter_stats),
                scaler=trainer.scaler,
            )

    return FineTuningResult(
        epochs=tuple(results),
        best_validation_loss=best_validation_loss,
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        global_step=trainer.total_optimizer_steps,
    )


def _parse_record(record: dict[object, object], line_number: int) -> dict[str, str]:
    if "text" in record:
        text = record["text"]
        if not isinstance(text, str):
            raise FineTuningError(
                f"Malformed fine-tuning record at line {line_number}: text must be a string."
            )
        return {"text": text}
    if "instruction" in record or "response" in record:
        instruction = record.get("instruction")
        response = record.get("response")
        if not isinstance(instruction, str) or not isinstance(response, str):
            raise FineTuningError(
                f"Malformed fine-tuning record at line {line_number}: "
                "instruction and response must be strings."
            )
        return {"instruction": instruction, "response": response}
    raise FineTuningError(
        f"Malformed fine-tuning record at line {line_number}: "
        "expected text or instruction/response fields."
    )


def _record_text(record: dict[str, str]) -> str:
    if "text" in record:
        return record["text"]
    return "\n".join(
        [
            INSTRUCTION_MARKER,
            record["instruction"],
            RESPONSE_MARKER,
            record["response"],
        ]
    )


def _example_from_token_ids(
    token_ids: Sequence[int],
    context_length: int,
    pad_id: int,
) -> FineTuningExample:
    input_ids = list(token_ids[:-1])
    target_ids = list(token_ids[1:])
    attention_mask = [1] * len(input_ids)
    pad_count = context_length - len(input_ids)
    if pad_count > 0:
        input_ids.extend([pad_id] * pad_count)
        target_ids.extend([pad_id] * pad_count)
        attention_mask.extend([0] * pad_count)
    return FineTuningExample(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
    )


def _split_examples(
    examples: Sequence[FineTuningExample],
    train_validation_ratio: float,
    seed: int,
) -> tuple[list[FineTuningExample], list[FineTuningExample]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    if len(indices) == 1 or train_validation_ratio >= 1:
        train_count = len(indices)
    else:
        train_count = int(len(indices) * train_validation_ratio)
        train_count = min(max(train_count, 1), len(indices) - 1)
    train_indices = indices[:train_count]
    validation_indices = indices[train_count:]
    return (
        [examples[index] for index in train_indices],
        [examples[index] for index in validation_indices],
    )


def _validate_prepare_inputs(
    tokenizer: TextTokenizer,
    vocabulary: Vocabulary,
    context_length: int,
    train_validation_ratio: float,
) -> None:
    if not isinstance(tokenizer, TextTokenizer):
        raise FineTuningError("tokenizer must be a TextTokenizer.")
    if not isinstance(vocabulary, Vocabulary):
        raise FineTuningError("vocabulary must be a Vocabulary.")
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length <= 0
    ):
        raise FineTuningError("context_length must be greater than zero.")
    if (
        not isinstance(train_validation_ratio, int | float)
        or isinstance(train_validation_ratio, bool)
        or not 0 < train_validation_ratio <= 1
    ):
        raise FineTuningError("train_validation_ratio must be greater than 0 and at most 1.")


def _loader(
    dataset: Dataset,
    batch_size: int,
    *,
    max_batches: int | None,
) -> DataLoader | list[dict[str, torch.Tensor]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if max_batches is None:
        return loader
    return [batch for index, batch in enumerate(loader) if index < max_batches]


def _parameter_stats(model: GPTModel) -> FineTuningParameterStats:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return FineTuningParameterStats(
        total_parameter_count=total,
        trainable_parameter_count=trainable,
        frozen_parameter_count=total - trainable,
        frozen_tensor_count=sum(
            1 for parameter in model.parameters() if not parameter.requires_grad
        ),
        trainable_tensor_count=sum(
            1 for parameter in model.parameters() if parameter.requires_grad
        ),
    )


def _extra_state(
    base_checkpoint_path: Path,
    dataset_path: Path,
    parameter_stats: FineTuningParameterStats,
) -> dict[str, object]:
    return {
        "fine_tuning": {
            "base_checkpoint_path": str(base_checkpoint_path),
            "dataset_path": str(dataset_path),
            "trainable_parameter_count": parameter_stats.trainable_parameter_count,
            "frozen_parameter_count": parameter_stats.frozen_parameter_count,
        }
    }


@dataclass(frozen=True)
class Phase7DataConfig:
    """Phase 7 instruction dataset settings."""

    train_path: Path
    validation_path: Path | None
    validation_fraction: float
    tokenizer: Path
    context_length: int | None
    mask_prompt_tokens: bool
    batch_size: int
    dataloader_workers: int
    prefetch_factor: int | None
    pin_memory: bool
    shuffle: bool
    seed: int


@dataclass(frozen=True)
class Phase7TrainingConfig:
    """Phase 7 training-loop settings."""

    device: str
    mixed_precision: str
    epochs: int
    max_steps: int | None
    gradient_accumulation_steps: int
    max_grad_norm: float | None
    log_every_steps: int
    save_every_steps: int
    eval_every_steps: int
    evaluation_steps: int
    resume: bool
    resume_from: Path | None
    seed: int


@dataclass(frozen=True)
class Phase7CheckpointConfig:
    """Phase 7 checkpoint settings."""

    base_checkpoint: Path
    output_dir: Path
    step_prefix: str
    best_filename: str
    last_filename: str
    keep_last: int


@dataclass(frozen=True)
class Phase7OutputConfig:
    """Phase 7 metrics, sample, and logging paths."""

    metrics_dir: Path
    samples_dir: Path
    log_file: Path


@dataclass(frozen=True)
class Phase7Config:
    """Complete Phase 7 SFT configuration."""

    project_root: Path
    model: Phase6ModelConfig
    data: Phase7DataConfig
    training: Phase7TrainingConfig
    optimizer: OptimizerConfig
    scheduler_warmup_steps: int
    checkpoint: Phase7CheckpointConfig
    template: ConversationTemplate
    generation: CodeGenerationSettings
    outputs: Phase7OutputConfig
    log_level: str


@dataclass(frozen=True)
class Phase7Result:
    """Summary returned by Phase 7 fine-tuning."""

    global_step: int
    latest_loss: float | None
    best_validation_loss: float | None
    last_checkpoint: Path | None
    best_checkpoint: Path | None
    metrics_path: Path
    latest_sample_path: Path | None


class Phase7Trainer:
    """Supervised instruction fine-tuning trainer for Phase 6 GPT checkpoints."""

    def __init__(self, config: Phase7Config) -> None:
        self.config = config
        random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        self.device = select_device(config.training.device)
        self.mixed_precision = resolve_mixed_precision(
            config.training.mixed_precision,
            self.device,
        )
        self.workers = self._effective_workers()
        self.prefetch_factor = self._effective_prefetch_factor()
        self.pin_memory = self._effective_pin_memory()
        self.tokenizer = CodeTokenizer.from_file(config.data.tokenizer)
        self.model = create_phase6_model(config.model, self.tokenizer)
        load_checkpoint(
            config.checkpoint.base_checkpoint,
            self.model,
            optimizer=None,
            map_location=self.device,
            restore_rng=False,
        )
        self.model.to(self.device)
        self.optimizer, self.optimizer_metadata = create_optimizer_with_metadata(
            self.model,
            config.optimizer,
        )
        max_steps = config.training.max_steps or max(1, config.training.epochs)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            max_steps=max_steps,
            warmup_steps=config.scheduler_warmup_steps,
            minimum_learning_rate_ratio=0.1,
        )
        self.scaler = create_grad_scaler(self.mixed_precision, self.device)
        self.train_dataset, self.validation_dataset = self._datasets()
        self.global_step = 0
        self.epoch = 0
        self.micro_step = 0
        self.best_validation_loss: float | None = None
        self.latest_loss: float | None = None
        self.last_checkpoint: Path | None = None
        self.best_checkpoint: Path | None = None
        self.latest_sample_path: Path | None = None
        self.metrics_path = config.outputs.metrics_dir / "fine_tuning_metrics.csv"
        self.json_metrics_path = config.outputs.metrics_dir / "fine_tuning_metrics.jsonl"
        if config.training.resume:
            self._resume()

    def train(self) -> Phase7Result:
        """Run supervised fine-tuning."""

        self._prepare_outputs()
        train_loader = self._loader(self.train_dataset, shuffle=self.config.data.shuffle)
        for epoch in range(self.epoch + 1, self.config.training.epochs + 1):
            self.epoch = epoch
            for batch in train_loader:
                metrics = self._train_micro_batch(batch)
                if metrics is None:
                    continue
                self.global_step += 1
                self.latest_loss = float(metrics["loss"])
                self._log_metrics(metrics)
                if self.global_step % self.config.training.log_every_steps == 0:
                    print(f"step={self.global_step} loss={self.latest_loss:.6f}")
                if self._should_evaluate():
                    eval_metrics = self.evaluate()
                    self._log_metrics(
                        {
                            "type": "evaluation",
                            "step": self.global_step,
                            "epoch": self.epoch,
                            "loss": eval_metrics.loss,
                            "perplexity": eval_metrics.perplexity,
                            "tokens": eval_metrics.tokens,
                            "batches": eval_metrics.batches,
                        }
                    )
                    self._write_samples()
                    if self._is_best(eval_metrics.loss):
                        self.best_validation_loss = eval_metrics.loss
                        self.best_checkpoint = self._save_checkpoint(best=True)
                if self.global_step % self.config.training.save_every_steps == 0:
                    self.last_checkpoint = self._save_checkpoint()
                if (
                    self.config.training.max_steps is not None
                    and self.global_step >= self.config.training.max_steps
                ):
                    self.last_checkpoint = self._save_checkpoint()
                    return self._result()
            train_loader = self._loader(self.train_dataset, shuffle=self.config.data.shuffle)
        self.last_checkpoint = self._save_checkpoint()
        return self._result()

    @torch.no_grad()
    def evaluate(self) -> EvaluationMetrics:
        """Evaluate validation loss and perplexity."""

        if self.validation_dataset is None or len(self.validation_dataset) == 0:
            return evaluation_metrics(0.0, 0, 0)
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = 0
        for batch in self._loader(self.validation_dataset, shuffle=False):
            if batches >= self.config.training.evaluation_steps:
                break
            input_ids, targets, attention = self._batch_to_device(batch)
            with autocast_context(self.mixed_precision, self.device):
                logits = self.model(input_ids, padding_mask=attention)
                loss = _sft_loss(logits, targets)
            tokens = int((targets != -100).sum().detach().cpu().item())
            total_loss += float(loss.detach().cpu().item()) * tokens
            total_tokens += tokens
            batches += 1
        if was_training:
            self.model.train()
        average_loss = total_loss / total_tokens if total_tokens else 0.0
        return evaluation_metrics(average_loss, total_tokens, batches)

    def _train_micro_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any] | None:
        self.model.train()
        self.micro_step += 1
        input_ids, targets, attention = self._batch_to_device(batch)
        reset_peak_memory(self.device)
        with StepTimer(self.device) as timer:
            with autocast_context(self.mixed_precision, self.device):
                logits = self.model(input_ids, padding_mask=attention)
                loss = _sft_loss(logits, targets)
                scaled_loss = loss / self.config.training.gradient_accumulation_steps
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if self.micro_step % self.config.training.gradient_accumulation_steps != 0:
                return None
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = self._clip_gradients()
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        tokens = int((targets != -100).sum().detach().cpu().item())
        elapsed = max(timer.elapsed_seconds, 1e-12)
        return {
            "type": "train",
            "step": self.global_step + 1,
            "epoch": self.epoch,
            "loss": float(loss.detach().cpu().item()),
            "perplexity": math.exp(min(20.0, float(loss.detach().cpu().item()))),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "tokens": tokens,
            "tokens_per_second": tokens / elapsed,
            "gradient_norm": grad_norm,
            "gpu_memory_mb": peak_memory_mb(self.device),
            "elapsed_seconds": elapsed,
        }

    def _datasets(self) -> tuple[InstructionDataset, InstructionDataset | None]:
        train = InstructionDataset.from_jsonl(
            self.config.data.train_path,
            tokenizer=self.tokenizer,
            template=self.config.template,
            context_length=self._dataset_context_length(),
            mask_prompt_tokens=self.config.data.mask_prompt_tokens,
        )
        if self.config.data.validation_path is not None:
            validation = InstructionDataset.from_jsonl(
                self.config.data.validation_path,
                tokenizer=self.tokenizer,
                template=self.config.template,
                context_length=self._dataset_context_length(),
                mask_prompt_tokens=self.config.data.mask_prompt_tokens,
            )
            return train, validation
        return train, None

    def _dataset_context_length(self) -> int:
        if self.config.data.context_length is None:
            return self.config.model.context_length
        return min(self.config.data.context_length, self.config.model.context_length)

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        kwargs: dict[str, Any] = {}
        if self.workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            **kwargs,
        )

    def _batch_to_device(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch["input_ids"].to(self.device, non_blocking=True),
            batch["target_ids"].to(self.device, non_blocking=True),
            batch["attention_mask"].to(self.device, non_blocking=True),
        )

    def _clip_gradients(self) -> float:
        if self.config.training.max_grad_norm is None:
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if not parameters:
                return 0.0
            norm = torch.linalg.vector_norm(
                torch.stack([p.grad.detach().norm() for p in parameters])
            )
            return float(norm.detach().cpu().item())
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.max_grad_norm,
        )
        return float(norm.detach().cpu().item())

    def _save_checkpoint(self, *, best: bool = False) -> Path:
        self.config.checkpoint.output_dir.mkdir(parents=True, exist_ok=True)
        if best:
            path = self.config.checkpoint.output_dir / self.config.checkpoint.best_filename
        else:
            path = self.config.checkpoint.output_dir / (
                f"{self.config.checkpoint.step_prefix}_{self.global_step:05d}.pt"
            )
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            epoch=self.epoch,
            global_step=self.global_step,
            training_loss=self.latest_loss,
            validation_loss=self.best_validation_loss,
            best_metric=self.best_validation_loss,
            scheduler=self.scheduler,
            scaler=self.scaler,
            model_config=asdict(self.config.model),
            vocabulary_metadata={
                "tokenizer": str(self.config.data.tokenizer),
                "tokenizer_sha256": tokenizer_file_hash(self.config.data.tokenizer),
            },
            extra_state={
                "phase": 7,
                "base_checkpoint": str(self.config.checkpoint.base_checkpoint),
                "dataset": str(self.config.data.train_path),
            },
        )
        if not best:
            last = self.config.checkpoint.output_dir / self.config.checkpoint.last_filename
            shutil.copy2(path, last)
            self._rotate_checkpoints()
            return last
        return path

    def _resume(self) -> None:
        path = self.config.training.resume_from
        if path is None:
            candidate = self.config.checkpoint.output_dir / self.config.checkpoint.last_filename
            path = candidate if candidate.is_file() else None
        if path is None:
            return
        loaded = load_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.epoch = loaded.epoch
        self.global_step = loaded.global_step
        self.best_validation_loss = loaded.best_metric

    def _write_samples(self) -> Path:
        self.config.outputs.samples_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.outputs.samples_dir / f"phase7_step_{self.global_step:05d}.json"
        samples = []
        for prompt in self.config.generation.prompts:
            formatted = format_generation_prompt(prompt, template=self.config.template)
            result = generate_code_sample(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=formatted,
                device=self.device,
                context_length=self.config.model.context_length,
                settings=self.config.generation,
            )
            samples.append(asdict(result))
        path.write_text(
            json.dumps({"step": self.global_step, "samples": samples}, indent=2),
            encoding="utf-8",
        )
        self.latest_sample_path = path
        return path

    def _log_metrics(self, metrics: Mapping[str, Any]) -> None:
        self.config.outputs.metrics_dir.mkdir(parents=True, exist_ok=True)
        row = {"timestamp": time.time(), **dict(metrics)}
        exists = self.metrics_path.is_file()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sorted(row))
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in sorted(row)})
        with self.json_metrics_path.open("a", encoding="utf-8") as file:
            json.dump(row, file, sort_keys=True, separators=(",", ":"))
            file.write("\n")

    def _should_evaluate(self) -> bool:
        return (
            self.validation_dataset is not None
            and self.config.training.eval_every_steps > 0
            and self.global_step > 0
            and self.global_step % self.config.training.eval_every_steps == 0
        )

    def _is_best(self, loss: float) -> bool:
        return self.best_validation_loss is None or loss < self.best_validation_loss

    def _prepare_outputs(self) -> None:
        for path in (
            self.config.checkpoint.output_dir,
            self.config.outputs.metrics_dir,
            self.config.outputs.samples_dir,
            self.config.outputs.log_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _rotate_checkpoints(self) -> None:
        keep = self.config.checkpoint.keep_last
        if keep <= 0:
            return
        checkpoints = sorted(
            self.config.checkpoint.output_dir.glob(f"{self.config.checkpoint.step_prefix}_*.pt"),
            key=lambda path: path.name,
            reverse=True,
        )
        for path in checkpoints[keep:]:
            path.unlink(missing_ok=True)

    def _effective_workers(self) -> int:
        if self.device.type == "mps" and self.config.data.dataloader_workers:
            print("Warning: forcing dataloader_workers=0 on Apple MPS.")
            return 0
        return self.config.data.dataloader_workers

    def _effective_prefetch_factor(self) -> int | None:
        if self.workers <= 0:
            return None
        return self.config.data.prefetch_factor

    def _effective_pin_memory(self) -> bool:
        return self.config.data.pin_memory and self.device.type == "cuda"

    def _result(self) -> Phase7Result:
        return Phase7Result(
            global_step=self.global_step,
            latest_loss=self.latest_loss,
            best_validation_loss=self.best_validation_loss,
            last_checkpoint=self.last_checkpoint,
            best_checkpoint=self.best_checkpoint,
            metrics_path=self.metrics_path,
            latest_sample_path=self.latest_sample_path,
        )


def load_phase7_config(path: Path | str = "configs/finetuning.yaml") -> Phase7Config:
    """Load Phase 7 YAML configuration."""

    root = Path(__file__).resolve().parents[2]
    raw = _yaml_mapping(_resolve_phase7(root, path), "phase7")
    section = _as_mapping(raw.get("phase7", {}), "phase7")
    model_raw = _yaml_mapping(
        _resolve_phase7(root, section.get("model_config", "configs/model.yaml")),
        "model",
    )
    optimizer_raw = _yaml_mapping(
        _resolve_phase7(root, section.get("optimizer_config", "configs/optimizer.yaml")),
        "optimizer",
    )
    data = _as_mapping(section.get("data", {}), "phase7.data")
    training = _as_mapping(section.get("training", {}), "phase7.training")
    checkpoint = _as_mapping(section.get("checkpoint", {}), "phase7.checkpoint")
    outputs = _as_mapping(section.get("outputs", {}), "phase7.outputs")
    generation = _as_mapping(section.get("generation", {}), "phase7.generation")
    model_section = _as_mapping(model_raw.get("model", {}), "model")
    optimizer_section = _as_mapping(optimizer_raw.get("optimizer", {}), "optimizer")
    prompts = generation_prompts(generation.get("prompts"))
    return Phase7Config(
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
        data=Phase7DataConfig(
            train_path=_resolve_phase7(
                root,
                data.get("train_path", "data/fine_tuning/train.jsonl"),
            ),
            validation_path=(
                _resolve_phase7(root, data["validation_path"])
                if data.get("validation_path") is not None
                else None
            ),
            validation_fraction=float(data.get("validation_fraction", 0.0)),
            tokenizer=_resolve_phase7(root, data.get("tokenizer", "data/tokenizer/tokenizer.json")),
            context_length=(
                int(data["context_length"]) if data.get("context_length") is not None else None
            ),
            mask_prompt_tokens=bool(data.get("mask_prompt_tokens", True)),
            batch_size=int(data.get("batch_size", 1)),
            dataloader_workers=int(data.get("dataloader_workers", 0)),
            prefetch_factor=(
                int(data["prefetch_factor"]) if data.get("prefetch_factor") is not None else None
            ),
            pin_memory=bool(data.get("pin_memory", False)),
            shuffle=bool(data.get("shuffle", True)),
            seed=int(data.get("seed", 42)),
        ),
        training=Phase7TrainingConfig(
            device=str(training.get("device", "auto")),
            mixed_precision=str(training.get("mixed_precision", "none")),
            epochs=int(training.get("epochs", 1)),
            max_steps=int(training["max_steps"]) if training.get("max_steps") is not None else None,
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
            max_grad_norm=(
                float(training["max_grad_norm"])
                if training.get("max_grad_norm") is not None
                else None
            ),
            log_every_steps=int(training.get("log_every_steps", 1)),
            save_every_steps=int(training.get("save_every_steps", 100)),
            eval_every_steps=int(training.get("eval_every_steps", 100)),
            evaluation_steps=int(training.get("evaluation_steps", 10)),
            resume=bool(training.get("resume", False)),
            resume_from=(
                _resolve_phase7(root, training["resume_from"])
                if training.get("resume_from") is not None
                else None
            ),
            seed=int(training.get("seed", 42)),
        ),
        optimizer=OptimizerConfig(
            type=str(optimizer_section.get("type", "adamw")),
            learning_rate=float(
                section.get("learning_rate", optimizer_section.get("learning_rate", 5e-5))
            ),
            weight_decay=float(optimizer_section.get("weight_decay", 0.1)),
            beta1=float(optimizer_section.get("beta1", 0.9)),
            beta2=float(optimizer_section.get("beta2", 0.95)),
            epsilon=float(optimizer_section.get("epsilon", 1e-8)),
            separate_weight_decay=bool(optimizer_section.get("separate_weight_decay", True)),
        ),
        scheduler_warmup_steps=int(section.get("warmup_steps", 0)),
        checkpoint=Phase7CheckpointConfig(
            base_checkpoint=_resolve_phase7(
                root,
                checkpoint.get("base_checkpoint", "checkpoints/last_checkpoint.pt"),
            ),
            output_dir=_resolve_phase7(
                root,
                checkpoint.get("output_dir", "checkpoints/fine_tuned"),
            ),
            step_prefix=str(checkpoint.get("step_prefix", "step")),
            best_filename=str(checkpoint.get("best_filename", "best_checkpoint.pt")),
            last_filename=str(checkpoint.get("last_filename", "last_checkpoint.pt")),
            keep_last=int(checkpoint.get("keep_last", 3)),
        ),
        template=ConversationTemplate.from_mapping(section.get("template")),
        generation=CodeGenerationSettings(
            prompts=prompts,
            max_new_tokens=int(generation.get("max_new_tokens", 64)),
            temperature=float(generation.get("temperature", 0.8)),
            top_k=int(generation["top_k"]) if generation.get("top_k") is not None else None,
            top_p=float(generation["top_p"]) if generation.get("top_p") is not None else None,
            do_sample=bool(generation.get("do_sample", False)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
            stop_tokens=tuple(generation.get("stop_tokens", ("<eos>",))),
        ),
        outputs=Phase7OutputConfig(
            metrics_dir=_resolve_phase7(root, outputs.get("metrics_dir", "metrics/phase7")),
            samples_dir=_resolve_phase7(root, outputs.get("samples_dir", "generated_samples")),
            log_file=_resolve_phase7(root, outputs.get("log_file", "logs/phase7_finetuning.jsonl")),
        ),
        log_level=str(section.get("log_level", "INFO")).upper(),
    )


def _sft_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


def _yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _as_mapping(payload, label)


def _as_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise FineTuningError(f"{label} must be a mapping.")
    return payload


def _resolve_phase7(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


__all__ = [
    "FineTuningDataset",
    "FineTuningError",
    "FineTuningExample",
    "FineTuningParameterStats",
    "FineTuningResult",
    "FineTuningStats",
    "configure_trainable_parameters",
    "create_fine_tuning_optimizer",
    "load_base_model_for_fine_tuning",
    "load_fine_tuning_records",
    "load_phase7_config",
    "prepare_fine_tuning_dataset",
    "Phase7Config",
    "Phase7DataConfig",
    "Phase7Trainer",
    "Phase7TrainingConfig",
    "Phase7Result",
    "run_fine_tuning",
]
