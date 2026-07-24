"""Phase 9 LoRA configuration and instruction-tuning pipeline."""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from genpy_llm.checkpointing import load_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash
from genpy_llm.device import select_device
from genpy_llm.evaluation import EvaluationMetrics, evaluation_metrics
from genpy_llm.fine_tuning import Phase7Config, load_phase7_config
from genpy_llm.instruction_dataset import InstructionDataset
from genpy_llm.lora import (
    DEFAULT_LORA_TARGETS,
    LoRAStats,
    apply_lora,
    load_lora_adapters,
    save_lora_adapters,
)
from genpy_llm.performance import (
    StepTimer,
    autocast_context,
    create_grad_scaler,
    resolve_mixed_precision,
)
from genpy_llm.pretraining import (
    CosineWarmupScheduler,
    compute_scheduler_total_steps,
    create_phase6_model,
)

LOGGER = logging.getLogger("genpy_llm.lora_training")


class LoRATrainingError(RuntimeError):
    """Raised when Phase 9 configuration or training cannot continue."""


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter hyperparameters."""

    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class LoRATrainingConfig:
    """Phase 9 optimization settings."""

    base_checkpoint: Path
    device: str
    mixed_precision: str
    epochs: int
    max_steps: int | None
    batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float | None
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    log_every_steps: int
    save_every_steps: int
    eval_every_steps: int
    evaluation_steps: int
    seed: int
    resume_from: Path | None


@dataclass(frozen=True)
class LoRACheckpointConfig:
    """Adapter-only checkpoint settings."""

    output_dir: Path
    adapter_filename: str
    best_filename: str
    step_prefix: str
    keep_last: int


@dataclass(frozen=True)
class LoRAEvaluationConfig:
    """Full-fine-tuning versus LoRA comparison settings."""

    full_fine_tuned_checkpoint: Path
    prompt_dataset: Path
    output_dir: Path
    max_new_tokens: int
    validation_batches: int


@dataclass(frozen=True)
class LoRAOutputConfig:
    """Phase 9 metrics and logging paths."""

    metrics_dir: Path
    log_file: Path
    log_level: str


@dataclass(frozen=True)
class Phase9Config:
    """Complete Phase 9 configuration."""

    project_root: Path
    phase7_config_path: Path
    phase7: Phase7Config
    adapter: LoRAConfig
    training: LoRATrainingConfig
    checkpoints: LoRACheckpointConfig
    evaluation: LoRAEvaluationConfig
    outputs: LoRAOutputConfig


@dataclass(frozen=True)
class LoRATrainingResult:
    """Artifacts and metrics returned by LoRA instruction tuning."""

    global_step: int
    latest_loss: float | None
    best_validation_loss: float | None
    last_adapter: Path
    best_adapter: Path | None
    metrics_path: Path
    parameter_stats: LoRAStats


def load_phase9_config(path: Path | str = "configs/lora.yaml") -> Phase9Config:
    """Load and validate the Phase 9 YAML configuration."""

    root = Path(__file__).resolve().parents[2]
    config_path = _resolve(root, path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Phase 9 config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = _mapping(_mapping(raw, "root").get("phase9"), "phase9")
    adapter = _mapping(section.get("adapter"), "phase9.adapter")
    training = _mapping(section.get("training"), "phase9.training")
    checkpoints = _mapping(section.get("checkpoints"), "phase9.checkpoints")
    evaluation = _mapping(section.get("evaluation"), "phase9.evaluation")
    outputs = _mapping(section.get("outputs"), "phase9.outputs")
    phase7_config_path = _resolve(
        root,
        section.get("phase7_config", "configs/finetuning.yaml"),
    )
    target_modules = adapter.get("target_modules", DEFAULT_LORA_TARGETS)
    if isinstance(target_modules, str) or not isinstance(target_modules, (list, tuple)):
        raise LoRATrainingError("phase9.adapter.target_modules must be an array.")
    config = Phase9Config(
        project_root=root,
        phase7_config_path=phase7_config_path,
        phase7=load_phase7_config(phase7_config_path),
        adapter=LoRAConfig(
            rank=int(adapter.get("rank", 8)),
            alpha=float(adapter.get("alpha", 16.0)),
            dropout=float(adapter.get("dropout", 0.05)),
            target_modules=tuple(str(value) for value in target_modules),
        ),
        training=LoRATrainingConfig(
            base_checkpoint=_resolve(
                root,
                training.get("base_checkpoint", "checkpoints/last_checkpoint.pt"),
            ),
            device=str(training.get("device", "auto")),
            mixed_precision=str(training.get("mixed_precision", "none")),
            epochs=int(training.get("epochs", 1)),
            max_steps=(
                int(training["max_steps"]) if training.get("max_steps") is not None else None
            ),
            batch_size=int(training.get("batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
            max_grad_norm=(
                float(training["max_grad_norm"])
                if training.get("max_grad_norm") is not None
                else None
            ),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            warmup_steps=int(training.get("warmup_steps", 0)),
            log_every_steps=int(training.get("log_every_steps", 10)),
            save_every_steps=int(training.get("save_every_steps", 100)),
            eval_every_steps=int(training.get("eval_every_steps", 100)),
            evaluation_steps=int(training.get("evaluation_steps", 10)),
            seed=int(training.get("seed", 42)),
            resume_from=(
                _resolve(root, training["resume_from"])
                if training.get("resume_from") is not None
                else None
            ),
        ),
        checkpoints=LoRACheckpointConfig(
            output_dir=_resolve(
                root,
                checkpoints.get("output_dir", "checkpoints/lora"),
            ),
            adapter_filename=str(checkpoints.get("adapter_filename", "last_adapter.pt")),
            best_filename=str(checkpoints.get("best_filename", "best_adapter.pt")),
            step_prefix=str(checkpoints.get("step_prefix", "step")),
            keep_last=int(checkpoints.get("keep_last", 3)),
        ),
        evaluation=LoRAEvaluationConfig(
            full_fine_tuned_checkpoint=_resolve(
                root,
                evaluation.get(
                    "full_fine_tuned_checkpoint",
                    "checkpoints/fine_tuned/last_checkpoint.pt",
                ),
            ),
            prompt_dataset=_resolve(
                root,
                evaluation.get("prompt_dataset", "data/evaluation/prompts.json"),
            ),
            output_dir=_resolve(
                root,
                evaluation.get("output_dir", "evaluation/lora_comparison"),
            ),
            max_new_tokens=int(evaluation.get("max_new_tokens", 64)),
            validation_batches=int(evaluation.get("validation_batches", 10)),
        ),
        outputs=LoRAOutputConfig(
            metrics_dir=_resolve(root, outputs.get("metrics_dir", "metrics/phase9")),
            log_file=_resolve(root, outputs.get("log_file", "logs/phase9_lora.jsonl")),
            log_level=str(outputs.get("log_level", "INFO")).upper(),
        ),
    )
    _validate_phase9_config(config)
    return config


class Phase9LoRATrainer:
    """Train only weight-parametrized LoRA adapters on Phase 7 instruction data."""

    def __init__(self, config: Phase9Config) -> None:
        self.config = config
        random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        self.device = select_device(config.training.device)
        self.mixed_precision = resolve_mixed_precision(
            config.training.mixed_precision,
            self.device,
            logger=LOGGER,
        )
        self.tokenizer = CodeTokenizer.from_file(config.phase7.data.tokenizer)
        self.model = create_phase6_model(config.phase7.model, self.tokenizer)
        load_checkpoint(
            config.training.base_checkpoint,
            self.model,
            optimizer=None,
            map_location="cpu",
            restore_rng=False,
        )
        self.model.to(self.device)
        self.parameter_stats = apply_lora(
            self.model,
            rank=config.adapter.rank,
            alpha=config.adapter.alpha,
            dropout=config.adapter.dropout,
            target_modules=config.adapter.target_modules,
        )
        self.global_step = 0
        self.latest_loss: float | None = None
        self.best_validation_loss: float | None = None
        if config.training.resume_from is not None:
            loaded = load_lora_adapters(
                self.model,
                config.training.resume_from,
                map_location=self.device,
            )
            self.global_step = int(loaded.metadata.get("global_step", 0))
            best = loaded.metadata.get("best_validation_loss")
            self.best_validation_loss = float(best) if isinstance(best, (int, float)) else None
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise LoRATrainingError("No trainable LoRA parameters were created.")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.train_dataset, self.validation_dataset = self._datasets()
        schedule_steps = compute_scheduler_total_steps(
            dataset_size=len(self.train_dataset),
            batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            epochs=config.training.epochs,
            max_steps=config.training.max_steps,
        )
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            max_steps=schedule_steps,
            warmup_steps=min(config.training.warmup_steps, schedule_steps),
            minimum_learning_rate_ratio=0.1,
            last_step=self.global_step,
        )
        self.scaler = create_grad_scaler(self.mixed_precision, self.device)
        self.metrics_path = config.outputs.metrics_dir / "lora_training_metrics.csv"
        self.json_metrics_path = config.outputs.metrics_dir / "lora_training_metrics.jsonl"

    def train(self) -> LoRATrainingResult:
        """Run LoRA-only supervised instruction tuning."""

        self._prepare_outputs()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        stop = False
        for epoch in range(1, self.config.training.epochs + 1):
            for batch in self._loader(self.train_dataset, shuffle=True):
                micro_step += 1
                self.model.train()
                input_ids, targets, attention = self._batch_to_device(batch)
                with StepTimer(self.device) as timer:
                    with autocast_context(self.mixed_precision, self.device):
                        logits = self.model(input_ids, padding_mask=attention)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1),
                            ignore_index=-100,
                        )
                        scaled_loss = loss / self.config.training.gradient_accumulation_steps
                    if self.scaler is None:
                        scaled_loss.backward()
                    else:
                        self.scaler.scale(scaled_loss).backward()
                if micro_step % self.config.training.gradient_accumulation_steps:
                    continue
                gradient_norm = self._optimizer_step()
                self.global_step += 1
                self.latest_loss = float(loss.detach().cpu().item())
                token_count = int((targets != -100).sum().detach().cpu().item())
                self._write_metrics(
                    {
                        "type": "train",
                        "epoch": epoch,
                        "step": self.global_step,
                        "loss": self.latest_loss,
                        "tokens": token_count,
                        "tokens_per_second": token_count / max(timer.elapsed_seconds, 1e-12),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "gradient_norm": gradient_norm,
                    }
                )
                if self.global_step % self.config.training.log_every_steps == 0:
                    LOGGER.info("step=%d lora_loss=%.6f", self.global_step, self.latest_loss)
                if self._should_evaluate():
                    metrics = self.evaluate()
                    self._write_metrics(
                        {
                            "type": "evaluation",
                            "epoch": epoch,
                            "step": self.global_step,
                            "loss": metrics.loss,
                            "perplexity": metrics.perplexity,
                            "tokens": metrics.tokens,
                            "batches": metrics.batches,
                        }
                    )
                    if (
                        self.best_validation_loss is None
                        or metrics.loss < self.best_validation_loss
                    ):
                        self.best_validation_loss = metrics.loss
                        self._save_adapter(self.config.checkpoints.best_filename)
                if self.global_step % self.config.training.save_every_steps == 0:
                    self._save_step_adapter()
                if (
                    self.config.training.max_steps is not None
                    and self.global_step >= self.config.training.max_steps
                ):
                    stop = True
                    break
            if stop:
                break
        last = self._save_adapter(self.config.checkpoints.adapter_filename)
        best = self.config.checkpoints.output_dir / self.config.checkpoints.best_filename
        return LoRATrainingResult(
            global_step=self.global_step,
            latest_loss=self.latest_loss,
            best_validation_loss=self.best_validation_loss,
            last_adapter=last,
            best_adapter=best.resolve() if best.is_file() else None,
            metrics_path=self.metrics_path,
            parameter_stats=self.parameter_stats,
        )

    @torch.no_grad()
    def evaluate(self) -> EvaluationMetrics:
        """Calculate validation loss with adapter dropout disabled."""

        if self.validation_dataset is None:
            return evaluation_metrics(0.0, 0, 0)
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
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
            token_count = int((targets != -100).sum().detach().cpu().item())
            total_loss += float(loss.detach().cpu().item()) * token_count
            total_tokens += token_count
            batches += 1
        average = total_loss / total_tokens if total_tokens else 0.0
        return evaluation_metrics(average, total_tokens, batches)

    def _datasets(self) -> tuple[InstructionDataset, InstructionDataset | None]:
        phase7 = self.config.phase7
        context_length = min(
            phase7.data.context_length or phase7.model.context_length,
            phase7.model.context_length,
        )
        train = InstructionDataset.from_jsonl(
            phase7.data.train_path,
            tokenizer=self.tokenizer,
            template=phase7.template,
            context_length=context_length,
            mask_prompt_tokens=phase7.data.mask_prompt_tokens,
        )
        validation = (
            InstructionDataset.from_jsonl(
                phase7.data.validation_path,
                tokenizer=self.tokenizer,
                template=phase7.template,
                context_length=context_length,
                mask_prompt_tokens=phase7.data.mask_prompt_tokens,
            )
            if phase7.data.validation_path is not None
            else None
        )
        return train, validation

    def _loader(self, dataset: InstructionDataset, *, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(self.config.training.seed + self.global_step)
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=generator,
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

    def _optimizer_step(self) -> float:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if self.config.training.max_grad_norm is None:
            gradients = [
                parameter.grad.norm()
                for parameter in parameters
                if parameter.grad is not None
            ]
            gradient_norm = (
                float(torch.linalg.vector_norm(torch.stack(gradients)).detach().cpu().item())
                if gradients
                else 0.0
            )
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.config.training.max_grad_norm,
            )
            gradient_norm = float(norm.detach().cpu().item())
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return gradient_norm

    def _should_evaluate(self) -> bool:
        return (
            self.validation_dataset is not None
            and self.config.training.eval_every_steps > 0
            and self.global_step % self.config.training.eval_every_steps == 0
        )

    def _save_step_adapter(self) -> Path:
        filename = f"{self.config.checkpoints.step_prefix}_{self.global_step:05d}.pt"
        path = self._save_adapter(filename)
        checkpoints = sorted(
            self.config.checkpoints.output_dir.glob(
                f"{self.config.checkpoints.step_prefix}_*.pt"
            ),
            key=lambda item: item.name,
            reverse=True,
        )
        for old in checkpoints[self.config.checkpoints.keep_last :]:
            old.unlink(missing_ok=True)
        return path

    def _save_adapter(self, filename: str) -> Path:
        return save_lora_adapters(
            self.model,
            self.config.checkpoints.output_dir / filename,
            metadata={
                "phase": 9,
                "global_step": self.global_step,
                "latest_loss": self.latest_loss,
                "best_validation_loss": self.best_validation_loss,
                "base_checkpoint": str(self.config.training.base_checkpoint),
                "phase7_config": str(self.config.phase7_config_path),
                "tokenizer": str(self.config.phase7.data.tokenizer),
                "tokenizer_sha256": tokenizer_file_hash(self.config.phase7.data.tokenizer),
            },
        )

    def _write_metrics(self, row: Mapping[str, Any]) -> None:
        self.config.outputs.metrics_dir.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": time.time(), **dict(row)}
        fields = (
            "timestamp",
            "type",
            "epoch",
            "step",
            "loss",
            "perplexity",
            "tokens",
            "batches",
            "tokens_per_second",
            "learning_rate",
            "gradient_norm",
        )
        exists = self.metrics_path.is_file()
        with self.metrics_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: payload.get(key) for key in fields})
        with self.json_metrics_path.open("a", encoding="utf-8") as file:
            json.dump(payload, file, sort_keys=True, separators=(",", ":"))
            file.write("\n")

    def _prepare_outputs(self) -> None:
        for path in (
            self.config.checkpoints.output_dir,
            self.config.outputs.metrics_dir,
            self.config.outputs.log_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


def override_phase9_config(
    config: Phase9Config,
    *,
    device: str | None = None,
    max_steps: int | None = None,
    base_checkpoint: Path | None = None,
    resume_from: Path | None = None,
) -> Phase9Config:
    """Apply supported CLI overrides without mutating loaded configuration."""

    training = config.training
    if device is not None:
        training = replace(training, device=device)
    if max_steps is not None:
        training = replace(training, max_steps=max_steps)
    if base_checkpoint is not None:
        training = replace(training, base_checkpoint=_resolve(config.project_root, base_checkpoint))
    if resume_from is not None:
        training = replace(training, resume_from=_resolve(config.project_root, resume_from))
    return replace(config, training=training)


def _validate_phase9_config(config: Phase9Config) -> None:
    training = config.training
    if config.adapter.rank <= 0 or config.adapter.alpha <= 0:
        raise LoRATrainingError("LoRA rank and alpha must be greater than zero.")
    if not 0 <= config.adapter.dropout < 1:
        raise LoRATrainingError("LoRA dropout must be at least zero and less than one.")
    positive_values = {
        "epochs": training.epochs,
        "batch_size": training.batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "log_every_steps": training.log_every_steps,
        "save_every_steps": training.save_every_steps,
        "evaluation_steps": training.evaluation_steps,
    }
    if any(value <= 0 for value in positive_values.values()):
        raise LoRATrainingError("Phase 9 positive training settings must be greater than zero.")
    if training.max_steps is not None and training.max_steps <= 0:
        raise LoRATrainingError("max_steps must be greater than zero when configured.")
    if training.weight_decay < 0 or training.warmup_steps < 0 or training.eval_every_steps < 0:
        raise LoRATrainingError("Phase 9 non-negative settings must not be negative.")
    if config.checkpoints.keep_last < 0:
        raise LoRATrainingError("keep_last must not be negative.")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LoRATrainingError(f"{label} must be a mapping.")
    return value


def _resolve(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


__all__ = [
    "LoRAConfig",
    "LoRACheckpointConfig",
    "LoRAEvaluationConfig",
    "LoRAOutputConfig",
    "LoRATrainingConfig",
    "LoRATrainingError",
    "LoRATrainingResult",
    "Phase9Config",
    "Phase9LoRATrainer",
    "load_phase9_config",
    "override_phase9_config",
]
