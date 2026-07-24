"""Continued pretraining (CPT) of the GenPy GPT model on the Final Corpus.

Resumes an existing checkpoint and continues training on the packed shards in
``python_corpus/final_corpus/packed`` by reusing the existing ``Phase6Trainer``
loop, ``PackedSequenceDataset`` loader, tokenizer, and checkpoint format.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Subset

from genpy_llm.benchmark_monitor import BenchmarkSettings, benchmark_phase63_checkpoints
from genpy_llm.checkpoint_manager import (
    resolve_latest_phase6_checkpoint,
    validate_checkpoint_tokenizer,
)
from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.code_tokenizer import tokenizer_file_hash
from genpy_llm.logging_utils import setup_structured_logging
from genpy_llm.performance import normalize_mixed_precision, peak_memory_mb
from genpy_llm.pretraining import (
    CosineWarmupScheduler,
    Phase6Config,
    Phase6Trainer,
    load_phase6_config,
)
from genpy_llm.pretraining_dataset import DeterministicSequenceSampler
from genpy_llm.training_monitor import (
    EarlyStoppingConfig,
    EarlyStoppingState,
    TrainingMonitor,
)

UTC = timezone.utc

LOGGER = logging.getLogger("genpy_llm.cpt")
CPT_PHASE = "cpt_final_corpus"
CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint_step_(?P<step>\d{5,})$")


class CPTError(RuntimeError):
    """Raised when continued pretraining cannot proceed safely."""


@dataclass(frozen=True)
class CPTPaths:
    """Config and artifact paths for continued pretraining."""

    training_config: Path
    model_config: Path
    optimizer_config: Path
    generation_config: Path
    corpus_directory: Path
    tokenizer: Path
    checkpoint_search_dir: Path
    checkpoint_output_dir: Path
    report_dir: Path
    log_file: Path


@dataclass(frozen=True)
class CPTTrainingSettings:
    """Continued-pretraining training controls."""

    learning_rate: float | None
    batch_size: int | None
    gradient_accumulation_steps: int
    epochs: int
    max_steps: int
    checkpoint_interval_steps: int
    validation_interval_steps: int
    log_interval_steps: int
    warmup_steps: int
    weight_decay: float | None
    sequence_length: int
    device: str
    precision: str
    max_grad_norm: float | None
    keep_last_checkpoints: int
    validation_fraction: float | None
    validation_steps: int | None
    shuffle: bool | None
    seed: int | None


@dataclass(frozen=True)
class CPTConfig:
    """Complete continued-pretraining configuration."""

    config_path: Path
    project_root: Path
    paths: CPTPaths
    training: CPTTrainingSettings
    early_stopping: EarlyStoppingConfig
    benchmark: BenchmarkSettings
    log_level: str


@dataclass(frozen=True)
class CPTResult:
    """Continued-pretraining run summary."""

    status: str
    global_step: int
    start_step: int
    source_checkpoint: Path
    last_checkpoint: Path | None
    best_checkpoint: Path | None
    checkpoint_directory: Path | None
    summary_path: Path
    benchmark_json: Path | None
    benchmark_markdown: Path | None


def load_cpt_config(path: Path | str = "configs/continued_pretraining.yaml") -> CPTConfig:
    """Load and validate the continued-pretraining YAML configuration."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Continued pretraining config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise CPTError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CPTError("Continued pretraining config must be a mapping.")
    root = _resolve(config_path.parent, raw.get("project_root", ".."))
    section = _mapping(raw.get("continued_pretraining", {}), "continued_pretraining")
    paths = _mapping(section.get("paths", {}), "continued_pretraining.paths")
    training = _mapping(section.get("training", {}), "continued_pretraining.training")
    early = _mapping(section.get("early_stopping", {}), "continued_pretraining.early_stopping")
    benchmark = _mapping(section.get("benchmark", {}), "continued_pretraining.benchmark")
    config = CPTConfig(
        config_path=config_path,
        project_root=root,
        paths=CPTPaths(
            training_config=_resolve(root, paths.get("training_config", "configs/training.yaml")),
            model_config=_resolve(root, paths.get("model_config", "configs/model.yaml")),
            optimizer_config=_resolve(
                root,
                paths.get("optimizer_config", "configs/optimizer.yaml"),
            ),
            generation_config=_resolve(
                root,
                paths.get("generation_config", "configs/generation.yaml"),
            ),
            corpus_directory=_resolve(
                root,
                paths.get("corpus_directory", "python_corpus/final_corpus/packed"),
            ),
            tokenizer=_resolve(root, paths.get("tokenizer", "data/tokenizer/tokenizer.json")),
            checkpoint_search_dir=_resolve(
                root,
                paths.get("checkpoint_search_dir", "checkpoints"),
            ),
            checkpoint_output_dir=_resolve(
                root,
                paths.get("checkpoint_output_dir", "checkpoints/continued_pretraining"),
            ),
            report_dir=_resolve(root, paths.get("report_dir", "reports/continued_pretraining")),
            log_file=_resolve(root, paths.get("log_file", "logs/continued_pretraining.jsonl")),
        ),
        training=CPTTrainingSettings(
            learning_rate=_optional_float(training.get("learning_rate"), "learning_rate"),
            batch_size=_optional_int(training.get("batch_size"), "batch_size"),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
            epochs=int(training.get("epochs", 1)),
            max_steps=int(training.get("max_steps", 1000)),
            checkpoint_interval_steps=int(training.get("checkpoint_interval_steps", 500)),
            validation_interval_steps=int(training.get("validation_interval_steps", 100)),
            log_interval_steps=int(training.get("log_interval_steps", 10)),
            warmup_steps=int(training.get("warmup_steps", 100)),
            weight_decay=_optional_float(training.get("weight_decay"), "weight_decay"),
            sequence_length=int(training.get("sequence_length", 1025)),
            device=str(training.get("device", "auto")),
            precision=normalize_mixed_precision(
                str(training.get("precision", "fp32")),
                allow_fp32_alias=True,
            ),
            max_grad_norm=_optional_float(training.get("max_grad_norm"), "max_grad_norm"),
            keep_last_checkpoints=int(training.get("keep_last_checkpoints", 3)),
            validation_fraction=_optional_float(
                training.get("validation_fraction"),
                "validation_fraction",
            ),
            validation_steps=_optional_int(training.get("validation_steps"), "validation_steps"),
            shuffle=(bool(training["shuffle"]) if training.get("shuffle") is not None else None),
            seed=_optional_int(training.get("seed"), "seed"),
        ),
        early_stopping=EarlyStoppingConfig(
            enabled=bool(early.get("enabled", False)),
            patience=int(early.get("patience", 3)),
            min_delta=float(early.get("min_improvement", early.get("min_delta", 0.0))),
            monitor="validation_loss",
            mode="min",
        ),
        benchmark=BenchmarkSettings(
            enabled=bool(benchmark.get("enabled", True)),
            validation_batches=int(benchmark.get("validation_batches", 2)),
            prompt_count=int(benchmark.get("prompt_count", 3)),
            max_new_tokens=int(benchmark.get("max_new_tokens", 32)),
        ),
        log_level=str(_mapping(raw.get("logging", {}), "logging").get("level", "INFO")).upper(),
    )
    _validate_cpt_config(config)
    return config


def resolve_cpt_checkpoint(resume: str | Path | None, config: CPTConfig) -> Path:
    """Resolve ``latest`` or an explicit checkpoint file/directory to a .pt path."""

    if resume is not None and str(resume) != "latest":
        candidate = Path(resume)
        if not candidate.is_absolute():
            candidate = config.project_root / candidate
        if candidate.is_dir():
            candidate = candidate / "model.pt"
        if not candidate.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
        return candidate.resolve()
    step_dirs = _cpt_step_directories(config.paths.checkpoint_output_dir)
    for _step, directory in step_dirs:
        model_path = directory / "model.pt"
        if model_path.is_file():
            return model_path.resolve()
    canonical = config.paths.checkpoint_output_dir / "last_checkpoint.pt"
    if canonical.is_file():
        return canonical.resolve()
    return resolve_latest_phase6_checkpoint(
        None,
        project_root=config.project_root,
        search_directory=config.paths.checkpoint_search_dir,
    )


def check_cpt_readiness(config: CPTConfig, resume: str | Path | None) -> dict[str, Any]:
    """Validate corpus shards, tokenizer, and source checkpoint before training."""

    index_path = config.paths.corpus_directory / "index.json"
    if not index_path.is_file():
        raise CPTError(f"Final corpus shard index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise CPTError(f"Final corpus shard index must be a JSON object: {index_path}")
    if index.get("format") != "genpy_uint16_packed_sequence_shards":
        raise CPTError("Final corpus shards use an unsupported format.")
    sequence_length = int(index.get("sequence_length") or 0)
    if sequence_length != config.training.sequence_length:
        raise CPTError(
            f"Configured sequence_length={config.training.sequence_length} does not match "
            f"packed shards (sequence_length={sequence_length})."
        )
    expected_hash = tokenizer_file_hash(config.paths.tokenizer)
    if index.get("tokenizer_sha256") != expected_hash:
        raise CPTError("Final corpus tokenizer hash does not match the tokenizer file.")
    shard_files = sorted(config.paths.corpus_directory.glob("*.bin"))
    if not shard_files:
        raise CPTError(f"No packed shards found in {config.paths.corpus_directory}")
    checkpoint = resolve_cpt_checkpoint(resume, config)
    validate_checkpoint_tokenizer(checkpoint, tokenizer_path=config.paths.tokenizer)
    return {
        "checkpoint": checkpoint,
        "index": index_path,
        "sequence_length": sequence_length,
        "shard_count": len(shard_files),
        "sequence_count": int(index.get("sequence_count") or 0),
        "token_count": int(index.get("token_count") or 0),
    }


class CPTCosineScheduler(CosineWarmupScheduler):
    """Cosine warmup/decay applied to the continued-pretraining leg only.

    ``step_count`` stays the absolute global step so checkpoints resume exactly;
    warmup and decay are computed relative to ``start_step``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        start_step: int,
        leg_steps: int,
        warmup_steps: int,
        minimum_learning_rate_ratio: float,
        last_step: int | None = None,
    ) -> None:
        self.start_step = int(start_step)
        super().__init__(
            optimizer,
            max_steps=self.start_step + max(1, int(leg_steps)),
            warmup_steps=warmup_steps,
            minimum_learning_rate_ratio=minimum_learning_rate_ratio,
            last_step=self.start_step if last_step is None else int(last_step),
        )

    def _factor(self, step: int) -> float:
        leg_step = max(0, int(step) - self.start_step)
        leg_total = max(1, self.max_steps - self.start_step)
        if self.warmup_steps and leg_step < self.warmup_steps:
            return max(1e-12, leg_step / self.warmup_steps)
        progress = min(
            1.0,
            max(0.0, (leg_step - self.warmup_steps) / max(1, leg_total - self.warmup_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.minimum_learning_rate_ratio
        return floor + (1.0 - floor) * cosine

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["start_step"] = self.start_step
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.start_step = int(state.get("start_step", self.start_step))
        super().load_state_dict(state)


@dataclass(frozen=True)
class _LoopResult:
    status: str
    reason: str | None
    last_checkpoint: Path | None
    checkpoint_directory: Path | None


class CPTTrainer(Phase6Trainer):
    """Existing Phase 6 training loop driven for one continued-pretraining leg."""

    def __init__(
        self,
        config: Phase6Config,
        *,
        cpt: CPTConfig,
        source_checkpoint: Path,
        monitor: TrainingMonitor,
    ) -> None:
        self.cpt = cpt
        self.source_checkpoint = source_checkpoint
        self.monitor = monitor
        self.early_stopping = EarlyStoppingState(cpt.early_stopping)
        self.start_step = 0
        self.target_step = 0
        self._validation_improved = False
        self._last_saved_step: int | None = None
        self._step_seconds: deque = deque(maxlen=50)
        super().__init__(config)
        self._resume_cpt()
        self.optimizer.zero_grad(set_to_none=True)

    def train_cpt(self) -> _LoopResult:
        """Run continued pretraining for the configured additional steps."""

        self._prepare_outputs()
        LOGGER.info(
            "cpt_training_started start_step=%d target_step=%d epochs=%d device=%s",
            self.start_step,
            self.target_step,
            self.cpt.training.epochs,
            self.device,
        )
        status = "completed"
        reason = None
        last_checkpoint: Path | None = None
        checkpoint_directory: Path | None = None
        training_started = time.perf_counter()
        epochs_completed = 0
        while self.global_step < self.target_step and epochs_completed < self.cpt.training.epochs:
            self.epoch += 1
            epochs_completed += 1
            sampler = DeterministicSequenceSampler(
                Subset(self.dataset, self.train_indices),
                shuffle=self.config.data.shuffle,
                seed=self.config.data.seed,
            )
            sampler.set_epoch(self.epoch)
            loader = self._loader(Subset(self.dataset, self.train_indices), sampler=sampler)
            for batch in loader:
                if self.global_step >= self.target_step:
                    break
                metrics = self._train_micro_batch(batch)
                if metrics is None:
                    continue
                self._check_finite(metrics)
                self.global_step += 1
                self.latest_training_loss = metrics["loss"]
                self._step_seconds.append(float(metrics["elapsed_seconds"]))
                validation_loss = None
                if self._cpt_should_validate():
                    validation_loss = self.evaluate()
                    self.latest_validation_loss = validation_loss
                self._log_cpt_step(metrics, validation_loss)
                if validation_loss is not None:
                    if self.best_metric is None or validation_loss < self.best_metric:
                        self.best_metric = validation_loss
                        self._validation_improved = True
                    _improved, should_stop = self.early_stopping.update(validation_loss)
                    if should_stop:
                        status = "early_stopped"
                        reason = "early_stopping"
                        break
                if self._cpt_should_checkpoint():
                    checkpoint_directory = self._save_cpt_checkpoint()
                    last_checkpoint = (
                        self.cpt.paths.checkpoint_output_dir / "last_checkpoint.pt"
                    )
            checkpoint_directory = self._save_cpt_checkpoint()
            last_checkpoint = self.cpt.paths.checkpoint_output_dir / "last_checkpoint.pt"
            if status == "early_stopped":
                break
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        LOGGER.info(
            "cpt_training_finished status=%s global_step=%d elapsed_seconds=%.2f",
            status,
            self.global_step,
            time.perf_counter() - training_started,
        )
        best_checkpoint = self.cpt.paths.checkpoint_output_dir / "best_checkpoint.pt"
        self.monitor.write_summary(
            status=status,
            reason=reason,
            source_checkpoint=self.source_checkpoint,
            last_checkpoint=last_checkpoint,
            best_checkpoint=best_checkpoint if best_checkpoint.is_file() else None,
            global_step=self.global_step,
            best_metric=self.best_metric,
        )
        return _LoopResult(status, reason, last_checkpoint, checkpoint_directory)

    def _resume_cpt(self) -> None:
        loaded = load_checkpoint(
            self.source_checkpoint,
            self.model,
            self.optimizer,
            scheduler=None,
            scaler=self.scaler,
            # CPU keeps the saved RNG state a CPU ByteTensor (set_rng_state requires
            # it); load_checkpoint then moves optimizer tensors to the model device.
            map_location="cpu",
        )
        self.epoch = loaded.epoch
        self.global_step = loaded.global_step
        self.micro_step = self.global_step * self.config.training.gradient_accumulation_steps
        self.latest_training_loss = loaded.training_loss
        self.latest_validation_loss = loaded.validation_loss
        cpt_state = loaded.extra_state.get("cpt") if isinstance(loaded.extra_state, dict) else None
        if loaded.extra_state.get("phase") == CPT_PHASE and isinstance(cpt_state, dict):
            self.start_step = int(cpt_state.get("start_step", self.global_step))
            self.best_metric = loaded.best_metric
            early = cpt_state.get("early_stopping")
            if isinstance(early, dict):
                if early.get("best_metric") is not None:
                    self.early_stopping.best_metric = float(early["best_metric"])
                self.early_stopping.bad_epochs = int(early.get("bad_epochs", 0))
            LOGGER.info(
                "cpt_resumed_interrupted_leg checkpoint=%s step=%d start_step=%d",
                loaded.checkpoint_path,
                self.global_step,
                self.start_step,
            )
        else:
            self.start_step = self.global_step
            self.best_metric = None
            LOGGER.info(
                "cpt_started_new_leg source=%s step=%d",
                loaded.checkpoint_path,
                self.global_step,
            )
        self.target_step = self.start_step + self.cpt.training.max_steps
        base_lr = self.config.optimizer.learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr
        self.scheduler = CPTCosineScheduler(
            self.optimizer,
            start_step=self.start_step,
            leg_steps=self.cpt.training.max_steps,
            warmup_steps=self.cpt.training.warmup_steps,
            minimum_learning_rate_ratio=self.config.scheduler.minimum_learning_rate_ratio,
            last_step=self.global_step,
        )

    def _save_cpt_checkpoint(self) -> Path:
        output_dir = self.cpt.paths.checkpoint_output_dir
        step_dir = output_dir / f"checkpoint_step_{self.global_step:05d}"
        if self._last_saved_step == self.global_step and step_dir.is_dir():
            return step_dir
        step_dir.mkdir(parents=True, exist_ok=True)
        extra_state = {
            "phase": CPT_PHASE,
            "source_checkpoint": str(self.source_checkpoint),
            "corpus_index": str(self.cpt.paths.corpus_directory / "index.json"),
            "training_config": str(self.cpt.config_path),
            "cpt": {
                "start_step": self.start_step,
                "target_step": self.target_step,
                "early_stopping": {
                    "best_metric": self.early_stopping.best_metric,
                    "bad_epochs": self.early_stopping.bad_epochs,
                },
            },
        }
        model_path = step_dir / "model.pt"
        save_checkpoint(
            model_path,
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
                "tokenizer": str(self.cpt.paths.tokenizer),
                "tokenizer_sha256": tokenizer_file_hash(self.cpt.paths.tokenizer),
            },
            extra_state=extra_state,
        )
        torch.save(self.optimizer.state_dict(), step_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), step_dir / "scheduler.pt")
        trainer_state = {
            "phase": CPT_PHASE,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "start_step": self.start_step,
            "target_step": self.target_step,
            "training_loss": self.latest_training_loss,
            "validation_loss": self.latest_validation_loss,
            "best_metric": self.best_metric,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "early_stopping": extra_state["cpt"]["early_stopping"],
            "source_checkpoint": str(self.source_checkpoint),
            "saved_at": datetime.now(UTC).isoformat(),
        }
        (step_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (step_dir / "config.json").write_text(
            json.dumps(_config_payload(self.cpt), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        last_path = output_dir / "last_checkpoint.pt"
        shutil.copy2(model_path, last_path)
        if self._validation_improved:
            shutil.copy2(model_path, output_dir / "best_checkpoint.pt")
            self._validation_improved = False
        self._rotate_cpt_checkpoints(output_dir)
        self._last_saved_step = self.global_step
        LOGGER.info(
            "cpt_checkpoint_saved step=%d directory=%s size_bytes=%d",
            self.global_step,
            step_dir,
            model_path.stat().st_size,
        )
        self.monitor.log_checkpoint(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "checkpoint": str(step_dir),
                "last_checkpoint": str(last_path),
                "checkpoint_size_bytes": model_path.stat().st_size,
            }
        )
        return step_dir

    def _rotate_cpt_checkpoints(self, output_dir: Path) -> None:
        keep_last = self.cpt.training.keep_last_checkpoints
        if keep_last <= 0:
            return
        for _step, directory in _cpt_step_directories(output_dir)[keep_last:]:
            shutil.rmtree(directory, ignore_errors=True)

    def _log_cpt_step(self, metrics: dict[str, Any], validation_loss: float | None) -> None:
        remaining_steps = max(0, self.target_step - self.global_step)
        average_seconds = (
            sum(self._step_seconds) / len(self._step_seconds) if self._step_seconds else 0.0
        )
        remaining_seconds = remaining_steps * average_seconds
        self.monitor.log(
            {
                "epoch": self.epoch,
                "step": self.global_step,
                "training_loss": metrics["loss"],
                "validation_loss": validation_loss,
                "perplexity": (
                    math.exp(min(20.0, validation_loss)) if validation_loss is not None else None
                ),
                "learning_rate": metrics["learning_rate"],
                "tokens_per_second": metrics["tokens_per_second"],
                "examples_per_second": metrics["examples_per_second"],
                "tokens_processed": metrics["tokens"],
                "gradient_norm": metrics["gradient_norm"],
                "gpu_memory_mb": metrics.get("gpu_memory_mb") or peak_memory_mb(self.device),
                "remaining_seconds": round(remaining_seconds, 3),
            }
        )
        if self.global_step % self.cpt.training.log_interval_steps == 0:
            LOGGER.info(
                "cpt step=%d/%d epoch=%d loss=%.6f lr=%.8f tokens_per_sec=%.1f "
                "examples_per_sec=%.2f gpu_memory_mb=%.1f remaining=%s",
                self.global_step,
                self.target_step,
                self.epoch,
                metrics["loss"],
                metrics["learning_rate"],
                metrics["tokens_per_second"],
                metrics["examples_per_second"],
                metrics.get("gpu_memory_mb") or 0.0,
                _format_duration(remaining_seconds),
            )

    def _cpt_should_validate(self) -> bool:
        interval = self.cpt.training.validation_interval_steps
        return bool(self.validation_indices) and interval > 0 and self.global_step % interval == 0

    def _cpt_should_checkpoint(self) -> bool:
        interval = self.cpt.training.checkpoint_interval_steps
        return interval > 0 and self.global_step % interval == 0

    def _check_finite(self, metrics: dict[str, Any]) -> None:
        loss = float(metrics["loss"])
        grad_norm = float(metrics.get("gradient_norm") or 0.0)
        if not math.isfinite(loss):
            raise CPTError("NaN or infinite loss detected; aborting continued pretraining.")
        if not math.isfinite(grad_norm):
            raise CPTError(
                "NaN or infinite gradient norm detected; aborting continued pretraining."
            )


def run_cpt(config: CPTConfig, *, resume: str | Path | None = "latest") -> CPTResult:
    """Run one continued-pretraining leg on the Final Corpus."""

    setup_structured_logging(config.paths.log_file, config.log_level)
    readiness = check_cpt_readiness(config, resume)
    source_checkpoint = readiness["checkpoint"]
    LOGGER.info(
        "cpt_readiness_passed checkpoint=%s shards=%d sequences=%d tokens=%d",
        source_checkpoint,
        readiness["shard_count"],
        readiness["sequence_count"],
        readiness["token_count"],
    )
    phase6 = build_phase6_config(config, source_checkpoint)
    monitor = TrainingMonitor(config.paths.report_dir)
    trainer = CPTTrainer(
        phase6,
        cpt=config,
        source_checkpoint=source_checkpoint,
        monitor=monitor,
    )
    loop = trainer.train_cpt()
    _retitle_summary(monitor.summary_path)
    benchmark_json = None
    benchmark_markdown = None
    if config.benchmark.enabled and loop.last_checkpoint is not None:
        comparison = benchmark_phase63_checkpoints(
            config=phase6,
            previous_checkpoint=source_checkpoint,
            continued_checkpoint=loop.last_checkpoint,
            output_dir=config.paths.report_dir,
            device=trainer.device,
            settings=config.benchmark,
        )
        benchmark_json = config.paths.report_dir / "comparison_report.json"
        benchmark_markdown = config.paths.report_dir / "comparison_report.md"
        LOGGER.info(
            "cpt_benchmark_completed validation_loss_delta=%s perplexity_delta=%s",
            comparison.validation_loss_delta,
            comparison.perplexity_delta,
        )
    best_checkpoint = config.paths.checkpoint_output_dir / "best_checkpoint.pt"
    return CPTResult(
        status=loop.status,
        global_step=trainer.global_step,
        start_step=trainer.start_step,
        source_checkpoint=source_checkpoint,
        last_checkpoint=loop.last_checkpoint,
        best_checkpoint=best_checkpoint if best_checkpoint.is_file() else None,
        checkpoint_directory=loop.checkpoint_directory,
        summary_path=monitor.summary_path,
        benchmark_json=benchmark_json,
        benchmark_markdown=benchmark_markdown,
    )


def build_phase6_config(config: CPTConfig, source_checkpoint: Path) -> Phase6Config:
    """Adapt the existing Phase 6 configuration to the Final Corpus CPT leg."""

    phase6 = load_phase6_config(
        config.paths.training_config,
        model_config=config.paths.model_config,
        optimizer_config=config.paths.optimizer_config,
        generation_config=config.paths.generation_config,
    )
    if phase6.model.context_length + 1 != config.training.sequence_length:
        raise CPTError(
            f"Model context_length={phase6.model.context_length} is incompatible with "
            f"sequence_length={config.training.sequence_length}; expected context_length + 1."
        )
    index_path = config.paths.corpus_directory / "index.json"
    data = replace(
        phase6.data,
        shard_pattern=str(config.paths.corpus_directory / "*.bin"),
        shard_index=index_path,
        training_manifest=index_path,
        tokenizer=config.paths.tokenizer,
        batch_size=config.training.batch_size or phase6.data.batch_size,
        validation_fraction=(
            config.training.validation_fraction
            if config.training.validation_fraction is not None
            else phase6.data.validation_fraction
        ),
        shuffle=(
            config.training.shuffle
            if config.training.shuffle is not None
            else phase6.data.shuffle
        ),
        seed=config.training.seed if config.training.seed is not None else phase6.data.seed,
    )
    training = replace(
        phase6.training,
        device=config.training.device,
        mixed_precision=config.training.precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_steps=max(phase6.training.max_steps, config.training.max_steps),
        log_every_steps=config.training.log_interval_steps,
        save_every_steps=config.training.checkpoint_interval_steps,
        validate_every_steps=config.training.validation_interval_steps,
        validation_steps=(
            config.training.validation_steps
            if config.training.validation_steps is not None
            else phase6.training.validation_steps
        ),
        max_grad_norm=(
            config.training.max_grad_norm
            if config.training.max_grad_norm is not None
            else phase6.training.max_grad_norm
        ),
        resume=False,
        resume_from=None,
    )
    optimizer = phase6.optimizer
    if config.training.learning_rate is not None:
        optimizer = replace(optimizer, learning_rate=config.training.learning_rate)
    if config.training.weight_decay is not None:
        optimizer = replace(optimizer, weight_decay=config.training.weight_decay)
    scheduler = replace(phase6.scheduler, warmup_steps=config.training.warmup_steps)
    checkpoint = replace(
        phase6.checkpoint,
        directory=config.paths.checkpoint_output_dir,
        best_filename="best_checkpoint.pt",
        last_filename="last_checkpoint.pt",
    )
    outputs = replace(
        phase6.outputs,
        metrics_directory=config.paths.report_dir,
        samples_directory=config.paths.report_dir / "samples",
        tensorboard_directory=config.paths.report_dir / "tensorboard",
        log_file=config.paths.log_file,
    )
    return replace(
        phase6,
        data=data,
        training=training,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint=checkpoint,
        outputs=outputs,
    )


def run_cpt_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for continued pretraining on the Final Corpus."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Continue pretraining the GenPy GPT model on the Final Corpus."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/continued_pretraining.yaml"),
    )
    parser.add_argument(
        "--resume",
        default="latest",
        help="'latest' or a checkpoint file/directory path",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_cpt_config(args.config)
        if args.max_steps is not None or args.device is not None or args.skip_benchmark:
            config = _override_cli(config, args)
            _validate_cpt_config(config)
        result = run_cpt(config, resume=args.resume)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Continued pretraining complete")
    print(f"Status: {result.status}")
    print(f"Source checkpoint: {result.source_checkpoint}")
    print(f"Start step: {result.start_step}")
    print(f"Global step: {result.global_step}")
    print(f"Checkpoint directory: {result.checkpoint_directory}")
    print(f"Last checkpoint: {result.last_checkpoint}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Summary: {result.summary_path}")
    print(f"Benchmark JSON: {result.benchmark_json}")
    print(f"Benchmark report: {result.benchmark_markdown}")
    return 0


def _override_cli(config: CPTConfig, args: argparse.Namespace) -> CPTConfig:
    training = config.training
    benchmark = config.benchmark
    if args.max_steps is not None:
        training = replace(training, max_steps=args.max_steps)
    if args.device is not None:
        training = replace(training, device=args.device)
    if args.skip_benchmark:
        benchmark = replace(benchmark, enabled=False)
    return replace(config, training=training, benchmark=benchmark)


def _retitle_summary(summary_path: Path) -> None:
    if not summary_path.is_file():
        return
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = "# GenPy Continued Pretraining (Final Corpus) Summary"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cpt_step_directories(output_dir: Path) -> list[tuple[int, Path]]:
    if not output_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        match = CHECKPOINT_DIR_PATTERN.fullmatch(path.name)
        if match is not None:
            found.append((int(match.group("step")), path))
    return sorted(found, key=lambda item: item[0], reverse=True)


def _config_payload(config: CPTConfig) -> dict[str, Any]:
    payload = asdict(config)
    return _stringify_paths(payload)


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value


def _validate_cpt_config(config: CPTConfig) -> None:
    training = config.training
    if training.max_steps <= 0:
        raise CPTError("continued_pretraining.training.max_steps must be positive.")
    if training.epochs <= 0:
        raise CPTError("continued_pretraining.training.epochs must be positive.")
    if training.gradient_accumulation_steps <= 0:
        raise CPTError("gradient_accumulation_steps must be positive.")
    if training.checkpoint_interval_steps <= 0:
        raise CPTError("checkpoint_interval_steps must be positive.")
    if training.validation_interval_steps < 0:
        raise CPTError("validation_interval_steps must be non-negative.")
    if training.log_interval_steps <= 0:
        raise CPTError("log_interval_steps must be positive.")
    if training.warmup_steps < 0:
        raise CPTError("warmup_steps must be non-negative.")
    if training.warmup_steps >= training.max_steps:
        raise CPTError("warmup_steps must be smaller than max_steps.")
    if training.sequence_length <= 1:
        raise CPTError("sequence_length must be greater than one.")
    if training.keep_last_checkpoints <= 0:
        raise CPTError("keep_last_checkpoints must be positive.")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CPTError(f"{name} must be a number or null.") from exc


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CPTError(f"{name} must be an integer or null.") from exc


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CPTError(f"{name} must be a mapping.")
    return value


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CPTError("Configured path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


__all__ = [
    "CPTConfig",
    "CPTCosineScheduler",
    "CPTError",
    "CPTResult",
    "CPTTrainer",
    "build_phase6_config",
    "check_cpt_readiness",
    "load_cpt_config",
    "resolve_cpt_checkpoint",
    "run_cpt",
    "run_cpt_cli",
]
