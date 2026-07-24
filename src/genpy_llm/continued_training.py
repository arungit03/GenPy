"""Phase 6.3 continued pretraining from Corpus V2."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import Subset

from genpy_llm.benchmark_monitor import BenchmarkSettings, benchmark_phase63_checkpoints
from genpy_llm.checkpoint_manager import (
    resolve_latest_phase6_checkpoint,
    save_phase63_checkpoint,
    validate_checkpoint_tokenizer,
)
from genpy_llm.code_tokenizer import tokenizer_file_hash
from genpy_llm.logging_utils import setup_structured_logging
from genpy_llm.performance import peak_memory_mb
from genpy_llm.pretraining import Phase6Config, Phase6Trainer, load_phase6_config
from genpy_llm.pretraining_dataset import DeterministicSequenceSampler
from genpy_llm.shard_builder import final_outputs_valid
from genpy_llm.training_monitor import (
    EarlyStoppingConfig,
    EarlyStoppingState,
    TrainingMonitor,
)

LOGGER = logging.getLogger("genpy_llm.continued_training")


class Phase63Error(RuntimeError):
    """Raised when Phase 6.3 continued pretraining cannot continue."""


@dataclass(frozen=True)
class Phase63Paths:
    """Phase 6.3 config and artifact paths."""

    phase6_training_config: Path
    model_config: Path
    optimizer_config: Path
    generation_config: Path
    corpus_index: Path
    corpus_manifest: Path
    corpus_statistics: Path
    corpus_report_manifest: Path
    corpus_quality_report: Path
    checkpoint_search_dir: Path
    checkpoint_output_dir: Path
    report_dir: Path
    log_file: Path


@dataclass(frozen=True)
class Phase63TrainingSettings:
    """Phase 6.3 training controls."""

    source_checkpoint: Path | None
    max_steps: int
    max_epochs: int
    batch_size: int | None
    learning_rate: float | None
    weight_decay: float | None
    device: str
    validation_interval_steps: int
    checkpoint_interval_steps: int
    max_grad_norm: float | None
    exploding_gradient_threshold: float


@dataclass(frozen=True)
class Phase63Config:
    """Complete Phase 6.3 configuration."""

    config_path: Path
    project_root: Path
    paths: Phase63Paths
    training: Phase63TrainingSettings
    early_stopping: EarlyStoppingConfig
    benchmark: BenchmarkSettings
    log_level: str


@dataclass(frozen=True)
class Phase63Result:
    """Phase 6.3 training result."""

    status: str
    global_step: int
    source_checkpoint: Path
    last_checkpoint: Path | None
    best_checkpoint: Path | None
    summary_path: Path
    benchmark_json: Path | None
    benchmark_markdown: Path | None


def load_phase63_config(path: Path | str = "configs/phase6_3.yaml") -> Phase63Config:
    """Load Phase 6.3 YAML config."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Phase 6.3 config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise Phase63Error(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise Phase63Error("Phase 6.3 config must be a mapping.")
    root = _resolve(config_path.parent, raw.get("project_root", ".."))
    section = _mapping(raw.get("phase6_3", {}), "phase6_3")
    paths = _mapping(section.get("paths", {}), "phase6_3.paths")
    training = _mapping(section.get("training", {}), "phase6_3.training")
    early = _mapping(section.get("early_stopping", {}), "phase6_3.early_stopping")
    benchmark = _mapping(section.get("benchmark", {}), "phase6_3.benchmark")
    report_dir = _resolve(root, paths.get("report_dir", "reports/pretraining_v2"))
    output_dir = _resolve(root, paths.get("checkpoint_output_dir", "checkpoints/pretraining_v2"))
    config = Phase63Config(
        config_path=config_path,
        project_root=root,
        paths=Phase63Paths(
            phase6_training_config=_resolve(
                root,
                paths.get("phase6_training_config", "configs/training.yaml"),
            ),
            model_config=_resolve(root, paths.get("model_config", "configs/model.yaml")),
            optimizer_config=_resolve(
                root,
                paths.get("optimizer_config", "configs/optimizer.yaml"),
            ),
            generation_config=_resolve(
                root,
                paths.get("generation_config", "configs/generation.yaml"),
            ),
            corpus_index=_resolve(root, paths.get("corpus_index", "data/corpus_v2/index.json")),
            corpus_manifest=_resolve(
                root,
                paths.get("corpus_manifest", "data/corpus_v2/document_manifest.jsonl"),
            ),
            corpus_statistics=_resolve(
                root,
                paths.get("corpus_statistics", "data/corpus_v2/statistics.json"),
            ),
            corpus_report_manifest=_resolve(
                root,
                paths.get("corpus_report_manifest", "reports/corpus_v2/manifest.json"),
            ),
            corpus_quality_report=_resolve(
                root,
                paths.get("corpus_quality_report", "reports/corpus_v2/quality_report.json"),
            ),
            checkpoint_search_dir=_resolve(
                root,
                paths.get("checkpoint_search_dir", "checkpoints"),
            ),
            checkpoint_output_dir=output_dir,
            report_dir=report_dir,
            log_file=_resolve(root, paths.get("log_file", "logs/phase6_3.jsonl")),
        ),
        training=Phase63TrainingSettings(
            source_checkpoint=(
                _resolve(root, training["source_checkpoint"])
                if training.get("source_checkpoint") is not None
                else None
            ),
            max_steps=int(training.get("max_steps", 1000)),
            max_epochs=int(training.get("max_epochs", 1)),
            batch_size=(
                int(training["batch_size"]) if training.get("batch_size") is not None else None
            ),
            learning_rate=(
                float(training["learning_rate"])
                if training.get("learning_rate") is not None
                else None
            ),
            weight_decay=(
                float(training["weight_decay"])
                if training.get("weight_decay") is not None
                else None
            ),
            device=str(training.get("device", "auto")),
            validation_interval_steps=int(training.get("validation_interval_steps", 100)),
            checkpoint_interval_steps=int(training.get("checkpoint_interval_steps", 500)),
            max_grad_norm=(
                float(training["max_grad_norm"])
                if training.get("max_grad_norm") is not None
                else None
            ),
            exploding_gradient_threshold=float(
                training.get("exploding_gradient_threshold", 1_000.0)
            ),
        ),
        early_stopping=EarlyStoppingConfig(
            enabled=bool(early.get("enabled", True)),
            patience=int(early.get("patience", 3)),
            min_delta=float(early.get("min_delta", 0.0)),
            monitor=str(early.get("monitor", "validation_loss")),
            mode=str(early.get("mode", "min")),
        ),
        benchmark=BenchmarkSettings(
            enabled=bool(benchmark.get("enabled", True)),
            validation_batches=int(benchmark.get("validation_batches", 1)),
            prompt_count=int(benchmark.get("prompt_count", 3)),
            max_new_tokens=int(benchmark.get("max_new_tokens", 16)),
        ),
        log_level=str(_mapping(raw.get("logging", {}), "logging").get("level", "INFO")).upper(),
    )
    _validate_phase63_config(config)
    return config


def run_phase63(config: Phase63Config) -> Phase63Result:
    """Run Phase 6.3 continued pretraining with hard readiness checks."""

    setup_structured_logging(config.paths.log_file, config.log_level)
    readiness = check_phase63_readiness(config)
    source_checkpoint = readiness["checkpoint"]
    phase6 = _phase6_config(config, source_checkpoint)
    monitor = TrainingMonitor(config.paths.report_dir)
    trainer = Phase63Trainer(
        phase6,
        phase63=config,
        source_checkpoint=source_checkpoint,
        monitor=monitor,
    )
    result = trainer.train_phase63()
    benchmark_json = None
    benchmark_markdown = None
    if config.benchmark.enabled and result.last_checkpoint is not None:
        comparison = benchmark_phase63_checkpoints(
            config=phase6,
            previous_checkpoint=source_checkpoint,
            continued_checkpoint=result.last_checkpoint,
            output_dir=config.paths.report_dir,
            device=trainer.device,
            settings=config.benchmark,
        )
        benchmark_json = config.paths.report_dir / "comparison_report.json"
        benchmark_markdown = config.paths.report_dir / "comparison_report.md"
        LOGGER.info(
            "phase63_benchmark_completed validation_loss_delta=%s perplexity_delta=%s",
            comparison.validation_loss_delta,
            comparison.perplexity_delta,
        )
    return Phase63Result(
        status=result.status,
        global_step=result.global_step,
        source_checkpoint=source_checkpoint,
        last_checkpoint=result.last_checkpoint,
        best_checkpoint=result.best_checkpoint,
        summary_path=monitor.summary_path,
        benchmark_json=benchmark_json,
        benchmark_markdown=benchmark_markdown,
    )


def check_phase63_readiness(config: Phase63Config) -> dict[str, Any]:
    """Validate corpus, reports, checkpoint, and tokenizer before training."""

    for path in (
        config.paths.corpus_index,
        config.paths.corpus_manifest,
        config.paths.corpus_statistics,
        config.paths.corpus_report_manifest,
        config.paths.corpus_quality_report,
    ):
        if not path.is_file():
            raise Phase63Error(f"Required Corpus V2 artifact is missing: {path}")
    report_manifest = _read_json(config.paths.corpus_report_manifest)
    quality_report = _read_json(config.paths.corpus_quality_report)
    corpus_statistics = _read_json(config.paths.corpus_statistics)
    readiness = _mapping(report_manifest.get("readiness", {}), "corpus readiness")
    if readiness.get("passed") is not True:
        failures = readiness.get("failures")
        raise Phase63Error(f"Corpus V2 readiness gate failed: {failures}")
    total_tokens = int(report_manifest.get("statistics", {}).get("total_tokens") or 0)
    minimum_tokens = int(
        quality_report.get("readiness", {}).get("settings", {}).get("minimum_tokens") or 0
    )
    if total_tokens < minimum_tokens:
        raise Phase63Error(
            f"Corpus token target not reached: tokens={total_tokens} target={minimum_tokens}"
        )
    build_fingerprint = str(report_manifest.get("build_fingerprint") or "")
    if not build_fingerprint:
        raise Phase63Error("Corpus V2 manifest is missing build_fingerprint.")
    if not final_outputs_valid(
        config.paths.corpus_index,
        config.paths.corpus_statistics,
        build_fingerprint,
    ):
        raise Phase63Error("Corpus V2 packed shard validation failed.")
    if corpus_statistics.get("build_fingerprint") != build_fingerprint:
        raise Phase63Error("Corpus V2 statistics fingerprint does not match manifest.")
    tokenizer_path = _resolve(config.project_root, report_manifest.get("tokenizer"))
    expected_hash = tokenizer_file_hash(tokenizer_path)
    if report_manifest.get("tokenizer_sha256") != expected_hash:
        raise Phase63Error("Corpus V2 tokenizer hash does not match tokenizer file.")
    checkpoint = resolve_latest_phase6_checkpoint(
        config.training.source_checkpoint,
        project_root=config.project_root,
        search_directory=config.paths.checkpoint_search_dir,
    )
    validate_checkpoint_tokenizer(checkpoint, tokenizer_path=tokenizer_path)
    return {
        "checkpoint": checkpoint,
        "tokenizer": tokenizer_path,
        "tokens": total_tokens,
        "build_fingerprint": build_fingerprint,
    }


@dataclass(frozen=True)
class _TrainingResult:
    status: str
    global_step: int
    last_checkpoint: Path | None
    best_checkpoint: Path | None


class Phase63Trainer(Phase6Trainer):
    """Phase 6 trainer with Phase 6.3 checkpoint names, monitoring, and early stop."""

    def __init__(
        self,
        config: Phase6Config,
        *,
        phase63: Phase63Config,
        source_checkpoint: Path,
        monitor: TrainingMonitor,
    ) -> None:
        self.phase63 = phase63
        self.source_checkpoint = source_checkpoint
        self.monitor = monitor
        self.early_stopping = EarlyStoppingState(phase63.early_stopping)
        super().__init__(config)
        self.target_step = self.global_step + phase63.training.max_steps
        self._latest_validation_improved = False
        self.optimizer.zero_grad(set_to_none=True)

    def train_phase63(self) -> _TrainingResult:
        """Run continued pretraining for configured additional steps."""

        self._prepare_outputs()
        status = "completed"
        reason = None
        last_checkpoint: Path | None = None
        best_checkpoint: Path | None = (
            self.phase63.paths.checkpoint_output_dir / "best_checkpoint.pt"
        )
        epochs_completed = 0
        while (
            self.global_step < self.target_step
            and epochs_completed < self.phase63.training.max_epochs
        ):
            self.epoch += 1
            epochs_completed += 1
            sampler = DeterministicSequenceSampler(
                Subset(self.dataset, self.train_indices),
                shuffle=self.config.data.shuffle,
                seed=self.config.data.seed,
            )
            sampler.set_epoch(self.epoch)
            loader = self._loader(Subset(self.dataset, self.train_indices), sampler=sampler)
            epoch_tokens = 0
            for batch in loader:
                if self.global_step >= self.target_step:
                    break
                metrics = self._train_micro_batch(batch)
                if metrics is None:
                    continue
                self._safety_check(metrics)
                self.global_step += 1
                self.latest_training_loss = metrics["loss"]
                epoch_tokens += int(metrics["tokens"])
                validation_loss = None
                if self._phase63_should_validate():
                    validation_loss = self.evaluate()
                    self.latest_validation_loss = validation_loss
                self.monitor.log(
                    {
                        "epoch": self.epoch,
                        "step": self.global_step,
                        "training_loss": metrics["loss"],
                        "validation_loss": validation_loss,
                        "perplexity": math.exp(min(20.0, validation_loss))
                        if validation_loss is not None
                        else None,
                        "learning_rate": metrics["learning_rate"],
                        "tokens_processed": epoch_tokens,
                        "tokens_per_second": metrics["tokens_per_second"],
                        "gradient_norm": metrics["gradient_norm"],
                        "gpu_memory_mb": (
                            metrics.get("gpu_memory_mb") or peak_memory_mb(self.device)
                        ),
                    }
                )
                if validation_loss is not None:
                    improved, should_stop = self.early_stopping.update(validation_loss)
                    self._latest_validation_improved = improved
                    if improved:
                        self.best_metric = validation_loss
                    if should_stop:
                        status = "early_stopped"
                        reason = "early_stopping"
                        break
                if self._phase63_should_checkpoint():
                    last_checkpoint = self._save_phase63_checkpoint()
            last_checkpoint = self._save_phase63_checkpoint()
            if status == "early_stopped":
                break
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        if best_checkpoint is not None and not best_checkpoint.is_file():
            best_checkpoint = None
        self.monitor.write_summary(
            status=status,
            reason=reason,
            source_checkpoint=self.source_checkpoint,
            last_checkpoint=last_checkpoint,
            best_checkpoint=best_checkpoint,
            global_step=self.global_step,
            best_metric=self.best_metric,
        )
        return _TrainingResult(status, self.global_step, last_checkpoint, best_checkpoint)

    def _save_phase63_checkpoint(self) -> Path:
        metric = self.latest_validation_loss
        improved = self._latest_validation_improved
        if metric is not None and improved:
            self.best_metric = metric
            self.early_stopping.best_metric = metric
            self.early_stopping.bad_epochs = 0
        record = save_phase63_checkpoint(
            output_dir=self.phase63.paths.checkpoint_output_dir,
            epoch=self.epoch,
            global_step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            model_config=asdict(self.config.model),
            tokenizer_path=self.config.data.tokenizer,
            training_loss=self.latest_training_loss,
            validation_loss=self.latest_validation_loss,
            best_metric=self.best_metric,
            save_best=improved,
            extra_state={
                "phase": "6.3",
                "source_checkpoint": str(self.source_checkpoint),
                "corpus_index": str(self.phase63.paths.corpus_index),
                "training_config": str(self.phase63.config_path),
            },
        )
        self._latest_validation_improved = False
        self.monitor.log_checkpoint(record)
        return Path(record["last_checkpoint"])

    def _phase63_should_validate(self) -> bool:
        interval = self.phase63.training.validation_interval_steps
        return interval > 0 and self.global_step % interval == 0

    def _phase63_should_checkpoint(self) -> bool:
        interval = self.phase63.training.checkpoint_interval_steps
        return interval > 0 and self.global_step % interval == 0

    def _safety_check(self, metrics: dict[str, Any]) -> None:
        loss = float(metrics["loss"])
        grad_norm = float(metrics.get("gradient_norm") or 0.0)
        if not math.isfinite(loss):
            raise Phase63Error("NaN or infinite loss detected; aborting Phase 6.3.")
        if not math.isfinite(grad_norm):
            raise Phase63Error("NaN or infinite gradient norm detected; aborting Phase 6.3.")
        if grad_norm > self.phase63.training.exploding_gradient_threshold:
            raise Phase63Error(
                f"Exploding gradients detected: {grad_norm:.6f} exceeds "
                f"{self.phase63.training.exploding_gradient_threshold:.6f}."
            )


def run_phase63_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Phase 6.3."""

    parser = argparse.ArgumentParser(description="Run Phase 6.3 continued pretraining.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase6_3.yaml"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_phase63_config(args.config)
        if args.max_steps is not None or args.device is not None or args.skip_benchmark:
            config = _override_cli(config, args)
        result = run_phase63(config)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Phase 6.3 continued pretraining complete")
    print(f"Status: {result.status}")
    print(f"Source checkpoint: {result.source_checkpoint}")
    print(f"Global step: {result.global_step}")
    print(f"Last checkpoint: {result.last_checkpoint}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Summary: {result.summary_path}")
    print(f"Benchmark JSON: {result.benchmark_json}")
    print(f"Benchmark report: {result.benchmark_markdown}")
    return 0


def _phase6_config(config: Phase63Config, source_checkpoint: Path) -> Phase6Config:
    phase6 = load_phase6_config(
        config.paths.phase6_training_config,
        model_config=config.paths.model_config,
        optimizer_config=config.paths.optimizer_config,
        generation_config=config.paths.generation_config,
    )
    data = replace(
        phase6.data,
        shard_pattern=str(config.paths.corpus_index.parent / "corpus_v2_*.bin"),
        shard_index=config.paths.corpus_index,
        training_manifest=config.paths.corpus_report_manifest,
        tokenizer=_resolve(
            config.project_root,
            _read_json(config.paths.corpus_report_manifest).get("tokenizer"),
        ),
        batch_size=config.training.batch_size or phase6.data.batch_size,
    )
    training = replace(
        phase6.training,
        device=config.training.device,
        max_steps=max(phase6.training.max_steps, config.training.max_steps),
        resume=True,
        resume_from=source_checkpoint,
        validate_every_steps=config.training.validation_interval_steps,
        save_every_steps=config.training.checkpoint_interval_steps,
        max_grad_norm=(
            config.training.max_grad_norm
            if config.training.max_grad_norm is not None
            else phase6.training.max_grad_norm
        ),
    )
    optimizer = phase6.optimizer
    if config.training.learning_rate is not None:
        optimizer = replace(optimizer, learning_rate=config.training.learning_rate)
    if config.training.weight_decay is not None:
        optimizer = replace(optimizer, weight_decay=config.training.weight_decay)
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
        checkpoint=checkpoint,
        outputs=outputs,
    )


def _override_cli(config: Phase63Config, args: argparse.Namespace) -> Phase63Config:
    training = config.training
    benchmark = config.benchmark
    if args.max_steps is not None:
        training = replace(training, max_steps=args.max_steps)
    if args.device is not None:
        training = replace(training, device=args.device)
    if args.skip_benchmark:
        benchmark = replace(benchmark, enabled=False)
    return replace(config, training=training, benchmark=benchmark)


def _validate_phase63_config(config: Phase63Config) -> None:
    if config.training.max_steps <= 0:
        raise Phase63Error("phase6_3.training.max_steps must be positive.")
    if config.training.max_epochs <= 0:
        raise Phase63Error("phase6_3.training.max_epochs must be positive.")
    if config.training.validation_interval_steps <= 0:
        raise Phase63Error("validation_interval_steps must be positive.")
    if config.training.checkpoint_interval_steps <= 0:
        raise Phase63Error("checkpoint_interval_steps must be positive.")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase63Error(f"Expected JSON object: {path}")
    return payload


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Phase63Error(f"{name} must be a mapping.")
    return value


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise Phase63Error("Configured path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


__all__ = [
    "Phase63Config",
    "Phase63Error",
    "Phase63Result",
    "Phase63Trainer",
    "check_phase63_readiness",
    "load_phase63_config",
    "run_phase63",
    "run_phase63_cli",
]
