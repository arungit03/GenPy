"""Phase 6.1 corpus expansion and continued-pretraining orchestration."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from genpy_llm.corpus_merger import build_pretraining_corpus, load_pretraining_config
from genpy_llm.logging_utils import setup_structured_logging
from genpy_llm.pretraining import Phase6Trainer, load_phase6_config

UTC = timezone.utc

LOGGER = logging.getLogger("genpy_llm.continued_pretraining")


class ContinuedPretrainingError(RuntimeError):
    """Raised when Phase 6.1 cannot continue safely."""


@dataclass(frozen=True)
class CorpusTargetConfig:
    """Phase 6.1 corpus size and balance gates."""

    minimum_tokens: int
    maximum_tokens: int
    min_python_code_fraction: float
    max_python_code_fraction: float
    min_technical_text_fraction: float
    max_technical_text_fraction: float
    allow_under_target: bool


@dataclass(frozen=True)
class Phase61StageConfig:
    """Phase 6.1 stage switches."""

    build_final_corpus: bool
    train: bool
    evaluate: bool


@dataclass(frozen=True)
class Phase61PathConfig:
    """Phase 6.1 configuration and artifact paths."""

    pretraining_corpus_config: Path
    training_config: Path
    model_config: Path
    optimizer_config: Path
    generation_config: Path
    report_directory: Path
    log_file: Path


@dataclass(frozen=True)
class Phase61TrainingOverrides:
    """Optional Phase 6 trainer overrides."""

    max_steps: int | None
    device: str | None
    resume_from: Path | None


@dataclass(frozen=True)
class Phase61EvaluationConfig:
    """Configured benchmark commands for Phase 6.1."""

    commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Phase61Config:
    """Complete Phase 6.1 configuration."""

    project_root: Path
    targets: CorpusTargetConfig
    stages: Phase61StageConfig
    paths: Phase61PathConfig
    training: Phase61TrainingOverrides
    evaluation: Phase61EvaluationConfig
    log_level: str


@dataclass(frozen=True)
class CorpusAssessment:
    """Readiness summary for the packed Phase 6.1 corpus."""

    manifest_path: Path
    total_tokens: int
    total_files: int
    content_type_tokens: Mapping[str, int]
    content_type_files: Mapping[str, int]
    token_target_met: bool
    token_target_exceeded: bool
    balance_target_met: bool
    python_code_fraction: float
    technical_text_fraction: float

    @property
    def ready_for_training(self) -> bool:
        """Return whether the corpus meets size and balance gates."""

        return self.token_target_met and not self.token_target_exceeded and self.balance_target_met


def load_phase61_config(path: Path | str = "configs/phase6_1.yaml") -> Phase61Config:
    """Load Phase 6.1 YAML configuration."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Phase 6.1 config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ContinuedPretrainingError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ContinuedPretrainingError("Phase 6.1 config must be a YAML mapping.")
    root = _resolve(config_path.parent, raw.get("project_root", ".."))
    section = _mapping(raw.get("phase6_1", {}), "phase6_1")
    corpus = _mapping(section.get("corpus", {}), "phase6_1.corpus")
    stages = _mapping(section.get("stages", {}), "phase6_1.stages")
    paths = _mapping(section.get("paths", {}), "phase6_1.paths")
    training = _mapping(section.get("training", {}), "phase6_1.training")
    evaluation = _mapping(section.get("evaluation", {}), "phase6_1.evaluation")
    logging_section = _mapping(raw.get("logging", {}), "logging")
    config = Phase61Config(
        project_root=root,
        targets=CorpusTargetConfig(
            minimum_tokens=int(corpus.get("minimum_tokens", 200_000_000)),
            maximum_tokens=int(corpus.get("maximum_tokens", 500_000_000)),
            min_python_code_fraction=float(corpus.get("min_python_code_fraction", 0.45)),
            max_python_code_fraction=float(corpus.get("max_python_code_fraction", 0.75)),
            min_technical_text_fraction=float(corpus.get("min_technical_text_fraction", 0.20)),
            max_technical_text_fraction=float(corpus.get("max_technical_text_fraction", 0.55)),
            allow_under_target=bool(corpus.get("allow_under_target", False)),
        ),
        stages=Phase61StageConfig(
            build_final_corpus=bool(stages.get("build_final_corpus", True)),
            train=bool(stages.get("train", True)),
            evaluate=bool(stages.get("evaluate", True)),
        ),
        paths=Phase61PathConfig(
            pretraining_corpus_config=_resolve(
                root,
                paths.get("pretraining_corpus_config", "configs/pretraining.yaml"),
            ),
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
            report_directory=_resolve(
                root,
                paths.get("report_directory", "reports/phase6_1"),
            ),
            log_file=_resolve(root, paths.get("log_file", "logs/phase6_1.jsonl")),
        ),
        training=Phase61TrainingOverrides(
            max_steps=(
                int(training["max_steps"]) if training.get("max_steps") is not None else None
            ),
            device=str(training["device"]) if training.get("device") is not None else None,
            resume_from=(
                _resolve(root, training["resume_from"])
                if training.get("resume_from") is not None
                else None
            ),
        ),
        evaluation=Phase61EvaluationConfig(commands=_commands(evaluation.get("commands", ()))),
        log_level=str(logging_section.get("level", "INFO")).upper(),
    )
    _validate_config(config)
    return config


def assess_pretraining_corpus(
    manifest_path: Path | str,
    targets: CorpusTargetConfig,
) -> CorpusAssessment:
    """Summarize token volume and code/text balance from a packed corpus manifest."""

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Packed corpus manifest not found: {path}")
    token_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    for record in _iter_jsonl(path):
        content_type = str(record.get("content_type") or _content_type_from_path(record))
        token_count = int(record.get("token_count") or 0)
        token_counts[content_type] += token_count
        file_counts[content_type] += 1
    total_tokens = sum(token_counts.values())
    total_files = sum(file_counts.values())
    python_fraction = _fraction(token_counts["python_code"], total_tokens)
    text_fraction = _fraction(token_counts["technical_text"], total_tokens)
    balance_ok = (
        targets.min_python_code_fraction <= python_fraction <= targets.max_python_code_fraction
        and targets.min_technical_text_fraction
        <= text_fraction
        <= targets.max_technical_text_fraction
    )
    return CorpusAssessment(
        manifest_path=path,
        total_tokens=total_tokens,
        total_files=total_files,
        content_type_tokens=dict(sorted(token_counts.items())),
        content_type_files=dict(sorted(file_counts.items())),
        token_target_met=total_tokens >= targets.minimum_tokens,
        token_target_exceeded=total_tokens > targets.maximum_tokens,
        balance_target_met=balance_ok,
        python_code_fraction=python_fraction,
        technical_text_fraction=text_fraction,
    )


def write_phase61_report(
    assessment: CorpusAssessment,
    config: Phase61Config,
) -> tuple[Path, Path]:
    """Write JSON and Markdown Phase 6.1 corpus-readiness reports."""

    config.paths.report_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": _timestamp(),
        "targets": asdict(config.targets),
        "assessment": _assessment_payload(assessment),
        "ready_for_training": assessment.ready_for_training,
    }
    json_path = config.paths.report_directory / "corpus_readiness.json"
    markdown_path = config.paths.report_directory / "corpus_readiness.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return json_path, markdown_path


def run_phase61(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Phase 6.1."""

    parser = argparse.ArgumentParser(
        description="Expand the GenPy corpus and resume continued Phase 6 pretraining."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase6_1.yaml"))
    parser.add_argument("--force-corpus", action="store_true")
    parser.add_argument("--skip-corpus-build", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_phase61_config(args.config)
        setup_structured_logging(config.paths.log_file, config.log_level)
        if config.stages.build_final_corpus and not args.skip_corpus_build:
            corpus_config = load_pretraining_config(config.paths.pretraining_corpus_config)
            build_pretraining_corpus(corpus_config, force=args.force_corpus)
        corpus_config = load_pretraining_config(config.paths.pretraining_corpus_config)
        assessment = assess_pretraining_corpus(
            corpus_config.paths.merged_manifest,
            config.targets,
        )
        report_json, report_markdown = write_phase61_report(assessment, config)
        if not assessment.ready_for_training and not config.targets.allow_under_target:
            raise ContinuedPretrainingError(
                "Phase 6.1 corpus gates failed. See "
                f"{report_markdown} before continuing training."
            )
        training_result = None
        if config.stages.train and not args.skip_training:
            phase6 = load_phase6_config(
                config.paths.training_config,
                model_config=config.paths.model_config,
                optimizer_config=config.paths.optimizer_config,
                generation_config=config.paths.generation_config,
            )
            phase6 = _apply_training_overrides(phase6, config.training)
            training_result = Phase6Trainer(phase6).train()
        benchmark_results: list[dict[str, Any]] = []
        if config.stages.evaluate and not args.skip_evaluation:
            benchmark_results = _run_benchmarks(config)
        _write_run_summary(config, assessment, training_result, benchmark_results)
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Phase 6.1 failed")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Phase 6.1 complete")
    print(f"Corpus readiness JSON: {report_json}")
    print(f"Corpus readiness report: {report_markdown}")
    if training_result is not None:
        print(f"Continued checkpoint: {training_result.last_checkpoint}")
    if benchmark_results:
        passed = sum(item["returncode"] == 0 for item in benchmark_results)
        print(f"Benchmark commands passed: {passed}/{len(benchmark_results)}")
    return 0


def _apply_training_overrides(phase6: Any, overrides: Phase61TrainingOverrides) -> Any:
    training = phase6.training
    if overrides.max_steps is not None:
        training = replace(training, max_steps=overrides.max_steps)
    if overrides.device is not None:
        training = replace(training, device=overrides.device)
    if overrides.resume_from is not None:
        training = replace(training, resume=True, resume_from=overrides.resume_from)
    return replace(phase6, training=training)


def _run_benchmarks(config: Phase61Config) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for command in config.evaluation.commands:
        resolved_command = [sys.executable if part == "{python}" else part for part in command]
        started = _timestamp()
        completed = subprocess.run(  # noqa: S603
            resolved_command,
            cwd=config.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "command": resolved_command,
                "started_at": started,
                "completed_at": _timestamp(),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    return results


def _write_run_summary(
    config: Phase61Config,
    assessment: CorpusAssessment,
    training_result: Any,
    benchmark_results: Sequence[Mapping[str, Any]],
) -> Path:
    path = config.paths.report_directory / "run_summary.json"
    payload = {
        "created_at": _timestamp(),
        "corpus": _assessment_payload(assessment),
        "training": None
        if training_result is None
        else {
            "global_step": training_result.global_step,
            "best_metric": training_result.best_metric,
            "last_checkpoint": str(training_result.last_checkpoint),
            "best_checkpoint": str(training_result.best_checkpoint),
            "metrics_path": str(training_result.metrics_path),
        },
        "benchmarks": list(benchmark_results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _assessment_payload(assessment: CorpusAssessment) -> dict[str, Any]:
    return {
        "manifest_path": str(assessment.manifest_path),
        "total_tokens": assessment.total_tokens,
        "total_files": assessment.total_files,
        "content_type_tokens": dict(assessment.content_type_tokens),
        "content_type_files": dict(assessment.content_type_files),
        "token_target_met": assessment.token_target_met,
        "token_target_exceeded": assessment.token_target_exceeded,
        "balance_target_met": assessment.balance_target_met,
        "python_code_fraction": assessment.python_code_fraction,
        "technical_text_fraction": assessment.technical_text_fraction,
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    assessment = _mapping(payload["assessment"], "assessment")
    targets = _mapping(payload["targets"], "targets")
    lines = [
        "# GenPy Phase 6.1 Corpus Readiness",
        "",
        f"- Total tokens: {assessment['total_tokens']:,}",
        f"- Target range: {targets['minimum_tokens']:,}-{targets['maximum_tokens']:,}",
        f"- Python code tokens: {assessment['python_code_fraction']:.2%}",
        f"- Technical text tokens: {assessment['technical_text_fraction']:.2%}",
        f"- Token target met: {assessment['token_target_met']}",
        f"- Balance target met: {assessment['balance_target_met']}",
        f"- Ready for continued training: {payload['ready_for_training']}",
        "",
        "## Token Mix",
        "",
    ]
    for content_type, tokens in dict(assessment["content_type_tokens"]).items():
        files = dict(assessment["content_type_files"]).get(content_type, 0)
        lines.append(f"- {content_type}: {tokens:,} tokens across {files:,} files")
    return "\n".join(lines).rstrip() + "\n"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContinuedPretrainingError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ContinuedPretrainingError(
                    f"JSONL record {path}:{line_number} is not an object."
                )
            yield value


def _content_type_from_path(record: Mapping[str, Any]) -> str:
    path = str(record.get("source_path") or record.get("stored_path") or "")
    return "python_code" if path.casefold().endswith(".py") else "technical_text"


def _commands(value: object) -> tuple[tuple[str, ...], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ContinuedPretrainingError("phase6_1.evaluation.commands must be a list.")
    commands: list[tuple[str, ...]] = []
    for index, command in enumerate(value, start=1):
        if not isinstance(command, (list, tuple)) or not command:
            raise ContinuedPretrainingError(f"evaluation command {index} must be a non-empty list.")
        if not all(isinstance(part, str) and part for part in command):
            raise ContinuedPretrainingError(f"evaluation command {index} must contain strings.")
        commands.append(tuple(command))
    return tuple(commands)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuedPretrainingError(f"{name} must be a mapping.")
    return value


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ContinuedPretrainingError("Configured path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _fraction(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _validate_config(config: Phase61Config) -> None:
    if config.targets.minimum_tokens <= 0:
        raise ContinuedPretrainingError("minimum_tokens must be positive.")
    if config.targets.maximum_tokens < config.targets.minimum_tokens:
        raise ContinuedPretrainingError("maximum_tokens must be >= minimum_tokens.")
    for name in (
        "min_python_code_fraction",
        "max_python_code_fraction",
        "min_technical_text_fraction",
        "max_technical_text_fraction",
    ):
        value = getattr(config.targets, name)
        if not 0 <= value <= 1:
            raise ContinuedPretrainingError(f"{name} must be between 0 and 1.")


__all__ = [
    "ContinuedPretrainingError",
    "CorpusAssessment",
    "CorpusTargetConfig",
    "Phase61Config",
    "assess_pretraining_corpus",
    "load_phase61_config",
    "run_phase61",
    "write_phase61_report",
]
