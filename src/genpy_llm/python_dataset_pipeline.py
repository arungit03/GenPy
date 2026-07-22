"""Auditable Python-source to instruction-dataset generation pipeline."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import fnmatch
import hashlib
import io
import json
import logging
import math
import os
import re
import sqlite3
import string
import sys
import textwrap
import tokenize
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, TypeVar

import yaml

from genpy_llm.code_filtering import normalize_python_source

LOGGER = logging.getLogger("genpy_llm.python_dataset_pipeline")
PIPELINE_VERSION = 1
PIPELINE_IMPLEMENTATION_VERSION = 2
STAGES = ("collect", "clean", "generate", "deduplicate", "validate", "split")
INSTRUCTION_CATEGORIES = (
    "code_generation",
    "explanation",
    "bug_fixing",
    "refactoring",
    "documentation",
    "unit_testing",
    "optimization",
    "complexity_analysis",
    "type_hints",
    "code_completion",
    "api_usage",
)
PYTHON_OUTPUT_CATEGORIES = frozenset(
    {
        "code_generation",
        "bug_fixing",
        "refactoring",
        "unit_testing",
        "optimization",
        "type_hints",
        "code_completion",
        "api_usage",
    }
)
DEFAULT_INSTRUCTION_TEMPLATES = {
    "code_generation": "{base_instruction}",
    "explanation": "Explain the behavior of the {kind} `{qualified_name}`.",
    "bug_fixing": "Fix the bug in the provided implementation of `{qualified_name}`.",
    "refactoring": (
        "Refactor `{qualified_name}` into clear, modern Python without changing behavior."
    ),
    "documentation": "Write concise API documentation for `{qualified_name}`.",
    "unit_testing": "Implement the existing unit test `{qualified_name}`.",
    "optimization": (
        "Optimize `{qualified_name}` while preserving its observable behavior."
    ),
    "complexity_analysis": (
        "Analyze the time and auxiliary-space complexity of `{qualified_name}`."
    ),
    "type_hints": "Add the demonstrated type hints to `{qualified_name}`.",
    "code_completion": "Complete the missing body of `{qualified_name}`.",
    "api_usage": "Demonstrate how `{qualified_name}` uses the `{api}` API.",
}
TEMPLATE_FIELDS = frozenset(
    {"api", "base_instruction", "description", "kind", "name", "qualified_name", "signature"}
)
_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


class DatasetPipelineError(RuntimeError):
    """Raised when dataset generation cannot continue safely."""


class JsonlValidationError(DatasetPipelineError):
    """Raised when a JSONL file is malformed."""


@dataclass(frozen=True)
class ApprovedSource:
    """One explicitly approved filesystem source root."""

    source_id: str
    path: Path
    repository: str
    approval: str
    license: str | None
    include: tuple[str, ...] = ("**/*.py",)
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelinePaths:
    """Intermediate and final dataset paths."""

    workspace: Path
    collected: Path
    cleaned: Path
    generated: Path
    deduplicated: Path
    validated: Path
    rejected: Path
    manifests: Path
    train: Path
    validation: Path
    test: Path
    statistics: Path
    log_file: Path


@dataclass(frozen=True)
class CleaningSettings:
    """Python source cleaning constraints."""

    minimum_file_bytes: int = 80
    maximum_file_bytes: int = 250_000
    require_python_definitions: bool = True


@dataclass(frozen=True)
class PairGenerationSettings:
    """Instruction-pair extraction constraints."""

    require_docstring: bool = True
    include_functions: bool = True
    include_async_functions: bool = True
    include_classes: bool = True
    include_methods: bool = True
    include_private: bool = True
    minimum_instruction_characters: int = 8
    maximum_instruction_characters: int = 8_000
    maximum_output_bytes: int = 100_000
    enabled_categories: tuple[str, ...] = ("code_generation",)
    templates: tuple[tuple[str, str], ...] = tuple(DEFAULT_INSTRUCTION_TEMPLATES.items())
    maximum_examples_per_file: int = 0
    selection_seed: int = 42


@dataclass(frozen=True)
class ValidationSettings:
    """Final instruction-record validation constraints."""

    minimum_instruction_characters: int = 8
    maximum_instruction_characters: int = 8_000
    minimum_output_characters: int = 1
    maximum_output_bytes: int = 100_000
    require_python_syntax: bool = True
    require_provenance: bool = True
    fail_on_invalid: bool = True


@dataclass(frozen=True)
class SplitSettings:
    """Deterministic source-group split settings."""

    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    group_by: str = "source_file"


@dataclass(frozen=True)
class PerformanceSettings:
    """Bounded-resource settings for large dataset builds."""

    workers: int = 1
    max_pending_tasks_per_worker: int = 4
    sqlite_batch_size: int = 1_000
    verify_output_hashes_on_resume: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Validated configuration for the complete dataset pipeline."""

    config_path: Path
    project_root: Path
    paths: PipelinePaths
    sources: tuple[ApprovedSource, ...]
    cleaning: CleaningSettings
    pair_generation: PairGenerationSettings
    validation: ValidationSettings
    split: SplitSettings
    performance: PerformanceSettings
    progress: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True)
class StageStatistics:
    """Serializable counters for one pipeline stage."""

    stage: str
    read_records: int
    written_records: int
    rejected_records: int = 0
    duplicate_records: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    split_category_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    output_files: tuple[str, ...] = ()
    output_hashes: dict[str, str] = field(default_factory=dict)
    output_metadata: dict[str, dict[str, int]] = field(default_factory=dict)
    input_fingerprint: str = ""
    resumed: bool = False
    completed_at: str = ""


@dataclass(frozen=True)
class BuildResult:
    """Final paths and per-stage statistics from a pipeline run."""

    train_path: Path
    validation_path: Path
    test_path: Path
    statistics_path: Path
    stages: tuple[StageStatistics, ...]


def load_pipeline_config(path: Path | str = "configs/dataset_pipeline.yaml") -> PipelineConfig:
    """Load and validate a dataset-pipeline YAML file."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset pipeline config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise DatasetPipelineError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetPipelineError("Dataset pipeline config must be a YAML mapping.")
    if int(raw.get("version", PIPELINE_VERSION)) != PIPELINE_VERSION:
        raise DatasetPipelineError("Unsupported dataset pipeline config version.")

    root_value = _required_string(raw.get("project_root", "."), "project_root")
    project_root = (config_path.parent / root_value).resolve()
    path_values = _required_mapping(raw.get("paths"), "paths")
    workspace = _resolve(project_root, path_values.get("workspace", "data/dataset_pipeline"))
    final_dir = _resolve(project_root, path_values.get("final_directory", "data/fine_tuning"))
    paths = PipelinePaths(
        workspace=workspace,
        collected=workspace / "01_collected.jsonl",
        cleaned=workspace / "02_cleaned.jsonl",
        generated=workspace / "03_instruction_pairs.jsonl",
        deduplicated=workspace / "04_deduplicated.jsonl",
        validated=workspace / "05_validated.jsonl",
        rejected=workspace / "validation_rejections.jsonl",
        manifests=workspace / "manifests",
        train=final_dir / "train.jsonl",
        validation=final_dir / "validation.jsonl",
        test=final_dir / "test.jsonl",
        statistics=final_dir / "dataset_statistics.json",
        log_file=_resolve(project_root, path_values.get("log_file", "logs/dataset_pipeline.log")),
    )

    collection = _required_mapping(raw.get("collection"), "collection")
    raw_sources = collection.get("approved_sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DatasetPipelineError("collection.approved_sources must be a non-empty list.")
    sources = tuple(_load_source(project_root, item) for item in raw_sources)

    cleaning_raw = _mapping_or_empty(raw.get("cleaning"), "cleaning")
    pair_raw = _mapping_or_empty(raw.get("instruction_generation"), "instruction_generation")
    validation_raw = _mapping_or_empty(raw.get("validation"), "validation")
    split_raw = _mapping_or_empty(raw.get("split"), "split")
    performance_raw = _mapping_or_empty(raw.get("performance"), "performance")
    logging_raw = _mapping_or_empty(raw.get("logging"), "logging")
    config = PipelineConfig(
        config_path=config_path,
        project_root=project_root,
        paths=paths,
        sources=sources,
        cleaning=CleaningSettings(
            minimum_file_bytes=int(cleaning_raw.get("minimum_file_bytes", 80)),
            maximum_file_bytes=int(cleaning_raw.get("maximum_file_bytes", 250_000)),
            require_python_definitions=bool(
                cleaning_raw.get("require_python_definitions", True)
            ),
        ),
        pair_generation=PairGenerationSettings(
            require_docstring=bool(pair_raw.get("require_docstring", True)),
            include_functions=bool(pair_raw.get("include_functions", True)),
            include_async_functions=bool(pair_raw.get("include_async_functions", True)),
            include_classes=bool(pair_raw.get("include_classes", True)),
            include_methods=bool(pair_raw.get("include_methods", True)),
            include_private=bool(pair_raw.get("include_private", True)),
            minimum_instruction_characters=int(
                pair_raw.get("minimum_instruction_characters", 8)
            ),
            maximum_instruction_characters=int(
                pair_raw.get("maximum_instruction_characters", 8_000)
            ),
            maximum_output_bytes=int(pair_raw.get("maximum_output_bytes", 100_000)),
            enabled_categories=_string_tuple(
                pair_raw.get("enabled_categories", ["code_generation"]),
                "instruction_generation.enabled_categories",
            ),
            templates=_load_instruction_templates(pair_raw.get("templates", {})),
            maximum_examples_per_file=int(
                pair_raw.get("maximum_examples_per_file", 0)
            ),
            selection_seed=int(pair_raw.get("selection_seed", 42)),
        ),
        validation=ValidationSettings(
            minimum_instruction_characters=int(
                validation_raw.get("minimum_instruction_characters", 8)
            ),
            maximum_instruction_characters=int(
                validation_raw.get("maximum_instruction_characters", 8_000)
            ),
            minimum_output_characters=int(
                validation_raw.get("minimum_output_characters", 1)
            ),
            maximum_output_bytes=int(validation_raw.get("maximum_output_bytes", 100_000)),
            require_python_syntax=bool(validation_raw.get("require_python_syntax", True)),
            require_provenance=bool(validation_raw.get("require_provenance", True)),
            fail_on_invalid=bool(validation_raw.get("fail_on_invalid", True)),
        ),
        split=SplitSettings(
            train_ratio=float(split_raw.get("train_ratio", 0.8)),
            validation_ratio=float(split_raw.get("validation_ratio", 0.1)),
            test_ratio=float(split_raw.get("test_ratio", 0.1)),
            seed=int(split_raw.get("seed", 42)),
            group_by=str(split_raw.get("group_by", "source_file")),
        ),
        performance=PerformanceSettings(
            workers=int(performance_raw.get("workers", 1)),
            max_pending_tasks_per_worker=int(
                performance_raw.get("max_pending_tasks_per_worker", 4)
            ),
            sqlite_batch_size=int(performance_raw.get("sqlite_batch_size", 1_000)),
            verify_output_hashes_on_resume=bool(
                performance_raw.get("verify_output_hashes_on_resume", False)
            ),
        ),
        progress=bool(raw.get("progress", True)),
        log_level=str(logging_raw.get("level", "INFO")).upper(),
    )
    _validate_config(config)
    return config


def configure_pipeline_logging(config: PipelineConfig, *, level: str | None = None) -> None:
    """Configure console and file logging for pipeline CLIs."""

    selected = (level or config.log_level).upper()
    numeric_level = getattr(logging, selected, None)
    if not isinstance(numeric_level, int):
        raise DatasetPipelineError(f"Unknown logging level: {selected}")
    config.paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(config.paths.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def collect_python_data(config: PipelineConfig, *, resume: bool = True) -> StageStatistics:
    """Collect Python files from only explicitly approved source roots."""

    discovered = _discover_approved_files(config.sources)
    fingerprint = _fingerprint(
        "collect",
        PIPELINE_IMPLEMENTATION_VERSION,
        [_source_as_mapping(source) for source in config.sources],
        [
            (
                source.source_id,
                path.relative_to(source.path.resolve()).as_posix(),
                *_file_metadata(path).values(),
            )
            for source, path in discovered
        ],
    )
    outputs = (config.paths.collected,)
    resumed = _resume_statistics(config, "collect", fingerprint, outputs, resume)
    if resumed is not None:
        return resumed

    LOGGER.info("Collecting %s approved Python files", len(discovered))
    progress = ProgressBar("collect", len(discovered), enabled=config.progress)
    written = 0
    with atomic_jsonl_writer(config.paths.collected) as output:
        for source, path in discovered:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise DatasetPipelineError(
                    f"Could not decode approved source {path}: {exc}"
                ) from exc
            relative_path = path.resolve().relative_to(source.path.resolve()).as_posix()
            normalized = normalize_python_source(content)
            content_digest = _text_hash(normalized)
            record_id = _text_hash(f"{source.source_id}\0{relative_path}\0{content_digest}")
            _write_jsonl_record(
                output,
                {
                    "schema_version": PIPELINE_VERSION,
                    "record_id": record_id,
                    "source_id": source.source_id,
                    "repository": source.repository,
                    "source_path": relative_path,
                    "approval": source.approval,
                    "license": source.license,
                    "content_hash": content_digest,
                    "byte_count": len(normalized.encode("utf-8")),
                    "content": normalized,
                },
            )
            written += 1
            progress.update(written)
    progress.close()
    return _finish_stage(config, "collect", written, written, fingerprint, outputs)


def clean_python_dataset(config: PipelineConfig, *, resume: bool = True) -> StageStatistics:
    """Normalize collected source and reject invalid or unsuitable Python files."""

    _require_file(config.paths.collected, "Run collect_python_data.py first.")
    fingerprint = _fingerprint(
        "clean",
        PIPELINE_IMPLEMENTATION_VERSION,
        asdict(config.cleaning),
        _upstream_output_hash(config, "collect", config.paths.collected),
    )
    outputs = (config.paths.cleaned,)
    resumed = _resume_statistics(config, "clean", fingerprint, outputs, resume)
    if resumed is not None:
        return resumed

    total = _upstream_record_count(config, "collect", config.paths.collected)
    progress = ProgressBar("clean", total, enabled=config.progress)
    reasons: Counter[str] = Counter()
    read = written = 0
    with atomic_jsonl_writer(config.paths.cleaned) as output:
        payloads = (
            (record, config.cleaning) for record in iter_jsonl(config.paths.collected)
        )
        for reason, cleaned in _bounded_parallel_map(
            _clean_source_worker,
            payloads,
            config.performance,
        ):
            read += 1
            if reason is not None:
                reasons[reason] += 1
            else:
                assert cleaned is not None
                _write_jsonl_record(output, cleaned)
                written += 1
            progress.update(read)
    progress.close()
    return _finish_stage(
        config,
        "clean",
        read,
        written,
        fingerprint,
        outputs,
        rejected=read - written,
        reasons=reasons,
    )


def generate_instruction_pairs(
    config: PipelineConfig,
    *,
    resume: bool = True,
) -> StageStatistics:
    """Extract grounded instruction/code pairs from real source docstrings and AST spans."""

    _require_file(config.paths.cleaned, "Run clean_python_dataset.py first.")
    fingerprint = _fingerprint(
        "generate",
        PIPELINE_IMPLEMENTATION_VERSION,
        asdict(config.pair_generation),
        _upstream_output_hash(config, "clean", config.paths.cleaned),
    )
    outputs = (config.paths.generated,)
    resumed = _resume_statistics(config, "generate", fingerprint, outputs, resume)
    if resumed is not None:
        return resumed

    total = _upstream_record_count(config, "clean", config.paths.cleaned)
    progress = ProgressBar("generate", total, enabled=config.progress)
    reasons: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    read = written = 0
    with atomic_jsonl_writer(config.paths.generated) as output:
        payloads = (
            (record, config.pair_generation) for record in iter_jsonl(config.paths.cleaned)
        )
        for pairs, pair_reasons in _bounded_parallel_map(
            _generate_pairs_worker,
            payloads,
            config.performance,
        ):
            read += 1
            reasons.update(pair_reasons)
            for pair in pairs:
                _write_jsonl_record(output, pair)
                written += 1
                categories[str(pair.get("category", "code_generation"))] += 1
            progress.update(read)
    progress.close()
    return _finish_stage(
        config,
        "generate",
        read,
        written,
        fingerprint,
        outputs,
        rejected=sum(reasons.values()),
        reasons=reasons,
        categories=categories,
    )


def deduplicate_dataset(config: PipelineConfig, *, resume: bool = True) -> StageStatistics:
    """Remove exact semantic duplicates from generated instruction pairs."""

    _require_file(config.paths.generated, "Run generate_instruction_pairs.py first.")
    fingerprint = _fingerprint(
        "deduplicate",
        PIPELINE_IMPLEMENTATION_VERSION,
        _upstream_output_hash(config, "generate", config.paths.generated),
    )
    outputs = (config.paths.deduplicated,)
    resumed = _resume_statistics(config, "deduplicate", fingerprint, outputs, resume)
    if resumed is not None:
        return resumed

    total = _upstream_record_count(config, "generate", config.paths.generated)
    progress = ProgressBar("deduplicate", total, enabled=config.progress)
    read = written = duplicates = 0
    index_path = config.paths.workspace / "indexes" / "deduplication.sqlite3"
    with (
        _temporary_sqlite(index_path) as database,
        atomic_jsonl_writer(config.paths.deduplicated) as output,
    ):
        database.execute(
            "CREATE TABLE seen_pairs (pair_hash TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        for record in iter_jsonl(config.paths.generated):
            read += 1
            key = instruction_pair_hash(record)
            if not _sqlite_insert_unique(database, "seen_pairs", "pair_hash", key):
                duplicates += 1
            else:
                record["deduplication_hash"] = key
                _write_jsonl_record(output, record)
                written += 1
            if read % config.performance.sqlite_batch_size == 0:
                database.commit()
            progress.update(read)
    progress.close()
    return _finish_stage(
        config,
        "deduplicate",
        read,
        written,
        fingerprint,
        outputs,
        duplicate=duplicates,
    )


def validate_dataset(config: PipelineConfig, *, resume: bool = True) -> StageStatistics:
    """Strictly validate JSONL schema, Python syntax, provenance, and uniqueness."""

    _require_file(config.paths.deduplicated, "Run deduplicate_dataset.py first.")
    fingerprint = _fingerprint(
        "validate",
        PIPELINE_IMPLEMENTATION_VERSION,
        asdict(config.validation),
        _upstream_output_hash(config, "deduplicate", config.paths.deduplicated),
    )
    outputs = (config.paths.validated, config.paths.rejected)
    resumed = _resume_statistics(config, "validate", fingerprint, outputs, resume)
    if resumed is not None:
        return resumed

    total = _upstream_record_count(config, "deduplicate", config.paths.deduplicated)
    progress = ProgressBar("validate", total, enabled=config.progress)
    reasons: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    read = written = 0
    index_path = config.paths.workspace / "indexes" / "validation.sqlite3"
    with (
        _temporary_sqlite(index_path) as database,
        atomic_jsonl_writer(config.paths.validated) as valid_output,
        atomic_jsonl_writer(config.paths.rejected) as rejected_output,
    ):
        database.execute(
            "CREATE TABLE seen_ids (record_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        database.execute(
            "CREATE TABLE seen_pairs (pair_hash TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        for line_number, record in enumerate(iter_jsonl(config.paths.deduplicated), start=1):
            read += 1
            errors = validate_instruction_record(record, config.validation)
            record_id = record.get("record_id")
            if isinstance(record_id, str) and record_id:
                if not _sqlite_insert_unique(database, "seen_ids", "record_id", record_id):
                    errors.append("duplicate_record_id")
            pair_hash = instruction_pair_hash(record)
            if not _sqlite_insert_unique(database, "seen_pairs", "pair_hash", pair_hash):
                errors.append("duplicate_instruction_pair")
            errors = sorted(set(errors))
            if errors:
                reasons.update(errors)
                _write_jsonl_record(
                    rejected_output,
                    {"line_number": line_number, "errors": errors, "record": record},
                )
            else:
                _write_jsonl_record(valid_output, record)
                written += 1
                categories[str(record.get("category", "code_generation"))] += 1
            if read % config.performance.sqlite_batch_size == 0:
                database.commit()
            progress.update(read)
    progress.close()
    statistics = _finish_stage(
        config,
        "validate",
        read,
        written,
        fingerprint,
        outputs,
        rejected=read - written,
        reasons=reasons,
        categories=categories,
    )
    if reasons and config.validation.fail_on_invalid:
        _manifest_path(config, "validate").unlink(missing_ok=True)
        raise DatasetPipelineError(
            f"Dataset validation rejected {read - written} records; "
            f"see {config.paths.rejected}."
        )
    return statistics


def split_dataset(config: PipelineConfig, *, resume: bool = True) -> StageStatistics:
    """Split records deterministically with a disk-backed external-memory sort."""

    _require_file(config.paths.validated, "Run validate_dataset.py first.")
    fingerprint = _fingerprint(
        "split",
        PIPELINE_IMPLEMENTATION_VERSION,
        asdict(config.split),
        _upstream_output_hash(config, "validate", config.paths.validated),
    )
    outputs = (config.paths.train, config.paths.validation, config.paths.test)
    resumed = _resume_statistics(config, "split", fingerprint, outputs, resume)
    if resumed is not None:
        return resumed

    total = _upstream_record_count(config, "validate", config.paths.validated)
    progress = ProgressBar("split", total * 3, enabled=config.progress)
    split_names = ("train", "validation", "test")
    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    split_categories: dict[str, Counter[str]] = {
        name: Counter() for name in split_names
    }
    index_path = config.paths.workspace / "indexes" / "split.sqlite3"
    processed = 0
    with _temporary_sqlite(index_path) as database:
        database.executescript(
            """
            CREATE TABLE source_groups (
                group_key TEXT PRIMARY KEY,
                sort_key TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE assignments (
                group_key TEXT PRIMARY KEY,
                split_name TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE records (
                split_name TEXT NOT NULL,
                sort_key TEXT NOT NULL,
                record_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )

        for record in iter_jsonl(config.paths.validated):
            group_key = _split_group_key(record, config.split.group_by)
            categories[str(record.get("category", "code_generation"))] += 1
            database.execute(
                "INSERT OR IGNORE INTO source_groups (group_key, sort_key) VALUES (?, ?)",
                (group_key, _text_hash(f"{config.split.seed}\0{group_key}")),
            )
            processed += 1
            if processed % config.performance.sqlite_batch_size == 0:
                database.commit()
            progress.update(processed)
        database.commit()

        group_total = int(
            database.execute("SELECT COUNT(*) FROM source_groups").fetchone()[0]
        )
        allocated = _allocate_split_counts(
            group_total,
            (
                config.split.train_ratio,
                config.split.validation_ratio,
                config.split.test_ratio,
            ),
        )
        boundaries = (allocated[0], allocated[0] + allocated[1])
        ordered_groups = database.execute(
            "SELECT group_key FROM source_groups ORDER BY sort_key, group_key"
        )
        assignment_batch: list[tuple[str, str]] = []
        for position, (group_key,) in enumerate(ordered_groups):
            split_name = (
                "train"
                if position < boundaries[0]
                else "validation"
                if position < boundaries[1]
                else "test"
            )
            assignment_batch.append((group_key, split_name))
            if len(assignment_batch) >= config.performance.sqlite_batch_size:
                database.executemany(
                    "INSERT INTO assignments (group_key, split_name) VALUES (?, ?)",
                    assignment_batch,
                )
                assignment_batch.clear()
        if assignment_batch:
            database.executemany(
                "INSERT INTO assignments (group_key, split_name) VALUES (?, ?)",
                assignment_batch,
            )
        database.commit()

        record_batch: list[tuple[str, str, str, str]] = []
        last_group = last_split = None
        for record in iter_jsonl(config.paths.validated):
            group_key = _split_group_key(record, config.split.group_by)
            if group_key != last_group:
                row = database.execute(
                    "SELECT split_name FROM assignments WHERE group_key = ?",
                    (group_key,),
                ).fetchone()
                if row is None:
                    raise DatasetPipelineError(f"Missing split assignment for {group_key!r}.")
                last_group, last_split = group_key, str(row[0])
            assert last_split is not None
            record_id = str(record.get("record_id", ""))
            record_batch.append(
                (
                    last_split,
                    _seeded_record_key(record, config.split.seed, last_split),
                    record_id,
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                )
            )
            counts[last_split] += 1
            split_categories[last_split][
                str(record.get("category", "code_generation"))
            ] += 1
            processed += 1
            if len(record_batch) >= config.performance.sqlite_batch_size:
                database.executemany(
                    "INSERT INTO records (split_name, sort_key, record_id, payload) "
                    "VALUES (?, ?, ?, ?)",
                    record_batch,
                )
                record_batch.clear()
                database.commit()
            progress.update(processed)
        if record_batch:
            database.executemany(
                "INSERT INTO records (split_name, sort_key, record_id, payload) "
                "VALUES (?, ?, ?, ?)",
                record_batch,
            )
        database.execute(
            "CREATE INDEX records_output_order "
            "ON records (split_name, sort_key, record_id)"
        )
        database.commit()

        partials = {
            name: Path(str(path) + ".partial")
            for name, path in zip(split_names, outputs, strict=True)
        }
        try:
            for split_name, destination in zip(split_names, outputs, strict=True):
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = partials[split_name]
                partial.unlink(missing_ok=True)
                with partial.open("w", encoding="utf-8", newline="\n") as output:
                    rows = database.execute(
                        "SELECT payload FROM records WHERE split_name = ? "
                        "ORDER BY sort_key, record_id",
                        (split_name,),
                    )
                    for (payload,) in rows:
                        output.write(payload)
                        output.write("\n")
                        processed += 1
                        progress.update(processed)
            for split_name, destination in zip(split_names, outputs, strict=True):
                os.replace(partials[split_name], destination)
        except Exception:
            for partial in partials.values():
                partial.unlink(missing_ok=True)
            raise
    progress.close()
    group_counts = dict(zip(split_names, allocated, strict=True))
    reasons = {
        **{f"{name}_records": counts.get(name, 0) for name in split_names},
        **{f"{name}_source_groups": group_counts.get(name, 0) for name in split_names},
    }
    return _finish_stage(
        config,
        "split",
        total,
        total,
        fingerprint,
        outputs,
        reasons=reasons,
        categories=categories,
        split_categories=split_categories,
    )


def build_dataset(config: PipelineConfig, *, resume: bool = True) -> BuildResult:
    """Run every stage and write aggregate final dataset statistics."""

    stages = (
        collect_python_data(config, resume=resume),
        clean_python_dataset(config, resume=resume),
        generate_instruction_pairs(config, resume=resume),
        deduplicate_dataset(config, resume=resume),
        validate_dataset(config, resume=resume),
        split_dataset(config, resume=resume),
    )
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_implementation_version": PIPELINE_IMPLEMENTATION_VERSION,
        "created_at": _timestamp(),
        "config": str(config.config_path),
        "config_sha256": _file_hash(config.config_path),
        "approved_sources": [_source_as_mapping(source) for source in config.sources],
        "settings": {
            "cleaning": asdict(config.cleaning),
            "instruction_generation": asdict(config.pair_generation),
            "validation": asdict(config.validation),
            "split": asdict(config.split),
            "performance": asdict(config.performance),
        },
        "stages": [asdict(stage) for stage in stages],
        "category_counts": stages[-1].category_counts,
        "split_category_counts": stages[-1].split_category_counts,
        "final_outputs": {
            name: _output_summary(
                path,
                records=stages[-1].reason_counts[f"{name}_records"],
                sha256=stages[-1].output_hashes[str(path)],
                byte_count=stages[-1].output_metadata[str(path)]["size"],
            )
            for name, path in (
                ("train", config.paths.train),
                ("validation", config.paths.validation),
                ("test", config.paths.test),
            )
        },
    }
    atomic_json_dump(payload, config.paths.statistics)
    LOGGER.info(
        "Dataset ready: train=%s validation=%s test=%s",
        stages[-1].reason_counts["train_records"],
        stages[-1].reason_counts["validation_records"],
        stages[-1].reason_counts["test_records"],
    )
    return BuildResult(
        train_path=config.paths.train,
        validation_path=config.paths.validation,
        test_path=config.paths.test,
        statistics_path=config.paths.statistics,
        stages=stages,
    )


def run_stage_cli(stage: str, argv: Sequence[str] | None = None) -> int:
    """Shared command-line implementation used by each modular stage script."""

    if stage not in {*STAGES, "build"}:
        raise DatasetPipelineError(f"Unknown dataset pipeline stage: {stage}")
    parser = argparse.ArgumentParser(description=f"Run the GenPy dataset {stage} stage.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_pipeline.yaml"),
        help="Pipeline YAML configuration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild this stage instead of resuming a current completed result.",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    try:
        config = load_pipeline_config(args.config)
        configure_pipeline_logging(config, level=args.log_level)
        resume = not args.force
        if stage == "build":
            result = build_dataset(config, resume=resume)
            print("Python dataset pipeline complete")
            print(f"Train: {result.train_path}")
            print(f"Validation: {result.validation_path}")
            print(f"Test: {result.test_path}")
            print(f"Statistics: {result.statistics_path}")
        else:
            function = {
                "collect": collect_python_data,
                "clean": clean_python_dataset,
                "generate": generate_instruction_pairs,
                "deduplicate": deduplicate_dataset,
                "validate": validate_dataset,
                "split": split_dataset,
            }[stage]
            statistics = function(config, resume=resume)
            print(format_stage_statistics(statistics))
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Dataset pipeline stage failed")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield strict JSON-object records with file and line diagnostics."""

    try:
        file = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise JsonlValidationError(f"Could not open JSONL file {path}: {exc}") from exc
    with file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlValidationError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise JsonlValidationError(
                    f"JSONL record in {path}:{line_number} must be an object."
                )
            yield record


def count_jsonl_records(path: Path) -> int:
    """Count non-empty JSONL records."""

    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def instruction_pair_hash(record: Mapping[str, Any]) -> str:
    """Return a canonical exact-deduplication hash for an instruction pair."""

    instruction = _normalized_instruction(record.get("instruction"))
    input_text = _normalized_instruction(record.get("input"))
    output = record.get("output")
    output_text = normalize_python_source(output).strip() if isinstance(output, str) else ""
    return _text_hash(f"{instruction}\0{input_text}\0{output_text}")


def validate_instruction_record(
    record: Mapping[str, Any],
    settings: ValidationSettings,
    *,
    seen_ids: set[str] | None = None,
    seen_pairs: set[str] | None = None,
) -> list[str]:
    """Return stable validation error codes for one instruction record."""

    errors: list[str] = []
    required_strings = ("record_id", "instruction", "input", "output")
    values: dict[str, str] = {}
    for field_name in required_strings:
        value = record.get(field_name)
        if not isinstance(value, str):
            errors.append(f"invalid_{field_name}_type")
            values[field_name] = ""
        else:
            values[field_name] = value
    instruction = values["instruction"].strip()
    output = values["output"]
    if len(instruction) < settings.minimum_instruction_characters:
        errors.append("instruction_too_short")
    if len(instruction) > settings.maximum_instruction_characters:
        errors.append("instruction_too_long")
    if len(output.strip()) < settings.minimum_output_characters:
        errors.append("output_too_short")
    if len(output.encode("utf-8")) > settings.maximum_output_bytes:
        errors.append("output_too_large")
    if "\x00" in output or "\x00" in instruction:
        errors.append("null_character")
    category = record.get("category", "code_generation")
    if not isinstance(category, str) or category not in INSTRUCTION_CATEGORIES:
        errors.append("invalid_category")
        category = "code_generation"
    if (
        settings.require_python_syntax
        and category in PYTHON_OUTPUT_CATEGORIES
        and output.strip()
    ):
        try:
            ast.parse(textwrap.dedent(output))
        except (SyntaxError, ValueError):
            errors.append("invalid_python_syntax")
    provenance = record.get("provenance")
    if settings.require_provenance:
        if not isinstance(provenance, dict):
            errors.append("missing_provenance")
        else:
            for key in ("source_id", "repository", "source_path", "content_hash", "approval"):
                if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                    errors.append(f"missing_provenance_{key}")
    record_id = values["record_id"]
    if seen_ids is not None and record_id:
        if record_id in seen_ids:
            errors.append("duplicate_record_id")
        seen_ids.add(record_id)
    pair_hash = instruction_pair_hash(record)
    if seen_pairs is not None:
        if pair_hash in seen_pairs:
            errors.append("duplicate_instruction_pair")
        seen_pairs.add(pair_hash)
    return sorted(set(errors))


def format_stage_statistics(statistics: StageStatistics) -> str:
    """Return a concise human-readable stage summary."""

    status = "resumed" if statistics.resumed else "completed"
    return (
        f"{statistics.stage}: {status}; read={statistics.read_records} "
        f"written={statistics.written_records} rejected={statistics.rejected_records} "
        f"duplicates={statistics.duplicate_records}"
    )


class ProgressBar:
    """Small dependency-free terminal progress bar."""

    def __init__(self, label: str, total: int, *, enabled: bool = True, width: int = 28) -> None:
        self.label = label
        self.total = max(int(total), 0)
        self.enabled = enabled
        self.width = width
        self.current = 0
        self._last_rendered = -1
        if self.enabled:
            self._render(0)

    def update(self, current: int) -> None:
        self.current = max(int(current), 0)
        if not self.enabled:
            return
        percent = 100 if self.total == 0 else int(min(self.current / self.total, 1.0) * 100)
        filled = int(self.width * percent / 100)
        if filled != self._last_rendered:
            self._render(percent)

    def close(self) -> None:
        if self.enabled:
            if self._last_rendered != self.width:
                self._render(100)
            print()

    def _render(self, percent: int) -> None:
        filled = int(self.width * percent / 100)
        self._last_rendered = filled
        bar = "#" * filled + "-" * (self.width - filled)
        print(
            f"\r{self.label:12s} [{bar}] {percent:3d}% "
            f"({self.current}/{self.total})",
            end="",
            flush=True,
        )


def _load_source(project_root: Path, raw: object) -> ApprovedSource:
    mapping = _required_mapping(raw, "approved source")
    source_id = _required_string(mapping.get("id"), "approved source id")
    source_path = _resolve(project_root, mapping.get("path"))
    repository = _required_string(mapping.get("repository"), f"{source_id}.repository")
    approval = _required_string(mapping.get("approval"), f"{source_id}.approval")
    license_value = mapping.get("license")
    if license_value is not None and not isinstance(license_value, str):
        raise DatasetPipelineError(f"{source_id}.license must be a string or null.")
    include = _string_tuple(mapping.get("include", ["**/*.py"]), f"{source_id}.include")
    exclude = _string_tuple(mapping.get("exclude", []), f"{source_id}.exclude")
    if not source_path.is_dir():
        raise FileNotFoundError(f"Approved source directory not found: {source_path}")
    return ApprovedSource(
        source_id=source_id,
        path=source_path,
        repository=repository,
        approval=approval,
        license=license_value.strip() if isinstance(license_value, str) else None,
        include=include,
        exclude=exclude,
    )


def _validate_config(config: PipelineConfig) -> None:
    if config.cleaning.minimum_file_bytes < 0:
        raise DatasetPipelineError("cleaning.minimum_file_bytes must be non-negative.")
    if config.cleaning.maximum_file_bytes <= config.cleaning.minimum_file_bytes:
        raise DatasetPipelineError(
            "cleaning.maximum_file_bytes must exceed minimum_file_bytes."
        )
    if config.pair_generation.minimum_instruction_characters <= 0:
        raise DatasetPipelineError(
            "instruction_generation.minimum_instruction_characters must be positive."
        )
    if (
        config.pair_generation.maximum_instruction_characters
        < config.pair_generation.minimum_instruction_characters
    ):
        raise DatasetPipelineError(
            "instruction_generation.maximum_instruction_characters must be at least "
            "minimum_instruction_characters."
        )
    if config.pair_generation.maximum_examples_per_file < 0:
        raise DatasetPipelineError(
            "instruction_generation.maximum_examples_per_file must be zero or positive."
        )
    invalid_categories = sorted(
        set(config.pair_generation.enabled_categories) - set(INSTRUCTION_CATEGORIES)
    )
    if invalid_categories:
        raise DatasetPipelineError(
            "Unsupported instruction categories: " + ", ".join(invalid_categories)
        )
    if not config.pair_generation.enabled_categories:
        raise DatasetPipelineError(
            "instruction_generation.enabled_categories must not be empty."
        )
    templates = dict(config.pair_generation.templates)
    missing_templates = sorted(set(config.pair_generation.enabled_categories) - templates.keys())
    if missing_templates:
        raise DatasetPipelineError(
            "Missing instruction templates: " + ", ".join(missing_templates)
        )
    for category, template in templates.items():
        _validate_instruction_template(category, template)
    ratios = (
        config.split.train_ratio,
        config.split.validation_ratio,
        config.split.test_ratio,
    )
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise DatasetPipelineError("split ratios must be non-negative and sum to 1.0.")
    if config.split.group_by not in {"source_file", "content_hash"}:
        raise DatasetPipelineError("split.group_by must be source_file or content_hash.")
    if config.performance.workers < 0:
        raise DatasetPipelineError("performance.workers must be zero or positive.")
    if config.performance.max_pending_tasks_per_worker <= 0:
        raise DatasetPipelineError(
            "performance.max_pending_tasks_per_worker must be positive."
        )
    if config.performance.sqlite_batch_size <= 0:
        raise DatasetPipelineError("performance.sqlite_batch_size must be positive.")
    source_ids = [source.source_id for source in config.sources]
    if len(source_ids) != len(set(source_ids)):
        raise DatasetPipelineError("Approved source IDs must be unique.")


def _discover_approved_files(
    sources: Iterable[ApprovedSource],
) -> list[tuple[ApprovedSource, Path]]:
    discovered: list[tuple[ApprovedSource, Path]] = []
    seen: set[Path] = set()
    for source in sources:
        root = source.path.resolve()
        for path in sorted(root.rglob("*.py")):
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file() or path.is_symlink():
                continue
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise DatasetPipelineError(f"Source path escapes approved root: {path}") from exc
            if not _matches_any(relative, source.include):
                continue
            if _matches_any(relative, source.exclude):
                continue
            seen.add(resolved)
            discovered.append((source, resolved))
    return sorted(discovered, key=lambda item: (item[0].source_id, str(item[1])))


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(
        fnmatch.fnmatch(path, pattern)
        or Path(path).match(pattern)
        or (pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]))
        for pattern in patterns
    )


def _clean_source_record(
    record: dict[str, Any],
    settings: CleaningSettings,
) -> tuple[str | None, dict[str, Any] | None]:
    content = record.get("content")
    if not isinstance(content, str):
        return "missing_content", None
    normalized = normalize_python_source(content)
    byte_count = len(normalized.encode("utf-8"))
    if byte_count < settings.minimum_file_bytes:
        return "too_small", None
    if byte_count > settings.maximum_file_bytes:
        return "too_large", None
    try:
        tree = ast.parse(normalized, filename=str(record.get("source_path", "<source>")))
    except (SyntaxError, ValueError):
        return "invalid_python_syntax", None
    if settings.require_python_definitions and not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    ):
        return "no_python_definitions", None
    cleaned = dict(record)
    cleaned["content"] = normalized
    cleaned["content_hash"] = _text_hash(normalized)
    cleaned["byte_count"] = byte_count
    return None, cleaned


def _clean_source_worker(
    payload: tuple[dict[str, Any], CleaningSettings],
) -> tuple[str | None, dict[str, Any] | None]:
    record, settings = payload
    return _clean_source_record(record, settings)


def _generate_pairs_worker(
    payload: tuple[dict[str, Any], PairGenerationSettings],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    record, settings = payload
    return _pairs_from_source(record, settings)


def _bounded_parallel_map(
    function: Callable[[_Input], _Output],
    values: Iterable[_Input],
    settings: PerformanceSettings,
) -> Iterator[_Output]:
    """Map in input order without allowing the process queue to grow unbounded."""

    workers = settings.workers or min(os.cpu_count() or 1, 8)
    if workers == 1:
        for value in values:
            yield function(value)
        return

    pending_limit = workers * settings.max_pending_tasks_per_worker
    LOGGER.debug(
        "Using %s worker processes with at most %s pending AST tasks",
        workers,
        pending_limit,
    )
    iterator = iter(values)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        pending: deque[concurrent.futures.Future[_Output]] = deque()
        for _ in range(pending_limit):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                continue


@dataclass(frozen=True)
class _DefinitionElement:
    qualified_name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    symbol_type: str


@dataclass(frozen=True)
class _ImportInfo:
    statement: str
    bindings: tuple[str, ...]


class _DefinitionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.definitions: list[_DefinitionElement] = []
        self.scope: list[tuple[str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_definition(node)

    def _visit_definition(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> None:
        qualified_name = ".".join([*(name for name, _ in self.scope), node.name])
        parent_type = self.scope[-1][1] if self.scope else None
        symbol_type = _symbol_type(node, parent_type=parent_type)
        self.definitions.append(_DefinitionElement(qualified_name, node, symbol_type))
        self.scope.append((node.name, symbol_type))
        self.generic_visit(node)
        self.scope.pop()


def _pairs_from_source(
    source_record: Mapping[str, Any],
    settings: PairGenerationSettings,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    content = source_record.get("content")
    if not isinstance(content, str):
        return [], Counter({"missing_content": 1})
    tree = ast.parse(content)
    collector = _DefinitionCollector()
    collector.visit(tree)
    imports = _collect_imports(tree)
    source_path = str(source_record.get("source_path", ""))
    reasons: Counter[str] = Counter()
    candidates: list[tuple[str, _DefinitionElement, str | None, str]] = []
    for element in collector.definitions:
        node = element.node
        if not _included_symbol_type(element.symbol_type, settings):
            reasons["excluded_symbol_type"] += 1
            continue
        name = getattr(node, "name", "")
        if not settings.include_private and name.startswith("_"):
            reasons["private_symbol"] += 1
            continue
        raw_docstring = ast.get_docstring(node, clean=True)
        docstring = _normalized_instruction(raw_docstring) or None
        if settings.require_docstring and not docstring:
            reasons["missing_docstring"] += 1
            continue
        for category in settings.enabled_categories:
            if not _category_maybe_applicable(
                category,
                element,
                source_path,
                docstring,
                imports,
            ):
                continue
            selection_key = _text_hash(
                f"{settings.selection_seed}\0{source_record.get('content_hash')}\0"
                f"{element.qualified_name}\0{getattr(node, 'lineno', 0)}\0{category}"
            )
            candidates.append((selection_key, element, docstring, category))
    limit = settings.maximum_examples_per_file
    pairs: list[dict[str, Any]] = []
    attempted: set[str] = set()

    def materialize(candidate: tuple[str, _DefinitionElement, str | None, str]) -> None:
        selection_key, element, docstring, category = candidate
        attempted.add(selection_key)
        pair = _category_pair(
            category,
            element,
            source_path,
            content,
            docstring,
            imports,
            source_record,
            settings,
            reasons,
        )
        if pair is not None:
            pairs.append(pair)

    if limit:
        for category in settings.enabled_categories:
            ranked = sorted(
                (candidate for candidate in candidates if candidate[3] == category),
                key=lambda candidate: candidate[0],
            )
            for candidate in ranked:
                before = len(pairs)
                materialize(candidate)
                if len(pairs) > before or len(pairs) >= limit:
                    break
            if len(pairs) >= limit:
                break
        if len(pairs) < limit:
            for candidate in sorted(candidates, key=lambda item: item[0]):
                if candidate[0] in attempted:
                    continue
                materialize(candidate)
                if len(pairs) >= limit:
                    break
    else:
        for candidate in candidates:
            materialize(candidate)
    if limit:
        pairs.sort(key=lambda record: str(record["record_id"]))
    return pairs, reasons


def _category_pair(
    category: str,
    element: _DefinitionElement,
    source_path: str,
    source: str,
    docstring: str | None,
    imports: tuple[_ImportInfo, ...],
    source_record: Mapping[str, Any],
    settings: PairGenerationSettings,
    reasons: Counter[str],
) -> dict[str, Any] | None:
    node = element.node
    original_output = ast.get_source_segment(source, node)
    if not isinstance(original_output, str) or not original_output.strip():
        reasons["missing_source_segment"] += 1
        return None
    if len(original_output.encode("utf-8")) > settings.maximum_output_bytes:
        reasons["output_too_large"] += 1
        return None
    pair = _category_payload(
        category,
        element,
        source_path,
        original_output,
        docstring,
        imports,
    )
    if pair is None:
        return None
    input_text, response, metadata = pair
    templates = dict(settings.templates)
    template = templates[category]
    base_instruction = ""
    if "{base_instruction" in template:
        base_instruction = (
            _truncate_instruction(docstring, settings.maximum_instruction_characters)
            if docstring
            else _infer_instruction(node, source)
        )
    context = {
        "api": "Python",
        "base_instruction": base_instruction,
        "description": _description_text(docstring, node),
        "kind": element.symbol_type.replace("_", " "),
        "name": node.name,
        "qualified_name": element.qualified_name,
        "signature": _definition_signature(node),
    }
    instruction = _render_instruction_template(template, context | metadata)
    if len(instruction) < settings.minimum_instruction_characters:
        reasons["instruction_too_short"] += 1
        return None
    if len(instruction) > settings.maximum_instruction_characters:
        reasons["instruction_too_long"] += 1
        return None
    if len(response.encode("utf-8")) > settings.maximum_output_bytes:
        reasons["output_too_large"] += 1
        return None
    provenance = {
        "source_id": source_record.get("source_id"),
        "repository": source_record.get("repository"),
        "source_path": source_record.get("source_path"),
        "content_hash": source_record.get("content_hash"),
        "approval": source_record.get("approval"),
        "license": source_record.get("license"),
        "symbol": element.qualified_name,
        "symbol_type": element.symbol_type,
        "line_start": getattr(node, "lineno", None),
        "line_end": getattr(node, "end_lineno", None),
        "imports": [item.statement for item in imports[:24]],
        "decorators": _decorator_names(node),
        "type_hints": _type_hint_mapping(node),
        "category": category,
        "instruction_source": (
            "docstring"
            if category == "code_generation" and docstring
            else "inferred"
            if category == "code_generation"
            else "template"
        ),
        **metadata,
    }
    record_id = _text_hash(
        f"{provenance['source_id']}\0{provenance['source_path']}\0"
        f"{element.qualified_name}\0{provenance['line_start']}\0{category}\0"
        f"{_text_hash(input_text)}\0{_text_hash(response)}"
    )
    return {
        "schema_version": PIPELINE_VERSION,
        "record_id": record_id,
        "category": category,
        "instruction": instruction,
        "input": input_text,
        "output": response,
        "provenance": provenance,
    }


def _category_maybe_applicable(
    category: str,
    element: _DefinitionElement,
    source_path: str,
    docstring: str | None,
    imports: tuple[_ImportInfo, ...],
) -> bool:
    node = element.node
    if category == "documentation":
        return bool(docstring)
    if category == "unit_testing":
        return _is_unit_test_element(element, source_path)
    if category == "type_hints":
        return bool(_type_hint_mapping(node))
    if category == "bug_fixing":
        return any(
            isinstance(child, ast.Compare)
            or isinstance(child, ast.Constant)
            and isinstance(child.value, bool)
            or isinstance(child, ast.BinOp)
            and isinstance(child.op, (ast.Add, ast.Sub))
            for child in ast.walk(node)
        )
    if category == "optimization":
        return _append_loop_parts(node) is not None
    if category == "complexity_analysis":
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(child, (ast.For, ast.AsyncFor, ast.While))
            or isinstance(child, ast.Call)
            and (
                isinstance(child.func, ast.Name)
                and child.func.id == node.name
                or isinstance(child.func, ast.Attribute)
                and child.func.attr == node.name
            )
            for child in _walk_definition_body(node)
        )
    if category == "api_usage":
        return _referenced_import(node, imports) is not None
    return True


def _category_payload(
    category: str,
    element: _DefinitionElement,
    source_path: str,
    original_output: str,
    docstring: str | None,
    imports: tuple[_ImportInfo, ...],
) -> tuple[str, str, dict[str, Any]] | None:
    node = element.node
    if category == "code_generation":
        return "", original_output, {}
    if category == "explanation":
        return original_output, _explain_definition(node, docstring), {}
    if category == "bug_fixing":
        buggy = _buggy_variant(node)
        if buggy:
            return buggy, original_output, {"transformation": "single_ast_mutation"}
        return None
    if category == "refactoring":
        refactored = _unparse_definition(node)
        if normalize_python_source(refactored).strip() == normalize_python_source(
            original_output
        ).strip():
            return None
        return original_output, refactored, {"transformation": "ast_normalization"}
    if category == "documentation":
        if not docstring:
            return None
        undocumented = _without_docstring(node)
        return undocumented, docstring, {"documentation_source": "source_docstring"}
    if category == "unit_testing":
        if not _is_unit_test_element(element, source_path):
            return None
        return "", original_output, {"test_source": "existing_test_code"}
    if category == "optimization":
        optimized = _optimized_variant(node)
        if optimized is None:
            return None
        return original_output, optimized, {"transformation": "list_comprehension"}
    if category == "complexity_analysis":
        analysis = _complexity_analysis(node)
        if analysis:
            return original_output, analysis, {"analysis_method": "ast_control_flow"}
        return None
    if category == "type_hints":
        untyped = _without_type_hints(node)
        if untyped is None:
            return None
        return untyped, original_output, {"transformation": "remove_type_hints"}
    if category == "code_completion":
        return _completion_skeleton(node), original_output, {}
    if category == "api_usage":
        used_import = _referenced_import(node, imports)
        if used_import is None:
            return None
        api = used_import.bindings[0] if used_import.bindings else "Python"
        return used_import.statement, original_output, {"api": api}
    raise DatasetPipelineError(f"Unsupported instruction category: {category}")


def _symbol_type(node: ast.AST, *, parent_type: str | None = None) -> str:
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_method" if parent_type == "class" else "async_function"
    if isinstance(node, ast.FunctionDef):
        return "method" if parent_type == "class" else "function"
    return "class"


def _included_symbol_type(symbol_type: str, settings: PairGenerationSettings) -> bool:
    return {
        "function": settings.include_functions,
        "async_function": settings.include_async_functions,
        "method": settings.include_methods,
        "async_method": settings.include_methods and settings.include_async_functions,
        "class": settings.include_classes,
    }[symbol_type]


def _render_instruction_template(template: str, context: Mapping[str, Any]) -> str:
    try:
        return template.format_map(context).strip()
    except (KeyError, ValueError) as exc:  # pragma: no cover - config validation guards this
        raise DatasetPipelineError(f"Could not render instruction template: {exc}") from exc


def _description_text(
    docstring: str | None,
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    if docstring:
        return re.sub(r"\s+", " ", docstring).strip().split(". ", 1)[0].rstrip(".")
    return _identifier_action(node.name)


def _definition_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(_annotation_text(base) for base in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    arguments = ast.unparse(node.args)
    returns = f" -> {_annotation_text(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({arguments}){returns}"


def _decorator_names(node: ast.AST) -> list[str]:
    decorators = getattr(node, "decorator_list", [])
    return [_annotation_text(decorator) for decorator in decorators]


def _type_hint_mapping(node: ast.AST) -> dict[str, str]:
    hints: dict[str, str] = {}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        for argument in arguments:
            hint = _annotation_text(argument.annotation)
            if hint:
                hints[argument.arg] = hint
        return_hint = _annotation_text(node.returns)
        if return_hint:
            hints["return"] = return_hint
    elif isinstance(node, ast.ClassDef):
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                hints[child.target.id] = _annotation_text(child.annotation)
    return hints


def _collect_imports(tree: ast.Module) -> tuple[_ImportInfo, ...]:
    imports: list[_ImportInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bindings = tuple(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bindings = tuple(
                alias.asname or alias.name for alias in node.names if alias.name != "*"
            )
        else:
            continue
        imports.append(_ImportInfo(ast.unparse(node), bindings))
    return tuple(imports)


def _unparse_definition(node: ast.AST) -> str:
    value = ast.unparse(copy.deepcopy(node)).strip()
    ast.parse(value)
    return value


def _explain_definition(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    docstring: str | None,
) -> str:
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    sentences = [f"`{node.name}` is a Python {kind}."]
    if docstring:
        purpose = re.sub(r"\s+", " ", docstring).strip().split(". ", 1)[0].rstrip(".")
        sentences.append(f"Its documented purpose is to {purpose[:1].lower() + purpose[1:]}.")
    if isinstance(node, ast.ClassDef):
        methods = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if methods:
            sentences.append(
                "It exposes behavior through "
                + ", ".join(f"`{name}`" for name in methods[:8])
                + "."
            )
        bases = [_annotation_text(base) for base in node.bases]
        if bases:
            sentences.append("It inherits from " + ", ".join(f"`{base}`" for base in bases) + ".")
    else:
        parameters = _format_parameters(node.args)
        if parameters:
            sentences.append(f"It accepts {parameters}.")
        return_hint = _annotation_text(node.returns)
        if return_hint:
            sentences.append(f"Its declared return type is `{return_hint}`.")
        guidance = _function_body_guidance(node)
        sentences.extend(guidance)
    decorators = _decorator_names(node)
    if decorators:
        sentences.append(
            "It is decorated with " + ", ".join(f"`{name}`" for name in decorators) + "."
        )
    return " ".join(_deduplicate_sentences(sentences))


class _BugMutation(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if self.changed or not node.ops:
            return node
        replacements: dict[type[ast.cmpop], ast.cmpop] = {
            ast.Eq: ast.NotEq(),
            ast.NotEq: ast.Eq(),
            ast.Lt: ast.GtE(),
            ast.LtE: ast.Gt(),
            ast.Gt: ast.LtE(),
            ast.GtE: ast.Lt(),
            ast.In: ast.NotIn(),
            ast.NotIn: ast.In(),
            ast.Is: ast.IsNot(),
            ast.IsNot: ast.Is(),
        }
        replacement = replacements.get(type(node.ops[0]))
        if replacement is not None:
            node.ops[0] = replacement
            self.changed = True
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if not self.changed and isinstance(node.value, bool):
            self.changed = True
            return ast.copy_location(ast.Constant(not node.value), node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if self.changed:
            return node
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
            self.changed = True
        elif isinstance(node.op, ast.Sub):
            node.op = ast.Add()
            self.changed = True
        return node


def _buggy_variant(node: ast.AST) -> str | None:
    candidate = copy.deepcopy(node)
    transformer = _BugMutation()
    candidate = transformer.visit(candidate)
    if not transformer.changed:
        return None
    ast.fix_missing_locations(candidate)
    value = ast.unparse(candidate).strip()
    ast.parse(value)
    return value


def _without_docstring(node: ast.AST) -> str:
    candidate = copy.deepcopy(node)
    body = getattr(candidate, "body", None)
    if isinstance(body, list) and body and _is_docstring_statement(body[0]):
        del body[0]
        if not body:
            body.append(ast.Pass())
    ast.fix_missing_locations(candidate)
    return ast.unparse(candidate).strip()


def _is_docstring_statement(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _without_type_hints(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if not _type_hint_mapping(node):
        return None
    candidate = copy.deepcopy(node)
    candidate.returns = None
    arguments = [
        *candidate.args.posonlyargs,
        *candidate.args.args,
        *candidate.args.kwonlyargs,
    ]
    if candidate.args.vararg is not None:
        arguments.append(candidate.args.vararg)
    if candidate.args.kwarg is not None:
        arguments.append(candidate.args.kwarg)
    for argument in arguments:
        argument.annotation = None
    ast.fix_missing_locations(candidate)
    return ast.unparse(candidate).strip()


def _completion_skeleton(node: ast.AST) -> str:
    candidate = copy.deepcopy(node)
    if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        candidate.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
    ast.fix_missing_locations(candidate)
    return ast.unparse(candidate).strip()


def _optimized_variant(node: ast.AST) -> str | None:
    candidate = copy.deepcopy(node)
    parts = _append_loop_parts(candidate)
    if parts is None or not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    prefix, loop, element, conditions = parts
    generator = ast.comprehension(
        target=loop.target,
        iter=loop.iter,
        ifs=conditions,
        is_async=int(isinstance(loop, ast.AsyncFor)),
    )
    optimized_return = ast.Return(
        value=ast.ListComp(elt=element, generators=[generator])
    )
    candidate.body = [*prefix, optimized_return]
    ast.fix_missing_locations(candidate)
    value = ast.unparse(candidate).strip()
    ast.parse(value)
    return value


def _append_loop_parts(
    node: ast.AST,
) -> tuple[list[ast.stmt], ast.For | ast.AsyncFor, ast.expr, list[ast.expr]] | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    prefix: list[ast.stmt] = []
    body = list(node.body)
    if body and _is_docstring_statement(body[0]):
        prefix.append(body.pop(0))
    if len(body) != 3:
        return None
    assignment, loop, returned = body
    result_name: str | None = None
    value: ast.expr | None = None
    if (
        isinstance(assignment, ast.Assign)
        and len(assignment.targets) == 1
        and isinstance(assignment.targets[0], ast.Name)
    ):
        result_name = assignment.targets[0].id
        value = assignment.value
    elif isinstance(assignment, ast.AnnAssign) and isinstance(assignment.target, ast.Name):
        result_name = assignment.target.id
        value = assignment.value
    if not (
        result_name
        and isinstance(value, ast.List)
        and not value.elts
        and isinstance(loop, (ast.For, ast.AsyncFor))
        and not loop.orelse
        and len(loop.body) == 1
        and isinstance(returned, ast.Return)
        and isinstance(returned.value, ast.Name)
        and returned.value.id == result_name
    ):
        return None
    statement = loop.body[0]
    conditions: list[ast.expr] = []
    if isinstance(statement, ast.If) and not statement.orelse and len(statement.body) == 1:
        conditions.append(statement.test)
        statement = statement.body[0]
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == result_name
        and statement.value.func.attr == "append"
        and len(statement.value.args) == 1
        and not statement.value.keywords
    ):
        return None
    return prefix, loop, statement.value.args[0], conditions


def _complexity_analysis(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    loop_depth = _maximum_loop_depth(node)
    recursive = any(
        isinstance(child, ast.Call)
        and (
            isinstance(child.func, ast.Name)
            and child.func.id == node.name
            or isinstance(child.func, ast.Attribute)
            and child.func.attr == node.name
        )
        for child in _walk_definition_body(node)
    )
    if loop_depth == 0 and not recursive:
        return None
    if loop_depth:
        time = "O(n)" if loop_depth == 1 else f"O(n^{loop_depth})"
        basis = f"a maximum visible loop nesting depth of {loop_depth}"
    else:
        time = "recurrence-dependent"
        basis = "a recursive self-call whose input reduction is not statically provable"
    allocates_collection = any(
        isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.List, ast.Set, ast.Dict))
        for child in _walk_definition_body(node)
    )
    space = "O(n)" if allocates_collection else "O(1) excluding call-stack and returned data"
    recursion_note = " It also contains recursion." if recursive else ""
    return (
        f"From the visible AST, the structural time bound is {time}, based on {basis}. "
        f"The visible auxiliary-space bound is {space}.{recursion_note} "
        "Costs inside called APIs and data-dependent iteration bounds may change the final bound."
    )


def _maximum_loop_depth(node: ast.AST) -> int:
    def visit(current: ast.AST, depth: int, *, root: bool = False) -> int:
        if not root and isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return depth
        next_depth = depth + int(isinstance(current, (ast.For, ast.AsyncFor, ast.While)))
        maximum = next_depth
        for child in ast.iter_child_nodes(current):
            maximum = max(maximum, visit(child, next_depth))
        return maximum

    return visit(node, 0, root=True)


def _is_unit_test_element(element: _DefinitionElement, source_path: str) -> bool:
    name = element.node.name.casefold()
    if element.symbol_type == "class":
        return name.startswith("test")
    return name.startswith("test_") or name == "test"


def _referenced_import(node: ast.AST, imports: tuple[_ImportInfo, ...]) -> _ImportInfo | None:
    loaded_names = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    for imported in imports:
        if any(binding in loaded_names for binding in imported.bindings):
            return imported
    return None


def _infer_instruction(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
) -> str:
    """Infer a grounded instruction from one undocumented Python definition."""

    if isinstance(node, ast.ClassDef):
        return _infer_class_instruction(node, source)
    return _infer_function_instruction(node, source)


def _infer_function_instruction(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
) -> str:
    parameters = _format_parameters(node.args)
    parameter_clause = f" with parameters {parameters}" if parameters else " without parameters"
    return_hint = _annotation_text(node.returns)
    return_clause = f" and return type `{return_hint}`" if return_hint else ""
    kind = "asynchronous function" if isinstance(node, ast.AsyncFunctionDef) else "function"
    sentences = [
        f"Implement the {kind} `{node.name}`{parameter_clause}{return_clause}.",
        f"Use it to {_identifier_action(node.name)}.",
    ]
    comments = _source_comments(node, source)
    if comments:
        sentences.append(f"Honor this source guidance: {'; '.join(comments)}.")
    sentences.extend(_function_body_guidance(node))
    return " ".join(_deduplicate_sentences(sentences))


def _infer_class_instruction(node: ast.ClassDef, source: str) -> str:
    sentences = [f"Implement the `{node.name}` class to {_identifier_action(node.name)}."]
    bases = [_annotation_text(base) for base in node.bases]
    bases = [base for base in bases if base]
    if bases:
        sentences.append(f"Derive it from {', '.join(f'`{base}`' for base in bases)}.")
    constructor = next(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__init__"
        ),
        None,
    )
    if constructor is not None:
        parameters = _format_parameters(constructor.args)
        if parameters:
            sentences.append(f"Initialize instances with parameters {parameters}.")
    fields = _class_fields(node)
    if fields:
        sentences.append(f"Maintain the fields {', '.join(fields)}.")
    methods = [
        child.name
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name != "__init__"
    ]
    if methods:
        displayed = ", ".join(f"`{name}`" for name in methods[:8])
        sentences.append(f"Provide behavior through {displayed}.")
    comments = _source_comments(node, source)
    if comments:
        sentences.append(f"Honor this source guidance: {'; '.join(comments)}.")
    if any(isinstance(child, ast.Raise) for child in ast.walk(node)):
        sentences.append("Validate unsupported states and raise the defined errors.")
    return " ".join(_deduplicate_sentences(sentences))


def _format_parameters(arguments: ast.arguments) -> str:
    positional = [*arguments.posonlyargs, *arguments.args]
    values: list[str] = []
    for argument in positional:
        if argument.arg in {"self", "cls"}:
            continue
        values.append(_format_parameter(argument.arg, argument.annotation))
    if arguments.vararg is not None:
        values.append("*" + _format_parameter(arguments.vararg.arg, arguments.vararg.annotation))
    for argument in arguments.kwonlyargs:
        values.append(_format_parameter(argument.arg, argument.annotation))
    if arguments.kwarg is not None:
        values.append("**" + _format_parameter(arguments.kwarg.arg, arguments.kwarg.annotation))
    return ", ".join(f"`{value}`" for value in values)


def _format_parameter(name: str, annotation: ast.expr | None) -> str:
    hint = _annotation_text(annotation)
    return f"{name}: {hint}" if hint else name


def _annotation_text(annotation: ast.expr | None) -> str:
    if annotation is None:
        return ""
    try:
        return ast.unparse(annotation)
    except (AttributeError, ValueError):
        return ""


def _identifier_action(name: str) -> str:
    special_names = {
        "__init__": "initialize an instance",
        "__call__": "make instances callable",
        "__enter__": "enter a managed context",
        "__exit__": "leave a managed context",
        "__iter__": "iterate over the available values",
        "__next__": "return the next available value",
        "__len__": "report the number of contained values",
        "__getitem__": "retrieve an item by key or index",
        "__setitem__": "store an item by key or index",
        "__str__": "produce a readable string representation",
        "__repr__": "produce a developer-facing representation",
    }
    if name in special_names:
        return special_names[name]
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name.strip("_")).replace("_", " ")
    words = re.sub(r"\s+", " ", words).strip().lower()
    return words or "provide the defined behavior"


def _source_comments(node: ast.AST, source: str) -> list[str]:
    start_line = max(int(getattr(node, "lineno", 1)) - 2, 1)
    end_line = int(getattr(node, "end_lineno", start_line))
    comments: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT or not start_line <= token.start[0] <= end_line:
                continue
            comment = token.string.lstrip("#").strip()
            folded = comment.lower()
            if not comment or any(
                marker in folded
                for marker in ("noqa", "type: ignore", "pragma:", "fmt:", "nosec")
            ):
                continue
            comment = re.sub(r"\s+", " ", comment)[:180].rstrip()
            if comment and comment not in comments:
                comments.append(comment)
            if len(comments) >= 2:
                break
    except (IndentationError, tokenize.TokenError):
        return []
    return comments


def _function_body_guidance(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    body_nodes = list(_walk_definition_body(node))
    guidance: list[str] = []
    if any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in body_nodes):
        guidance.append("Produce values incrementally as an iterator.")
    elif any(isinstance(child, ast.Return) for child in body_nodes):
        guidance.append("Return the computed result.")
    if any(isinstance(child, ast.Await) for child in body_nodes):
        guidance.append("Await the required asynchronous operations.")
    if any(isinstance(child, (ast.For, ast.AsyncFor, ast.While)) for child in body_nodes):
        guidance.append("Process the relevant values iteratively.")
    if any(isinstance(child, ast.If) for child in body_nodes):
        guidance.append("Handle the conditional cases represented by the inputs.")
    if any(isinstance(child, ast.Raise) for child in body_nodes):
        guidance.append("Validate invalid cases and raise the defined errors.")
    if any(isinstance(child, (ast.With, ast.AsyncWith)) for child in body_nodes):
        guidance.append("Manage acquired resources with the appropriate context.")
    if any(_assigns_instance_state(child) for child in body_nodes):
        guidance.append("Update the relevant instance state.")
    if not guidance:
        guidance.append("Perform the operations defined by the function body.")
    return guidance


def _walk_definition_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    for statement in node.body:
        for child in ast.walk(statement):
            if child is not statement and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            yield child


def _assigns_instance_state(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return False
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    else:
        targets.append(node.target)
    return any(
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id in {"self", "cls"}
        for target in targets
    )


def _class_fields(node: ast.ClassDef) -> list[str]:
    fields: list[str] = []
    for child in node.body:
        if not isinstance(child, ast.AnnAssign) or not isinstance(child.target, ast.Name):
            continue
        hint = _annotation_text(child.annotation)
        field = f"`{child.target.id}: {hint}`" if hint else f"`{child.target.id}`"
        fields.append(field)
        if len(fields) >= 8:
            break
    return fields


def _deduplicate_sentences(sentences: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _normalized_instruction(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "\n".join(line.rstrip() for line in value.strip().splitlines()).strip()


def _truncate_instruction(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    shortened = value[: max(maximum - 1, 0)].rstrip()
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0].rstrip()
    return shortened + "…"


def _split_group_key(record: Mapping[str, Any], group_by: str) -> str:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise DatasetPipelineError("Validated record is missing provenance during split.")
    if group_by == "content_hash":
        value = provenance.get("content_hash")
    else:
        value = f"{provenance.get('source_id')}:{provenance.get('source_path')}"
    if not isinstance(value, str) or not value.strip():
        raise DatasetPipelineError(f"Could not determine {group_by} split group.")
    return value


def _assign_groups(groups: tuple[str, ...], settings: SplitSettings) -> dict[str, str]:
    split_names = ("train", "validation", "test")
    ratios = (settings.train_ratio, settings.validation_ratio, settings.test_ratio)
    counts = _allocate_split_counts(len(groups), ratios)
    ordered = sorted(groups, key=lambda group: _text_hash(f"{settings.seed}\0{group}"))
    assignments: dict[str, str] = {}
    offset = 0
    for name, count in zip(split_names, counts, strict=True):
        for group in ordered[offset : offset + count]:
            assignments[group] = name
        offset += count
    return assignments


def _allocate_split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total == 0:
        return (0, 0, 0)
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (raw[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(positive):
        for empty_index in [index for index in positive if counts[index] == 0]:
            donor = max(positive, key=lambda index: counts[index])
            counts[donor] -= 1
            counts[empty_index] += 1
    return tuple(counts)  # type: ignore[return-value]


def _seeded_record_key(record: Mapping[str, Any], seed: int, split_name: str) -> str:
    return _text_hash(f"{seed}\0{split_name}\0{record.get('record_id', '')}")


def _resume_statistics(
    config: PipelineConfig,
    stage: str,
    fingerprint: str,
    outputs: tuple[Path, ...],
    resume: bool,
) -> StageStatistics | None:
    if not resume or not all(path.is_file() for path in outputs):
        return None
    manifest_path = _manifest_path(config, stage)
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("input_fingerprint") != fingerprint:
        return None
    expected_metadata = payload.get("output_metadata")
    if not isinstance(expected_metadata, dict):
        return None
    for output in outputs:
        if expected_metadata.get(str(output)) != _file_metadata(output):
            return None
    if config.performance.verify_output_hashes_on_resume:
        expected_hashes = payload.get("output_hashes")
        if not isinstance(expected_hashes, dict):
            return None
        for output in outputs:
            if expected_hashes.get(str(output)) != _file_hash(output):
                return None
    try:
        statistics = StageStatistics(**payload)
    except TypeError:
        return None
    LOGGER.info("Resuming completed %s stage", stage)
    return StageStatistics(**{**asdict(statistics), "resumed": True})


def _finish_stage(
    config: PipelineConfig,
    stage: str,
    read: int,
    written: int,
    fingerprint: str,
    outputs: tuple[Path, ...],
    *,
    rejected: int = 0,
    duplicate: int = 0,
    reasons: Mapping[str, int] | None = None,
    categories: Mapping[str, int] | None = None,
    split_categories: Mapping[str, Mapping[str, int]] | None = None,
) -> StageStatistics:
    statistics = StageStatistics(
        stage=stage,
        read_records=read,
        written_records=written,
        rejected_records=rejected,
        duplicate_records=duplicate,
        reason_counts=dict(sorted((reasons or {}).items())),
        category_counts=dict(sorted((categories or {}).items())),
        split_category_counts={
            split: dict(sorted(counts.items()))
            for split, counts in sorted((split_categories or {}).items())
        },
        output_files=tuple(str(path) for path in outputs),
        output_hashes={str(path): _file_hash(path) for path in outputs},
        output_metadata={str(path): _file_metadata(path) for path in outputs},
        input_fingerprint=fingerprint,
        completed_at=_timestamp(),
    )
    atomic_json_dump(asdict(statistics), _manifest_path(config, stage))
    LOGGER.info(format_stage_statistics(statistics))
    return statistics


def _manifest_path(config: PipelineConfig, stage: str) -> Path:
    return config.paths.manifests / f"{stage}.json"


def _output_summary(
    path: Path,
    *,
    records: int | None = None,
    sha256: str | None = None,
    byte_count: int | None = None,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "records": count_jsonl_records(path) if records is None else records,
        "bytes": path.stat().st_size if byte_count is None else byte_count,
        "sha256": _file_hash(path) if sha256 is None else sha256,
    }


def _file_metadata(path: Path) -> dict[str, int]:
    statistics = path.stat()
    return {"size": statistics.st_size, "mtime_ns": statistics.st_mtime_ns}


def _upstream_manifest(config: PipelineConfig, stage: str) -> dict[str, Any] | None:
    path = _manifest_path(config, stage)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _upstream_output_hash(config: PipelineConfig, stage: str, path: Path) -> str:
    """Reuse an upstream manifest hash when its inexpensive file signature matches."""

    manifest = _upstream_manifest(config, stage)
    if manifest is not None and not config.performance.verify_output_hashes_on_resume:
        metadata = manifest.get("output_metadata")
        hashes = manifest.get("output_hashes")
        if (
            isinstance(metadata, dict)
            and isinstance(hashes, dict)
            and metadata.get(str(path)) == _file_metadata(path)
            and isinstance(hashes.get(str(path)), str)
        ):
            return str(hashes[str(path)])
    return _file_hash(path)


def _upstream_record_count(config: PipelineConfig, stage: str, path: Path) -> int:
    """Read an upstream count from its manifest, falling back to a streaming count."""

    manifest = _upstream_manifest(config, stage)
    if manifest is not None:
        metadata = manifest.get("output_metadata")
        written = manifest.get("written_records")
        if (
            isinstance(metadata, dict)
            and metadata.get(str(path)) == _file_metadata(path)
            and isinstance(written, int)
            and written >= 0
        ):
            return written
    return count_jsonl_records(path)


@contextmanager
def _temporary_sqlite(path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a bounded-memory, disposable SQLite index and remove it afterwards."""

    path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (path, Path(f"{path}-journal"), Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)
    database = sqlite3.connect(path)
    try:
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("PRAGMA temp_store=FILE")
        database.execute("PRAGMA cache_size=-32768")
        yield database
        database.commit()
    finally:
        database.close()
        for candidate in (
            path,
            Path(f"{path}-journal"),
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        ):
            candidate.unlink(missing_ok=True)


def _sqlite_insert_unique(
    database: sqlite3.Connection,
    table: str,
    column: str,
    value: str,
) -> bool:
    try:
        database.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
    except sqlite3.IntegrityError:
        return False
    return True


@contextmanager
def atomic_jsonl_writer(path: Path) -> Iterator[TextIO]:
    """Yield an atomic UTF-8 JSONL output file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            yield file
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    """Atomically write a formatted UTF-8 JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _write_jsonl_record(file: TextIO, record: Mapping[str, Any]) -> None:
    json.dump(record, file, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    file.write("\n")


def _fingerprint(*values: object) -> str:
    encoded = json.dumps(
        values,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return _text_hash(encoded)


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _source_as_mapping(source: ApprovedSource) -> dict[str, Any]:
    return {
        "id": source.source_id,
        "path": str(source.path),
        "repository": source.repository,
        "approval": source.approval,
        "license": source.license,
        "include": list(source.include),
        "exclude": list(source.exclude),
    }


def _resolve(root: Path, value: object) -> Path:
    text = _required_string(value, "path")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetPipelineError(f"{name} must be a non-empty string.")
    return value.strip()


def _required_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetPipelineError(f"{name} must be a mapping.")
    return value


def _mapping_or_empty(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _required_mapping(value, name)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DatasetPipelineError(f"{name} must be a list of strings.")
    return tuple(item for item in value if item)


def _load_instruction_templates(value: object) -> tuple[tuple[str, str], ...]:
    templates = dict(DEFAULT_INSTRUCTION_TEMPLATES)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise DatasetPipelineError("instruction_generation.templates must be a mapping.")
    for category, template in value.items():
        if not isinstance(category, str) or category not in INSTRUCTION_CATEGORIES:
            raise DatasetPipelineError(f"Unsupported instruction template: {category!r}.")
        if not isinstance(template, str) or not template.strip():
            raise DatasetPipelineError(
                f"Instruction template {category!r} must be a non-empty string."
            )
        templates[category] = template.strip()
    return tuple((category, templates[category]) for category in INSTRUCTION_CATEGORIES)


def _validate_instruction_template(category: str, template: str) -> None:
    try:
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name is not None
        }
    except ValueError as exc:
        raise DatasetPipelineError(
            f"Invalid instruction template for {category!r}: {exc}"
        ) from exc
    unsupported = sorted(fields - TEMPLATE_FIELDS)
    if unsupported:
        raise DatasetPipelineError(
            f"Unsupported template fields for {category!r}: {', '.join(unsupported)}"
        )


def _require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required pipeline input not found: {path}. {hint}")


__all__ = [
    "ApprovedSource",
    "BuildResult",
    "CleaningSettings",
    "DatasetPipelineError",
    "JsonlValidationError",
    "PairGenerationSettings",
    "PipelineConfig",
    "PipelinePaths",
    "ProgressBar",
    "SplitSettings",
    "StageStatistics",
    "ValidationSettings",
    "build_dataset",
    "clean_python_dataset",
    "collect_python_data",
    "configure_pipeline_logging",
    "count_jsonl_records",
    "deduplicate_dataset",
    "format_stage_statistics",
    "generate_instruction_pairs",
    "instruction_pair_hash",
    "iter_jsonl",
    "load_pipeline_config",
    "run_stage_cli",
    "split_dataset",
    "validate_dataset",
    "validate_instruction_record",
]
