"""Phase 5.5C final corpus merge, deduplication, packing, and shard build."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from genpy_llm.code_filtering import CodeFilterSettings
from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash
from genpy_llm.global_deduplicator import GlobalDeduplicationConfig, GlobalDeduplicator
from genpy_llm.license_analyzer import build_license_report
from genpy_llm.logging_utils import setup_structured_logging
from genpy_llm.manifest_builder import (
    build_training_manifest,
    manifest_fingerprint,
    write_json,
    write_jsonl,
)
from genpy_llm.python_corpus_expansion import expand_python_corpus, load_corpus_expansion_config
from genpy_llm.python_dataset_pipeline import ProgressBar
from genpy_llm.sequence_packer import (
    SequencePacker,
    SequencePackingConfig,
    prepare_document_tokens,
)
from genpy_llm.shard_builder import (
    SequenceShardStatistics,
    SequenceShardWriter,
    final_outputs_valid,
    prepare_sequence_output,
    write_sequence_shard_index,
)
from genpy_llm.source_analyzer import build_source_report
from genpy_llm.statistics_builder import (
    build_corpus_report,
    build_quality_report,
    build_shard_statistics,
    build_token_statistics,
)
from genpy_llm.validation_report import (
    FinalValidationConfig,
    ValidatedCorpusRecord,
    ValidationReporter,
    validate_manifest_record,
)

LOGGER = logging.getLogger("genpy_llm.corpus_merger")
FINAL_CORPUS_VERSION = 1
_WORKER_TOKENIZER: CodeTokenizer | None = None


class CorpusMergeError(RuntimeError):
    """Raised when the final pretraining corpus cannot be built."""


@dataclass(frozen=True)
class CorpusMergePaths:
    """Configured Phase 5.5C paths."""

    source_manifest: Path
    corpus_root: Path
    corpus_index: Path | None
    output_directory: Path
    merged_manifest: Path
    index: Path
    manifest: Path
    statistics: Path
    report_directory: Path
    corpus_report: Path
    quality_report: Path
    duplicate_report: Path
    validation_report: Path
    license_report: Path
    source_report: Path
    report_statistics: Path
    token_statistics: Path
    shard_statistics: Path
    checkpoint: Path
    log_file: Path


@dataclass(frozen=True)
class TokenizationSettings:
    """Final corpus tokenization settings."""

    tokenizer_path: Path
    workers: int
    max_pending_tasks_per_worker: int


@dataclass(frozen=True)
class ShardSettings:
    """Final binary shard settings."""

    prefix: str
    max_tokens_per_shard: int
    compression: str


@dataclass(frozen=True)
class CorpusMergeConfig:
    """Validated YAML configuration for Phase 5.5C."""

    config_path: Path
    project_root: Path
    enabled: bool
    progress: bool
    source_types: tuple[str, ...]
    rebuild_corpus_index: bool
    dataset_pipeline_config: Path
    validation: FinalValidationConfig
    deduplication: GlobalDeduplicationConfig
    packing: SequencePackingConfig
    tokenization: TokenizationSettings
    shards: ShardSettings
    paths: CorpusMergePaths
    resume: bool
    log_level: str


@dataclass(frozen=True)
class CorpusMergeResult:
    """Summary returned by the final corpus builder."""

    manifest_path: Path
    index_path: Path
    statistics_path: Path
    accepted_files: int
    rejected_files: int
    duplicates_removed: int
    total_tokens: int
    training_sequences: int
    shard_count: int
    resumed: bool = False


@dataclass(frozen=True)
class TokenizedRecord:
    """A validated record with token IDs."""

    record: dict[str, Any]
    token_ids: list[int]
    token_count: int
    used_token_ids: set[int]


def load_pretraining_config(path: Path | str = "configs/pretraining.yaml") -> CorpusMergeConfig:
    """Load Phase 5.5C pretraining corpus configuration."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Pretraining config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise CorpusMergeError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CorpusMergeError("Pretraining config must be a YAML mapping.")
    project_root = _resolve(config_path.parent, raw.get("project_root", ".."))
    section = _mapping(raw.get("pretraining_corpus", {}), "pretraining_corpus")
    paths_raw = _mapping(section.get("paths", {}), "pretraining_corpus.paths")
    validation_raw = _mapping(section.get("validation", {}), "pretraining_corpus.validation")
    dedup_raw = _mapping(section.get("deduplication", {}), "pretraining_corpus.deduplication")
    token_raw = _mapping(section.get("tokenization", {}), "pretraining_corpus.tokenization")
    packing_raw = _mapping(section.get("packing", {}), "pretraining_corpus.packing")
    shard_raw = _mapping(section.get("shards", {}), "pretraining_corpus.shards")
    output_directory = _resolve(
        project_root,
        paths_raw.get("output_directory", "data/pretraining"),
    )
    report_directory = _resolve(
        project_root,
        paths_raw.get("report_directory", "reports/pretraining"),
    )
    source_manifest = _resolve(
        project_root,
        paths_raw.get("source_manifest", "data/raw/collection_manifest.jsonl"),
    )
    tokenizer_path = _resolve(
        project_root,
        token_raw.get("tokenizer", "data/tokenizer/tokenizer.json"),
    )
    cleaner_raw = _mapping(
        validation_raw.get("cleaner", {}),
        "pretraining_corpus.validation.cleaner",
    )
    workers = int(token_raw.get("workers", 0))
    if workers <= 0:
        workers = min(8, os.cpu_count() or 1)
    context_length = int(packing_raw.get("context_length", 1024))
    source_types = _string_tuple(
        section.get(
            "source_types",
            ("github", "pypi", "local", "git", "zip", "file", "manual_raw"),
        ),
        "pretraining_corpus.source_types",
    )
    paths = CorpusMergePaths(
        source_manifest=source_manifest,
        corpus_root=_resolve(project_root, paths_raw.get("corpus_root", "data/raw")),
        corpus_index=(
            _resolve(project_root, paths_raw["corpus_index"])
            if paths_raw.get("corpus_index") is not None
            else None
        ),
        output_directory=output_directory,
        merged_manifest=_resolve(
            project_root,
            paths_raw.get("merged_manifest", "data/pretraining/corpus_manifest.jsonl"),
        ),
        index=_resolve(project_root, paths_raw.get("index", "data/pretraining/index.json")),
        manifest=_resolve(
            project_root,
            paths_raw.get("manifest", "data/pretraining/manifest.json"),
        ),
        statistics=_resolve(
            project_root,
            paths_raw.get("statistics", "data/pretraining/statistics.json"),
        ),
        report_directory=report_directory,
        corpus_report=report_directory / "corpus_report.json",
        quality_report=report_directory / "quality_report.json",
        duplicate_report=report_directory / "duplicate_report.json",
        validation_report=report_directory / "validation_report.json",
        license_report=report_directory / "license_report.json",
        source_report=report_directory / "source_report.json",
        report_statistics=report_directory / "statistics.json",
        token_statistics=report_directory / "token_statistics.json",
        shard_statistics=report_directory / "shard_statistics.json",
        checkpoint=_resolve(
            project_root,
            paths_raw.get("checkpoint", "data/pretraining/checkpoint.json"),
        ),
        log_file=_resolve(project_root, paths_raw.get("log_file", "logs/pretraining_corpus.jsonl")),
    )
    config = CorpusMergeConfig(
        config_path=config_path,
        project_root=project_root,
        enabled=bool(section.get("enabled", True)),
        progress=bool(section.get("progress", raw.get("progress", True))),
        source_types=source_types,
        rebuild_corpus_index=bool(section.get("rebuild_corpus_index", False)),
        dataset_pipeline_config=_resolve(
            project_root,
            section.get("dataset_pipeline_config", "configs/dataset_pipeline.yaml"),
        ),
        validation=FinalValidationConfig(
            minimum_file_bytes=int(validation_raw.get("minimum_file_bytes", 1)),
            maximum_file_bytes=int(validation_raw.get("maximum_file_bytes", 250_000)),
            require_python_syntax=bool(validation_raw.get("require_python_syntax", True)),
            reject_generated=bool(validation_raw.get("reject_generated", True)),
            reject_vendor=bool(validation_raw.get("reject_vendor", True)),
            cleaner=CodeFilterSettings(
                minimum_file_bytes=int(validation_raw.get("minimum_file_bytes", 1)),
                maximum_file_bytes=int(validation_raw.get("maximum_file_bytes", 250_000)),
                accepted_licenses=_string_tuple(
                    cleaner_raw.get(
                        "accepted_licenses",
                        (
                            "MIT",
                            "Apache-2.0",
                            "BSD-2-Clause",
                            "BSD-3-Clause",
                            "ISC",
                            "Unlicense",
                            "MPL-2.0",
                        ),
                    ),
                    "pretraining_corpus.validation.cleaner.accepted_licenses",
                ),
                require_known_license=bool(cleaner_raw.get("require_known_license", False)),
            ),
        ),
        deduplication=GlobalDeduplicationConfig(
            exact_sha256=bool(dedup_raw.get("exact_sha256", True)),
            whitespace_normalization=bool(dedup_raw.get("whitespace_normalization", True)),
            comment_normalization=bool(dedup_raw.get("comment_normalization", True)),
            newline_normalization=bool(dedup_raw.get("newline_normalization", True)),
            ast_normalization=bool(dedup_raw.get("ast_normalization", False)),
            include_duplicate_groups=bool(dedup_raw.get("include_duplicate_groups", True)),
            maximum_duplicate_groups=int(dedup_raw.get("maximum_duplicate_groups", 1000)),
        ),
        packing=SequencePackingConfig(
            context_length=context_length,
            add_bos=bool(packing_raw.get("add_bos", False)),
            add_eos=bool(packing_raw.get("add_eos", True)),
            document_boundary=str(packing_raw.get("document_boundary", "eos")),
            pad_final_sequence=bool(packing_raw.get("pad_final_sequence", False)),
        ),
        tokenization=TokenizationSettings(
            tokenizer_path=tokenizer_path,
            workers=workers,
            max_pending_tasks_per_worker=int(token_raw.get("max_pending_tasks_per_worker", 4)),
        ),
        shards=ShardSettings(
            prefix=_required_string(
                shard_raw.get("prefix", "shard"),
                "pretraining_corpus.shards.prefix",
            ),
            max_tokens_per_shard=int(shard_raw.get("max_tokens_per_shard", 10_000_000)),
            compression=str(shard_raw.get("compression", "metadata_gzip")),
        ),
        paths=paths,
        resume=bool(section.get("resume", True)),
        log_level=str(_mapping(raw.get("logging", {}), "logging").get("level", "INFO")).upper(),
    )
    _validate_config(config)
    return config


def build_pretraining_corpus(
    config: CorpusMergeConfig,
    *,
    force: bool = False,
) -> CorpusMergeResult:
    """Build the final merged pretraining corpus for Phase 6."""

    if not config.enabled:
        raise CorpusMergeError("pretraining_corpus.enabled is false.")
    if not config.tokenization.tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Existing GenPy tokenizer not found: {config.tokenization.tokenizer_path}. "
            "Phase 5.5C never retrains it."
        )
    started_at = _timestamp()
    config.paths.output_directory.mkdir(parents=True, exist_ok=True)
    config.paths.report_directory.mkdir(parents=True, exist_ok=True)
    if config.rebuild_corpus_index:
        expansion = load_corpus_expansion_config(config.dataset_pipeline_config)
        expand_python_corpus(expansion, collect=False)
    category_index = _load_corpus_index(config.paths.corpus_index)
    source_records = list(_iter_manifest(config.paths.source_manifest, set(config.source_types)))
    source_records.sort(key=_record_sort_key)
    validator = ValidationReporter()
    deduplicator = GlobalDeduplicator(config.deduplication)
    accepted: list[ValidatedCorpusRecord] = []
    bar = ProgressBar("merge", len(source_records), enabled=config.progress)
    try:
        for index, record in enumerate(source_records, start=1):
            validated, reason = validate_manifest_record(
                record,
                corpus_root=config.paths.corpus_root,
                config=config.validation,
            )
            if validated is None:
                validator.reject(record, str(reason))
                bar.update(index)
                continue
            duplicate = deduplicator.check(record, validated.text)
            if duplicate is not None:
                validator.reject(record, duplicate.reason)
                bar.update(index)
                continue
            validator.accept()
            accepted.append(validated)
            bar.update(index)
    finally:
        bar.close()
    enriched = [
        _enrich_record(item, category_index.get(str(item.provenance.get("stored_path"))))
        for item in accepted
    ]
    tokenizer_hash = tokenizer_file_hash(config.tokenization.tokenizer_path)
    build_options = {
        "tokenizer_sha256": tokenizer_hash,
        "context_length": config.packing.context_length,
        "packing": config.packing.__dict__,
        "deduplication": config.deduplication.__dict__,
        "source_types": config.source_types,
    }
    fingerprint = manifest_fingerprint(enriched, build_options)
    resumed = config.resume and not force and final_outputs_valid(
        config.paths.index,
        config.paths.statistics,
        fingerprint,
    )
    if resumed:
        shard_index = json.loads(config.paths.index.read_text(encoding="utf-8"))
        token_statistics = json.loads(config.paths.token_statistics.read_text(encoding="utf-8"))
        manifest_records = _read_jsonl(config.paths.merged_manifest)
        if manifest_records:
            enriched = manifest_records
    else:
        output_artifacts = [
            config.paths.index,
            config.paths.manifest,
            config.paths.statistics,
            config.paths.merged_manifest,
        ]
        prepare_sequence_output(
            config.paths.output_directory,
            config.shards.prefix,
            output_artifacts,
        )
        shard_stats, shard_index, _statistics, token_statistics, enriched = _tokenize_pack_shard(
            config,
            enriched,
            tokenizer_hash=tokenizer_hash,
            build_fingerprint=fingerprint,
        )
        _write_checkpoint(
            config.paths.checkpoint,
            {
                "stage": "complete",
                "build_fingerprint": fingerprint,
                "sequence_count": shard_stats.sequence_count,
                "token_count": shard_stats.token_count,
                "completed_at": _timestamp(),
            },
        )
    training_statistics = json.loads(config.paths.statistics.read_text(encoding="utf-8"))
    completed_at = _timestamp()
    validation_payload = validator.report()
    duplicate_payload = deduplicator.report()
    source_payload = build_source_report(enriched)
    license_payload = build_license_report(enriched)
    corpus_payload = build_corpus_report(
        started_at=started_at,
        completed_at=completed_at,
        input_files=len(source_records),
        accepted_files=len(enriched),
        rejected_files=validator.rejected,
        duplicates_removed=int(duplicate_payload["duplicate_count"]),
        final_files=len(enriched),
        total_tokens=int(token_statistics.get("total_tokens") or 0),
        total_sequences=int(shard_index.get("sequence_count") or 0),
        shard_count=len(shard_index.get("shards") or []),
    )
    manifest_payload = build_training_manifest(
        corpus_version=FINAL_CORPUS_VERSION,
        creation_date=completed_at,
        tokenizer_path=config.tokenization.tokenizer_path,
        tokenizer_hash=tokenizer_hash,
        repositories=int(source_payload["total_repositories"]),
        packages=int(source_payload["total_packages"]),
        accepted_files=len(enriched),
        rejected_files=validator.rejected,
        duplicates_removed=int(duplicate_payload["duplicate_count"]),
        total_files=len(source_records),
        total_tokens=int(token_statistics.get("total_tokens") or 0),
        context_length=config.packing.context_length,
        shard_index=shard_index,
        source_manifest=config.paths.source_manifest,
        merged_manifest=config.paths.merged_manifest,
        build_fingerprint=fingerprint,
    )
    write_jsonl(config.paths.merged_manifest, enriched)
    write_json(config.paths.manifest, manifest_payload)
    write_json(config.paths.corpus_report, corpus_payload)
    write_json(config.paths.validation_report, validation_payload)
    write_json(config.paths.duplicate_report, duplicate_payload)
    write_json(config.paths.source_report, source_payload)
    write_json(config.paths.license_report, license_payload)
    write_json(config.paths.report_statistics, training_statistics)
    write_json(
        config.paths.quality_report,
        build_quality_report(
            enriched,
            validation_report=validation_payload,
            duplicate_report=duplicate_payload,
        ),
    )
    write_json(config.paths.shard_statistics, build_shard_statistics(shard_index))
    LOGGER.info(
        "pretraining_corpus_complete files=%d tokens=%d sequences=%d shards=%d resumed=%s",
        len(enriched),
        int(token_statistics.get("total_tokens") or 0),
        int(shard_index.get("sequence_count") or 0),
        len(shard_index.get("shards") or []),
        resumed,
    )
    return CorpusMergeResult(
        manifest_path=config.paths.manifest,
        index_path=config.paths.index,
        statistics_path=config.paths.statistics,
        accepted_files=len(enriched),
        rejected_files=validator.rejected,
        duplicates_removed=int(duplicate_payload["duplicate_count"]),
        total_tokens=int(token_statistics.get("total_tokens") or 0),
        training_sequences=int(shard_index.get("sequence_count") or 0),
        shard_count=len(shard_index.get("shards") or []),
        resumed=resumed,
    )


def run_pretraining_corpus_cli(argv: Sequence[str] | None = None) -> int:
    """Run the final corpus preparation command-line interface."""

    parser = argparse.ArgumentParser(
        description="Merge, deduplicate, tokenize, pack, and shard the GenPy corpus."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretraining.yaml"),
        help="Phase 5.5C YAML configuration.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild final shards.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    try:
        config = load_pretraining_config(args.config)
        setup_structured_logging(config.paths.log_file, args.log_level or config.log_level)
        result = build_pretraining_corpus(config, force=args.force)
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Final pretraining corpus build failed")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Final pretraining corpus ready")
    print(f"Manifest: {result.manifest_path}")
    print(f"Index: {result.index_path}")
    print(f"Statistics: {result.statistics_path}")
    print(
        f"Files={result.accepted_files} rejected={result.rejected_files} "
        f"duplicates={result.duplicates_removed} tokens={result.total_tokens} "
        f"sequences={result.training_sequences} shards={result.shard_count} "
        f"resumed={result.resumed}"
    )
    return 0


def _tokenize_pack_shard(
    config: CorpusMergeConfig,
    records: list[dict[str, Any]],
    *,
    tokenizer_hash: str,
    build_fingerprint: str,
) -> tuple[
    SequenceShardStatistics,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    tokenizer = CodeTokenizer.from_file(config.tokenization.tokenizer_path)
    if tokenizer.vocab_size > 65_536:
        raise CorpusMergeError("uint16 final shards require a vocabulary <= 65,536.")
    packer = SequencePacker(config.packing, pad_token_id=tokenizer.pad_token_id)
    writer = SequenceShardWriter(
        config.paths.output_directory,
        max_tokens_per_shard=config.shards.max_tokens_per_shard,
        context_length=config.packing.context_length,
        prefix=config.shards.prefix,
    )
    token_counts: list[int] = []
    used_token_ids: set[int] = set()
    tokenized_records: list[dict[str, Any]] = []
    bar = ProgressBar("tokenize-pack", len(records), enabled=config.progress)
    try:
        for index, tokenized in enumerate(
            _ordered_tokenized_records(records, tokenizer, config),
            start=1,
        ):
            record = tokenized.record
            prepared = prepare_document_tokens(
                tokenized.token_ids,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                config=config.packing,
            )
            token_count = len(prepared)
            record["token_count"] = token_count
            token_counts.append(token_count)
            used_token_ids.update(prepared)
            for sequence in packer.add_document(prepared, _sequence_metadata(record)):
                writer.write_sequence(sequence)
            tokenized_records.append(record)
            bar.update(index)
        for sequence in packer.finish():
            writer.write_sequence(sequence)
        shard_statistics = writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        bar.close()
    created_at = _timestamp()
    shard_index = write_sequence_shard_index(
        config.paths.index,
        shard_statistics,
        tokenizer_path=config.tokenization.tokenizer_path,
        tokenizer_sha256=tokenizer_hash,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        context_length=config.packing.context_length,
        source_manifest=config.paths.merged_manifest,
        creation_timestamp=created_at,
        build_fingerprint=build_fingerprint,
    )
    token_statistics = build_token_statistics(
        token_counts,
        used_token_ids=used_token_ids,
        vocab_size=tokenizer.vocab_size,
    )
    token_statistics.update(
        {
            "creation_timestamp": created_at,
            "build_fingerprint": build_fingerprint,
            "tokenizer_sha256": tokenizer_hash,
        }
    )
    statistics = {
        "format_version": 1,
        "creation_timestamp": created_at,
        "build_fingerprint": build_fingerprint,
        "corpus_files": len(tokenized_records),
        "total_tokens": token_statistics["total_tokens"],
        "training_sequences": shard_statistics.sequence_count,
        "context_length": config.packing.context_length,
        "sequence_length": config.packing.sequence_length,
        "shard_count": len(shard_statistics.shards),
        "byte_count": shard_statistics.byte_count,
    }
    write_json(config.paths.statistics, statistics)
    write_json(config.paths.token_statistics, token_statistics)
    return shard_statistics, shard_index, statistics, token_statistics, tokenized_records


def _ordered_tokenized_records(
    records: list[dict[str, Any]],
    tokenizer: CodeTokenizer,
    config: CorpusMergeConfig,
) -> Iterator[TokenizedRecord]:
    if config.tokenization.workers == 1:
        for record in records:
            yield _tokenize_record(record, tokenizer)
        return
    limit = config.tokenization.workers * config.tokenization.max_pending_tasks_per_worker
    iterator = iter(records)
    pending: list[Future[TokenizedRecord]] = []
    with ProcessPoolExecutor(
        max_workers=config.tokenization.workers,
        initializer=_initialize_tokenizer_worker,
        initargs=(str(config.tokenization.tokenizer_path),),
    ) as executor:
        for _ in range(limit):
            record = next(iterator, None)
            if record is None:
                break
            pending.append(executor.submit(_worker_tokenize_record, record))
        while pending:
            yield pending.pop(0).result()
            record = next(iterator, None)
            if record is not None:
                pending.append(executor.submit(_worker_tokenize_record, record))


def _initialize_tokenizer_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = CodeTokenizer.from_file(tokenizer_path)


def _worker_tokenize_record(record: dict[str, Any]) -> TokenizedRecord:
    if _WORKER_TOKENIZER is None:  # pragma: no cover
        raise CorpusMergeError("Tokenizer worker was not initialized.")
    return _tokenize_record(record, _WORKER_TOKENIZER)


def _tokenize_record(record: dict[str, Any], tokenizer: CodeTokenizer) -> TokenizedRecord:
    text = record.pop("_text")
    token_ids = tokenizer.encode(text)
    return TokenizedRecord(
        record=record,
        token_ids=token_ids,
        token_count=len(token_ids),
        used_token_ids=set(token_ids),
    )


def _enrich_record(
    item: ValidatedCorpusRecord,
    index_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    provenance = dict(item.provenance)
    source = provenance.get("source")
    source = dict(source) if isinstance(source, Mapping) else {}
    origin_url = (
        source.get("download_url")
        or source.get("repository_url")
        or source.get("location")
    )
    record = {
        **provenance,
        "repository": source.get("repository_url"),
        "package": source.get("package"),
        "source": source,
        "origin_url": origin_url,
        "relative_path": provenance.get("stored_path"),
        "sha256": provenance.get("content_sha256"),
        "size": item.byte_size,
        "size_bytes": item.byte_size,
        "line_count": item.line_count,
        "language": "Python",
        "validation_status": "accepted",
        "_text": item.text,
    }
    if index_metadata:
        record.update(index_metadata)
    return record


def _sequence_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record.get("source")
    source = source if isinstance(source, Mapping) else {}
    return {
        "stored_path": record.get("stored_path"),
        "content_sha256": record.get("content_sha256"),
        "source_type": source.get("type"),
        "source_id": source.get("id"),
        "repository": source.get("repository_url"),
        "package": source.get("package"),
    }


def _load_corpus_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    try:
        with sqlite3.connect(path) as database:
            rows = database.execute(
                """
                SELECT stored_path, primary_category, function_count, class_count,
                       estimated_instruction_pairs
                FROM files
                ORDER BY stored_path
                """
            ).fetchall()
            categories = database.execute(
                """
                SELECT files.stored_path, file_categories.category
                FROM file_categories
                JOIN files ON files.file_id = file_categories.file_id
                ORDER BY files.stored_path, file_categories.category
                """
            ).fetchall()
    except sqlite3.Error:
        LOGGER.warning("Could not read corpus index metadata from %s", path)
        return {}
    for stored_path, primary, functions, classes, estimated_pairs in rows:
        records[str(stored_path)] = {
            "primary_category": primary,
            "function_count": int(functions),
            "class_count": int(classes),
            "estimated_instruction_pairs": int(estimated_pairs),
            "categories": [],
        }
    for stored_path, category in categories:
        if str(stored_path) in records:
            records[str(stored_path)]["categories"].append(category)
    return records


def _iter_manifest(path: Path, source_types: set[str]) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise CorpusMergeError(f"Corpus provenance manifest not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusMergeError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise CorpusMergeError(f"Manifest record {path}:{line_number} is not an object.")
            source = record.get("source")
            source_type = source.get("type") if isinstance(source, Mapping) else None
            if source_type in source_types:
                yield record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    source = record.get("source")
    source = source if isinstance(source, Mapping) else {}
    return (
        str(source.get("type") or ""),
        str(source.get("id") or ""),
        str(record.get("stored_path") or ""),
    )


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def _validate_config(config: CorpusMergeConfig) -> None:
    if config.tokenization.max_pending_tasks_per_worker <= 0:
        raise CorpusMergeError("max_pending_tasks_per_worker must be positive.")
    if config.shards.max_tokens_per_shard <= 0:
        raise CorpusMergeError("max_tokens_per_shard must be positive.")
    if config.shards.compression != "metadata_gzip":
        raise CorpusMergeError("Only metadata_gzip compression is currently supported.")
    for artifact in (
        config.paths.index,
        config.paths.manifest,
        config.paths.statistics,
        config.paths.merged_manifest,
    ):
        try:
            artifact.resolve().relative_to(config.paths.output_directory.resolve())
        except ValueError as exc:
            raise CorpusMergeError(
                f"Final corpus output artifact must be under {config.paths.output_directory}: "
                f"{artifact}"
            ) from exc


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CorpusMergeError(f"{name} must be a mapping.")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable):
        raise CorpusMergeError(f"{name} must be a string or list of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise CorpusMergeError(f"{name} must contain only strings.")
        result.append(item)
    return tuple(result)


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusMergeError(f"{name} must be a non-empty string.")
    return value


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CorpusMergeError("Configured path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CorpusMergeConfig",
    "CorpusMergeError",
    "CorpusMergeResult",
    "CorpusMergePaths",
    "build_pretraining_corpus",
    "load_pretraining_config",
    "run_pretraining_corpus_cli",
]
