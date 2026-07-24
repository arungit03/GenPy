"""Corpus V2 pipeline orchestration for Phase 6.2."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from genpy_llm.corpus_v2.cleaner import CleanSettings, clean_document
from genpy_llm.corpus_v2.collector import DEFAULT_SKIP_DIRECTORIES, collect_documents
from genpy_llm.corpus_v2.deduplicator import DeduplicationSettings, Deduplicator
from genpy_llm.corpus_v2.manifest import (
    SourceSpec,
    TokenizedDocument,
    fingerprint_json,
    timestamp,
    write_jsonl,
)
from genpy_llm.corpus_v2.packer import PackingSettings, pack_documents
from genpy_llm.corpus_v2.quality import (
    QualitySettings,
    ReadinessSettings,
    evaluate_readiness,
    write_quality_reports,
)
from genpy_llm.corpus_v2.statistics import build_statistics, write_statistics_csv
from genpy_llm.corpus_v2.tokenizer import CorpusV2Tokenizer, TokenizationSettings
from genpy_llm.corpus_v2.validator import ValidationSettings, validate_document
from genpy_llm.shard_builder import final_outputs_valid

LOGGER = logging.getLogger("genpy_llm.corpus_v2")


class CorpusV2Error(RuntimeError):
    """Raised when Corpus V2 cannot continue."""


@dataclass(frozen=True)
class CorpusV2Paths:
    """Corpus V2 artifact paths."""

    output_directory: Path
    report_directory: Path
    document_manifest: Path
    log_file: Path


@dataclass(frozen=True)
class CorpusV2Config:
    """Complete Corpus V2 configuration."""

    config_path: Path
    project_root: Path
    sources: tuple[SourceSpec, ...]
    skip_directories: tuple[str, ...]
    clean: CleanSettings
    deduplication: DeduplicationSettings
    validation: ValidationSettings
    tokenization: TokenizationSettings
    packing: PackingSettings
    readiness: ReadinessSettings
    paths: CorpusV2Paths
    resume: bool
    log_level: str


@dataclass(frozen=True)
class CorpusV2Result:
    """Corpus V2 build result."""

    accepted_documents: int
    rejected_documents: int
    total_tokens: int
    estimated_token_count: int
    readiness_passed: bool
    readiness_failures: tuple[str, ...]
    duplicate_percentage: float
    manifest_path: Path
    shard_index_path: Path
    quality_report_json: Path
    quality_report_markdown: Path
    statistics_csv: Path
    resumed: bool


def load_corpus_v2_config(path: Path | str = "configs/corpus_v2.yaml") -> CorpusV2Config:
    """Load Corpus V2 YAML config."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Corpus V2 config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise CorpusV2Error(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise CorpusV2Error("Corpus V2 config must be a mapping.")
    root = _resolve(config_path.parent, raw.get("project_root", ".."))
    section = _mapping(raw.get("corpus_v2", {}), "corpus_v2")
    paths_raw = _mapping(section.get("paths", {}), "corpus_v2.paths")
    clean_raw = _mapping(section.get("cleaning", {}), "corpus_v2.cleaning")
    dedup_raw = _mapping(section.get("deduplication", {}), "corpus_v2.deduplication")
    quality_raw = _mapping(section.get("quality", {}), "corpus_v2.quality")
    validation_raw = _mapping(section.get("validation", {}), "corpus_v2.validation")
    token_raw = _mapping(section.get("tokenization", {}), "corpus_v2.tokenization")
    packing_raw = _mapping(section.get("packing", {}), "corpus_v2.packing")
    readiness_raw = _mapping(section.get("readiness", {}), "corpus_v2.readiness")
    report_dir = _resolve(root, paths_raw.get("report_directory", "reports/corpus_v2"))
    output_dir = _resolve(root, paths_raw.get("output_directory", "data/corpus_v2"))
    quality = QualitySettings(
        minimum_entropy=float(quality_raw.get("minimum_entropy", 2.5)),
        sample_characters=int(quality_raw.get("sample_characters", 200_000)),
        maximum_base64_fraction=float(quality_raw.get("maximum_base64_fraction", 0.35)),
        maximum_hex_fraction=float(quality_raw.get("maximum_hex_fraction", 0.35)),
        maximum_repeated_line_fraction=float(
            quality_raw.get("maximum_repeated_line_fraction", 0.35)
        ),
        minimum_technical_score=int(quality_raw.get("minimum_technical_score", 2)),
    )
    config = CorpusV2Config(
        config_path=config_path,
        project_root=root,
        sources=_load_sources(section.get("sources", ()), root),
        skip_directories=_string_tuple(
            section.get("skip_directories", DEFAULT_SKIP_DIRECTORIES),
            "corpus_v2.skip_directories",
        ),
        clean=CleanSettings(
            minimum_file_bytes=int(clean_raw.get("minimum_file_bytes", 80)),
            maximum_file_bytes=int(clean_raw.get("maximum_file_bytes", 2_000_000)),
            normalize_tabs=bool(clean_raw.get("normalize_tabs", True)),
            tab_width=int(clean_raw.get("tab_width", 4)),
        ),
        deduplication=DeduplicationSettings(
            exact=bool(dedup_raw.get("exact", True)),
            normalized=bool(dedup_raw.get("normalized", True)),
            near_duplicate=bool(dedup_raw.get("near_duplicate", True)),
            near_duplicate_threshold=float(dedup_raw.get("near_duplicate_threshold", 0.92)),
            shingle_size=int(dedup_raw.get("shingle_size", 5)),
            maximum_near_duplicate_index=int(
                dedup_raw.get("maximum_near_duplicate_index", 100_000)
            ),
        ),
        validation=ValidationSettings(
            require_python_syntax=bool(validation_raw.get("require_python_syntax", True)),
            quality=quality,
        ),
        tokenization=TokenizationSettings(
            tokenizer_path=_resolve(
                root,
                token_raw.get("tokenizer", "data/tokenizer/tokenizer.json"),
            ),
            minimum_tokens=int(token_raw.get("minimum_tokens", 4)),
        ),
        packing=PackingSettings(
            output_directory=output_dir,
            shard_prefix=str(packing_raw.get("shard_prefix", "corpus_v2")),
            context_length=int(packing_raw.get("context_length", 1024)),
            max_tokens_per_shard=int(packing_raw.get("max_tokens_per_shard", 10_000_000)),
            add_bos=bool(packing_raw.get("add_bos", False)),
            add_eos=bool(packing_raw.get("add_eos", True)),
            pad_final_sequence=bool(packing_raw.get("pad_final_sequence", False)),
        ),
        readiness=ReadinessSettings(
            minimum_tokens=int(readiness_raw.get("minimum_tokens", 200_000_000)),
            min_python_ratio=float(readiness_raw.get("min_python_ratio", 0.45)),
            max_python_ratio=float(readiness_raw.get("max_python_ratio", 0.75)),
            min_technical_text_ratio=float(
                readiness_raw.get("min_technical_text_ratio", 0.20)
            ),
            max_technical_text_ratio=float(
                readiness_raw.get("max_technical_text_ratio", 0.55)
            ),
            max_duplicate_percentage=float(
                readiness_raw.get("max_duplicate_percentage", 0.15)
            ),
        ),
        paths=CorpusV2Paths(
            output_directory=output_dir,
            report_directory=report_dir,
            document_manifest=_resolve(
                root,
                paths_raw.get("document_manifest", "data/corpus_v2/document_manifest.jsonl"),
            ),
            log_file=_resolve(root, paths_raw.get("log_file", "logs/corpus_v2.jsonl")),
        ),
        resume=bool(section.get("resume", True)),
        log_level=str(_mapping(raw.get("logging", {}), "logging").get("level", "INFO")).upper(),
    )
    _validate_config(config)
    return config


def run_corpus_v2_pipeline(config: CorpusV2Config, *, force: bool = False) -> CorpusV2Result:
    """Build the Corpus V2 manifest, shards, reports, and readiness decision."""

    tokenizer = CorpusV2Tokenizer(config.tokenization)
    pass_result = _process_documents(config, tokenizer)
    statistics = build_statistics(pass_result.documents)
    duplicate_report = pass_result.deduplicator.report()
    duplicate_percentage = float(duplicate_report["duplicate_percentage"])
    readiness = evaluate_readiness(
        statistics,
        duplicate_percentage=duplicate_percentage,
        validation_failures=0,
        settings=config.readiness,
    )
    manifest_records = [document.manifest_record() for document in pass_result.documents]
    build_fingerprint = fingerprint_json(
        {
            "config": _fingerprint_config(config, tokenizer.tokenizer_hash),
            "documents": [
                {
                    "stored_path": record["stored_path"],
                    "sha256": record["sha256"],
                    "normalized_sha256": record["normalized_sha256"],
                    "token_count": record["token_count"],
                }
                for record in manifest_records
            ],
        }
    )
    write_jsonl(config.paths.document_manifest, manifest_records)
    manifest_payload = _manifest_payload(
        config,
        tokenizer_hash=tokenizer.tokenizer_hash,
        statistics=statistics.to_json(),
        duplicate_report=duplicate_report,
        build_fingerprint=build_fingerprint,
        readiness_passed=readiness.passed,
        readiness_failures=readiness.failures,
    )
    config.paths.output_directory.mkdir(parents=True, exist_ok=True)
    resumed = (
        config.resume
        and not force
        and final_outputs_valid(
            config.packing.index_path,
            config.packing.statistics_path,
            build_fingerprint,
        )
    )
    if resumed:
        shard_index = json.loads(config.packing.index_path.read_text(encoding="utf-8"))
    else:
        pack_result = pack_documents(
            _iter_documents_for_packing(config, tokenizer),
            tokenizer=tokenizer.tokenizer,
            tokenizer_path=config.tokenization.tokenizer_path,
            tokenizer_hash=tokenizer.tokenizer_hash,
            source_manifest=config.paths.document_manifest,
            settings=config.packing,
            build_fingerprint=build_fingerprint,
            force=force,
        )
        shard_index = pack_result.shard_index
        resumed = pack_result.resumed
    manifest_payload["shards"] = {
        "index": str(config.packing.index_path),
        "sequence_count": shard_index.get("sequence_count", 0),
        "token_count": shard_index.get("token_count", 0),
        "shard_count": len(shard_index.get("shards") or []),
    }
    write_statistics_csv(
        config.paths.report_directory / "statistics.csv",
        statistics,
        duplicate_percentage,
    )
    quality_json, quality_md, report_manifest = write_quality_reports(
        output_directory=config.paths.report_directory,
        statistics=statistics,
        duplicate_report=duplicate_report,
        rejection_reasons=dict(sorted(pass_result.rejections.items())),
        readiness=readiness,
        readiness_settings=config.readiness,
        manifest_payload=manifest_payload,
    )
    return CorpusV2Result(
        accepted_documents=len(pass_result.documents),
        rejected_documents=sum(pass_result.rejections.values()),
        total_tokens=statistics.total_tokens,
        estimated_token_count=statistics.total_tokens,
        readiness_passed=readiness.passed,
        readiness_failures=readiness.failures,
        duplicate_percentage=duplicate_percentage,
        manifest_path=report_manifest,
        shard_index_path=config.packing.index_path,
        quality_report_json=quality_json,
        quality_report_markdown=quality_md,
        statistics_csv=config.paths.report_directory / "statistics.csv",
        resumed=resumed,
    )


def run_corpus_v2_cli(argv: Sequence[str] | None = None) -> int:
    """Run the Corpus V2 build command."""

    parser = argparse.ArgumentParser(description="Build Phase 6.2 Corpus V2 artifacts.")
    parser.add_argument("--config", type=Path, default=Path("configs/corpus_v2.yaml"))
    parser.add_argument("--force", action="store_true", help="Rebuild packed shards.")
    args = parser.parse_args(argv)
    try:
        config = load_corpus_v2_config(args.config)
        _configure_logging(config)
        result = run_corpus_v2_pipeline(config, force=args.force)
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Corpus V2 build failed")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


def run_corpus_v2_analysis_cli(argv: Sequence[str] | None = None) -> int:
    """Print a concise summary from existing Corpus V2 reports."""

    parser = argparse.ArgumentParser(description="Analyze existing Corpus V2 artifacts.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/corpus_v2"))
    args = parser.parse_args(argv)
    report = args.report_dir / "quality_report.json"
    if not report.is_file():
        print(f"Error: Corpus V2 quality report not found: {report}", file=sys.stderr)
        return 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    statistics = payload["statistics"]
    readiness = payload["readiness"]
    print("Corpus V2 analysis")
    print(f"Tokens: {statistics['total_tokens']}")
    print(f"Documents: {statistics['total_documents']}")
    print(f"Python ratio: {statistics['python_ratio']:.2%}")
    print(f"Technical text ratio: {statistics['technical_text_ratio']:.2%}")
    print(f"Duplicate percentage: {payload['duplicate_percentage']:.2%}")
    print(f"Readiness: {'PASS' if readiness['passed'] else 'FAIL'}")
    return 0


@dataclass
class _PassResult:
    documents: list[TokenizedDocument]
    rejections: Counter[str]
    deduplicator: Deduplicator


def _process_documents(config: CorpusV2Config, tokenizer: CorpusV2Tokenizer) -> _PassResult:
    documents: list[TokenizedDocument] = []
    rejections: Counter[str] = Counter()
    deduplicator = Deduplicator(config.deduplication)
    for tokenized in _iter_processed_documents(config, tokenizer, rejections, deduplicator):
        documents.append(replace(tokenized, token_ids=[]))
    return _PassResult(documents, rejections, deduplicator)


def _iter_documents_for_packing(
    config: CorpusV2Config,
    tokenizer: CorpusV2Tokenizer,
) -> Iterator[TokenizedDocument]:
    rejections: Counter[str] = Counter()
    deduplicator = Deduplicator(config.deduplication)
    yield from _iter_processed_documents(config, tokenizer, rejections, deduplicator)


def _iter_processed_documents(
    config: CorpusV2Config,
    tokenizer: CorpusV2Tokenizer,
    rejections: Counter[str],
    deduplicator: Deduplicator,
) -> Iterator[TokenizedDocument]:
    collected = collect_documents(
        config.sources,
        skip_directories=config.skip_directories,
        max_file_bytes=config.clean.maximum_file_bytes,
    )
    for raw_document in collected:
        cleaned = clean_document(raw_document, config.clean)
        if cleaned.document is None:
            rejections[cleaned.rejection_reason or "cleaning_rejected"] += 1
            continue
        validation = validate_document(cleaned.document, config.validation)
        if not validation.accepted:
            rejections[validation.reason] += 1
            continue
        decision = deduplicator.check(cleaned.document)
        if not decision.accepted:
            rejections[decision.reason] += 1
            continue
        tokenized = tokenizer.tokenize(
            cleaned.document,
            normalized_sha256=decision.normalized_sha256,
            quality=validation.quality,
        )
        if tokenized.document is None:
            rejections[tokenized.rejection_reason or "tokenization_rejected"] += 1
            continue
        yield tokenized.document


def _load_sources(value: object, root: Path) -> tuple[SourceSpec, ...]:
    if not isinstance(value, (list, tuple)):
        raise CorpusV2Error("corpus_v2.sources must be a list.")
    sources: list[SourceSpec] = []
    for index, item in enumerate(value, start=1):
        raw = _mapping(item, f"corpus_v2.sources[{index}]")
        if not bool(raw.get("enabled", True)):
            continue
        source_id = _required_string(raw.get("id"), f"source {index} id")
        source_type = _required_string(raw.get("type"), f"{source_id}.type")
        sources.append(
            SourceSpec(
                source_id=source_id,
                source_type=source_type,
                path=_resolve(root, raw.get("path") or raw.get("location")),
                include=_string_tuple(raw.get("include", ("**/*",)), f"{source_id}.include"),
                exclude=_string_tuple(raw.get("exclude", ()), f"{source_id}.exclude"),
                license=_optional_string(raw.get("license")),
                approval=_optional_string(raw.get("approval")),
            )
        )
    return tuple(sources)


def _manifest_payload(
    config: CorpusV2Config,
    *,
    tokenizer_hash: str,
    statistics: dict[str, Any],
    duplicate_report: dict[str, Any],
    build_fingerprint: str,
    readiness_passed: bool,
    readiness_failures: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "phase": "6.2",
        "created_at": timestamp(),
        "config": str(config.config_path),
        "document_manifest": str(config.paths.document_manifest),
        "tokenizer": str(config.tokenization.tokenizer_path),
        "tokenizer_sha256": tokenizer_hash,
        "vocabulary_retrained": False,
        "build_fingerprint": build_fingerprint,
        "statistics": statistics,
        "duplicate_report": duplicate_report,
        "readiness": {
            "passed": readiness_passed,
            "failures": list(readiness_failures),
        },
    }


def _fingerprint_config(config: CorpusV2Config, tokenizer_hash: str) -> dict[str, Any]:
    return {
        "sources": [
            {
                "id": source.source_id,
                "type": source.source_type,
                "path": str(source.path),
                "include": source.include,
                "exclude": source.exclude,
            }
            for source in config.sources
        ],
        "clean": config.clean.__dict__,
        "deduplication": config.deduplication.__dict__,
        "validation": {
            "require_python_syntax": config.validation.require_python_syntax,
            "quality": config.validation.quality.__dict__,
        },
        "packing": {
            "output_directory": str(config.packing.output_directory),
            "shard_prefix": config.packing.shard_prefix,
            "context_length": config.packing.context_length,
            "max_tokens_per_shard": config.packing.max_tokens_per_shard,
            "add_bos": config.packing.add_bos,
            "add_eos": config.packing.add_eos,
            "pad_final_sequence": config.packing.pad_final_sequence,
        },
        "tokenizer_hash": tokenizer_hash,
    }


def _configure_logging(config: CorpusV2Config) -> None:
    config.paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.paths.log_file, encoding="utf-8"),
        ],
        force=True,
    )


def _print_result(result: CorpusV2Result) -> None:
    print("Corpus V2 build complete")
    print(f"Accepted documents: {result.accepted_documents}")
    print(f"Rejected documents: {result.rejected_documents}")
    print(f"Estimated token count: {result.estimated_token_count}")
    print(f"Duplicate percentage: {result.duplicate_percentage:.2%}")
    print(f"Readiness: {'PASS' if result.readiness_passed else 'FAIL'}")
    print(f"Readiness failures: {', '.join(result.readiness_failures) or 'none'}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Shard index: {result.shard_index_path}")
    print(f"Quality JSON: {result.quality_report_json}")
    print(f"Quality report: {result.quality_report_markdown}")
    print(f"Statistics CSV: {result.statistics_csv}")


def _validate_config(config: CorpusV2Config) -> None:
    if not config.sources:
        raise CorpusV2Error("At least one Corpus V2 source is required.")
    if config.clean.minimum_file_bytes < 0:
        raise CorpusV2Error("minimum_file_bytes must be non-negative.")
    if config.clean.maximum_file_bytes <= config.clean.minimum_file_bytes:
        raise CorpusV2Error("maximum_file_bytes must exceed minimum_file_bytes.")
    if config.tokenization.minimum_tokens < 0:
        raise CorpusV2Error("minimum_tokens must be non-negative.")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusV2Error(f"{label} must be a mapping.")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise CorpusV2Error(f"{label} must be a list of non-empty strings.")
    return tuple(item.strip() for item in value)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusV2Error(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CorpusV2Error("Optional string value must be a string or null.")
    return value.strip() or None


def _resolve(root: Path, value: object) -> Path:
    text = _required_string(value, "path")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


__all__ = [
    "CorpusV2Config",
    "CorpusV2Error",
    "CorpusV2Result",
    "load_corpus_v2_config",
    "run_corpus_v2_analysis_cli",
    "run_corpus_v2_cli",
    "run_corpus_v2_pipeline",
]
