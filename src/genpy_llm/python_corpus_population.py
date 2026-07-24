"""Approved-source population and search interface for the GenPy Python corpus."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from genpy_llm.python_corpus_collector import configure_collector_logging
from genpy_llm.python_corpus_expansion import (
    CorpusExpansionConfig,
    CorpusExpansionResult,
    expand_python_corpus,
    load_corpus_expansion_config,
)

UTC = timezone.utc

LOGGER = logging.getLogger("genpy_llm.python_corpus_population")
POPULATION_VERSION = 1
POPULATION_SOURCE_TYPES = {"local", "git", "zip"}


class CorpusPopulationError(RuntimeError):
    """Raised when corpus population policy or index search cannot be satisfied."""


@dataclass(frozen=True)
class CorpusPopulationConfig:
    """Population configuration layered on the collector and expansion settings."""

    expansion: CorpusExpansionConfig
    report_path: Path


@dataclass(frozen=True)
class CorpusPopulationResult:
    """Consolidated import and indexed-corpus statistics."""

    report_path: Path
    index_path: Path
    python_files_imported: int
    python_files_unchanged: int
    total_python_files: int
    functions_discovered: int
    classes_discovered: int
    duplicate_files: int
    estimated_instruction_pairs: int
    categories: dict[str, int]
    expansion: CorpusExpansionResult


@dataclass(frozen=True)
class CorpusSearchResult:
    """One searchable file result from the SQLite corpus index."""

    stored_path: str
    source_path: str
    source_id: str
    source_type: str
    license: str | None
    primary_category: str
    categories: tuple[str, ...]
    function_count: int
    class_count: int
    matching_symbols: tuple[str, ...]


def load_corpus_population_config(
    path: Path | str = "configs/dataset_pipeline.yaml",
) -> CorpusPopulationConfig:
    """Load population settings and enforce local approved-source policy."""

    expansion = load_corpus_expansion_config(path)
    try:
        raw = yaml.safe_load(expansion.collector.config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - parsed by lower layers first
        raise CorpusPopulationError(
            f"Invalid YAML in {expansion.collector.config_path}: {exc}"
        ) from exc
    section = raw.get("corpus_population", {})
    if not isinstance(section, dict):
        raise CorpusPopulationError("corpus_population must be a YAML mapping.")
    report_path = _resolve(
        expansion.collector.project_root,
        section.get("report", "data/raw/corpus_population_report.json"),
    )
    try:
        report_path.relative_to(expansion.collector.output_directory.resolve())
    except ValueError as exc:
        raise CorpusPopulationError(
            f"Corpus population report must be under "
            f"{expansion.collector.output_directory}: {report_path}"
        ) from exc
    _validate_population_sources(expansion)
    return CorpusPopulationConfig(expansion=expansion, report_path=report_path)


def populate_python_corpus(config: CorpusPopulationConfig) -> CorpusPopulationResult:
    """Import all configured approved sources and rebuild the searchable index."""

    _validate_population_sources(config.expansion)
    started_at = _timestamp()
    expansion = expand_python_corpus(config.expansion, collect=True)
    collection = expansion.collection
    if collection is None:  # pragma: no cover - collect=True guarantees this
        raise CorpusPopulationError("Corpus population did not return collection statistics.")
    duplicate_files = collection.rejection_reasons.get("duplicate_content", 0)
    license_metadata = _summarize_license_metadata(expansion.index_path)
    report = {
        "population_version": POPULATION_VERSION,
        "started_at": started_at,
        "completed_at": _timestamp(),
        "configuration": str(config.expansion.collector.config_path),
        "approved_sources": [
            {
                "id": source.source_id,
                "type": source.source_type,
                "location": source.location,
                "license": source.license,
                "approval": source.approval,
                "revision": source.revision,
                "discovered_automatically": source.discovered_automatically,
            }
            for source in config.expansion.collector.sources
        ],
        "python_files_imported": collection.files_accepted,
        "python_files_unchanged": collection.files_unchanged,
        "python_files_rejected": collection.files_rejected,
        "total_python_files": expansion.total_python_files,
        # Explicit corpus-report names. Keep the established fields above for
        # backwards compatibility with existing tooling.
        "total_repositories": expansion.total_repositories,
        "python_files_scanned": collection.files_scanned,
        "accepted_files": expansion.total_python_files,
        "rejected_files": collection.files_rejected,
        "rejection_reasons": collection.rejection_reasons,
        "functions_discovered": expansion.total_functions,
        "classes_discovered": expansion.total_classes,
        "categories": expansion.category_files,
        "category_distribution": expansion.category_files,
        "license_metadata": license_metadata,
        "duplicate_files": duplicate_files,
        "estimated_instruction_pairs": expansion.estimated_instruction_pairs,
        "collection_rejection_reasons": collection.rejection_reasons,
        "index_rejected_records": expansion.rejected_index_records,
        "corpus_index": str(expansion.index_path),
        "provenance_manifest": str(config.expansion.collector.provenance_manifest),
    }
    _atomic_json_dump(report, config.report_path)
    LOGGER.info(
        "Corpus populated: imported=%s unchanged=%s total=%s functions=%s "
        "classes=%s duplicates=%s estimated_pairs=%s",
        collection.files_accepted,
        collection.files_unchanged,
        expansion.total_python_files,
        expansion.total_functions,
        expansion.total_classes,
        duplicate_files,
        expansion.estimated_instruction_pairs,
    )
    return CorpusPopulationResult(
        report_path=config.report_path,
        index_path=expansion.index_path,
        python_files_imported=collection.files_accepted,
        python_files_unchanged=collection.files_unchanged,
        total_python_files=expansion.total_python_files,
        functions_discovered=expansion.total_functions,
        classes_discovered=expansion.total_classes,
        duplicate_files=duplicate_files,
        estimated_instruction_pairs=expansion.estimated_instruction_pairs,
        categories=expansion.category_files,
        expansion=expansion,
    )


def _summarize_license_metadata(index_path: Path) -> dict[str, Any]:
    """Report declared and explicitly unspecified licenses for indexed files."""

    distribution: dict[str, int] = {}
    database = sqlite3.connect(index_path)
    try:
        rows = database.execute(
            """
            SELECT COALESCE(NULLIF(TRIM(license), ''), '<unspecified>'), COUNT(*)
            FROM files
            GROUP BY COALESCE(NULLIF(TRIM(license), ''), '<unspecified>')
            ORDER BY 1
            """
        )
        distribution = {str(license_name): int(count) for license_name, count in rows}
    finally:
        database.close()
    unspecified = distribution.get("<unspecified>", 0)
    return {
        "declared_files": sum(distribution.values()) - unspecified,
        "unspecified_files": unspecified,
        "distribution": distribution,
    }


def search_python_corpus(
    config: CorpusPopulationConfig,
    *,
    query: str = "",
    category: str | None = None,
    limit: int = 20,
) -> tuple[CorpusSearchResult, ...]:
    """Search indexed paths, sources, and qualified symbol names."""

    if limit <= 0 or limit > 1_000:
        raise CorpusPopulationError("Search limit must be between 1 and 1000.")
    if not config.expansion.index_path.is_file():
        raise CorpusPopulationError(
            f"Corpus index not found: {config.expansion.index_path}. Populate it first."
        )
    normalized_query = query.strip().casefold()
    normalized_category = category.strip() if isinstance(category, str) else None
    pattern = f"%{normalized_query}%"
    statement = """
        SELECT DISTINCT
            files.file_id, files.stored_path, files.source_path, files.source_id,
            files.source_type, files.license, files.primary_category,
            files.function_count, files.class_count
        FROM files
        WHERE (
            ? = ''
            OR LOWER(files.stored_path) LIKE ?
            OR LOWER(files.source_path) LIKE ?
            OR LOWER(files.source_id) LIKE ?
            OR EXISTS (
                SELECT 1 FROM symbols
                WHERE symbols.file_id = files.file_id
                  AND LOWER(symbols.qualified_name) LIKE ?
            )
        )
        AND (
            ? IS NULL
            OR EXISTS (
                SELECT 1 FROM file_categories
                WHERE file_categories.file_id = files.file_id
                  AND LOWER(file_categories.category) = LOWER(?)
            )
        )
        ORDER BY files.stored_path
        LIMIT ?
    """
    results: list[CorpusSearchResult] = []
    database = sqlite3.connect(config.expansion.index_path)
    try:
        rows = database.execute(
            statement,
            (
                normalized_query,
                pattern,
                pattern,
                pattern,
                pattern,
                normalized_category,
                normalized_category,
                limit,
            ),
        )
        for row in rows:
            file_id = int(row[0])
            categories = tuple(
                item[0]
                for item in database.execute(
                    "SELECT category FROM file_categories WHERE file_id = ? "
                    "ORDER BY category",
                    (file_id,),
                )
            )
            matching_symbols = tuple(
                item[0]
                for item in database.execute(
                    "SELECT qualified_name FROM symbols WHERE file_id = ? "
                    "AND (? = '' OR LOWER(qualified_name) LIKE ?) "
                    "ORDER BY qualified_name LIMIT 20",
                    (file_id, normalized_query, pattern),
                )
            )
            results.append(
                CorpusSearchResult(
                    stored_path=str(row[1]),
                    source_path=str(row[2]),
                    source_id=str(row[3]),
                    source_type=str(row[4]),
                    license=str(row[5]) if row[5] is not None else None,
                    primary_category=str(row[6]),
                    categories=categories,
                    function_count=int(row[7]),
                    class_count=int(row[8]),
                    matching_symbols=matching_symbols,
                )
            )
    finally:
        database.close()
    return tuple(results)


def run_corpus_population_cli(argv: list[str] | None = None) -> int:
    """Populate the corpus or search its current index."""

    parser = argparse.ArgumentParser(
        description="Populate and search the approved GenPy Python corpus."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_pipeline.yaml"),
        help="Dataset pipeline YAML containing corpus source settings.",
    )
    parser.add_argument("--search", default=None, help="Search path/source/symbol text.")
    parser.add_argument("--category", default=None, help="Restrict search to one category.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum search results.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    try:
        config = load_corpus_population_config(args.config)
        configure_collector_logging(config.expansion.collector, level=args.log_level)
        if args.search is not None or args.category is not None:
            results = search_python_corpus(
                config,
                query=args.search or "",
                category=args.category,
                limit=args.limit,
            )
            for result in results:
                print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
            print(f"Search results: {len(results)}")
        else:
            result = populate_python_corpus(config)
            print("Python corpus population complete")
            print(f"Index: {result.index_path}")
            print(f"Report: {result.report_path}")
            print(
                f"Imported={result.python_files_imported} "
                f"unchanged={result.python_files_unchanged} "
                f"total={result.total_python_files} "
                f"functions={result.functions_discovered} "
                f"classes={result.classes_discovered} "
                f"duplicates={result.duplicate_files} "
                f"estimated_pairs={result.estimated_instruction_pairs}"
            )
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Corpus population failed")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _validate_population_sources(expansion: CorpusExpansionConfig) -> None:
    for source in expansion.collector.sources:
        if source.source_type not in POPULATION_SOURCE_TYPES:
            raise CorpusPopulationError(
                f"Population source {source.source_id!r} uses unsupported type "
                f"{source.source_type!r}; expected local, git, or zip."
            )
        if not source.approval:
            raise CorpusPopulationError(
                f"Population source {source.source_id!r} must include a non-empty "
                "approval statement."
            )
        if source.source_type != "git":
            continue
        location = Path(source.location).expanduser()
        if not location.is_absolute():
            location = expansion.collector.project_root / location
        if not location.is_dir():
            raise CorpusPopulationError(
                f"Population Git source {source.source_id!r} must be a local repository: "
                f"{location.resolve()}"
            )


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CorpusPopulationError("Corpus population path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CorpusPopulationConfig",
    "CorpusPopulationError",
    "CorpusPopulationResult",
    "CorpusSearchResult",
    "load_corpus_population_config",
    "populate_python_corpus",
    "run_corpus_population_cli",
    "search_python_corpus",
]
