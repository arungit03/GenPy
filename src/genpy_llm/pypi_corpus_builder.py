"""Production PyPI sdist discovery, ingestion, and binary shard pipeline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import stat
import tarfile
import tempfile
import time
import tokenize
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from genpy_llm.code_filtering import CodeFilterSettings, filter_code_record
from genpy_llm.code_tokenizer import tokenizer_file_hash
from genpy_llm.corpus_tokenization import (
    CorpusTokenShardConfig,
    atomic_json,
    binary_outputs_valid,
    build_manifest_token_shards,
    prepare_binary_output,
    stable_manifest_fingerprint,
)
from genpy_llm.logging_utils import setup_structured_logging
from genpy_llm.python_corpus_collector import (
    CollectionResult,
    CorpusCollectorConfig,
    CorpusSource,
    SourceCandidate,
    collect_python_corpus,
    load_corpus_collector_config,
)
from genpy_llm.python_corpus_expansion import (
    CorpusExpansionConfig,
    expand_python_corpus,
    load_corpus_expansion_config,
)
from genpy_llm.python_dataset_pipeline import ProgressBar

LOGGER = logging.getLogger("genpy_llm.pypi_corpus_builder")
PYPI_CORPUS_VERSION = 1
SUPPORTED_ARCHIVES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip")
CANONICAL_NAME = re.compile(r"[-_.]+")
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[.*?\])?")


class PyPICorpusError(RuntimeError):
    """Raised when the PyPI corpus cannot be built safely."""


@dataclass(frozen=True)
class PyPIAPISettings:
    base_url: str
    simple_index_url: str
    top_packages_url: str
    request_timeout_seconds: int
    retries: int
    retry_backoff_seconds: float
    user_agent: str


@dataclass(frozen=True)
class PyPISelectionSettings:
    top_downloaded: bool
    top_downloaded_limit: int
    minimum_downloads: int
    keywords: tuple[str, ...]
    keyword_scan_limit: int
    categories: Mapping[str, tuple[str, ...]]
    enabled_categories: tuple[str, ...]
    requirements_files: tuple[Path, ...]
    manual_packages: tuple[str, ...]
    maximum_packages: int
    ignored_licenses: tuple[str, ...]


@dataclass(frozen=True)
class PyPIDownloadSettings:
    directory: Path
    workers: int
    retries: int
    timeout_seconds: int
    retry_backoff_seconds: float
    resume: bool


@dataclass(frozen=True)
class PyPIExtractionSettings:
    directory: Path
    workers: int
    ignored_directories: tuple[str, ...]
    ignore_migrations: bool
    maximum_members: int
    maximum_expanded_bytes: int


@dataclass(frozen=True)
class PyPIDeduplicationSettings:
    normalized: bool
    near_duplicate: bool
    near_duplicate_distance: int
    index_path: Path


@dataclass(frozen=True)
class PyPITokenSettings:
    tokenizer_path: Path
    output_directory: Path
    shard_index_path: Path
    statistics_path: Path
    document_index_filename: str
    shard_prefix: str
    max_tokens_per_shard: int
    workers: int
    max_pending_tasks_per_worker: int


@dataclass(frozen=True)
class PyPICorpusPaths:
    checkpoint: Path
    report_directory: Path
    pypi_report: Path
    package_statistics: Path
    license_report: Path
    quality_report: Path
    duplicate_report: Path
    token_statistics: Path
    log_file: Path


@dataclass(frozen=True)
class PyPICorpusConfig:
    config_path: Path
    project_root: Path
    enabled: bool
    approval: str
    api: PyPIAPISettings
    selection: PyPISelectionSettings
    download: PyPIDownloadSettings
    extraction: PyPIExtractionSettings
    cleaner_enabled: bool
    cleaner: CodeFilterSettings
    deduplication: PyPIDeduplicationSettings
    tokens: PyPITokenSettings
    paths: PyPICorpusPaths
    collector: CorpusCollectorConfig
    corpus_manager: CorpusExpansionConfig
    progress: bool
    log_level: str


@dataclass(frozen=True)
class PyPIPackage:
    name: str
    canonical_name: str
    version: str
    release_date: str | None
    homepage: str | None
    project_url: str
    repository_url: str | None
    author: str | None
    license: str
    summary: str | None
    keywords: str | None
    download_url: str
    filename: str
    sha256: str
    download_count: int | None = None


@dataclass(frozen=True)
class DownloadedSdist:
    package: PyPIPackage
    archive_path: Path
    resumed: bool


@dataclass(frozen=True)
class ExtractedSdist:
    downloaded: DownloadedSdist
    extraction_path: Path
    python_files: int


@dataclass(frozen=True)
class PyPICorpusResult:
    packages_discovered: int
    packages_downloaded: int
    packages_failed: int
    collection: CollectionResult
    documents: int
    token_count: int
    shard_count: int
    shard_index_path: Path
    statistics_path: Path
    resumed: bool = False


def load_pypi_corpus_config(path: Path | str = "configs/pypi.yaml") -> PyPICorpusConfig:
    """Load and validate the standalone PyPI YAML using collector path semantics."""

    collector = load_corpus_collector_config(path)
    corpus_manager = load_corpus_expansion_config(path)
    try:
        raw: Any = yaml.safe_load(collector.config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - collector parses first
        raise PyPICorpusError(f"Invalid YAML in {collector.config_path}: {exc}") from exc
    section = _mapping(raw.get("pypi_corpus", {}), "pypi_corpus")
    api_raw = _mapping(section.get("api", {}), "pypi_corpus.api")
    selection_raw = _mapping(section.get("selection", {}), "pypi_corpus.selection")
    download_raw = _mapping(section.get("download", {}), "pypi_corpus.download")
    extraction_raw = _mapping(section.get("extraction", {}), "pypi_corpus.extraction")
    cleaner_raw = _mapping(section.get("cleaner", {}), "pypi_corpus.cleaner")
    dedup_raw = _mapping(section.get("deduplication", {}), "pypi_corpus.deduplication")
    tokens_raw = _mapping(section.get("tokenization", {}), "pypi_corpus.tokenization")
    paths_raw = _mapping(section.get("paths", {}), "pypi_corpus.paths")
    categories_raw = _mapping(selection_raw.get("categories", {}), "selection.categories")
    categories = {
        str(name): _string_tuple(values, f"selection.categories.{name}", allow_empty=True)
        for name, values in categories_raw.items()
    }
    requirements = tuple(
        _resolve(collector.project_root, item)
        for item in _string_tuple(
            selection_raw.get("requirements_files", []),
            "selection.requirements_files",
            allow_empty=True,
        )
    )
    output_directory = _resolve(
        collector.project_root,
        tokens_raw.get("output_directory", "data/pretraining"),
    )
    report_directory = _resolve(
        collector.project_root,
        paths_raw.get("report_directory", "reports"),
    )
    config = PyPICorpusConfig(
        config_path=collector.config_path,
        project_root=collector.project_root,
        enabled=bool(section.get("enabled", False)),
        approval=_required_string(
            section.get("approval", "Approved by configured PyPI source policy"),
            "pypi_corpus.approval",
        ),
        api=PyPIAPISettings(
            base_url=_required_string(
                api_raw.get("base_url", "https://pypi.org/pypi"), "api.base_url"
            ).rstrip("/"),
            simple_index_url=_required_string(
                api_raw.get("simple_index_url", "https://pypi.org/simple/"),
                "api.simple_index_url",
            ),
            top_packages_url=_required_string(
                api_raw.get(
                    "top_packages_url",
                    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json",
                ),
                "api.top_packages_url",
            ),
            request_timeout_seconds=int(api_raw.get("request_timeout_seconds", 30)),
            retries=int(api_raw.get("retries", 4)),
            retry_backoff_seconds=float(api_raw.get("retry_backoff_seconds", 2.0)),
            user_agent=_required_string(
                api_raw.get("user_agent", "GenPy-PyPI-Corpus-Builder/1"),
                "api.user_agent",
            ),
        ),
        selection=PyPISelectionSettings(
            top_downloaded=bool(selection_raw.get("top_downloaded", False)),
            top_downloaded_limit=int(selection_raw.get("top_downloaded_limit", 1_000)),
            minimum_downloads=int(selection_raw.get("minimum_downloads", 0)),
            keywords=_string_tuple(
                selection_raw.get("keywords", []), "selection.keywords", allow_empty=True
            ),
            keyword_scan_limit=int(selection_raw.get("keyword_scan_limit", 2_000)),
            categories=categories,
            enabled_categories=_string_tuple(
                selection_raw.get("enabled_categories", []),
                "selection.enabled_categories",
                allow_empty=True,
            ),
            requirements_files=requirements,
            manual_packages=_string_tuple(
                selection_raw.get("manual_packages", []),
                "selection.manual_packages",
                allow_empty=True,
            ),
            maximum_packages=int(selection_raw.get("maximum_packages", 10_000)),
            ignored_licenses=_string_tuple(
                selection_raw.get("ignored_licenses", []),
                "selection.ignored_licenses",
                allow_empty=True,
            ),
        ),
        download=PyPIDownloadSettings(
            directory=_resolve(
                collector.project_root,
                download_raw.get("directory", "data/pypi/downloads"),
            ),
            workers=_worker_count(int(download_raw.get("workers", 8))),
            retries=int(download_raw.get("retries", 4)),
            timeout_seconds=int(download_raw.get("timeout_seconds", 120)),
            retry_backoff_seconds=float(download_raw.get("retry_backoff_seconds", 2.0)),
            resume=bool(download_raw.get("resume", True)),
        ),
        extraction=PyPIExtractionSettings(
            directory=_resolve(
                collector.project_root,
                extraction_raw.get("directory", "data/pypi/extraction"),
            ),
            workers=_worker_count(int(extraction_raw.get("workers", 8))),
            ignored_directories=_string_tuple(
                extraction_raw.get(
                    "ignored_directories",
                    [
                        "tests", "test", "docs", "examples", "demo", "benchmarks",
                        "build", "dist", "node_modules", "__pycache__", "venv", ".venv",
                        "vendor", "vendors", "vendored", "third_party", "external",
                    ],
                ),
                "extraction.ignored_directories",
            ),
            ignore_migrations=bool(extraction_raw.get("ignore_migrations", True)),
            maximum_members=int(extraction_raw.get("maximum_members", 200_000)),
            maximum_expanded_bytes=int(
                extraction_raw.get("maximum_expanded_bytes", 2_000_000_000)
            ),
        ),
        cleaner_enabled=bool(cleaner_raw.get("enabled", True)),
        cleaner=CodeFilterSettings(
            minimum_file_bytes=collector.minimum_file_bytes,
            maximum_file_bytes=collector.maximum_file_bytes,
            accepted_licenses=_string_tuple(
                cleaner_raw.get(
                    "accepted_licenses", list(CodeFilterSettings().accepted_licenses)
                ),
                "cleaner.accepted_licenses",
            ),
            require_known_license=bool(cleaner_raw.get("require_known_license", False)),
        ),
        deduplication=PyPIDeduplicationSettings(
            normalized=bool(dedup_raw.get("normalized", False)),
            near_duplicate=bool(dedup_raw.get("near_duplicate", False)),
            near_duplicate_distance=int(dedup_raw.get("near_duplicate_distance", 3)),
            index_path=_resolve(
                collector.project_root,
                dedup_raw.get("index", "data/pypi/duplicate_index.sqlite3"),
            ),
        ),
        tokens=PyPITokenSettings(
            tokenizer_path=_resolve(
                collector.project_root,
                tokens_raw.get("tokenizer", "data/tokenizer/tokenizer.json"),
            ),
            output_directory=output_directory,
            shard_index_path=output_directory
            / _filename(tokens_raw.get("index", "index.json")),
            statistics_path=output_directory
            / _filename(tokens_raw.get("statistics", "statistics.json")),
            document_index_filename=_filename(
                tokens_raw.get("document_index", "pypi_document_index.jsonl")
            ),
            shard_prefix=_filename(tokens_raw.get("shard_prefix", "shard")),
            max_tokens_per_shard=int(tokens_raw.get("max_tokens_per_shard", 10_000_000)),
            workers=_worker_count(int(tokens_raw.get("workers", 0))),
            max_pending_tasks_per_worker=int(
                tokens_raw.get("max_pending_tasks_per_worker", 4)
            ),
        ),
        paths=PyPICorpusPaths(
            checkpoint=_resolve(
                collector.project_root,
                paths_raw.get("checkpoint", "data/pypi/checkpoint.sqlite3"),
            ),
            report_directory=report_directory,
            pypi_report=report_directory / "pypi_report.json",
            package_statistics=report_directory / "package_statistics.json",
            license_report=report_directory / "license_report.json",
            quality_report=report_directory / "quality_report.json",
            duplicate_report=report_directory / "duplicate_report.json",
            token_statistics=report_directory / "token_statistics.json",
            log_file=_resolve(
                collector.project_root,
                paths_raw.get("log_file", "logs/pypi_corpus_builder.jsonl"),
            ),
        ),
        collector=collector,
        corpus_manager=corpus_manager,
        progress=bool(section.get("progress", raw.get("progress", True))),
        log_level=str(raw.get("logging", {}).get("level", "INFO")).upper(),
    )
    _validate_config(config)
    return config


class _SimpleIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []
        self._inside = False

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        self._inside = tag.casefold() == "a"

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a":
            self._inside = False

    def handle_data(self, data: str) -> None:
        if self._inside and data.strip():
            self.names.append(data.strip())


class PyPIClient:
    """Retried PyPI JSON/Simple API client with injectable settings."""

    def __init__(self, settings: PyPIAPISettings) -> None:
        self.settings = settings
        self._metadata_cache: dict[tuple[str, str | None], dict[str, Any]] = {}

    def package_metadata(self, name: str, version: str | None = None) -> dict[str, Any]:
        key = (_canonicalize(name), version)
        if key not in self._metadata_cache:
            quoted = urllib.parse.quote(name, safe="")
            endpoint = f"{self.settings.base_url}/{quoted}"
            if version:
                endpoint += f"/{urllib.parse.quote(version, safe='')}"
            self._metadata_cache[key] = self._request_json(f"{endpoint}/json")
        return self._metadata_cache[key]

    def top_packages(self, limit: int, minimum_downloads: int) -> list[tuple[str, int]]:
        if limit <= 0:
            return []
        payload = self._request_json(self.settings.top_packages_url)
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            raise PyPICorpusError("Top-package response must contain a rows list.")
        packages: list[tuple[str, int]] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("project"), str):
                continue
            downloads = int(row.get("download_count", 0))
            if downloads >= minimum_downloads:
                packages.append((row["project"], downloads))
            if len(packages) >= limit:
                break
        return packages

    def simple_package_names(self, limit: int) -> list[str]:
        if limit <= 0:
            return []
        content = self._request_bytes(self.settings.simple_index_url).decode("utf-8")
        try:
            payload: Any = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("projects"), list):
            names = [
                item["name"]
                for item in payload["projects"]
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
            return names[:limit]
        parser = _SimpleIndexParser()
        parser.feed(content)
        return parser.names[:limit]

    def _request_json(self, url: str) -> dict[str, Any]:
        try:
            payload: Any = json.loads(self._request_bytes(url).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PyPICorpusError(f"Invalid JSON response from {url}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PyPICorpusError(f"JSON response from {url} must be an object.")
        return payload

    def _request_bytes(self, url: str) -> bytes:
        headers = {"Accept": "application/json", "User-Agent": self.settings.user_agent}
        for attempt in range(self.settings.retries + 1):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(  # noqa: S310
                    request, timeout=self.settings.request_timeout_seconds
                ) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt >= self.settings.retries:
                    raise PyPICorpusError(f"Request failed for {url}: {exc}") from exc
                time.sleep(self.settings.retry_backoff_seconds * (2**attempt))
        raise PyPICorpusError("PyPI request retry loop ended unexpectedly.")


class PyPICheckpoint:
    """SQLite checkpoint for package metadata, downloads, and build stages."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database = sqlite3.connect(path)
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=NORMAL")
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS packages (
                canonical_name TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                status TEXT NOT NULL,
                archive_path TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS package_status ON packages(status);
            CREATE TABLE IF NOT EXISTS stages (
                stage TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        self.database.commit()

    def __enter__(self) -> PyPICheckpoint:
        return self

    def __exit__(self, *_args: object) -> None:
        self.database.close()

    def save_packages(self, packages: Sequence[PyPIPackage]) -> None:
        now = _timestamp()
        for package in packages:
            old = self.database.execute(
                "SELECT metadata_json, status FROM packages WHERE canonical_name=?",
                (package.canonical_name,),
            ).fetchone()
            status = "discovered"
            if old is not None:
                previous = json.loads(old[0])
                if previous.get("sha256") == package.sha256:
                    status = str(old[1])
            self.database.execute(
                """
                INSERT INTO packages(canonical_name, metadata_json, status, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(canonical_name) DO UPDATE SET
                    metadata_json=excluded.metadata_json,
                    status=excluded.status,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    package.canonical_name,
                    json.dumps(asdict(package), sort_keys=True, separators=(",", ":")),
                    status,
                    now,
                ),
            )
        self.database.commit()

    def packages(self, names: Sequence[str] | None = None) -> list[PyPIPackage]:
        rows = self.database.execute(
            "SELECT metadata_json FROM packages ORDER BY canonical_name"
        ).fetchall()
        selected = set(names) if names is not None else None
        packages = [PyPIPackage(**json.loads(row[0])) for row in rows]
        return [
            package
            for package in packages
            if selected is None or package.canonical_name in selected
        ]

    def status(self, canonical_name: str) -> tuple[str, Path | None] | None:
        row = self.database.execute(
            "SELECT status, archive_path FROM packages WHERE canonical_name=?",
            (canonical_name,),
        ).fetchone()
        return None if row is None else (str(row[0]), Path(row[1]) if row[1] else None)

    def mark_package(
        self,
        canonical_name: str,
        status: str,
        *,
        archive_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        self.database.execute(
            """UPDATE packages SET status=?, archive_path=?, error=?, updated_at=?
            WHERE canonical_name=?""",
            (
                status,
                str(archive_path) if archive_path else None,
                error,
                _timestamp(),
                canonical_name,
            ),
        )
        self.database.commit()

    def stage_payload(self, stage: str, fingerprint: str) -> dict[str, Any] | None:
        row = self.database.execute(
            "SELECT status, payload_json FROM stages WHERE stage=? AND fingerprint=?",
            (stage, fingerprint),
        ).fetchone()
        if row is None or row[0] != "complete":
            return None
        payload: Any = json.loads(row[1])
        return payload if isinstance(payload, dict) else None

    def mark_stage(
        self, stage: str, fingerprint: str, status: str, payload: Mapping[str, Any]
    ) -> None:
        self.database.execute(
            """
            INSERT INTO stages(stage, fingerprint, status, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(stage) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                status=excluded.status,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """,
            (
                stage,
                fingerprint,
                status,
                json.dumps(dict(payload), sort_keys=True, separators=(",", ":")),
                _timestamp(),
            ),
        )
        self.database.commit()


class PyPIDuplicateIndex:
    """Optional scalable AST-normalized and token-SimHash duplicate index."""

    def __init__(self, settings: PyPIDeduplicationSettings) -> None:
        settings.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.database = sqlite3.connect(settings.index_path)
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS normalized (
                normalized_sha256 TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS normalized_content ON normalized(content_sha256);
            CREATE TABLE IF NOT EXISTS near_duplicates (
                content_sha256 TEXT PRIMARY KEY, simhash INTEGER NOT NULL,
                band0 INTEGER NOT NULL, band1 INTEGER NOT NULL,
                band2 INTEGER NOT NULL, band3 INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS near_band0 ON near_duplicates(band0);
            CREATE INDEX IF NOT EXISTS near_band1 ON near_duplicates(band1);
            CREATE INDEX IF NOT EXISTS near_band2 ON near_duplicates(band2);
            CREATE INDEX IF NOT EXISTS near_band3 ON near_duplicates(band3);
            """
        )
        self.database.commit()
        self.reasons: Counter[str] = Counter()

    def close(self) -> None:
        self.database.commit()
        self.database.close()

    def __enter__(self) -> PyPIDuplicateIndex:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def check(self, _candidate: SourceCandidate, source: str, digest: str) -> str | None:
        normalized_hash = _normalized_source_hash(source) if self.settings.normalized else None
        if self.settings.normalized:
            row = self.database.execute(
                "SELECT content_sha256 FROM normalized WHERE normalized_sha256=?",
                (normalized_hash,),
            ).fetchone()
            if row is not None and row[0] != digest:
                self.reasons["normalized_duplicate"] += 1
                return "normalized_duplicate"
        simhash = _source_simhash(source) if self.settings.near_duplicate else None
        if self.settings.near_duplicate:
            assert simhash is not None
            bands = _simhash_bands(simhash)
            rows = self.database.execute(
                """SELECT content_sha256, simhash FROM near_duplicates
                WHERE band0=? OR band1=? OR band2=? OR band3=?""",
                bands,
            ).fetchall()
            for existing_digest, encoded_hash in rows:
                existing = _unsigned64(int(encoded_hash))
                if existing_digest != digest and (existing ^ simhash).bit_count() <= (
                    self.settings.near_duplicate_distance
                ):
                    self.reasons["near_duplicate"] += 1
                    return "near_duplicate"
        if self.settings.normalized:
            assert normalized_hash is not None
            self.database.execute(
                "INSERT OR IGNORE INTO normalized VALUES (?, ?)",
                (normalized_hash, digest),
            )
        if self.settings.near_duplicate:
            assert simhash is not None
            bands = _simhash_bands(simhash)
            self.database.execute(
                "INSERT OR IGNORE INTO near_duplicates VALUES (?, ?, ?, ?, ?, ?)",
                (digest, _signed64(simhash), *bands),
            )
        self.database.commit()
        return None

    def seed_manifest(self, manifest_path: Path, corpus_root: Path) -> None:
        """Index existing validated corpus files before checking newly imported files."""

        if not manifest_path.is_file():
            return
        with manifest_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record: Any = json.loads(line)
                if not isinstance(record, dict):
                    continue
                stored_path = record.get("stored_path")
                digest = record.get("content_sha256")
                if not isinstance(stored_path, str) or not isinstance(digest, str):
                    continue
                normalized_known = not self.settings.normalized or self.database.execute(
                    "SELECT 1 FROM normalized WHERE content_sha256=? LIMIT 1", (digest,)
                ).fetchone() is not None
                near_known = not self.settings.near_duplicate or self.database.execute(
                    "SELECT 1 FROM near_duplicates WHERE content_sha256=?", (digest,)
                ).fetchone() is not None
                if normalized_known and near_known:
                    continue
                path = corpus_root / stored_path
                try:
                    source = path.read_text(encoding="utf-8-sig")
                    ast.parse(source)
                except (OSError, UnicodeDecodeError, SyntaxError, ValueError, TypeError):
                    continue
                normalized_hash = (
                    _normalized_source_hash(source) if self.settings.normalized else None
                )
                if normalized_hash is not None:
                    self.database.execute(
                        "INSERT OR IGNORE INTO normalized VALUES (?, ?)",
                        (normalized_hash, digest),
                    )
                if self.settings.near_duplicate:
                    simhash = _source_simhash(source)
                    self.database.execute(
                        "INSERT OR IGNORE INTO near_duplicates VALUES (?, ?, ?, ?, ?, ?)",
                        (digest, _signed64(simhash), *_simhash_bands(simhash)),
                    )
        self.database.commit()


def discover_packages(
    config: PyPICorpusConfig,
    client: PyPIClient,
    *,
    errors: dict[str, str] | None = None,
) -> list[PyPIPackage]:
    """Resolve all configured selectors into deterministic, sdist-only metadata."""

    requested: dict[str, tuple[str, str | None, int | None]] = {}

    def add(specification: str, downloads: int | None = None) -> None:
        name, version = _parse_package_spec(specification)
        key = _canonicalize(name)
        previous = requested.get(key)
        if previous is None or (previous[1] is None and version is not None):
            requested[key] = (name, version, downloads)

    if config.selection.top_downloaded:
        for name, downloads in client.top_packages(
            config.selection.top_downloaded_limit, config.selection.minimum_downloads
        ):
            add(name, downloads)
    for category in config.selection.enabled_categories:
        if category not in config.selection.categories:
            raise PyPICorpusError(f"Unknown enabled package category: {category}")
        for package in config.selection.categories[category]:
            add(package)
    for path in config.selection.requirements_files:
        for package in _requirements_packages(path):
            add(package)
    for package in config.selection.manual_packages:
        add(package)

    if config.selection.keywords:
        keywords = tuple(value.casefold() for value in config.selection.keywords)
        for name in client.simple_package_names(config.selection.keyword_scan_limit):
            metadata = client.package_metadata(name)
            info = metadata.get("info", {})
            haystack = " ".join(
                str(info.get(field) or "") for field in ("name", "summary", "keywords")
            ).casefold()
            if any(keyword in haystack for keyword in keywords):
                add(name)

    packages: list[PyPIPackage] = []
    ignored = {value.casefold() for value in config.selection.ignored_licenses}
    for canonical_name in sorted(requested):
        if len(packages) >= config.selection.maximum_packages:
            break
        name, version, downloads = requested[canonical_name]
        try:
            package = _package_from_api(client.package_metadata(name, version), downloads)
        except PyPICorpusError as exc:
            if errors is not None:
                errors[canonical_name] = str(exc)
            LOGGER.warning("package_metadata_rejected package=%s error=%s", name, exc)
            continue
        if package.license.casefold() in ignored:
            LOGGER.info("package_license_ignored package=%s license=%s", name, package.license)
            continue
        packages.append(package)
    return packages


def build_pypi_corpus(
    config: PyPICorpusConfig,
    *,
    api_client: PyPIClient | None = None,
    force: bool = False,
) -> PyPICorpusResult:
    """Discover, download, extract, collect, deduplicate, tokenize, and report."""

    if not config.enabled:
        raise PyPICorpusError(
            "pypi_corpus.enabled is false; review source and license policy before enabling."
        )
    if not config.tokens.tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Existing GenPy tokenizer not found: {config.tokens.tokenizer_path}. "
            "This pipeline never retrains it."
        )
    config.download.directory.mkdir(parents=True, exist_ok=True)
    config.extraction.directory.mkdir(parents=True, exist_ok=True)
    config.paths.report_directory.mkdir(parents=True, exist_ok=True)
    client = api_client or PyPIClient(config.api)
    discovery_fingerprint = _discovery_fingerprint(config)

    with PyPICheckpoint(config.paths.checkpoint) as checkpoint:
        discovery_errors: dict[str, str] = {}
        cached = (
            checkpoint.stage_payload("discovery", discovery_fingerprint)
            if config.download.resume and not force
            else None
        )
        cached_names = cached.get("package_names") if cached is not None else None
        if isinstance(cached_names, list) and all(
            isinstance(name, str) for name in cached_names
        ):
            packages = checkpoint.packages(cached_names)
            LOGGER.info("package_discovery_resumed packages=%d", len(packages))
        else:
            LOGGER.info("package_discovery_started")
            packages = discover_packages(config, client, errors=discovery_errors)
            checkpoint.save_packages(packages)
            checkpoint.mark_stage(
                "discovery",
                discovery_fingerprint,
                "complete" if not discovery_errors else "incomplete",
                {
                    "packages": len(packages),
                    "package_names": [package.canonical_name for package in packages],
                    "metadata_errors": discovery_errors,
                },
            )
            LOGGER.info("package_discovery_completed packages=%d", len(packages))

        downloaded, download_errors = _download_packages(
            config, packages, checkpoint, force=force
        )
        extraction_errors: dict[str, str] = {}
        with tempfile.TemporaryDirectory(
            prefix="genpy-pypi-", dir=config.extraction.directory
        ) as temporary:
            extracted, extraction_errors = _extract_packages(
                config, downloaded, Path(temporary)
            )
            extraction_counts = {
                item.downloaded.package.canonical_name: item.python_files
                for item in extracted
            }
            sources = tuple(_corpus_source(config, item) for item in extracted)
            with PyPIDuplicateIndex(config.deduplication) as duplicate_index:
                if config.deduplication.normalized or config.deduplication.near_duplicate:
                    duplicate_index.seed_manifest(
                        config.collector.provenance_manifest,
                        config.collector.output_directory,
                    )
                LOGGER.info("package_validation_started sources=%d", len(sources))
                collection = collect_python_corpus(
                    config.collector,
                    sources=sources,
                    collect_manual=False,
                    duplicate_filter=(
                        lambda candidate, source, digest: _filter_candidate(
                            config, duplicate_index, candidate, source, digest
                        )
                        if config.cleaner_enabled
                        or config.deduplication.normalized
                        or config.deduplication.near_duplicate
                        else None
                    ),
                )
                optional_duplicate_reasons = dict(duplicate_index.reasons)
                LOGGER.info(
                    "package_validation_completed scanned=%d accepted=%d unchanged=%d rejected=%d",
                    collection.files_scanned,
                    collection.files_accepted,
                    collection.files_unchanged,
                    collection.files_rejected,
                )

        corpus_index = expand_python_corpus(config.corpus_manager, collect=False)

        manifest_hash = stable_manifest_fingerprint(
            config.collector.provenance_manifest, {"pypi"}
        )
        tokenizer_hash = tokenizer_file_hash(config.tokens.tokenizer_path)
        fingerprint = _build_fingerprint(
            config,
            downloaded,
            manifest_hash=manifest_hash,
            tokenizer_hash=tokenizer_hash,
        )
        token_config = _token_shard_config(config)
        resumed_payload = (
            checkpoint.stage_payload("binary_shards", fingerprint)
            if config.download.resume and not force
            else None
        )
        resumed = resumed_payload is not None and binary_outputs_valid(
            token_config, fingerprint
        )
        if resumed:
            token_statistics = resumed_payload
        else:
            checkpoint.mark_stage("binary_shards", fingerprint, "running", {})
            prepare_binary_output(token_config)
            _, token_statistics = build_manifest_token_shards(
                manifest_path=config.collector.provenance_manifest,
                corpus_root=config.collector.output_directory,
                source_types={"pypi"},
                config=token_config,
                tokenizer_sha256=tokenizer_hash,
                manifest_fingerprint=manifest_hash,
                build_fingerprint=fingerprint,
                metadata_builder=_pypi_document_metadata,
                progress=config.progress,
            )
            checkpoint.mark_stage(
                "binary_shards", fingerprint, "complete", token_statistics
            )
        statistics_path = _write_reports(
            config,
            packages,
            downloaded,
            extraction_counts,
            discovery_errors,
            download_errors,
            extraction_errors,
            collection,
            optional_duplicate_reasons,
            token_statistics,
            corpus_index_path=corpus_index.index_path,
            resumed=resumed,
        )
    return PyPICorpusResult(
        packages_discovered=len(packages),
        packages_downloaded=len(downloaded),
        packages_failed=len(
            set(discovery_errors) | set(download_errors) | set(extraction_errors)
        ),
        collection=collection,
        documents=int(token_statistics["documents"]),
        token_count=int(token_statistics["token_count"]),
        shard_count=int(token_statistics["shard_count"]),
        shard_index_path=config.tokens.shard_index_path,
        statistics_path=statistics_path,
        resumed=resumed,
    )


def _download_packages(
    config: PyPICorpusConfig,
    packages: Sequence[PyPIPackage],
    checkpoint: PyPICheckpoint,
    *,
    force: bool,
) -> tuple[list[DownloadedSdist], dict[str, str]]:
    downloaded: list[DownloadedSdist] = []
    errors: dict[str, str] = {}
    progress = ProgressBar("sdists", len(packages), enabled=config.progress)
    futures: dict[Future[DownloadedSdist], PyPIPackage] = {}
    with ProcessPoolExecutor(max_workers=config.download.workers) as executor:
        for package in packages:
            prior = checkpoint.status(package.canonical_name)
            futures[
                executor.submit(
                    _download_sdist,
                    package,
                    config.download,
                    prior,
                    force,
                    config.api.user_agent,
                )
            ] = package
        for completed, future in enumerate(as_completed(futures), start=1):
            package = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - one package must not stop a large run
                errors[package.canonical_name] = str(exc)
                checkpoint.mark_package(package.canonical_name, "failed", error=str(exc))
                LOGGER.warning(
                    "package_download_failed package=%s error=%s",
                    package.name,
                    exc,
                )
            else:
                downloaded.append(result)
                checkpoint.mark_package(
                    package.canonical_name,
                    "downloaded",
                    archive_path=result.archive_path,
                )
                LOGGER.info(
                    "package_download_completed package=%s version=%s archive=%s resumed=%s",
                    package.name,
                    package.version,
                    result.archive_path,
                    result.resumed,
                )
            progress.update(completed)
    progress.close()
    downloaded.sort(key=lambda item: item.package.canonical_name)
    return downloaded, errors


def _download_sdist(
    package: PyPIPackage,
    settings: PyPIDownloadSettings,
    prior: tuple[str, Path | None] | None,
    force: bool,
    user_agent: str,
) -> DownloadedSdist:
    if not _supported_archive(package.filename):
        raise PyPICorpusError(f"Refusing non-sdist archive: {package.filename}")
    destination = (
        settings.directory
        / package.canonical_name
        / _safe_version_component(package.version)
        / _safe_archive_filename(package.filename)
    )
    if (
        not force
        and settings.resume
        and destination.is_file()
        and _file_hash(destination) == package.sha256
    ):
        return DownloadedSdist(package, destination, resumed=True)
    if (
        not force
        and settings.resume
        and prior is not None
        and prior[0] == "downloaded"
        and prior[1] is not None
        and prior[1].is_file()
        and _file_hash(prior[1]) == package.sha256
    ):
        return DownloadedSdist(package, prior[1], resumed=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{destination}.partial")
    if force:
        partial.unlink(missing_ok=True)
    elif partial.is_file() and _file_hash(partial) == package.sha256:
        os.replace(partial, destination)
        return DownloadedSdist(package, destination, resumed=True)
    for attempt in range(settings.retries + 1):
        try:
            _stream_download(
                package.download_url,
                partial,
                timeout=settings.timeout_seconds,
                resume=settings.resume,
                user_agent=user_agent,
            )
            actual = _file_hash(partial)
            if actual != package.sha256:
                partial.unlink(missing_ok=True)
                raise PyPICorpusError(
                    f"Checksum mismatch for {package.name}: expected {package.sha256}, got {actual}"
                )
            os.replace(partial, destination)
            return DownloadedSdist(package, destination, resumed=False)
        except Exception as exc:  # noqa: BLE001 - retried and contextualized
            if attempt >= settings.retries:
                raise PyPICorpusError(f"Download failed for {package.name}: {exc}") from exc
            time.sleep(settings.retry_backoff_seconds * (2**attempt))
    raise PyPICorpusError("Download retry loop ended unexpectedly.")


def _stream_download(
    url: str,
    partial: Path,
    *,
    timeout: int,
    resume: bool,
    user_agent: str,
) -> None:
    offset = partial.stat().st_size if resume and partial.is_file() else 0
    headers = {"User-Agent": user_agent}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        status = getattr(response, "status", None)
        append = bool(offset and status == 206)
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _extract_packages(
    config: PyPICorpusConfig,
    downloaded: Sequence[DownloadedSdist],
    temporary_root: Path,
) -> tuple[list[ExtractedSdist], dict[str, str]]:
    extracted: list[ExtractedSdist] = []
    errors: dict[str, str] = {}
    progress = ProgressBar("extract", len(downloaded), enabled=config.progress)
    futures: dict[Future[ExtractedSdist], DownloadedSdist] = {}
    with ProcessPoolExecutor(max_workers=config.extraction.workers) as executor:
        for item in downloaded:
            destination = temporary_root / _safe_package_directory(item.package)
            futures[
                executor.submit(_extract_sdist, item, destination, config.extraction)
            ] = item
        for completed, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                result = future.result()
                extracted.append(result)
                LOGGER.info(
                    "package_extraction_completed package=%s python_files=%d",
                    item.package.name,
                    result.python_files,
                )
            except Exception as exc:  # noqa: BLE001
                errors[item.package.canonical_name] = str(exc)
                LOGGER.warning(
                    "package_extraction_failed package=%s error=%s",
                    item.package.name,
                    exc,
                )
            progress.update(completed)
    progress.close()
    extracted.sort(key=lambda item: item.downloaded.package.canonical_name)
    return extracted, errors


def _extract_sdist(
    item: DownloadedSdist,
    destination: Path,
    settings: PyPIExtractionSettings,
) -> ExtractedSdist:
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    expanded = 0
    members = 0
    try:
        if item.archive_path.name.casefold().endswith(".zip"):
            with zipfile.ZipFile(item.archive_path) as archive:
                for member in sorted(archive.infolist(), key=lambda value: value.filename):
                    members += 1
                    if members > settings.maximum_members:
                        raise PyPICorpusError("Archive member limit exceeded.")
                    relative = PurePosixPath(member.filename)
                    if not _selected_archive_path(relative, settings) or member.is_dir():
                        continue
                    if _zip_symlink(member):
                        raise PyPICorpusError(f"Archive symlink rejected: {member.filename}")
                    expanded += member.file_size
                    _check_expanded_size(expanded, settings)
                    _write_archive_member(destination, relative, archive.open(member))
                    count += 1
        else:
            with tarfile.open(item.archive_path, mode="r|*") as archive:
                for member in archive:
                    members += 1
                    if members > settings.maximum_members:
                        raise PyPICorpusError("Archive member limit exceeded.")
                    relative = PurePosixPath(member.name)
                    if not _selected_archive_path(relative, settings) or not member.isfile():
                        continue
                    expanded += member.size
                    _check_expanded_size(expanded, settings)
                    source = archive.extractfile(member)
                    if source is None:
                        raise PyPICorpusError(f"Cannot read archive member: {member.name}")
                    _write_archive_member(destination, relative, source)
                    count += 1
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
        raise PyPICorpusError(f"Corrupt source archive {item.archive_path}: {exc}") from exc
    return ExtractedSdist(item, destination, count)


def _write_archive_member(destination: Path, relative: PurePosixPath, source: Any) -> None:
    target = destination.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError as exc:  # pragma: no cover - guarded by selection
        raise PyPICorpusError(f"Unsafe archive path: {relative}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{target}.partial")
    try:
        with source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        os.replace(partial, target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _selected_archive_path(
    relative: PurePosixPath, settings: PyPIExtractionSettings
) -> bool:
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise PyPICorpusError(f"Unsafe archive path: {relative}")
    if relative.suffix.casefold() != ".py":
        return False
    ignored = {value.casefold() for value in settings.ignored_directories}
    if settings.ignore_migrations:
        ignored.add("migrations")
    return not any(part.casefold() in ignored for part in relative.parts[:-1])


def _corpus_source(config: PyPICorpusConfig, item: ExtractedSdist) -> CorpusSource:
    package = item.downloaded.package
    return CorpusSource(
        source_id=_pypi_source_id(package),
        source_type="pypi",
        location=_portable_path(config.project_root, item.extraction_path),
        license=package.license,
        approval=config.approval,
        revision=package.version,
        include=("**/*.py",),
        discovered_automatically=True,
        repository_url=package.repository_url,
        package_name=package.name,
        package_version=package.version,
        release_date=package.release_date,
        homepage=package.homepage,
        project_url=package.project_url,
        author=package.author,
        summary=package.summary,
        keywords=package.keywords,
        source_archive=package.filename,
        download_url=package.download_url,
        archive_sha256=package.sha256,
    )


def _filter_candidate(
    config: PyPICorpusConfig,
    duplicate_index: PyPIDuplicateIndex,
    candidate: SourceCandidate,
    source: str,
    digest: str,
) -> str | None:
    if config.cleaner_enabled:
        result = filter_code_record(
            {
                "content": source,
                "path": candidate.relative_path.as_posix(),
                "license": candidate.source.license,
            },
            settings=config.cleaner,
        )
        if not result.accepted:
            return f"cleaner_{result.reason}"
    if config.deduplication.normalized or config.deduplication.near_duplicate:
        return duplicate_index.check(candidate, source, digest)
    return None


def _pypi_document_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record.get("source")
    source = source if isinstance(source, dict) else {}
    return {
        "package": source.get("package"),
        "version": source.get("version"),
        "repository": source.get("repository_url"),
        "license": record.get("license"),
        "source_archive": source.get("source_archive"),
        "source_path": record.get("source_path"),
        "stored_path": record.get("stored_path"),
        "sha256": record.get("content_sha256"),
        "byte_size": record.get("size_bytes"),
        "import_timestamp": record.get("collection_timestamp"),
        "download_url": source.get("download_url"),
        "archive_sha256": source.get("archive_sha256"),
        "release_date": source.get("release_date"),
    }


def _write_reports(
    config: PyPICorpusConfig,
    packages: Sequence[PyPIPackage],
    downloaded: Sequence[DownloadedSdist],
    extraction_counts: Mapping[str, int],
    discovery_errors: Mapping[str, str],
    download_errors: Mapping[str, str],
    extraction_errors: Mapping[str, str],
    collection: CollectionResult,
    optional_duplicates: Mapping[str, int],
    token_statistics: Mapping[str, Any],
    *,
    corpus_index_path: Path,
    resumed: bool,
) -> Path:
    downloaded_by_name = {
        item.package.canonical_name: item for item in downloaded
    }
    package_rows = []
    for package in packages:
        row = asdict(package)
        download = downloaded_by_name.get(package.canonical_name)
        row["downloaded"] = download is not None
        row["download_resumed"] = download.resumed if download else False
        row["archive_path"] = str(download.archive_path) if download else None
        row["extracted"] = package.canonical_name in extraction_counts
        row["extracted_python_files"] = extraction_counts.get(package.canonical_name, 0)
        row["download_error"] = download_errors.get(package.canonical_name)
        row["extraction_error"] = extraction_errors.get(package.canonical_name)
        package_rows.append(row)
    atomic_json(
        config.paths.package_statistics,
        {
            "packages_discovered": len(packages),
            "packages_downloaded": len(downloaded),
            "packages_failed": len(
                set(discovery_errors) | set(download_errors) | set(extraction_errors)
            ),
            "metadata_errors": dict(sorted(discovery_errors.items())),
            "packages": package_rows,
        },
    )
    licenses = Counter(package.license for package in packages)
    downloaded_licenses = Counter(item.package.license for item in downloaded)
    atomic_json(
        config.paths.license_report,
        {
            "discovered": dict(sorted(licenses.items())),
            "downloaded": dict(sorted(downloaded_licenses.items())),
            "packages": len(packages),
        },
    )
    quality = {
        "files_scanned": collection.files_scanned,
        "files_accepted": collection.files_accepted,
        "files_unchanged": collection.files_unchanged,
        "files_rejected": collection.files_rejected,
        "acceptance_rate": _ratio(
            collection.files_accepted + collection.files_unchanged,
            collection.files_scanned,
        ),
        "rejection_reasons": collection.rejection_reasons,
        "metadata_failures": dict(sorted(discovery_errors.items())),
        "download_failures": dict(sorted(download_errors.items())),
        "extraction_failures": dict(sorted(extraction_errors.items())),
        "tokenization_rejections": token_statistics.get("rejection_reasons", {}),
    }
    atomic_json(config.paths.quality_report, quality)
    exact = int(collection.rejection_reasons.get("duplicate_content", 0))
    duplicate_report = {
        "sha256_duplicates": exact,
        "normalized_duplicates": int(optional_duplicates.get("normalized_duplicate", 0)),
        "near_duplicates": int(optional_duplicates.get("near_duplicate", 0)),
        "total_duplicates": exact + sum(optional_duplicates.values()),
        "normalized_detection_enabled": config.deduplication.normalized,
        "near_duplicate_detection_enabled": config.deduplication.near_duplicate,
    }
    atomic_json(config.paths.duplicate_report, duplicate_report)
    atomic_json(config.paths.token_statistics, dict(token_statistics))
    aggregate = {
        "pypi_corpus_version": PYPI_CORPUS_VERSION,
        "creation_timestamp": _timestamp(),
        "configuration": str(config.config_path),
        "resumed": resumed,
        "packages_discovered": len(packages),
        "packages_downloaded": len(downloaded),
        "packages_failed": len(
            set(discovery_errors) | set(download_errors) | set(extraction_errors)
        ),
        "files_scanned": collection.files_scanned,
        "files_accepted": collection.files_accepted,
        "files_unchanged": collection.files_unchanged,
        "files_rejected": collection.files_rejected,
        "documents": token_statistics.get("documents", 0),
        "token_count": token_statistics.get("token_count", 0),
        "shard_count": token_statistics.get("shard_count", 0),
        "corpus_index": str(corpus_index_path),
        "reports": {
            "packages": str(config.paths.package_statistics),
            "licenses": str(config.paths.license_report),
            "quality": str(config.paths.quality_report),
            "duplicates": str(config.paths.duplicate_report),
            "tokens": str(config.paths.token_statistics),
        },
    }
    atomic_json(config.paths.pypi_report, aggregate)
    return config.paths.pypi_report


def configure_pypi_logging(config: PyPICorpusConfig, *, level: str | None = None) -> None:
    setup_structured_logging(config.paths.log_file, level or config.log_level)


def run_pypi_corpus_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build validated PyPI sdist corpus and binary token shards."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pypi.yaml"))
    parser.add_argument("--force", action="store_true", help="Redownload and rebuild outputs.")
    parser.add_argument("--max-packages", type=int)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    try:
        config = load_pypi_corpus_config(args.config)
        if args.max_packages is not None:
            config = replace(
                config,
                selection=replace(config.selection, maximum_packages=args.max_packages),
            )
            _validate_config(config)
        configure_pypi_logging(config, level=args.log_level)
        result = build_pypi_corpus(config, force=args.force)
    except KeyboardInterrupt:
        print("Interrupted; completed PyPI checkpoints can be resumed.", file=os.sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("PyPI corpus build failed")
        else:
            print(f"Error: {exc}", file=os.sys.stderr)
        return 1
    print("GenPy Phase 5.5B PyPI corpus complete")
    print(f"Packages discovered: {result.packages_discovered}")
    print(f"Packages downloaded: {result.packages_downloaded}")
    print(f"Packages failed: {result.packages_failed}")
    print(f"Python files accepted: {result.collection.files_accepted}")
    print(f"Python files unchanged: {result.collection.files_unchanged}")
    print(f"Documents tokenized: {result.documents}")
    print(f"Tokens: {result.token_count}")
    print(f"Binary shards: {result.shard_count}")
    print(f"Shard index: {result.shard_index_path}")
    print(f"Statistics: {result.statistics_path}")
    print(f"Resumed: {result.resumed}")
    return 0


def _package_from_api(payload: Mapping[str, Any], downloads: int | None) -> PyPIPackage:
    info = payload.get("info")
    urls = payload.get("urls")
    if not isinstance(info, dict) or not isinstance(urls, list):
        raise PyPICorpusError("PyPI package metadata is missing info or urls.")
    name = _required_string(info.get("name"), "package name")
    version = _required_string(info.get("version"), "package version")
    candidates = [
        item
        for item in urls
        if isinstance(item, dict)
        and item.get("packagetype") == "sdist"
        and not bool(item.get("yanked", False))
        and isinstance(item.get("filename"), str)
        and _supported_archive(item["filename"])
    ]
    if not candidates:
        raise PyPICorpusError(f"No supported source distribution for {name}=={version}.")
    candidate = sorted(candidates, key=lambda item: str(item["filename"]))[0]
    filename = _safe_archive_filename(str(candidate["filename"]))
    digests = candidate.get("digests")
    sha256 = digests.get("sha256") if isinstance(digests, dict) else None
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
        raise PyPICorpusError(f"Source distribution for {name} lacks a SHA256 digest.")
    project_urls = info.get("project_urls")
    project_urls = project_urls if isinstance(project_urls, dict) else {}
    repository = _repository_url(project_urls, info.get("home_page"))
    upload_time = candidate.get("upload_time_iso_8601") or candidate.get("upload_time")
    license_value = _license_value(info)
    project_url = (
        f"https://pypi.org/project/{urllib.parse.quote(name, safe='')}/"
        f"{urllib.parse.quote(version, safe='')}/"
    )
    return PyPIPackage(
        name=name,
        canonical_name=_canonicalize(name),
        version=version,
        release_date=str(upload_time) if upload_time else None,
        homepage=_optional_string(info.get("home_page")),
        project_url=project_url,
        repository_url=repository,
        author=_optional_string(info.get("author")),
        license=license_value,
        summary=_optional_string(info.get("summary")),
        keywords=_optional_string(info.get("keywords")),
        download_url=_required_string(candidate.get("url"), "sdist URL"),
        filename=filename,
        sha256=sha256.casefold(),
        download_count=downloads,
    )


def _license_value(info: Mapping[str, Any]) -> str:
    expression = _optional_string(info.get("license_expression"))
    if expression:
        return expression
    value = info.get("license")
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())[:500]
    classifiers = info.get("classifiers")
    if isinstance(classifiers, list):
        licenses = [
            item.rsplit(" :: ", 1)[-1]
            for item in classifiers
            if isinstance(item, str) and item.startswith("License ::")
        ]
        if licenses:
            return "; ".join(licenses)
    return "NOASSERTION"


def _repository_url(project_urls: Mapping[str, Any], homepage: object) -> str | None:
    for key, value in project_urls.items():
        if isinstance(value, str) and any(
            word in str(key).casefold() for word in ("source", "repository", "code", "github")
        ):
            return value
    if isinstance(homepage, str) and "github.com" in homepage.casefold():
        return homepage
    return None


def _requirements_packages(path: Path) -> Iterable[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Requirements file not found: {path}")
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http:", "https:", "git+")):
            continue
        match = REQUIREMENT_NAME.match(line)
        if match is None:
            raise PyPICorpusError(f"Invalid requirement at {path}:{line_number}")
        name = match.group(1)
        exact = re.search(r"==\s*([A-Za-z0-9][A-Za-z0-9.!+_-]*)", line)
        yield f"{name}=={exact.group(1)}" if exact else name


def _parse_package_spec(specification: str) -> tuple[str, str | None]:
    match = REQUIREMENT_NAME.match(specification)
    if match is None:
        raise PyPICorpusError(f"Invalid package specification: {specification!r}")
    name = match.group(1)
    exact = re.search(r"==\s*([A-Za-z0-9][A-Za-z0-9.!+_-]*)", specification)
    return name, exact.group(1) if exact else None


def _normalized_source_hash(source: str) -> str:
    tree = ast.parse(source)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _source_simhash(source: str) -> int:
    tokens: list[str] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(source).readline)
        for item in stream:
            if item.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.COMMENT,
            }:
                continue
            value = item.string
            if item.type == tokenize.NAME:
                value = "NAME"
            elif item.type == tokenize.NUMBER:
                value = "NUMBER"
            elif item.type == tokenize.STRING:
                value = "STRING"
            tokens.append(f"{item.type}:{value}")
    except tokenize.TokenError:
        tokens = source.split()
    shingles = ["\x1f".join(tokens[index : index + 5]) for index in range(max(1, len(tokens) - 4))]
    if not shingles:
        shingles = [source]
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _simhash_bands(value: int) -> tuple[int, int, int, int]:
    return tuple((value >> (index * 16)) & 0xFFFF for index in range(4))  # type: ignore[return-value]


def _signed64(value: int) -> int:
    return value if value < 2**63 else value - 2**64


def _unsigned64(value: int) -> int:
    return value if value >= 0 else value + 2**64


def _token_shard_config(config: PyPICorpusConfig) -> CorpusTokenShardConfig:
    return CorpusTokenShardConfig(
        tokenizer_path=config.tokens.tokenizer_path,
        output_directory=config.tokens.output_directory,
        shard_index_path=config.tokens.shard_index_path,
        statistics_path=config.tokens.statistics_path,
        max_tokens_per_shard=config.tokens.max_tokens_per_shard,
        workers=config.tokens.workers,
        max_pending_tasks_per_worker=config.tokens.max_pending_tasks_per_worker,
        shard_prefix=config.tokens.shard_prefix,
        document_index_filename=config.tokens.document_index_filename,
    )


def _discovery_fingerprint(config: PyPICorpusConfig) -> str:
    payload = {
        "version": PYPI_CORPUS_VERSION,
        "selection": {
            **asdict(config.selection),
            "requirements_files": [
                [str(path), _file_hash(path) if path.is_file() else None]
                for path in config.selection.requirements_files
            ],
        },
    }
    return _json_hash(payload)


def _build_fingerprint(
    config: PyPICorpusConfig,
    downloaded: Sequence[DownloadedSdist],
    *,
    manifest_hash: str,
    tokenizer_hash: str,
) -> str:
    return _json_hash(
        {
            "version": PYPI_CORPUS_VERSION,
            "manifest": manifest_hash,
            "tokenizer": tokenizer_hash,
            "max_tokens_per_shard": config.tokens.max_tokens_per_shard,
            "shard_prefix": config.tokens.shard_prefix,
            "packages": [
                [item.package.canonical_name, item.package.version, item.package.sha256]
                for item in downloaded
            ],
        }
    )


def _validate_config(config: PyPICorpusConfig) -> None:
    if config.selection.maximum_packages <= 0:
        raise PyPICorpusError("maximum_packages must be positive.")
    if config.selection.top_downloaded_limit < 0 or config.selection.minimum_downloads < 0:
        raise PyPICorpusError("Top-package limits must be non-negative.")
    if config.selection.keyword_scan_limit < 0:
        raise PyPICorpusError("keyword_scan_limit must be non-negative.")
    if config.api.retries < 0 or config.api.request_timeout_seconds <= 0:
        raise PyPICorpusError("PyPI API timeout/retry settings are invalid.")
    if config.download.retries < 0 or config.download.workers <= 0:
        raise PyPICorpusError("Download retry/worker settings are invalid.")
    if config.download.timeout_seconds <= 0 or config.download.retry_backoff_seconds < 0:
        raise PyPICorpusError("Download timeout/backoff settings are invalid.")
    if config.extraction.workers <= 0 or config.extraction.maximum_members <= 0:
        raise PyPICorpusError("Extraction worker/member settings must be positive.")
    if config.extraction.maximum_expanded_bytes <= 0:
        raise PyPICorpusError("maximum_expanded_bytes must be positive.")
    if not 0 <= config.deduplication.near_duplicate_distance <= 16:
        raise PyPICorpusError("near_duplicate_distance must be between 0 and 16.")
    if config.tokens.max_tokens_per_shard <= 0 or config.tokens.workers <= 0:
        raise PyPICorpusError("Tokenization shard/worker settings must be positive.")
    if config.tokens.max_pending_tasks_per_worker <= 0:
        raise PyPICorpusError("max_pending_tasks_per_worker must be positive.")
    for artifact in (config.tokens.shard_index_path, config.tokens.statistics_path):
        try:
            artifact.resolve().relative_to(config.tokens.output_directory.resolve())
        except ValueError as exc:
            raise PyPICorpusError(
                f"Token artifact must be inside {config.tokens.output_directory}: {artifact}"
            ) from exc


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PyPICorpusError(f"{label} must be a YAML mapping.")
    return value


def _string_tuple(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise PyPICorpusError(f"{label} must be a list of strings.")
    result = tuple(item.strip() for item in value if item.strip())
    if not result and not allow_empty:
        raise PyPICorpusError(f"{label} must not be empty.")
    return result


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PyPICorpusError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _filename(value: object) -> str:
    result = _required_string(value, "artifact filename")
    if Path(result).name != result:
        raise PyPICorpusError("Artifact names must be plain filenames.")
    return result


def _resolve(root: Path, value: object) -> Path:
    path = Path(_required_string(value, "path")).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _portable_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _worker_count(value: int) -> int:
    if value < 0:
        raise PyPICorpusError("Worker counts must be non-negative.")
    return value or min(8, os.cpu_count() or 1)


def _supported_archive(filename: str) -> bool:
    lowered = filename.casefold()
    return any(lowered.endswith(suffix) for suffix in SUPPORTED_ARCHIVES)


def _safe_archive_filename(filename: str) -> str:
    result = _required_string(filename, "source archive filename")
    if PurePosixPath(result).name != result or Path(result).name != result:
        raise PyPICorpusError(f"Unsafe source archive filename: {result!r}")
    if not _supported_archive(result):
        raise PyPICorpusError(f"Unsupported source archive: {result!r}")
    return result


def _safe_version_component(version: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9.+_-]+", "-", version).strip(".-_")
    if not slug:
        raise PyPICorpusError(f"Unsafe package version: {version!r}")
    digest = hashlib.sha256(version.encode()).hexdigest()[:8]
    return f"{slug[:80]}-{digest}"


def _canonicalize(name: str) -> str:
    canonical = CANONICAL_NAME.sub("-", name).casefold()
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", canonical) is None:
        raise PyPICorpusError(f"Unsafe package name: {name!r}")
    return canonical


def _safe_package_directory(package: PyPIPackage) -> str:
    digest = hashlib.sha256(
        f"{package.canonical_name}=={package.version}".encode()
    ).hexdigest()[:10]
    return f"{package.canonical_name[:100]}-{digest}"


def _pypi_source_id(package: PyPIPackage) -> str:
    return f"pypi-{_safe_package_directory(package)}"


def _zip_symlink(member: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(member.external_attr >> 16)


def _check_expanded_size(size: int, settings: PyPIExtractionSettings) -> None:
    if size > settings.maximum_expanded_bytes:
        raise PyPICorpusError("Archive expanded-size limit exceeded.")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DownloadedSdist",
    "ExtractedSdist",
    "PyPIClient",
    "PyPICorpusConfig",
    "PyPICorpusError",
    "PyPICorpusResult",
    "PyPIDuplicateIndex",
    "PyPIPackage",
    "build_pypi_corpus",
    "configure_pypi_logging",
    "discover_packages",
    "load_pypi_corpus_config",
    "run_pypi_corpus_cli",
]
