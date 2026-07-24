"""Production GitHub discovery, collection, and binary token-shard pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from genpy_llm.binary_sharding import BinaryShardStatistics
from genpy_llm.code_tokenizer import tokenizer_file_hash
from genpy_llm.corpus_tokenization import (
    CorpusTokenShardConfig,
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
    collect_python_corpus,
    load_corpus_collector_config,
)
from genpy_llm.python_dataset_pipeline import ProgressBar

UTC = timezone.utc

LOGGER = logging.getLogger("genpy_llm.github_corpus_builder")
GITHUB_CORPUS_VERSION = 1
DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_TEST_EXCLUDES = (
    "test/**",
    "tests/**",
    "**/test/**",
    "**/tests/**",
    "**/fixtures/**",
    "**/testdata/**",
    "**/test_data/**",
)
DEFAULT_VENDOR_EXCLUDES = (
    "vendor/**",
    "vendors/**",
    "**/vendor/**",
    "**/vendors/**",
    "third_party/**",
    "**/third_party/**",
    "**/third-party/**",
    "**/extern/**",
    "**/external/**",
    "**/node_modules/**",
)


class GitHubCorpusError(RuntimeError):
    """Raised when GitHub discovery or corpus construction cannot continue."""


@dataclass(frozen=True)
class GitHubRepository:
    """Validated GitHub repository search result."""

    full_name: str
    html_url: str
    clone_url: str
    stars: int
    license: str
    language: str
    archived: bool
    fork: bool
    default_branch: str
    created_at: str
    updated_at: str
    pushed_at: str


@dataclass(frozen=True)
class GitHubAPISettings:
    base_url: str
    token_env: str
    request_timeout_seconds: int
    retries: int
    retry_backoff_seconds: float
    maximum_rate_limit_wait_seconds: int
    user_agent: str


@dataclass(frozen=True)
class GitHubSearchSettings:
    language: str
    minimum_stars: int
    updated_after: date
    updated_before: date
    include_archived: bool
    include_forks: bool
    allowed_licenses: tuple[str, ...]
    queries: tuple[str, ...]
    maximum_repositories: int
    results_per_page: int


@dataclass(frozen=True)
class GitHubDownloadSettings:
    cache_directory: Path
    workers: int
    timeout_seconds: int
    shallow: bool
    resume: bool
    refresh_existing: bool


@dataclass(frozen=True)
class GitHubFilterSettings:
    include_test_fixtures: bool
    exclude_vendor_code: bool
    additional_exclude: tuple[str, ...]


@dataclass(frozen=True)
class GitHubTokenSettings:
    tokenizer_path: Path
    output_directory: Path
    shard_index_path: Path
    statistics_path: Path
    max_tokens_per_shard: int
    workers: int
    max_pending_tasks_per_worker: int


@dataclass(frozen=True)
class GitHubCorpusPaths:
    checkpoint: Path
    report_directory: Path
    repositories_report: Path
    licenses_report: Path
    quality_report: Path
    rejected_report: Path
    statistics_report: Path
    log_file: Path


@dataclass(frozen=True)
class GitHubCorpusConfig:
    config_path: Path
    project_root: Path
    enabled: bool
    approval: str
    api: GitHubAPISettings
    search: GitHubSearchSettings
    download: GitHubDownloadSettings
    filters: GitHubFilterSettings
    tokens: GitHubTokenSettings
    paths: GitHubCorpusPaths
    collector: CorpusCollectorConfig
    progress: bool
    log_level: str


@dataclass(frozen=True)
class DownloadedRepository:
    repository: GitHubRepository
    checkout_path: Path
    commit_hash: str
    resumed: bool


@dataclass(frozen=True)
class GitHubCorpusResult:
    repositories_discovered: int
    repositories_downloaded: int
    repositories_failed: int
    collection: CollectionResult
    documents: int
    token_count: int
    shard_count: int
    shard_index_path: Path
    statistics_path: Path
    resumed: bool = False


def load_github_corpus_config(
    path: Path | str = "configs/dataset_pipeline.yaml",
) -> GitHubCorpusConfig:
    """Load Phase 5.5A settings alongside the existing collector configuration."""

    collector = load_corpus_collector_config(path)
    try:
        raw: Any = yaml.safe_load(collector.config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - collector parses the same file
        raise GitHubCorpusError(f"Invalid YAML in {collector.config_path}: {exc}") from exc
    section = _mapping(raw.get("github_corpus", {}), "github_corpus")
    api_raw = _mapping(section.get("api", {}), "github_corpus.api")
    search_raw = _mapping(section.get("search", {}), "github_corpus.search")
    download_raw = _mapping(section.get("download", {}), "github_corpus.download")
    filters_raw = _mapping(section.get("filters", {}), "github_corpus.filters")
    tokens_raw = _mapping(section.get("tokenization", {}), "github_corpus.tokenization")
    paths_raw = _mapping(section.get("paths", {}), "github_corpus.paths")
    today = datetime.now(UTC).date()
    updated_before = _date_value(search_raw.get("updated_before"), today)
    updated_after = _date_value(search_raw.get("updated_after"), date(2020, 1, 1))
    output_directory = _resolve(
        collector.project_root,
        tokens_raw.get("output_directory", "data/pretraining/github"),
    )
    report_directory = _resolve(
        collector.project_root,
        paths_raw.get("report_directory", "reports/github_corpus"),
    )
    config = GitHubCorpusConfig(
        config_path=collector.config_path,
        project_root=collector.project_root,
        enabled=bool(section.get("enabled", False)),
        approval=_required_string(
            section.get("approval", "Approved by configured GitHub corpus policy"),
            "github_corpus.approval",
        ),
        api=GitHubAPISettings(
            base_url=_required_string(
                api_raw.get("base_url", DEFAULT_GITHUB_API_URL),
                "github_corpus.api.base_url",
            ).rstrip("/"),
            token_env=_required_string(
                api_raw.get("token_env", "GITHUB_TOKEN"),
                "github_corpus.api.token_env",
            ),
            request_timeout_seconds=int(api_raw.get("request_timeout_seconds", 30)),
            retries=int(api_raw.get("retries", 4)),
            retry_backoff_seconds=float(api_raw.get("retry_backoff_seconds", 2.0)),
            maximum_rate_limit_wait_seconds=int(
                api_raw.get("maximum_rate_limit_wait_seconds", 300)
            ),
            user_agent=_required_string(
                api_raw.get("user_agent", "GenPy-GitHub-Corpus-Builder/1"),
                "github_corpus.api.user_agent",
            ),
        ),
        search=GitHubSearchSettings(
            language=_required_string(
                search_raw.get("language", "Python"), "github_corpus.search.language"
            ),
            minimum_stars=int(search_raw.get("minimum_stars", 100)),
            updated_after=updated_after,
            updated_before=updated_before,
            include_archived=bool(search_raw.get("include_archived", False)),
            include_forks=bool(search_raw.get("include_forks", False)),
            allowed_licenses=_string_tuple(
                search_raw.get(
                    "allowed_licenses",
                    ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"],
                ),
                "github_corpus.search.allowed_licenses",
            ),
            queries=_string_tuple(
                search_raw.get("queries", [""]),
                "github_corpus.search.queries",
                allow_empty_items=True,
            ),
            maximum_repositories=int(search_raw.get("maximum_repositories", 10_000)),
            results_per_page=int(search_raw.get("results_per_page", 100)),
        ),
        download=GitHubDownloadSettings(
            cache_directory=_resolve(
                collector.project_root,
                download_raw.get("cache_directory", "data/github_cache"),
            ),
            workers=_worker_count(int(download_raw.get("workers", 8))),
            timeout_seconds=int(download_raw.get("timeout_seconds", 600)),
            shallow=bool(download_raw.get("shallow", True)),
            resume=bool(download_raw.get("resume", True)),
            refresh_existing=bool(download_raw.get("refresh_existing", False)),
        ),
        filters=GitHubFilterSettings(
            include_test_fixtures=bool(filters_raw.get("include_test_fixtures", False)),
            exclude_vendor_code=bool(filters_raw.get("exclude_vendor_code", True)),
            additional_exclude=_string_tuple(
                filters_raw.get("additional_exclude", []),
                "github_corpus.filters.additional_exclude",
                allow_empty_list=True,
            ),
        ),
        tokens=GitHubTokenSettings(
            tokenizer_path=_resolve(
                collector.project_root,
                tokens_raw.get("tokenizer", "data/tokenizer/tokenizer.json"),
            ),
            output_directory=output_directory,
            shard_index_path=output_directory
            / _filename(tokens_raw.get("shard_index", "shard_index.json")),
            statistics_path=output_directory
            / _filename(tokens_raw.get("statistics", "tokenizer_statistics.json")),
            max_tokens_per_shard=int(tokens_raw.get("max_tokens_per_shard", 10_000_000)),
            workers=_worker_count(int(tokens_raw.get("workers", 0))),
            max_pending_tasks_per_worker=int(
                tokens_raw.get("max_pending_tasks_per_worker", 4)
            ),
        ),
        paths=GitHubCorpusPaths(
            checkpoint=_resolve(
                collector.project_root,
                paths_raw.get("checkpoint", "data/github_corpus/checkpoint.sqlite3"),
            ),
            report_directory=report_directory,
            repositories_report=report_directory / "repositories.json",
            licenses_report=report_directory / "licenses.json",
            quality_report=report_directory / "quality.json",
            rejected_report=report_directory / "rejected_files.json",
            statistics_report=report_directory / "statistics.json",
            log_file=_resolve(
                collector.project_root,
                paths_raw.get("log_file", "logs/github_corpus_builder.jsonl"),
            ),
        ),
        collector=collector,
        progress=bool(section.get("progress", raw.get("progress", True))),
        log_level=str(raw.get("logging", {}).get("level", "INFO")).upper(),
    )
    _validate_config(config)
    return config


class GitHubAPIClient:
    """Minimal authenticated GitHub REST client with retry/rate-limit handling."""

    def __init__(self, settings: GitHubAPISettings, token: str | None = None) -> None:
        self.settings = settings
        self.token = token if token is not None else os.environ.get(settings.token_env)

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def search_repositories(self, settings: GitHubSearchSettings) -> list[GitHubRepository]:
        """Search deterministically, partitioning date windows around GitHub's 1K cap."""

        repositories: dict[str, GitHubRepository] = {}
        license_queries = settings.allowed_licenses or ("",)
        for terms in settings.queries:
            for license_name in license_queries:
                if len(repositories) >= settings.maximum_repositories:
                    break
                for item in self._search_window(
                    settings,
                    terms=terms,
                    license_name=license_name,
                    start=settings.updated_after,
                    end=settings.updated_before,
                ):
                    repository = _repository_from_api(item)
                    if _repository_allowed(repository, settings):
                        repositories.setdefault(repository.full_name.casefold(), repository)
                    if len(repositories) >= settings.maximum_repositories:
                        break
        ordered = sorted(repositories.values(), key=lambda item: (-item.stars, item.full_name))
        return ordered[: settings.maximum_repositories]

    def _search_window(
        self,
        settings: GitHubSearchSettings,
        *,
        terms: str,
        license_name: str,
        start: date,
        end: date,
    ) -> Iterator[dict[str, Any]]:
        query = _search_query(settings, terms, license_name, start, end)
        first = self._request_search(query, settings, page=1)
        total = int(first.get("total_count", 0))
        if total > 1_000 and start < end:
            midpoint = start + timedelta(days=(end - start).days // 2)
            yield from self._search_window(
                settings,
                terms=terms,
                license_name=license_name,
                start=start,
                end=midpoint,
            )
            if midpoint < end:
                yield from self._search_window(
                    settings,
                    terms=terms,
                    license_name=license_name,
                    start=midpoint + timedelta(days=1),
                    end=end,
                )
            return
        if total > 1_000:
            LOGGER.warning(
                "GitHub search window %s contains %d results; API limits it to 1,000",
                start,
                total,
            )
        yield from _api_items(first)
        maximum = min(total, 1_000)
        pages = (maximum + settings.results_per_page - 1) // settings.results_per_page
        for page in range(2, pages + 1):
            yield from _api_items(self._request_search(query, settings, page=page))

    def _request_search(
        self,
        query: str,
        settings: GitHubSearchSettings,
        *,
        page: int,
    ) -> dict[str, Any]:
        return self.request_json(
            "/search/repositories",
            {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": settings.results_per_page,
                "page": page,
            },
        )

    def request_json(self, endpoint: str, parameters: Mapping[str, object]) -> dict[str, Any]:
        """Issue one retried REST request and return a JSON object."""

        query = urllib.parse.urlencode(parameters)
        url = f"{self.settings.base_url}{endpoint}?{query}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.settings.user_agent,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in range(self.settings.retries + 1):
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(  # noqa: S310
                    request,
                    timeout=self.settings.request_timeout_seconds,
                ) as response:
                    payload: Any = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise GitHubCorpusError("GitHub API response must be a JSON object.")
                return payload
            except urllib.error.HTTPError as exc:
                retry_seconds = self._retry_delay(exc, attempt)
                if retry_seconds is None:
                    details = exc.read().decode("utf-8", errors="replace")[:500]
                    raise GitHubCorpusError(
                        f"GitHub API request failed with HTTP {exc.code}: {details}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt >= self.settings.retries:
                    raise GitHubCorpusError(f"GitHub API request failed: {exc}") from exc
                retry_seconds = self.settings.retry_backoff_seconds * (2**attempt)
            LOGGER.warning(
                "GitHub API retry %d/%d in %.1fs",
                attempt + 1,
                self.settings.retries,
                retry_seconds,
            )
            time.sleep(retry_seconds)
        raise GitHubCorpusError("GitHub API retry loop ended unexpectedly.")

    def _retry_delay(
        self,
        error: urllib.error.HTTPError,
        attempt: int,
    ) -> float | None:
        if attempt >= self.settings.retries:
            return None
        if error.code not in {403, 429, 500, 502, 503, 504}:
            return None
        retry_after = error.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        elif error.code in {403, 429} and error.headers.get("X-RateLimit-Reset"):
            reset = float(error.headers["X-RateLimit-Reset"])
            delay = max(0.0, reset - time.time()) + 1.0
        else:
            delay = self.settings.retry_backoff_seconds * (2**attempt)
        if delay > self.settings.maximum_rate_limit_wait_seconds:
            raise GitHubCorpusError(
                "GitHub API rate-limit reset exceeds maximum configured wait. "
                f"Set {self.settings.token_env} for authenticated requests or resume later."
            )
        return delay


class GitHubCheckpoint:
    """SQLite-backed repository and stage checkpoint suitable for large runs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.database = sqlite3.connect(path)
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=NORMAL")
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS repositories (
                full_name TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                status TEXT NOT NULL,
                checkout_path TEXT,
                commit_hash TEXT,
                error TEXT,
                discovered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS repository_status ON repositories(status);
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

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> GitHubCheckpoint:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def upsert_discovered(self, repository: GitHubRepository) -> None:
        now = _timestamp()
        metadata = json.dumps(asdict(repository), sort_keys=True, separators=(",", ":"))
        prior = self.database.execute(
            "SELECT metadata_json, status FROM repositories WHERE full_name = ?",
            (repository.full_name,),
        ).fetchone()
        status = "discovered"
        if prior is not None:
            old = json.loads(prior[0])
            unchanged = (
                old.get("pushed_at") == repository.pushed_at
                and old.get("default_branch") == repository.default_branch
            )
            status = str(prior[1]) if unchanged else "discovered"
        self.database.execute(
            """
            INSERT INTO repositories (
                full_name, metadata_json, status, discovered_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(full_name) DO UPDATE SET
                metadata_json=excluded.metadata_json,
                status=excluded.status,
                error=NULL,
                updated_at=excluded.updated_at
            """,
            (repository.full_name, metadata, status, now, now),
        )
        self.database.commit()

    def repository_status(self, full_name: str) -> tuple[str, Path | None, str | None] | None:
        row = self.database.execute(
            "SELECT status, checkout_path, commit_hash FROM repositories WHERE full_name = ?",
            (full_name,),
        ).fetchone()
        if row is None:
            return None
        return str(row[0]), Path(row[1]) if row[1] else None, row[2]

    def mark_repository(
        self,
        full_name: str,
        status: str,
        *,
        checkout_path: Path | None = None,
        commit_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        self.database.execute(
            """
            UPDATE repositories
            SET status=?, checkout_path=?, commit_hash=?, error=?, updated_at=?
            WHERE full_name=?
            """,
            (
                status,
                str(checkout_path) if checkout_path is not None else None,
                commit_hash,
                error,
                _timestamp(),
                full_name,
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
        self,
        stage: str,
        fingerprint: str,
        status: str,
        payload: Mapping[str, Any],
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


def build_github_corpus(
    config: GitHubCorpusConfig,
    *,
    api_client: GitHubAPIClient | None = None,
    force: bool = False,
) -> GitHubCorpusResult:
    """Discover, download, validate, deduplicate, tokenize, shard, and report."""

    if not config.enabled:
        raise GitHubCorpusError(
            "github_corpus.enabled is false; review the source/license policy before enabling it."
        )
    if not config.tokens.tokenizer_path.is_file():
        raise FileNotFoundError(
            f"GenPy tokenizer not found: {config.tokens.tokenizer_path}. "
            "Run scripts/train_code_tokenizer.py first."
        )
    config.download.cache_directory.mkdir(parents=True, exist_ok=True)
    config.paths.report_directory.mkdir(parents=True, exist_ok=True)
    client = api_client or GitHubAPIClient(config.api)
    LOGGER.info(
        "github_discovery_started authenticated=%s maximum_repositories=%d",
        getattr(client, "authenticated", False),
        config.search.maximum_repositories,
    )
    repositories = client.search_repositories(config.search)
    LOGGER.info("github_discovery_completed repositories=%d", len(repositories))

    with GitHubCheckpoint(config.paths.checkpoint) as checkpoint:
        for repository in repositories:
            checkpoint.upsert_discovered(repository)
        downloaded, download_errors = _download_repositories(
            config,
            repositories,
            checkpoint,
            force=force,
        )
        sources = tuple(_corpus_source(config, item) for item in downloaded)
        collection = collect_python_corpus(
            config.collector,
            sources=sources,
            collect_manual=False,
        )
        manifest_hash = stable_manifest_fingerprint(
            config.collector.provenance_manifest, {"github"}
        )
        tokenizer_hash = tokenizer_file_hash(config.tokens.tokenizer_path)
        fingerprint = _build_fingerprint(
            config,
            downloaded,
            manifest_hash=manifest_hash,
            tokenizer_hash=tokenizer_hash,
        )
        resumed_payload = (
            checkpoint.stage_payload("binary_shards", fingerprint)
            if config.download.resume and not force
            else None
        )
        if resumed_payload is not None and binary_outputs_valid(
            _token_shard_config(config), fingerprint
        ):
            reports = _write_reports(
                config,
                repositories,
                downloaded,
                download_errors,
                collection,
                token_statistics=resumed_payload,
                resumed=True,
            )
            return GitHubCorpusResult(
                repositories_discovered=len(repositories),
                repositories_downloaded=len(downloaded),
                repositories_failed=len(download_errors),
                collection=collection,
                documents=int(resumed_payload["documents"]),
                token_count=int(resumed_payload["token_count"]),
                shard_count=int(resumed_payload["shard_count"]),
                shard_index_path=config.tokens.shard_index_path,
                statistics_path=reports,
                resumed=True,
            )
        checkpoint.mark_stage("binary_shards", fingerprint, "running", {})
        prepare_binary_output(_token_shard_config(config))
        shard_statistics, token_statistics = _tokenize_corpus(
            config,
            tokenizer_hash=tokenizer_hash,
            manifest_hash=manifest_hash,
            fingerprint=fingerprint,
        )
        checkpoint.mark_stage("binary_shards", fingerprint, "complete", token_statistics)
        reports = _write_reports(
            config,
            repositories,
            downloaded,
            download_errors,
            collection,
            token_statistics=token_statistics,
            resumed=False,
        )
    return GitHubCorpusResult(
        repositories_discovered=len(repositories),
        repositories_downloaded=len(downloaded),
        repositories_failed=len(download_errors),
        collection=collection,
        documents=shard_statistics.documents,
        token_count=shard_statistics.token_count,
        shard_count=len(shard_statistics.shards),
        shard_index_path=config.tokens.shard_index_path,
        statistics_path=reports,
    )


def _download_repositories(
    config: GitHubCorpusConfig,
    repositories: Sequence[GitHubRepository],
    checkpoint: GitHubCheckpoint,
    *,
    force: bool,
) -> tuple[list[DownloadedRepository], dict[str, str]]:
    downloaded: list[DownloadedRepository] = []
    errors: dict[str, str] = {}
    progress = ProgressBar("repositories", len(repositories), enabled=config.progress)
    futures: dict[Future[DownloadedRepository], GitHubRepository] = {}
    with ThreadPoolExecutor(
        max_workers=config.download.workers,
        thread_name_prefix="github-clone",
    ) as executor:
        for repository in repositories:
            prior = checkpoint.repository_status(repository.full_name)
            futures[
                executor.submit(
                    _download_repository,
                    repository,
                    config.download,
                    prior,
                    force,
                )
            ] = repository
        completed = 0
        for future in as_completed(futures):
            repository = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - recorded per repository
                message = str(exc)
                errors[repository.full_name] = message
                checkpoint.mark_repository(repository.full_name, "failed", error=message)
                LOGGER.warning(
                    "repository_download_failed repository=%s error=%s",
                    repository.full_name,
                    message,
                )
            else:
                downloaded.append(result)
                checkpoint.mark_repository(
                    repository.full_name,
                    "downloaded",
                    checkout_path=result.checkout_path,
                    commit_hash=result.commit_hash,
                )
            progress.update(completed)
    progress.close()
    downloaded.sort(key=lambda item: item.repository.full_name.casefold())
    return downloaded, errors


def _download_repository(
    repository: GitHubRepository,
    settings: GitHubDownloadSettings,
    prior: tuple[str, Path | None, str | None] | None,
    force: bool,
) -> DownloadedRepository:
    destination = settings.cache_directory.joinpath(*repository.full_name.split("/"))
    if (
        not force
        and settings.resume
        and not settings.refresh_existing
        and prior is not None
        and prior[0] == "downloaded"
        and prior[1] == destination
        and _valid_git_checkout(destination)
    ):
        commit = _git_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            settings.timeout_seconds,
            repository.full_name,
        )
        return DownloadedRepository(repository, destination, commit, resumed=True)

    if _valid_git_checkout(destination):
        _run_git(
            [
                "git",
                "-C",
                str(destination),
                "fetch",
                "--quiet",
                "--depth",
                "1",
                "origin",
                repository.default_branch,
            ],
            settings.timeout_seconds,
            repository.full_name,
        )
        _run_git(
            ["git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            settings.timeout_seconds,
            repository.full_name,
        )
        resumed = True
    else:
        if destination.exists():
            quarantine = destination.with_name(
                f"{destination.name}.invalid-{int(time.time())}"
            )
            os.replace(destination, quarantine)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.partial")
        if partial.exists():
            shutil.rmtree(partial)
        command = ["git", "clone", "--quiet", "--single-branch"]
        if settings.shallow:
            command.extend(["--depth", "1", "--filter=blob:none"])
        command.extend(
            ["--branch", repository.default_branch, repository.clone_url, str(partial)]
        )
        try:
            _run_git(command, settings.timeout_seconds, repository.full_name)
            os.replace(partial, destination)
        except Exception:
            if partial.exists():
                shutil.rmtree(partial)
            raise
        resumed = False
    commit = _git_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        settings.timeout_seconds,
        repository.full_name,
    )
    return DownloadedRepository(repository, destination, commit, resumed=resumed)


def _corpus_source(config: GitHubCorpusConfig, item: DownloadedRepository) -> CorpusSource:
    repository = item.repository
    exclude: list[str] = list(config.filters.additional_exclude)
    if not config.filters.include_test_fixtures:
        exclude.extend(DEFAULT_TEST_EXCLUDES)
    if config.filters.exclude_vendor_code:
        exclude.extend(DEFAULT_VENDOR_EXCLUDES)
    source_id = _github_source_id(repository.full_name)
    return CorpusSource(
        source_id=source_id,
        source_type="github",
        location=_portable_path(config.project_root, item.checkout_path),
        license=repository.license,
        approval=config.approval,
        revision=item.commit_hash,
        include=("**/*.py",),
        exclude=tuple(dict.fromkeys(exclude)),
        discovered_automatically=True,
        repository_url=repository.html_url,
        stars=repository.stars,
        repository_created_at=repository.created_at,
        repository_updated_at=repository.updated_at,
        repository_pushed_at=repository.pushed_at,
        default_branch=repository.default_branch,
    )


def _tokenize_corpus(
    config: GitHubCorpusConfig,
    *,
    tokenizer_hash: str,
    manifest_hash: str,
    fingerprint: str,
) -> tuple[BinaryShardStatistics, dict[str, Any]]:
    return build_manifest_token_shards(
        manifest_path=config.collector.provenance_manifest,
        corpus_root=config.collector.output_directory,
        source_types={"github"},
        config=_token_shard_config(config),
        tokenizer_sha256=tokenizer_hash,
        manifest_fingerprint=manifest_hash,
        build_fingerprint=fingerprint,
        progress=config.progress,
    )


def _token_shard_config(config: GitHubCorpusConfig) -> CorpusTokenShardConfig:
    return CorpusTokenShardConfig(
        tokenizer_path=config.tokens.tokenizer_path,
        output_directory=config.tokens.output_directory,
        shard_index_path=config.tokens.shard_index_path,
        statistics_path=config.tokens.statistics_path,
        max_tokens_per_shard=config.tokens.max_tokens_per_shard,
        workers=config.tokens.workers,
        max_pending_tasks_per_worker=config.tokens.max_pending_tasks_per_worker,
        shard_prefix="github_tokens",
    )


def _write_reports(
    config: GitHubCorpusConfig,
    repositories: Sequence[GitHubRepository],
    downloaded: Sequence[DownloadedRepository],
    download_errors: Mapping[str, str],
    collection: CollectionResult,
    *,
    token_statistics: Mapping[str, Any],
    resumed: bool,
) -> Path:
    downloaded_by_name = {item.repository.full_name: item for item in downloaded}
    repository_rows = []
    for repository in repositories:
        row = asdict(repository)
        row["downloaded"] = repository.full_name in downloaded_by_name
        row["download_error"] = download_errors.get(repository.full_name)
        match = downloaded_by_name.get(repository.full_name)
        row["commit_hash"] = match.commit_hash if match else None
        row["checkout_path"] = str(match.checkout_path) if match else None
        row["resumed"] = match.resumed if match else False
        repository_rows.append(row)
    _atomic_json(
        config.paths.repositories_report,
        {
            "repositories_discovered": len(repositories),
            "repositories_downloaded": len(downloaded),
            "repositories_failed": len(download_errors),
            "repositories": repository_rows,
        },
    )
    license_discovered = Counter(repository.license for repository in repositories)
    license_downloaded = Counter(item.repository.license for item in downloaded)
    _atomic_json(
        config.paths.licenses_report,
        {
            "discovered": dict(sorted(license_discovered.items())),
            "downloaded": dict(sorted(license_downloaded.items())),
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
        "tokenization_rejections": token_statistics.get("rejection_reasons", {}),
    }
    _atomic_json(config.paths.quality_report, quality)
    _atomic_json(
        config.paths.rejected_report,
        {
            "repository_download_errors": dict(sorted(download_errors.items())),
            "file_rejection_reasons": collection.rejection_reasons,
            "tokenization_rejection_reasons": token_statistics.get(
                "rejection_reasons", {}
            ),
        },
    )
    aggregate = {
        "github_corpus_version": GITHUB_CORPUS_VERSION,
        "creation_timestamp": _timestamp(),
        "configuration": str(config.config_path),
        "resumed": resumed,
        "repositories_discovered": len(repositories),
        "repositories_downloaded": len(downloaded),
        "repositories_failed": len(download_errors),
        "files_scanned": collection.files_scanned,
        "files_accepted": collection.files_accepted,
        "files_unchanged": collection.files_unchanged,
        "files_rejected": collection.files_rejected,
        **dict(token_statistics),
        "reports": {
            "repositories": str(config.paths.repositories_report),
            "licenses": str(config.paths.licenses_report),
            "quality": str(config.paths.quality_report),
            "rejected_files": str(config.paths.rejected_report),
        },
    }
    _atomic_json(config.paths.statistics_report, aggregate)
    return config.paths.statistics_report


def configure_github_logging(config: GitHubCorpusConfig, *, level: str | None = None) -> None:
    """Configure readable console logs and structured JSONL file logs."""

    setup_structured_logging(config.paths.log_file, level or config.log_level)


def run_github_corpus_cli(argv: Sequence[str] | None = None) -> int:
    """Run the Phase 5.5A GitHub corpus command."""

    parser = argparse.ArgumentParser(
        description="Build validated GitHub Python corpus and binary token shards."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_pipeline.yaml"))
    parser.add_argument("--force", action="store_true", help="Refresh clones and rebuild shards.")
    parser.add_argument("--max-repositories", type=int, default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    try:
        config = load_github_corpus_config(args.config)
        if args.max_repositories is not None:
            config = replace(
                config,
                search=replace(
                    config.search,
                    maximum_repositories=args.max_repositories,
                ),
            )
            _validate_config(config)
        configure_github_logging(config, level=args.log_level)
        result = build_github_corpus(config, force=args.force)
    except KeyboardInterrupt:
        print("Interrupted; completed repository checkpoints can be resumed.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("GitHub corpus build failed")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("GenPy Phase 5.5A GitHub corpus complete")
    print(f"Repositories discovered: {result.repositories_discovered}")
    print(f"Repositories downloaded: {result.repositories_downloaded}")
    print(f"Repositories failed: {result.repositories_failed}")
    print(f"Python files accepted: {result.collection.files_accepted}")
    print(f"Python files unchanged: {result.collection.files_unchanged}")
    print(f"Documents tokenized: {result.documents}")
    print(f"Tokens: {result.token_count}")
    print(f"Binary shards: {result.shard_count}")
    print(f"Shard index: {result.shard_index_path}")
    print(f"Statistics: {result.statistics_path}")
    print(f"Resumed: {result.resumed}")
    return 0


def _build_fingerprint(
    config: GitHubCorpusConfig,
    downloaded: Sequence[DownloadedRepository],
    *,
    manifest_hash: str,
    tokenizer_hash: str,
) -> str:
    payload = {
        "version": GITHUB_CORPUS_VERSION,
        "manifest": manifest_hash,
        "tokenizer": tokenizer_hash,
        "max_tokens_per_shard": config.tokens.max_tokens_per_shard,
        "repositories": [
            [item.repository.full_name, item.commit_hash] for item in downloaded
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _repository_from_api(item: Mapping[str, Any]) -> GitHubRepository:
    license_data = item.get("license")
    license_name = license_data.get("spdx_id") if isinstance(license_data, dict) else None
    repository = GitHubRepository(
        full_name=_required_string(item.get("full_name"), "repository full_name"),
        html_url=_required_string(item.get("html_url"), "repository html_url"),
        clone_url=_required_string(item.get("clone_url"), "repository clone_url"),
        stars=int(item.get("stargazers_count", 0)),
        license=str(license_name or "NOASSERTION"),
        language=str(item.get("language") or ""),
        archived=bool(item.get("archived", False)),
        fork=bool(item.get("fork", False)),
        default_branch=_required_string(
            item.get("default_branch", "main"), "repository default_branch"
        ),
        created_at=_required_string(item.get("created_at"), "repository created_at"),
        updated_at=_required_string(item.get("updated_at"), "repository updated_at"),
        pushed_at=_required_string(item.get("pushed_at"), "repository pushed_at"),
    )
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository.full_name) is None:
        raise GitHubCorpusError(f"Unsafe GitHub repository name: {repository.full_name!r}")
    return repository


def _repository_allowed(
    repository: GitHubRepository,
    settings: GitHubSearchSettings,
) -> bool:
    allowed_licenses = {value.casefold() for value in settings.allowed_licenses}
    return (
        repository.stars >= settings.minimum_stars
        and repository.language.casefold() == settings.language.casefold()
        and (settings.include_archived or not repository.archived)
        and (settings.include_forks or not repository.fork)
        and (not allowed_licenses or repository.license.casefold() in allowed_licenses)
    )


def _search_query(
    settings: GitHubSearchSettings,
    terms: str,
    license_name: str,
    start: date,
    end: date,
) -> str:
    qualifiers = [
        terms.strip(),
        f"language:{settings.language}",
        f"stars:>={settings.minimum_stars}",
        f"updated:{start.isoformat()}..{end.isoformat()}",
    ]
    if not settings.include_archived:
        qualifiers.append("archived:false")
    if not settings.include_forks:
        qualifiers.append("fork:false")
    if license_name:
        qualifiers.append(f"license:{license_name.casefold()}")
    return " ".join(value for value in qualifiers if value)


def _api_items(payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise GitHubCorpusError("GitHub search response items must be a list.")
    for item in items:
        if not isinstance(item, dict):
            raise GitHubCorpusError("GitHub repository result must be an object.")
        yield item


def _run_git(command: list[str], timeout: int, repository: str) -> None:
    if shutil.which("git") is None:
        raise GitHubCorpusError("Git is required for GitHub corpus cloning.")
    try:
        subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        details = getattr(exc, "stderr", "") or str(exc)
        raise GitHubCorpusError(
            f"Git operation failed for {repository}: {details.strip()}"
        ) from exc


def _git_output(command: list[str], timeout: int, repository: str) -> str:
    if shutil.which("git") is None:
        raise GitHubCorpusError("Git is required for GitHub corpus cloning.")
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        details = getattr(exc, "stderr", "") or str(exc)
        raise GitHubCorpusError(
            f"Git operation failed for {repository}: {details.strip()}"
        ) from exc
    return result.stdout.strip()


def _valid_git_checkout(path: Path) -> bool:
    return path.is_dir() and (path / ".git").is_dir()


def _github_source_id(full_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", full_name).strip(".-_").lower()
    digest = hashlib.sha256(full_name.casefold().encode("utf-8")).hexdigest()[:10]
    return f"github-{slug[:100]}-{digest}"


def _validate_config(config: GitHubCorpusConfig) -> None:
    if config.search.minimum_stars < 0:
        raise GitHubCorpusError("github_corpus.search.minimum_stars must be non-negative.")
    if config.search.updated_after > config.search.updated_before:
        raise GitHubCorpusError("GitHub updated_after must not exceed updated_before.")
    if config.search.maximum_repositories <= 0:
        raise GitHubCorpusError("maximum_repositories must be positive.")
    if not 1 <= config.search.results_per_page <= 100:
        raise GitHubCorpusError("results_per_page must be between 1 and 100.")
    if config.api.request_timeout_seconds <= 0 or config.api.retries < 0:
        raise GitHubCorpusError("GitHub API timeout/retry settings are invalid.")
    if config.api.retry_backoff_seconds < 0:
        raise GitHubCorpusError("retry_backoff_seconds must be non-negative.")
    if config.download.workers <= 0 or config.download.timeout_seconds <= 0:
        raise GitHubCorpusError("GitHub download worker/timeout settings must be positive.")
    if config.tokens.max_tokens_per_shard <= 0 or config.tokens.workers <= 0:
        raise GitHubCorpusError("GitHub tokenization worker/shard settings must be positive.")
    if config.tokens.max_pending_tasks_per_worker <= 0:
        raise GitHubCorpusError("max_pending_tasks_per_worker must be positive.")
    if config.tokens.shard_index_path == config.tokens.statistics_path:
        raise GitHubCorpusError("Shard index and statistics paths must differ.")
    for artifact in (config.tokens.shard_index_path, config.tokens.statistics_path):
        try:
            artifact.resolve().relative_to(config.tokens.output_directory.resolve())
        except ValueError as exc:
            raise GitHubCorpusError(
                f"Token artifact must be inside {config.tokens.output_directory}: {artifact}"
            ) from exc


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GitHubCorpusError(f"{label} must be a YAML mapping.")
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GitHubCorpusError(f"{label} must be a non-empty string.")
    return value.strip()


def _string_tuple(
    value: object,
    label: str,
    *,
    allow_empty_items: bool = False,
    allow_empty_list: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise GitHubCorpusError(f"{label} must be a list of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or (not allow_empty_items and not item.strip()):
            raise GitHubCorpusError(f"{label} must be a list of strings.")
        result.append(item.strip())
    if not result and not allow_empty_list:
        raise GitHubCorpusError(f"{label} must not be empty.")
    return tuple(result)


def _date_value(value: object, default: date) -> date:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise GitHubCorpusError(f"Invalid ISO date: {value}") from exc
    raise GitHubCorpusError("GitHub updated dates must be ISO dates or null.")


def _filename(value: object) -> str:
    filename = _required_string(value, "artifact filename")
    if Path(filename).name != filename:
        raise GitHubCorpusError("Artifact names must be plain filenames.")
    return filename


def _resolve(root: Path, value: object) -> Path:
    text = _required_string(value, "path")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _portable_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _worker_count(value: int) -> int:
    if value < 0:
        raise GitHubCorpusError("Worker counts must be non-negative.")
    return value or min(8, os.cpu_count() or 1)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


__all__ = [
    "DownloadedRepository",
    "GitHubAPIClient",
    "GitHubCheckpoint",
    "GitHubCorpusConfig",
    "GitHubCorpusError",
    "GitHubCorpusResult",
    "GitHubRepository",
    "build_github_corpus",
    "configure_github_logging",
    "load_github_corpus_config",
    "run_github_corpus_cli",
]
