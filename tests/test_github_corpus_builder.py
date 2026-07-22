from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest
import yaml

from genpy_llm.binary_sharding import (
    BinaryTokenShardWriter,
    read_binary_tokens,
    write_binary_shard_index,
)
from genpy_llm.code_tokenizer import (
    CodeTokenizer,
    tokenizer_file_hash,
    train_byte_level_bpe_tokenizer,
)
from genpy_llm.github_corpus_builder import (
    GitHubAPIClient,
    GitHubRepository,
    build_github_corpus,
    load_github_corpus_config,
)
from genpy_llm.streaming_dataset import StreamingGPTDataset


class _FixtureAPI:
    authenticated = True

    def __init__(self, repository: GitHubRepository) -> None:
        self.repository = repository

    def search_repositories(self, _settings):
        return [self.repository]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_github_pipeline_reuses_collector_tracks_metadata_and_resumes(tmp_path: Path) -> None:
    remote, branch, revision = _git_repository(tmp_path)
    config = _config(tmp_path)
    repository = GitHubRepository(
        full_name="approved/example",
        html_url="https://github.com/approved/example",
        clone_url=str(remote),
        stars=250,
        license="MIT",
        language="Python",
        archived=False,
        fork=False,
        default_branch=branch,
        created_at="2021-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        pushed_at="2026-01-01T00:00:00Z",
    )

    first = build_github_corpus(config, api_client=_FixtureAPI(repository))

    assert first.repositories_discovered == 1
    assert first.repositories_downloaded == 1
    assert first.repositories_failed == 0
    assert first.collection.files_accepted == 1
    assert first.collection.files_rejected == 3
    assert first.collection.rejection_reasons == {
        "duplicate_content": 1,
        "generated_file": 1,
        "invalid_python_syntax": 1,
    }
    assert first.documents == 1
    assert first.token_count > 1
    assert first.shard_count >= 1
    assert first.shard_index_path.is_file()

    manifest = [
        json.loads(line)
        for line in config.collector.provenance_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(manifest) == 1
    source = manifest[0]["source"]
    assert source["type"] == "github"
    assert source["repository_url"] == repository.html_url
    assert source["stars"] == 250
    assert source["commit_hash"] == revision
    assert source["revision"] == revision
    assert manifest[0]["license"] == "MIT"
    assert "tests" not in manifest[0]["source_path"]
    assert "vendor" not in manifest[0]["source_path"]

    shard_index = json.loads(first.shard_index_path.read_text(encoding="utf-8"))
    assert shard_index["format"] == "genpy_uint16_token_shards"
    assert shard_index["vocab_size"] == 320
    assert shard_index["documents"] == 1
    assert shard_index["source_manifest_sha256"]
    assert (config.tokens.output_directory / "document_index.jsonl").is_file()
    assert config.paths.repositories_report.is_file()
    assert config.paths.licenses_report.is_file()
    assert config.paths.quality_report.is_file()
    assert config.paths.rejected_report.is_file()

    second = build_github_corpus(config, api_client=_FixtureAPI(repository))

    assert second.resumed is True
    assert second.collection.files_unchanged == 1
    report = json.loads(config.paths.repositories_report.read_text(encoding="utf-8"))
    assert report["repositories"][0]["resumed"] is True


def test_binary_shards_feed_existing_streaming_dataset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tokenizer = CodeTokenizer.from_file(config.tokens.tokenizer_path)
    output = tmp_path / "binary"
    writer = BinaryTokenShardWriter(output, max_tokens_per_shard=8)
    first_ids = tokenizer.encode("def first():\n    return 1\n") + [tokenizer.eos_token_id]
    second_ids = tokenizer.encode("def second():\n    return 2\n") + [tokenizer.eos_token_id]
    writer.write_document(first_ids, {"stored_path": "first.py"})
    writer.write_document(second_ids, {"stored_path": "second.py"})
    statistics = writer.close()
    write_binary_shard_index(
        output / "shard_index.json",
        statistics,
        tokenizer_path=config.tokens.tokenizer_path,
        tokenizer_sha256=tokenizer_file_hash(config.tokens.tokenizer_path),
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        source_manifest=tmp_path / "manifest.jsonl",
        creation_timestamp="2026-01-01T00:00:00Z",
    )

    raw_ids = []
    for shard in statistics.shards:
        raw_ids.extend(read_binary_tokens(output / shard.filename))
    dataset = StreamingGPTDataset(
        output / "*.bin",
        tokenizer,
        context_length=4,
        stride=4,
        pack_across_files=True,
        incomplete_window_policy="pad",
    )
    samples = list(dataset)

    assert raw_ids == first_ids + second_ids
    assert samples
    assert samples[0]["input_ids"].tolist() == raw_ids[:4]
    assert samples[0]["target_ids"].tolist() == raw_ids[1:5]


def test_search_partitions_large_date_windows_and_applies_filters(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class RecordingClient(GitHubAPIClient):
        def __init__(self) -> None:
            super().__init__(config.api, token="secret")
            self.queries: list[str] = []

        def _request_search(self, query, settings, *, page):
            self.queries.append(query)
            if "2026-01-01..2026-01-04" in query:
                return {"total_count": 1001, "items": []}
            return {"total_count": 1, "items": [_api_repository()]}

    client = RecordingClient()
    settings = config.search
    settings = type(settings)(
        **{
            **settings.__dict__,
            "updated_after": date(2026, 1, 1),
            "updated_before": date(2026, 1, 4),
        }
    )

    repositories = client.search_repositories(settings)

    assert len(repositories) == 1
    assert len(client.queries) == 3
    assert all("language:Python" in query for query in client.queries)
    assert all("stars:>=100" in query for query in client.queries)
    assert all("archived:false" in query for query in client.queries)
    assert all("license:mit" in query for query in client.queries)


def _config(root: Path):
    corpus = root / "tokenizer_corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "instruction": "Implement a function.",
                "output": "def value():\n    return 1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer_path = root / "tokenizer" / "tokenizer.json"
    train_byte_level_bpe_tokenizer(
        [corpus],
        output_path=tokenizer_path,
        metadata_path=tokenizer_path.with_name("tokenizer_metadata.json"),
        vocab_size=320,
        min_frequency=1,
        show_progress=False,
    )
    payload = {
        "version": 1,
        "project_root": ".",
        "progress": False,
        "corpus_collection": {
            "output_directory": "raw",
            "provenance_manifest": "raw/collection_manifest.jsonl",
            "report": "raw/collection_report.json",
            "log_file": "logs/collector.log",
            "minimum_file_bytes": 5,
            "maximum_file_bytes": 10000,
            "sources": [],
        },
        "github_corpus": {
            "enabled": True,
            "approval": "Approved test policy",
            "progress": False,
            "search": {
                "language": "Python",
                "minimum_stars": 100,
                "updated_after": "2026-01-01",
                "updated_before": "2026-01-04",
                "include_archived": False,
                "include_forks": False,
                "allowed_licenses": ["MIT"],
                "queries": [""],
                "maximum_repositories": 100,
                "results_per_page": 100,
            },
            "download": {
                "cache_directory": "cache",
                "workers": 2,
                "timeout_seconds": 30,
                "shallow": False,
                "resume": True,
                "refresh_existing": False,
            },
            "filters": {
                "include_test_fixtures": False,
                "exclude_vendor_code": True,
                "additional_exclude": [],
            },
            "tokenization": {
                "tokenizer": "tokenizer/tokenizer.json",
                "output_directory": "binary",
                "max_tokens_per_shard": 20,
                "workers": 2,
                "max_pending_tasks_per_worker": 2,
            },
            "paths": {
                "checkpoint": "state/checkpoint.sqlite3",
                "report_directory": "reports",
                "log_file": "logs/github.jsonl",
            },
        },
        "logging": {"level": "INFO"},
    }
    path = root / "dataset_pipeline.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_github_corpus_config(path)


def _git_repository(root: Path) -> tuple[Path, str, str]:
    repository = root / "remote"
    (repository / "pkg").mkdir(parents=True)
    code = "def value(number: int) -> int:\n    return number + 1\n"
    (repository / "pkg" / "main.py").write_text(code, encoding="utf-8")
    (repository / "pkg" / "copy.py").write_text(code, encoding="utf-8")
    (repository / "pkg" / "client_generated.py").write_text(
        "# generated by fixture\nvalue = 1\n", encoding="utf-8"
    )
    (repository / "pkg" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (repository / "tests").mkdir()
    (repository / "tests" / "test_value.py").write_text("assert True\n", encoding="utf-8")
    (repository / "vendor").mkdir()
    (repository / "vendor" / "vendored.py").write_text("value = 2\n", encoding="utf-8")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "github-builder@example.invalid")
    _git(repository, "config", "user.name", "GitHub Builder Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    branch = _git(repository, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    revision = _git(repository, "rev-parse", "HEAD").stdout.strip()
    return repository, branch, revision


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _api_repository() -> dict[str, object]:
    return {
        "full_name": "approved/example",
        "html_url": "https://github.com/approved/example",
        "clone_url": "https://github.com/approved/example.git",
        "stargazers_count": 250,
        "license": {"spdx_id": "MIT"},
        "language": "Python",
        "archived": False,
        "fork": False,
        "default_branch": "main",
        "created_at": "2021-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "pushed_at": "2026-01-02T00:00:00Z",
    }
