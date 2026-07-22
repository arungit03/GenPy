from __future__ import annotations

import gzip
import hashlib
import json
from array import array
from pathlib import Path

import yaml

from genpy_llm.code_tokenizer import train_byte_level_bpe_tokenizer
from genpy_llm.corpus_merger import build_pretraining_corpus, load_pretraining_config
from genpy_llm.global_deduplicator import GlobalDeduplicationConfig, GlobalDeduplicator
from genpy_llm.sequence_packer import SequencePacker, SequencePackingConfig


def test_pretraining_corpus_merges_deduplicates_packs_shards_and_resumes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    first = build_pretraining_corpus(config, force=True)

    assert first.accepted_files == 2
    assert first.rejected_files == 2
    assert first.duplicates_removed == 1
    assert first.total_tokens > 0
    assert first.training_sequences > 0
    assert first.shard_count >= 1
    assert first.manifest_path.is_file()
    assert first.index_path.is_file()

    index = json.loads(config.paths.index.read_text(encoding="utf-8"))
    assert index["format"] == "genpy_uint16_packed_sequence_shards"
    assert index["context_length"] == 4
    assert index["sequence_length"] == 5
    assert index["sequence_count"] == first.training_sequences
    first_shard = index["shards"][0]
    assert (config.paths.output_directory / first_shard["filename"]).is_file()
    assert (config.paths.output_directory / first_shard["metadata_filename"]).is_file()
    assert _read_uint16(config.paths.output_directory / first_shard["filename"])

    with gzip.open(
        config.paths.output_directory / first_shard["metadata_filename"],
        "rt",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)
    assert metadata["format"] == "genpy_sequence_shard_metadata"
    assert metadata["sequences"][0]["documents"]

    manifest = json.loads(config.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["accepted_files"] == 2
    assert manifest["duplicates_removed"] == 1
    assert manifest["shard_count"] == first.shard_count
    assert manifest["number_of_repositories"] == 1
    assert manifest["number_of_packages"] == 1

    records = [
        json.loads(line)
        for line in config.paths.merged_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 2
    assert all("_text" not in record for record in records)
    assert all(record["validation_status"] == "accepted" for record in records)
    assert all(record["token_count"] > 0 for record in records)

    for report in (
        config.paths.corpus_report,
        config.paths.quality_report,
        config.paths.duplicate_report,
        config.paths.validation_report,
        config.paths.license_report,
        config.paths.source_report,
        config.paths.report_statistics,
        config.paths.token_statistics,
        config.paths.shard_statistics,
    ):
        assert report.is_file()
    duplicate_report = json.loads(config.paths.duplicate_report.read_text(encoding="utf-8"))
    assert duplicate_report["reasons"] == {"exact_sha256_duplicate": 1}
    validation_report = json.loads(config.paths.validation_report.read_text(encoding="utf-8"))
    assert validation_report["rejection_reasons"]["invalid_python_syntax"] == 1

    second = build_pretraining_corpus(config)

    assert second.resumed is True
    assert second.training_sequences == first.training_sequences


def test_sequence_packer_tracks_document_offsets_and_padding() -> None:
    packer = SequencePacker(
        SequencePackingConfig(context_length=2, pad_final_sequence=True),
        pad_token_id=0,
    )

    complete = packer.add_document([1, 2, 3], {"stored_path": "a.py"})
    final = packer.finish()

    assert complete[0].token_ids == [1, 2, 3]
    assert complete[0].document_offsets[0]["sequence_token_start"] == 0
    assert complete[0].document_offsets[0]["sequence_token_end"] == 3
    assert final == []


def test_global_deduplicator_supports_comment_and_ast_normalization() -> None:
    comment_config = GlobalDeduplicationConfig(
        exact_sha256=False,
        whitespace_normalization=True,
        comment_normalization=True,
        newline_normalization=True,
        ast_normalization=False,
    )
    deduplicator = GlobalDeduplicator(comment_config)
    first = _record("a.py", "a" * 64)
    second = _record("b.py", "b" * 64)

    assert deduplicator.check(first, "def value():\n    return 1\n") is None
    duplicate = deduplicator.check(
        second,
        "# comment\n\ndef value():\n    return 1\n",
    )

    assert duplicate is not None
    assert duplicate.reason == "normalized_duplicate"

    ast_config = GlobalDeduplicationConfig(
        exact_sha256=False,
        whitespace_normalization=False,
        comment_normalization=False,
        newline_normalization=False,
        ast_normalization=True,
    )
    ast_deduplicator = GlobalDeduplicator(ast_config)
    assert ast_deduplicator.check(first, "def value():\n    return 1\n") is None
    ast_duplicate = ast_deduplicator.check(
        second,
        "def value():\n\n    return 1\n",
    )
    assert ast_duplicate is not None
    assert ast_duplicate.reason == "ast_duplicate"


def _config(root: Path):
    raw = root / "raw"
    raw.mkdir()
    github_content = "def add(a, b):\n    return a + b\n"
    pypi_content = "class Counter:\n    def value(self):\n        return 1\n"
    invalid_content = "def broken(:\n"
    _write(raw / "github_repo" / "math.py", github_content)
    _write(raw / "github_duplicate" / "math.py", github_content)
    _write(raw / "pypi_package" / "counter.py", pypi_content)
    _write(raw / "invalid" / "broken.py", invalid_content)
    manifest = raw / "collection_manifest.jsonl"
    records = [
        _manifest_record(
            "github_repo/math.py",
            "src/math.py",
            github_content,
            {
                "id": "github_repo",
                "type": "github",
                "location": "https://github.com/acme/repo",
                "repository_url": "https://github.com/acme/repo",
                "revision": "abc123",
            },
            "MIT",
        ),
        _manifest_record(
            "github_duplicate/math.py",
            "copy/math.py",
            github_content,
            {
                "id": "github_duplicate",
                "type": "github",
                "location": "https://github.com/acme/duplicate",
                "repository_url": "https://github.com/acme/duplicate",
                "revision": "def456",
            },
            "MIT",
        ),
        _manifest_record(
            "pypi_package/counter.py",
            "pkg/counter.py",
            pypi_content,
            {
                "id": "pypi_package",
                "type": "pypi",
                "location": "https://files.pythonhosted.org/pkg.tar.gz",
                "package": "demo-package",
                "version": "1.0.0",
                "download_url": "https://files.pythonhosted.org/pkg.tar.gz",
            },
            "Apache-2.0",
        ),
        _manifest_record(
            "invalid/broken.py",
            "invalid/broken.py",
            invalid_content,
            {"id": "invalid", "type": "local", "location": "data/imports/invalid"},
            None,
        ),
    ]
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    tokenizer_path = _tokenizer(root)
    config_path = root / "pretraining.yaml"
    config_payload = {
        "version": 1,
        "project_root": ".",
        "progress": False,
        "pretraining_corpus": {
            "enabled": True,
            "progress": False,
            "resume": True,
            "source_types": ["github", "pypi", "local"],
            "paths": {
                "source_manifest": "raw/collection_manifest.jsonl",
                "corpus_root": "raw",
                "corpus_index": None,
                "output_directory": "pretraining",
                "merged_manifest": "pretraining/corpus_manifest.jsonl",
                "index": "pretraining/index.json",
                "manifest": "pretraining/manifest.json",
                "statistics": "pretraining/statistics.json",
                "report_directory": "reports",
                "checkpoint": "pretraining/checkpoint.json",
                "log_file": "logs/pretraining.jsonl",
            },
            "validation": {
                "minimum_file_bytes": 1,
                "maximum_file_bytes": 10000,
                "require_python_syntax": True,
                "cleaner": {"require_known_license": False},
            },
            "deduplication": {
                "exact_sha256": True,
                "whitespace_normalization": False,
                "comment_normalization": False,
                "newline_normalization": True,
                "ast_normalization": False,
            },
            "tokenization": {
                "tokenizer": str(tokenizer_path.relative_to(root)),
                "workers": 1,
                "max_pending_tasks_per_worker": 2,
            },
            "packing": {
                "context_length": 4,
                "add_bos": True,
                "add_eos": True,
                "document_boundary": "eos",
                "pad_final_sequence": True,
            },
            "shards": {
                "prefix": "shard",
                "max_tokens_per_shard": 10,
                "compression": "metadata_gzip",
            },
        },
        "logging": {"level": "INFO"},
    }
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")
    return load_pretraining_config(config_path)


def _tokenizer(root: Path) -> Path:
    corpus = root / "tokenizer_corpus.jsonl"
    corpus.write_text(
        json.dumps({"instruction": "Write code.", "output": "def add(a, b):\n    return a + b"})
        + "\n"
        + json.dumps({"instruction": "Write class.", "output": "class Counter:\n    pass"})
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
    return tokenizer_path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _manifest_record(
    stored_path: str,
    source_path: str,
    content: str,
    source: dict,
    license_value: str | None,
) -> dict:
    return {
        "stored_path": stored_path,
        "source_path": source_path,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "license": license_value,
        "collection_timestamp": "2026-01-01T00:00:00+00:00",
        "source": source,
    }


def _record(stored_path: str, digest: str) -> dict:
    return {
        "stored_path": stored_path,
        "source_path": stored_path,
        "content_sha256": digest,
        "license": "MIT",
        "source": {"id": "test", "type": "local", "location": "fixture"},
    }


def _read_uint16(path: Path) -> list[int]:
    values = array("H")
    with path.open("rb") as file:
        values.fromfile(file, path.stat().st_size // 2)
    return list(values)
