from __future__ import annotations

import json
from pathlib import Path

import yaml

from genpy_llm.code_tokenizer import train_byte_level_bpe_tokenizer
from genpy_llm.corpus_v2.cleaner import CleanSettings, clean_document
from genpy_llm.corpus_v2.collector import collect_documents
from genpy_llm.corpus_v2.deduplicator import DeduplicationSettings, Deduplicator
from genpy_llm.corpus_v2.manifest import CollectedDocument, SourceSpec
from genpy_llm.corpus_v2.pipeline import (
    load_corpus_v2_config,
    run_corpus_v2_analysis_cli,
    run_corpus_v2_pipeline,
)
from genpy_llm.corpus_v2.quality import QualitySettings, evaluate_quality
from genpy_llm.corpus_v2.validator import ValidationSettings, validate_document


def test_corpus_v2_pipeline_builds_reports_shards_and_resumes(tmp_path: Path) -> None:
    tokenizer_path = _tokenizer(tmp_path)
    source = _source_tree(tmp_path)
    config_path = _config(tmp_path, tokenizer_path, source)

    config = load_corpus_v2_config(config_path)
    first = run_corpus_v2_pipeline(config, force=True)
    second = run_corpus_v2_pipeline(config)

    assert first.accepted_documents >= 4
    assert first.rejected_documents >= 3
    assert first.estimated_token_count == first.total_tokens
    assert first.readiness_passed is True
    assert first.duplicate_percentage > 0
    assert first.quality_report_json.is_file()
    assert first.quality_report_markdown.is_file()
    assert first.statistics_csv.is_file()
    assert first.shard_index_path.is_file()
    assert second.resumed is True

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["vocabulary_retrained"] is False
    assert manifest["readiness"]["passed"] is True
    assert manifest["statistics"]["total_tokens"] > 0
    records = [
        json.loads(line)
        for line in config.paths.document_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert {record["content_type"] for record in records} == {
        "python_code",
        "technical_text",
    }

    assert run_corpus_v2_analysis_cli(["--report-dir", str(config.paths.report_directory)]) == 0


def test_corpus_v2_components_cover_collection_cleaning_validation_and_dedup(
    tmp_path: Path,
) -> None:
    source_path = _source_tree(tmp_path)
    source = SourceSpec(
        source_id="fixture",
        source_type="local_dataset",
        path=source_path,
        include=("**/*.py", "**/*.md", "**/*.rst", "**/*.txt"),
        exclude=("**/excluded/**",),
        license="MIT",
        approval="test fixture",
    )

    collected = list(collect_documents((source,), max_file_bytes=100_000))
    names = {document.relative_path for document in collected}

    assert "package/module.py" in names
    assert "docs/guide.md" in names
    assert "archive.zip" not in names
    assert "binary.txt" not in names

    settings = CleanSettings(minimum_file_bytes=20, maximum_file_bytes=100_000)
    cleaned = [
        result.document
        for result in (clean_document(document, settings) for document in collected)
        if result.document is not None
    ]
    assert any(document and document.relative_path == "docs/guide.md" for document in cleaned)

    valid_doc = next(
        document for document in cleaned if document.relative_path == "package/module.py"
    )
    invalid_raw = CollectedDocument(
        source=source,
        path=source_path / "bad.py",
        relative_path="bad.py",
        content=b"def broken(:\n",
        content_type="python_code",
    )
    invalid_clean = clean_document(invalid_raw, CleanSettings(minimum_file_bytes=1))

    assert validate_document(valid_doc, ValidationSettings()).accepted is True
    assert invalid_clean.document is not None
    assert validate_document(invalid_clean.document, ValidationSettings()).reason == (
        "invalid_python_syntax"
    )

    quality = evaluate_quality(
        "This guide explains Python API configuration, tokens, schema, tests, and validation.",
        content_type="technical_text",
        settings=QualitySettings(minimum_technical_score=2),
    )
    assert quality.accepted is True

    deduplicator = Deduplicator(DeduplicationSettings(near_duplicate=False))
    first = deduplicator.check(valid_doc)
    duplicate = deduplicator.check(valid_doc)

    assert first.accepted is True
    assert duplicate.accepted is False
    assert duplicate.reason == "exact_duplicate"


def test_corpus_v2_readiness_fails_for_under_target(tmp_path: Path) -> None:
    tokenizer_path = _tokenizer(tmp_path)
    source = _source_tree(tmp_path)
    config_path = _config(tmp_path, tokenizer_path, source, minimum_tokens=999_999)

    result = run_corpus_v2_pipeline(load_corpus_v2_config(config_path), force=True)

    assert result.readiness_passed is False
    assert "token_target" in result.readiness_failures


def _source_tree(root: Path) -> Path:
    source = root / "source"
    (source / "package").mkdir(parents=True)
    (source / "docs").mkdir()
    (source / "build").mkdir()
    (source / "package" / "module.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    value = left + right\n"
        "    return value\n",
        encoding="utf-8",
    )
    (source / "package" / "duplicate.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    value = left + right\n"
        "    return value\n",
        encoding="utf-8",
    )
    (source / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    (source / "generated.py").write_text(
        "# This file is automatically generated. Do not edit.\nvalue = 1\n",
        encoding="utf-8",
    )
    (source / "build" / "ignored.py").write_text("def ignored():\n    return 1\n", encoding="utf-8")
    (source / "docs" / "guide.md").write_text(
        "# Tokenizer API Guide\n\n"
        "Configure Python token validation, schema checks, tests, and dataset packing.\n"
        "Use `build_corpus_v2.py` after reviewing source licenses.\n",
        encoding="utf-8",
    )
    (source / "docs" / "usage.rst").write_text(
        "Training Configuration\n======================\n\n"
        "The Python API accepts YAML config, tokenizer paths, validation reports, "
        "and checkpoint metadata for tests.\n",
        encoding="utf-8",
    )
    (source / "docs" / "notes.txt").write_text(
        "Technical notes: API config, JSON schema, validation, Python modules, "
        "token counts, and dataset tests are tracked.\n",
        encoding="utf-8",
    )
    (source / "random.txt").write_text(
        "A soft story about sunsets and feelings without implementation details.\n",
        encoding="utf-8",
    )
    (source / "binary.txt").write_bytes(b"\x00\x01\x02\x03")
    (source / "archive.zip").write_bytes(b"PK\x03\x04")
    return source


def _tokenizer(root: Path) -> Path:
    corpus = root / "tokenizer_corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "instruction": "Write Python technical code.",
                "output": (
                    "def add(left, right):\n"
                    "    return left + right\n"
                    "Tokenizer API config validation schema dataset tests.\n"
                ),
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
    return tokenizer_path


def _config(
    root: Path,
    tokenizer_path: Path,
    source: Path,
    *,
    minimum_tokens: int = 1,
) -> Path:
    config_path = root / "corpus_v2.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "project_root": ".",
                "corpus_v2": {
                    "paths": {
                        "output_directory": "out",
                        "report_directory": "reports",
                        "document_manifest": "out/document_manifest.jsonl",
                        "log_file": "logs/corpus_v2.jsonl",
                    },
                    "sources": [
                        {
                            "id": "fixture",
                            "type": "local_dataset",
                            "path": str(source),
                            "license": "MIT",
                            "approval": "test fixture",
                            "include": ["**/*.py", "**/*.md", "**/*.rst", "**/*.txt"],
                            "exclude": [],
                        }
                    ],
                    "cleaning": {
                        "minimum_file_bytes": 20,
                        "maximum_file_bytes": 100000,
                    },
                    "deduplication": {
                        "exact": True,
                        "normalized": True,
                        "near_duplicate": True,
                        "near_duplicate_threshold": 0.92,
                    },
                    "quality": {
                        "minimum_entropy": 2.0,
                        "minimum_technical_score": 2,
                        "maximum_base64_fraction": 0.4,
                        "maximum_hex_fraction": 0.4,
                        "maximum_repeated_line_fraction": 0.5,
                    },
                    "tokenization": {
                        "tokenizer": str(tokenizer_path),
                        "minimum_tokens": 1,
                    },
                    "packing": {
                        "output_directory": "out",
                        "shard_prefix": "corpus_v2",
                        "context_length": 16,
                        "max_tokens_per_shard": 128,
                        "add_eos": True,
                        "pad_final_sequence": True,
                    },
                    "readiness": {
                        "minimum_tokens": minimum_tokens,
                        "min_python_ratio": 0.01,
                        "max_python_ratio": 0.99,
                        "min_technical_text_ratio": 0.01,
                        "max_technical_text_ratio": 0.99,
                        "max_duplicate_percentage": 0.5,
                    },
                },
                "logging": {"level": "INFO"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
