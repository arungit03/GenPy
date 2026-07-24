from __future__ import annotations

import json
from pathlib import Path

import yaml

from genpy_llm.code_tokenizer import CodeTokenizer, train_byte_level_bpe_tokenizer
from genpy_llm.python_corpus_builder import (
    add_token_counts,
    build_statistics,
    clean_and_filter_files,
    load_python_corpus_config,
    pack_documents,
    run_python_corpus_builder,
    scan_python_corpus,
    write_metadata,
)


def test_directory_scanning_ignores_unsupported_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)

    files = scan_python_corpus(config)
    names = {file.relative_path.as_posix() for file in files}

    assert "github/repo1/package/module.py" in names
    assert "docs/guide.md" in names
    assert "peps/pep_0008.rst" in names
    assert "tutorials/lesson.txt" in names
    assert "github/repo1/.git/config.py" not in names
    assert "github/repo1/node_modules/generated.py" not in names
    assert "github/repo1/image.png" not in names


def test_filtering_deduplication_utf8_and_statistics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scanned = scan_python_corpus(config)

    documents, rejected = clean_and_filter_files(scanned, config)
    reasons = {item.relative_path: item.reason for item in rejected}
    stats = build_statistics(documents, rejected, config, shard_index=None)

    assert any(
        document.relative_path.as_posix() == "github/repo1/package/module.py"
        for document in documents
    )
    assert reasons["github/repo2/duplicate.py"] == "duplicate"
    assert reasons["github/repo2/empty.py"] == "empty_file"
    assert reasons["docs/binary.txt"] == "invalid_utf8"
    assert reasons["tutorials/tiny.txt"] == "too_small"
    assert reasons["docs/huge.md"] == "too_large"
    assert stats["number_of_repositories"] == 1
    assert stats["python_files"] == 1
    assert stats["documentation_files"] == 3
    assert stats["duplicate_count"] == 1
    assert stats["total_characters"] > 0
    assert stats["total_words"] > 0


def test_token_counting_and_packing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)
    documents, rejected = clean_and_filter_files(scan_python_corpus(config), config)
    documents = add_token_counts(documents, tokenizer)
    manifest = write_metadata(documents, rejected, config)

    index_path = pack_documents(documents, tokenizer, config, manifest, force=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert all(document.token_ids for document in documents)
    assert index["format"] == "genpy_uint16_packed_sequence_shards"
    assert index["sequence_count"] > 0
    assert (config.packed_directory / index["shards"][0]["filename"]).is_file()


def test_full_python_corpus_builder_writes_expected_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = run_python_corpus_builder(config, force=True)

    assert result.accepted_files == 4
    assert result.duplicate_count == 1
    assert result.total_tokens > 0
    assert result.statistics_path.is_file()
    assert result.metadata_path.is_file()
    assert result.shard_index_path.is_file()


def test_yaml_configuration_loads_defaults(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    source = _source_tree(tmp_path)
    config_path = tmp_path / "python_corpus.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "python_corpus": {
                    "input_directory": str(source),
                    "output_directory": str(tmp_path / "out"),
                    "tokenizer": str(tokenizer),
                    "min_file_size": 20,
                    "max_file_size": 5_000,
                    "deduplication": True,
                    "preserve_comments": False,
                    "packing": {
                        "context_length": 16,
                        "max_tokens_per_shard": 128,
                        "pad_final_sequence": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_python_corpus_config(config_path)

    assert config.input_directory == source
    assert config.output_directory == tmp_path / "out"
    assert config.tokenizer_path == tokenizer
    assert config.preserve_comments is False
    assert config.packing.context_length == 16


def _config(root: Path):
    tokenizer = _tokenizer(root)
    source = _source_tree(root)
    config_path = root / "python_corpus.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(root),
                "python_corpus": {
                    "input_directory": str(source),
                    "output_directory": str(root / "output"),
                    "tokenizer": str(tokenizer),
                    "min_file_size": 20,
                    "max_file_size": 1_000,
                    "deduplication": True,
                    "preserve_comments": True,
                    "packing": {
                        "context_length": 16,
                        "max_tokens_per_shard": 128,
                        "pad_final_sequence": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return load_python_corpus_config(config_path)


def _source_tree(root: Path) -> Path:
    source = root / "python_corpus"
    (source / "github" / "repo1" / "package").mkdir(parents=True)
    (source / "github" / "repo1" / ".git").mkdir()
    (source / "github" / "repo1" / "node_modules").mkdir()
    (source / "github" / "repo2").mkdir()
    (source / "docs").mkdir()
    (source / "peps").mkdir()
    (source / "tutorials").mkdir()

    python_text = (
        "# Keep this useful comment\n"
        "def is_even(number: int) -> bool:\n"
        "    \"\"\"Return whether a number is even.\"\"\"\n"
        "    return number % 2 == 0\n"
    )
    (source / "github" / "repo1" / "package" / "module.py").write_text(
        python_text,
        encoding="utf-8",
    )
    (source / "github" / "repo2" / "duplicate.py").write_text(python_text, encoding="utf-8")
    (source / "github" / "repo2" / "empty.py").write_text("", encoding="utf-8")
    (source / "github" / "repo1" / ".git" / "config.py").write_text(
        "def ignored():\n    return True\n",
        encoding="utf-8",
    )
    (source / "github" / "repo1" / "node_modules" / "generated.py").write_text(
        "def ignored():\n    return True\n",
        encoding="utf-8",
    )
    (source / "github" / "repo1" / "image.png").write_bytes(b"not text")
    (source / "docs" / "guide.md").write_text(
        "# Guide\n\nPython corpus building validates UTF-8 text and packs tokens for training.\n",
        encoding="utf-8",
    )
    (source / "docs" / "binary.txt").write_bytes(b"\xff\xfe\x00" * 10)
    (source / "docs" / "huge.md").write_text("x" * 1_500, encoding="utf-8")
    (source / "peps" / "pep_0008.rst").write_text(
        "PEP Guide\n=========\n\nPython style documentation with examples and explanations.\n",
        encoding="utf-8",
    )
    (source / "tutorials" / "lesson.txt").write_text(
        "Tutorial lesson with Python functions, tokenizers, datasets, and examples.\n",
        encoding="utf-8",
    )
    (source / "tutorials" / "tiny.txt").write_text("tiny", encoding="utf-8")
    return source


def _tokenizer(root: Path) -> Path:
    corpus = root / "tokenizer_corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "text": (
                    "def is_even(number: int) -> bool:\n"
                    "    return number % 2 == 0\n"
                    "Python corpus building validates UTF-8 text and packs tokens.\n"
                    "Tutorial lesson with tokenizers and datasets.\n"
                )
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
