from __future__ import annotations

import json
from pathlib import Path

import yaml

from genpy_llm.code_tokenizer import CodeTokenizer, train_byte_level_bpe_tokenizer
from genpy_llm.documentation_corpus import (
    build_documentation_statistics,
    clean_documentation_text,
    ensure_documentation_folders,
    load_documentation_corpus_config,
    pack_documentation,
    process_documentation,
    run_documentation_corpus_pipeline,
    scan_documentation_sources,
    write_documentation_metadata,
)


def test_ensure_folders_and_scanning_ignore_noise(tmp_path: Path) -> None:
    config = _config(tmp_path)
    ensure_documentation_folders(config)

    sources = scan_documentation_sources(config)
    names = {source.relative_path.as_posix() for source in sources}

    assert (tmp_path / "python_corpus" / "docs").is_dir()
    assert (tmp_path / "python_corpus" / "peps").is_dir()
    assert (tmp_path / "python_corpus" / "tutorials").is_dir()
    assert "docs/guide.md" in names
    assert "docs/page.html" in names
    assert "peps/pep.rst" in names
    assert "tutorials/tutorial.txt" in names
    assert "docs/_build/ignored.md" not in names
    assert "docs/image.png" not in names


def test_cleaner_removes_navigation_and_preserves_code() -> None:
    html = """
    <html>
      <head><style>.hidden{}</style><script>alert(1)</script></head>
      <body>
        <nav>Previous Next Search</nav>
        <header>Header menu</header>
        <main>
          <h1>Python API Guide</h1>
          <p>This tutorial explains useful Python functions.</p>
          <pre>def add(left, right):
    return left + right</pre>
          <table><tr><td>name</td><td>value</td></tr></table>
        </main>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """

    cleaned = clean_documentation_text(html, ".html")

    assert "Python API Guide" in cleaned
    assert "def add(left, right):" in cleaned
    assert "return left + right" in cleaned
    assert "name" in cleaned and "value" in cleaned
    assert "Previous Next Search" not in cleaned
    assert "Copyright 2026" not in cleaned
    assert "alert" not in cleaned


def test_filtering_deduplication_utf8_statistics_and_packing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)
    sources = scan_documentation_sources(config)

    documents, rejected = process_documentation(sources, config, tokenizer)
    manifest = write_documentation_metadata(documents, rejected, config)
    shard_index_path = pack_documentation(documents, tokenizer, config, manifest, force=True)
    shard_index = json.loads(shard_index_path.read_text(encoding="utf-8"))
    stats = build_documentation_statistics(documents, rejected, config, shard_index)
    reasons = {item.relative_path: item.reason for item in rejected}

    assert len(documents) == 4
    assert reasons["docs/z_duplicate.md"] == "duplicate"
    assert reasons["docs/binary.txt"] == "invalid_utf8"
    assert reasons["tutorials/tiny.txt"] == "too_small"
    assert stats["files_processed"] == 4
    assert stats["files_skipped"] == 3
    assert stats["duplicates_removed"] == 1
    assert stats["total_characters"] > 0
    assert stats["total_tokens"] > 0
    assert stats["average_tokens_per_document"] > 0
    assert shard_index["sequence_count"] > 0
    assert (config.packed_directory / shard_index["shards"][0]["filename"]).is_file()


def test_full_documentation_corpus_pipeline(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = run_documentation_corpus_pipeline(config, force=True)

    assert result.files_processed == 4
    assert result.files_skipped == 3
    assert result.duplicates_removed == 1
    assert result.total_tokens > 0
    assert result.statistics_path.is_file()
    assert result.manifest_path.is_file()
    assert result.shard_index_path.is_file()


def test_empty_documentation_folders_produce_empty_reports(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    corpus = tmp_path / "python_corpus"
    config_path = tmp_path / "documentation_corpus.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "documentation_corpus": {
                    "corpus_root": str(corpus),
                    "source_directories": ["docs", "peps", "tutorials"],
                    "tokenizer": str(tokenizer),
                    "min_file_size": 20,
                    "max_file_size": 5_000,
                    "packing": {
                        "context_length": 16,
                        "max_tokens_per_shard": 128,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_documentation_corpus_pipeline(
        load_documentation_corpus_config(config_path),
        force=True,
    )
    index = json.loads(result.shard_index_path.read_text(encoding="utf-8"))

    assert result.files_processed == 0
    assert result.total_tokens == 0
    assert index["sequence_count"] == 0
    assert index["shards"] == []


def test_yaml_configuration_loads(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    corpus = _documentation_tree(tmp_path)
    config_path = tmp_path / "documentation_corpus.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "documentation_corpus": {
                    "corpus_root": str(corpus),
                    "source_directories": ["docs", "peps", "tutorials"],
                    "tokenizer": str(tokenizer),
                    "min_file_size": 20,
                    "max_file_size": 5_000,
                    "deduplication": True,
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

    config = load_documentation_corpus_config(config_path)

    assert config.corpus_root == corpus
    assert config.source_directories == (
        corpus / "docs",
        corpus / "peps",
        corpus / "tutorials",
    )
    assert config.tokenizer_path == tokenizer
    assert config.packing.context_length == 16


def _config(root: Path):
    tokenizer = _tokenizer(root)
    corpus = _documentation_tree(root)
    config_path = root / "documentation_corpus.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(root),
                "documentation_corpus": {
                    "corpus_root": str(corpus),
                    "source_directories": ["docs", "peps", "tutorials"],
                    "tokenizer": str(tokenizer),
                    "min_file_size": 20,
                    "max_file_size": 5_000,
                    "deduplication": True,
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
    return load_documentation_corpus_config(config_path)


def _documentation_tree(root: Path) -> Path:
    corpus = root / "python_corpus"
    (corpus / "docs" / "_build").mkdir(parents=True)
    (corpus / "peps").mkdir(parents=True)
    (corpus / "tutorials").mkdir(parents=True)
    markdown = (
        "# Python Tutorial\n\n"
        "Table of Contents\n\n"
        "This guide explains Python functions and API usage.\n\n"
        "```python\n"
        "def greet(name):\n"
        "    return f\"Hello {name}\"\n"
        "```\n"
    )
    (corpus / "docs" / "guide.md").write_text(markdown, encoding="utf-8")
    (corpus / "docs" / "z_duplicate.md").write_text(markdown, encoding="utf-8")
    (corpus / "docs" / "page.html").write_text(
        "<html><body><nav>Next Previous</nav><main><h1>API Reference</h1>"
        "<p>Use this Python API to tokenize documents.</p>"
        "<pre>tokens = tokenizer.encode(text)</pre></main>"
        "<footer>Copyright 2026</footer></body></html>",
        encoding="utf-8",
    )
    (corpus / "docs" / "_build" / "ignored.md").write_text(
        "# Ignored\n\nThis generated page should not be scanned.\n",
        encoding="utf-8",
    )
    (corpus / "docs" / "image.png").write_bytes(b"not text")
    (corpus / "docs" / "binary.txt").write_bytes(b"\xff\xfe" * 20)
    (corpus / "peps" / "pep.rst").write_text(
        "PEP Example\n===========\n\n"
        ".. contents::\n"
        "   :local:\n\n"
        "This PEP explains Python syntax and documentation examples.\n\n"
        "::\n\n"
        "    value = 42\n",
        encoding="utf-8",
    )
    (corpus / "tutorials" / "tutorial.txt").write_text(
        "Tutorial text explaining Python modules, functions, examples, tables, and APIs.\n",
        encoding="utf-8",
    )
    (corpus / "tutorials" / "tiny.txt").write_text("tiny", encoding="utf-8")
    return corpus


def _tokenizer(root: Path) -> Path:
    corpus = root / "tokenizer_docs.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "text": (
                    "Python tutorial API reference tokenize documents. "
                    "def greet(name): return name. value = 42."
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
