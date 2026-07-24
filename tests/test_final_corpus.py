from __future__ import annotations

import json
from pathlib import Path

import yaml

from genpy_llm.code_tokenizer import CodeTokenizer, train_byte_level_bpe_tokenizer
from genpy_llm.final_corpus import (
    build_final_statistics,
    clean_final_text,
    load_final_corpus_config,
    pack_final_corpus,
    process_final_corpus,
    run_final_corpus_pipeline,
    scan_final_corpus_sources,
    write_final_metadata,
)


def test_scan_final_sources_skips_missing_and_noise(tmp_path: Path) -> None:
    config = _config(tmp_path)

    sources = scan_final_corpus_sources(config)
    names = {source.relative_path.as_posix() for source in sources}

    assert "github/repo1/pkg/module.py" in names
    assert "docs/guide.md" in names
    assert "docs/page.html" in names
    assert "cleaned/repo_copy/module.py" in names
    assert "cleaned_docs/docs/guide.txt" in names
    assert "github/repo1/.git/config.py" not in names
    assert "github/repo1/node_modules/generated.py" not in names
    assert "github/repo1/image.png" not in names


def test_cleaning_preserves_code_and_removes_doc_noise(tmp_path: Path) -> None:
    config = _config(tmp_path)
    html = """
    <html><body><nav>Next Previous Search</nav><main>
    <h1>API Guide</h1><p>Use this Python function.</p>
    <pre>def add(left, right):
    return left + right</pre></main><footer>Copyright 2026</footer></body></html>
    """

    cleaned = clean_final_text(html, ".html", config)
    code = clean_final_text("# useful comment\ndef run():\n    return True\n", ".py", config)
    unicode_text = clean_final_text("Cafe\u0301 API reference\n", ".md", config)

    assert "API Guide" in cleaned
    assert "def add(left, right):" in cleaned
    assert "Next Previous Search" not in cleaned
    assert "Copyright 2026" not in cleaned
    assert "# useful comment" in code
    assert "Caf\u00e9 API reference" in unicode_text


def test_process_deduplicates_across_all_sources_and_counts_tokens(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)

    documents, rejected = process_final_corpus(
        scan_final_corpus_sources(config),
        config,
        tokenizer,
    )
    reasons = {item.relative_path: item.reason for item in rejected}
    stats = build_final_statistics(
        len(documents) + len(rejected),
        documents,
        rejected,
        config,
        shard_index={"sequence_count": 0, "token_count": 0},
    )

    assert any(
        document.relative_path.as_posix() == "github/repo1/pkg/module.py"
        for document in documents
    )
    assert reasons["github/repo2/duplicate.py"] == "duplicate"
    assert reasons["cleaned/repo_copy/module.py"] == "duplicate"
    assert reasons["cleaned_docs/docs/guide.txt"] == "duplicate"
    assert reasons["docs/binary.txt"] == "invalid_utf8"
    assert reasons["tutorials/tiny.txt"] == "too_small"
    assert reasons["docs/huge.md"] == "too_large"
    assert reasons["docs/noisy.html"] == "too_small_after_cleaning"
    assert stats["processed_files"] == 4
    assert stats["duplicates_removed"] == 3
    assert stats["total_code_files"] == 1
    assert stats["total_documentation_files"] == 3
    assert stats["total_tokens"] > 0


def test_pack_final_corpus_reuses_train_ready_shards(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)
    documents, rejected = process_final_corpus(scan_final_corpus_sources(config), config, tokenizer)
    manifest = write_final_metadata(documents, rejected, config)

    index_path = pack_final_corpus(documents, tokenizer, config, manifest, force=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["format"] == "genpy_uint16_packed_sequence_shards"
    assert index["sequence_length"] == 17
    assert index["sequence_count"] > 0
    assert (config.packed_directory / index["shards"][0]["filename"]).is_file()


def test_full_final_corpus_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = run_final_corpus_pipeline(config, force=True)
    statistics = json.loads(result.statistics_path.read_text(encoding="utf-8"))

    assert result.total_input_files == 11
    assert result.processed_files == 4
    assert result.duplicates_removed == 3
    assert result.total_tokens > 0
    assert result.packed_sequences > 0
    assert result.statistics_path.is_file()
    assert result.manifest_path.is_file()
    assert result.shard_index_path.is_file()
    assert statistics["average_sequence_length"] == 17


def test_sliding_window_overlap_configuration_packs(tmp_path: Path) -> None:
    config = _config(tmp_path, packing_strategy="sliding_window", overlap=4)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)
    documents, rejected = process_final_corpus(scan_final_corpus_sources(config), config, tokenizer)
    manifest = write_final_metadata(documents, rejected, config)

    index_path = pack_final_corpus(documents, tokenizer, config, manifest, force=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert config.packing.packing_strategy == "sliding_window"
    assert config.packing.overlap == 4
    assert index["sequence_count"] > 0


def test_yaml_configuration_loads_context_length_alias(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    corpus = _source_tree(tmp_path)
    config_path = tmp_path / "final_corpus.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "final_corpus": {
                    "corpus_root": str(corpus),
                    "source_directories": ["github", "docs", "cleaned_docs"],
                    "tokenizer": str(tokenizer),
                    "minimum_size": 20,
                    "maximum_size": 5_000,
                    "comment_removal": True,
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

    config = load_final_corpus_config(config_path)

    assert config.corpus_root == corpus
    assert config.source_directories == (
        corpus / "github",
        corpus / "docs",
        corpus / "cleaned_docs",
    )
    assert config.tokenizer_path == tokenizer
    assert config.comment_removal is True
    assert config.packing.sequence_length == 17
    assert config.packing.context_length == 16


def _config(root: Path, *, packing_strategy: str = "packed", overlap: int = 0):
    tokenizer = _tokenizer(root)
    corpus = _source_tree(root)
    config_path = root / "final_corpus.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_root": str(root),
                "final_corpus": {
                    "corpus_root": str(corpus),
                    "output_directory": "final_corpus",
                    "source_directories": [
                        "github",
                        "docs",
                        "peps",
                        "tutorials",
                        "cleaned",
                        "cleaned_docs",
                    ],
                    "tokenizer": str(tokenizer),
                    "minimum_size": 20,
                    "maximum_size": 1_000,
                    "deduplication": True,
                    "comment_removal": False,
                    "packing": {
                        "sequence_length": 17,
                        "overlap": overlap,
                        "packing_strategy": packing_strategy,
                        "max_tokens_per_shard": 128,
                        "pad_final_sequence": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return load_final_corpus_config(config_path)


def _source_tree(root: Path) -> Path:
    corpus = root / "python_corpus"
    (corpus / "github" / "repo1" / "pkg").mkdir(parents=True)
    (corpus / "github" / "repo1" / ".git").mkdir()
    (corpus / "github" / "repo1" / "node_modules").mkdir()
    (corpus / "github" / "repo2").mkdir()
    (corpus / "docs").mkdir()
    (corpus / "peps").mkdir()
    (corpus / "tutorials").mkdir()
    (corpus / "cleaned" / "repo_copy").mkdir(parents=True)
    (corpus / "cleaned_docs" / "docs").mkdir(parents=True)

    python_text = (
        "# Keep this useful comment\n"
        "def is_even(number: int) -> bool:\n"
        "    \"\"\"Return whether a number is even.\"\"\"\n"
        "    return number % 2 == 0\n"
    )
    guide = (
        "# Python Guide\n\n"
        "Table of Contents\n\n"
        "This guide explains Python functions, APIs, tokenizers, datasets, and examples.\n\n"
        "```python\n"
        "def greet(name):\n"
        "    return f\"Hello {name}\"\n"
        "```\n"
    )
    (corpus / "github" / "repo1" / "pkg" / "module.py").write_text(python_text, encoding="utf-8")
    (corpus / "github" / "repo2" / "duplicate.py").write_text(python_text, encoding="utf-8")
    (corpus / "cleaned" / "repo_copy" / "module.py").write_text(python_text, encoding="utf-8")
    (corpus / "github" / "repo1" / ".git" / "config.py").write_text(
        "def ignored():\n    return True\n",
        encoding="utf-8",
    )
    (corpus / "github" / "repo1" / "node_modules" / "generated.py").write_text(
        "def ignored():\n    return True\n",
        encoding="utf-8",
    )
    (corpus / "github" / "repo1" / "image.png").write_bytes(b"not text")
    (corpus / "docs" / "guide.md").write_text(guide, encoding="utf-8")
    (corpus / "cleaned_docs" / "docs" / "guide.txt").write_text(
        (
            "# Python Guide\n\n"
            "This guide explains Python functions, APIs, tokenizers, datasets, and examples.\n\n"
            "```python\n"
            "def greet(name):\n"
            "    return f\"Hello {name}\"\n"
            "```\n"
        ),
        encoding="utf-8",
    )
    (corpus / "docs" / "page.html").write_text(
        "<html><body><nav>Next Previous</nav><main><h1>API Reference</h1>"
        "<p>Use this Python API to tokenize documents.</p>"
        "<pre>tokens = tokenizer.encode(text)</pre></main>"
        "<footer>Copyright 2026</footer></body></html>",
        encoding="utf-8",
    )
    (corpus / "docs" / "noisy.html").write_text(
        "<html><body><nav>"
        + ("Next Previous Search " * 20)
        + "</nav><main><p>Hi.</p></main><footer>Copyright 2026</footer></body></html>",
        encoding="utf-8",
    )
    (corpus / "docs" / "binary.txt").write_bytes(b"\xff\xfe" * 20)
    (corpus / "docs" / "huge.md").write_text("x" * 1_500, encoding="utf-8")
    (corpus / "peps" / "pep.rst").write_text(
        "PEP Example\n===========\n\nThis PEP explains Python syntax and examples.\n",
        encoding="utf-8",
    )
    (corpus / "tutorials" / "tiny.txt").write_text("tiny", encoding="utf-8")
    return corpus


def _tokenizer(root: Path) -> Path:
    corpus = root / "tokenizer_final.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "text": (
                    "def is_even(number: int) -> bool: return number % 2 == 0. "
                    "Python guide API reference tokenizer datasets examples."
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
