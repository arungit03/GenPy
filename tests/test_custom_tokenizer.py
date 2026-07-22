from __future__ import annotations

import json
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from genpy_llm.code_tokenizer import (
    SPECIAL_TOKENS,
    CodeTokenizer,
    build_tokenizer_artifacts,
    ensure_code_tokenizer,
    load_tokenizer_pipeline_config,
    tokenizer_file_hash,
    validate_python_tokenization,
)


def test_phase5_build_writes_all_artifacts_and_tokenizes_python(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)

    metadata = build_tokenizer_artifacts(config)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)

    expected = {
        "tokenizer.json",
        "code_tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens.json",
        "tokenizer_metadata.json",
        "tokenizer_statistics.json",
    }
    assert expected <= {path.name for path in config.output_directory.iterdir()}
    assert metadata.normalization == "NFC"
    assert metadata.minimum_frequency == 2
    assert tokenizer.is_phase5
    assert [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] == list(range(8))

    code = "@decorator\ndef café(value: list[int]) -> str:\n    return f'{value!r} 🐍'\n"
    decoded = tokenizer.decode(tokenizer.encode(code))
    assert decoded == unicodedata.normalize("NFC", code)
    assert all(item["round_trip"] for item in validate_python_tokenization(tokenizer))

    configuration = json.loads(
        (config.output_directory / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    statistics = json.loads(
        (config.output_directory / "tokenizer_statistics.json").read_text(encoding="utf-8")
    )
    assert configuration["vocab_size"] == 320
    assert configuration["additional_special_tokens"] == [
        "<instruction>",
        "<input>",
        "<output>",
    ]
    assert statistics["corpus_files"] == 3
    assert statistics["corpus_records"] == 3
    assert statistics["validation"]


def test_phase5_training_is_deterministic(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    first = replace(config, output_directory=tmp_path / "first")
    second = replace(config, output_directory=tmp_path / "second")

    build_tokenizer_artifacts(first)
    build_tokenizer_artifacts(second)

    assert tokenizer_file_hash(first.tokenizer_path) == tokenizer_file_hash(second.tokenizer_path)
    assert (first.output_directory / "vocab.json").read_bytes() == (
        second.output_directory / "vocab.json"
    ).read_bytes()
    assert (first.output_directory / "merges.txt").read_bytes() == (
        second.output_directory / "merges.txt"
    ).read_bytes()


def test_add_special_tokens_wraps_with_bos_and_eos(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    build_tokenizer_artifacts(config)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)

    ids = tokenizer.encode("def identity(value): return value", add_special_tokens=True)

    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id


def test_training_auto_build_uses_phase5_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _pipeline_config(tmp_path)

    tokenizer = ensure_code_tokenizer(
        tokenizer_path=config.tokenizer_path,
        metadata_path=config.metadata_path,
        project_root=tmp_path,
        vocab_size=320,
        min_frequency=2,
    )

    output = capsys.readouterr().out
    assert "Tokenizer not found. Building tokenizer..." in output
    assert "✓ Tokenizer built successfully" in output
    assert tokenizer.is_phase5
    assert (config.output_directory / "tokenizer_statistics.json").is_file()
    assert (config.output_directory / "vocab.json").is_file()
    assert (config.output_directory / "merges.txt").is_file()


def _pipeline_config(tmp_path: Path):
    corpus = tmp_path / "data" / "fine_tuning"
    corpus.mkdir(parents=True)
    records = (
        {
            "instruction": "Implement add.",
            "input": "Two integers",
            "output": "def add(a, b):\n    return a + b",
        },
        {"instruction": "Explain Point.", "input": "", "output": "class Point:\n    pass"},
        {"instruction": "Write Unicode output.", "input": "", "output": "print('வணக்கம் 🐍')"},
    )
    names = ("train.jsonl", "validation.jsonl", "test.jsonl")
    for name, record in zip(names, records, strict=True):
        (corpus / name).write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    raw = {
        "corpus": {
            "files": [f"data/fine_tuning/{name}" for name in names],
            "max_training_bytes": None,
        },
        "tokenizer": {
            "vocab_size": 320,
            "min_frequency": 2,
            "normalization": "NFC",
            "special_tokens": list(SPECIAL_TOKENS),
        },
        "training": {"seed": 42, "show_progress": False},
        "statistics": {"sample_size": None},
        "artifacts": {
            "output_directory": "data/tokenizer",
            "tokenizer": "tokenizer.json",
            "legacy_tokenizer": "code_tokenizer.json",
            "vocab": "vocab.json",
            "merges": "merges.txt",
            "tokenizer_config": "tokenizer_config.json",
            "special_tokens": "special_tokens.json",
            "metadata": "tokenizer_metadata.json",
            "statistics": "tokenizer_statistics.json",
        },
    }
    path = tmp_path / "configs" / "tokenizer.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_tokenizer_pipeline_config(path, project_root=tmp_path)
