from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from genpy_llm.config import ConfigError, PreprocessingConfig, load_config
from genpy_llm.preprocessing import TextPreprocessor


def make_config(**overrides: object) -> PreprocessingConfig:
    values = {
        "unicode_normalization": "NFKC",
        "lowercase": False,
        "normalize_whitespace": True,
        "preserve_newlines": True,
        "remove_control_characters": True,
        "remove_empty_lines": True,
        "strip_lines": True,
        "min_line_length": 1,
        "max_line_length": None,
    }
    values.update(overrides)
    return PreprocessingConfig(**values)


def test_unicode_normalization() -> None:
    preprocessor = TextPreprocessor(make_config(unicode_normalization="NFKC"))

    assert preprocessor.clean_text("ＡＢＣ") == "ABC"


def test_repeated_whitespace_normalization() -> None:
    preprocessor = TextPreprocessor(make_config())

    assert preprocessor.clean_text("Hello     World!\tAgain") == "Hello World! Again"


def test_lowercase_enabled() -> None:
    preprocessor = TextPreprocessor(make_config(lowercase=True))

    assert preprocessor.clean_text("Hello WORLD") == "hello world"


def test_lowercase_disabled() -> None:
    preprocessor = TextPreprocessor(make_config(lowercase=False))

    assert preprocessor.clean_text("Hello WORLD") == "Hello WORLD"


def test_empty_line_removal() -> None:
    preprocessor = TextPreprocessor(make_config())

    assert preprocessor.clean_text("First\n\n   \nSecond") == "First\nSecond"


def test_newline_preservation() -> None:
    preprocessor = TextPreprocessor(make_config(preserve_newlines=True))

    assert preprocessor.clean_text("First line\nSecond line") == "First line\nSecond line"


def test_control_character_removal() -> None:
    preprocessor = TextPreprocessor(make_config())

    assert preprocessor.clean_text("a\x00b\x1fc") == "abc"


def test_tamil_unicode_preservation() -> None:
    tamil_text = "வணக்கம்! இது தமிழ் உரை."
    preprocessor = TextPreprocessor(make_config())

    assert preprocessor.clean_text(tamil_text) == tamil_text


def test_minimum_line_length_filtering() -> None:
    preprocessor = TextPreprocessor(make_config(min_line_length=3))

    assert preprocessor.clean_line("ab") is None
    assert preprocessor.clean_line("abc") == "abc"


def test_maximum_line_length_filtering() -> None:
    preprocessor = TextPreprocessor(make_config(max_line_length=3))

    assert preprocessor.clean_line("abcd") is None
    assert preprocessor.clean_line("abc") == "abc"


def test_process_file_successfully(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.txt"
    output_path = tmp_path / "processed" / "cleaned.txt"
    input_path.write_text("Hello     World!\n\nவணக்கம்!\n", encoding="utf-8")
    preprocessor = TextPreprocessor(make_config())

    stats = preprocessor.process_file(input_path, output_path)

    assert output_path.read_text(encoding="utf-8") == "Hello World!\nவணக்கம்!\n"
    assert stats.original_lines == 3
    assert stats.written_lines == 2
    assert stats.skipped_empty_lines == 1


def test_process_file_creates_missing_output_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.txt"
    output_path = tmp_path / "missing" / "nested" / "cleaned.txt"
    input_path.write_text("Hello     World!\n", encoding="utf-8")
    preprocessor = TextPreprocessor(make_config())

    preprocessor.process_file(input_path, output_path)

    assert output_path.exists()


def test_rejects_identical_input_and_output_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "same.txt"
    input_path.write_text("Hello\n", encoding="utf-8")
    preprocessor = TextPreprocessor(make_config())

    with pytest.raises(ValueError, match="different files"):
        preprocessor.process_file(input_path, input_path)


def test_missing_input_file(tmp_path: Path) -> None:
    preprocessor = TextPreprocessor(make_config())

    with pytest.raises(FileNotFoundError):
        preprocessor.process_file(tmp_path / "missing.txt", tmp_path / "out.txt")


def test_returns_accurate_processing_statistics(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.txt"
    output_path = tmp_path / "out.txt"
    input_path.write_text("a\nabcd\nabcde\n\nxy\n", encoding="utf-8")
    preprocessor = TextPreprocessor(make_config(min_line_length=2, max_line_length=4))

    stats = preprocessor.process_file(input_path, output_path)

    assert stats.original_lines == 5
    assert stats.written_lines == 2
    assert stats.skipped_short_lines == 1
    assert stats.skipped_long_lines == 1
    assert stats.skipped_empty_lines == 1
    assert stats.cleaned_characters == len("abcd\nxy\n")


def test_avoids_partial_output_when_processing_fails(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.txt"
    output_path = tmp_path / "out.txt"
    input_path.write_bytes(b"valid\n\xff")
    output_path.write_text("original output", encoding="utf-8")
    preprocessor = TextPreprocessor(make_config())

    with pytest.raises(UnicodeDecodeError):
        preprocessor.process_file(input_path, output_path, encoding="utf-8")

    assert output_path.read_text(encoding="utf-8") == "original output"
    assert not list(tmp_path.glob(".out.txt.*.tmp"))


def test_configuration_validation_rejects_invalid_preprocessing(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_config.yaml"
    config_data = _base_config()
    config_data["preprocessing"]["unicode_normalization"] = "BAD"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="unicode_normalization"):
        load_config(config_path)


def _base_config() -> dict[str, object]:
    return {
        "project": {
            "name": "GenPy LLM",
            "version": "0.1.0",
            "description": "Test config.",
        },
        "paths": {
            "data_dir": "data",
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "checkpoints_dir": "checkpoints",
            "logs_dir": "logs",
            "notebooks_dir": "notebooks",
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "tokenized_dir": "data/tokenized",
            "input_file": "data/raw/sample.txt",
            "output_file": "data/processed/cleaned_text.txt",
            "tokenized_file": "data/tokenized/tokens.jsonl",
            "encoding": "utf-8",
        },
        "preprocessing": {
            "unicode_normalization": "NFKC",
            "lowercase": False,
            "normalize_whitespace": True,
            "preserve_newlines": True,
            "remove_control_characters": True,
            "remove_empty_lines": True,
            "strip_lines": True,
            "min_line_length": 1,
            "max_line_length": None,
        },
        "tokenizer": {
            "type": "word",
            "vocab_size": 5000,
            "min_frequency": 1,
            "lowercase": True,
        },
        "model": {
            "context_length": 128,
            "vocab_size": 5000,
            "embedding_dim": 128,
            "num_heads": 4,
            "num_layers": 4,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.0003,
            "epochs": 10,
            "seed": 42,
            "device": "auto",
        },
        "generation": {
            "max_new_tokens": 100,
            "temperature": 1.0,
            "top_k": 50,
        },
        "logging": {
            "level": "INFO",
            "log_file": "genpy_llm.log",
        },
    }
