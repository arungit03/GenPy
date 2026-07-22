from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from genpy_llm.config import ConfigError, TokenizationConfig, load_config
from genpy_llm.preprocessing import TextPreprocessor
from genpy_llm.tokenization import TextTokenizer
from tests.test_preprocessing import _base_config
from tests.test_preprocessing import make_config as make_preprocessing_config


def make_config(**overrides: object) -> TokenizationConfig:
    values = {
        "method": "word",
        "preserve_case": True,
        "preserve_punctuation": True,
        "preserve_newlines": True,
        "split_contractions": False,
        "normalize_quotes": True,
        "normalize_dashes": True,
        "add_bos_token": False,
        "add_eos_token": False,
        "add_newline_token": True,
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "newline_token": "<NL>",
        "unknown_token": "<UNK>",
    }
    values.update(overrides)
    return TokenizationConfig(**values)


def test_basic_english_word_tokenization() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize("Hello, GenPy LLM!") == ["Hello", ",", "GenPy", "LLM", "!"]


def test_punctuation_tokenization() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize("Wait... yes!") == ["Wait", ".", ".", ".", "yes", "!"]


def test_tamil_word_preservation() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize("\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd!") == [
        "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd",
        "!",
    ]


def test_mixed_tamil_and_english_text() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize(
        "\u0b87\u0ba4\u0bc1 GenPy \u0ba4\u0bbf\u0b9f\u0bcd\u0b9f\u0bae\u0bcd."
    ) == [
        "\u0b87\u0ba4\u0bc1",
        "GenPy",
        "\u0ba4\u0bbf\u0b9f\u0bcd\u0b9f\u0bae\u0bcd",
        ".",
    ]


def test_character_tokenization() -> None:
    tokenizer = TextTokenizer(make_config(method="character"))

    assert tokenizer.tokenize("Hi!") == ["H", "i", "!"]


def test_emoji_preservation() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize("Hi 🙂") == ["Hi", "🙂"]


def test_unicode_symbol_preservation() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize("Cost €5") == ["Cost", "€", "5"]


def test_combining_mark_handling() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize("Cafe\u0301") == ["Cafe\u0301"]


def test_case_preservation_enabled() -> None:
    tokenizer = TextTokenizer(make_config(preserve_case=True))

    assert tokenizer.tokenize("GenPy") == ["GenPy"]


def test_case_preservation_disabled() -> None:
    tokenizer = TextTokenizer(make_config(preserve_case=False))

    assert tokenizer.tokenize("GenPy") == ["genpy"]


def test_punctuation_preservation_enabled() -> None:
    tokenizer = TextTokenizer(make_config(preserve_punctuation=True))

    assert tokenizer.tokenize("Hello!") == ["Hello", "!"]


def test_punctuation_preservation_disabled() -> None:
    tokenizer = TextTokenizer(make_config(preserve_punctuation=False))

    assert tokenizer.tokenize("Hello! €") == ["Hello", "€"]


def test_quote_normalization() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize("\u201cHello\u201d") == ['"', "Hello", '"']


def test_dash_normalization() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.tokenize("word\u2014word") == ["word", "-", "word"]


def test_contraction_splitting_enabled() -> None:
    tokenizer = TextTokenizer(make_config(split_contractions=True))

    assert tokenizer.tokenize("GenPy's tokenizer") == ["GenPy", "'", "s", "tokenizer"]


def test_contraction_splitting_disabled() -> None:
    tokenizer = TextTokenizer(make_config(split_contractions=False))

    assert tokenizer.tokenize("GenPy's tokenizer") == ["GenPy's", "tokenizer"]


def test_newline_token_enabled() -> None:
    tokenizer = TextTokenizer(make_config(preserve_newlines=True, add_newline_token=True))

    assert tokenizer.tokenize("Hello\nworld") == ["Hello", "<NL>", "world"]


def test_newline_token_disabled() -> None:
    tokenizer = TextTokenizer(make_config(preserve_newlines=False, add_newline_token=False))

    assert tokenizer.tokenize("Hello\nworld") == ["Hello", "world"]


def test_bos_token_enabled() -> None:
    tokenizer = TextTokenizer(make_config(add_bos_token=True))

    assert tokenizer.tokenize("Hello") == ["<BOS>", "Hello"]


def test_eos_token_enabled() -> None:
    tokenizer = TextTokenizer(make_config(add_eos_token=True))

    assert tokenizer.tokenize("Hello") == ["Hello", "<EOS>"]


def test_bos_and_eos_added_only_once() -> None:
    tokenizer = TextTokenizer(make_config(add_bos_token=True, add_eos_token=True))

    assert tokenizer.tokenize("Hello world") == ["<BOS>", "Hello", "world", "<EOS>"]


def test_empty_text_input() -> None:
    tokenizer = TextTokenizer(make_config(add_bos_token=True, add_eos_token=True))

    assert tokenizer.tokenize("") == []


def test_whitespace_only_input() -> None:
    tokenizer = TextTokenizer(make_config(add_bos_token=True, add_eos_token=True))

    assert tokenizer.tokenize("   \n\t") == []


def test_detokenization_of_simple_english_text() -> None:
    tokenizer = TextTokenizer(make_config())

    assert tokenizer.detokenize(["Hello", ",", "world", "!"]) == "Hello, world!"


def test_detokenization_with_tamil_text() -> None:
    tokenizer = TextTokenizer(make_config())

    assert (
        tokenizer.detokenize(["\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd", "!"])
        == "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd!"
    )


def test_successful_file_tokenization(tmp_path: Path) -> None:
    input_path = tmp_path / "cleaned.txt"
    output_path = tmp_path / "tokenized" / "tokens.jsonl"
    input_path.write_text("Hello, world!\n\n", encoding="utf-8")
    tokenizer = TextTokenizer(make_config(add_eos_token=True))

    stats = tokenizer.process_file(input_path, output_path)

    assert output_path.exists()
    assert stats.tokenized_sequences == 1


def test_correct_jsonl_output(tmp_path: Path) -> None:
    input_path = tmp_path / "cleaned.txt"
    output_path = tmp_path / "tokens.jsonl"
    input_path.write_text("Hello, world!\n", encoding="utf-8")
    tokenizer = TextTokenizer(make_config(add_eos_token=True))

    tokenizer.process_file(input_path, output_path)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert records == [
        {
            "sequence_id": 0,
            "tokens": ["Hello", ",", "world", "!", "<EOS>"],
            "token_count": 5,
        }
    ]


def test_unicode_readable_json_output(tmp_path: Path) -> None:
    input_path = tmp_path / "cleaned.txt"
    output_path = tmp_path / "tokens.jsonl"
    input_path.write_text("\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd!\n", encoding="utf-8")
    tokenizer = TextTokenizer(make_config(add_eos_token=True))

    tokenizer.process_file(input_path, output_path)
    output_text = output_path.read_text(encoding="utf-8")

    assert "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd" in output_text
    assert "\\u0bb5" not in output_text


def test_correct_sequence_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "cleaned.txt"
    output_path = tmp_path / "tokens.jsonl"
    input_path.write_text("one\ntwo\n", encoding="utf-8")
    tokenizer = TextTokenizer(make_config(add_eos_token=True))

    tokenizer.process_file(input_path, output_path)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert [record["sequence_id"] for record in records] == [0, 1]


def test_accurate_tokenization_statistics(tmp_path: Path) -> None:
    input_path = tmp_path / "cleaned.txt"
    output_path = tmp_path / "tokens.jsonl"
    input_path.write_text("Hello, world!\n\nAgain.\n", encoding="utf-8")
    tokenizer = TextTokenizer(make_config(add_eos_token=True))

    stats = tokenizer.process_file(input_path, output_path)

    assert stats.input_lines == 3
    assert stats.tokenized_sequences == 2
    assert stats.empty_lines_skipped == 1
    assert stats.total_tokens == 8
    assert stats.punctuation_tokens == 3
    assert stats.special_tokens == 2
    assert stats.unique_tokens == 7


def test_output_directory_creation(tmp_path: Path) -> None:
    input_path = tmp_path / "cleaned.txt"
    output_path = tmp_path / "missing" / "tokens.jsonl"
    input_path.write_text("Hello\n", encoding="utf-8")
    tokenizer = TextTokenizer(make_config())

    tokenizer.process_file(input_path, output_path)

    assert output_path.exists()


def test_missing_input_file_handling(tmp_path: Path) -> None:
    tokenizer = TextTokenizer(make_config())

    with pytest.raises(FileNotFoundError):
        tokenizer.process_file(tmp_path / "missing.txt", tmp_path / "tokens.jsonl")


def test_directory_used_as_input_handling(tmp_path: Path) -> None:
    tokenizer = TextTokenizer(make_config())

    with pytest.raises(IsADirectoryError):
        tokenizer.process_file(tmp_path, tmp_path / "tokens.jsonl")


def test_identical_input_and_output_rejection(tmp_path: Path) -> None:
    input_path = tmp_path / "same.txt"
    input_path.write_text("Hello\n", encoding="utf-8")
    tokenizer = TextTokenizer(make_config())

    with pytest.raises(ValueError, match="different files"):
        tokenizer.process_file(input_path, input_path)


def test_invalid_tokenization_method() -> None:
    tokenizer = TextTokenizer(make_config(method="bpe"))

    with pytest.raises(ValueError, match="Unsupported tokenization method"):
        tokenizer.tokenize("Hello")


def test_invalid_special_token_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_config.yaml"
    config_data = _config_with_tokenization()
    config_data["tokenization"]["eos_token"] = "<BOS>"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="unique"):
        load_config(config_path)


def test_temporary_output_cleanup_after_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.txt"
    output_path = tmp_path / "tokens.jsonl"
    input_path.write_bytes(b"valid\n\xff")
    output_path.write_text("original output", encoding="utf-8")
    tokenizer = TextTokenizer(make_config())

    with pytest.raises(UnicodeDecodeError):
        tokenizer.process_file(input_path, output_path, encoding="utf-8")

    assert output_path.read_text(encoding="utf-8") == "original output"
    assert not list(tmp_path.glob(".tokens.jsonl.*.tmp"))


def test_existing_step_2_behavior_remains_functional() -> None:
    preprocessor = TextPreprocessor(make_preprocessing_config())

    assert preprocessor.clean_text("Hello     World!") == "Hello World!"


def _config_with_tokenization() -> dict[str, object]:
    config_data = _base_config()
    config_data["paths"]["tokenized_data_dir"] = "data/tokenized"
    config_data["data"]["tokenized_dir"] = "data/tokenized"
    config_data["data"]["tokenized_file"] = "data/tokenized/tokens.jsonl"
    config_data["tokenization"] = {
        "method": "word",
        "preserve_case": True,
        "preserve_punctuation": True,
        "preserve_newlines": True,
        "split_contractions": False,
        "normalize_quotes": True,
        "normalize_dashes": True,
        "add_bos_token": False,
        "add_eos_token": True,
        "add_newline_token": True,
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "newline_token": "<NL>",
        "unknown_token": "<UNK>",
    }
    return config_data
