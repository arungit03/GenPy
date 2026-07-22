from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from genpy_llm.config import ConfigError, VocabularyConfig, load_config
from genpy_llm.preprocessing import TextPreprocessor
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.vocabulary import (
    VOCABULARY_FORMAT_VERSION,
    Vocabulary,
    VocabularyError,
    encode_jsonl_file,
)
from tests.test_preprocessing import _base_config
from tests.test_preprocessing import make_config as make_preprocessing_config
from tests.test_tokenization import make_config as make_tokenization_config


def make_config(**overrides: object) -> VocabularyConfig:
    values = {
        "min_frequency": 1,
        "max_size": 5000,
        "include_special_tokens": True,
        "save_frequencies": True,
        "strict_special_token_validation": True,
        "pad_token": "<PAD>",
        "unknown_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "newline_token": "<NL>",
        "special_token_order": ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"),
    }
    values.update(overrides)
    return VocabularyConfig(**values)


def write_jsonl(path: Path, sequences: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sequence_id, tokens in enumerate(sequences):
            file.write(
                json.dumps(
                    {
                        "sequence_id": sequence_id,
                        "tokens": tokens,
                        "token_count": len(tokens),
                    },
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def test_basic_vocabulary_construction() -> None:
    vocabulary = Vocabulary.build([["Hello", "<EOS>"], ["world", "<EOS>"]], make_config())

    assert len(vocabulary) == 7
    assert vocabulary.token_id("<PAD>") == 0


def test_token_frequency_counting() -> None:
    vocabulary = Vocabulary.build([["apple", "apple", "banana"]], make_config())

    assert vocabulary.frequencies["apple"] == 2
    assert vocabulary.frequencies["banana"] == 1


def test_deterministic_id_assignment() -> None:
    sequences = [["banana", "apple"], ["banana", "cat"]]

    first = Vocabulary.build(sequences, make_config())
    second = Vocabulary.build(sequences, make_config())

    assert first.token_to_id == second.token_to_id


def test_frequency_descending_ordering() -> None:
    vocabulary = Vocabulary.build([["cat", "dog", "dog"]], make_config())

    assert vocabulary.token_id("dog") < vocabulary.token_id("cat")


def test_unicode_lexical_tie_breaking() -> None:
    vocabulary = Vocabulary.build([["banana", "apple"]], make_config())

    assert vocabulary.token_id("apple") < vocabulary.token_id("banana")


def test_special_token_ordering() -> None:
    config = make_config(special_token_order=("<UNK>", "<PAD>", "<BOS>", "<EOS>", "<NL>"))
    vocabulary = Vocabulary.build([["Hello"]], config)

    assert vocabulary.unknown_id == 0
    assert vocabulary.pad_id == 1


def test_default_special_token_ids() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.pad_id == 0
    assert vocabulary.unknown_id == 1
    assert vocabulary.bos_id == 2
    assert vocabulary.eos_id == 3
    assert vocabulary.newline_id == 4


def test_tamil_token_preservation() -> None:
    tamil = "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd"
    vocabulary = Vocabulary.build([[tamil]], make_config())

    assert vocabulary.decode(vocabulary.encode([tamil])) == [tamil]


def test_emoji_token_preservation() -> None:
    vocabulary = Vocabulary.build([["🙂"]], make_config())

    assert vocabulary.decode(vocabulary.encode(["🙂"])) == ["🙂"]


def test_punctuation_token_inclusion() -> None:
    vocabulary = Vocabulary.build([["Hello", "!"]], make_config())

    assert "!" in vocabulary.token_to_id


def test_duplicate_token_handling() -> None:
    vocabulary = Vocabulary.build([["same", "same", "same"]], make_config())

    assert list(vocabulary.token_to_id).count("same") == 1


def test_special_token_appearing_in_input() -> None:
    vocabulary = Vocabulary.build([["<EOS>", "<EOS>", "Hello"]], make_config())

    assert vocabulary.eos_id == 3
    assert vocabulary.frequencies["<EOS>"] == 2


def test_minimum_frequency_filtering() -> None:
    vocabulary = Vocabulary.build([["keep", "keep", "drop"]], make_config(min_frequency=2))

    assert "keep" in vocabulary.token_to_id
    assert "drop" not in vocabulary.token_to_id


def test_maximum_size_filtering() -> None:
    vocabulary = Vocabulary.build([["a", "b", "c"]], make_config(max_size=6))

    assert len(vocabulary) == 6


def test_maximum_size_includes_special_tokens() -> None:
    vocabulary = Vocabulary.build([["a", "b", "c"]], make_config(max_size=5))

    assert len(vocabulary) == 5
    assert set(vocabulary.token_to_id) == {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"}


def test_unlimited_vocabulary_size() -> None:
    vocabulary = Vocabulary.build([["a", "b", "c"]], make_config(max_size=None))

    assert len(vocabulary) == 8


def test_invalid_minimum_frequency(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_data = config_with_vocabulary()
    config_data["vocabulary"]["min_frequency"] = 0
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="min_frequency"):
        load_config(config_path)


def test_invalid_maximum_size(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_data = config_with_vocabulary()
    config_data["vocabulary"]["max_size"] = 0
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="max_size"):
        load_config(config_path)


def test_maximum_size_smaller_than_special_token_count(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_data = config_with_vocabulary()
    config_data["vocabulary"]["max_size"] = 4
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="special tokens"):
        load_config(config_path)


def test_empty_special_token_string(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_data = config_with_vocabulary()
    config_data["vocabulary"]["pad_token"] = ""
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="pad_token"):
        load_config(config_path)


def test_duplicate_special_tokens(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_data = config_with_vocabulary()
    config_data["vocabulary"]["unknown_token"] = "<PAD>"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="unique"):
        load_config(config_path)


def test_invalid_special_token_order(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_data = config_with_vocabulary()
    config_data["vocabulary"]["special_token_order"] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<BAD>"]
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="missing|required|unknown"):
        load_config(config_path)


def test_encoding_known_tokens() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.encode(["Hello"]) == [vocabulary.token_id("Hello")]


def test_unknown_token_mapping() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.encode(["missing"]) == [vocabulary.unknown_id]


def test_optional_bos_insertion() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.encode(["Hello"], add_bos=True)[0] == vocabulary.bos_id


def test_optional_eos_insertion() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.encode(["Hello"], add_eos=True)[-1] == vocabulary.eos_id


def test_avoiding_duplicate_bos() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.encode(["<BOS>", "Hello"], add_bos=True).count(vocabulary.bos_id) == 1


def test_avoiding_duplicate_eos() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.encode(["Hello", "<EOS>"], add_eos=True).count(vocabulary.eos_id) == 1


def test_decoding_valid_token_ids() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.decode([vocabulary.token_id("Hello")]) == ["Hello"]


def test_decoding_with_special_token_skipping() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    assert vocabulary.decode([vocabulary.bos_id, vocabulary.token_id("Hello")], True) == ["Hello"]


def test_rejecting_negative_token_ids() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    with pytest.raises(VocabularyError, match="-1"):
        vocabulary.decode([-1])


def test_rejecting_out_of_range_token_ids() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    with pytest.raises(VocabularyError, match=str(len(vocabulary))):
        vocabulary.decode([len(vocabulary)])


def test_rejecting_non_integer_token_ids() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    with pytest.raises(VocabularyError, match="integer"):
        vocabulary.decode(["1"])  # type: ignore[list-item]


def test_saving_vocabulary_json(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    vocabulary.save(path)

    assert json.loads(path.read_text(encoding="utf-8"))["format_version"] == 1


def test_loading_vocabulary_json(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    vocabulary = Vocabulary.build([["Hello"]], make_config())
    vocabulary.save(path)

    loaded = Vocabulary.load(path, make_config())

    assert loaded.token_to_id == vocabulary.token_to_id


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    vocabulary = Vocabulary.build([["Hello", "world"]], make_config())
    vocabulary.save(path)

    loaded = Vocabulary.load(path, make_config())

    assert loaded.decode(vocabulary.encode(["Hello", "world"])) == ["Hello", "world"]


def test_tamil_readable_json_output(tmp_path: Path) -> None:
    tamil = "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd"
    path = tmp_path / "vocab.json"
    Vocabulary.build([[tamil]], make_config()).save(path)

    assert tamil in path.read_text(encoding="utf-8")


def test_ensure_ascii_false_behavior(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    Vocabulary.build([["\u0bb5"]], make_config()).save(path)

    assert "\\u0bb5" not in path.read_text(encoding="utf-8")


def test_vocabulary_mapping_integrity() -> None:
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    for token, token_id in vocabulary.token_to_id.items():
        assert vocabulary.id_to_token[token_id] == token


def test_detecting_inconsistent_token_to_id() -> None:
    with pytest.raises(VocabularyError, match="continuous"):
        Vocabulary(
            {"<PAD>": 0, "<UNK>": 2},
            None,
            make_config(strict_special_token_validation=False),
        )


def test_detecting_inconsistent_id_to_token(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    vocabulary = Vocabulary.build([["Hello"]], make_config())
    data = vocabulary.to_json_dict()
    data["id_to_token"][0] = "WRONG"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(VocabularyError, match="id_to_token"):
        Vocabulary.load(path, make_config(strict_special_token_validation=False))


def test_detecting_non_continuous_ids() -> None:
    with pytest.raises(VocabularyError, match="continuous"):
        Vocabulary(
            {"<PAD>": 0, "<UNK>": 1, "<BOS>": 3},
            None,
            make_config(strict_special_token_validation=False),
        )


def test_detecting_duplicate_ids() -> None:
    with pytest.raises(VocabularyError, match="unique"):
        Vocabulary(
            {"<PAD>": 0, "<UNK>": 0},
            None,
            make_config(strict_special_token_validation=False),
        )


def test_detecting_unsupported_format_version(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    vocabulary = Vocabulary.build([["Hello"]], make_config())
    data = vocabulary.to_json_dict()
    data["format_version"] = VOCABULARY_FORMAT_VERSION + 1
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(VocabularyError, match="Unsupported"):
        Vocabulary.load(path, make_config())


def test_successful_jsonl_vocabulary_build(tmp_path: Path) -> None:
    input_path = tmp_path / "tokens.jsonl"
    write_jsonl(input_path, [["Hello", "<EOS>"], ["Hello", "!"]])

    vocabulary, stats = Vocabulary.build_from_jsonl(input_path, make_config())

    assert "Hello" in vocabulary.token_to_id
    assert stats.processed_sequences == 2


def test_malformed_jsonl_handling(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(VocabularyError, match="line 1"):
        Vocabulary.build_from_jsonl(input_path, make_config())


def test_missing_tokens_field(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"sequence_id": 0}\n', encoding="utf-8")

    with pytest.raises(VocabularyError, match="line 1"):
        Vocabulary.build_from_jsonl(input_path, make_config())


def test_non_list_tokens_field(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"tokens": "Hello"}\n', encoding="utf-8")

    with pytest.raises(VocabularyError, match="line 1"):
        Vocabulary.build_from_jsonl(input_path, make_config())


def test_non_string_token_handling(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"tokens": ["Hello", 1]}\n', encoding="utf-8")

    with pytest.raises(VocabularyError, match="line 1"):
        Vocabulary.build_from_jsonl(input_path, make_config())


def test_incorrect_token_count(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"tokens": ["Hello"], "token_count": 2}\n', encoding="utf-8")

    with pytest.raises(VocabularyError, match="line 1"):
        Vocabulary.build_from_jsonl(input_path, make_config())


def test_error_messages_contain_jsonl_line_number(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"tokens": ["ok"]}\n{"tokens": ["", "bad"]}\n', encoding="utf-8")

    with pytest.raises(VocabularyError, match="line 2"):
        Vocabulary.build_from_jsonl(input_path, make_config())


def test_accurate_build_statistics(tmp_path: Path) -> None:
    input_path = tmp_path / "tokens.jsonl"
    write_jsonl(input_path, [["keep", "keep"], ["drop"], []])

    _vocabulary, stats = Vocabulary.build_from_jsonl(input_path, make_config(min_frequency=2))

    assert stats.processed_sequences == 2
    assert stats.empty_sequences == 1
    assert stats.total_tokens == 3
    assert stats.unique_tokens_observed == 2
    assert stats.excluded_below_min_frequency == 1


def test_atomic_vocabulary_saving(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "vocab.json"
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    vocabulary.save(path)

    assert path.exists()
    assert not list(path.parent.glob(".vocab.json.*.tmp"))


def test_temporary_file_cleanup_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "vocab.json"
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    def fail_dump(*_args: object, **_kwargs: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr("genpy_llm.vocabulary.json.dump", fail_dump)

    with pytest.raises(OSError):
        vocabulary.save(path)

    assert not list(tmp_path.glob(".vocab.json.*.tmp"))


def test_encoded_jsonl_generation(tmp_path: Path) -> None:
    input_path = tmp_path / "tokens.jsonl"
    output_path = tmp_path / "encoded.jsonl"
    write_jsonl(input_path, [["Hello", "<EOS>"]])
    vocabulary = Vocabulary.build([["Hello", "<EOS>"]], make_config())

    encode_jsonl_file(input_path, output_path, vocabulary)

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["token_ids"] == vocabulary.encode(["Hello", "<EOS>"])


def test_correct_encoded_sequence_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "tokens.jsonl"
    output_path = tmp_path / "encoded.jsonl"
    write_jsonl(input_path, [["one"], ["two"]])
    vocabulary = Vocabulary.build([["one"], ["two"]], make_config())

    encode_jsonl_file(input_path, output_path, vocabulary)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert [record["sequence_id"] for record in records] == [0, 1]


def test_correct_encoded_token_counts(tmp_path: Path) -> None:
    input_path = tmp_path / "tokens.jsonl"
    output_path = tmp_path / "encoded.jsonl"
    write_jsonl(input_path, [["one", "two"]])
    vocabulary = Vocabulary.build([["one", "two"]], make_config())

    encode_jsonl_file(input_path, output_path, vocabulary)
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert record["token_count"] == len(record["tokens"]) == len(record["token_ids"])


def test_unknown_tokens_in_later_encoding() -> None:
    vocabulary = Vocabulary.build([["known"]], make_config())

    assert vocabulary.encode(["later-unknown"]) == [vocabulary.unknown_id]


def test_missing_input_file_handling(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Vocabulary.build_from_jsonl(tmp_path / "missing.jsonl", make_config())


def test_input_path_being_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        Vocabulary.build_from_jsonl(tmp_path, make_config())


def test_output_directory_creation(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "vocab.json"
    Vocabulary.build([["Hello"]], make_config()).save(path)

    assert path.exists()


def test_input_and_output_path_conflict(tmp_path: Path) -> None:
    input_path = tmp_path / "tokens.jsonl"
    write_jsonl(input_path, [["Hello"]])
    vocabulary = Vocabulary.build([["Hello"]], make_config())

    with pytest.raises(ValueError, match="different files"):
        encode_jsonl_file(input_path, input_path, vocabulary)


def test_existing_steps_1_to_3_remain_functional() -> None:
    preprocessor = TextPreprocessor(make_preprocessing_config())
    tokenizer = TextTokenizer(make_tokenization_config(add_eos_token=True))

    cleaned = preprocessor.clean_text("Hello     World!")
    tokens = tokenizer.tokenize(cleaned)

    assert tokens == ["Hello", "World", "!", "<EOS>"]


def config_with_vocabulary() -> dict[str, object]:
    config_data = _base_config()
    config_data["paths"]["tokenized_data_dir"] = "data/tokenized"
    config_data["data"]["vocabulary_dir"] = "data/vocabulary"
    config_data["data"]["vocabulary_file"] = "data/vocabulary/vocab.json"
    config_data["data"]["vocabulary_metadata_file"] = "data/vocabulary/vocab_metadata.json"
    config_data["data"]["encoded_file"] = "data/vocabulary/encoded_tokens.jsonl"
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
    config_data["vocabulary"] = {
        "min_frequency": 1,
        "max_size": 5000,
        "include_special_tokens": True,
        "save_frequencies": True,
        "strict_special_token_validation": True,
        "pad_token": "<PAD>",
        "unknown_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "newline_token": "<NL>",
        "special_token_order": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"],
    }
    return config_data
