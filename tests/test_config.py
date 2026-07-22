from __future__ import annotations

from pathlib import Path

import pytest

from genpy_llm.config import AppConfig, load_config


def test_load_valid_configuration() -> None:
    config = load_config()

    assert isinstance(config, AppConfig)
    assert config.project.name == "GenPy LLM"
    assert config.model.context_length == 128
    assert config.training.device == "auto"
    assert config.data.input_file.name == "sample.txt"
    assert config.data.tokenized_file.name == "tokens.jsonl"
    assert config.data.vocabulary_file.name == "vocab.json"
    assert config.preprocessing.unicode_normalization == "NFKC"
    assert config.tokenization.method == "word"
    assert config.vocabulary.special_token_order == ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>")
    assert config.dataset.context_length == 128
    assert config.dataset.split_unit == "sequence"
    assert config.embeddings.embedding_dim == 128
    assert config.embeddings.initialization == "normal"
    assert config.embeddings.zero_padding_embedding is True
    assert config.positional_encoding.type == "learned"
    assert config.positional_encoding.max_sequence_length == 128
    assert config.positional_encoding.dropout == 0.0
    assert config.attention.dropout == 0.1
    assert config.attention.use_bias is True
    assert config.attention.causal is True
    assert config.feed_forward.hidden_dim is None
    assert config.feed_forward.hidden_multiplier == 4
    assert config.feed_forward.activation == "gelu"
    assert config.feed_forward.dropout == 0.1
    assert config.web_interface.title == "GenPy LLM"
    assert config.web_interface.host == "127.0.0.1"
    assert config.web_interface.port == 7860
    assert config.web_interface.share is False
    assert config.web_interface.default_checkpoint.name == "genpy_best.pt"
    assert config.web_interface.max_prompt_characters == 2000
    assert config.paths.data_dir.is_absolute()


def test_missing_configuration_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(missing_path)
