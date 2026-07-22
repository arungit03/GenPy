from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.config import ConfigError, EmbeddingConfig, VocabularyConfig, load_config
from genpy_llm.embeddings import (
    EMBEDDING_CHECKPOINT_FORMAT_VERSION,
    EmbeddingError,
    TokenEmbedding,
    build_embedding_metadata,
    calculate_embedding_statistics,
    cosine_similarity_between_tokens,
    create_token_embedding,
    inspect_token_embeddings,
    load_embedding_checkpoint,
    save_embedding_checkpoint,
)
from genpy_llm.vocabulary import Vocabulary, VocabularyError


def make_embedding_config(**overrides: object) -> EmbeddingConfig:
    values = {
        "embedding_dim": 8,
        "initialization": "normal",
        "initialization_std": 0.02,
        "scale_embeddings": False,
        "freeze_embeddings": False,
        "zero_padding_embedding": True,
    }
    values.update(overrides)
    return EmbeddingConfig(**values)


def make_vocabulary_config(**overrides: object) -> VocabularyConfig:
    values = {
        "min_frequency": 1,
        "max_size": 32,
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


def make_vocabulary() -> Vocabulary:
    return Vocabulary.build(
        [["GenPy", "LLM", "token", "embedding"], ["வணக்கம்", "GenPy"]],
        make_vocabulary_config(),
    )


def save_vocabulary(tmp_path: Path) -> tuple[Vocabulary, Path]:
    vocabulary = make_vocabulary()
    path = tmp_path / "vocab.json"
    vocabulary.save(path)
    return vocabulary, path


def test_constructs_embedding_with_expected_shape() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    assert embedding.weight.shape == (10, 8)
    assert embedding.num_embeddings == 10
    assert embedding.embedding_dim == 8


def test_forward_accepts_one_dimensional_ids() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    output = embedding(torch.tensor([1, 2, 3], dtype=torch.long))

    assert output.shape == (3, 8)
    assert output.is_floating_point()


def test_forward_accepts_two_dimensional_ids() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    output = embedding(torch.tensor([[1, 2], [3, 4]], dtype=torch.long))

    assert output.shape == (2, 2, 8)


def test_forward_preserves_input_device() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    token_ids = torch.tensor([1, 2], dtype=torch.long)

    assert embedding(token_ids).device == token_ids.device


def test_empty_one_dimensional_input_is_allowed() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    output = embedding(torch.empty((0,), dtype=torch.long))

    assert output.shape == (0, 8)


def test_empty_two_dimensional_input_is_allowed() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    output = embedding(torch.empty((0, 4), dtype=torch.long))

    assert output.shape == (0, 4, 8)


def test_forward_is_differentiable() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    output = embedding(torch.tensor([[1, 2]], dtype=torch.long))
    output.sum().backward()

    assert embedding.weight.grad is not None
    assert embedding.weight.grad[1].abs().sum().item() > 0


def test_rejects_non_tensor_input() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="torch.Tensor"):
        embedding([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32, torch.bool])
def test_rejects_non_long_token_ids(dtype: torch.dtype) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="torch.long"):
        embedding(torch.tensor([1, 2], dtype=dtype))


def test_rejects_three_dimensional_token_ids() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="one- or two-dimensional"):
        embedding(torch.zeros((1, 2, 3), dtype=torch.long))


def test_rejects_zero_dimensional_token_ids() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="one- or two-dimensional"):
        embedding(torch.tensor(1, dtype=torch.long))


def test_rejects_negative_token_id_with_position() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="negative ID -1 at position \\(0, 1\\)"):
        embedding(torch.tensor([[1, -1]], dtype=torch.long))


def test_rejects_out_of_range_token_id_with_position() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="out-of-range ID 10 at position 1"):
        embedding(torch.tensor([1, 10], dtype=torch.long))


@pytest.mark.parametrize(
    ("vocab_size", "embedding_dim"),
    [(0, 8), (10, 0), (-1, 8), (True, 8), (10, False)],
)
def test_rejects_invalid_constructor_dimensions(vocab_size: int, embedding_dim: int) -> None:
    with pytest.raises(EmbeddingError, match="greater than zero"):
        TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)


@pytest.mark.parametrize("padding_idx", [-1, 10, True])
def test_rejects_invalid_padding_idx(padding_idx: int) -> None:
    with pytest.raises(EmbeddingError, match="padding_idx"):
        TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=padding_idx)


def test_rejects_config_dimension_mismatch() -> None:
    with pytest.raises(EmbeddingError, match="dimension"):
        TokenEmbedding(
            vocab_size=10,
            embedding_dim=8,
            config=make_embedding_config(embedding_dim=4),
        )


def test_rejects_unknown_initialization() -> None:
    with pytest.raises(EmbeddingError, match="Unsupported"):
        TokenEmbedding(
            vocab_size=10,
            embedding_dim=8,
            config=make_embedding_config(initialization="bad"),
        )


def test_rejects_non_positive_initialization_std() -> None:
    with pytest.raises(EmbeddingError, match="initialization_std"):
        TokenEmbedding(
            vocab_size=10,
            embedding_dim=8,
            config=make_embedding_config(initialization_std=0.0),
        )


def test_normal_initialization_uses_configured_std() -> None:
    torch.manual_seed(1)
    embedding = TokenEmbedding(
        vocab_size=200,
        embedding_dim=8,
        config=make_embedding_config(initialization_std=0.05, zero_padding_embedding=False),
    )
    std = float(embedding.weight.detach().std(unbiased=False).item())

    assert 0.035 < std < 0.065


def test_uniform_initialization_range() -> None:
    torch.manual_seed(1)
    embedding = TokenEmbedding(
        vocab_size=50,
        embedding_dim=8,
        config=make_embedding_config(initialization="uniform", zero_padding_embedding=False),
    )
    bound = 1.0 / 8**0.5

    assert float(embedding.weight.max().item()) <= bound
    assert float(embedding.weight.min().item()) >= -bound


def test_xavier_uniform_initialization_range() -> None:
    torch.manual_seed(1)
    embedding = TokenEmbedding(
        vocab_size=50,
        embedding_dim=8,
        config=make_embedding_config(initialization="xavier_uniform", zero_padding_embedding=False),
    )
    bound = (6.0 / (50 + 8)) ** 0.5

    assert float(embedding.weight.max().item()) <= bound
    assert float(embedding.weight.min().item()) >= -bound


def test_padding_row_is_zeroed_by_default() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    assert torch.equal(embedding.weight[0], torch.zeros(8))


def test_padding_row_can_remain_random() -> None:
    torch.manual_seed(1)
    embedding = TokenEmbedding(
        vocab_size=10,
        embedding_dim=8,
        padding_idx=0,
        config=make_embedding_config(zero_padding_embedding=False),
    )

    assert embedding.weight[0].abs().sum().item() > 0


def test_padding_row_stays_zero_after_optimizer_step() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    before = embedding.weight[1].detach().clone()
    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)

    loss = embedding(torch.tensor([[0, 1]], dtype=torch.long)).sum()
    loss.backward()
    optimizer.step()

    assert torch.equal(embedding.weight[0].detach(), torch.zeros(8))
    assert not torch.equal(embedding.weight[1].detach(), before)


def test_frozen_embeddings_disable_gradients() -> None:
    embedding = TokenEmbedding(
        vocab_size=10,
        embedding_dim=8,
        config=make_embedding_config(freeze_embeddings=True),
    )

    assert embedding.weight.requires_grad is False
    assert build_embedding_metadata(embedding).trainable_parameter_count == 0


def test_unfrozen_embedding_has_trainable_parameters() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)

    assert embedding.weight.requires_grad is True
    assert build_embedding_metadata(embedding).trainable_parameter_count == 80


def test_scaling_disabled_returns_raw_lookup() -> None:
    embedding = TokenEmbedding(
        vocab_size=10,
        embedding_dim=8,
        config=make_embedding_config(scale_embeddings=False),
    )
    token_ids = torch.tensor([1, 2], dtype=torch.long)

    assert torch.equal(embedding(token_ids), embedding.embedding(token_ids))


def test_scaled_embeddings_multiply_by_sqrt_dimension() -> None:
    raw = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    scaled = TokenEmbedding(
        vocab_size=10,
        embedding_dim=8,
        padding_idx=0,
        config=make_embedding_config(scale_embeddings=True),
    )
    with torch.no_grad():
        scaled.weight.copy_(raw.weight)

    token_ids = torch.tensor([1, 2], dtype=torch.long)

    assert torch.allclose(scaled(token_ids), raw(token_ids) * 8**0.5)


def test_metadata_counts_parameters() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    metadata = build_embedding_metadata(embedding)

    assert metadata.vocab_size == 10
    assert metadata.parameter_count == 80
    assert metadata.trainable_parameter_count == 80
    assert "Vocabulary size: 10" in metadata.summary()


def test_metadata_reports_frozen_parameter_count() -> None:
    embedding = TokenEmbedding(
        vocab_size=10,
        embedding_dim=8,
        config=make_embedding_config(freeze_embeddings=True),
    )
    metadata = build_embedding_metadata(embedding)

    assert metadata.frozen is True
    assert metadata.trainable_parameter_count == 0


def test_statistics_report_expected_fields() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    stats = calculate_embedding_statistics(embedding)

    assert stats.zero_rows == 1
    assert stats.l2_norm > 0
    assert "Zero rows: 1" in stats.summary()


def test_statistics_count_multiple_zero_rows() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    with torch.no_grad():
        embedding.weight[3].zero_()

    assert calculate_embedding_statistics(embedding).zero_rows == 2


def test_factory_uses_actual_vocabulary_size(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _vocabulary, path = save_vocabulary(tmp_path)

    with caplog.at_level("WARNING", logger="genpy_llm"):
        embedding, metadata = create_token_embedding(
            path,
            make_embedding_config(),
            expected_vocab_size=5000,
        )

    assert embedding.num_embeddings == metadata.vocab_size
    assert "differs from actual vocabulary size" in caplog.text


def test_factory_uses_vocabulary_padding_id(tmp_path: Path) -> None:
    vocabulary, path = save_vocabulary(tmp_path)
    embedding, metadata = create_token_embedding(path, make_embedding_config())

    assert embedding.padding_idx == vocabulary.pad_id
    assert metadata.padding_idx == vocabulary.pad_id


def test_factory_rejects_missing_vocabulary(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        create_token_embedding(tmp_path / "missing.json", make_embedding_config())


def test_factory_rejects_directory_vocabulary(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        create_token_embedding(tmp_path, make_embedding_config())


def test_factory_rejects_vocabulary_missing_padding_token(tmp_path: Path) -> None:
    vocabulary = Vocabulary(
        token_to_id={"<UNK>": 0, "GenPy": 1},
        frequencies=None,
        config=make_vocabulary_config(
            include_special_tokens=False,
            strict_special_token_validation=False,
        ),
    )
    path = tmp_path / "vocab.json"
    vocabulary.save(path)

    with pytest.raises(VocabularyError, match="padding|PAD|special token"):
        create_token_embedding(path, make_embedding_config())


def test_inspect_known_token_returns_vector_preview() -> None:
    vocabulary = make_vocabulary()
    embedding = TokenEmbedding(vocab_size=len(vocabulary), embedding_dim=8, padding_idx=0)
    record = inspect_token_embeddings(embedding, vocabulary, ["GenPy"], max_dimensions=3)[0]

    assert record.requested_token == "GenPy"
    assert record.token == "GenPy"
    assert record.mapped_to_unknown is False
    assert len(record.vector) == 3


def test_inspect_unknown_token_marks_mapping() -> None:
    vocabulary = make_vocabulary()
    embedding = TokenEmbedding(vocab_size=len(vocabulary), embedding_dim=8, padding_idx=0)
    record = inspect_token_embeddings(embedding, vocabulary, ["missing"], max_dimensions=2)[0]

    assert record.requested_token == "missing"
    assert record.token == "<UNK>"
    assert record.token_id == vocabulary.unknown_id
    assert record.mapped_to_unknown is True


def test_inspect_preserves_tamil_token() -> None:
    vocabulary = make_vocabulary()
    embedding = TokenEmbedding(vocab_size=len(vocabulary), embedding_dim=8, padding_idx=0)
    record = inspect_token_embeddings(embedding, vocabulary, ["வணக்கம்"], max_dimensions=2)[0]

    assert record.token == "வணக்கம்"
    assert record.mapped_to_unknown is False


def test_inspect_rejects_empty_token() -> None:
    vocabulary = make_vocabulary()
    embedding = TokenEmbedding(vocab_size=len(vocabulary), embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="non-empty"):
        inspect_token_embeddings(embedding, vocabulary, [""])


def test_inspect_rejects_invalid_max_dimensions() -> None:
    vocabulary = make_vocabulary()
    embedding = TokenEmbedding(vocab_size=len(vocabulary), embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="max_dimensions"):
        inspect_token_embeddings(embedding, vocabulary, ["GenPy"], max_dimensions=0)


def test_cosine_similarity_identical_token_is_one() -> None:
    vocabulary = make_vocabulary()
    embedding = TokenEmbedding(vocab_size=len(vocabulary), embedding_dim=8, padding_idx=0)

    similarity = cosine_similarity_between_tokens(embedding, vocabulary, "GenPy", "GenPy")

    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_maps_unknown_tokens() -> None:
    vocabulary = make_vocabulary()
    embedding = TokenEmbedding(vocab_size=len(vocabulary), embedding_dim=8, padding_idx=0)

    similarity = cosine_similarity_between_tokens(embedding, vocabulary, "missing-a", "missing-b")

    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_rejects_zero_vector() -> None:
    vocabulary = make_vocabulary()
    embedding = TokenEmbedding(vocab_size=len(vocabulary), embedding_dim=8, padding_idx=0)

    with pytest.raises(EmbeddingError, match="zero vector"):
        cosine_similarity_between_tokens(embedding, vocabulary, "<PAD>", "GenPy")


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    path = tmp_path / "embedding.pt"
    save_embedding_checkpoint(embedding, path)

    loaded, metadata = load_embedding_checkpoint(path)

    assert metadata.vocab_size == 10
    assert torch.equal(loaded.weight, embedding.weight)


def test_checkpoint_preserves_scaled_config(tmp_path: Path) -> None:
    embedding = TokenEmbedding(
        vocab_size=10,
        embedding_dim=8,
        padding_idx=0,
        config=make_embedding_config(scale_embeddings=True),
    )
    path = tmp_path / "embedding.pt"
    save_embedding_checkpoint(embedding, path)

    loaded, _metadata = load_embedding_checkpoint(path)

    assert loaded.config.scale_embeddings is True


def test_checkpoint_preserves_frozen_flag(tmp_path: Path) -> None:
    embedding = TokenEmbedding(
        vocab_size=10,
        embedding_dim=8,
        padding_idx=0,
        config=make_embedding_config(freeze_embeddings=True),
    )
    path = tmp_path / "embedding.pt"
    save_embedding_checkpoint(embedding, path)

    loaded, metadata = load_embedding_checkpoint(path)

    assert metadata.frozen is True
    assert loaded.weight.requires_grad is False


def test_load_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_embedding_checkpoint(tmp_path / "missing.pt")


def test_load_rejects_checkpoint_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        load_embedding_checkpoint(tmp_path)


def test_load_rejects_non_dictionary_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(["bad"], path)

    with pytest.raises(EmbeddingError, match="dictionary"):
        load_embedding_checkpoint(path)


def test_load_rejects_unsupported_checkpoint_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "format_version": EMBEDDING_CHECKPOINT_FORMAT_VERSION + 1,
            "module_type": "TokenEmbedding",
        },
        path,
    )

    with pytest.raises(EmbeddingError, match="Unsupported"):
        load_embedding_checkpoint(path)


def test_load_rejects_wrong_module_type(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {"format_version": EMBEDDING_CHECKPOINT_FORMAT_VERSION, "module_type": "Other"},
        path,
    )

    with pytest.raises(EmbeddingError, match="module_type"):
        load_embedding_checkpoint(path)


def test_load_rejects_missing_state_dict(tmp_path: Path) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    path = tmp_path / "bad.pt"
    save_embedding_checkpoint(embedding, path)
    payload = torch.load(path, weights_only=True)
    payload.pop("state_dict")
    torch.save(payload, path)

    with pytest.raises(EmbeddingError, match="state_dict"):
        load_embedding_checkpoint(path)


def test_load_rejects_missing_weight(tmp_path: Path) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    path = tmp_path / "bad.pt"
    save_embedding_checkpoint(embedding, path)
    payload = torch.load(path, weights_only=True)
    payload["state_dict"] = {}
    torch.save(payload, path)

    with pytest.raises(EmbeddingError, match="embedding.weight"):
        load_embedding_checkpoint(path)


def test_load_rejects_wrong_weight_shape(tmp_path: Path) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    path = tmp_path / "bad.pt"
    save_embedding_checkpoint(embedding, path)
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["embedding.weight"] = torch.zeros((9, 8))
    torch.save(payload, path)

    with pytest.raises(EmbeddingError, match="shape"):
        load_embedding_checkpoint(path)


def test_load_rejects_invalid_metadata(tmp_path: Path) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    path = tmp_path / "bad.pt"
    save_embedding_checkpoint(embedding, path)
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["vocab_size"] = 0
    torch.save(payload, path)

    with pytest.raises(EmbeddingError, match="vocab_size"):
        load_embedding_checkpoint(path)


def test_save_checkpoint_rejects_directory(tmp_path: Path) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    with pytest.raises(IsADirectoryError):
        save_embedding_checkpoint(embedding, tmp_path)


def test_failed_checkpoint_save_cleans_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("genpy_llm.embeddings.torch.save", fail_save)

    with pytest.raises(RuntimeError, match="boom"):
        save_embedding_checkpoint(embedding, tmp_path / "embedding.pt")

    assert list(tmp_path.glob("*.tmp")) == []


def test_load_default_config_includes_embeddings() -> None:
    config = load_config()

    assert config.embeddings.embedding_dim == config.model.embedding_dim
    assert config.embeddings.initialization_std == 0.02


def test_config_allows_missing_embeddings_section(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data.pop("embeddings", None)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = load_config(path)

    assert config.embeddings.embedding_dim == config.model.embedding_dim


def test_config_rejects_embedding_dim_not_divisible_by_heads(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["model"]["embedding_dim"] = 130
    config_data["embeddings"]["embedding_dim"] = 130
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="embedding_dim=130 and num_heads=4"):
        load_config(path)


@pytest.mark.parametrize("initialization", ["bad", "", None])
def test_config_rejects_invalid_initialization(
    tmp_path: Path,
    initialization: object,
) -> None:
    config_data = _load_base_config_data()
    config_data["embeddings"]["initialization"] = initialization
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="embeddings.initialization"):
        load_config(path)


@pytest.mark.parametrize(
    "field",
    ["scale_embeddings", "freeze_embeddings", "zero_padding_embedding"],
)
def test_config_rejects_non_bool_embedding_flags(tmp_path: Path, field: str) -> None:
    config_data = _load_base_config_data()
    config_data["embeddings"][field] = "yes"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match=field):
        load_config(path)


def test_config_rejects_non_positive_embedding_std(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["embeddings"]["initialization_std"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="initialization_std"):
        load_config(path)


def test_checkpoint_can_load_to_cpu(tmp_path: Path) -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0)
    path = tmp_path / "embedding.pt"
    save_embedding_checkpoint(embedding, path)

    loaded, _metadata = load_embedding_checkpoint(path, map_location=torch.device("cpu"))

    assert loaded.weight.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_embedding_forward_on_cuda() -> None:
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, padding_idx=0).to("cuda")
    token_ids = torch.tensor([[1, 2]], dtype=torch.long, device="cuda")

    assert embedding(token_ids).device.type == "cuda"


def _load_base_config_data() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
