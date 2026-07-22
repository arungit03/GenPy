from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.config import ConfigError, load_config
from genpy_llm.embeddings import TokenEmbedding
from genpy_llm.positional_encoding import (
    GPTInputEmbedding,
    PositionalEncoding,
    PositionalEncodingError,
)


def test_learned_encoding_output_shape() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=16, encoding_type="learned")
    token_embeddings = torch.zeros((2, 4, 8), dtype=torch.float32)

    output = encoding(token_embeddings)

    assert output.shape == (2, 4, 8)
    assert output.dtype == token_embeddings.dtype
    assert output.device == token_embeddings.device


def test_sinusoidal_encoding_output_shape() -> None:
    encoding = PositionalEncoding(
        embedding_dim=8,
        max_sequence_length=16,
        encoding_type="sinusoidal",
    )
    token_embeddings = torch.zeros((2, 4, 8), dtype=torch.float32)

    output = encoding(token_embeddings)

    assert output.shape == (2, 4, 8)


def test_same_token_at_different_positions_produces_different_vectors() -> None:
    token_embedding = TokenEmbedding(vocab_size=5, embedding_dim=8)
    positional = PositionalEncoding(8, 16, encoding_type="sinusoidal")
    combined = GPTInputEmbedding(token_embedding, positional)

    output = combined(torch.tensor([[1, 1]], dtype=torch.long))

    assert not torch.equal(output[0, 0], output[0, 1])


def test_deterministic_sinusoidal_values() -> None:
    encoding = PositionalEncoding(
        embedding_dim=4,
        max_sequence_length=3,
        encoding_type="sinusoidal",
    )
    token_embeddings = torch.zeros((1, 2, 4), dtype=torch.float32)

    output = encoding(token_embeddings)

    assert output[0, 0, 0].item() == pytest.approx(0.0)
    assert output[0, 0, 1].item() == pytest.approx(1.0)
    assert output[0, 1, 0].item() == pytest.approx(torch.sin(torch.tensor(1.0)).item())
    assert output[0, 1, 1].item() == pytest.approx(torch.cos(torch.tensor(1.0)).item())
    assert output[0, 1, 2].item() == pytest.approx(torch.sin(torch.tensor(0.01)).item())
    assert output[0, 1, 3].item() == pytest.approx(torch.cos(torch.tensor(0.01)).item())


def test_sinusoidal_supports_odd_embedding_dimension() -> None:
    encoding = PositionalEncoding(
        embedding_dim=5,
        max_sequence_length=4,
        encoding_type="sinusoidal",
    )
    output = encoding(torch.zeros((1, 4, 5), dtype=torch.float32))

    assert output.shape == (1, 4, 5)


def test_learned_parameters_require_gradients() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=16, encoding_type="learned")

    assert encoding.position_embedding is not None
    assert encoding.position_embedding.weight.requires_grad is True
    assert encoding.trainable_parameter_count == 128


def test_sinusoidal_values_do_not_require_gradients() -> None:
    encoding = PositionalEncoding(
        embedding_dim=8,
        max_sequence_length=16,
        encoding_type="sinusoidal",
    )

    assert list(encoding.parameters()) == []
    assert encoding.sinusoidal_encoding.requires_grad is False
    assert encoding.trainable_parameter_count == 0


def test_position_offset_selects_later_positions() -> None:
    encoding = PositionalEncoding(
        embedding_dim=4,
        max_sequence_length=8,
        encoding_type="sinusoidal",
    )
    token_embeddings = torch.zeros((1, 2, 4), dtype=torch.float32)

    output = encoding(token_embeddings, position_offset=3)

    assert torch.allclose(output[0], encoding.sinusoidal_encoding[3:5])


def test_dropout_disabled_is_plain_addition() -> None:
    encoding = PositionalEncoding(
        embedding_dim=4,
        max_sequence_length=8,
        encoding_type="learned",
        dropout=0.0,
    )
    with torch.no_grad():
        encoding.position_embedding.weight.fill_(0.5)
    token_embeddings = torch.ones((1, 3, 4), dtype=torch.float32)
    original = token_embeddings.clone()

    output = encoding(token_embeddings)

    assert torch.equal(token_embeddings, original)
    assert torch.allclose(output, torch.full((1, 3, 4), 1.5))


def test_rejects_invalid_encoding_type() -> None:
    with pytest.raises(PositionalEncodingError, match="encoding_type"):
        PositionalEncoding(embedding_dim=8, max_sequence_length=16, encoding_type="bad")


@pytest.mark.parametrize(
    ("embedding_dim", "max_sequence_length"),
    [(0, 16), (8, 0), (True, 16), (8, False)],
)
def test_rejects_invalid_dimensions_and_limits(
    embedding_dim: int,
    max_sequence_length: int,
) -> None:
    with pytest.raises(PositionalEncodingError, match="greater than zero"):
        PositionalEncoding(embedding_dim, max_sequence_length)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, True])
def test_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(PositionalEncodingError, match="dropout"):
        PositionalEncoding(embedding_dim=8, max_sequence_length=16, dropout=dropout)


def test_rejects_invalid_initialization_std() -> None:
    with pytest.raises(PositionalEncodingError, match="initialization_std"):
        PositionalEncoding(embedding_dim=8, max_sequence_length=16, initialization_std=0)


def test_rejects_sequence_exceeding_maximum_length() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)

    with pytest.raises(PositionalEncodingError, match="exceeds"):
        encoding(torch.zeros((1, 5, 8), dtype=torch.float32))


def test_rejects_sequence_exceeding_maximum_with_offset() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)

    with pytest.raises(PositionalEncodingError, match="exceeds"):
        encoding(torch.zeros((1, 2, 8), dtype=torch.float32), position_offset=3)


def test_rejects_non_tensor_input() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)

    with pytest.raises(PositionalEncodingError, match="torch.Tensor"):
        encoding([1, 2, 3])  # type: ignore[arg-type]


def test_rejects_non_floating_input_dtype() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)

    with pytest.raises(PositionalEncodingError, match="floating-point"):
        encoding(torch.zeros((1, 2, 8), dtype=torch.long))


@pytest.mark.parametrize("shape", [(2, 8), (1, 2, 3, 8)])
def test_rejects_invalid_input_rank(shape: tuple[int, ...]) -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)

    with pytest.raises(PositionalEncodingError, match="three-dimensional"):
        encoding(torch.zeros(shape, dtype=torch.float32))


def test_rejects_wrong_embedding_dimension() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)

    with pytest.raises(PositionalEncodingError, match="last dimension"):
        encoding(torch.zeros((1, 2, 7), dtype=torch.float32))


@pytest.mark.parametrize("offset", [-1, True, 1.5])
def test_rejects_invalid_position_offset(offset: int) -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)

    with pytest.raises(PositionalEncodingError, match="position_offset"):
        encoding(torch.zeros((1, 2, 8), dtype=torch.float32), position_offset=offset)


def test_preserves_float64_dtype() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)
    token_embeddings = torch.zeros((1, 2, 8), dtype=torch.float64)

    assert encoding(token_embeddings).dtype == torch.float64


def test_empty_sequence_is_supported() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4)
    output = encoding(torch.zeros((2, 0, 8), dtype=torch.float32))

    assert output.shape == (2, 0, 8)


def test_cpu_behavior() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4).to("cpu")
    output = encoding(torch.zeros((1, 2, 8), dtype=torch.float32, device="cpu"))

    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    encoding = PositionalEncoding(embedding_dim=8, max_sequence_length=4).to("cuda")
    token_embeddings = torch.zeros((1, 2, 8), dtype=torch.float32, device="cuda")

    assert encoding(token_embeddings).device.type == "cuda"


def test_combined_gpt_input_embedding_output_shape() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)
    positional = PositionalEncoding(embedding_dim=8, max_sequence_length=16)
    combined = GPTInputEmbedding(token_embedding, positional)

    output = combined(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert output.shape == (1, 3, 8)


def test_combined_gpt_input_embedding_rejects_mismatched_dimensions() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)
    positional = PositionalEncoding(embedding_dim=4, max_sequence_length=16)

    with pytest.raises(PositionalEncodingError, match="dimensions"):
        GPTInputEmbedding(token_embedding, positional)


def test_combined_gpt_input_embedding_uses_position_offset() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=4)
    positional = PositionalEncoding(
        embedding_dim=4,
        max_sequence_length=16,
        encoding_type="sinusoidal",
    )
    combined = GPTInputEmbedding(token_embedding, positional)
    token_ids = torch.tensor([[1, 2]], dtype=torch.long)
    token_vectors = token_embedding(token_ids)

    output = combined(token_ids, position_offset=2)

    assert torch.allclose(output, token_vectors + positional.sinusoidal_encoding[2:4].unsqueeze(0))


def test_existing_step_6_token_embedding_still_works() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)

    assert token_embedding(torch.tensor([[1, 2]], dtype=torch.long)).shape == (1, 2, 8)


def test_load_default_config_includes_positional_encoding() -> None:
    config = load_config()

    assert config.positional_encoding.type == "learned"
    assert config.positional_encoding.max_sequence_length >= config.dataset.context_length


@pytest.mark.parametrize("encoding_type", ["bad", "", None])
def test_config_rejects_invalid_positional_encoding_type(
    tmp_path: Path,
    encoding_type: object,
) -> None:
    config_data = _load_base_config_data()
    config_data["positional_encoding"]["type"] = encoding_type
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="positional_encoding.type"):
        load_config(path)


def test_config_rejects_short_positional_max_length(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["positional_encoding"]["max_sequence_length"] = 64
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="max_sequence_length"):
        load_config(path)


def test_config_rejects_invalid_positional_dropout(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["positional_encoding"]["dropout"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="positional_encoding.dropout"):
        load_config(path)


def test_config_rejects_invalid_positional_initialization_std(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["positional_encoding"]["initialization_std"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="initialization_std"):
        load_config(path)


def _load_base_config_data() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
