from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.attention import AttentionError, CausalSelfAttention
from genpy_llm.config import ConfigError, load_config
from genpy_llm.embeddings import TokenEmbedding
from genpy_llm.positional_encoding import GPTInputEmbedding, PositionalEncoding


def test_correct_output_shape() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.0)
    hidden_states = torch.randn(2, 3, 8)

    output = attention(hidden_states)

    assert output.shape == (2, 3, 8)
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device


def test_correct_attention_weight_shape() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.0)
    hidden_states = torch.randn(2, 3, 8)

    output, weights = attention(hidden_states, return_attention=True)

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 3, 3)


def test_single_token_sequence() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.0)
    output, weights = attention(torch.randn(2, 1, 8), return_attention=True)

    assert output.shape == (2, 1, 8)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_multiple_token_sequence() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.0)
    output = attention(torch.randn(2, 4, 8))

    assert output.shape == (2, 4, 8)


def test_future_positions_are_masked() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.0)
    _output, weights = attention(torch.randn(1, 4, 8), return_attention=True)

    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.equal(weights[0][future_mask], torch.zeros_like(weights[0][future_mask]))


def test_attention_rows_sum_to_one() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.0)
    _output, weights = attention(torch.randn(2, 4, 8), return_attention=True)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_padding_mask_prevents_padded_key_attention() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.0)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    _output, weights = attention(torch.randn(1, 4, 8), padding_mask=mask, return_attention=True)

    assert torch.equal(weights[:, :, 2:], torch.zeros_like(weights[:, :, 2:]))
    assert torch.allclose(weights[0, 1, :2].sum(), torch.tensor(1.0))


def test_bool_padding_mask_is_supported() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)
    mask = torch.tensor([[True, False, True]])

    _output, weights = attention(torch.randn(1, 3, 8), padding_mask=mask, return_attention=True)

    assert torch.equal(weights[:, :, 1], torch.zeros_like(weights[:, :, 1]))


def test_float_padding_mask_is_supported() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    _output, weights = attention(torch.randn(1, 3, 8), padding_mask=mask, return_attention=True)

    assert torch.equal(weights[:, :, 1], torch.zeros_like(weights[:, :, 1]))


def test_fully_masked_padding_rows_do_not_create_nan() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)
    mask = torch.zeros((1, 3), dtype=torch.long)

    output, weights = attention(torch.randn(1, 3, 8), padding_mask=mask, return_attention=True)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert torch.equal(weights, torch.zeros_like(weights))


def test_invalid_padding_mask_shape() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)

    with pytest.raises(AttentionError, match="padding_mask shape"):
        attention(torch.randn(1, 3, 8), padding_mask=torch.ones(1, 2))


def test_invalid_padding_mask_values() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)

    with pytest.raises(AttentionError, match="0/1"):
        attention(torch.randn(1, 3, 8), padding_mask=torch.tensor([[1, 2, 0]]))


def test_sequence_exceeding_maximum() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)

    with pytest.raises(AttentionError, match="exceeds"):
        attention(torch.randn(1, 4, 8))


def test_invalid_input_rank() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)

    with pytest.raises(AttentionError, match="three-dimensional"):
        attention(torch.randn(3, 8))


def test_invalid_input_dtype() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)

    with pytest.raises(AttentionError, match="floating-point"):
        attention(torch.ones(1, 3, 8, dtype=torch.long))


def test_incorrect_embedding_dimension() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)

    with pytest.raises(AttentionError, match="last dimension"):
        attention(torch.randn(1, 3, 7))


def test_rejects_invalid_constructor_values() -> None:
    with pytest.raises(AttentionError, match="embedding_dim"):
        CausalSelfAttention(embedding_dim=0, max_sequence_length=3)
    with pytest.raises(AttentionError, match="max_sequence_length"):
        CausalSelfAttention(embedding_dim=8, max_sequence_length=0)
    with pytest.raises(AttentionError, match="dropout"):
        CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=1.0)
    with pytest.raises(AttentionError, match="use_bias"):
        CausalSelfAttention(embedding_dim=8, max_sequence_length=3, use_bias="yes")  # type: ignore[arg-type]


def test_dropout_disabled_is_deterministic_in_training() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=3, dropout=0.0)
    attention.train()
    hidden_states = torch.randn(1, 3, 8)

    first = attention(hidden_states)
    second = attention(hidden_states)

    assert torch.equal(first, second)


def test_dropout_training_changes_outputs() -> None:
    torch.manual_seed(1)
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.8)
    attention.train()
    hidden_states = torch.randn(2, 4, 8)

    first = attention(hidden_states)
    second = attention(hidden_states)

    assert not torch.equal(first, second)


def test_dropout_eval_is_deterministic() -> None:
    torch.manual_seed(1)
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.8)
    attention.eval()
    hidden_states = torch.randn(2, 4, 8)

    first = attention(hidden_states)
    second = attention(hidden_states)

    assert torch.equal(first, second)


def test_gradients_reach_all_projection_layers() -> None:
    torch.manual_seed(1)
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4, dropout=0.0)
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    output = attention(hidden_states)
    loss = output.pow(2).mean()
    loss.backward()

    for layer in [
        attention.query_projection,
        attention.key_projection,
        attention.value_projection,
        attention.output_projection,
    ]:
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum().item() > 0


def test_registered_causal_mask_is_non_trainable() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4)

    assert "causal_mask" in dict(attention.named_buffers())
    assert "causal_mask" not in dict(attention.named_parameters())
    assert attention.causal_mask.requires_grad is False


def test_cpu_behavior() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4).to("cpu")
    output = attention(torch.randn(1, 3, 8, device="cpu"))

    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4).to("cuda")
    hidden_states = torch.randn(1, 3, 8, device="cuda")

    assert attention(hidden_states).device.type == "cuda"


def test_return_attention_false_returns_tensor() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4)
    output = attention(torch.randn(1, 3, 8), return_attention=False)

    assert isinstance(output, torch.Tensor)


def test_return_attention_true_returns_tuple() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4)
    result = attention(torch.randn(1, 3, 8), return_attention=True)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_no_nan_or_infinite_values() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4)
    output, weights = attention(torch.randn(2, 4, 8), return_attention=True)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()


def test_preserves_float64_dtype() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4)
    hidden_states = torch.randn(1, 3, 8, dtype=torch.float64)

    output = attention(hidden_states)

    assert output.dtype == torch.float64


def test_existing_steps_6_and_7_remain_functional() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)
    positional = PositionalEncoding(embedding_dim=8, max_sequence_length=4)
    input_embedding = GPTInputEmbedding(token_embedding, positional)
    hidden_states = input_embedding(torch.tensor([[1, 2, 3]], dtype=torch.long))
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4)

    assert attention(hidden_states).shape == (1, 3, 8)


def test_load_default_config_includes_attention() -> None:
    config = load_config()

    assert config.attention.dropout == 0.1
    assert config.attention.use_bias is True
    assert config.attention.causal is True


def test_config_rejects_invalid_attention_dropout(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["attention"]["dropout"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="attention.dropout"):
        load_config(path)


@pytest.mark.parametrize("field", ["use_bias", "causal"])
def test_config_rejects_non_bool_attention_values(tmp_path: Path, field: str) -> None:
    config_data = _load_base_config_data()
    config_data["attention"][field] = "yes"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match=field):
        load_config(path)


def test_config_rejects_non_causal_attention(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["attention"]["causal"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="causal"):
        load_config(path)


def _load_base_config_data() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
