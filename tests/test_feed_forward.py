from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.attention import MultiHeadCausalSelfAttention
from genpy_llm.config import ConfigError, load_config
from genpy_llm.embeddings import TokenEmbedding
from genpy_llm.feed_forward import (
    FeedForwardError,
    FeedForwardNetwork,
    resolve_feed_forward_hidden_dim,
)
from genpy_llm.positional_encoding import GPTInputEmbedding, PositionalEncoding


def test_correct_construction() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    assert ffn.embedding_dim == 8
    assert ffn.hidden_dim == 32
    assert ffn.activation_name == "gelu"


def test_correct_input_and_output_shape() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    hidden_states = torch.randn(2, 3, 8)

    output = ffn(hidden_states)

    assert output.shape == (2, 3, 8)
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device


def test_hidden_dimension_calculation() -> None:
    assert resolve_feed_forward_hidden_dim(embedding_dim=8, hidden_multiplier=4) == 32


def test_explicit_hidden_dimension() -> None:
    assert (
        resolve_feed_forward_hidden_dim(embedding_dim=8, hidden_multiplier=4, hidden_dim=24) == 24
    )


def test_gelu_activation() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, activation="gelu")

    assert isinstance(ffn.activation, torch.nn.GELU)


def test_relu_activation() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, activation="relu")

    assert isinstance(ffn.activation, torch.nn.ReLU)


def test_silu_activation() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, activation="silu")

    assert isinstance(ffn.activation, torch.nn.SiLU)


def test_invalid_activation() -> None:
    with pytest.raises(FeedForwardError, match="activation"):
        FeedForwardNetwork(embedding_dim=8, hidden_dim=32, activation="bad")


@pytest.mark.parametrize("embedding_dim", [0, -1, True])
def test_invalid_embedding_dimension(embedding_dim: int) -> None:
    with pytest.raises(FeedForwardError, match="embedding_dim"):
        FeedForwardNetwork(embedding_dim=embedding_dim, hidden_dim=32)


@pytest.mark.parametrize("hidden_dim", [0, -1, True])
def test_invalid_hidden_dimension(hidden_dim: int) -> None:
    with pytest.raises(FeedForwardError, match="hidden_dim"):
        FeedForwardNetwork(embedding_dim=8, hidden_dim=hidden_dim)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, True])
def test_invalid_dropout(dropout: float) -> None:
    with pytest.raises(FeedForwardError, match="dropout"):
        FeedForwardNetwork(embedding_dim=8, hidden_dim=32, dropout=dropout)


def test_invalid_initialization_std() -> None:
    with pytest.raises(FeedForwardError, match="initialization_std"):
        FeedForwardNetwork(embedding_dim=8, hidden_dim=32, initialization_std=0)


def test_invalid_input_rank() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    with pytest.raises(FeedForwardError, match="three-dimensional"):
        ffn(torch.randn(2, 8))


def test_invalid_input_dtype() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    with pytest.raises(FeedForwardError, match="floating-point"):
        ffn(torch.ones(2, 3, 8, dtype=torch.long))


def test_incorrect_final_dimension() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    with pytest.raises(FeedForwardError, match="last dimension"):
        ffn(torch.randn(2, 3, 7))


def test_single_token_sequence() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    assert ffn(torch.randn(2, 1, 8)).shape == (2, 1, 8)


def test_multiple_token_sequence() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    assert ffn(torch.randn(2, 5, 8)).shape == (2, 5, 8)


def test_batch_size_of_one() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    assert ffn(torch.randn(1, 5, 8)).shape == (1, 5, 8)


def test_zero_sequence_length_is_supported() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    assert ffn(torch.randn(2, 0, 8)).shape == (2, 0, 8)


def test_rejects_zero_batch_size() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    with pytest.raises(FeedForwardError, match="batch"):
        ffn(torch.randn(0, 2, 8))


def test_gradients_reach_input_projection() -> None:
    torch.manual_seed(1)
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, dropout=0.0)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    ffn(hidden_states).pow(2).mean().backward()

    assert ffn.input_projection.weight.grad is not None
    assert ffn.input_projection.weight.grad.abs().sum().item() > 0


def test_gradients_reach_output_projection() -> None:
    torch.manual_seed(1)
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, dropout=0.0)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    ffn(hidden_states).pow(2).mean().backward()

    assert ffn.output_projection.weight.grad is not None
    assert ffn.output_projection.weight.grad.abs().sum().item() > 0


def test_dropout_disabled_determinism() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, dropout=0.0)
    ffn.train()
    hidden_states = torch.randn(2, 3, 8)

    assert torch.equal(ffn(hidden_states), ffn(hidden_states))


def test_dropout_train_eval_behavior() -> None:
    torch.manual_seed(1)
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, dropout=0.8)
    hidden_states = torch.randn(2, 3, 8)
    ffn.train()
    first = ffn(hidden_states)
    second = ffn(hidden_states)
    ffn.eval()
    third = ffn(hidden_states)
    fourth = ffn(hidden_states)

    assert not torch.equal(first, second)
    assert torch.equal(third, fourth)


def test_position_wise_independence() -> None:
    torch.manual_seed(1)
    ffn = FeedForwardNetwork(embedding_dim=4, hidden_dim=16, dropout=0.0)
    ffn.eval()
    original = torch.randn(1, 3, 4)
    changed = original.clone()
    changed[:, 2, :] += 100.0

    original_output = ffn(original)
    changed_output = ffn(changed)

    assert torch.allclose(original_output[:, :2, :], changed_output[:, :2, :])
    assert not torch.allclose(original_output[:, 2, :], changed_output[:, 2, :])


def test_bias_enabled_and_disabled() -> None:
    with_bias = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, use_bias=True)
    without_bias = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, use_bias=False)

    assert with_bias.input_projection.bias is not None
    assert with_bias.output_projection.bias is not None
    assert without_bias.input_projection.bias is None
    assert without_bias.output_projection.bias is None


def test_parameter_count_correctness() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, use_bias=True)

    assert ffn.parameter_count == 552
    assert ffn.trainable_parameter_count == 552
    assert ffn.metadata().parameter_count == 552


def test_parameter_count_without_bias() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, use_bias=False)

    assert ffn.parameter_count == 512


def test_metadata_summary() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, activation="relu")

    summary = ffn.metadata().summary()

    assert "Hidden dimension: 32" in summary
    assert "Activation: relu" in summary


def test_cpu_behavior() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32).to("cpu")

    assert ffn(torch.randn(1, 3, 8, device="cpu")).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32).to("cuda")

    assert ffn(torch.randn(1, 3, 8, device="cuda")).device.type == "cuda"


def test_preserves_float64_dtype() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    hidden_states = torch.randn(1, 3, 8, dtype=torch.float64)

    assert ffn(hidden_states).dtype == torch.float64


def test_no_nan_or_infinite_outputs() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)

    assert torch.isfinite(ffn(torch.randn(2, 3, 8))).all()


def test_weight_initialization_and_zero_bias() -> None:
    torch.manual_seed(1)
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=128, initialization_std=0.03)

    assert 0.02 < float(ffn.input_projection.weight.std(unbiased=False).item()) < 0.04
    assert torch.equal(ffn.input_projection.bias, torch.zeros_like(ffn.input_projection.bias))
    assert torch.equal(ffn.output_projection.bias, torch.zeros_like(ffn.output_projection.bias))


def test_steps_1_to_9_remain_functional() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)
    positional = PositionalEncoding(embedding_dim=8, max_sequence_length=4)
    input_embedding = GPTInputEmbedding(token_embedding, positional)
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    hidden_states = input_embedding(torch.tensor([[1, 2, 3]], dtype=torch.long))
    attended = attention(hidden_states)

    assert ffn(attended).shape == (1, 3, 8)


def test_default_config_includes_feed_forward() -> None:
    config = load_config()

    assert config.feed_forward.hidden_dim is None
    assert config.feed_forward.hidden_multiplier == 4
    assert config.feed_forward.activation == "gelu"


def test_config_rejects_invalid_hidden_dim(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["feed_forward"]["hidden_dim"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="feed_forward.hidden_dim"):
        load_config(path)


def test_config_rejects_invalid_hidden_multiplier(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["feed_forward"]["hidden_multiplier"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="hidden_multiplier"):
        load_config(path)


def test_config_rejects_invalid_activation(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["feed_forward"]["activation"] = "bad"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="feed_forward.activation"):
        load_config(path)


def test_config_rejects_invalid_dropout(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["feed_forward"]["dropout"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="feed_forward.dropout"):
        load_config(path)


def test_config_rejects_invalid_use_bias(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["feed_forward"]["use_bias"] = "yes"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="feed_forward.use_bias"):
        load_config(path)


def test_config_rejects_invalid_initialization_std(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["feed_forward"]["initialization_std"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="initialization_std"):
        load_config(path)


def _load_base_config_data() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
