from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from genpy_llm.attention import MultiHeadCausalSelfAttention
from genpy_llm.config import ConfigError, load_config
from genpy_llm.embeddings import TokenEmbedding
from genpy_llm.feed_forward import FeedForwardNetwork
from genpy_llm.normalization import GPTLayerNorm, NormalizationError
from genpy_llm.positional_encoding import GPTInputEmbedding, PositionalEncoding
from genpy_llm.residual import PreNormResidual, ResidualConnection, ResidualError


def test_layer_norm_preserves_shape_dtype_and_device() -> None:
    layer_norm = GPTLayerNorm(embedding_dim=8)
    hidden_states = torch.randn(2, 3, 8, dtype=torch.float64)

    output = layer_norm(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == torch.float64
    assert output.device == hidden_states.device


def test_layer_norm_mean_is_approximately_zero() -> None:
    hidden_states = torch.randn(2, 3, 8)
    output = GPTLayerNorm(embedding_dim=8)(hidden_states)

    assert torch.allclose(output.mean(dim=-1), torch.zeros(2, 3), atol=1e-6)


def test_layer_norm_variance_is_approximately_one() -> None:
    hidden_states = torch.randn(2, 3, 8)
    output = GPTLayerNorm(embedding_dim=8)(hidden_states)

    assert torch.allclose(output.var(dim=-1, unbiased=False), torch.ones(2, 3), atol=2e-4)


def test_layer_norm_supports_single_token_and_single_batch() -> None:
    layer_norm = GPTLayerNorm(embedding_dim=8)

    assert layer_norm(torch.randn(1, 1, 8)).shape == (1, 1, 8)


def test_layer_norm_affine_enabled_and_disabled() -> None:
    with_affine = GPTLayerNorm(embedding_dim=8, elementwise_affine=True)
    without_affine = GPTLayerNorm(embedding_dim=8, elementwise_affine=False)

    assert with_affine.layer_norm.weight is not None
    assert with_affine.parameter_count == 16
    assert without_affine.layer_norm.weight is None
    assert without_affine.parameter_count == 0


@pytest.mark.parametrize("embedding_dim", [0, -1, True])
def test_layer_norm_rejects_invalid_embedding_dimension(embedding_dim: int) -> None:
    with pytest.raises(NormalizationError, match="embedding_dim"):
        GPTLayerNorm(embedding_dim=embedding_dim)


@pytest.mark.parametrize("epsilon", [0.0, -1e-5, True])
def test_layer_norm_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(NormalizationError, match="epsilon"):
        GPTLayerNorm(embedding_dim=8, epsilon=epsilon)


def test_layer_norm_rejects_invalid_input_rank() -> None:
    layer_norm = GPTLayerNorm(embedding_dim=8)

    with pytest.raises(NormalizationError, match="three-dimensional"):
        layer_norm(torch.randn(2, 8))


def test_layer_norm_rejects_incorrect_final_dimension() -> None:
    layer_norm = GPTLayerNorm(embedding_dim=8)

    with pytest.raises(NormalizationError, match="last dimension"):
        layer_norm(torch.randn(2, 3, 7))


def test_layer_norm_rejects_non_floating_input() -> None:
    layer_norm = GPTLayerNorm(embedding_dim=8)

    with pytest.raises(NormalizationError, match="floating-point"):
        layer_norm(torch.ones(2, 3, 8, dtype=torch.long))


def test_layer_norm_is_differentiable_and_does_not_mutate_input() -> None:
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)
    original = hidden_states.detach().clone()
    layer_norm = GPTLayerNorm(embedding_dim=8)

    layer_norm(hidden_states).pow(2).sum().backward()

    assert hidden_states.grad is not None
    assert layer_norm.layer_norm.weight.grad is not None
    assert torch.equal(hidden_states.detach(), original)


def test_residual_addition_correctness_with_dropout_disabled() -> None:
    residual = torch.randn(2, 3, 8)
    sublayer_output = torch.randn(2, 3, 8)

    output = ResidualConnection(dropout=0.0)(residual, sublayer_output)

    assert torch.equal(output, residual + sublayer_output)


def test_residual_zero_sublayer_identity_behavior() -> None:
    hidden_states = torch.randn(2, 3, 8)

    output = ResidualConnection(dropout=0.0)(hidden_states, torch.zeros_like(hidden_states))

    assert torch.equal(output, hidden_states)


def test_residual_rejects_shape_mismatch() -> None:
    residual = ResidualConnection(dropout=0.0)

    with pytest.raises(ResidualError, match="shapes"):
        residual(torch.randn(2, 3, 8), torch.randn(2, 3, 7))


def test_residual_rejects_dtype_mismatch() -> None:
    residual = ResidualConnection(dropout=0.0)

    with pytest.raises(ResidualError, match="dtypes"):
        residual(torch.randn(2, 3, 8), torch.randn(2, 3, 8, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_residual_rejects_device_mismatch() -> None:
    residual = ResidualConnection(dropout=0.0)

    with pytest.raises(ResidualError, match="devices"):
        residual(torch.randn(2, 3, 8), torch.randn(2, 3, 8, device="cuda"))


def test_residual_rejects_non_floating_inputs() -> None:
    residual = ResidualConnection(dropout=0.0)

    with pytest.raises(ResidualError, match="floating-point"):
        residual(torch.ones(2, 3, 8, dtype=torch.long), torch.randn(2, 3, 8))


def test_residual_dropout_disabled_is_deterministic() -> None:
    connection = ResidualConnection(dropout=0.0)
    connection.train()
    residual = torch.randn(2, 3, 8)
    sublayer_output = torch.randn(2, 3, 8)

    assert torch.equal(connection(residual, sublayer_output), connection(residual, sublayer_output))


def test_residual_dropout_follows_train_and_eval_mode() -> None:
    torch.manual_seed(1)
    connection = ResidualConnection(dropout=0.8)
    residual = torch.zeros(4, 4, 8)
    sublayer_output = torch.ones(4, 4, 8)

    connection.train()
    first = connection(residual, sublayer_output)
    second = connection(residual, sublayer_output)
    connection.eval()
    third = connection(residual, sublayer_output)
    fourth = connection(residual, sublayer_output)

    assert not torch.equal(first, second)
    assert torch.equal(third, fourth)


def test_residual_gradients_flow_through_both_paths() -> None:
    residual = torch.randn(2, 3, 8, requires_grad=True)
    sublayer_output = torch.randn(2, 3, 8, requires_grad=True)

    ResidualConnection(dropout=0.0)(residual, sublayer_output).sum().backward()

    assert residual.grad is not None
    assert sublayer_output.grad is not None
    assert torch.equal(residual.grad, torch.ones_like(residual))
    assert torch.equal(sublayer_output.grad, torch.ones_like(sublayer_output))


def test_pre_norm_wrapper_with_ffn_preserves_shape() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, dropout=0.0)
    wrapper = PreNormResidual(embedding_dim=8, sublayer=ffn, residual_dropout=0.0)
    hidden_states = torch.randn(2, 3, 8)

    output = wrapper(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_pre_norm_wrapper_with_multi_head_attention_preserves_shape() -> None:
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
        dropout=0.0,
    )
    wrapper = PreNormResidual(embedding_dim=8, sublayer=attention, residual_dropout=0.0)
    hidden_states = torch.randn(2, 4, 8)
    padding_mask = torch.ones(2, 4, dtype=torch.long)

    output = wrapper(hidden_states, padding_mask=padding_mask)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_pre_norm_rejects_tuple_sublayer_output() -> None:
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
        dropout=0.0,
    )
    wrapper = PreNormResidual(embedding_dim=8, sublayer=attention, residual_dropout=0.0)

    with pytest.raises(ResidualError, match="return_attention=False"):
        wrapper(torch.randn(2, 4, 8), return_attention=True)


def test_pre_norm_rejects_non_tensor_sublayer_output() -> None:
    class BadSublayer(nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> str:
            return "not a tensor"

    wrapper = PreNormResidual(embedding_dim=8, sublayer=BadSublayer(), residual_dropout=0.0)

    with pytest.raises(ResidualError, match="torch.Tensor"):
        wrapper(torch.randn(2, 4, 8))


def test_pre_norm_rejects_sublayer_shape_mismatch() -> None:
    class ShapeChangingSublayer(nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states[..., :-1]

    wrapper = PreNormResidual(
        embedding_dim=8,
        sublayer=ShapeChangingSublayer(),
        residual_dropout=0.0,
    )

    with pytest.raises(ResidualError, match="shapes"):
        wrapper(torch.randn(2, 4, 8))


def test_pre_norm_gradients_reach_sublayer_and_input() -> None:
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32, dropout=0.0)
    wrapper = PreNormResidual(embedding_dim=8, sublayer=ffn, residual_dropout=0.0)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    wrapper(hidden_states).pow(2).mean().backward()

    assert hidden_states.grad is not None
    assert ffn.input_projection.weight.grad is not None
    assert wrapper.layer_norm.layer_norm.weight.grad is not None


def test_cpu_behavior() -> None:
    wrapper = PreNormResidual(
        embedding_dim=8,
        sublayer=FeedForwardNetwork(embedding_dim=8, hidden_dim=32),
    ).to("cpu")

    assert wrapper(torch.randn(1, 3, 8, device="cpu")).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    wrapper = PreNormResidual(
        embedding_dim=8,
        sublayer=FeedForwardNetwork(embedding_dim=8, hidden_dim=32).to("cuda"),
    ).to("cuda")

    assert wrapper(torch.randn(1, 3, 8, device="cuda")).device.type == "cuda"


def test_steps_1_to_10_remain_functional() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)
    positional = PositionalEncoding(embedding_dim=8, max_sequence_length=4)
    input_embedding = GPTInputEmbedding(token_embedding, positional)
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    attention_block = PreNormResidual(embedding_dim=8, sublayer=attention, residual_dropout=0.0)
    ffn_block = PreNormResidual(embedding_dim=8, sublayer=ffn, residual_dropout=0.0)

    hidden_states = input_embedding(torch.tensor([[1, 2, 3]], dtype=torch.long))
    output = ffn_block(attention_block(hidden_states))

    assert output.shape == (1, 3, 8)


def test_default_config_includes_normalization_and_residual() -> None:
    config = load_config()

    assert config.normalization.type == "layer_norm"
    assert config.normalization.epsilon == 1e-5
    assert config.normalization.elementwise_affine is True
    assert config.residual.dropout == 0.1


def test_config_rejects_invalid_normalization_type(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["normalization"]["type"] = "batch_norm"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="normalization.type"):
        load_config(path)


def test_config_rejects_invalid_normalization_epsilon(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["normalization"]["epsilon"] = 0.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="normalization.epsilon"):
        load_config(path)


def test_config_rejects_invalid_normalization_affine_flag(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["normalization"]["elementwise_affine"] = "yes"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="elementwise_affine"):
        load_config(path)


def test_config_rejects_invalid_residual_dropout(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["residual"]["dropout"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="residual.dropout"):
        load_config(path)


def _load_base_config_data() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
