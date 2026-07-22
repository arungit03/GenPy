from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.config import ConfigError, load_config
from genpy_llm.embeddings import TokenEmbedding
from genpy_llm.feed_forward import FeedForwardNetwork
from genpy_llm.normalization import GPTLayerNorm
from genpy_llm.positional_encoding import GPTInputEmbedding, PositionalEncoding
from genpy_llm.residual import PreNormResidual
from genpy_llm.transformer_block import TransformerBlock, TransformerBlockError


def _block(dropout: float = 0.0) -> TransformerBlock:
    return TransformerBlock(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
        feed_forward_hidden_dim=32,
        attention_dropout=dropout,
        feed_forward_dropout=dropout,
        residual_dropout=dropout,
    )


def test_correct_construction() -> None:
    block = _block()

    assert block.embedding_dim == 8
    assert block.num_heads == 2
    assert block.head_dim == 4
    assert block.feed_forward_hidden_dim == 32


def test_correct_output_shape() -> None:
    output = _block()(torch.randn(2, 3, 8))

    assert output.shape == (2, 3, 8)


def test_attention_weight_return_shape() -> None:
    output, weights = _block()(torch.randn(2, 3, 8), return_attention=True)

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 3)


def test_single_token_sequence() -> None:
    assert _block()(torch.randn(2, 1, 8)).shape == (2, 1, 8)


def test_multiple_token_sequence() -> None:
    assert _block()(torch.randn(2, 4, 8)).shape == (2, 4, 8)


def test_padding_mask_support() -> None:
    hidden_states = torch.randn(2, 4, 8)
    padding_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)

    output = _block()(hidden_states, padding_mask=padding_mask)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_causal_masking() -> None:
    _output, weights = _block()(torch.randn(2, 4, 8), return_attention=True)
    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.equal(weights[..., future_mask], torch.zeros_like(weights[..., future_mask]))


def test_invalid_input_rank() -> None:
    with pytest.raises(TransformerBlockError, match="three-dimensional"):
        _block()(torch.randn(2, 8))


def test_invalid_input_dtype() -> None:
    with pytest.raises(TransformerBlockError, match="floating-point"):
        _block()(torch.ones(2, 3, 8, dtype=torch.long))


def test_incorrect_embedding_dimension() -> None:
    with pytest.raises(TransformerBlockError, match="last dimension"):
        _block()(torch.randn(2, 3, 7))


def test_sequence_exceeding_maximum() -> None:
    with pytest.raises(TransformerBlockError, match="maximum length"):
        _block()(torch.randn(2, 5, 8))


def test_invalid_padding_mask_shape() -> None:
    with pytest.raises(TransformerBlockError, match="padding_mask shape"):
        _block()(torch.randn(2, 4, 8), padding_mask=torch.ones(2, 3))


def test_invalid_padding_mask_values() -> None:
    with pytest.raises(TransformerBlockError, match="0/1"):
        _block()(torch.randn(2, 4, 8), padding_mask=torch.full((2, 4), 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_invalid_padding_mask_device() -> None:
    with pytest.raises(TransformerBlockError, match="same device"):
        _block()(torch.randn(2, 4, 8), padding_mask=torch.ones(2, 4, device="cuda"))


def test_dropout_train_eval_behavior() -> None:
    torch.manual_seed(1)
    block = _block(dropout=0.8)
    hidden_states = torch.randn(2, 4, 8)
    block.train()
    first = block(hidden_states)
    second = block(hidden_states)
    block.eval()
    third = block(hidden_states)
    fourth = block(hidden_states)

    assert not torch.equal(first, second)
    assert torch.equal(third, fourth)


def test_gradients_reach_attention_parameters() -> None:
    block = _block()
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    block(hidden_states).pow(2).mean().backward()

    assert block.attention.qkv_projection.weight.grad is not None
    assert block.attention.qkv_projection.weight.grad.abs().sum().item() > 0


def test_gradients_reach_ffn_parameters() -> None:
    block = _block()
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    block(hidden_states).pow(2).mean().backward()

    assert block.feed_forward.input_projection.weight.grad is not None
    assert block.feed_forward.input_projection.weight.grad.abs().sum().item() > 0


def test_gradients_reach_layer_norm_parameters() -> None:
    block = _block()
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    block(hidden_states).pow(2).mean().backward()

    assert block.attention_norm.layer_norm.weight.grad is not None
    assert block.feed_forward_norm.layer_norm.weight.grad is not None


def test_residual_connections_preserve_input_when_sublayers_are_zero() -> None:
    block = _block()
    for parameter in block.attention.parameters():
        parameter.data.zero_()
    for parameter in block.feed_forward.parameters():
        parameter.data.zero_()
    hidden_states = torch.randn(2, 4, 8)

    output = block(hidden_states)

    assert torch.allclose(output, hidden_states)


def test_input_tensor_is_not_mutated() -> None:
    block = _block()
    hidden_states = torch.randn(2, 4, 8)
    original = hidden_states.clone()

    _ = block(hidden_states)

    assert torch.equal(hidden_states, original)


def test_no_nan_or_infinite_outputs() -> None:
    assert torch.isfinite(_block()(torch.randn(2, 4, 8))).all()


def test_cpu_behavior() -> None:
    block = _block().to("cpu")

    assert block(torch.randn(1, 3, 8, device="cpu")).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    block = _block().to("cuda")

    assert block(torch.randn(1, 3, 8, device="cuda")).device.type == "cuda"


def test_parameter_count_correctness() -> None:
    block = _block()

    assert block.attention_parameter_count == 288
    assert block.feed_forward_parameter_count == 552
    assert block.layer_norm_parameter_count == 32
    assert block.parameter_count == 872
    assert block.trainable_parameter_count == 872


def test_preserves_float64_dtype() -> None:
    block = _block()
    hidden_states = torch.randn(2, 4, 8, dtype=torch.float64)

    assert block(hidden_states).dtype == torch.float64


def test_autocast_matches_residual_and_attention_dtypes() -> None:
    block = _block()
    hidden_states = torch.randn(2, 4, 8, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = block(hidden_states)

    assert output.dtype == torch.bfloat16


def test_invalid_construction_values() -> None:
    with pytest.raises(TransformerBlockError, match="embedding_dim"):
        TransformerBlock(0, 2, 4, 32)
    with pytest.raises(TransformerBlockError, match="divisible"):
        TransformerBlock(8, 3, 4, 32)
    with pytest.raises(TransformerBlockError, match="dropout"):
        TransformerBlock(8, 2, 4, 32, attention_dropout=1.0)


def test_steps_1_to_11_remain_functional() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)
    positional = PositionalEncoding(embedding_dim=8, max_sequence_length=4)
    input_embedding = GPTInputEmbedding(token_embedding, positional)
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    pre_norm = PreNormResidual(embedding_dim=8, sublayer=ffn, residual_dropout=0.0)
    layer_norm = GPTLayerNorm(embedding_dim=8)
    block = _block()

    hidden_states = input_embedding(torch.tensor([[1, 2, 3]], dtype=torch.long))
    output = block(pre_norm(layer_norm(hidden_states)))

    assert output.shape == (1, 3, 8)


def test_default_config_includes_transformer_block() -> None:
    config = load_config()

    assert config.transformer_block.attention_dropout == 0.1
    assert config.transformer_block.residual_dropout == 0.1
    assert config.transformer_block.feed_forward_dropout == 0.1


def test_config_rejects_invalid_transformer_block_dropout(tmp_path: Path) -> None:
    config_data = _load_base_config_data()
    config_data["transformer_block"]["residual_dropout"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="transformer_block.residual_dropout"):
        load_config(path)


def _load_base_config_data() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
