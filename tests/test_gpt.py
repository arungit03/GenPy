from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from genpy_llm.attention import MultiHeadCausalSelfAttention
from genpy_llm.config import load_config
from genpy_llm.embeddings import TokenEmbedding
from genpy_llm.feed_forward import FeedForwardNetwork
from genpy_llm.gpt import GPTModel, GPTModelError, create_gpt_model
from genpy_llm.normalization import GPTLayerNorm
from genpy_llm.positional_encoding import GPTInputEmbedding, PositionalEncoding
from genpy_llm.residual import PreNormResidual
from genpy_llm.transformer_block import TransformerBlock


def _model(dropout: float = 0.0, tie_embeddings: bool = True) -> GPTModel:
    return GPTModel(
        vocab_size=11,
        embedding_dim=8,
        num_heads=2,
        num_layers=2,
        context_length=4,
        feed_forward_hidden_dim=32,
        padding_idx=0,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
    )


def test_correct_model_construction() -> None:
    model = _model()

    assert model.vocab_size == 11
    assert model.embedding_dim == 8
    assert model.num_heads == 2
    assert model.context_length == 4


def test_correct_number_of_transformer_blocks() -> None:
    model = _model()

    assert len(model.blocks) == 2
    assert model.blocks[0] is not model.blocks[1]


def test_input_and_logits_shapes() -> None:
    logits = _model()(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long))

    assert logits.shape == (2, 3, 11)


def test_single_token_input() -> None:
    assert _model()(torch.tensor([[1]], dtype=torch.long)).shape == (1, 1, 11)


def test_maximum_context_length_input() -> None:
    assert _model()(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)).shape == (1, 4, 11)


def test_sequence_exceeding_context_length() -> None:
    with pytest.raises(GPTModelError, match="context length"):
        _model()(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))


def test_invalid_token_dtype() -> None:
    with pytest.raises(GPTModelError, match="torch.long"):
        _model()(torch.ones(1, 3, dtype=torch.float32))


def test_invalid_rank() -> None:
    with pytest.raises(GPTModelError, match="two-dimensional"):
        _model()(torch.ones(1, 2, 3, dtype=torch.long))


def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(GPTModelError, match="sequence dimension"):
        _model()(torch.empty(1, 0, dtype=torch.long))


def test_negative_token_ids() -> None:
    with pytest.raises(GPTModelError, match="negative"):
        _model()(torch.tensor([[1, -1]], dtype=torch.long))


def test_out_of_range_token_ids() -> None:
    with pytest.raises(GPTModelError, match="below vocab_size"):
        _model()(torch.tensor([[1, 11]], dtype=torch.long))


def test_padding_mask_support() -> None:
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)
    padding_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)

    logits = _model()(input_ids, padding_mask=padding_mask)

    assert logits.shape == (2, 4, 11)
    assert torch.isfinite(logits).all()


def test_invalid_padding_mask() -> None:
    with pytest.raises(GPTModelError, match="padding_mask shape"):
        _model()(torch.tensor([[1, 2]], dtype=torch.long), padding_mask=torch.ones(1, 3))


def test_attention_map_count_equals_number_of_layers() -> None:
    logits, attention_maps = _model()(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        return_attention=True,
    )

    assert logits.shape == (1, 3, 11)
    assert len(attention_maps) == 2


def test_attention_map_shape() -> None:
    _logits, attention_maps = _model()(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        return_attention=True,
    )

    assert attention_maps[0].shape == (1, 2, 3, 3)


def test_causal_masking_remains_correct() -> None:
    _logits, attention_maps = _model()(
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        return_attention=True,
    )
    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.equal(
        attention_maps[0][..., future_mask],
        torch.zeros_like(attention_maps[0][..., future_mask]),
    )


def test_weight_tying_enabled() -> None:
    model = _model(tie_embeddings=True)

    assert model.embeddings_are_tied()
    assert model.lm_head.weight is model.token_embedding.weight


def test_weight_tying_disabled() -> None:
    model = _model(tie_embeddings=False)

    assert not model.embeddings_are_tied()
    assert model.lm_head.weight is not model.token_embedding.weight


def test_actual_vocabulary_size_is_used(tmp_path: Path) -> None:
    config = load_config()
    vocabulary_path = _write_vocabulary(tmp_path, ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>", "x"])

    model, metadata = create_gpt_model(vocabulary_path, config)

    assert model.vocab_size == 6
    assert metadata.vocab_size == 6


def test_padding_embedding_remains_zero() -> None:
    model = _model()

    assert torch.equal(
        model.token_embedding.weight[0],
        torch.zeros_like(model.token_embedding.weight[0]),
    )


def test_gradient_flow_reaches_embeddings() -> None:
    model = _model()

    model(torch.tensor([[1, 2, 3]], dtype=torch.long)).pow(2).mean().backward()

    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.grad.abs().sum().item() > 0


def test_gradient_flow_reaches_every_transformer_block() -> None:
    model = _model()

    model(torch.tensor([[1, 2, 3]], dtype=torch.long)).pow(2).mean().backward()

    for block in model.blocks:
        assert block.attention.qkv_projection.weight.grad is not None
        assert block.feed_forward.input_projection.weight.grad is not None


def test_gradient_flow_reaches_final_layer_norm() -> None:
    model = _model()

    model(torch.tensor([[1, 2, 3]], dtype=torch.long)).pow(2).mean().backward()

    assert model.final_norm.layer_norm.weight.grad is not None


def test_gradient_flow_reaches_lm_head() -> None:
    model = _model(tie_embeddings=False)

    model(torch.tensor([[1, 2, 3]], dtype=torch.long)).pow(2).mean().backward()

    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum().item() > 0


def test_output_contains_no_nan_or_infinite_values() -> None:
    assert torch.isfinite(_model()(torch.tensor([[1, 2, 3]], dtype=torch.long))).all()


def test_deterministic_output_in_eval_mode_with_fixed_seed() -> None:
    torch.manual_seed(123)
    first = _model(dropout=0.5)
    torch.manual_seed(123)
    second = _model(dropout=0.5)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    first.eval()
    second.eval()

    assert torch.allclose(first(input_ids), second(input_ids))


def test_dropout_behavior_in_train_and_eval_modes() -> None:
    torch.manual_seed(1)
    model = _model(dropout=0.8)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    model.train()
    first = model(input_ids)
    second = model(input_ids)
    model.eval()
    third = model(input_ids)
    fourth = model(input_ids)

    assert not torch.equal(first, second)
    assert torch.equal(third, fourth)


def test_cpu_behavior() -> None:
    model = _model().to("cpu")

    assert model(torch.tensor([[1, 2, 3]], dtype=torch.long, device="cpu")).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    model = _model().to("cuda")

    assert model(torch.tensor([[1, 2, 3]], dtype=torch.long, device="cuda")).device.type == "cuda"


def test_autocast_forward_keeps_residual_connections_dtype_safe() -> None:
    model = _model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(input_ids)

    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()


def test_parameter_count_correctness() -> None:
    tied = _model(tie_embeddings=True)
    untied = _model(tie_embeddings=False)

    assert tied.parameter_count == 1880
    assert tied.trainable_parameter_count == 1880
    assert tied.metadata().total_parameters == 1880
    assert untied.parameter_count == 1968


def test_invalid_construction_values() -> None:
    with pytest.raises(GPTModelError, match="vocab_size"):
        GPTModel(0, 8, 2, 2, 4, 32)
    with pytest.raises(GPTModelError, match="divisible"):
        GPTModel(11, 8, 3, 2, 4, 32)
    with pytest.raises(GPTModelError, match="dropout"):
        GPTModel(11, 8, 2, 2, 4, 32, dropout=1.0)


def test_steps_1_to_12_remain_functional() -> None:
    token_embedding = TokenEmbedding(vocab_size=11, embedding_dim=8)
    positional = PositionalEncoding(embedding_dim=8, max_sequence_length=4)
    input_embedding = GPTInputEmbedding(token_embedding, positional)
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)
    ffn = FeedForwardNetwork(embedding_dim=8, hidden_dim=32)
    pre_norm = PreNormResidual(embedding_dim=8, sublayer=ffn, residual_dropout=0.0)
    layer_norm = GPTLayerNorm(embedding_dim=8)
    block = TransformerBlock(8, 2, 4, 32)
    gpt = _model()

    hidden_states = input_embedding(torch.tensor([[1, 2, 3]], dtype=torch.long))
    transformed = block(pre_norm(layer_norm(attention(hidden_states))))
    logits = gpt(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert transformed.shape == (1, 3, 8)
    assert logits.shape == (1, 3, 11)


def _write_vocabulary(tmp_path: Path, tokens: list[str]) -> Path:
    path = tmp_path / "vocab.json"
    special_tokens = {
        "pad_token": "<PAD>",
        "unknown_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "newline_token": "<NL>",
    }
    payload = {
        "format_version": 1,
        "vocab_size": len(tokens),
        "special_tokens": special_tokens,
        "token_to_id": {token: index for index, token in enumerate(tokens)},
        "id_to_token": tokens,
        "frequencies": {token: 1 for token in tokens},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
