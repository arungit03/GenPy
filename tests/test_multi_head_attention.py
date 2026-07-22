from __future__ import annotations

import pytest
import torch

from genpy_llm.attention import (
    AttentionError,
    CausalSelfAttention,
    MultiHeadCausalSelfAttention,
)
from genpy_llm.embeddings import TokenEmbedding
from genpy_llm.positional_encoding import GPTInputEmbedding, PositionalEncoding


def test_correct_output_shape() -> None:
    attention = MultiHeadCausalSelfAttention(8, 2, 4, dropout=0.0)

    output = attention(torch.randn(2, 3, 8))

    assert output.shape == (2, 3, 8)


def test_correct_attention_weight_shape() -> None:
    attention = MultiHeadCausalSelfAttention(8, 2, 4, dropout=0.0)

    output, weights = attention(torch.randn(2, 3, 8), return_attention=True)

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 3)


def test_correct_head_dimension() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=12, num_heads=3, max_sequence_length=4)

    assert attention.embedding_dim == 12
    assert attention.num_heads == 3
    assert attention.head_dim == 4


@pytest.mark.parametrize("num_heads", [0, -1, True])
def test_invalid_number_of_heads(num_heads: int) -> None:
    with pytest.raises(AttentionError, match="num_heads"):
        MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=num_heads, max_sequence_length=4)


def test_embedding_dimension_not_divisible_by_heads() -> None:
    with pytest.raises(AttentionError, match="divisible"):
        MultiHeadCausalSelfAttention(embedding_dim=10, num_heads=3, max_sequence_length=4)


def test_one_attention_head() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=1, max_sequence_length=4)

    output, weights = attention(torch.randn(2, 3, 8), return_attention=True)

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 1, 3, 3)


def test_multiple_attention_heads() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=4, max_sequence_length=4)

    output, weights = attention(torch.randn(2, 3, 8), return_attention=True)

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 4, 3, 3)


def test_single_token_sequence() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    _output, weights = attention(torch.randn(2, 1, 8), return_attention=True)

    assert torch.allclose(weights, torch.ones_like(weights))


def test_future_positions_masked_for_every_head() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    _output, weights = attention(torch.randn(1, 4, 8), return_attention=True)
    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    assert torch.equal(weights[0, :, future_mask], torch.zeros_like(weights[0, :, future_mask]))


def test_attention_rows_sum_to_one() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    _output, weights = attention(torch.randn(2, 4, 8), return_attention=True)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 2, 4), atol=1e-6)


def test_padding_mask_behavior_across_heads() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    output, weights = attention(torch.randn(1, 4, 8), padding_mask=mask, return_attention=True)

    assert torch.equal(weights[:, :, :, 2:], torch.zeros_like(weights[:, :, :, 2:]))
    assert torch.equal(output[:, 2:, :], torch.zeros_like(output[:, 2:, :]))


def test_invalid_mask_shape() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    with pytest.raises(AttentionError, match="padding_mask shape"):
        attention(torch.randn(1, 3, 8), padding_mask=torch.ones(1, 2))


def test_invalid_mask_values() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    with pytest.raises(AttentionError, match="0/1"):
        attention(torch.randn(1, 3, 8), padding_mask=torch.tensor([[1, 2, 0]]))


def test_sequence_exceeding_maximum() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=3)

    with pytest.raises(AttentionError, match="exceeds"):
        attention(torch.randn(1, 4, 8))


def test_invalid_input_rank() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=3)

    with pytest.raises(AttentionError, match="three-dimensional"):
        attention(torch.randn(3, 8))


def test_invalid_input_dtype() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=3)

    with pytest.raises(AttentionError, match="floating-point"):
        attention(torch.ones(1, 3, 8, dtype=torch.long))


def test_incorrect_embedding_dimension() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=3)

    with pytest.raises(AttentionError, match="last dimension"):
        attention(torch.randn(1, 3, 7))


def test_dropout_disabled_determinism() -> None:
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
        dropout=0.0,
    )
    attention.train()
    hidden_states = torch.randn(1, 3, 8)

    assert torch.equal(attention(hidden_states), attention(hidden_states))


def test_training_versus_evaluation_dropout() -> None:
    torch.manual_seed(1)
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
        dropout=0.8,
    )
    hidden_states = torch.randn(2, 4, 8)
    attention.train()
    first = attention(hidden_states)
    second = attention(hidden_states)
    attention.eval()
    third = attention(hidden_states)
    fourth = attention(hidden_states)

    assert not torch.equal(first, second)
    assert torch.equal(third, fourth)


def test_gradients_reach_qkv_and_output_projections() -> None:
    torch.manual_seed(1)
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
        dropout=0.0,
    )
    hidden_states = torch.randn(2, 4, 8, requires_grad=True)

    loss = attention(hidden_states).pow(2).mean()
    loss.backward()

    assert attention.qkv_projection.weight.grad is not None
    assert attention.qkv_projection.weight.grad.abs().sum().item() > 0
    assert attention.output_projection.weight.grad is not None
    assert attention.output_projection.weight.grad.abs().sum().item() > 0


def test_causal_mask_buffer_is_non_trainable() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    assert "causal_mask" in dict(attention.named_buffers())
    assert "causal_mask" not in dict(attention.named_parameters())
    assert attention.causal_mask.requires_grad is False


def test_no_nan_or_infinite_outputs() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    output, weights = attention(torch.randn(2, 4, 8), return_attention=True)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()


def test_cpu_behavior() -> None:
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
    ).to("cpu")

    assert attention(torch.randn(1, 3, 8, device="cpu")).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
    ).to("cuda")
    hidden_states = torch.randn(1, 3, 8, device="cuda")

    assert attention(hidden_states).device.type == "cuda"


def test_return_attention_disabled() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    assert isinstance(attention(torch.randn(1, 3, 8), return_attention=False), torch.Tensor)


def test_return_attention_enabled() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)
    result = attention(torch.randn(1, 3, 8), return_attention=True)

    assert isinstance(result, tuple)
    assert result[0].shape == (1, 3, 8)
    assert result[1].shape == (1, 2, 3, 3)


def test_head_merging_preserves_expected_shape() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=12, num_heads=3, max_sequence_length=5)

    output = attention(torch.randn(2, 5, 12))

    assert output.shape == (2, 5, 12)


def test_existing_single_head_attention_remains_functional() -> None:
    attention = CausalSelfAttention(embedding_dim=8, max_sequence_length=4)

    assert attention(torch.randn(1, 3, 8)).shape == (1, 3, 8)


def test_steps_6_to_8_remain_functional() -> None:
    token_embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)
    positional = PositionalEncoding(embedding_dim=8, max_sequence_length=4)
    input_embedding = GPTInputEmbedding(token_embedding, positional)
    hidden_states = input_embedding(torch.tensor([[1, 2, 3]], dtype=torch.long))
    single_head = CausalSelfAttention(embedding_dim=8, max_sequence_length=4)
    multi_head = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    assert single_head(hidden_states).shape == (1, 3, 8)
    assert multi_head(hidden_states).shape == (1, 3, 8)


def test_parameter_counts_with_bias() -> None:
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2, max_sequence_length=4)

    assert attention.parameter_count == 288
    assert attention.trainable_parameter_count == 288


def test_parameter_counts_without_bias() -> None:
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=8,
        num_heads=2,
        max_sequence_length=4,
        use_bias=False,
    )

    assert attention.parameter_count == 256
    assert attention.trainable_parameter_count == 256
