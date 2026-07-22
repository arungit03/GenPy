"""Pre-normalization GPT transformer block for GenPy LLM."""

from __future__ import annotations

import torch
from torch import nn

from genpy_llm.attention import AttentionError, MultiHeadCausalSelfAttention
from genpy_llm.feed_forward import FeedForwardError, FeedForwardNetwork
from genpy_llm.normalization import GPTLayerNorm, NormalizationError
from genpy_llm.residual import (
    ResidualConnection,
    ResidualError,
    cast_residual_stream_for_autocast,
)


class TransformerBlockError(ValueError):
    """Raised when transformer block configuration or input is invalid."""


class TransformerBlock(nn.Module):
    """A single GPT-style pre-norm transformer block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_sequence_length: int,
        feed_forward_hidden_dim: int,
        attention_dropout: float = 0.1,
        feed_forward_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        normalization_epsilon: float = 1e-5,
        activation: str = "gelu",
        use_bias: bool = True,
        rotary_embeddings: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("num_heads", num_heads)
        _validate_positive_int("max_sequence_length", max_sequence_length)
        _validate_positive_int("feed_forward_hidden_dim", feed_forward_hidden_dim)
        if embedding_dim % num_heads != 0:
            raise TransformerBlockError("embedding_dim must be divisible by num_heads.")
        if not isinstance(use_bias, bool):
            raise TransformerBlockError("use_bias must be true or false.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.max_sequence_length = max_sequence_length
        self.feed_forward_hidden_dim = feed_forward_hidden_dim

        try:
            self.attention_norm = GPTLayerNorm(
                embedding_dim=embedding_dim,
                epsilon=normalization_epsilon,
            )
            self.attention = MultiHeadCausalSelfAttention(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                max_sequence_length=max_sequence_length,
                dropout=attention_dropout,
                use_bias=use_bias,
                rotary_embeddings=rotary_embeddings,
            )
            self.attention_residual = ResidualConnection(dropout=residual_dropout)
            self.feed_forward_norm = GPTLayerNorm(
                embedding_dim=embedding_dim,
                epsilon=normalization_epsilon,
            )
            self.feed_forward = FeedForwardNetwork(
                embedding_dim=embedding_dim,
                hidden_dim=feed_forward_hidden_dim,
                activation=activation,
                dropout=feed_forward_dropout,
                use_bias=use_bias,
            )
            self.feed_forward_residual = ResidualConnection(dropout=residual_dropout)
        except (AttentionError, FeedForwardError, NormalizationError, ResidualError) as exc:
            raise TransformerBlockError(str(exc)) from exc

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run a single pre-norm attention and FFN block."""

        _validate_hidden_states(hidden_states, self.embedding_dim, self.max_sequence_length)
        _validate_padding_mask(padding_mask, hidden_states)
        hidden_states = cast_residual_stream_for_autocast(hidden_states)

        normalized_attention_input = self.attention_norm(hidden_states)
        attention_result = self.attention(
            normalized_attention_input,
            padding_mask=padding_mask,
            return_attention=return_attention,
        )
        attention_weights = None
        if return_attention:
            attention_output, attention_weights = attention_result
        else:
            attention_output = attention_result

        hidden_states = self.attention_residual(hidden_states, attention_output)
        normalized_ffn_input = self.feed_forward_norm(hidden_states)
        feed_forward_output = self.feed_forward(normalized_ffn_input)
        hidden_states = self.feed_forward_residual(hidden_states, feed_forward_output)

        if return_attention:
            return hidden_states, attention_weights
        return hidden_states

    @property
    def parameter_count(self) -> int:
        """Return total trainable and frozen parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Return trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def attention_parameter_count(self) -> int:
        """Return parameters owned by the attention module."""

        return self.attention.parameter_count

    @property
    def feed_forward_parameter_count(self) -> int:
        """Return parameters owned by the feed-forward network."""

        return self.feed_forward.parameter_count

    @property
    def layer_norm_parameter_count(self) -> int:
        """Return parameters owned by both layer normalization modules."""

        return self.attention_norm.parameter_count + self.feed_forward_norm.parameter_count


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TransformerBlockError(f"{name} must be an integer greater than zero.")


def _validate_hidden_states(
    hidden_states: torch.Tensor,
    embedding_dim: int,
    max_sequence_length: int,
) -> None:
    if not isinstance(hidden_states, torch.Tensor):
        raise TransformerBlockError("hidden_states must be a torch.Tensor.")
    if hidden_states.ndim != 3:
        raise TransformerBlockError("hidden_states must be a three-dimensional tensor.")
    if not hidden_states.is_floating_point():
        raise TransformerBlockError("hidden_states must use a floating-point dtype.")
    batch_size, sequence_length, actual_embedding_dim = hidden_states.shape
    if batch_size <= 0:
        raise TransformerBlockError("hidden_states batch dimension must be greater than zero.")
    if sequence_length > max_sequence_length:
        raise TransformerBlockError(
            "Sequence length exceeds transformer block maximum length. "
            f"Received {sequence_length} with max_sequence_length {max_sequence_length}."
        )
    if actual_embedding_dim != embedding_dim:
        raise TransformerBlockError(
            "hidden_states last dimension must match embedding_dim. "
            f"Received {actual_embedding_dim} and expected {embedding_dim}."
        )


def _validate_padding_mask(
    padding_mask: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> None:
    if padding_mask is None:
        return
    if not isinstance(padding_mask, torch.Tensor):
        raise TransformerBlockError("padding_mask must be a torch.Tensor.")
    if padding_mask.device != hidden_states.device:
        raise TransformerBlockError("padding_mask must be on the same device as hidden_states.")
    if tuple(padding_mask.shape) != tuple(hidden_states.shape[:2]):
        raise TransformerBlockError(
            "padding_mask shape must match hidden_states batch and sequence dimensions."
        )
    if padding_mask.dtype == torch.bool or padding_mask.numel() == 0:
        return
    if not padding_mask.dtype.is_floating_point and padding_mask.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TransformerBlockError("padding_mask must be bool, integer, or floating zero/one.")
    if not _should_validate_tensor_values(padding_mask):
        return
    is_zero_or_one = (padding_mask == 0) | (padding_mask == 1)
    if not bool(is_zero_or_one.all().item()):
        raise TransformerBlockError("padding_mask values must be 0/1 or boolean.")


def _should_validate_tensor_values(tensor: torch.Tensor) -> bool:
    return tensor.device.type != "mps"


__all__ = ["TransformerBlock", "TransformerBlockError"]
