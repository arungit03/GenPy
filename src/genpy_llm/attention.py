"""Causal self-attention layers for GenPy LLM."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class AttentionError(ValueError):
    """Raised when attention configuration or input is invalid."""


class CausalSelfAttention(nn.Module):
    """Single-head scaled dot-product causal self-attention."""

    def __init__(
        self,
        embedding_dim: int,
        max_sequence_length: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("max_sequence_length", max_sequence_length)
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0.0 <= dropout < 1.0
        ):
            raise AttentionError("dropout must be at least 0.0 and less than 1.0.")
        if not isinstance(use_bias, bool):
            raise AttentionError("use_bias must be true or false.")

        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.attention_dropout = nn.Dropout(float(dropout))
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply causal self-attention to hidden states."""

        self._validate_hidden_states(hidden_states)
        normalized_padding_mask = self._validate_padding_mask(padding_mask, hidden_states)
        query = _linear_preserve_input_dtype(hidden_states, self.query_projection)
        key = _linear_preserve_input_dtype(hidden_states, self.key_projection)
        value = _linear_preserve_input_dtype(hidden_states, self.value_projection)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embedding_dim)
        allowed_mask = self._combined_allowed_mask(
            hidden_states=hidden_states,
            padding_mask=normalized_padding_mask,
        )
        scores = scores.masked_fill(~allowed_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~allowed_mask, 0.0)
        row_sums = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(row_sums > 0, weights / row_sums.clamp_min(1e-12), weights)
        dropped_weights = self.attention_dropout(weights)
        output = torch.matmul(dropped_weights, value)
        output = _linear_preserve_input_dtype(output, self.output_projection)

        if return_attention:
            return output, weights
        return output

    @property
    def parameter_count(self) -> int:
        """Return total trainable and frozen parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if not isinstance(hidden_states, torch.Tensor):
            raise AttentionError("hidden_states must be a torch.Tensor.")
        if hidden_states.ndim != 3:
            raise AttentionError("hidden_states must be a three-dimensional tensor.")
        if not hidden_states.is_floating_point():
            raise AttentionError("hidden_states must use a floating-point dtype.")
        batch_size, sequence_length, embedding_dim = hidden_states.shape
        if batch_size <= 0:
            raise AttentionError("hidden_states batch dimension must be greater than zero.")
        if sequence_length < 0:
            raise AttentionError("hidden_states sequence dimension must not be negative.")
        if sequence_length > self.max_sequence_length:
            raise AttentionError(
                "Sequence length exceeds attention maximum length. "
                f"Received {sequence_length} with max_sequence_length "
                f"{self.max_sequence_length}."
            )
        if embedding_dim != self.embedding_dim:
            raise AttentionError(
                "hidden_states last dimension must match embedding_dim. "
                f"Received {embedding_dim} and expected {self.embedding_dim}."
            )

    def _validate_padding_mask(
        self,
        padding_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if padding_mask is None:
            return None
        if not isinstance(padding_mask, torch.Tensor):
            raise AttentionError("padding_mask must be a torch.Tensor.")
        if padding_mask.device != hidden_states.device:
            raise AttentionError("padding_mask must be on the same device as hidden_states.")
        expected_shape = hidden_states.shape[:2]
        if tuple(padding_mask.shape) != tuple(expected_shape):
            raise AttentionError(
                "padding_mask shape must match hidden_states batch and sequence dimensions. "
                f"Received {tuple(padding_mask.shape)} and expected {tuple(expected_shape)}."
            )
        if padding_mask.dtype == torch.bool:
            return padding_mask
        if not padding_mask.dtype.is_floating_point and padding_mask.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise AttentionError("padding_mask must be bool, integer, or floating zero/one values.")
        if padding_mask.numel() == 0:
            return padding_mask.to(dtype=torch.bool)
        if not _should_validate_tensor_values(padding_mask):
            return padding_mask.to(dtype=torch.bool)
        is_zero_or_one = (padding_mask == 0) | (padding_mask == 1)
        if not bool(is_zero_or_one.all().item()):
            raise AttentionError("padding_mask values must be 0/1 or boolean.")
        return padding_mask.to(dtype=torch.bool)

    def _combined_allowed_mask(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        causal = self.causal_mask[:sequence_length, :sequence_length].to(hidden_states.device)
        allowed = causal.unsqueeze(0)
        if padding_mask is not None:
            allowed = allowed & padding_mask.unsqueeze(1)
        return allowed


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product causal self-attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_sequence_length: int,
        dropout: float = 0.1,
        use_bias: bool = True,
        rotary_embeddings: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("num_heads", num_heads)
        _validate_positive_int("max_sequence_length", max_sequence_length)
        if embedding_dim % num_heads != 0:
            raise AttentionError(
                "embedding_dim must be divisible by num_heads. "
                f"Received embedding_dim={embedding_dim} and num_heads={num_heads}."
            )
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0.0 <= dropout < 1.0
        ):
            raise AttentionError("dropout must be at least 0.0 and less than 1.0.")
        if not isinstance(use_bias, bool):
            raise AttentionError("use_bias must be true or false.")
        if not isinstance(rotary_embeddings, bool):
            raise AttentionError("rotary_embeddings must be true or false.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        if rotary_embeddings and self.head_dim % 2:
            raise AttentionError("rotary embeddings require an even head dimension.")
        self.max_sequence_length = max_sequence_length
        self.qkv_projection = nn.Linear(embedding_dim, 3 * embedding_dim, bias=use_bias)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.attention_dropout = nn.Dropout(float(dropout))
        self.rotary_embeddings = rotary_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool)),
            persistent=False,
        )
        if rotary_embeddings:
            cos, sin = _build_rotary_cache(max_sequence_length, self.head_dim)
            self.register_buffer("rotary_cos", cos, persistent=False)
            self.register_buffer("rotary_sin", sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head causal self-attention to hidden states."""

        _validate_hidden_states(
            hidden_states=hidden_states,
            embedding_dim=self.embedding_dim,
            max_sequence_length=self.max_sequence_length,
        )
        normalized_padding_mask = _validate_padding_mask(padding_mask, hidden_states)
        batch_size, sequence_length, _embedding_dim = hidden_states.shape

        qkv = _linear_preserve_input_dtype(hidden_states, self.qkv_projection)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._split_heads(query, batch_size, sequence_length)
        key = self._split_heads(key, batch_size, sequence_length)
        value = self._split_heads(value, batch_size, sequence_length)
        if self.rotary_embeddings:
            query, key = _apply_rotary_embeddings(query, key, self.rotary_cos, self.rotary_sin)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed_mask = self._combined_allowed_mask(
            hidden_states=hidden_states,
            padding_mask=normalized_padding_mask,
        )
        scores = scores.masked_fill(~allowed_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~allowed_mask, 0.0)
        row_sums = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(row_sums > 0, weights / row_sums.clamp_min(1e-12), weights)

        dropped_weights = self.attention_dropout(weights)
        context = torch.matmul(dropped_weights, value)
        context = self._merge_heads(context, batch_size, sequence_length)
        output = _linear_preserve_input_dtype(context, self.output_projection)
        if normalized_padding_mask is not None:
            output = output.masked_fill(~normalized_padding_mask.unsqueeze(-1), 0.0)

        if return_attention:
            return output, weights
        return output

    @property
    def parameter_count(self) -> int:
        """Return total trainable and frozen parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Return trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _split_heads(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def _merge_heads(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        return (
            tensor.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.embedding_dim,
            )
        )

    def _combined_allowed_mask(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        causal = self.causal_mask[:sequence_length, :sequence_length].to(hidden_states.device)
        allowed = causal.unsqueeze(0).unsqueeze(0)
        if padding_mask is not None:
            allowed = allowed & padding_mask.unsqueeze(1).unsqueeze(2)
        return allowed


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AttentionError(f"{name} must be an integer greater than zero.")


def _validate_hidden_states(
    hidden_states: torch.Tensor,
    embedding_dim: int,
    max_sequence_length: int,
) -> None:
    if not isinstance(hidden_states, torch.Tensor):
        raise AttentionError("hidden_states must be a torch.Tensor.")
    if hidden_states.ndim != 3:
        raise AttentionError("hidden_states must be a three-dimensional tensor.")
    if not hidden_states.is_floating_point():
        raise AttentionError("hidden_states must use a floating-point dtype.")
    batch_size, sequence_length, actual_embedding_dim = hidden_states.shape
    if batch_size <= 0:
        raise AttentionError("hidden_states batch dimension must be greater than zero.")
    if sequence_length > max_sequence_length:
        raise AttentionError(
            "Sequence length exceeds attention maximum length. "
            f"Received {sequence_length} with max_sequence_length {max_sequence_length}."
        )
    if actual_embedding_dim != embedding_dim:
        raise AttentionError(
            "hidden_states last dimension must match embedding_dim. "
            f"Received {actual_embedding_dim} and expected {embedding_dim}."
        )


def _validate_padding_mask(
    padding_mask: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    if padding_mask is None:
        return None
    if not isinstance(padding_mask, torch.Tensor):
        raise AttentionError("padding_mask must be a torch.Tensor.")
    if padding_mask.device != hidden_states.device:
        raise AttentionError("padding_mask must be on the same device as hidden_states.")
    expected_shape = hidden_states.shape[:2]
    if tuple(padding_mask.shape) != tuple(expected_shape):
        raise AttentionError(
            "padding_mask shape must match hidden_states batch and sequence dimensions. "
            f"Received {tuple(padding_mask.shape)} and expected {tuple(expected_shape)}."
        )
    if padding_mask.dtype == torch.bool:
        return padding_mask
    if not padding_mask.dtype.is_floating_point and padding_mask.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise AttentionError("padding_mask must be bool, integer, or floating zero/one values.")
    if padding_mask.numel() == 0:
        return padding_mask.to(dtype=torch.bool)
    if not _should_validate_tensor_values(padding_mask):
        return padding_mask.to(dtype=torch.bool)
    is_zero_or_one = (padding_mask == 0) | (padding_mask == 1)
    if not bool(is_zero_or_one.all().item()):
        raise AttentionError("padding_mask values must be 0/1 or boolean.")
    return padding_mask.to(dtype=torch.bool)


def _should_validate_tensor_values(tensor: torch.Tensor) -> bool:
    return tensor.device.type != "mps"


def _linear_preserve_input_dtype(input_tensor: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return layer(input_tensor)
    bias = getattr(layer, "bias", None)
    bias = bias.to(dtype=input_tensor.dtype) if isinstance(bias, torch.Tensor) else None
    weight = weight.to(dtype=input_tensor.dtype)
    return F.linear(input_tensor, weight, bias)


def _build_rotary_cache(
    max_sequence_length: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inverse_frequency = 1.0 / (10000.0 ** (dimensions / head_dim))
    angles = positions * inverse_frequency
    return torch.cos(angles), torch.sin(angles)


def _apply_rotary_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = query.shape[-2]
    cos = (
        cos[:sequence_length]
        .to(device=query.device, dtype=query.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sin = (
        sin[:sequence_length]
        .to(device=query.device, dtype=query.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    return _rotate(query, cos, sin), _rotate(key, cos, sin)


def _rotate(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    rotated = torch.empty_like(tensor)
    rotated[..., 0::2] = (even * cos) - (odd * sin)
    rotated[..., 1::2] = (even * sin) + (odd * cos)
    return rotated


__all__ = ["AttentionError", "CausalSelfAttention", "MultiHeadCausalSelfAttention"]
