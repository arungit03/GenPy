"""Position-wise feed-forward network for GenPy LLM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class FeedForwardError(ValueError):
    """Raised when feed-forward configuration or input is invalid."""


@dataclass(frozen=True)
class FeedForwardMetadata:
    """Readable metadata for a feed-forward network."""

    embedding_dim: int
    hidden_dim: int
    expansion_ratio: float
    activation: str
    dropout: float
    parameter_count: int
    trainable_parameter_count: int

    def summary(self) -> str:
        """Return a readable multi-line summary."""

        return "\n".join(
            [
                "Feed-forward network summary",
                "============================",
                f"Embedding dimension: {self.embedding_dim}",
                f"Hidden dimension: {self.hidden_dim}",
                f"Expansion ratio: {self.expansion_ratio:.2f}",
                f"Activation: {self.activation}",
                f"Dropout: {self.dropout}",
                f"Parameters: {self.parameter_count}",
                f"Trainable parameters: {self.trainable_parameter_count}",
            ]
        )

    def __str__(self) -> str:
        """Return the same readable summary used by CLI output."""

        return self.summary()


class FeedForwardNetwork(nn.Module):
    """Apply the GPT position-wise feed-forward transformation."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_bias: bool = True,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("hidden_dim", hidden_dim)
        if activation not in {"gelu", "relu", "silu", "swiglu"}:
            raise FeedForwardError("activation must be one of: gelu, relu, silu, swiglu.")
        if (
            not isinstance(dropout, int | float)
            or isinstance(dropout, bool)
            or not 0.0 <= dropout < 1.0
        ):
            raise FeedForwardError("dropout must be at least 0.0 and less than 1.0.")
        if not isinstance(use_bias, bool):
            raise FeedForwardError("use_bias must be true or false.")
        if (
            not isinstance(initialization_std, int | float)
            or isinstance(initialization_std, bool)
            or initialization_std <= 0
        ):
            raise FeedForwardError("initialization_std must be greater than zero.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        input_width = hidden_dim * 2 if activation == "swiglu" else hidden_dim
        self.input_projection = nn.Linear(embedding_dim, input_width, bias=use_bias)
        self.activation = _build_activation(activation)
        self.dropout = nn.Dropout(float(dropout))
        self.output_projection = nn.Linear(hidden_dim, embedding_dim, bias=use_bias)
        self._initialize_weights(float(initialization_std))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform each token position independently."""

        self._validate_hidden_states(hidden_states)
        hidden = _linear_preserve_input_dtype(hidden_states, self.input_projection)
        if self.activation_name == "swiglu":
            values, gate = hidden.chunk(2, dim=-1)
            hidden = values * self.activation(gate)
        else:
            hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return _linear_preserve_input_dtype(hidden, self.output_projection)

    @property
    def parameter_count(self) -> int:
        """Return total trainable and frozen parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Return trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def metadata(self) -> FeedForwardMetadata:
        """Return a compact metadata object."""

        return FeedForwardMetadata(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            expansion_ratio=self.hidden_dim / self.embedding_dim,
            activation=self.activation_name,
            dropout=float(self.dropout.p),
            parameter_count=self.parameter_count,
            trainable_parameter_count=self.trainable_parameter_count,
        )

    def _initialize_weights(self, initialization_std: float) -> None:
        for layer in [self.input_projection, self.output_projection]:
            nn.init.normal_(layer.weight, mean=0.0, std=initialization_std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if not isinstance(hidden_states, torch.Tensor):
            raise FeedForwardError("hidden_states must be a torch.Tensor.")
        if hidden_states.ndim != 3:
            raise FeedForwardError("hidden_states must be a three-dimensional tensor.")
        if not hidden_states.is_floating_point():
            raise FeedForwardError("hidden_states must use a floating-point dtype.")
        batch_size, _sequence_length, actual_embedding_dim = hidden_states.shape
        if batch_size <= 0:
            raise FeedForwardError("hidden_states batch dimension must be greater than zero.")
        if actual_embedding_dim != self.embedding_dim:
            raise FeedForwardError(
                "hidden_states last dimension must match embedding_dim. "
                f"Received {actual_embedding_dim} and expected {self.embedding_dim}."
            )


def resolve_feed_forward_hidden_dim(
    embedding_dim: int,
    hidden_multiplier: int,
    hidden_dim: int | None = None,
) -> int:
    """Resolve explicit or multiplier-based FFN hidden dimension."""

    _validate_positive_int("embedding_dim", embedding_dim)
    _validate_positive_int("hidden_multiplier", hidden_multiplier)
    if hidden_dim is not None:
        _validate_positive_int("hidden_dim", hidden_dim)
        return hidden_dim
    return embedding_dim * hidden_multiplier


def _build_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "swiglu":
        return nn.SiLU()
    raise FeedForwardError("activation must be one of: gelu, relu, silu, swiglu.")


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FeedForwardError(f"{name} must be an integer greater than zero.")


def _linear_preserve_input_dtype(input_tensor: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return layer(input_tensor)
    bias = getattr(layer, "bias", None)
    bias = bias.to(dtype=input_tensor.dtype) if isinstance(bias, torch.Tensor) else None
    weight = weight.to(dtype=input_tensor.dtype)
    return F.linear(input_tensor, weight, bias)


__all__ = [
    "FeedForwardError",
    "FeedForwardMetadata",
    "FeedForwardNetwork",
    "resolve_feed_forward_hidden_dim",
]
