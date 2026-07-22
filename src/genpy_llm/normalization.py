"""GPT-style layer normalization for GenPy LLM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NormalizationError(ValueError):
    """Raised when layer normalization configuration or input is invalid."""


class GPTLayerNorm(nn.Module):
    """Layer normalization across the final embedding dimension."""

    def __init__(
        self,
        embedding_dim: int,
        epsilon: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise NormalizationError("embedding_dim must be an integer greater than zero.")
        if not isinstance(epsilon, int | float) or isinstance(epsilon, bool) or epsilon <= 0:
            raise NormalizationError("epsilon must be greater than zero.")
        if not isinstance(elementwise_affine, bool):
            raise NormalizationError("elementwise_affine must be true or false.")

        self.embedding_dim = embedding_dim
        self.epsilon = float(epsilon)
        self.elementwise_affine = elementwise_affine
        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim,
            eps=self.epsilon,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize hidden states over their final dimension."""

        self._validate_hidden_states(hidden_states)
        weight = (
            self.layer_norm.weight.to(dtype=hidden_states.dtype)
            if self.layer_norm.weight is not None
            else None
        )
        bias = (
            self.layer_norm.bias.to(dtype=hidden_states.dtype)
            if self.layer_norm.bias is not None
            else None
        )
        return F.layer_norm(
            hidden_states,
            normalized_shape=(self.embedding_dim,),
            weight=weight,
            bias=bias,
            eps=self.epsilon,
        )

    @property
    def parameter_count(self) -> int:
        """Return total trainable and frozen parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Return trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if not isinstance(hidden_states, torch.Tensor):
            raise NormalizationError("hidden_states must be a torch.Tensor.")
        if hidden_states.ndim != 3:
            raise NormalizationError("hidden_states must be a three-dimensional tensor.")
        if not hidden_states.is_floating_point():
            raise NormalizationError("hidden_states must use a floating-point dtype.")
        batch_size, _sequence_length, actual_embedding_dim = hidden_states.shape
        if batch_size <= 0:
            raise NormalizationError("hidden_states batch dimension must be greater than zero.")
        if actual_embedding_dim != self.embedding_dim:
            raise NormalizationError(
                "hidden_states last dimension must match embedding_dim. "
                f"Received {actual_embedding_dim} and expected {self.embedding_dim}."
            )


__all__ = ["GPTLayerNorm", "NormalizationError"]
