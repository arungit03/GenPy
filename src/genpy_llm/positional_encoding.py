"""Positional encoding modules for GenPy LLM input embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn

from genpy_llm.embeddings import TokenEmbedding


class PositionalEncodingError(ValueError):
    """Raised when positional encoding configuration or input is invalid."""


class PositionalEncoding(nn.Module):
    """Add position information to token embedding vectors."""

    def __init__(
        self,
        embedding_dim: int,
        max_sequence_length: int,
        encoding_type: str = "learned",
        dropout: float = 0.0,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("max_sequence_length", max_sequence_length)
        if encoding_type not in {"learned", "sinusoidal", "none"}:
            raise PositionalEncodingError(
                "encoding_type must be either 'learned', 'sinusoidal', or 'none'."
            )
        if (
            not isinstance(dropout, int | float)
            or isinstance(dropout, bool)
            or not 0.0 <= dropout < 1.0
        ):
            raise PositionalEncodingError("dropout must be at least 0.0 and less than 1.0.")
        if (
            not isinstance(initialization_std, int | float)
            or isinstance(initialization_std, bool)
            or initialization_std <= 0
        ):
            raise PositionalEncodingError("initialization_std must be greater than zero.")

        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(p=float(dropout))

        if encoding_type == "learned":
            self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
            nn.init.normal_(
                self.position_embedding.weight,
                mean=0.0,
                std=float(initialization_std),
            )
        elif encoding_type == "sinusoidal":
            self.position_embedding = None
            self.register_buffer(
                "sinusoidal_encoding",
                _build_sinusoidal_encoding(max_sequence_length, embedding_dim),
                persistent=True,
            )
        else:
            self.position_embedding = None

    def forward(
        self,
        token_embeddings: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Return token embeddings plus positional information."""

        self._validate_input(token_embeddings, position_offset)
        sequence_length = token_embeddings.shape[1]
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=token_embeddings.device,
            dtype=torch.long,
        )

        if self.encoding_type == "none":
            return self.dropout(token_embeddings)
        if self.encoding_type == "learned":
            positional = self.position_embedding(positions)
        else:
            positional = self.sinusoidal_encoding[
                position_offset : position_offset + sequence_length
            ]
        positional = positional.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        output = token_embeddings + positional.unsqueeze(0)
        return self.dropout(output)

    @property
    def trainable_parameter_count(self) -> int:
        """Return trainable positional parameter count."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_input(self, token_embeddings: torch.Tensor, position_offset: int) -> None:
        if not isinstance(token_embeddings, torch.Tensor):
            raise PositionalEncodingError("token_embeddings must be a torch.Tensor.")
        if token_embeddings.ndim != 3:
            raise PositionalEncodingError("token_embeddings must be a three-dimensional tensor.")
        if not token_embeddings.is_floating_point():
            raise PositionalEncodingError("token_embeddings must use a floating-point dtype.")
        if token_embeddings.shape[2] != self.embedding_dim:
            raise PositionalEncodingError(
                "token_embeddings last dimension must match embedding_dim. "
                f"Received {token_embeddings.shape[2]} and expected {self.embedding_dim}."
            )
        if not isinstance(position_offset, int) or isinstance(position_offset, bool):
            raise PositionalEncodingError("position_offset must be an integer.")
        if position_offset < 0:
            raise PositionalEncodingError("position_offset must be greater than or equal to zero.")

        sequence_length = token_embeddings.shape[1]
        end_position = position_offset + sequence_length
        if end_position > self.max_sequence_length:
            raise PositionalEncodingError(
                "Sequence exceeds positional encoding maximum length. "
                f"Received end position {end_position} with max_sequence_length "
                f"{self.max_sequence_length}."
            )


class GPTInputEmbedding(nn.Module):
    """Combine token identity embeddings with positional encodings."""

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        positional_encoding: PositionalEncoding,
    ) -> None:
        super().__init__()
        if not isinstance(token_embedding, TokenEmbedding):
            raise PositionalEncodingError("token_embedding must be a TokenEmbedding.")
        if not isinstance(positional_encoding, PositionalEncoding):
            raise PositionalEncodingError("positional_encoding must be a PositionalEncoding.")
        if token_embedding.embedding_dim != positional_encoding.embedding_dim:
            raise PositionalEncodingError(
                "Token and positional embedding dimensions must match. "
                f"Received {token_embedding.embedding_dim} and "
                f"{positional_encoding.embedding_dim}."
            )
        self.token_embedding = token_embedding
        self.positional_encoding = positional_encoding

    def forward(
        self,
        token_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Embed token IDs and add positional information."""

        token_vectors = self.token_embedding(token_ids)
        return self.positional_encoding(token_vectors, position_offset=position_offset)


def _build_sinusoidal_encoding(max_sequence_length: int, embedding_dim: int) -> torch.Tensor:
    positions = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
    even_dimensions = torch.arange(0, embedding_dim, 2, dtype=torch.float32)
    div_term = torch.exp(even_dimensions * (-math.log(10000.0) / embedding_dim))
    encoding = torch.zeros(max_sequence_length, embedding_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    if embedding_dim > 1:
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PositionalEncodingError(f"{name} must be an integer greater than zero.")


__all__ = [
    "GPTInputEmbedding",
    "PositionalEncoding",
    "PositionalEncodingError",
]
