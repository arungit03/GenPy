"""Token embedding layer and inspection helpers for GenPy LLM."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from genpy_llm.config import EmbeddingConfig
from genpy_llm.vocabulary import Vocabulary

EMBEDDING_CHECKPOINT_FORMAT_VERSION = 1
LOGGER = logging.getLogger("genpy_llm")


class EmbeddingError(ValueError):
    """Raised when token embedding construction or use is invalid."""


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Serializable summary of a token embedding matrix."""

    vocab_size: int
    embedding_dim: int
    padding_idx: int | None
    parameter_count: int
    trainable_parameter_count: int
    initialization: str
    initialization_std: float
    scale_embeddings: bool
    frozen: bool

    def summary(self) -> str:
        """Return a readable multi-line summary."""

        return "\n".join(
            [
                "Token embedding summary",
                "=======================",
                f"Vocabulary size: {self.vocab_size}",
                f"Embedding dimension: {self.embedding_dim}",
                f"Padding index: {self.padding_idx}",
                f"Parameters: {self.parameter_count}",
                f"Trainable parameters: {self.trainable_parameter_count}",
                f"Initialization: {self.initialization}",
                f"Initialization std: {self.initialization_std}",
                f"Scale embeddings: {self.scale_embeddings}",
                f"Frozen: {self.frozen}",
            ]
        )

    def __str__(self) -> str:
        """Return the same readable summary used by the CLI."""

        return self.summary()


@dataclass(frozen=True)
class EmbeddingWeightStats:
    """Basic numerical diagnostics for embedding weights."""

    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    l2_norm: float
    zero_rows: int

    def summary(self) -> str:
        """Return a readable multi-line summary."""

        return "\n".join(
            [
                "Embedding weight statistics",
                "===========================",
                f"Minimum: {self.minimum:.8f}",
                f"Maximum: {self.maximum:.8f}",
                f"Mean: {self.mean:.8f}",
                f"Standard deviation: {self.standard_deviation:.8f}",
                f"L2 norm: {self.l2_norm:.8f}",
                f"Zero rows: {self.zero_rows}",
            ]
        )

    def __str__(self) -> str:
        """Return the same readable summary used by the CLI."""

        return self.summary()


@dataclass(frozen=True)
class TokenEmbeddingRecord:
    """One inspected token embedding vector."""

    requested_token: str
    token: str
    token_id: int
    vector: tuple[float, ...]
    l2_norm: float
    mapped_to_unknown: bool


class TokenEmbedding(nn.Module):
    """Lookup token IDs and return trainable embedding vectors."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        config: EmbeddingConfig | None = None,
    ) -> None:
        super().__init__()
        _validate_positive_int("vocab_size", vocab_size)
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_padding_idx(padding_idx, vocab_size)

        self.config = config or _default_config(embedding_dim)
        if self.config.embedding_dim != embedding_dim:
            raise EmbeddingError(
                "Embedding config dimension must match module dimension. "
                f"Received config.embedding_dim={self.config.embedding_dim} and "
                f"embedding_dim={embedding_dim}."
            )
        _validate_config(self.config)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.padding_idx = padding_idx
        self._initialize_weights()
        if self.config.freeze_embeddings:
            self.embedding.weight.requires_grad_(False)

    @property
    def weight(self) -> torch.nn.Parameter:
        """Return the underlying embedding weight parameter."""

        return self.embedding.weight

    @property
    def num_embeddings(self) -> int:
        """Return vocabulary size."""

        return self.embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        """Return embedding vector width."""

        return self.embedding.embedding_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert a one- or two-dimensional tensor of token IDs into vectors."""

        self._validate_token_ids(token_ids)
        output = self.embedding(token_ids)
        if self.config.scale_embeddings:
            return output * math.sqrt(self.embedding_dim)
        return output

    def _initialize_weights(self) -> None:
        with torch.no_grad():
            if self.config.initialization == "normal":
                nn.init.normal_(self.weight, mean=0.0, std=float(self.config.initialization_std))
            elif self.config.initialization == "uniform":
                bound = 1.0 / math.sqrt(self.embedding_dim)
                nn.init.uniform_(self.weight, a=-bound, b=bound)
            elif self.config.initialization == "xavier_uniform":
                nn.init.xavier_uniform_(self.weight)
            else:
                raise EmbeddingError(
                    f"Unsupported embedding initialization: {self.config.initialization}"
                )

            if self.config.zero_padding_embedding and self.padding_idx is not None:
                self.weight[self.padding_idx].zero_()

    def _validate_token_ids(self, token_ids: torch.Tensor) -> None:
        if not isinstance(token_ids, torch.Tensor):
            raise EmbeddingError("token_ids must be a torch.Tensor.")
        if token_ids.dtype != torch.long:
            raise EmbeddingError("token_ids must use torch.long dtype.")
        if token_ids.ndim not in {1, 2}:
            raise EmbeddingError("token_ids must be one- or two-dimensional.")
        if token_ids.numel() == 0:
            return
        if not _should_validate_tensor_values(token_ids):
            return

        negative_positions = (token_ids < 0).nonzero(as_tuple=False)
        if negative_positions.numel() > 0:
            position = _format_position(negative_positions[0])
            value = int(token_ids[tuple(negative_positions[0].tolist())].item())
            raise EmbeddingError(f"token_ids contains negative ID {value} at position {position}.")

        high_positions = (token_ids >= self.num_embeddings).nonzero(as_tuple=False)
        if high_positions.numel() > 0:
            position = _format_position(high_positions[0])
            value = int(token_ids[tuple(high_positions[0].tolist())].item())
            raise EmbeddingError(
                f"token_ids contains out-of-range ID {value} at position {position}; "
                f"vocabulary size is {self.num_embeddings}."
            )


def _should_validate_tensor_values(tensor: torch.Tensor) -> bool:
    return tensor.device.type != "mps"


def create_token_embedding(
    vocabulary_path: Path,
    embedding_config: EmbeddingConfig,
    expected_vocab_size: int | None = None,
    encoding: str = "utf-8",
) -> tuple[TokenEmbedding, EmbeddingMetadata]:
    """Create a token embedding module sized from a saved vocabulary file."""

    vocabulary = Vocabulary.load(vocabulary_path, encoding=encoding)
    actual_vocab_size = len(vocabulary)
    if expected_vocab_size is not None and expected_vocab_size != actual_vocab_size:
        LOGGER.warning(
            "Configured vocabulary size %s differs from actual vocabulary size %s. "
            "Using the actual vocabulary size.",
            expected_vocab_size,
            actual_vocab_size,
        )

    embedding = TokenEmbedding(
        vocab_size=actual_vocab_size,
        embedding_dim=embedding_config.embedding_dim,
        padding_idx=vocabulary.pad_id,
        config=embedding_config,
    )
    return embedding, build_embedding_metadata(embedding)


def build_embedding_metadata(embedding: TokenEmbedding) -> EmbeddingMetadata:
    """Build metadata from a token embedding module."""

    parameter_count = sum(parameter.numel() for parameter in embedding.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in embedding.parameters() if parameter.requires_grad
    )
    return EmbeddingMetadata(
        vocab_size=embedding.num_embeddings,
        embedding_dim=embedding.embedding_dim,
        padding_idx=embedding.padding_idx,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        initialization=embedding.config.initialization,
        initialization_std=float(embedding.config.initialization_std),
        scale_embeddings=embedding.config.scale_embeddings,
        frozen=not embedding.weight.requires_grad,
    )


def calculate_embedding_statistics(embedding: TokenEmbedding) -> EmbeddingWeightStats:
    """Calculate basic numerical statistics for embedding weights."""

    with torch.no_grad():
        weight = embedding.weight.detach().float().cpu()
        zero_rows = int((weight == 0).all(dim=1).sum().item()) if weight.numel() else 0
        return EmbeddingWeightStats(
            minimum=float(weight.min().item()),
            maximum=float(weight.max().item()),
            mean=float(weight.mean().item()),
            standard_deviation=float(weight.std(unbiased=False).item()),
            l2_norm=float(torch.linalg.vector_norm(weight).item()),
            zero_rows=zero_rows,
        )


def inspect_token_embeddings(
    embedding: TokenEmbedding,
    vocabulary: Vocabulary,
    tokens: Sequence[str],
    max_dimensions: int = 8,
) -> list[TokenEmbeddingRecord]:
    """Return compact embedding vectors for selected tokens."""

    if not isinstance(max_dimensions, int) or max_dimensions <= 0:
        raise EmbeddingError("max_dimensions must be an integer greater than zero.")
    records: list[TokenEmbeddingRecord] = []
    with torch.no_grad():
        weight = embedding.weight.detach().cpu()
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise EmbeddingError("Tokens to inspect must be non-empty strings.")
            token_id = vocabulary.token_id(token)
            mapped_to_unknown = token not in vocabulary.token_to_id
            resolved_token = vocabulary.id_token(token_id)
            vector = weight[token_id]
            records.append(
                TokenEmbeddingRecord(
                    requested_token=token,
                    token=resolved_token,
                    token_id=token_id,
                    vector=tuple(float(value) for value in vector[:max_dimensions].tolist()),
                    l2_norm=float(torch.linalg.vector_norm(vector.float()).item()),
                    mapped_to_unknown=mapped_to_unknown,
                )
            )
    return records


def cosine_similarity_between_tokens(
    embedding: TokenEmbedding,
    vocabulary: Vocabulary,
    first_token: str,
    second_token: str,
) -> float:
    """Return cosine similarity between two token vectors."""

    with torch.no_grad():
        first_id = vocabulary.token_id(first_token)
        second_id = vocabulary.token_id(second_token)
        first = embedding.weight[first_id].detach().float()
        second = embedding.weight[second_id].detach().float()
        first_norm = torch.linalg.vector_norm(first)
        second_norm = torch.linalg.vector_norm(second)
        if first_norm.item() == 0.0 or second_norm.item() == 0.0:
            raise EmbeddingError("Cannot compute cosine similarity for a zero vector.")
        return float(torch.dot(first, second).div(first_norm * second_norm).item())


def save_embedding_checkpoint(
    embedding: TokenEmbedding,
    output_path: Path,
    metadata: EmbeddingMetadata | None = None,
) -> None:
    """Save an embedding-only checkpoint atomically."""

    output_path = output_path.resolve()
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"Embedding checkpoint path is a directory: {output_path}")
    metadata = metadata or build_embedding_metadata(embedding)
    payload = {
        "format_version": EMBEDDING_CHECKPOINT_FORMAT_VERSION,
        "module_type": "TokenEmbedding",
        "metadata": asdict(metadata),
        "configuration": asdict(embedding.config),
        "state_dict": _cpu_state_dict(embedding),
    }
    _torch_save_atomic(output_path, payload)


def load_embedding_checkpoint(
    input_path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[TokenEmbedding, EmbeddingMetadata]:
    """Load an embedding-only checkpoint and validate its contents."""

    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Embedding checkpoint not found: {input_path}")
    if not input_path.is_file():
        raise IsADirectoryError(f"Embedding checkpoint path is not a file: {input_path}")

    payload = _safe_torch_load(input_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise EmbeddingError("Embedding checkpoint must contain a dictionary.")
    if payload.get("format_version") != EMBEDDING_CHECKPOINT_FORMAT_VERSION:
        raise EmbeddingError(
            f"Unsupported embedding checkpoint format version: {payload.get('format_version')}"
        )
    if payload.get("module_type") != "TokenEmbedding":
        raise EmbeddingError("Embedding checkpoint module_type must be TokenEmbedding.")

    metadata = _metadata_from_payload(payload.get("metadata"))
    config = _config_from_payload(payload.get("configuration"), metadata.embedding_dim)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise EmbeddingError("Embedding checkpoint state_dict must be a dictionary.")
    weight = state_dict.get("embedding.weight")
    if not isinstance(weight, torch.Tensor):
        raise EmbeddingError("Embedding checkpoint is missing embedding.weight.")
    expected_shape = (metadata.vocab_size, metadata.embedding_dim)
    if tuple(weight.shape) != expected_shape:
        raise EmbeddingError(
            "Embedding weight shape "
            f"{tuple(weight.shape)} does not match expected {expected_shape}."
        )

    embedding = TokenEmbedding(
        vocab_size=metadata.vocab_size,
        embedding_dim=metadata.embedding_dim,
        padding_idx=metadata.padding_idx,
        config=config,
    )
    try:
        embedding.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise EmbeddingError(f"Could not load embedding checkpoint state_dict: {exc}") from exc
    if metadata.frozen:
        embedding.weight.requires_grad_(False)
    else:
        embedding.weight.requires_grad_(True)
    embedding.to(map_location)
    return embedding, metadata


def _default_config(embedding_dim: int) -> EmbeddingConfig:
    return EmbeddingConfig(
        embedding_dim=embedding_dim,
        initialization="normal",
        initialization_std=0.02,
        scale_embeddings=False,
        freeze_embeddings=False,
        zero_padding_embedding=True,
    )


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EmbeddingError(f"{name} must be an integer greater than zero.")


def _validate_padding_idx(padding_idx: int | None, vocab_size: int) -> None:
    if padding_idx is None:
        return
    if not isinstance(padding_idx, int) or isinstance(padding_idx, bool):
        raise EmbeddingError("padding_idx must be an integer or None.")
    if padding_idx < 0 or padding_idx >= vocab_size:
        raise EmbeddingError(
            f"padding_idx must be between 0 and vocab_size - 1. Received {padding_idx}."
        )


def _validate_config(config: EmbeddingConfig) -> None:
    _validate_positive_int("embedding_dim", config.embedding_dim)
    if config.initialization not in {"normal", "uniform", "xavier_uniform"}:
        raise EmbeddingError(f"Unsupported embedding initialization: {config.initialization}")
    if (
        not isinstance(config.initialization_std, int | float)
        or isinstance(config.initialization_std, bool)
        or config.initialization_std <= 0
    ):
        raise EmbeddingError("initialization_std must be greater than zero.")
    for name in ["scale_embeddings", "freeze_embeddings", "zero_padding_embedding"]:
        if not isinstance(getattr(config, name), bool):
            raise EmbeddingError(f"{name} must be true or false.")


def _format_position(position: torch.Tensor) -> str:
    values = position.tolist()
    if len(values) == 1:
        return str(values[0])
    return f"({values[0]}, {values[1]})"


def _cpu_state_dict(embedding: TokenEmbedding) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in embedding.state_dict().items()}


def _metadata_from_payload(data: Any) -> EmbeddingMetadata:
    if not isinstance(data, dict):
        raise EmbeddingError("Embedding checkpoint metadata must be a dictionary.")
    try:
        metadata = EmbeddingMetadata(**data)
    except TypeError as exc:
        raise EmbeddingError(f"Embedding checkpoint metadata is invalid: {exc}") from exc
    _validate_positive_int("metadata.vocab_size", metadata.vocab_size)
    _validate_positive_int("metadata.embedding_dim", metadata.embedding_dim)
    _validate_padding_idx(metadata.padding_idx, metadata.vocab_size)
    return metadata


def _config_from_payload(data: Any, embedding_dim: int) -> EmbeddingConfig:
    if not isinstance(data, dict):
        raise EmbeddingError("Embedding checkpoint configuration must be a dictionary.")
    values = dict(data)
    values["embedding_dim"] = embedding_dim
    try:
        config = EmbeddingConfig(**values)
    except TypeError as exc:
        raise EmbeddingError(f"Embedding checkpoint configuration is invalid: {exc}") from exc
    _validate_config(config)
    return config


def _safe_torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except RuntimeError as exc:
        raise EmbeddingError(f"Could not load embedding checkpoint: {exc}") from exc


def _torch_save_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        temp_path = _create_temp_path(path)
        torch.save(payload, temp_path)
        temp_path.replace(path)
        temp_path = None
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _create_temp_path(output_path: Path) -> Path:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    return Path(temp_name)


__all__ = [
    "EMBEDDING_CHECKPOINT_FORMAT_VERSION",
    "EmbeddingError",
    "EmbeddingMetadata",
    "EmbeddingWeightStats",
    "TokenEmbedding",
    "TokenEmbeddingRecord",
    "build_embedding_metadata",
    "calculate_embedding_statistics",
    "cosine_similarity_between_tokens",
    "create_token_embedding",
    "inspect_token_embeddings",
    "load_embedding_checkpoint",
    "save_embedding_checkpoint",
]
