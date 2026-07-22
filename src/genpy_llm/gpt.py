"""Complete untrained GPT decoder architecture for GenPy LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from genpy_llm.config import AppConfig, EmbeddingConfig
from genpy_llm.embeddings import TokenEmbedding
from genpy_llm.feed_forward import resolve_feed_forward_hidden_dim
from genpy_llm.normalization import GPTLayerNorm, NormalizationError
from genpy_llm.positional_encoding import PositionalEncoding, PositionalEncodingError
from genpy_llm.transformer_block import TransformerBlock, TransformerBlockError
from genpy_llm.vocabulary import Vocabulary


class GPTModelError(ValueError):
    """Raised when GPT model configuration or input is invalid."""


@dataclass(frozen=True)
class GPTModelMetadata:
    """Readable metadata for an untrained GPT decoder."""

    vocab_size: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    context_length: int
    feed_forward_hidden_dim: int
    total_parameters: int
    trainable_parameters: int
    tie_embeddings: bool

    def summary(self) -> str:
        """Return a readable multi-line summary."""

        return "\n".join(
            [
                "GPT decoder summary",
                "===================",
                f"Vocabulary size: {self.vocab_size}",
                f"Embedding dimension: {self.embedding_dim}",
                f"Head count: {self.num_heads}",
                f"Layer count: {self.num_layers}",
                f"Context length: {self.context_length}",
                f"FFN hidden dimension: {self.feed_forward_hidden_dim}",
                f"Total parameters: {self.total_parameters}",
                f"Trainable parameters: {self.trainable_parameters}",
                f"Tied token/output embeddings: {self.tie_embeddings}",
            ]
        )


class GPTModel(nn.Module):
    """An untrained GPT-style decoder that returns vocabulary logits."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        context_length: int,
        feed_forward_hidden_dim: int,
        padding_idx: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float | None = None,
        feed_forward_dropout: float | None = None,
        residual_dropout: float | None = None,
        normalization_epsilon: float = 1e-5,
        activation: str = "gelu",
        use_bias: bool = True,
        positional_encoding_type: str = "learned",
        tie_embeddings: bool = True,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        _validate_positive_int("vocab_size", vocab_size)
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("num_heads", num_heads)
        _validate_positive_int("num_layers", num_layers)
        _validate_positive_int("context_length", context_length)
        _validate_positive_int("feed_forward_hidden_dim", feed_forward_hidden_dim)
        _validate_dropout(dropout)
        if attention_dropout is not None:
            _validate_dropout(attention_dropout)
        if feed_forward_dropout is not None:
            _validate_dropout(feed_forward_dropout)
        if residual_dropout is not None:
            _validate_dropout(residual_dropout)
        _validate_positive_float("initialization_std", initialization_std)
        _validate_padding_idx(padding_idx, vocab_size)
        if embedding_dim % num_heads != 0:
            raise GPTModelError("embedding_dim must be divisible by num_heads.")
        if not isinstance(use_bias, bool):
            raise GPTModelError("use_bias must be true or false.")
        if not isinstance(tie_embeddings, bool):
            raise GPTModelError("tie_embeddings must be true or false.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context_length = context_length
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.tie_embeddings = tie_embeddings
        self.padding_idx = padding_idx
        self.initialization_std = float(initialization_std)
        self.gradient_checkpointing = False
        selected_attention_dropout = float(
            dropout if attention_dropout is None else attention_dropout
        )
        selected_ffn_dropout = float(
            dropout if feed_forward_dropout is None else feed_forward_dropout
        )
        selected_residual_dropout = float(dropout if residual_dropout is None else residual_dropout)

        embedding_config = EmbeddingConfig(
            embedding_dim=embedding_dim,
            initialization="normal",
            initialization_std=float(initialization_std),
            scale_embeddings=False,
            freeze_embeddings=False,
            zero_padding_embedding=True,
        )
        try:
            rotary_embeddings = positional_encoding_type == "rotary"
            additive_position_type = "none" if rotary_embeddings else positional_encoding_type
            self.token_embedding = TokenEmbedding(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                config=embedding_config,
            )
            self.positional_encoding = PositionalEncoding(
                embedding_dim=embedding_dim,
                max_sequence_length=context_length,
                encoding_type=additive_position_type,
                dropout=0.0,
                initialization_std=float(initialization_std),
            )
            self.embedding_dropout = nn.Dropout(float(dropout))
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        max_sequence_length=context_length,
                        feed_forward_hidden_dim=feed_forward_hidden_dim,
                        attention_dropout=selected_attention_dropout,
                        feed_forward_dropout=selected_ffn_dropout,
                        residual_dropout=selected_residual_dropout,
                        normalization_epsilon=normalization_epsilon,
                        activation=activation,
                        use_bias=use_bias,
                        rotary_embeddings=rotary_embeddings,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final_norm = GPTLayerNorm(
                embedding_dim=embedding_dim,
                epsilon=normalization_epsilon,
            )
        except (NormalizationError, PositionalEncodingError, TransformerBlockError) as exc:
            raise GPTModelError(str(exc)) from exc

        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self._initialize_decoder_weights()
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._zero_padding_embedding()

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Return unnormalized vocabulary logits for token IDs."""

        self._validate_inputs(input_ids, padding_mask)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.positional_encoding(hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                if return_attention:
                    raise GPTModelError(
                        "return_attention is not supported while gradient checkpointing is enabled."
                    )

                def block_forward(states: torch.Tensor, block=block) -> torch.Tensor:
                    return block(states, padding_mask=padding_mask)

                hidden_states = checkpoint(block_forward, hidden_states, use_reentrant=False)
                continue
            if return_attention:
                hidden_states, attention = block(
                    hidden_states,
                    padding_mask=padding_mask,
                    return_attention=True,
                )
                attention_maps.append(attention)
            else:
                hidden_states = block(hidden_states, padding_mask=padding_mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if return_attention:
            return logits, attention_maps
        return logits

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing across transformer blocks during training."""

        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable activation checkpointing."""

        self.gradient_checkpointing = False

    @property
    def parameter_count(self) -> int:
        """Return total unique trainable and frozen parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Return unique trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def metadata(self) -> GPTModelMetadata:
        """Return compact model metadata."""

        return GPTModelMetadata(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            context_length=self.context_length,
            feed_forward_hidden_dim=self.feed_forward_hidden_dim,
            total_parameters=self.parameter_count,
            trainable_parameters=self.trainable_parameter_count,
            tie_embeddings=self.tie_embeddings,
        )

    def embeddings_are_tied(self) -> bool:
        """Return whether token embedding and output head share one parameter."""

        return self.lm_head.weight is self.token_embedding.weight

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> None:
        if not isinstance(input_ids, torch.Tensor):
            raise GPTModelError("input_ids must be a torch.Tensor.")
        if input_ids.dtype != torch.long:
            raise GPTModelError("input_ids must use torch.long dtype.")
        if input_ids.ndim != 2:
            raise GPTModelError("input_ids must be a two-dimensional tensor.")
        batch_size, sequence_length = input_ids.shape
        if batch_size <= 0:
            raise GPTModelError("input_ids batch dimension must be greater than zero.")
        if sequence_length <= 0:
            raise GPTModelError("input_ids sequence dimension must be greater than zero.")
        if sequence_length > self.context_length:
            raise GPTModelError(
                "Sequence length exceeds context length. "
                f"Received {sequence_length} with context_length {self.context_length}."
            )
        if input_ids.numel() > 0 and _should_validate_tensor_values(input_ids):
            if bool((input_ids < 0).any().item()):
                raise GPTModelError("input_ids must not contain negative token IDs.")
            if bool((input_ids >= self.vocab_size).any().item()):
                raise GPTModelError("input_ids must be below vocab_size.")
        _validate_padding_mask(padding_mask, input_ids)

    def _initialize_decoder_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.initialization_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _zero_padding_embedding(self) -> None:
        if self.padding_idx is None:
            return
        with torch.no_grad():
            self.token_embedding.weight[self.padding_idx].zero_()


def create_gpt_model(
    vocabulary_path: Path,
    config: AppConfig,
) -> tuple[GPTModel, GPTModelMetadata]:
    """Create a GPT decoder from config using the saved vocabulary size."""

    vocabulary = Vocabulary.load(vocabulary_path, encoding=config.data.encoding)
    feed_forward_hidden_dim = resolve_feed_forward_hidden_dim(
        embedding_dim=config.model.embedding_dim,
        hidden_multiplier=config.feed_forward.hidden_multiplier,
        hidden_dim=config.feed_forward.hidden_dim,
    )
    model = GPTModel(
        vocab_size=len(vocabulary),
        embedding_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        context_length=config.model.context_length,
        feed_forward_hidden_dim=feed_forward_hidden_dim,
        padding_idx=vocabulary.pad_id,
        dropout=config.model.dropout,
        normalization_epsilon=config.normalization.epsilon,
        activation=config.feed_forward.activation,
        use_bias=config.model.use_bias,
        positional_encoding_type=config.positional_encoding.type,
        tie_embeddings=config.model.tie_embeddings,
        initialization_std=config.model.initialization_std,
    )
    return model, model.metadata()


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GPTModelError(f"{name} must be an integer greater than zero.")


def _validate_positive_float(name: str, value: float) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise GPTModelError(f"{name} must be greater than zero.")


def _validate_dropout(value: float) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not 0.0 <= value < 1.0:
        raise GPTModelError("dropout must be at least 0.0 and less than 1.0.")


def _validate_padding_idx(padding_idx: int | None, vocab_size: int) -> None:
    if padding_idx is None:
        return
    if not isinstance(padding_idx, int) or isinstance(padding_idx, bool):
        raise GPTModelError("padding_idx must be an integer or None.")
    if padding_idx < 0 or padding_idx >= vocab_size:
        raise GPTModelError("padding_idx must be between 0 and vocab_size - 1.")


def _validate_padding_mask(
    padding_mask: torch.Tensor | None,
    input_ids: torch.Tensor,
) -> None:
    if padding_mask is None:
        return
    if not isinstance(padding_mask, torch.Tensor):
        raise GPTModelError("padding_mask must be a torch.Tensor.")
    if padding_mask.shape != input_ids.shape:
        raise GPTModelError("padding_mask shape must match input_ids shape.")
    if padding_mask.device != input_ids.device:
        raise GPTModelError("padding_mask must be on the same device as input_ids.")
    if padding_mask.dtype == torch.bool or padding_mask.numel() == 0:
        return
    if not padding_mask.dtype.is_floating_point and padding_mask.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise GPTModelError("padding_mask must be bool, integer, or floating zero/one.")
    if not _should_validate_tensor_values(padding_mask):
        return
    is_zero_or_one = (padding_mask == 0) | (padding_mask == 1)
    if not bool(is_zero_or_one.all().item()):
        raise GPTModelError("padding_mask values must be 0/1 or boolean.")


def _should_validate_tensor_values(tensor: torch.Tensor) -> bool:
    return tensor.device.type != "mps"


__all__ = ["GPTModel", "GPTModelError", "GPTModelMetadata", "create_gpt_model"]
