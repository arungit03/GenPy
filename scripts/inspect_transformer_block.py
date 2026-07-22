"""Inspect one GPT-style pre-norm transformer block."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import ConfigError, load_config
from genpy_llm.dataset import DatasetPreparationError, load_dataset_split
from genpy_llm.device import select_device
from genpy_llm.embeddings import EmbeddingError, create_token_embedding
from genpy_llm.feed_forward import FeedForwardError, resolve_feed_forward_hidden_dim
from genpy_llm.logging_utils import setup_logging
from genpy_llm.positional_encoding import (
    GPTInputEmbedding,
    PositionalEncoding,
    PositionalEncodingError,
)
from genpy_llm.transformer_block import TransformerBlock, TransformerBlockError
from genpy_llm.utils import count_trainable_parameters, set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Parse arguments and inspect a single transformer block."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        app_config = load_config(_resolve_optional_path(args.config))
        logger = setup_logging(
            log_dir=app_config.paths.logs_dir,
            log_file=app_config.logging.log_file,
            level="DEBUG" if args.debug else app_config.logging.level,
        )
        set_seed(args.seed if args.seed is not None else app_config.training.seed)
        device = select_device(args.device or app_config.training.device)

        input_embedding, actual_vocab_size = _build_input_embedding(app_config)
        input_embedding = input_embedding.to(device)
        input_embedding.eval()

        hidden_dim = resolve_feed_forward_hidden_dim(
            embedding_dim=app_config.model.embedding_dim,
            hidden_multiplier=app_config.feed_forward.hidden_multiplier,
            hidden_dim=app_config.feed_forward.hidden_dim,
        )
        block = TransformerBlock(
            embedding_dim=app_config.model.embedding_dim,
            num_heads=app_config.model.num_heads,
            max_sequence_length=app_config.model.context_length,
            feed_forward_hidden_dim=hidden_dim,
            attention_dropout=app_config.transformer_block.attention_dropout,
            feed_forward_dropout=app_config.transformer_block.feed_forward_dropout,
            residual_dropout=app_config.transformer_block.residual_dropout,
            normalization_epsilon=app_config.normalization.epsilon,
            activation=app_config.feed_forward.activation,
            use_bias=app_config.feed_forward.use_bias,
        ).to(device)
        block.eval()

        hidden_states, padding_mask, dataset_path = _load_hidden_states(
            args=args,
            app_config=app_config,
            input_embedding=input_embedding,
            actual_vocab_size=actual_vocab_size,
            device=device,
        )
        with torch.no_grad():
            output, attention_weights = block(
                hidden_states,
                padding_mask=padding_mask,
                return_attention=True,
            )
    except (
        ConfigError,
        DatasetPreparationError,
        EmbeddingError,
        FeedForwardError,
        FileNotFoundError,
        IsADirectoryError,
        OSError,
        PositionalEncodingError,
        RuntimeError,
        TransformerBlockError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Transformer block inspection completed successfully.")
    _print_summary(
        block=block,
        hidden_states=hidden_states,
        output=output,
        attention_weights=attention_weights,
        dataset_path=dataset_path,
        device=device,
        show_attention=args.show_attention,
        show_head=args.show_head,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GenPy LLM transformer block.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--dataset", type=str, default=None, help="Prepared dataset split file.")
    parser.add_argument("--show-batch", action="store_true")
    parser.add_argument("--show-attention", action="store_true")
    parser.add_argument("--show-head", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _build_input_embedding(app_config) -> tuple[GPTInputEmbedding, int]:
    token_embedding, token_metadata = create_token_embedding(
        vocabulary_path=app_config.data.vocabulary_file,
        embedding_config=app_config.embeddings,
        expected_vocab_size=app_config.model.vocab_size,
        encoding=app_config.data.encoding,
    )
    positional_encoding = PositionalEncoding(
        embedding_dim=app_config.embeddings.embedding_dim,
        max_sequence_length=app_config.positional_encoding.max_sequence_length,
        encoding_type=app_config.positional_encoding.type,
        dropout=app_config.positional_encoding.dropout,
        initialization_std=app_config.positional_encoding.initialization_std,
    )
    return GPTInputEmbedding(token_embedding, positional_encoding), token_metadata.vocab_size


def _load_hidden_states(
    args: argparse.Namespace,
    app_config,
    input_embedding: GPTInputEmbedding,
    actual_vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, Path | None]:
    if not args.show_batch:
        sequence_length = min(4, app_config.model.context_length)
        token_ids = torch.randint(0, actual_vocab_size, (2, sequence_length), device=device)
        with torch.no_grad():
            hidden_states = input_embedding(token_ids)
        return hidden_states, None, None

    dataset_path = _resolve_path(args.dataset, app_config.data.train_dataset_file)
    dataset = load_dataset_split(dataset_path)
    if len(dataset) == 0:
        raise DatasetPreparationError("Dataset split is empty.")
    loader = DataLoader(dataset, batch_size=app_config.dataset.batch_size, shuffle=False)
    batch = next(iter(loader))
    token_ids = batch["input_ids"].to(device)
    padding_mask = batch.get("attention_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)
    with torch.no_grad():
        hidden_states = input_embedding(token_ids)
    return hidden_states, padding_mask, dataset_path


def _print_summary(
    block: TransformerBlock,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    attention_weights: torch.Tensor,
    dataset_path: Path | None,
    device: torch.device,
    show_attention: bool,
    show_head: int,
) -> None:
    _validate_head_index(show_head, block.num_heads)
    print("GenPy LLM Transformer Block")
    print("===========================")
    print(f"Dataset file: {dataset_path if dataset_path is not None else 'synthetic sample'}")
    print(f"Device: {device}")
    print(f"Embedding dimension: {block.embedding_dim}")
    print(f"Head count: {block.num_heads}")
    print(f"Head dimension: {block.head_dim}")
    print(f"FFN hidden dimension: {block.feed_forward_hidden_dim}")
    print(f"Input shape: {list(hidden_states.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"Attention weights shape: {list(attention_weights.shape)}")
    print(f"Causal masking confirmed: {_future_probabilities_are_zero(attention_weights)}")
    print(f"Total parameters: {block.parameter_count}")
    print(f"Trainable parameters: {count_trainable_parameters(block)}")
    print(f"Attention parameters: {block.attention_parameter_count}")
    print(f"FFN parameters: {block.feed_forward_parameter_count}")
    print(f"LayerNorm parameters: {block.layer_norm_parameter_count}")
    print(f"Finite output: {bool(torch.isfinite(output).all().item())}")
    if show_attention:
        _print_attention_matrix(attention_weights[0, show_head], show_head)
    print()
    print("Outputs and attention patterns are random before training.")


def _future_probabilities_are_zero(weights: torch.Tensor) -> bool:
    sequence_length = weights.shape[-1]
    future_mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=weights.device),
        diagonal=1,
    )
    future_values = weights[..., future_mask]
    return bool(torch.allclose(future_values, torch.zeros_like(future_values), atol=1e-7))


def _print_attention_matrix(matrix: torch.Tensor, head_index: int, max_rows: int = 8) -> None:
    shown = min(matrix.shape[0], max_rows)
    print()
    print(f"Attention matrix preview for head {head_index}")
    print("===================================")
    for row_index in range(shown):
        values = [f"{value:.4f}" for value in matrix[row_index, :shown].tolist()]
        print(f"{row_index:02d}: {' '.join(values)}")
    if matrix.shape[0] > shown:
        print(f"... truncated to first {shown} rows/columns")


def _validate_head_index(head_index: int, num_heads: int) -> None:
    if head_index < 0 or head_index >= num_heads:
        raise TransformerBlockError(
            f"show_head must be between 0 and {num_heads - 1}. Received {head_index}."
        )


def _resolve_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return _resolve_against_project_root(Path(value))


def _resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    return _resolve_against_project_root(Path(value))


def _resolve_against_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Transformer block inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
