"""Inspect multi-head causal self-attention without building a full GPT model."""

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

from genpy_llm.attention import AttentionError, MultiHeadCausalSelfAttention
from genpy_llm.config import ConfigError, load_config
from genpy_llm.dataset import DatasetPreparationError, load_dataset_split
from genpy_llm.device import select_device
from genpy_llm.embeddings import EmbeddingError, create_token_embedding
from genpy_llm.logging_utils import setup_logging
from genpy_llm.positional_encoding import (
    GPTInputEmbedding,
    PositionalEncoding,
    PositionalEncodingError,
)
from genpy_llm.utils import count_trainable_parameters, set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Parse arguments and inspect multi-head causal self-attention."""

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
        num_heads = args.num_heads if args.num_heads is not None else app_config.model.num_heads

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
        input_embedding = GPTInputEmbedding(token_embedding, positional_encoding).to(device)
        attention = MultiHeadCausalSelfAttention(
            embedding_dim=app_config.embeddings.embedding_dim,
            num_heads=num_heads,
            max_sequence_length=app_config.positional_encoding.max_sequence_length,
            dropout=app_config.attention.dropout,
            use_bias=app_config.attention.use_bias,
        ).to(device)
        _validate_head_index(args.show_head, attention.num_heads)
        input_embedding.eval()
        attention.eval()
    except (
        AttentionError,
        ConfigError,
        EmbeddingError,
        FileNotFoundError,
        IsADirectoryError,
        OSError,
        PositionalEncodingError,
        RuntimeError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Multi-head attention inspection completed successfully.")
    _print_summary(
        attention=attention,
        actual_vocab_size=token_metadata.vocab_size,
        configured_vocab_size=app_config.model.vocab_size,
        trainable_parameters=count_trainable_parameters(attention),
        device=device,
    )

    if args.show_batch or args.show_matrix:
        try:
            dataset_path = _resolve_path(args.dataset, app_config.data.train_dataset_file)
            _print_batch_forward(
                dataset_path=dataset_path,
                input_embedding=input_embedding,
                attention=attention,
                batch_size=app_config.dataset.batch_size,
                device=device,
                show_matrix=args.show_matrix,
                show_head=args.show_head,
            )
        except (
            AttentionError,
            DatasetPreparationError,
            EmbeddingError,
            FileNotFoundError,
            IsADirectoryError,
            PositionalEncodingError,
        ) as exc:
            _report_error(exc, debug=args.debug)
            return 1

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GenPy LLM multi-head attention.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--dataset", type=str, default=None, help="Prepared dataset split file.")
    parser.add_argument("--num-heads", type=_positive_int, default=None)
    parser.add_argument("--show-batch", action="store_true")
    parser.add_argument("--show-head", type=int, default=0)
    parser.add_argument("--show-matrix", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _print_summary(
    attention: MultiHeadCausalSelfAttention,
    actual_vocab_size: int,
    configured_vocab_size: int,
    trainable_parameters: int,
    device: torch.device,
) -> None:
    print("GenPy LLM Multi-Head Causal Self-Attention")
    print("==========================================")
    print(f"Actual vocabulary size: {actual_vocab_size}")
    print(f"Configured vocabulary size: {configured_vocab_size}")
    print(f"Embedding dimension: {attention.embedding_dim}")
    print(f"Head count: {attention.num_heads}")
    print(f"Head dimension: {attention.head_dim}")
    print(f"Maximum sequence length: {attention.max_sequence_length}")
    print(f"Attention dropout: {attention.attention_dropout.p}")
    print(f"Attention parameter count: {attention.parameter_count}")
    print(f"Trainable attention parameters: {trainable_parameters}")
    print(f"Device: {device}")
    print()
    print("These are random attention weights before training; they are not meaningful yet.")


def _print_batch_forward(
    dataset_path: Path,
    input_embedding: GPTInputEmbedding,
    attention: MultiHeadCausalSelfAttention,
    batch_size: int,
    device: torch.device,
    show_matrix: bool,
    show_head: int,
) -> None:
    dataset = load_dataset_split(dataset_path)
    print()
    print("Prepared batch multi-head attention pass")
    print("========================================")
    if len(dataset) == 0:
        print("Dataset split is empty.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    token_ids = batch["input_ids"].to(device)
    padding_mask = batch.get("attention_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)

    with torch.no_grad():
        hidden_states = input_embedding(token_ids)
        output, weights = attention(
            hidden_states,
            padding_mask=padding_mask,
            return_attention=True,
        )

    print(f"Dataset file: {dataset_path}")
    print(f"Input token IDs shape: {tuple(token_ids.shape)}")
    print(f"Input embedding shape: {tuple(hidden_states.shape)}")
    print(f"Attention output shape: {tuple(output.shape)}")
    print(f"Attention weights shape: {tuple(weights.shape)}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    print(f"Future-token probabilities zero: {_future_probabilities_are_zero(weights)}")

    if show_matrix:
        _print_attention_matrix(weights[0, show_head], show_head)


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
        raise AttentionError(
            f"show_head must be between 0 and {num_heads - 1}. Received {head_index}."
        )


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


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
        logger.exception("Multi-head attention inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
