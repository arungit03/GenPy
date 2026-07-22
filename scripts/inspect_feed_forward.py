"""Inspect the GenPy LLM feed-forward network without building a transformer block."""

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
from genpy_llm.feed_forward import (
    FeedForwardError,
    FeedForwardNetwork,
    resolve_feed_forward_hidden_dim,
)
from genpy_llm.logging_utils import setup_logging
from genpy_llm.positional_encoding import (
    GPTInputEmbedding,
    PositionalEncoding,
    PositionalEncodingError,
)
from genpy_llm.utils import count_trainable_parameters, set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Parse arguments and inspect the position-wise FFN."""

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

        hidden_multiplier = (
            args.hidden_multiplier
            if args.hidden_multiplier is not None
            else app_config.feed_forward.hidden_multiplier
        )
        hidden_dim = resolve_feed_forward_hidden_dim(
            embedding_dim=app_config.embeddings.embedding_dim,
            hidden_multiplier=hidden_multiplier,
            hidden_dim=args.hidden_dim
            if args.hidden_dim is not None
            else app_config.feed_forward.hidden_dim,
        )
        activation = args.activation or app_config.feed_forward.activation
        dropout = args.dropout if args.dropout is not None else app_config.feed_forward.dropout

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
            num_heads=app_config.model.num_heads,
            max_sequence_length=app_config.positional_encoding.max_sequence_length,
            dropout=app_config.attention.dropout,
            use_bias=app_config.attention.use_bias,
        ).to(device)
        ffn = FeedForwardNetwork(
            embedding_dim=app_config.embeddings.embedding_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
            use_bias=app_config.feed_forward.use_bias,
            initialization_std=app_config.feed_forward.initialization_std,
        ).to(device)
        input_embedding.eval()
        attention.eval()
        ffn.eval()
    except (
        AttentionError,
        ConfigError,
        EmbeddingError,
        FeedForwardError,
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

    logger.info("Feed-forward inspection completed successfully.")
    _print_summary(ffn, token_metadata.vocab_size, app_config.model.vocab_size, device)

    if args.show_batch:
        try:
            dataset_path = _resolve_path(args.dataset, app_config.data.train_dataset_file)
            _print_batch_forward(
                dataset_path=dataset_path,
                input_embedding=input_embedding,
                attention=attention,
                ffn=ffn,
                batch_size=app_config.dataset.batch_size,
                device=device,
                use_attention_output=args.use_attention_output,
            )
        except (
            AttentionError,
            DatasetPreparationError,
            EmbeddingError,
            FeedForwardError,
            FileNotFoundError,
            IsADirectoryError,
            PositionalEncodingError,
        ) as exc:
            _report_error(exc, debug=args.debug)
            return 1

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GenPy LLM feed-forward network.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--dataset", type=str, default=None, help="Prepared dataset split file.")
    parser.add_argument("--hidden-dim", type=_positive_int, default=None)
    parser.add_argument("--hidden-multiplier", type=_positive_int, default=None)
    parser.add_argument("--activation", choices=["gelu", "relu", "silu"], default=None)
    parser.add_argument("--dropout", type=_dropout, default=None)
    parser.add_argument("--show-batch", action="store_true")
    parser.add_argument("--use-attention-output", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _print_summary(
    ffn: FeedForwardNetwork,
    actual_vocab_size: int,
    configured_vocab_size: int,
    device: torch.device,
) -> None:
    print("GenPy LLM Feed-Forward Network")
    print("==============================")
    print(f"Actual vocabulary size: {actual_vocab_size}")
    print(f"Configured vocabulary size: {configured_vocab_size}")
    print(ffn.metadata().summary())
    print(f"Trainable parameters via utility: {count_trainable_parameters(ffn)}")
    print(f"Device: {device}")
    print()
    print("FFN weights are randomly initialized before training.")


def _print_batch_forward(
    dataset_path: Path,
    input_embedding: GPTInputEmbedding,
    attention: MultiHeadCausalSelfAttention,
    ffn: FeedForwardNetwork,
    batch_size: int,
    device: torch.device,
    use_attention_output: bool,
) -> None:
    dataset = load_dataset_split(dataset_path)
    print()
    print("Prepared batch feed-forward pass")
    print("================================")
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
        if use_attention_output:
            hidden_states = attention(hidden_states, padding_mask=padding_mask)
        hidden_projection = _hidden_projection(ffn, hidden_states)
        output = ffn(hidden_states)

    print(f"Dataset file: {dataset_path}")
    print(f"Using attention output: {use_attention_output}")
    print(f"Input shape: {tuple(hidden_states.shape)}")
    print(f"Hidden projection shape: {tuple(hidden_projection.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")


def _hidden_projection(ffn: FeedForwardNetwork, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden = ffn.input_projection(hidden_states)
    return ffn.activation(hidden)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _dropout(value: str) -> float:
    number = float(value)
    if not 0.0 <= number < 1.0:
        raise argparse.ArgumentTypeError("dropout must be at least 0.0 and less than 1.0.")
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
        logger.exception("Feed-forward inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
