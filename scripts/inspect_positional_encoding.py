"""Inspect GenPy LLM positional encoding without building attention or a model."""

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
from genpy_llm.embeddings import EmbeddingError, TokenEmbedding, create_token_embedding
from genpy_llm.logging_utils import setup_logging
from genpy_llm.positional_encoding import (
    GPTInputEmbedding,
    PositionalEncoding,
    PositionalEncodingError,
)
from genpy_llm.utils import set_seed
from genpy_llm.vocabulary import Vocabulary, VocabularyError


def main() -> int:
    """Parse arguments and inspect positional encodings."""

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
        set_seed(app_config.training.seed)
        device = select_device(args.device or app_config.training.device)

        vocabulary_path = _resolve_path(args.vocabulary, app_config.data.vocabulary_file)
        vocabulary = Vocabulary.load(vocabulary_path, encoding=app_config.data.encoding)
        token_embedding, token_metadata = create_token_embedding(
            vocabulary_path=vocabulary_path,
            embedding_config=app_config.embeddings,
            expected_vocab_size=app_config.model.vocab_size,
            encoding=app_config.data.encoding,
        )
        positional_encoding = PositionalEncoding(
            embedding_dim=app_config.embeddings.embedding_dim,
            max_sequence_length=app_config.positional_encoding.max_sequence_length,
            encoding_type=args.type or app_config.positional_encoding.type,
            dropout=app_config.positional_encoding.dropout,
            initialization_std=app_config.positional_encoding.initialization_std,
        )
        input_embedding = GPTInputEmbedding(token_embedding, positional_encoding).to(device)
        input_embedding.eval()
    except (
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

    logger.info("Positional encoding inspection completed successfully.")
    _print_summary(
        positional_encoding=positional_encoding,
        token_vocab_size=token_metadata.vocab_size,
        configured_vocab_size=app_config.model.vocab_size,
        device=device,
    )
    _print_position_comparison(input_embedding, vocabulary, device)

    if args.show_batch:
        try:
            dataset_path = _resolve_path(args.dataset, app_config.data.train_dataset_file)
            _print_batch_forward(
                input_embedding=input_embedding,
                token_embedding=token_embedding,
                dataset_path=dataset_path,
                batch_size=app_config.dataset.batch_size,
                device=device,
            )
        except (
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
    parser = argparse.ArgumentParser(description="Inspect GenPy LLM positional encodings.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--vocabulary", type=str, default=None, help="Vocabulary JSON file.")
    parser.add_argument("--dataset", type=str, default=None, help="Prepared dataset split file.")
    parser.add_argument("--type", choices=["learned", "sinusoidal"], default=None)
    parser.add_argument("--show-batch", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _print_summary(
    positional_encoding: PositionalEncoding,
    token_vocab_size: int,
    configured_vocab_size: int,
    device: torch.device,
) -> None:
    print("GenPy LLM Positional Encoding")
    print("=============================")
    print(f"Encoding type: {positional_encoding.encoding_type}")
    print(f"Actual vocabulary size: {token_vocab_size}")
    print(f"Configured vocabulary size: {configured_vocab_size}")
    print(f"Embedding dimension: {positional_encoding.embedding_dim}")
    print(f"Maximum sequence length: {positional_encoding.max_sequence_length}")
    print(f"Dropout: {positional_encoding.dropout.p}")
    print(f"Trainable positional parameters: {positional_encoding.trainable_parameter_count}")
    print(f"Device: {device}")
    print()
    print("GPT input embeddings are token embeddings plus positional encodings.")


def _print_position_comparison(
    input_embedding: GPTInputEmbedding,
    vocabulary: Vocabulary,
    device: torch.device,
) -> None:
    token = (
        "GenPy" if "GenPy" in vocabulary.token_to_id else vocabulary.id_token(vocabulary.unknown_id)
    )
    token_id = vocabulary.token_id(token)
    token_ids = torch.tensor([[token_id, token_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        output = input_embedding(token_ids)
    first = output[0, 0]
    second = output[0, 1]
    max_difference = torch.max(torch.abs(first - second)).item()
    print()
    print("Same token at different positions")
    print("=================================")
    print(f"Token: {token}")
    print(f"Token ID: {token_id}")
    print(f"Position 0 first value: {first[0].item():.8f}")
    print(f"Position 1 first value: {second[0].item():.8f}")
    print(f"Maximum absolute difference: {max_difference:.8f}")


def _print_batch_forward(
    input_embedding: GPTInputEmbedding,
    token_embedding: TokenEmbedding,
    dataset_path: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    dataset = load_dataset_split(dataset_path)
    print()
    print("Prepared batch positional pass")
    print("==============================")
    if len(dataset) == 0:
        print("Dataset split is empty.")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    token_ids = batch["input_ids"].to(device)
    with torch.no_grad():
        token_vectors = token_embedding(token_ids)
        output = input_embedding(token_ids)
    print(f"Dataset file: {dataset_path}")
    print(f"Input token IDs shape: {tuple(token_ids.shape)}")
    print(f"Token embedding shape: {tuple(token_vectors.shape)}")
    print(f"Positional output shape: {tuple(output.shape)}")
    print(f"Input dtype: {token_ids.dtype}")
    print(f"Output dtype: {output.dtype}")
    print(f"Input device: {token_ids.device}")
    print(f"Output device: {output.device}")


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
        logger.exception("Positional encoding inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
