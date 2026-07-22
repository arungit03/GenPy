"""Inspect GenPy LLM token embeddings without building a full model."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import ConfigError, EmbeddingConfig, load_config
from genpy_llm.dataset import DatasetPreparationError, load_dataset_split
from genpy_llm.device import select_device
from genpy_llm.embeddings import (
    EmbeddingError,
    EmbeddingMetadata,
    EmbeddingWeightStats,
    TokenEmbedding,
    TokenEmbeddingRecord,
    build_embedding_metadata,
    calculate_embedding_statistics,
    cosine_similarity_between_tokens,
    create_token_embedding,
    inspect_token_embeddings,
    load_embedding_checkpoint,
    save_embedding_checkpoint,
)
from genpy_llm.logging_utils import setup_logging
from genpy_llm.utils import set_seed
from genpy_llm.vocabulary import Vocabulary, VocabularyError


def main() -> int:
    """Parse arguments and inspect token embeddings."""

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

        embedding_config = _override_embedding_config(app_config.embeddings, args)
        vocabulary_path = _resolve_path(args.vocabulary, app_config.data.vocabulary_file)
        vocabulary = Vocabulary.load(vocabulary_path, encoding=app_config.data.encoding)

        if args.load_checkpoint is not None:
            checkpoint_path = _resolve_against_project_root(Path(args.load_checkpoint))
            embedding, metadata = load_embedding_checkpoint(checkpoint_path, map_location=device)
        else:
            embedding, metadata = create_token_embedding(
                vocabulary_path=vocabulary_path,
                embedding_config=embedding_config,
                expected_vocab_size=app_config.model.vocab_size,
                encoding=app_config.data.encoding,
            )
            embedding.to(device)

        metadata = build_embedding_metadata(embedding)
        stats = calculate_embedding_statistics(embedding)
    except (
        ConfigError,
        DatasetPreparationError,
        EmbeddingError,
        FileNotFoundError,
        IsADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Embedding inspection completed successfully.")
    _print_summary(
        metadata=metadata,
        configured_vocab_size=app_config.model.vocab_size,
        device=device,
        stats=stats,
        checkpoint_loaded=args.load_checkpoint is not None,
    )

    if args.show_batch:
        try:
            dataset_path = _resolve_path(args.dataset, app_config.data.train_dataset_file)
            _print_batch_forward(embedding, dataset_path, app_config.dataset.batch_size, device)
        except (
            DatasetPreparationError,
            EmbeddingError,
            FileNotFoundError,
            IsADirectoryError,
        ) as exc:
            _report_error(exc, debug=args.debug)
            return 1

    if args.tokens:
        try:
            records = inspect_token_embeddings(
                embedding=embedding,
                vocabulary=vocabulary,
                tokens=args.tokens,
                max_dimensions=args.max_dimensions,
            )
            _print_token_records(records)
        except (EmbeddingError, VocabularyError) as exc:
            _report_error(exc, debug=args.debug)
            return 1

    if args.compare is not None:
        try:
            similarity = cosine_similarity_between_tokens(
                embedding=embedding,
                vocabulary=vocabulary,
                first_token=args.compare[0],
                second_token=args.compare[1],
            )
            print()
            print("Token cosine similarity")
            print("=======================")
            print(f"{args.compare[0]!r} vs {args.compare[1]!r}: {similarity:.8f}")
            print(
                "The embedding weights are randomly initialized; this similarity is not "
                "semantically meaningful until training."
            )
        except (EmbeddingError, VocabularyError) as exc:
            _report_error(exc, debug=args.debug)
            return 1

    if args.save_checkpoint is not None:
        try:
            save_path = _resolve_against_project_root(Path(args.save_checkpoint))
            save_embedding_checkpoint(embedding, save_path, metadata)
            print()
            print(f"Saved embedding checkpoint: {save_path}")
        except (EmbeddingError, IsADirectoryError, OSError, RuntimeError) as exc:
            _report_error(exc, debug=args.debug)
            return 1

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GenPy LLM token embeddings.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--vocabulary", type=str, default=None, help="Vocabulary JSON file.")
    parser.add_argument("--dataset", type=str, default=None, help="Prepared dataset split file.")
    parser.add_argument("--embedding-dim", type=_positive_int, default=None)
    parser.add_argument(
        "--initialization",
        choices=["normal", "uniform", "xavier_uniform"],
        default=None,
    )
    parser.add_argument("--initialization-std", type=_positive_float, default=None)
    parser.add_argument("--scale-embeddings", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("TOKEN_A", "TOKEN_B"), default=None)
    parser.add_argument("--max-dimensions", type=_positive_int, default=8)
    parser.add_argument("--show-batch", action="store_true")
    parser.add_argument("--save-checkpoint", type=str, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _override_embedding_config(
    config: EmbeddingConfig,
    args: argparse.Namespace,
) -> EmbeddingConfig:
    updates = {}
    if args.embedding_dim is not None:
        updates["embedding_dim"] = args.embedding_dim
    if args.initialization is not None:
        updates["initialization"] = args.initialization
    if args.initialization_std is not None:
        updates["initialization_std"] = args.initialization_std
    if args.scale_embeddings:
        updates["scale_embeddings"] = True
    if args.freeze_embeddings:
        updates["freeze_embeddings"] = True
    return replace(config, **updates)


def _print_summary(
    metadata: EmbeddingMetadata,
    configured_vocab_size: int,
    device: torch.device,
    stats: EmbeddingWeightStats,
    checkpoint_loaded: bool,
) -> None:
    print("GenPy LLM Token Embeddings")
    print("==========================")
    print(metadata.summary())
    print()
    print(f"Configured model vocab size: {configured_vocab_size}")
    print(f"Embedding matrix shape: ({metadata.vocab_size}, {metadata.embedding_dim})")
    print(f"Device: {device}")
    print(f"Checkpoint loaded: {checkpoint_loaded}")
    print()
    print(stats.summary())
    print()
    print("Note: random initial embeddings have no learned semantic meaning yet.")


def _print_batch_forward(
    embedding: TokenEmbedding,
    dataset_path: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    dataset = load_dataset_split(dataset_path)
    print()
    print("Prepared batch embedding pass")
    print("=============================")
    if len(dataset) == 0:
        print("Dataset split is empty.")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    with torch.no_grad():
        vectors = embedding(input_ids)
    print(f"Dataset file: {dataset_path}")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"embedding output shape: {tuple(vectors.shape)}")
    print(f"dtype: {vectors.dtype}")
    print(f"device: {vectors.device}")


def _print_token_records(records: list[TokenEmbeddingRecord]) -> None:
    print()
    print("Token embedding inspection")
    print("==========================")
    for record in records:
        label = record.token
        if record.mapped_to_unknown:
            label = f"{record.requested_token} -> {record.token}"
        preview = ", ".join(f"{value:.6f}" for value in record.vector)
        print(f"{label}\tid={record.token_id}\tl2={record.l2_norm:.6f}\t[{preview}]")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
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
        logger.exception("Embedding inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
