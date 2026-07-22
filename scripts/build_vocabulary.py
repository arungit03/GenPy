"""Build a deterministic vocabulary from tokenized JSONL."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import ConfigError, VocabularyConfig, load_config
from genpy_llm.logging_utils import setup_logging
from genpy_llm.vocabulary import (
    Vocabulary,
    VocabularyBuildStats,
    VocabularyError,
    encode_jsonl_file,
    save_build_metadata,
)


def main() -> int:
    """Parse arguments and build the vocabulary."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        config_path = _resolve_optional_path(args.config)
        app_config = load_config(config_path)
        logger = setup_logging(
            log_dir=app_config.paths.logs_dir,
            log_file=app_config.logging.log_file,
            level="DEBUG" if args.debug else app_config.logging.level,
        )

        vocabulary_config = _override_vocabulary_config(app_config.vocabulary, args)
        input_path = _resolve_path(args.input, app_config.data.tokenized_file)
        vocabulary_path = _resolve_path(args.output, app_config.data.vocabulary_file)
        metadata_path = _resolve_path(
            args.metadata_output,
            app_config.data.vocabulary_metadata_file,
        )
        encoded_path = _resolve_path(args.encode_output, app_config.data.encoded_file)

        _validate_outputs(
            input_path=input_path,
            vocabulary_path=vocabulary_path,
            metadata_path=metadata_path,
            encoded_path=None if args.no_encode else encoded_path,
            force=args.force,
        )

        vocabulary, stats = Vocabulary.build_from_jsonl(
            input_path=input_path,
            config=vocabulary_config,
            encoding=app_config.data.encoding,
        )
        stats = replace(stats, vocabulary_file=vocabulary_path)

        vocabulary.save(vocabulary_path, encoding=app_config.data.encoding)
        final_encoded_path = None
        if not args.no_encode:
            encode_jsonl_file(
                input_path=input_path,
                output_path=encoded_path,
                vocabulary=vocabulary,
                encoding=app_config.data.encoding,
            )
            final_encoded_path = encoded_path

        save_build_metadata(
            metadata_path=metadata_path,
            stats=stats,
            vocabulary_path=vocabulary_path,
            encoded_path=final_encoded_path,
            project_root=app_config.project_root,
            config=vocabulary_config,
            encoding=app_config.data.encoding,
        )
    except (
        ConfigError,
        FileExistsError,
        FileNotFoundError,
        IsADirectoryError,
        LookupError,
        OSError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Vocabulary build completed successfully.")
    _print_summary(
        stats=stats,
        input_path=input_path,
        vocabulary_path=vocabulary_path,
        metadata_path=metadata_path,
        encoded_path=final_encoded_path,
    )
    if args.show_special_tokens:
        _print_special_tokens(vocabulary)
    if args.show_top > 0:
        _print_top_tokens(vocabulary, args.show_top)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GenPy LLM vocabulary.")
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML config file.")
    parser.add_argument("--input", type=str, default=None, help="Tokenized JSONL input file.")
    parser.add_argument("--output", type=str, default=None, help="Vocabulary JSON output file.")
    parser.add_argument(
        "--metadata-output",
        type=str,
        default=None,
        help="Metadata JSON output file.",
    )
    parser.add_argument(
        "--encode-output",
        type=str,
        default=None,
        help="Encoded JSONL output file.",
    )
    parser.add_argument("--min-frequency", type=_positive_int, default=None)
    parser.add_argument("--max-size", type=_positive_int, default=None)
    parser.add_argument("--show-top", type=_non_negative_int, default=0)
    parser.add_argument("--show-special-tokens", action="store_true")
    parser.add_argument(
        "--no-encode",
        action="store_true",
        help="Only write vocabulary and metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files safely.",
    )
    parser.add_argument("--debug", action="store_true", help="Show detailed error tracebacks.")
    return parser.parse_args()


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer.") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return number


def _override_vocabulary_config(
    config: VocabularyConfig,
    args: argparse.Namespace,
) -> VocabularyConfig:
    if args.min_frequency is not None:
        config = replace(config, min_frequency=args.min_frequency)
    if args.max_size is not None:
        config = replace(config, max_size=args.max_size)
    if config.max_size is not None and config.max_size < len(config.special_token_order):
        raise ConfigError("max_size must be large enough to include all special tokens.")
    return config


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


def _validate_outputs(
    input_path: Path,
    vocabulary_path: Path,
    metadata_path: Path,
    encoded_path: Path | None,
    force: bool,
) -> None:
    del force
    output_paths = [vocabulary_path, metadata_path]
    if encoded_path is not None:
        output_paths.append(encoded_path)

    for output_path in output_paths:
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Input and output paths must be different files.")
        if output_path.exists() and output_path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {output_path}")


def _print_summary(
    stats: VocabularyBuildStats,
    input_path: Path,
    vocabulary_path: Path,
    metadata_path: Path,
    encoded_path: Path | None,
) -> None:
    print("GenPy LLM Vocabulary Build Complete")
    print("===================================")
    print(f"Input file:                  {input_path}")
    print(f"Vocabulary file:             {vocabulary_path}")
    print(f"Metadata file:               {metadata_path}")
    print(f"Encoded output:              {encoded_path if encoded_path else 'disabled'}")
    print()
    print(f"Sequences processed:         {stats.processed_sequences}")
    print(f"Total tokens observed:       {stats.total_tokens}")
    print(f"Unique tokens observed:      {stats.unique_tokens_observed}")
    print(f"Final vocabulary size:       {stats.vocabulary_size}")
    print(f"Special tokens:              {stats.special_token_count}")
    print(f"Normal tokens:               {stats.normal_token_count}")
    print(f"Below minimum frequency:     {stats.excluded_below_min_frequency}")
    print(f"Excluded by maximum size:    {stats.excluded_by_max_size}")


def _print_special_tokens(vocabulary: Vocabulary) -> None:
    print()
    print("Special tokens")
    print("==============")
    for token in vocabulary.config.special_token_order:
        print(f"{token}\t{vocabulary.token_id(token)}")


def _print_top_tokens(vocabulary: Vocabulary, count: int) -> None:
    if vocabulary.frequencies is None:
        print()
        print("Top tokens unavailable because frequencies were not saved.")
        return

    special_tokens = vocabulary.special_tokens
    ranked_tokens = sorted(
        (token for token in vocabulary.token_to_id if token not in special_tokens),
        key=lambda token: (-vocabulary.frequencies.get(token, 0), token),
    )[:count]
    print()
    print("Top normal tokens")
    print("=================")
    print("Rank\tToken\tFrequency\tID")
    for rank, token in enumerate(ranked_tokens, start=1):
        print(
            f"{rank}\t{token}\t{vocabulary.frequencies.get(token, 0)}\t{vocabulary.token_id(token)}"
        )


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Vocabulary build failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
