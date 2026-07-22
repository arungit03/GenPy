"""Build the production GenPy Phase 5 ByteLevel BPE tokenizer."""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.code_tokenizer import (
    DEFAULT_TOKENIZER_CONFIG_PATH,
    TokenizerPipelineConfig,
    build_tokenizer_artifacts,
    load_tokenizer_pipeline_config,
    verify_code_tokenizer,
)

LOGGER = logging.getLogger("genpy_llm.train_code_tokenizer")


def main() -> int:
    """Run the configured tokenizer build."""

    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        config = load_tokenizer_pipeline_config(
            _resolve(args.config),
            project_root=PROJECT_ROOT,
        )
        config = _apply_overrides(config, args)
        existing = _existing_artifacts(config)
        if existing and not args.force:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"Tokenizer artifacts already exist ({names}). Use --force.")
        LOGGER.info("Configuration: %s", config.config_path)
        LOGGER.info("Corpus: %s", ", ".join(_display(path) for path in config.corpus_paths))
        LOGGER.info(
            "Settings: vocab_size=%d min_frequency=%d normalization=%s seed=%d",
            config.vocab_size,
            config.min_frequency,
            config.normalization,
            config.seed,
        )
        result = build_tokenizer_artifacts(config)
        tokenizer = verify_code_tokenizer(config.tokenizer_path)
    except Exception as exc:  # noqa: BLE001
        _report_error(exc, debug=args.debug)
        return 1

    print("GenPy Phase 5 tokenizer")
    print("=======================")
    print(f"Tokenizer: {_display(config.tokenizer_path)}")
    print(f"Artifacts: {_display(config.output_directory)}")
    print(f"Vocabulary size: {result.actual_vocab_size}")
    print(f"Minimum frequency: {result.minimum_frequency}")
    print(f"Normalization: {result.normalization}")
    print(f"Training corpus files: {len(result.training_corpus)}")
    print(f"Training bytes: {result.actual_sample_bytes}")
    print(f"Special token IDs: {result.special_token_ids}")
    print(f"Verification vocabulary size: {tokenizer.vocab_size}")
    print("✓ Tokenizer built successfully")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the YAML-configured GenPy ByteLevel BPE tokenizer."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TOKENIZER_CONFIG_PATH)
    parser.add_argument(
        "--train-pattern",
        action="append",
        default=None,
        help="Override corpus files with a glob; repeat for multiple globs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=None)
    parser.add_argument("--max-training-bytes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _apply_overrides(
    config: TokenizerPipelineConfig,
    args: argparse.Namespace,
) -> TokenizerPipelineConfig:
    updates: dict[str, object] = {}
    if args.train_pattern:
        corpus: list[Path] = []
        for pattern in args.train_pattern:
            search = pattern if Path(pattern).is_absolute() else str(PROJECT_ROOT / pattern)
            matched = sorted(glob.glob(search, recursive=True))
            corpus.extend(Path(item).resolve() for item in matched)
        updates["corpus_paths"] = tuple(dict.fromkeys(corpus))
    if args.output is not None:
        output = _resolve(args.output)
        updates.update(
            output_directory=output.parent,
            tokenizer_filename=output.name,
            legacy_tokenizer_filename=None,
        )
    for argument, field_name in (
        (args.vocab_size, "vocab_size"),
        (args.min_frequency, "min_frequency"),
        (args.max_training_bytes, "max_training_bytes"),
        (args.seed, "seed"),
    ):
        if argument is not None:
            updates[field_name] = argument
    if args.no_progress:
        updates["show_progress"] = False
    return replace(config, **updates)


def _existing_artifacts(config: TokenizerPipelineConfig) -> list[Path]:
    filenames = (
        config.tokenizer_filename,
        config.vocab_filename,
        config.merges_filename,
        config.tokenizer_config_filename,
        config.special_tokens_filename,
        config.metadata_filename,
        config.statistics_filename,
    )
    paths = (config.output_directory / name for name in filenames)
    return [path for path in paths if path.exists()]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _report_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        LOGGER.exception("Tokenizer training failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
