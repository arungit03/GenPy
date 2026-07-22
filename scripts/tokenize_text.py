"""Run GenPy LLM tokenization from the command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import ConfigError, load_config
from genpy_llm.logging_utils import setup_logging
from genpy_llm.tokenization import TextTokenizer


def main() -> int:
    """Parse arguments and run tokenization."""

    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        config_path = _resolve_optional_path(args.config)
        config = load_config(config_path)
        logger = setup_logging(
            log_dir=config.paths.logs_dir,
            log_file=config.logging.log_file,
            level="DEBUG" if args.debug else config.logging.level,
        )

        tokenization_config = config.tokenization
        if args.method is not None:
            tokenization_config = replace(tokenization_config, method=args.method)

        input_path = _resolve_path(args.input, config.data.output_file)
        output_path = _resolve_path(args.output, config.data.tokenized_file)

        tokenizer = TextTokenizer(tokenization_config)
        stats = tokenizer.process_file(
            input_path=input_path,
            output_path=output_path,
            encoding=config.data.encoding,
        )
    except (
        ConfigError,
        FileNotFoundError,
        IsADirectoryError,
        LookupError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Text tokenization completed successfully.")
    print(stats.summary())
    if args.show_sample > 0:
        print()
        _print_sample(output_path, args.show_sample, encoding=config.data.encoding)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize cleaned text for GenPy LLM.")
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML config file.")
    parser.add_argument("--input", type=str, default=None, help="Cleaned text input file.")
    parser.add_argument("--output", type=str, default=None, help="Tokenized JSONL output file.")
    parser.add_argument("--method", choices=["word", "character"], default=None)
    parser.add_argument("--show-sample", type=_non_negative_int, default=0)
    parser.add_argument("--debug", action="store_true", help="Show detailed error tracebacks.")
    return parser.parse_args()


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--show-sample must be an integer.") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("--show-sample must be non-negative.")
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


def _print_sample(output_path: Path, count: int, encoding: str) -> None:
    print(f"First {count} tokenized sequence(s)")
    print("============================")
    with output_path.open("r", encoding=encoding) as output_file:
        for index, line in enumerate(output_file):
            if index >= count:
                break
            record = json.loads(line)
            tokens = record["tokens"]
            display_tokens = tokens[:80]
            suffix = " ..." if len(tokens) > len(display_tokens) else ""
            print(f"sequence_id: {record['sequence_id']}")
            print(f"token_count: {record['token_count']}")
            print(f"tokens: {display_tokens}{suffix}")


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Tokenization failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
