"""Run GenPy LLM text preprocessing from the command line."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import ConfigError, load_config
from genpy_llm.logging_utils import setup_logging
from genpy_llm.preprocessing import TextPreprocessor


def main() -> int:
    """Parse arguments and run preprocessing."""

    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        config_path = _resolve_optional_path(args.config)
        config = load_config(config_path)
        logger = setup_logging(
            log_dir=config.paths.logs_dir,
            log_file=config.logging.log_file,
            level=config.logging.level,
        )

        input_path = _resolve_path(args.input, config.data.input_file)
        output_path = _resolve_path(args.output, config.data.output_file)

        preprocessor = TextPreprocessor(config.preprocessing)
        stats = preprocessor.process_file(
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
    ) as exc:
        _report_error(exc)
        return 1

    logger.info("Text preprocessing completed successfully.")
    print(stats.summary())
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw text for GenPy LLM.")
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML config file.")
    parser.add_argument("--input", type=str, default=None, help="Raw text input file.")
    parser.add_argument("--output", type=str, default=None, help="Cleaned text output file.")
    return parser.parse_args()


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


def _report_error(exc: Exception) -> None:
    logger = logging.getLogger("genpy_llm")
    if logger.isEnabledFor(logging.DEBUG):
        logger.exception("Preprocessing failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
