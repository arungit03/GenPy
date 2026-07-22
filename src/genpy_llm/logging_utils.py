"""Logging setup for GenPy LLM."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class JsonLogFormatter(logging.Formatter):
    """Stable structured formatter shared by corpus ingestion pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def setup_structured_logging(log_path: Path, level: str = "INFO") -> None:
    """Configure root console logging plus a structured JSONL file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = _parse_log_level(level)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    structured = logging.FileHandler(log_path, encoding="utf-8")
    structured.setFormatter(JsonLogFormatter())
    logging.basicConfig(
        level=numeric_level,
        handlers=[console, structured],
        force=True,
    )


def setup_logging(log_dir: Path, log_file: str, level: str = "INFO") -> logging.Logger:
    """Configure console and file logging without duplicate handlers."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file
    numeric_level = _parse_log_level(level)

    logger = logging.getLogger("genpy_llm")
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    _add_handler_once(
        logger=logger,
        handler=logging.StreamHandler(),
        handler_key="console",
        formatter=formatter,
        level=numeric_level,
    )
    _add_handler_once(
        logger=logger,
        handler=logging.FileHandler(log_path, encoding="utf-8"),
        handler_key=str(log_path.resolve()),
        formatter=formatter,
        level=numeric_level,
    )

    return logger


def _parse_log_level(level: str) -> int:
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level '{level}'.")
    return numeric_level


def _add_handler_once(
    logger: logging.Logger,
    handler: logging.Handler,
    handler_key: str,
    formatter: logging.Formatter,
    level: int,
) -> None:
    for existing_handler in logger.handlers:
        if getattr(existing_handler, "_genpy_handler_key", None) == handler_key:
            existing_handler.setLevel(level)
            return

    handler.setFormatter(formatter)
    handler.setLevel(level)
    handler._genpy_handler_key = handler_key
    logger.addHandler(handler)
