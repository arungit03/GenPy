"""Final corpus validation report helpers."""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from genpy_llm.code_filtering import CodeFilterSettings, filter_code_record


@dataclass(frozen=True)
class FinalValidationConfig:
    """Validation settings for Phase 5.5C."""

    minimum_file_bytes: int
    maximum_file_bytes: int
    cleaner: CodeFilterSettings
    require_python_syntax: bool = True
    reject_generated: bool = True
    reject_vendor: bool = True
    allow_technical_text: bool = False
    technical_text_extensions: tuple[str, ...] = (".md", ".rst", ".txt")


@dataclass(frozen=True)
class ValidatedCorpusRecord:
    """A manifest record whose file was revalidated from disk."""

    provenance: dict[str, Any]
    path: Path
    text: str
    byte_size: int
    line_count: int


class ValidationReporter:
    """Accumulate final validation outcomes."""

    def __init__(self) -> None:
        self.accepted = 0
        self.rejected = 0
        self.reasons: Counter[str] = Counter()
        self.records: list[dict[str, Any]] = []

    def accept(self) -> None:
        self.accepted += 1

    def reject(self, record: Mapping[str, Any], reason: str) -> None:
        self.rejected += 1
        self.reasons[reason] += 1
        self.records.append(
            {
                "reason": reason,
                "stored_path": record.get("stored_path"),
                "source_path": record.get("source_path"),
                "source": record.get("source"),
            }
        )

    def report(self) -> dict[str, Any]:
        total = self.accepted + self.rejected
        return {
            "accepted_files": self.accepted,
            "rejected_files": self.rejected,
            "total_candidates": total,
            "rejection_rate": round(self.rejected / total, 6) if total else 0.0,
            "rejection_reasons": dict(sorted(self.reasons.items())),
            "rejections": self.records,
        }


def validate_manifest_record(
    record: Mapping[str, Any],
    *,
    corpus_root: Path,
    config: FinalValidationConfig,
) -> tuple[ValidatedCorpusRecord | None, str | None]:
    """Re-read and validate one provenance manifest record."""

    stored_path = record.get("stored_path")
    if not isinstance(stored_path, str) or not stored_path.strip():
        return None, "invalid_metadata"
    source = record.get("source")
    if not isinstance(source, Mapping) or not source.get("type"):
        return None, "invalid_metadata"
    expected_hash = record.get("content_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return None, "invalid_metadata"
    path = (corpus_root / stored_path).resolve()
    try:
        path.relative_to(corpus_root.resolve())
    except ValueError:
        return None, "unsafe_path"
    if not path.is_file():
        return None, "missing_file"
    try:
        content = path.read_bytes()
    except OSError:
        return None, "file_read_error"
    byte_size = len(content)
    if byte_size < config.minimum_file_bytes:
        return None, "too_small"
    if byte_size > config.maximum_file_bytes:
        return None, "too_large"
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash:
        return None, "hash_mismatch"
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, "invalid_utf8"
    if not text.strip():
        return None, "empty_file"
    content_type = _content_type(record, stored_path, config)
    if content_type == "python_code" and config.require_python_syntax:
        try:
            ast.parse(text, filename=stored_path)
        except (SyntaxError, ValueError, TypeError):
            return None, "invalid_python_syntax"
    if content_type == "python_code":
        filter_result = filter_code_record(
            {
                "text": text,
                "path": record.get("source_path") or stored_path,
                "license": record.get("license"),
                "repo_name": source.get("id"),
            },
            settings=config.cleaner,
        )
        if not filter_result.accepted:
            return None, f"cleaner_{filter_result.reason}"
    elif not config.allow_technical_text:
        return None, "technical_text_disabled"
    elif not _technical_text_has_signal(text):
        return None, "low_text_signal"
    return (
        ValidatedCorpusRecord(
            provenance=dict(record),
            path=path,
            text=text,
            byte_size=byte_size,
            line_count=len(text.splitlines()),
        ),
        None,
    )


def _content_type(
    record: Mapping[str, Any],
    stored_path: str,
    config: FinalValidationConfig,
) -> str:
    declared = record.get("content_type")
    if declared == "python_code":
        return "python_code"
    if declared == "technical_text":
        return "technical_text"
    suffix = PurePosixPath(str(record.get("source_path") or stored_path)).suffix.casefold()
    if suffix == ".py":
        return "python_code"
    if suffix in {item.casefold() for item in config.technical_text_extensions}:
        return "technical_text"
    return "unknown"


def _technical_text_has_signal(text: str) -> bool:
    words = [word for word in text.replace("_", " ").split() if any(ch.isalpha() for ch in word)]
    if len(words) < 8:
        return False
    printable = sum(char.isprintable() or char in {"\n", "\t"} for char in text)
    return printable / max(1, len(text)) >= 0.95


__all__ = [
    "FinalValidationConfig",
    "ValidatedCorpusRecord",
    "ValidationReporter",
    "validate_manifest_record",
]
