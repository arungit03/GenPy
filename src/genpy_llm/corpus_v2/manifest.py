"""Manifest records and atomic artifact helpers for Corpus V2."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


@dataclass(frozen=True)
class SourceSpec:
    """One approved local source root."""

    source_id: str
    source_type: str
    path: Path
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    license: str | None
    approval: str | None


@dataclass(frozen=True)
class CollectedDocument:
    """Raw collected local file content."""

    source: SourceSpec
    path: Path
    relative_path: str
    content: bytes
    content_type: str


@dataclass(frozen=True)
class CleanDocument:
    """Cleaned UTF-8 document ready for validation."""

    source: SourceSpec
    path: Path
    relative_path: str
    text: str
    content_type: str
    sha256: str
    byte_count: int
    line_count: int


@dataclass(frozen=True)
class TokenizedDocument:
    """Validated, deduplicated, and tokenized document."""

    stored_path: str
    source_id: str
    source_type: str
    source_path: str
    content_type: str
    language: str
    sha256: str
    normalized_sha256: str
    token_count: int
    byte_count: int
    line_count: int
    license: str | None
    approval: str | None
    token_ids: list[int]
    quality: dict[str, Any]

    def manifest_record(self) -> dict[str, Any]:
        """Return a JSON-serializable record without token IDs."""

        payload = asdict(self)
        payload.pop("token_ids", None)
        payload["collection_timestamp"] = timestamp()
        return payload


def content_hash(content: bytes) -> str:
    """Return SHA-256 for bytes."""

    return hashlib.sha256(content).hexdigest()


def text_hash(text: str) -> str:
    """Return SHA-256 for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_json(payload: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                json.dump(record, file, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    """Read a JSON artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CleanDocument",
    "CollectedDocument",
    "SourceSpec",
    "TokenizedDocument",
    "content_hash",
    "fingerprint_json",
    "read_json",
    "text_hash",
    "timestamp",
    "write_json",
    "write_jsonl",
]
