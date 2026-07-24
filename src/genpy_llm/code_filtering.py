"""Filtering helpers for Python source-code training records."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


class CodeFilteringError(ValueError):
    """Raised when a code record cannot be filtered safely."""


DEFAULT_ACCEPTED_LICENSES = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "MPL-2.0",
    }
)

PYTHON_MARKERS = (
    "def ",
    "class ",
    "import ",
    "from ",
    "async def ",
    "if __name__",
    "try:",
    "for ",
    "while ",
    "return ",
    "print(",
)

REJECTED_PATH_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "vendor",
    "vendors",
    "dist",
    "build",
    "site-packages",
}


@dataclass(frozen=True)
class CodeFilterSettings:
    """Configurable source-code filtering settings."""

    minimum_file_bytes: int = 200
    maximum_file_bytes: int = 250_000
    accepted_licenses: tuple[str, ...] = tuple(sorted(DEFAULT_ACCEPTED_LICENSES))
    require_known_license: bool = True


@dataclass(frozen=True)
class FilteredCodeRecord:
    """Accepted normalized source record ready for sharding."""

    text: str
    repo_name: str | None
    path: str | None
    license: str | None
    content_hash: str
    byte_count: int
    split: str

    def to_json_record(self) -> dict[str, str]:
        """Return the compact JSONL representation."""

        record = {
            "text": self.text,
            "content_hash": self.content_hash,
        }
        if self.repo_name:
            record["repo_name"] = self.repo_name
        if self.path:
            record["path"] = self.path
        if self.license:
            record["license"] = self.license
        return record


@dataclass(frozen=True)
class CodeFilterResult:
    """Result of applying source-code filters to a raw dataset record."""

    accepted: bool
    reason: str
    record: FilteredCodeRecord | None = None


def normalize_license(value: object) -> str | None:
    """Normalize common license spellings to SPDX-like names."""

    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = normalize_license(item)
            if normalized is not None:
                return normalized
        return None
    text = str(value).strip()
    if not text:
        return None
    folded = re.sub(r"[^a-z0-9]+", "", text.lower())
    aliases = {
        "mit": "MIT",
        "mitlicense": "MIT",
        "apache20": "Apache-2.0",
        "apache2": "Apache-2.0",
        "apachelicense20": "Apache-2.0",
        "bsd2clause": "BSD-2-Clause",
        "bsd2": "BSD-2-Clause",
        "bsd3clause": "BSD-3-Clause",
        "bsd3": "BSD-3-Clause",
        "isc": "ISC",
        "isclicense": "ISC",
        "unlicense": "Unlicense",
        "theunlicense": "Unlicense",
        "mpl20": "MPL-2.0",
        "mozilla public license 20".replace(" ", ""): "MPL-2.0",
    }
    return aliases.get(folded)


def normalize_python_source(text: str) -> str:
    """Normalize line endings and invalid control characters without touching indentation."""

    if not isinstance(text, str):
        raise CodeFilteringError("source text must be a string.")
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\x00")
    return "".join(char for char in text if char in {"\n", "\t"} or ord(char) >= 32)


def content_hash(text: str) -> str:
    """Return deterministic SHA-256 hash for normalized content."""

    normalized = normalize_python_source(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_split(hash_value: str, validation_percent: int | float = 2) -> str:
    """Assign a stable split from a content hash."""

    if not isinstance(hash_value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", hash_value):
        raise CodeFilteringError("hash_value must be a SHA-256 hex digest.")
    if not isinstance(validation_percent, (int, float)) or not 0 <= validation_percent <= 100:
        raise CodeFilteringError("validation_percent must be between 0 and 100.")
    bucket = int(hash_value[:8], 16) % 10_000
    return "validation" if bucket < int(float(validation_percent) * 100) else "train"


def filter_code_record(
    raw_record: dict[str, Any],
    *,
    settings: CodeFilterSettings | None = None,
    validation_percent: int | float = 2,
) -> CodeFilterResult:
    """Return an accepted normalized record or the rejection reason."""

    settings = settings or CodeFilterSettings()
    text = _extract_text(raw_record)
    if text is None:
        return CodeFilterResult(False, "missing_content")
    normalized = normalize_python_source(text)
    if not normalized.strip():
        return CodeFilterResult(False, "empty_content")
    byte_count = len(normalized.encode("utf-8"))
    if byte_count < settings.minimum_file_bytes:
        return CodeFilterResult(False, "too_small")
    if byte_count > settings.maximum_file_bytes:
        return CodeFilterResult(False, "too_large")

    path = _extract_path(raw_record)
    if not _looks_like_python(path, normalized):
        return CodeFilterResult(False, "not_python")
    if _is_bad_path(path):
        return CodeFilterResult(False, "rejected_path")
    if _is_autogenerated(raw_record, normalized):
        return CodeFilterResult(False, "autogenerated")
    if _is_binary_like(normalized):
        return CodeFilterResult(False, "binary_like")
    if _is_minified(normalized):
        return CodeFilterResult(False, "minified")
    if not _has_python_structure(normalized):
        return CodeFilterResult(False, "low_python_signal")

    normalized_license = normalize_license(_extract_license(raw_record))
    accepted_licenses = {normalize_license(item) for item in settings.accepted_licenses}
    if normalized_license not in accepted_licenses:
        reason = "unknown_license" if normalized_license is None else "rejected_license"
        if settings.require_known_license:
            return CodeFilterResult(False, reason)

    hash_value = content_hash(normalized)
    return CodeFilterResult(
        True,
        "accepted",
        FilteredCodeRecord(
            text=normalized,
            repo_name=_extract_repo_name(raw_record),
            path=path,
            license=normalized_license,
            content_hash=hash_value,
            byte_count=byte_count,
            split=stable_split(hash_value, validation_percent),
        ),
    )


def _extract_text(record: dict[str, Any]) -> str | None:
    for key in ("text", "content", "code"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def _extract_path(record: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "filepath", "filename"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.replace("\\", "/")
    return None


def _extract_repo_name(record: dict[str, Any]) -> str | None:
    for key in ("repo_name", "repository_name", "repo"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_license(record: dict[str, Any]) -> object:
    for key in ("license", "licence", "licenses"):
        if key in record:
            return record[key]
    return None


def _looks_like_python(path: str | None, text: str) -> bool:
    if path and PurePosixPath(path).suffix == ".py":
        return True
    first_line = text.split("\n", 1)[0].lower()
    return "python" in first_line and first_line.startswith("#!")


def _is_bad_path(path: str | None) -> bool:
    if not path:
        return False
    lowered_parts = {part.lower() for part in PurePosixPath(path).parts}
    if lowered_parts & REJECTED_PATH_PARTS:
        return True
    lowered = path.lower()
    generated_markers = ("generated", "minified", ".min.py", "bundle.py")
    if any(marker in lowered for marker in generated_markers):
        return True
    return "migrations" in lowered_parts and ("auto" in lowered or "generated" in lowered)


def _is_autogenerated(record: dict[str, Any], text: str) -> bool:
    for key in ("is_generated", "generated", "autogenerated"):
        value = record.get(key)
        if isinstance(value, bool) and value:
            return True
    head = text[:2000].lower()
    return any(
        marker in head
        for marker in (
            "auto-generated",
            "autogenerated",
            "automatically generated",
            "generated by",
            "do not edit",
        )
    )


def _is_binary_like(text: str) -> bool:
    if "\ufffd" in text:
        return True
    control_count = sum(1 for char in text if ord(char) < 32 and char not in "\n\t")
    return control_count / max(len(text), 1) > 0.01


def _is_minified(text: str) -> bool:
    lines = text.splitlines()
    if not lines:
        return True
    longest = max(len(line) for line in lines)
    average = sum(len(line) for line in lines) / len(lines)
    return longest > 5000 or (len(lines) <= 3 and average > 1000)


def _has_python_structure(text: str) -> bool:
    return any(marker in text for marker in PYTHON_MARKERS)


__all__ = [
    "CodeFilteringError",
    "CodeFilterResult",
    "CodeFilterSettings",
    "FilteredCodeRecord",
    "content_hash",
    "filter_code_record",
    "normalize_license",
    "normalize_python_source",
    "stable_split",
]
