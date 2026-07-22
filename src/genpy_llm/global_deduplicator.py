"""Global duplicate detection for the final GenPy pretraining corpus."""

from __future__ import annotations

import ast
import hashlib
import io
import tokenize
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class GlobalDeduplicationConfig:
    """Configurable final deduplication settings."""

    exact_sha256: bool = True
    whitespace_normalization: bool = True
    comment_normalization: bool = True
    newline_normalization: bool = True
    ast_normalization: bool = False
    include_duplicate_groups: bool = True
    maximum_duplicate_groups: int = 1000


@dataclass(frozen=True)
class DuplicateDecision:
    """One rejected duplicate and the canonical record it matched."""

    reason: str
    duplicate_key: str
    canonical_path: str


@dataclass
class GlobalDeduplicator:
    """Incremental deterministic duplicate detector across all corpus sources."""

    config: GlobalDeduplicationConfig
    _seen: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict, init=False)
    _groups: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list),
        init=False,
    )
    _reasons: Counter[str] = field(default_factory=Counter, init=False)
    _accepted: int = 0

    def check(
        self,
        record: Mapping[str, Any],
        text: str,
    ) -> DuplicateDecision | None:
        """Return a duplicate decision, or register the record as canonical."""

        for reason, value in self._keys(record, text):
            key = (reason, value)
            canonical = self._seen.get(key)
            if canonical is not None:
                duplicate = _location(record)
                self._groups[key].append(duplicate)
                self._reasons[reason] += 1
                return DuplicateDecision(
                    reason=reason,
                    duplicate_key=value,
                    canonical_path=str(canonical["stored_path"]),
                )
        for reason, value in self._keys(record, text):
            key = (reason, value)
            self._seen.setdefault(key, _location(record))
        self._accepted += 1
        return None

    def report(self) -> dict[str, Any]:
        """Return a JSON-serializable duplicate report."""

        duplicate_total = sum(self._reasons.values())
        total_seen = self._accepted + duplicate_total
        groups: list[dict[str, Any]] = []
        if self.config.include_duplicate_groups:
            for (reason, duplicate_key), duplicates in sorted(self._groups.items()):
                canonical = self._seen[(reason, duplicate_key)]
                groups.append(
                    {
                        "reason": reason,
                        "key": duplicate_key,
                        "canonical": canonical,
                        "duplicates": duplicates,
                        "duplicate_count": len(duplicates),
                    }
                )
                if len(groups) >= self.config.maximum_duplicate_groups:
                    break
        return {
            "duplicate_count": duplicate_total,
            "accepted_count": self._accepted,
            "total_candidates": total_seen,
            "duplicate_percentage": _percentage(duplicate_total, total_seen),
            "reasons": dict(sorted(self._reasons.items())),
            "duplicate_groups": groups,
            "duplicate_groups_truncated": len(self._groups) > len(groups),
        }

    def _keys(self, record: Mapping[str, Any], text: str) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        content_sha256 = record.get("content_sha256")
        if self.config.exact_sha256 and isinstance(content_sha256, str):
            keys.append(("exact_sha256_duplicate", content_sha256))
        normalized_text = text
        if self.config.newline_normalization:
            normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n")
        if self.config.comment_normalization:
            normalized_text = _strip_comments(normalized_text)
        if self.config.whitespace_normalization:
            normalized_text = _normalize_whitespace(normalized_text)
        if (
            self.config.whitespace_normalization
            or self.config.comment_normalization
            or self.config.newline_normalization
        ):
            keys.append(("normalized_duplicate", _hash_text(normalized_text)))
        if self.config.ast_normalization:
            ast_dump = _ast_fingerprint(text)
            if ast_dump is not None:
                keys.append(("ast_duplicate", _hash_text(ast_dump)))
        return keys


def _strip_comments(text: str) -> str:
    output: list[tokenize.TokenInfo] = []
    stream = io.StringIO(text).readline
    try:
        tokens = tokenize.generate_tokens(stream)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                continue
            output.append(token)
    except tokenize.TokenError:
        return text
    return tokenize.untokenize(output)


def _normalize_whitespace(text: str) -> str:
    return "\n".join(" ".join(line.strip().split()) for line in text.splitlines() if line.strip())


def _ast_fingerprint(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        return None
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _location(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record.get("source")
    source = source if isinstance(source, Mapping) else {}
    return {
        "stored_path": record.get("stored_path"),
        "source_path": record.get("source_path"),
        "source_type": source.get("type"),
        "source_id": source.get("id"),
        "repository": source.get("repository_url"),
        "package": source.get("package"),
        "origin_url": (
            source.get("download_url")
            or source.get("repository_url")
            or source.get("location")
        ),
        "relative_path": PurePosixPath(str(record.get("stored_path", ""))).as_posix(),
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _percentage(value: int, total: int) -> float:
    return round((value / total) * 100, 6) if total else 0.0


__all__ = [
    "DuplicateDecision",
    "GlobalDeduplicationConfig",
    "GlobalDeduplicator",
]
