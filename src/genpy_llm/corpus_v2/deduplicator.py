"""Exact, normalized, and near-duplicate filtering for Corpus V2."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, deque
from dataclasses import dataclass, field

from genpy_llm.corpus_v2.manifest import CleanDocument, text_hash


@dataclass(frozen=True)
class DeduplicationSettings:
    """Deduplication controls."""

    exact: bool = True
    normalized: bool = True
    near_duplicate: bool = True
    near_duplicate_threshold: float = 0.92
    shingle_size: int = 5
    maximum_near_duplicate_index: int = 100_000


@dataclass(frozen=True)
class DeduplicationDecision:
    """Deduplication outcome for one document."""

    accepted: bool
    reason: str
    normalized_sha256: str
    duplicate_of: str | None = None


@dataclass
class Deduplicator:
    """Streaming duplicate detector."""

    settings: DeduplicationSettings
    exact_hashes: dict[str, str] = field(default_factory=dict)
    normalized_hashes: dict[str, str] = field(default_factory=dict)
    near_index: deque[tuple[str, frozenset[str]]] = field(default_factory=deque)
    reasons: Counter[str] = field(default_factory=Counter)
    accepted_count: int = 0

    def check(self, document: CleanDocument) -> DeduplicationDecision:
        """Return a duplicate decision and register accepted documents."""

        normalized = normalize_for_dedup(document.text)
        normalized_sha256 = text_hash(normalized)
        if self.settings.exact and document.sha256 in self.exact_hashes:
            return self._reject(
                "exact_duplicate",
                normalized_sha256,
                self.exact_hashes[document.sha256],
            )
        if self.settings.normalized and normalized_sha256 in self.normalized_hashes:
            return self._reject(
                "normalized_duplicate",
                normalized_sha256,
                self.normalized_hashes[normalized_sha256],
            )
        shingles = _shingles(normalized, self.settings.shingle_size)
        if self.settings.near_duplicate and shingles:
            for stored_path, indexed in self.near_index:
                similarity = _jaccard(shingles, indexed)
                if similarity >= self.settings.near_duplicate_threshold:
                    return self._reject("near_duplicate", normalized_sha256, stored_path)
        stored_path = f"{document.source.source_id}/{document.relative_path}"
        self.exact_hashes.setdefault(document.sha256, stored_path)
        self.normalized_hashes.setdefault(normalized_sha256, stored_path)
        if shingles:
            self.near_index.append((stored_path, shingles))
            while len(self.near_index) > self.settings.maximum_near_duplicate_index:
                self.near_index.popleft()
        self.accepted_count += 1
        return DeduplicationDecision(True, "accepted", normalized_sha256)

    def report(self) -> dict[str, object]:
        """Return aggregate duplicate statistics."""

        duplicate_count = sum(self.reasons.values())
        total = duplicate_count + self.accepted_count
        return {
            "accepted_count": self.accepted_count,
            "duplicate_count": duplicate_count,
            "duplicate_percentage": round(duplicate_count / total, 6) if total else 0.0,
            "reasons": dict(sorted(self.reasons.items())),
        }

    def _reject(
        self,
        reason: str,
        normalized_sha256: str,
        duplicate_of: str,
    ) -> DeduplicationDecision:
        self.reasons[reason] += 1
        return DeduplicationDecision(False, reason, normalized_sha256, duplicate_of)


def normalize_for_dedup(text: str) -> str:
    """Normalize text for document-level duplicate detection."""

    text = re.sub(r"#.*", "", text)
    text = re.sub(r"\s+", " ", text.casefold()).strip()
    return text


def _shingles(text: str, size: int) -> frozenset[str]:
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.casefold())
    if len(words) < size:
        return frozenset()
    return frozenset(
        hashlib.sha1(" ".join(words[index : index + size]).encode()).hexdigest()
        for index in range(0, len(words) - size + 1)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


__all__ = [
    "DeduplicationDecision",
    "DeduplicationSettings",
    "Deduplicator",
    "normalize_for_dedup",
]
