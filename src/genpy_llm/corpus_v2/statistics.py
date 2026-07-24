"""Corpus V2 aggregate statistics."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from genpy_llm.corpus_v2.manifest import TokenizedDocument


@dataclass(frozen=True)
class CorpusStatistics:
    """Computed corpus statistics."""

    total_documents: int
    total_files: int
    total_tokens: int
    python_tokens: int
    technical_text_tokens: int
    python_ratio: float
    technical_text_ratio: float
    average_tokens: float
    median_tokens: int
    largest_file: str | None
    smallest_file: str | None
    largest_file_tokens: int
    smallest_file_tokens: int
    token_distribution: dict[str, int]
    content_type_documents: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        """Return JSON-compatible statistics."""

        return self.__dict__.copy()


def build_statistics(documents: list[TokenizedDocument]) -> CorpusStatistics:
    """Compute aggregate corpus statistics."""

    token_counts = [document.token_count for document in documents]
    total_tokens = sum(token_counts)
    content_tokens: Counter[str] = Counter()
    content_docs: Counter[str] = Counter()
    for document in documents:
        content_tokens[document.content_type] += document.token_count
        content_docs[document.content_type] += 1
    sorted_docs = sorted(documents, key=lambda item: item.token_count)
    smallest = sorted_docs[0] if sorted_docs else None
    largest = sorted_docs[-1] if sorted_docs else None
    return CorpusStatistics(
        total_documents=len(documents),
        total_files=len(documents),
        total_tokens=total_tokens,
        python_tokens=content_tokens["python_code"],
        technical_text_tokens=content_tokens["technical_text"],
        python_ratio=_ratio(content_tokens["python_code"], total_tokens),
        technical_text_ratio=_ratio(content_tokens["technical_text"], total_tokens),
        average_tokens=round(total_tokens / len(documents), 6) if documents else 0.0,
        median_tokens=int(median(token_counts)) if token_counts else 0,
        largest_file=largest.stored_path if largest else None,
        smallest_file=smallest.stored_path if smallest else None,
        largest_file_tokens=largest.token_count if largest else 0,
        smallest_file_tokens=smallest.token_count if smallest else 0,
        token_distribution=_distribution(token_counts),
        content_type_documents=dict(sorted(content_docs.items())),
    )


def write_statistics_csv(path: Path, statistics: CorpusStatistics, duplicate_rate: float) -> None:
    """Write a compact CSV summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        payload = statistics.to_json()
        payload["duplicate_rate"] = duplicate_rate
        for key, value in payload.items():
            writer.writerow({"metric": key, "value": value})


def _distribution(values: list[int]) -> dict[str, int]:
    buckets = {
        "lt_128": 0,
        "128_511": 0,
        "512_2047": 0,
        "2048_8191": 0,
        "gte_8192": 0,
    }
    for value in values:
        if value < 128:
            buckets["lt_128"] += 1
        elif value < 512:
            buckets["128_511"] += 1
        elif value < 2048:
            buckets["512_2047"] += 1
        elif value < 8192:
            buckets["2048_8191"] += 1
        else:
            buckets["gte_8192"] += 1
    return buckets


def _ratio(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


__all__ = [
    "CorpusStatistics",
    "build_statistics",
    "write_statistics_csv",
]
