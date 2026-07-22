"""Statistics helpers for final GenPy pretraining corpora."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from statistics import median
from typing import Any


def build_corpus_report(
    *,
    started_at: str,
    completed_at: str,
    input_files: int,
    accepted_files: int,
    rejected_files: int,
    duplicates_removed: int,
    final_files: int,
    total_tokens: int,
    total_sequences: int,
    shard_count: int,
) -> dict[str, Any]:
    """Create the top-level final corpus report."""

    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "input_files": input_files,
        "accepted_files": accepted_files,
        "rejected_files": rejected_files,
        "duplicates_removed": duplicates_removed,
        "final_files": final_files,
        "total_tokens": total_tokens,
        "training_sequences": total_sequences,
        "binary_shards": shard_count,
    }


def build_quality_report(
    records: Iterable[Mapping[str, Any]],
    *,
    validation_report: Mapping[str, Any],
    duplicate_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize source quality and rejection health."""

    byte_sizes: list[int] = []
    line_counts: list[int] = []
    categories: Counter[str] = Counter()
    for record in records:
        byte_sizes.append(int(record.get("size_bytes") or record.get("size") or 0))
        line_counts.append(int(record.get("line_count") or 0))
        category = record.get("primary_category")
        if isinstance(category, str) and category:
            categories[category] += 1
    return {
        "accepted_files": len(byte_sizes),
        "validation": dict(validation_report),
        "duplicates": dict(duplicate_report),
        "average_file_bytes": _average(byte_sizes),
        "median_file_bytes": _median_int(byte_sizes),
        "maximum_file_bytes": max(byte_sizes, default=0),
        "minimum_file_bytes": min(byte_sizes, default=0),
        "average_lines_per_file": _average(line_counts),
        "category_distribution": dict(sorted(categories.items())),
    }


def build_token_statistics(
    token_counts: Iterable[int],
    *,
    used_token_ids: set[int],
    vocab_size: int,
) -> dict[str, Any]:
    """Summarize token counts and vocabulary utilization."""

    counts = sorted(int(value) for value in token_counts)
    total = sum(counts)
    return {
        "total_tokens": total,
        "files": len(counts),
        "average_tokens_per_file": round(total / len(counts), 6) if counts else 0.0,
        "median_tokens_per_file": _median_int(counts),
        "p95_tokens_per_file": _percentile(counts, 0.95),
        "maximum_tokens_per_file": max(counts, default=0),
        "minimum_tokens_per_file": min(counts, default=0),
        "vocab_size": vocab_size,
        "used_token_ids": len(used_token_ids),
        "vocabulary_utilization": round(len(used_token_ids) / vocab_size, 6)
        if vocab_size
        else 0.0,
    }


def build_shard_statistics(shard_index: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize final binary shards from an index payload."""

    shards = shard_index.get("shards")
    shards = shards if isinstance(shards, list) else []
    return {
        "shard_count": len(shards),
        "total_tokens": int(shard_index.get("token_count") or 0),
        "training_sequences": int(shard_index.get("sequence_count") or 0),
        "byte_count": int(shard_index.get("byte_count") or 0),
        "shards": shards,
    }


def _average(values: list[int]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _median_int(values: list[int]) -> int:
    return int(median(values)) if values else 0


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, int(round((len(values) - 1) * percentile)))
    return values[index]


__all__ = [
    "build_corpus_report",
    "build_quality_report",
    "build_shard_statistics",
    "build_token_statistics",
]
