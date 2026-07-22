"""License summaries for merged GenPy corpora."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from genpy_llm.code_filtering import normalize_license


def build_license_report(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize accepted corpus licenses."""

    counts: Counter[str] = Counter()
    unknown = 0
    for record in records:
        normalized = normalize_license(record.get("license"))
        if normalized is None:
            unknown += 1
            counts["unknown"] += 1
        else:
            counts[normalized] += 1
    total = sum(counts.values())
    return {
        "total_files": total,
        "unknown_license_files": unknown,
        "licenses": dict(sorted(counts.items())),
        "license_percentages": {
            license_name: round(count / total, 6) if total else 0.0
            for license_name, count in sorted(counts.items())
        },
    }


__all__ = ["build_license_report"]
