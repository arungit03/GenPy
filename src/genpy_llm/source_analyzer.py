"""Source summaries for merged GenPy corpora."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def build_source_report(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize source systems, repositories, packages, and byte counts."""

    source_types: Counter[str] = Counter()
    repositories: set[str] = set()
    packages: set[str] = set()
    sources: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"files": 0, "bytes": 0, "source_type": None}
    )
    for record in records:
        source = record.get("source")
        source = source if isinstance(source, Mapping) else {}
        source_type = str(source.get("type") or "unknown")
        source_id = str(source.get("id") or source_type)
        source_types[source_type] += 1
        sources[source_id]["files"] += 1
        sources[source_id]["bytes"] += int(record.get("size") or record.get("size_bytes") or 0)
        sources[source_id]["source_type"] = source_type
        repository = source.get("repository_url")
        if isinstance(repository, str) and repository:
            repositories.add(repository)
        package = source.get("package")
        if isinstance(package, str) and package:
            packages.add(package)
    return {
        "total_sources": len(sources),
        "total_repositories": len(repositories),
        "total_packages": len(packages),
        "source_types": dict(sorted(source_types.items())),
        "sources": dict(sorted(sources.items())),
    }


__all__ = ["build_source_report"]
