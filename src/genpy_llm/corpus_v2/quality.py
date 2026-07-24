"""Quality filters for Corpus V2."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genpy_llm.corpus_v2.manifest import write_json
from genpy_llm.corpus_v2.statistics import CorpusStatistics


@dataclass(frozen=True)
class QualitySettings:
    """Quality thresholds for code and technical text."""

    minimum_entropy: float = 2.5
    sample_characters: int = 200_000
    maximum_base64_fraction: float = 0.35
    maximum_hex_fraction: float = 0.35
    maximum_repeated_line_fraction: float = 0.35
    minimum_technical_score: int = 2


@dataclass(frozen=True)
class QualityResult:
    """Quality decision and metrics for one document."""

    accepted: bool
    reason: str
    entropy: float
    base64_fraction: float
    hex_fraction: float
    repeated_line_fraction: float
    technical_score: int


@dataclass(frozen=True)
class ReadinessSettings:
    """Corpus readiness gates."""

    minimum_tokens: int = 200_000_000
    min_python_ratio: float = 0.45
    max_python_ratio: float = 0.75
    min_technical_text_ratio: float = 0.20
    max_technical_text_ratio: float = 0.55
    max_duplicate_percentage: float = 0.15


@dataclass(frozen=True)
class ReadinessResult:
    """Outcome of corpus readiness gates."""

    passed: bool
    failures: tuple[str, ...]


TECHNICAL_TERMS = {
    "api",
    "argument",
    "async",
    "class",
    "cli",
    "code",
    "config",
    "dataset",
    "debug",
    "dependency",
    "exception",
    "function",
    "install",
    "json",
    "module",
    "package",
    "parameter",
    "python",
    "return",
    "schema",
    "server",
    "test",
    "token",
    "training",
    "type",
    "validation",
    "yaml",
}


def evaluate_quality(
    text: str,
    *,
    content_type: str,
    settings: QualitySettings,
) -> QualityResult:
    """Run non-semantic quality checks over cleaned text."""

    sample = _quality_sample(text, settings.sample_characters)
    entropy = _entropy(sample)
    base64_fraction = _base64_fraction(sample)
    hex_fraction = _hex_fraction(sample)
    repeated_line_fraction = _repeated_line_fraction(sample)
    technical_score = technical_content_score(sample)
    if entropy < settings.minimum_entropy:
        reason = "low_entropy"
    elif base64_fraction > settings.maximum_base64_fraction:
        reason = "base64_blob"
    elif hex_fraction > settings.maximum_hex_fraction:
        reason = "hex_dump"
    elif repeated_line_fraction > settings.maximum_repeated_line_fraction:
        reason = "repeated_sequences"
    elif content_type == "technical_text" and technical_score < settings.minimum_technical_score:
        reason = "low_technical_content"
    else:
        reason = "accepted"
    return QualityResult(
        accepted=reason == "accepted",
        reason=reason,
        entropy=round(entropy, 6),
        base64_fraction=round(base64_fraction, 6),
        hex_fraction=round(hex_fraction, 6),
        repeated_line_fraction=round(repeated_line_fraction, 6),
        technical_score=technical_score,
    )


def technical_content_score(text: str) -> int:
    """Return a rough technical-content score for documentation."""

    lowered = text.casefold()
    score = sum(1 for term in TECHNICAL_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered))
    score += min(3, len(re.findall(r"`[^`]+`|```|::|^\s{4,}\S", text, flags=re.MULTILINE)))
    score += min(3, len(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\(", text)))
    return score


def _quality_sample(text: str, sample_characters: int) -> str:
    if sample_characters <= 0 or len(text) <= sample_characters:
        return text
    head = sample_characters // 2
    tail = sample_characters - head
    return text[:head] + "\n" + text[-tail:]


def evaluate_readiness(
    statistics: CorpusStatistics,
    *,
    duplicate_percentage: float,
    validation_failures: int,
    settings: ReadinessSettings,
) -> ReadinessResult:
    """Evaluate final corpus readiness gates."""

    failures: list[str] = []
    if statistics.total_tokens < settings.minimum_tokens:
        failures.append("token_target")
    if not settings.min_python_ratio <= statistics.python_ratio <= settings.max_python_ratio:
        failures.append("python_ratio")
    if not (
        settings.min_technical_text_ratio
        <= statistics.technical_text_ratio
        <= settings.max_technical_text_ratio
    ):
        failures.append("technical_text_ratio")
    if duplicate_percentage > settings.max_duplicate_percentage:
        failures.append("duplicate_rate")
    if validation_failures:
        failures.append("validation_failures")
    return ReadinessResult(not failures, tuple(failures))


def write_quality_reports(
    *,
    output_directory: Path,
    statistics: CorpusStatistics,
    duplicate_report: dict[str, Any],
    rejection_reasons: dict[str, int],
    readiness: ReadinessResult,
    readiness_settings: ReadinessSettings,
    manifest_payload: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Write JSON, Markdown, and manifest quality artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    duplicate_percentage = float(duplicate_report.get("duplicate_percentage") or 0.0)
    payload = {
        "statistics": statistics.to_json(),
        "duplicate_report": duplicate_report,
        "rejection_reasons": rejection_reasons,
        "duplicate_percentage": duplicate_percentage,
        "readiness": {
            "passed": readiness.passed,
            "failures": list(readiness.failures),
            "settings": readiness_settings.__dict__,
        },
    }
    json_path = output_directory / "quality_report.json"
    markdown_path = output_directory / "quality_report.md"
    manifest_path = output_directory / "manifest.json"
    write_json(json_path, payload)
    write_json(manifest_path, manifest_payload)
    markdown_path.write_text(_quality_markdown(payload), encoding="utf-8")
    return json_path, markdown_path, manifest_path


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _base64_fraction(text: str) -> float:
    chunks = re.findall(r"\b[A-Za-z0-9+/]{80,}={0,2}\b", text)
    return sum(len(chunk) for chunk in chunks) / max(1, len(text))


def _hex_fraction(text: str) -> float:
    chunks = re.findall(r"\b(?:[0-9a-fA-F]{2}\s*){40,}\b", text)
    return sum(len(chunk) for chunk in chunks) / max(1, len(text))


def _repeated_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count for count in counts.values() if count > 3)
    return repeated / len(lines)


def _quality_markdown(payload: dict[str, Any]) -> str:
    statistics = payload["statistics"]
    readiness = payload["readiness"]
    lines = [
        "# GenPy Phase 6.2 Corpus V2 Quality Report",
        "",
        f"- Readiness: {'PASS' if readiness['passed'] else 'FAIL'}",
        f"- Failures: {', '.join(readiness['failures']) or 'none'}",
        f"- Total documents: {statistics['total_documents']:,}",
        f"- Total tokens: {statistics['total_tokens']:,}",
        f"- Python ratio: {statistics['python_ratio']:.2%}",
        f"- Technical text ratio: {statistics['technical_text_ratio']:.2%}",
        f"- Duplicate percentage: {payload['duplicate_percentage']:.2%}",
        f"- Average tokens/document: {statistics['average_tokens']:.2f}",
        f"- Median tokens/document: {statistics['median_tokens']:,}",
        (
            f"- Largest file: `{statistics['largest_file']}` "
            f"({statistics['largest_file_tokens']:,})"
        ),
        (
            f"- Smallest file: `{statistics['smallest_file']}` "
            f"({statistics['smallest_file_tokens']:,})"
        ),
        "",
        "## Rejections",
        "",
    ]
    for reason, count in sorted(payload["rejection_reasons"].items()):
        lines.append(f"- {reason}: {count:,}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "QualityResult",
    "QualitySettings",
    "ReadinessResult",
    "ReadinessSettings",
    "evaluate_readiness",
    "evaluate_quality",
    "technical_content_score",
    "write_quality_reports",
]
