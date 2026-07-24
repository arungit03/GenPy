"""Syntax and source-type validation for Corpus V2."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass

from genpy_llm.corpus_v2.manifest import CleanDocument
from genpy_llm.corpus_v2.quality import QualityResult, QualitySettings, evaluate_quality


@dataclass(frozen=True)
class ValidationSettings:
    """Document validation settings."""

    require_python_syntax: bool = True
    quality: QualitySettings = QualitySettings()


@dataclass(frozen=True)
class ValidationResult:
    """Validation result with attached quality metrics."""

    accepted: bool
    reason: str
    quality: dict[str, object]


def validate_document(
    document: CleanDocument,
    settings: ValidationSettings,
) -> ValidationResult:
    """Validate Python syntax or technical text quality."""

    if document.content_type == "python_code" and settings.require_python_syntax:
        try:
            ast.parse(document.text, filename=document.relative_path)
        except (SyntaxError, ValueError, TypeError):
            return _rejected("invalid_python_syntax", _quality(document, settings.quality))
    quality = _quality(document, settings.quality)
    if not quality.accepted:
        return _rejected(quality.reason, quality)
    return ValidationResult(True, "accepted", asdict(quality))


def _quality(document: CleanDocument, settings: QualitySettings) -> QualityResult:
    return evaluate_quality(
        document.text,
        content_type=document.content_type,
        settings=settings,
    )


def _rejected(reason: str, quality: QualityResult) -> ValidationResult:
    return ValidationResult(False, reason, asdict(quality))


__all__ = [
    "ValidationResult",
    "ValidationSettings",
    "validate_document",
]
