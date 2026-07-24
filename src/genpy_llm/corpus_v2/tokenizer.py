"""Existing-tokenizer reuse for Corpus V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash
from genpy_llm.corpus_v2.manifest import CleanDocument, TokenizedDocument


@dataclass(frozen=True)
class TokenizationSettings:
    """Tokenizer path and token filtering settings."""

    tokenizer_path: Path
    minimum_tokens: int = 4


@dataclass(frozen=True)
class TokenizationResult:
    """Tokenization result for one document."""

    document: TokenizedDocument | None
    rejection_reason: str | None = None


class CorpusV2Tokenizer:
    """Thin adapter around the existing GenPy tokenizer."""

    def __init__(self, settings: TokenizationSettings) -> None:
        self.settings = settings
        self.tokenizer = CodeTokenizer.from_file(settings.tokenizer_path)
        self.tokenizer_hash = tokenizer_file_hash(settings.tokenizer_path)

    def tokenize(
        self,
        document: CleanDocument,
        *,
        normalized_sha256: str,
        quality: dict[str, Any],
    ) -> TokenizationResult:
        """Tokenize one cleaned document without changing the vocabulary."""

        token_ids = self.tokenizer.encode(document.text, add_special_tokens=False)
        if len(token_ids) < self.settings.minimum_tokens:
            return TokenizationResult(None, "too_few_tokens")
        suffix = Path(document.relative_path).suffix.casefold()
        language = {
            ".py": "Python",
            ".md": "Markdown",
            ".rst": "reStructuredText",
            ".txt": "Text",
        }.get(suffix, "Text")
        return TokenizationResult(
            TokenizedDocument(
                stored_path=f"{document.source.source_id}/{document.relative_path}",
                source_id=document.source.source_id,
                source_type=document.source.source_type,
                source_path=document.relative_path,
                content_type=document.content_type,
                language=language,
                sha256=document.sha256,
                normalized_sha256=normalized_sha256,
                token_count=len(token_ids),
                byte_count=document.byte_count,
                line_count=document.line_count,
                license=document.source.license,
                approval=document.source.approval,
                token_ids=token_ids,
                quality=quality,
            )
        )


__all__ = [
    "CorpusV2Tokenizer",
    "TokenizationResult",
    "TokenizationSettings",
]
