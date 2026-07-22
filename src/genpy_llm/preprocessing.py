"""Configurable text preprocessing for GenPy LLM."""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from genpy_llm.config import PreprocessingConfig

_WHITESPACE_PATTERN = re.compile(r"[^\S\n]+")


@dataclass(frozen=True)
class PreprocessingStats:
    """Summary of a text preprocessing run."""

    input_file: Path
    output_file: Path
    original_characters: int
    cleaned_characters: int
    original_lines: int
    written_lines: int
    skipped_empty_lines: int
    skipped_short_lines: int
    skipped_long_lines: int

    def summary(self) -> str:
        """Return a readable multi-line summary."""

        return "\n".join(
            [
                "Preprocessing summary",
                "=====================",
                f"Input file: {self.input_file}",
                f"Output file: {self.output_file}",
                f"Original characters: {self.original_characters}",
                f"Cleaned characters: {self.cleaned_characters}",
                f"Original lines: {self.original_lines}",
                f"Written lines: {self.written_lines}",
                f"Skipped empty lines: {self.skipped_empty_lines}",
                f"Skipped short lines: {self.skipped_short_lines}",
                f"Skipped long lines: {self.skipped_long_lines}",
            ]
        )

    def __str__(self) -> str:
        """Return the same readable summary used by the CLI."""

        return self.summary()


@dataclass
class _MutableStats:
    original_characters: int = 0
    cleaned_characters: int = 0
    original_lines: int = 0
    written_lines: int = 0
    skipped_empty_lines: int = 0
    skipped_short_lines: int = 0
    skipped_long_lines: int = 0


class TextPreprocessor:
    """Clean raw text using configurable, Unicode-friendly rules."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def clean_text(self, text: str) -> str:
        """Clean a small text string using the configured line rules."""

        cleaned_lines = []
        for line in text.splitlines():
            cleaned_line = self.clean_line(line)
            if cleaned_line is not None:
                cleaned_lines.append(cleaned_line)

        separator = "\n" if self.config.preserve_newlines else " "
        cleaned_text = separator.join(cleaned_lines)
        if not self.config.preserve_newlines and self.config.normalize_whitespace:
            cleaned_text = _WHITESPACE_PATTERN.sub(" ", cleaned_text)
        return cleaned_text.strip() if self.config.strip_lines else cleaned_text

    def clean_line(self, line: str) -> str | None:
        """Clean one line of text and return None when it should be skipped."""

        cleaned_line, _reason = self._clean_line_with_reason(line)
        return cleaned_line

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        encoding: str = "utf-8",
    ) -> PreprocessingStats:
        """Process a text file line by line and write cleaned output atomically."""

        input_path = input_path.resolve()
        output_path = output_path.resolve()
        self._validate_file_paths(input_path, output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        stats = _MutableStats()

        try:
            temp_path = _create_temp_path(output_path)
            with input_path.open("r", encoding=encoding) as input_file:
                with temp_path.open("w", encoding=encoding, newline="\n") as output_file:
                    for line in input_file:
                        stats.original_lines += 1
                        stats.original_characters += len(line)

                        cleaned_line, skip_reason = self._clean_line_with_reason(line)
                        if cleaned_line is None:
                            _record_skip(stats, skip_reason)
                            continue

                        written_text = f"{cleaned_line}\n"
                        output_file.write(written_text)
                        stats.cleaned_characters += len(written_text)
                        stats.written_lines += 1

            temp_path.replace(output_path)
            temp_path = None
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            raise

        return PreprocessingStats(
            input_file=input_path,
            output_file=output_path,
            original_characters=stats.original_characters,
            cleaned_characters=stats.cleaned_characters,
            original_lines=stats.original_lines,
            written_lines=stats.written_lines,
            skipped_empty_lines=stats.skipped_empty_lines,
            skipped_short_lines=stats.skipped_short_lines,
            skipped_long_lines=stats.skipped_long_lines,
        )

    def _clean_line_with_reason(self, line: str) -> tuple[str | None, str | None]:
        line = line.rstrip("\r\n")
        line = self._normalize_unicode(line)

        if self.config.remove_control_characters:
            line = _remove_control_characters(line)

        if self.config.lowercase:
            line = line.lower()

        if self.config.normalize_whitespace:
            line = _WHITESPACE_PATTERN.sub(" ", line)

        if self.config.strip_lines:
            line = line.strip()

        if self.config.remove_empty_lines and not line:
            return None, "empty"

        line_length = len(line)
        if line_length < self.config.min_line_length:
            return None, "short"
        if self.config.max_line_length is not None and line_length > self.config.max_line_length:
            return None, "long"

        return line, None

    def _normalize_unicode(self, text: str) -> str:
        if self.config.unicode_normalization == "none":
            return text
        return unicodedata.normalize(self.config.unicode_normalization, text)

    @staticmethod
    def _validate_file_paths(input_path: Path, output_path: Path) -> None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not input_path.is_file():
            raise IsADirectoryError(f"Input path is not a file: {input_path}")
        if output_path.exists() and output_path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {output_path}")
        if input_path == output_path:
            raise ValueError("Input and output paths must be different files.")


def _remove_control_characters(text: str) -> str:
    return "".join(
        character
        for character in text
        if character in {"\n", "\t"} or not unicodedata.category(character).startswith("C")
    )


def _create_temp_path(output_path: Path) -> Path:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    os.close(file_descriptor)
    return Path(temp_name)


def _record_skip(stats: _MutableStats, reason: str | None) -> None:
    if reason == "empty":
        stats.skipped_empty_lines += 1
    elif reason == "short":
        stats.skipped_short_lines += 1
    elif reason == "long":
        stats.skipped_long_lines += 1
