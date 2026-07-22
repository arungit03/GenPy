"""Unicode-aware tokenization for GenPy LLM."""

from __future__ import annotations

import json
import os
import tempfile
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from genpy_llm.config import TokenizationConfig

QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
    }
)
DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "'", '"'}
NO_SPACE_AFTER = {"(", "[", "{", "$", "#", "'", '"'}


@dataclass(frozen=True)
class TokenizationStats:
    """Summary of a tokenization run.

    Step 3 only counts token strings. Step 4 will build the permanent vocabulary
    and assign integer token IDs.
    """

    input_file: Path
    output_file: Path
    input_lines: int
    tokenized_sequences: int
    total_tokens: int
    word_tokens: int
    punctuation_tokens: int
    special_tokens: int
    empty_lines_skipped: int
    unique_tokens: int

    def summary(self) -> str:
        """Return a readable multi-line summary."""

        return "\n".join(
            [
                "Tokenization summary",
                "====================",
                f"Input file: {self.input_file}",
                f"Output file: {self.output_file}",
                f"Input lines: {self.input_lines}",
                f"Tokenized sequences: {self.tokenized_sequences}",
                f"Total tokens: {self.total_tokens}",
                f"Word tokens: {self.word_tokens}",
                f"Punctuation tokens: {self.punctuation_tokens}",
                f"Special tokens: {self.special_tokens}",
                f"Empty lines skipped: {self.empty_lines_skipped}",
                f"Unique tokens: {self.unique_tokens}",
            ]
        )

    def __str__(self) -> str:
        """Return the same readable summary used by the CLI."""

        return self.summary()


@dataclass
class _MutableStats:
    input_lines: int = 0
    tokenized_sequences: int = 0
    total_tokens: int = 0
    word_tokens: int = 0
    punctuation_tokens: int = 0
    special_tokens: int = 0
    empty_lines_skipped: int = 0


class TextTokenizer:
    """Convert cleaned Unicode text into human-readable token strings."""

    def __init__(self, config: TokenizationConfig) -> None:
        self.config = config
        self.special_tokens = {
            config.bos_token,
            config.eos_token,
            config.newline_token,
            config.unknown_token,
        }

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a complete text string and add sequence tokens once."""

        normalized_text = self._normalize_text(text)
        body_tokens = self._tokenize_body(normalized_text)
        if not body_tokens:
            return []
        return self._add_sequence_tokens(body_tokens)

    def tokenize_line(self, line: str) -> list[str]:
        """Tokenize one cleaned line as one training sequence."""

        normalized_line = self._normalize_text(line.rstrip("\r\n"))
        body_tokens = self._tokenize_body(normalized_line)
        if not body_tokens:
            return []
        return self._add_sequence_tokens(body_tokens)

    def detokenize(self, tokens: Sequence[str]) -> str:
        """Reconstruct readable text for debugging.

        Perfect original-text reconstruction is not guaranteed for this
        educational rule-based tokenizer.
        """

        pieces: list[str] = []
        for token in tokens:
            if token in {self.config.bos_token, self.config.eos_token}:
                continue
            if token == self.config.newline_token:
                pieces.append("\n")
                continue
            if not pieces or pieces[-1].endswith(("\n", " ")):
                pieces.append(token)
            elif token in NO_SPACE_BEFORE:
                pieces.append(token)
            elif pieces[-1] in NO_SPACE_AFTER:
                pieces.append(token)
            else:
                pieces.append(f" {token}")

        return "".join(pieces)

    def iter_file_tokens(
        self,
        input_path: Path,
        encoding: str = "utf-8",
    ) -> Iterator[list[str]]:
        """Yield token lists for each non-empty line in a file."""

        input_path = input_path.resolve()
        self._validate_input_path(input_path)
        with input_path.open("r", encoding=encoding) as input_file:
            for line in input_file:
                tokens = self.tokenize_line(line)
                if tokens:
                    yield tokens

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        encoding: str = "utf-8",
    ) -> TokenizationStats:
        """Tokenize a file line by line and write JSONL output atomically."""

        input_path = input_path.resolve()
        output_path = output_path.resolve()
        self._validate_file_paths(input_path, output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        stats = _MutableStats()
        unique_tokens: set[str] = set()
        sequence_id = 0

        try:
            temp_path = _create_temp_path(output_path)
            with input_path.open("r", encoding=encoding) as input_file:
                with temp_path.open("w", encoding=encoding, newline="\n") as output_file:
                    for line in input_file:
                        stats.input_lines += 1
                        tokens = self.tokenize_line(line)
                        if not tokens:
                            stats.empty_lines_skipped += 1
                            continue

                        record = {
                            "sequence_id": sequence_id,
                            "tokens": tokens,
                            "token_count": len(tokens),
                        }
                        output_file.write(json.dumps(record, ensure_ascii=False))
                        output_file.write("\n")

                        self._update_stats(stats, tokens)
                        unique_tokens.update(tokens)
                        stats.tokenized_sequences += 1
                        sequence_id += 1

            temp_path.replace(output_path)
            temp_path = None
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            raise

        return TokenizationStats(
            input_file=input_path,
            output_file=output_path,
            input_lines=stats.input_lines,
            tokenized_sequences=stats.tokenized_sequences,
            total_tokens=stats.total_tokens,
            word_tokens=stats.word_tokens,
            punctuation_tokens=stats.punctuation_tokens,
            special_tokens=stats.special_tokens,
            empty_lines_skipped=stats.empty_lines_skipped,
            unique_tokens=len(unique_tokens),
        )

    def _tokenize_body(self, text: str) -> list[str]:
        if self.config.method == "character":
            return self._tokenize_characters(text)
        if self.config.method == "word":
            return self._tokenize_words(text)
        raise ValueError(f"Unsupported tokenization method: {self.config.method}")

    def _tokenize_characters(self, text: str) -> list[str]:
        tokens: list[str] = []
        for character in text:
            if character in "\r\n":
                if self.config.preserve_newlines and self.config.add_newline_token:
                    if tokens and tokens[-1] != self.config.newline_token:
                        tokens.append(self.config.newline_token)
                continue
            if character.isspace():
                continue
            if not self.config.preserve_punctuation and _is_punctuation(character):
                continue
            tokens.append(character)
        return tokens

    def _tokenize_words(self, text: str) -> list[str]:
        tokens: list[str] = []
        current_token: list[str] = []

        for index, character in enumerate(text):
            if character in "\r\n":
                self._flush_current_token(current_token, tokens)
                if self.config.preserve_newlines and self.config.add_newline_token:
                    if tokens and tokens[-1] != self.config.newline_token:
                        tokens.append(self.config.newline_token)
                continue

            if character.isspace():
                self._flush_current_token(current_token, tokens)
                continue

            if self._is_apostrophe_inside_word(text, index, current_token):
                current_token.append(character)
                continue

            if _is_word_character(character):
                current_token.append(character)
                continue

            self._flush_current_token(current_token, tokens)
            if _is_punctuation(character):
                if self.config.preserve_punctuation:
                    tokens.append(character)
            else:
                tokens.append(character)

        self._flush_current_token(current_token, tokens)
        return tokens

    def _normalize_text(self, text: str) -> str:
        if not self.config.preserve_case:
            text = text.lower()
        if self.config.normalize_quotes:
            text = text.translate(QUOTE_TRANSLATION)
        if self.config.normalize_dashes:
            text = text.translate(DASH_TRANSLATION)
        return text

    def _add_sequence_tokens(self, tokens: list[str]) -> list[str]:
        output_tokens = list(tokens)
        if self.config.add_bos_token:
            output_tokens.insert(0, self.config.bos_token)
        if self.config.add_eos_token:
            output_tokens.append(self.config.eos_token)
        return output_tokens

    def _is_apostrophe_inside_word(
        self,
        text: str,
        index: int,
        current_token: list[str],
    ) -> bool:
        if self.config.split_contractions or text[index] != "'":
            return False
        next_character = text[index + 1] if index + 1 < len(text) else ""
        return bool(current_token and next_character and _is_word_character(next_character))

    def _update_stats(self, stats: _MutableStats, tokens: Sequence[str]) -> None:
        for token in tokens:
            stats.total_tokens += 1
            if token in self.special_tokens:
                stats.special_tokens += 1
            elif len(token) == 1 and _is_punctuation(token):
                stats.punctuation_tokens += 1
            else:
                stats.word_tokens += 1

    @staticmethod
    def _flush_current_token(current_token: list[str], tokens: list[str]) -> None:
        if current_token:
            tokens.append("".join(current_token))
            current_token.clear()

    @staticmethod
    def _validate_input_path(input_path: Path) -> None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not input_path.is_file():
            raise IsADirectoryError(f"Input path is not a file: {input_path}")

    @classmethod
    def _validate_file_paths(cls, input_path: Path, output_path: Path) -> None:
        cls._validate_input_path(input_path)
        if output_path.exists() and output_path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {output_path}")
        if input_path == output_path:
            raise ValueError("Input and output paths must be different files.")


def _is_word_character(character: str) -> bool:
    return unicodedata.category(character)[0] in {"L", "M", "N"}


def _is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")


def _create_temp_path(output_path: Path) -> Path:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    os.close(file_descriptor)
    return Path(temp_name)
