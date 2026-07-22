"""Deterministic vocabulary building and token encoding for GenPy LLM."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genpy_llm.config import VocabularyConfig

VOCABULARY_FORMAT_VERSION = 1


class VocabularyError(ValueError):
    """Raised when vocabulary data is invalid."""


@dataclass(frozen=True)
class VocabularyBuildStats:
    """Summary of a vocabulary build.

    Counts for excluded tokens are counts of unique token types, not token occurrences.
    """

    input_file: Path
    vocabulary_file: Path | None
    processed_sequences: int
    total_tokens: int
    unique_tokens_observed: int
    vocabulary_size: int
    special_token_count: int
    normal_token_count: int
    excluded_below_min_frequency: int
    excluded_by_max_size: int
    empty_sequences: int

    def summary(self) -> str:
        """Return a readable multi-line summary."""

        return "\n".join(
            [
                "Vocabulary build summary",
                "========================",
                f"Input file: {self.input_file}",
                f"Vocabulary file: {self.vocabulary_file}",
                f"Processed sequences: {self.processed_sequences}",
                f"Total tokens observed: {self.total_tokens}",
                f"Unique tokens observed: {self.unique_tokens_observed}",
                f"Final vocabulary size: {self.vocabulary_size}",
                f"Special tokens: {self.special_token_count}",
                f"Normal tokens: {self.normal_token_count}",
                f"Below minimum frequency: {self.excluded_below_min_frequency}",
                f"Excluded by maximum size: {self.excluded_by_max_size}",
                f"Empty sequences: {self.empty_sequences}",
            ]
        )

    def __str__(self) -> str:
        """Return the same readable summary used by the CLI."""

        return self.summary()


class Vocabulary:
    """A reversible mapping between token strings and integer IDs."""

    def __init__(
        self,
        token_to_id: dict[str, int],
        frequencies: dict[str, int] | None,
        config: VocabularyConfig,
    ) -> None:
        self.config = config
        self.token_to_id = dict(token_to_id)
        self.id_to_token = _build_id_to_token(self.token_to_id)
        self.frequencies = dict(frequencies) if frequencies is not None else None
        self._validate_integrity()

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        config: VocabularyConfig,
    ) -> Vocabulary:
        """Build a vocabulary from in-memory token sequences."""

        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            _validate_token_sequence(tokens, line_number=None)
            counter.update(tokens)
        token_to_id, frequencies = _build_mapping(counter, config)
        return cls(token_to_id=token_to_id, frequencies=frequencies, config=config)

    @classmethod
    def build_from_jsonl(
        cls,
        input_path: Path,
        config: VocabularyConfig,
        encoding: str = "utf-8",
    ) -> tuple[Vocabulary, VocabularyBuildStats]:
        """Build a vocabulary from Step 3 tokenized JSONL without loading all rows."""

        input_path = input_path.resolve()
        _validate_input_file(input_path)

        counter: Counter[str] = Counter()
        processed_sequences = 0
        empty_sequences = 0
        total_tokens = 0

        with input_path.open("r", encoding=encoding) as input_file:
            for line_number, line in enumerate(input_file, start=1):
                tokens = _read_jsonl_tokens(line, line_number)
                if not tokens:
                    empty_sequences += 1
                    continue
                counter.update(tokens)
                processed_sequences += 1
                total_tokens += len(tokens)

        token_to_id, frequencies = _build_mapping(counter, config)
        vocabulary = cls(token_to_id=token_to_id, frequencies=frequencies, config=config)
        stats = _build_stats(
            input_path=input_path,
            vocabulary_file=None,
            counter=counter,
            vocabulary=vocabulary,
            processed_sequences=processed_sequences,
            total_tokens=total_tokens,
            empty_sequences=empty_sequences,
            config=config,
        )
        return vocabulary, stats

    @classmethod
    def load(
        cls,
        vocabulary_path: Path,
        config: VocabularyConfig | None = None,
        encoding: str = "utf-8",
    ) -> Vocabulary:
        """Load and validate a saved vocabulary JSON file."""

        vocabulary_path = vocabulary_path.resolve()
        if not vocabulary_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_path}")
        if not vocabulary_path.is_file():
            raise IsADirectoryError(f"Vocabulary path is not a file: {vocabulary_path}")

        with vocabulary_path.open("r", encoding=encoding) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise VocabularyError("Vocabulary file must contain a JSON object.")
        if data.get("format_version") != VOCABULARY_FORMAT_VERSION:
            raise VocabularyError(
                f"Unsupported vocabulary format version: {data.get('format_version')}"
            )

        token_to_id = data.get("token_to_id")
        id_to_token = data.get("id_to_token")
        frequencies = data.get("frequencies")
        declared_size = data.get("vocab_size")

        if not isinstance(token_to_id, dict):
            raise VocabularyError("Vocabulary file must contain token_to_id mapping.")
        if not isinstance(id_to_token, list):
            raise VocabularyError("Vocabulary file must contain id_to_token list.")
        if not isinstance(declared_size, int):
            raise VocabularyError("Vocabulary file must contain integer vocab_size.")
        if declared_size != len(token_to_id):
            raise VocabularyError("Declared vocab_size does not match token_to_id size.")

        if config is None:
            config = _config_from_saved_data(data)

        vocabulary = cls(
            token_to_id=_parse_token_to_id(token_to_id),
            frequencies=_parse_frequencies(frequencies),
            config=config,
        )
        if vocabulary.id_to_token != id_to_token:
            raise VocabularyError("Saved id_to_token is not the inverse of token_to_id.")
        return vocabulary

    def save(
        self,
        vocabulary_path: Path,
        metadata_path: Path | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """Save the vocabulary JSON, and optionally a compact metadata file."""

        vocabulary_path = vocabulary_path.resolve()
        if vocabulary_path.exists() and vocabulary_path.is_dir():
            raise IsADirectoryError(f"Vocabulary path is a directory: {vocabulary_path}")
        data = self.to_json_dict()
        _write_json_atomic(vocabulary_path, data, encoding=encoding)

        if metadata_path is not None:
            metadata_path = metadata_path.resolve()
            if metadata_path.exists() and metadata_path.is_dir():
                raise IsADirectoryError(f"Metadata path is a directory: {metadata_path}")
            _write_json_atomic(
                metadata_path,
                {
                    "format_version": VOCABULARY_FORMAT_VERSION,
                    "vocabulary_file": vocabulary_path.name,
                    "vocab_size": len(self),
                    "special_token_count": len(self.special_tokens),
                    "normal_token_count": len(self) - len(self.special_tokens),
                    "min_frequency": self.config.min_frequency,
                    "max_size": self.config.max_size,
                },
                encoding=encoding,
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-serializable vocabulary representation."""

        frequencies = self.frequencies if self.config.save_frequencies else None
        return {
            "format_version": VOCABULARY_FORMAT_VERSION,
            "vocab_size": len(self),
            "special_tokens": {
                "pad_token": self.config.pad_token,
                "unknown_token": self.config.unknown_token,
                "bos_token": self.config.bos_token,
                "eos_token": self.config.eos_token,
                "newline_token": self.config.newline_token,
            },
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "frequencies": frequencies,
        }

    def encode(
        self,
        tokens: Sequence[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert tokens to IDs.

        When requested, BOS is inserted only if the first token is not already BOS,
        and EOS is appended only if the last token is not already EOS.
        """

        output_tokens = list(tokens)
        if add_bos and (not output_tokens or output_tokens[0] != self.config.bos_token):
            output_tokens.insert(0, self.config.bos_token)
        if add_eos and (not output_tokens or output_tokens[-1] != self.config.eos_token):
            output_tokens.append(self.config.eos_token)
        return [self.token_id(token) for token in output_tokens]

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Convert token IDs back to token strings."""

        tokens: list[str] = []
        for position, token_id in enumerate(token_ids):
            if not isinstance(token_id, int):
                raise VocabularyError(
                    f"Token ID at position {position} must be an integer: {token_id!r}"
                )
            if token_id < 0 or token_id >= len(self):
                raise VocabularyError(f"Invalid token ID {token_id} at position {position}.")
            token = self.id_to_token[token_id]
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return tokens

    def token_id(self, token: str) -> int:
        """Return the ID for a token, using UNK for unknown normal tokens."""

        if not isinstance(token, str) or not token:
            raise VocabularyError("Token must be a non-empty string.")
        return self.token_to_id.get(token, self.unknown_id)

    def id_token(self, token_id: int) -> str:
        """Return the token string for a valid token ID."""

        return self.decode([token_id])[0]

    @property
    def pad_id(self) -> int:
        """Return the configured PAD token ID."""

        return self._required_token_id(self.config.pad_token)

    @property
    def unknown_id(self) -> int:
        """Return the configured UNK token ID."""

        return self._required_token_id(self.config.unknown_token)

    @property
    def bos_id(self) -> int:
        """Return the configured BOS token ID."""

        return self._required_token_id(self.config.bos_token)

    @property
    def eos_id(self) -> int:
        """Return the configured EOS token ID."""

        return self._required_token_id(self.config.eos_token)

    @property
    def newline_id(self) -> int:
        """Return the configured newline token ID."""

        return self._required_token_id(self.config.newline_token)

    @property
    def special_tokens(self) -> set[str]:
        """Return configured special tokens."""

        if not self.config.include_special_tokens:
            return set()
        return set(self.config.special_token_order)

    @property
    def special_token_ids(self) -> set[int]:
        """Return configured special token IDs."""

        return {self._required_token_id(token) for token in self.special_tokens}

    def __len__(self) -> int:
        """Return vocabulary size."""

        return len(self.token_to_id)

    def _required_token_id(self, token: str) -> int:
        try:
            return self.token_to_id[token]
        except KeyError as exc:
            raise VocabularyError(f"Required special token is missing: {token}") from exc

    def _validate_integrity(self) -> None:
        _validate_token_to_id(self.token_to_id)
        if _build_id_to_token(self.token_to_id) != self.id_to_token:
            raise VocabularyError("token_to_id and id_to_token are inconsistent.")
        if self.config.include_special_tokens:
            for expected_id, token in enumerate(self.config.special_token_order):
                actual_id = self.token_to_id.get(token)
                if actual_id is None:
                    raise VocabularyError(f"Required special token is missing: {token}")
                if self.config.strict_special_token_validation and actual_id != expected_id:
                    raise VocabularyError(
                        f"Special token {token} has ID {actual_id}, expected {expected_id}."
                    )
        if self.config.unknown_token not in self.token_to_id:
            raise VocabularyError("Unknown token must be present in the vocabulary.")


def encode_jsonl_file(
    input_path: Path,
    output_path: Path,
    vocabulary: Vocabulary,
    encoding: str = "utf-8",
) -> None:
    """Encode Step 3 tokenized JSONL into token IDs without creating windows."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    _validate_output_conflicts(input_path, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        temp_path = _create_temp_path(output_path)
        with input_path.open("r", encoding=encoding) as input_file:
            with temp_path.open("w", encoding=encoding, newline="\n") as output_file:
                for line_number, line in enumerate(input_file, start=1):
                    record, tokens = _read_jsonl_record(line, line_number)
                    token_ids = vocabulary.encode(tokens)
                    encoded_record = {
                        "sequence_id": record.get("sequence_id"),
                        "tokens": tokens,
                        "token_ids": token_ids,
                        "token_count": len(tokens),
                    }
                    output_file.write(json.dumps(encoded_record, ensure_ascii=False))
                    output_file.write("\n")
        temp_path.replace(output_path)
        temp_path = None
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def save_build_metadata(
    metadata_path: Path,
    stats: VocabularyBuildStats,
    vocabulary_path: Path,
    encoded_path: Path | None,
    project_root: Path,
    config: VocabularyConfig,
    encoding: str = "utf-8",
) -> None:
    """Save portable metadata for a vocabulary build."""

    metadata = {
        "format_version": VOCABULARY_FORMAT_VERSION,
        "source_file": _portable_path(stats.input_file, project_root),
        "vocabulary_file": _portable_path(vocabulary_path, project_root),
        "encoded_file": _portable_path(encoded_path, project_root) if encoded_path else None,
        "vocab_size": stats.vocabulary_size,
        "special_token_count": stats.special_token_count,
        "normal_token_count": stats.normal_token_count,
        "unique_tokens_observed": stats.unique_tokens_observed,
        "total_tokens_observed": stats.total_tokens,
        "excluded_below_min_frequency": stats.excluded_below_min_frequency,
        "excluded_by_max_size": stats.excluded_by_max_size,
        "processed_sequences": stats.processed_sequences,
        "empty_sequences": stats.empty_sequences,
        "min_frequency": config.min_frequency,
        "max_size": config.max_size,
    }
    _write_json_atomic(metadata_path.resolve(), metadata, encoding=encoding)


def _build_mapping(
    counter: Counter[str],
    config: VocabularyConfig,
) -> tuple[dict[str, int], dict[str, int] | None]:
    token_to_id: dict[str, int] = {}
    if config.include_special_tokens:
        for token in config.special_token_order:
            token_to_id[token] = len(token_to_id)

    normal_tokens = _sorted_normal_tokens(counter, config)
    normal_limit = None
    if config.max_size is not None:
        normal_limit = config.max_size - len(token_to_id)

    selected_tokens = normal_tokens if normal_limit is None else normal_tokens[:normal_limit]
    for token in selected_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    frequencies = None
    if config.save_frequencies:
        frequencies = {token: counter.get(token, 0) for token in token_to_id}
    return token_to_id, frequencies


def _sorted_normal_tokens(counter: Counter[str], config: VocabularyConfig) -> list[str]:
    special_tokens = set(config.special_token_order) if config.include_special_tokens else set()
    return sorted(
        (
            token
            for token, frequency in counter.items()
            if token not in special_tokens and frequency >= config.min_frequency
        ),
        key=lambda token: (-counter[token], token),
    )


def _build_stats(
    input_path: Path,
    vocabulary_file: Path | None,
    counter: Counter[str],
    vocabulary: Vocabulary,
    processed_sequences: int,
    total_tokens: int,
    empty_sequences: int,
    config: VocabularyConfig,
) -> VocabularyBuildStats:
    special_tokens = set(config.special_token_order) if config.include_special_tokens else set()
    normal_observed = {token for token in counter if token not in special_tokens}
    below_min = {token for token in normal_observed if counter[token] < config.min_frequency}
    eligible = {token for token in normal_observed if counter[token] >= config.min_frequency}
    included_normal = set(vocabulary.token_to_id) - special_tokens
    return VocabularyBuildStats(
        input_file=input_path,
        vocabulary_file=vocabulary_file,
        processed_sequences=processed_sequences,
        total_tokens=total_tokens,
        unique_tokens_observed=len(counter),
        vocabulary_size=len(vocabulary),
        special_token_count=len(vocabulary.special_tokens),
        normal_token_count=len(included_normal),
        excluded_below_min_frequency=len(below_min),
        excluded_by_max_size=max(len(eligible) - len(included_normal), 0),
        empty_sequences=empty_sequences,
    )


def _read_jsonl_tokens(line: str, line_number: int) -> list[str]:
    _record, tokens = _read_jsonl_record(line, line_number)
    return tokens


def _read_jsonl_record(line: str, line_number: int) -> tuple[dict[str, Any], list[str]]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise VocabularyError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise VocabularyError(f"JSONL line {line_number} must contain a JSON object.")
    if "tokens" not in record:
        raise VocabularyError(f"JSONL line {line_number} is missing required 'tokens' field.")
    tokens = record["tokens"]
    if not isinstance(tokens, list):
        raise VocabularyError(f"JSONL line {line_number} field 'tokens' must be a list.")
    _validate_token_sequence(tokens, line_number=line_number)
    if "token_count" in record and record["token_count"] != len(tokens):
        raise VocabularyError(f"JSONL line {line_number} token_count does not match tokens length.")
    if "sequence_id" in record and not isinstance(record["sequence_id"], int):
        raise VocabularyError(f"JSONL line {line_number} sequence_id must be an integer.")
    return record, tokens


def _validate_token_sequence(tokens: Sequence[str], line_number: int | None) -> None:
    location = f" at JSONL line {line_number}" if line_number is not None else ""
    for index, token in enumerate(tokens):
        if not isinstance(token, str):
            raise VocabularyError(f"Token {index}{location} must be a string.")
        if not token:
            raise VocabularyError(f"Token {index}{location} must not be empty.")


def _validate_input_file(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise IsADirectoryError(f"Input path is not a file: {input_path}")


def _validate_output_conflicts(input_path: Path, output_path: Path) -> None:
    _validate_input_file(input_path)
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {output_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different files.")


def _parse_token_to_id(data: dict[Any, Any]) -> dict[str, int]:
    token_to_id: dict[str, int] = {}
    for token, token_id in data.items():
        if not isinstance(token, str) or not token:
            raise VocabularyError("All vocabulary tokens must be non-empty strings.")
        if not isinstance(token_id, int):
            raise VocabularyError(f"ID for token {token!r} must be an integer.")
        token_to_id[token] = token_id
    return token_to_id


def _parse_frequencies(data: Any) -> dict[str, int] | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise VocabularyError("frequencies must be an object or null.")
    frequencies: dict[str, int] = {}
    for token, frequency in data.items():
        if not isinstance(token, str) or not token:
            raise VocabularyError("Frequency tokens must be non-empty strings.")
        if not isinstance(frequency, int) or frequency < 0:
            raise VocabularyError(f"Frequency for token {token!r} must be a non-negative integer.")
        frequencies[token] = frequency
    return frequencies


def _validate_token_to_id(token_to_id: dict[str, int]) -> None:
    if not token_to_id:
        raise VocabularyError("Vocabulary must not be empty.")
    if any(not isinstance(token, str) or not token for token in token_to_id):
        raise VocabularyError("All vocabulary tokens must be non-empty strings.")
    ids = list(token_to_id.values())
    if any(not isinstance(token_id, int) for token_id in ids):
        raise VocabularyError("All vocabulary IDs must be integers.")
    if any(token_id < 0 for token_id in ids):
        raise VocabularyError("Vocabulary IDs must not be negative.")
    if len(set(ids)) != len(ids):
        raise VocabularyError("Vocabulary IDs must be unique.")
    expected_ids = list(range(len(ids)))
    if sorted(ids) != expected_ids:
        raise VocabularyError("Vocabulary IDs must be continuous from zero.")


def _build_id_to_token(token_to_id: dict[str, int]) -> list[str]:
    _validate_token_to_id(token_to_id)
    id_to_token = [""] * len(token_to_id)
    for token, token_id in token_to_id.items():
        id_to_token[token_id] = token
    return id_to_token


def _config_from_saved_data(data: dict[str, Any]) -> VocabularyConfig:
    special_tokens = data.get("special_tokens")
    id_to_token = data.get("id_to_token")
    if not isinstance(special_tokens, dict) or not isinstance(id_to_token, list):
        raise VocabularyError("Cannot infer vocabulary config from saved file.")

    required_names = ["pad_token", "unknown_token", "bos_token", "eos_token", "newline_token"]
    values: dict[str, str] = {}
    for name in required_names:
        value = special_tokens.get(name)
        if not isinstance(value, str) or not value:
            raise VocabularyError(f"Saved special token {name} is missing or invalid.")
        values[name] = value

    order = tuple(token for token in id_to_token if token in set(values.values()))
    return VocabularyConfig(
        min_frequency=1,
        max_size=None,
        include_special_tokens=True,
        save_frequencies=data.get("frequencies") is not None,
        strict_special_token_validation=False,
        pad_token=values["pad_token"],
        unknown_token=values["unknown_token"],
        bos_token=values["bos_token"],
        eos_token=values["eos_token"],
        newline_token=values["newline_token"],
        special_token_order=order,
    )


def _write_json_atomic(path: Path, data: dict[str, Any], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        temp_path = _create_temp_path(path)
        with temp_path.open("w", encoding=encoding, newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temp_path.replace(path)
        temp_path = None
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _create_temp_path(output_path: Path) -> Path:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    os.close(file_descriptor)
    return Path(temp_name)


def _portable_path(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name
