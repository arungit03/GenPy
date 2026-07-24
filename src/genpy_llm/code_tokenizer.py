"""Custom byte-level BPE tokenizer support for GenPy Code LLM."""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from genpy_llm.compat import zip_strict

UTC = timezone.utc


class CodeTokenizerError(ValueError):
    """Raised when the code tokenizer is missing, corrupted, or invalid."""


SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<mask>",
    "<instruction>",
    "<input>",
    "<output>",
)
LEGACY_SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
DEFAULT_TOKENIZER_PATH = Path("data/tokenizer/tokenizer.json")
DEFAULT_LEGACY_TOKENIZER_PATH = Path("data/tokenizer/code_tokenizer.json")
DEFAULT_TOKENIZER_METADATA_PATH = Path("data/tokenizer/tokenizer_metadata.json")
DEFAULT_TOKENIZER_CONFIG_PATH = Path("configs/tokenizer.yaml")
LOGGER = logging.getLogger("genpy_llm.code_tokenizer")
DEFAULT_VERIFICATION_SAMPLE = (
    "def greet(name):\n"
    "    return f\"Hello, {name}!\"\n"
    "# தமிழ் Unicode round trip\n"
)
_FALLBACK_CORPUS_PATTERNS = (
    "data/fine_tuning/**/*.jsonl",
    "data/fine_tuning/**/*.json",
    "data/fine_tuning/**/*.txt",
    "data/raw/**/*.jsonl",
    "data/raw/**/*.json",
    "data/raw/**/*.txt",
    "src/**/*.py",
    "scripts/**/*.py",
)


@dataclass(frozen=True)
class CodeTokenizerMetadata:
    """Metadata for a trained code tokenizer."""

    tokenizer_type: str
    requested_vocab_size: int
    actual_vocab_size: int
    special_tokens: tuple[str, ...]
    special_token_ids: dict[str, int]
    minimum_frequency: int
    requested_sample_bytes: int | None
    actual_sample_bytes: int
    training_shard_names: tuple[str, ...]
    training_corpus: tuple[str, ...]
    creation_timestamp: str
    normalization: str = "NFC"
    seed: int = 42
    format_version: int = 2


@dataclass(frozen=True)
class TokenizerPipelineConfig:
    """Validated settings for the Phase 5 tokenizer build."""

    config_path: Path
    project_root: Path
    corpus_paths: tuple[Path, ...]
    output_directory: Path
    tokenizer_filename: str
    legacy_tokenizer_filename: str | None
    vocab_filename: str
    merges_filename: str
    tokenizer_config_filename: str
    special_tokens_filename: str
    metadata_filename: str
    statistics_filename: str
    vocab_size: int
    min_frequency: int
    max_training_bytes: int | None
    normalization: str
    special_tokens: tuple[str, ...]
    seed: int
    show_progress: bool
    statistics_sample_size: int | None

    @property
    def tokenizer_path(self) -> Path:
        return self.output_directory / self.tokenizer_filename

    @property
    def metadata_path(self) -> Path:
        return self.output_directory / self.metadata_filename


class CodeTokenizer:
    """Small adapter around a custom tokenizers byte-level BPE tokenizer."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.source_path: Path | None = None
        self._vocab_size_cache = int(self.tokenizer.get_vocab_size())
        self.special_tokens = self._detect_special_tokens()
        self._validate_special_ids()

    @classmethod
    def from_file(cls, path: Path | str) -> CodeTokenizer:
        """Load a tokenizer JSON file."""

        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise CodeTokenizerError("The tokenizers package is required.") from exc
        tokenizer_path = Path(path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        try:
            adapter = cls(Tokenizer.from_file(str(tokenizer_path)))
            adapter.source_path = tokenizer_path.resolve()
            return adapter
        except Exception as exc:
            raise CodeTokenizerError(f"Could not load tokenizer {tokenizer_path}: {exc}") from exc

    def save(self, path: Path | str) -> None:
        """Save tokenizer JSON atomically."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = Path(str(output_path) + ".partial")
        try:
            self.tokenizer.save(str(partial_path))
            os.replace(partial_path, output_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token IDs."""

        if not isinstance(text, str):
            raise CodeTokenizerError("text must be a string.")
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids = list(encoded.ids)
        self._validate_ids(ids)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""

        ids_list = list(ids)
        self._validate_ids(ids_list)
        return self.tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        """Encode a batch of strings."""

        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_ids: Iterable[Iterable[int]]) -> list[str]:
        """Decode a batch of ID sequences."""

        return [self.decode(ids) for ids in batch_ids]

    def token_to_id(self, token: str) -> int | None:
        """Return token ID or None."""

        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        """Return token string or None."""

        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise CodeTokenizerError("token_id must be an integer.")
        return self.tokenizer.id_to_token(token_id)

    @property
    def vocab_size(self) -> int:
        """Actual tokenizer vocabulary size."""

        return self._vocab_size_cache

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""

        return self._required_id(self.special_tokens[0])

    @property
    def unknown_token_id(self) -> int:
        """Unknown token ID."""

        return self._required_id(self.special_tokens[1])

    @property
    def bos_token_id(self) -> int:
        """BOS token ID."""

        return self._required_id(self.special_tokens[2])

    @property
    def eos_token_id(self) -> int:
        """EOS token ID."""

        return self._required_id(self.special_tokens[3])

    @property
    def mask_token_id(self) -> int | None:
        """Mask token ID, when available in the tokenizer profile."""

        token_id = self.token_to_id("<mask>")
        return None if token_id is None else int(token_id)

    @property
    def is_phase5(self) -> bool:
        """Whether this tokenizer contains the complete Phase 5 token set."""

        return self.special_tokens == SPECIAL_TOKENS

    def _required_id(self, token: str) -> int:
        token_id = self.token_to_id(token)
        if token_id is None:
            raise CodeTokenizerError(f"Missing required special token: {token}")
        return int(token_id)

    def _validate_special_ids(self) -> None:
        for expected_id, token in enumerate(self.special_tokens):
            actual = self.token_to_id(token)
            if actual != expected_id:
                raise CodeTokenizerError(f"{token} must have ID {expected_id}; found {actual}.")

    def _detect_special_tokens(self) -> tuple[str, ...]:
        if all(self.token_to_id(token) is not None for token in SPECIAL_TOKENS):
            return SPECIAL_TOKENS
        if all(self.token_to_id(token) is not None for token in LEGACY_SPECIAL_TOKENS):
            return LEGACY_SPECIAL_TOKENS
        required = ", ".join(SPECIAL_TOKENS)
        raise CodeTokenizerError(
            f"Tokenizer is missing the required special-token profile: {required}."
        )

    def _validate_ids(self, ids: Iterable[int]) -> None:
        vocab_size = self.vocab_size
        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise CodeTokenizerError("token IDs must be integers.")
            if token_id < 0 or token_id >= vocab_size:
                raise CodeTokenizerError("token ID is outside the tokenizer vocabulary.")


def discover_tokenizer_corpus(
    project_root: Path,
    *,
    preferred_paths: Iterable[Path] = (),
    train_pattern: str = "data/code_shards/train/*.jsonl.gz",
) -> list[Path]:
    """Discover a deterministic, repository-local tokenizer training corpus."""

    root = project_root.resolve()
    shards = _glob_paths(root, train_pattern)
    if shards:
        return shards

    discovered = [_resolve_corpus_path(root, path) for path in preferred_paths]
    for pattern in _FALLBACK_CORPUS_PATTERNS:
        discovered.extend(sorted(root.glob(pattern)))
    return _deduplicate_files(discovered)


def iter_code_texts_from_corpus(
    corpus_paths: Iterable[Path],
    *,
    max_training_bytes: int,
) -> tuple[Iterable[str], callable, list[str]]:
    """Return training texts from gzip JSONL, JSONL, JSON, or plain-text files."""

    paths = list(dict.fromkeys(Path(path) for path in corpus_paths))
    consumed = 0
    used_names: list[str] = []

    def iterator() -> Iterator[str]:
        nonlocal consumed
        for path in paths:
            used_this_file = False
            for text in _iter_corpus_file(path):
                if consumed >= max_training_bytes:
                    return
                text = _truncate_utf8(text, max_training_bytes - consumed)
                if not text:
                    continue
                if not used_this_file:
                    used_names.append(str(path))
                    used_this_file = True
                consumed += len(text.encode("utf-8"))
                yield text

    return iterator(), lambda: consumed, used_names


def iter_code_texts_from_shards(
    shard_paths: Iterable[Path],
    *,
    max_training_bytes: int,
) -> tuple[Iterable[str], callable, list[str]]:
    """Backward-compatible alias for reading gzip JSONL shard text."""

    return iter_code_texts_from_corpus(
        shard_paths,
        max_training_bytes=max_training_bytes,
    )


def train_byte_level_bpe_tokenizer(
    shard_paths: Iterable[Path],
    *,
    output_path: Path,
    metadata_path: Path,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    max_training_bytes: int | None = 500 * 1024 * 1024,
    corpus_root: Path | None = None,
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
    normalization: str = "NFC",
    seed: int = 42,
    show_progress: bool = True,
) -> CodeTokenizerMetadata:
    """Train a byte-level BPE tokenizer from supported local corpus files."""

    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.normalizers import NFC
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise CodeTokenizerError("The tokenizers package is required.") from exc
    paths = list(dict.fromkeys(Path(path) for path in shard_paths))
    if not paths:
        raise CodeTokenizerError("At least one tokenizer training corpus file is required.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer training corpus file not found: {path}")
    if vocab_size < len(special_tokens):
        raise CodeTokenizerError("vocab_size must include all special tokens.")
    if min_frequency <= 0:
        raise CodeTokenizerError("min_frequency must be greater than zero.")
    if max_training_bytes is not None and max_training_bytes <= 0:
        raise CodeTokenizerError("max_training_bytes must be greater than zero.")

    if normalization != "NFC":
        raise CodeTokenizerError("Only Unicode NFC normalization is supported.")
    if len(special_tokens) < 4 or len(set(special_tokens)) != len(special_tokens):
        raise CodeTokenizerError("special_tokens must contain at least four unique values.")

    random.seed(seed)
    tokenizer = Tokenizer(BPE(unk_token=special_tokens[1]))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    iterator, consumed_bytes, used_names = iter_code_texts_from_corpus(
        paths,
        max_training_bytes=max_training_bytes or 2**63 - 1,
    )

    LOGGER.info("Training ByteLevel BPE tokenizer from %d corpus files", len(paths))
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    if consumed_bytes() == 0:
        raise CodeTokenizerError("Tokenizer training corpus contains no usable text.")
    _pad_vocabulary(tokenizer, vocab_size)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{special_tokens[2]} $A {special_tokens[3]}",
        pair=f"{special_tokens[2]} $A {special_tokens[3]} $B:1 {special_tokens[3]}:1",
        special_tokens=[
            (special_tokens[2], int(tokenizer.token_to_id(special_tokens[2]))),
            (special_tokens[3], int(tokenizer.token_to_id(special_tokens[3]))),
        ],
    )
    LOGGER.info(
        "Tokenizer training complete: vocab=%d bytes=%d",
        tokenizer.get_vocab_size(),
        consumed_bytes(),
    )

    adapter = CodeTokenizer(tokenizer)
    adapter.save(output_path)

    training_corpus = tuple(_display_path(Path(name), corpus_root) for name in used_names)
    metadata = CodeTokenizerMetadata(
        tokenizer_type="byte_bpe",
        requested_vocab_size=vocab_size,
        actual_vocab_size=adapter.vocab_size,
        special_tokens=special_tokens,
        special_token_ids={
            token: adapter.token_to_id(token)
            for token in special_tokens
        },
        minimum_frequency=min_frequency,
        requested_sample_bytes=max_training_bytes,
        actual_sample_bytes=consumed_bytes(),
        training_shard_names=tuple(Path(name).name for name in used_names),
        training_corpus=training_corpus,
        creation_timestamp=datetime.now(UTC).isoformat(),
        normalization=normalization,
        seed=seed,
    )

    write_tokenizer_metadata(metadata, metadata_path)
    return metadata


def write_tokenizer_metadata(metadata: CodeTokenizerMetadata, path: Path) -> None:
    """Write tokenizer metadata atomically."""

    payload = {
        "format_version": metadata.format_version,
        "tokenizer_type": metadata.tokenizer_type,
        "vocab_size": metadata.actual_vocab_size,
        "requested_vocab_size": metadata.requested_vocab_size,
        "actual_vocab_size": metadata.actual_vocab_size,
        "special_tokens": list(metadata.special_tokens),
        "special_token_ids": metadata.special_token_ids,
        "minimum_frequency": metadata.minimum_frequency,
        "requested_sample_bytes": metadata.requested_sample_bytes,
        "actual_sample_bytes": metadata.actual_sample_bytes,
        "training_shard_names": list(metadata.training_shard_names),
        "training_corpus": list(metadata.training_corpus),
        "creation_timestamp": metadata.creation_timestamp,
        "normalization": metadata.normalization,
        "seed": metadata.seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(str(path) + ".partial")
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        os.replace(partial_path, path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def load_tokenizer_pipeline_config(
    path: Path | str = DEFAULT_TOKENIZER_CONFIG_PATH,
    *,
    project_root: Path | None = None,
) -> TokenizerPipelineConfig:
    """Load and validate the YAML configuration for Phase 5."""

    root = (project_root or Path.cwd()).resolve()
    config_path = Path(path)
    config_path = config_path if config_path.is_absolute() else root / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Tokenizer configuration not found: {config_path}")
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CodeTokenizerError(f"Invalid tokenizer YAML configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise CodeTokenizerError("Tokenizer configuration must be a YAML mapping.")
    try:
        corpus = raw["corpus"]
        tokenizer = raw["tokenizer"]
        artifacts = raw["artifacts"]
        training = raw.get("training", {})
        statistics = raw.get("statistics", {})
        corpus_paths = tuple(_resolve_corpus_path(root, Path(item)) for item in corpus["files"])
        max_bytes_raw = corpus.get("max_training_bytes")
        max_training_bytes = None if max_bytes_raw is None else int(max_bytes_raw)
        sample_size_raw = statistics.get("sample_size")
        sample_size = None if sample_size_raw is None else int(sample_size_raw)
        config = TokenizerPipelineConfig(
            config_path=config_path,
            project_root=root,
            corpus_paths=corpus_paths,
            output_directory=_resolve_corpus_path(root, Path(artifacts["output_directory"])),
            tokenizer_filename=str(artifacts.get("tokenizer", "tokenizer.json")),
            legacy_tokenizer_filename=_optional_filename(
                artifacts.get("legacy_tokenizer", "code_tokenizer.json")
            ),
            vocab_filename=str(artifacts.get("vocab", "vocab.json")),
            merges_filename=str(artifacts.get("merges", "merges.txt")),
            tokenizer_config_filename=str(
                artifacts.get("tokenizer_config", "tokenizer_config.json")
            ),
            special_tokens_filename=str(
                artifacts.get("special_tokens", "special_tokens.json")
            ),
            metadata_filename=str(artifacts.get("metadata", "tokenizer_metadata.json")),
            statistics_filename=str(
                artifacts.get("statistics", "tokenizer_statistics.json")
            ),
            vocab_size=int(tokenizer.get("vocab_size", 32_000)),
            min_frequency=int(tokenizer.get("min_frequency", 2)),
            max_training_bytes=max_training_bytes,
            normalization=str(tokenizer.get("normalization", "NFC")),
            special_tokens=tuple(str(item) for item in tokenizer["special_tokens"]),
            seed=int(training.get("seed", 42)),
            show_progress=bool(training.get("show_progress", True)),
            statistics_sample_size=sample_size,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CodeTokenizerError(f"Invalid tokenizer configuration: {exc}") from exc
    _validate_pipeline_config(config)
    return config


def build_tokenizer_artifacts(config: TokenizerPipelineConfig) -> CodeTokenizerMetadata:
    """Train, validate, and write the complete Phase 5 artifact set."""

    for path in config.corpus_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer corpus file not found: {path}")
    config.output_directory.mkdir(parents=True, exist_ok=True)
    metadata = train_byte_level_bpe_tokenizer(
        config.corpus_paths,
        output_path=config.tokenizer_path,
        metadata_path=config.metadata_path,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        max_training_bytes=config.max_training_bytes,
        corpus_root=config.project_root,
        special_tokens=config.special_tokens,
        normalization=config.normalization,
        seed=config.seed,
        show_progress=config.show_progress,
    )
    tokenizer = verify_code_tokenizer(config.tokenizer_path)
    _write_bpe_model_files(tokenizer, config)
    _write_phase5_configuration(tokenizer, metadata, config)
    if config.legacy_tokenizer_filename:
        _atomic_copy(
            config.tokenizer_path,
            config.output_directory / config.legacy_tokenizer_filename,
        )
    statistics = calculate_tokenizer_statistics(tokenizer, metadata, config)
    _atomic_write_json(
        config.output_directory / config.statistics_filename,
        statistics,
    )
    return metadata


def calculate_tokenizer_statistics(
    tokenizer: CodeTokenizer,
    metadata: CodeTokenizerMetadata,
    config: TokenizerPipelineConfig,
) -> dict[str, Any]:
    """Calculate corpus, compression, vocabulary, and validation statistics."""

    records = 0
    sampled_records = 0
    sampled_characters = 0
    characters = 0
    utf8_bytes = 0
    token_count = 0
    observed_ids: set[int] = set()
    for path in config.corpus_paths:
        for text_value in _iter_corpus_file(path):
            if not text_value:
                continue
            records += 1
            characters += len(text_value)
            utf8_bytes += len(text_value.encode("utf-8"))
            if (
                config.statistics_sample_size is not None
                and sampled_records >= config.statistics_sample_size
            ):
                continue
            ids = tokenizer.encode(text_value)
            sampled_records += 1
            sampled_characters += len(text_value)
            token_count += len(ids)
            observed_ids.update(ids)
    validation = validate_python_tokenization(tokenizer)
    artifact_names = (
        config.tokenizer_filename,
        config.vocab_filename,
        config.merges_filename,
        config.tokenizer_config_filename,
        config.special_tokens_filename,
        config.metadata_filename,
    )
    artifact_hashes = {
        name: tokenizer_file_hash(config.output_directory / name)
        for name in artifact_names
    }
    return {
        "format_version": 1,
        "creation_timestamp": metadata.creation_timestamp,
        "tokenizer_type": "ByteLevel BPE",
        "library": "tokenizers",
        "normalization": metadata.normalization,
        "deterministic_training": True,
        "seed": metadata.seed,
        "vocab_size": tokenizer.vocab_size,
        "requested_vocab_size": metadata.requested_vocab_size,
        "min_frequency": metadata.minimum_frequency,
        "special_tokens": metadata.special_token_ids,
        "training_corpus": list(metadata.training_corpus),
        "corpus_files": len(config.corpus_paths),
        "corpus_records": records,
        "corpus_characters": characters,
        "corpus_utf8_bytes": utf8_bytes,
        "training_bytes": metadata.actual_sample_bytes,
        "statistics_sample_records": sampled_records,
        "sample_tokens": token_count,
        "average_tokens_per_record": _safe_ratio(token_count, sampled_records),
        "average_characters_per_token": _safe_ratio(sampled_characters, token_count),
        "observed_vocabulary_tokens": len(observed_ids),
        "observed_vocabulary_percent": round(
            100.0 * _safe_ratio(len(observed_ids), tokenizer.vocab_size), 4
        ),
        "validation": validation,
        "artifact_sha256": artifact_hashes,
    }


def validate_python_tokenization(tokenizer: CodeTokenizer) -> list[dict[str, Any]]:
    """Run lossless Python and Unicode encode/decode checks."""

    samples = (
        "def add(left: int, right: int) -> int:\n    return left + right\n",
        "@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\n",
        "values = [item ** 2 for item in range(10) if item % 2 == 0]\n",
        "message = f\"வணக்கம், {name}! 🐍\"\n",
        "# decomposed Unicode is normalized: Cafe\u0301\n",
    )
    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        expected = unicodedata.normalize("NFC", sample)
        ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(ids)
        if decoded != expected:
            raise CodeTokenizerError(f"Python tokenization validation sample {index} failed.")
        results.append(
            {
                "sample": index,
                "characters": len(sample),
                "tokens": len(ids),
                "round_trip": True,
            }
        )
    special_ids = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
    if tokenizer.is_phase5 and special_ids != list(range(len(SPECIAL_TOKENS))):
        raise CodeTokenizerError("Phase 5 special token IDs are not deterministic.")
    return results


def _write_bpe_model_files(tokenizer: CodeTokenizer, config: TokenizerPipelineConfig) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".tokenizer-model-",
        dir=config.output_directory,
    ) as temporary_directory:
        saved = tokenizer.tokenizer.model.save(temporary_directory)
        by_suffix = {Path(path).suffix: Path(path) for path in saved}
        try:
            vocab_source = by_suffix[".json"]
            merges_source = by_suffix[".txt"]
        except KeyError as exc:
            raise CodeTokenizerError(
                "BPE model did not produce vocab.json and merges.txt."
            ) from exc
        _atomic_copy(vocab_source, config.output_directory / config.vocab_filename)
        _atomic_copy(merges_source, config.output_directory / config.merges_filename)


def _write_phase5_configuration(
    tokenizer: CodeTokenizer,
    metadata: CodeTokenizerMetadata,
    config: TokenizerPipelineConfig,
) -> None:
    roles = (
        "pad_token",
        "unk_token",
        "bos_token",
        "eos_token",
        "mask_token",
        "instruction_token",
        "input_token",
        "output_token",
    )
    special_payload = {
        role: {
            "content": token,
            "id": tokenizer.token_to_id(token),
            "special": True,
        }
        for role, token in zip_strict(roles, config.special_tokens)
    }
    _atomic_write_json(
        config.output_directory / config.special_tokens_filename,
        special_payload,
    )
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "tokenizer_file": config.tokenizer_filename,
        "tokenizer_type": "ByteLevel BPE",
        "vocab_size": tokenizer.vocab_size,
        "min_frequency": config.min_frequency,
        "normalization": config.normalization,
        "add_prefix_space": False,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
        "add_eos_token": False,
        "pad_token": config.special_tokens[0],
        "unk_token": config.special_tokens[1],
        "bos_token": config.special_tokens[2],
        "eos_token": config.special_tokens[3],
        "mask_token": config.special_tokens[4],
        "additional_special_tokens": list(config.special_tokens[5:]),
        "seed": config.seed,
        "training_corpus": list(metadata.training_corpus),
        "creation_timestamp": metadata.creation_timestamp,
    }
    _atomic_write_json(
        config.output_directory / config.tokenizer_config_filename,
        tokenizer_config,
    )


def _validate_pipeline_config(config: TokenizerPipelineConfig) -> None:
    if not config.corpus_paths:
        raise CodeTokenizerError("Tokenizer corpus.files must not be empty.")
    if config.vocab_size < 256 + len(config.special_tokens):
        raise CodeTokenizerError("vocab_size is too small for byte-level BPE training.")
    if config.min_frequency <= 0:
        raise CodeTokenizerError("min_frequency must be greater than zero.")
    if config.max_training_bytes is not None and config.max_training_bytes <= 0:
        raise CodeTokenizerError("max_training_bytes must be greater than zero or null.")
    if config.normalization != "NFC":
        raise CodeTokenizerError("tokenizer.normalization must be NFC.")
    if config.special_tokens != SPECIAL_TOKENS:
        raise CodeTokenizerError(
            "tokenizer.special_tokens must contain the eight Phase 5 tokens in order."
        )
    if config.statistics_sample_size is not None and config.statistics_sample_size <= 0:
        raise CodeTokenizerError("statistics.sample_size must be positive or null.")
    filenames = (
        config.tokenizer_filename,
        config.vocab_filename,
        config.merges_filename,
        config.tokenizer_config_filename,
        config.special_tokens_filename,
        config.metadata_filename,
        config.statistics_filename,
    )
    if config.legacy_tokenizer_filename:
        filenames += (config.legacy_tokenizer_filename,)
    if len(set(filenames)) != len(filenames):
        raise CodeTokenizerError("Tokenizer artifact filenames must be unique.")
    if any(Path(name).name != name or not name for name in filenames):
        raise CodeTokenizerError("Tokenizer artifacts must be plain filenames.")


def _optional_filename(value: Any) -> str | None:
    if value is None or value is False:
        return None
    filename = str(value).strip()
    return filename or None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(f"{path}.partial")
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        os.replace(partial_path, path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(f"{destination}.partial")
    try:
        shutil.copyfile(source, partial_path)
        os.replace(partial_path, destination)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def ensure_code_tokenizer(
    *,
    tokenizer_path: Path,
    metadata_path: Path,
    project_root: Path,
    vocab_size: int,
    preferred_corpus_paths: Iterable[Path] = (),
    train_pattern: str = "data/code_shards/train/*.jsonl.gz",
    min_frequency: int = 2,
    max_training_bytes: int = 500 * 1024 * 1024,
) -> CodeTokenizer:
    """Build missing tokenizer artifacts from the repository's available corpus."""

    tokenizer_missing = not tokenizer_path.is_file()
    metadata_missing = not metadata_path.is_file()
    if not tokenizer_missing and not metadata_missing:
        return CodeTokenizer.from_file(tokenizer_path)

    if tokenizer_missing:
        print("Tokenizer not found. Building tokenizer...")
    else:
        print("Tokenizer metadata not found. Rebuilding tokenizer...")

    phase5_config_path = project_root / DEFAULT_TOKENIZER_CONFIG_PATH
    if phase5_config_path.is_file():
        phase5_config = load_tokenizer_pipeline_config(
            phase5_config_path,
            project_root=project_root,
        )
        if tokenizer_path.resolve() == phase5_config.tokenizer_path.resolve():
            phase5_config = replace(
                phase5_config,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                max_training_bytes=max_training_bytes,
            )
            build_tokenizer_artifacts(phase5_config)
            tokenizer = verify_code_tokenizer(tokenizer_path)
            print("✓ Tokenizer built successfully")
            return tokenizer

    corpus = discover_tokenizer_corpus(
        project_root,
        preferred_paths=preferred_corpus_paths,
        train_pattern=train_pattern,
    )
    if not corpus:
        raise CodeTokenizerError("No supported tokenizer training corpus was found.")
    train_byte_level_bpe_tokenizer(
        corpus,
        output_path=tokenizer_path,
        metadata_path=metadata_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_training_bytes=max_training_bytes,
        corpus_root=project_root,
    )
    tokenizer = verify_code_tokenizer(tokenizer_path)
    print("✓ Tokenizer built successfully")
    return tokenizer


def verify_code_tokenizer(
    path: Path,
    sample: str = DEFAULT_VERIFICATION_SAMPLE,
) -> CodeTokenizer:
    """Load a tokenizer and verify an exact byte-level encode/decode round trip."""

    tokenizer = CodeTokenizer.from_file(path)
    decoded = tokenizer.decode(tokenizer.encode(sample))
    expected = unicodedata.normalize("NFC", sample) if tokenizer.is_phase5 else sample
    if decoded != expected:
        raise CodeTokenizerError("Tokenizer encode/decode verification failed.")
    return tokenizer


def _iter_corpus_file(path: Path) -> Iterator[str]:
    suffixes = path.suffixes
    if suffixes[-2:] == [".jsonl", ".gz"]:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            yield from _iter_jsonl(file, path)
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            yield from _iter_jsonl(file, path)
        return
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload: Any = json.load(file)
        yield from _texts_from_json(payload)
        return
    yield path.read_text(encoding="utf-8")


def _iter_jsonl(file: Iterable[str], path: Path) -> Iterator[str]:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodeTokenizerError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        yield from _texts_from_json(payload)


def _texts_from_json(payload: Any) -> Iterator[str]:
    if isinstance(payload, list):
        for item in payload:
            yield from _texts_from_json(item)
        return
    if isinstance(payload, str):
        if payload:
            yield payload
        return
    if not isinstance(payload, dict):
        return
    for container_key in ("examples", "data", "records"):
        if isinstance(payload.get(container_key), list):
            yield from _texts_from_json(payload[container_key])
            return
    text = _text_from_record(payload)
    if text:
        yield text


def _text_from_record(record: dict[str, Any]) -> str:
    text = record.get("text")
    if isinstance(text, str) and text:
        return text
    instruction = record.get("instruction")
    input_text = record.get("input")
    response = record.get("output", record.get("response"))
    if isinstance(instruction, str) and instruction.strip():
        result = f"<instruction>\n{instruction.strip()}\n"
        if isinstance(input_text, str) and input_text.strip():
            result += f"<input>\n{input_text.strip()}\n"
        if isinstance(response, str) and response.strip():
            result += f"<output>\n{response.strip()}"
        return result
    return response if isinstance(response, str) else ""


def _truncate_utf8(text: str, byte_limit: int) -> str:
    if byte_limit <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def _pad_vocabulary(tokenizer: Any, requested_vocab_size: int) -> None:
    missing = requested_vocab_size - int(tokenizer.get_vocab_size())
    if missing <= 0:
        return
    start = int(tokenizer.get_vocab_size())
    reserved = [f"<|reserved_token_{index:05d}|>" for index in range(start, start + missing)]
    added = tokenizer.add_tokens(reserved)
    if added != missing or tokenizer.get_vocab_size() != requested_vocab_size:
        raise CodeTokenizerError("Could not create the requested tokenizer vocabulary size.")


def _resolve_corpus_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _glob_paths(root: Path, pattern: str) -> list[Path]:
    search_pattern = pattern if Path(pattern).is_absolute() else str(root / pattern)
    return sorted(Path(path) for path in glob.glob(search_pattern, recursive=True))


def _deduplicate_files(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def _display_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def tokenizer_file_hash(path: Path) -> str:
    """Return SHA-256 hash of a tokenizer file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CodeTokenizer",
    "CodeTokenizerError",
    "CodeTokenizerMetadata",
    "TokenizerPipelineConfig",
    "DEFAULT_LEGACY_TOKENIZER_PATH",
    "DEFAULT_TOKENIZER_CONFIG_PATH",
    "DEFAULT_TOKENIZER_METADATA_PATH",
    "DEFAULT_TOKENIZER_PATH",
    "SPECIAL_TOKENS",
    "LEGACY_SPECIAL_TOKENS",
    "build_tokenizer_artifacts",
    "calculate_tokenizer_statistics",
    "discover_tokenizer_corpus",
    "ensure_code_tokenizer",
    "iter_code_texts_from_corpus",
    "load_tokenizer_pipeline_config",
    "tokenizer_file_hash",
    "train_byte_level_bpe_tokenizer",
    "verify_code_tokenizer",
    "validate_python_tokenization",
    "write_tokenizer_metadata",
]
