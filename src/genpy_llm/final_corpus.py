"""Final local corpus builder for GenPy continued pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

import yaml

from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash
from genpy_llm.documentation_corpus import clean_documentation_text
from genpy_llm.python_corpus_builder import count_words, normalize_text
from genpy_llm.sequence_packer import (
    PackedSequence,
    SequencePacker,
    SequencePackingConfig,
    prepare_document_tokens,
)
from genpy_llm.shard_builder import (
    SequenceShardStatistics,
    SequenceShardWriter,
    prepare_sequence_output,
    write_sequence_shard_index,
)

UTC = timezone.utc

DEFAULT_CONFIG_PATH = Path("configs/final_corpus.yaml")
DEFAULT_SOURCE_DIRECTORIES = (
    "github",
    "docs",
    "peps",
    "tutorials",
    "cleaned",
    "cleaned_docs",
)
DEFAULT_ALLOWED_EXTENSIONS = (".py", ".pyi", ".md", ".rst", ".html", ".txt")
DEFAULT_IGNORED_DIRECTORIES = (
    ".git",
    ".github",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "ENV",
    "build",
    "dist",
    "_build",
    "site",
    "final_corpus",
    "packed",
    "packed_docs",
    "metadata",
    "statistics",
)
DEFAULT_IGNORED_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".exe",
    ".dll",
    ".so",
    ".bin",
)
PYTHON_EXTENSIONS = {".py", ".pyi"}
DOCUMENTATION_EXTENSIONS = {".md", ".rst", ".html", ".txt"}
PACKING_STRATEGIES = {"packed", "sliding_window"}


class FinalCorpusError(RuntimeError):
    """Raised when the final corpus builder cannot continue safely."""


@dataclass(frozen=True)
class FinalCorpusPackingConfig:
    """Packed-shard generation settings for the final corpus."""

    sequence_length: int = 1025
    overlap: int = 0
    packing_strategy: str = "packed"
    max_tokens_per_shard: int = 10_000_000
    shard_prefix: str = "final_corpus"
    add_bos: bool = False
    add_eos: bool = True
    document_boundary: str = "eos"
    pad_final_sequence: bool = False

    @property
    def context_length(self) -> int:
        """Number of input tokens used by the existing GenPy training contract."""

        return self.sequence_length - 1


@dataclass(frozen=True)
class FinalCorpusConfig:
    """Validated final local corpus configuration."""

    project_root: Path
    corpus_root: Path
    output_directory: Path
    source_directories: tuple[Path, ...]
    tokenizer_path: Path
    min_file_size: int = 80
    max_file_size: int = 2_000_000
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    ignored_directories: tuple[str, ...] = DEFAULT_IGNORED_DIRECTORIES
    ignored_suffixes: tuple[str, ...] = DEFAULT_IGNORED_SUFFIXES
    deduplication: bool = True
    comment_removal: bool = False
    target_training_tokens: int | None = None
    packing: FinalCorpusPackingConfig = field(default_factory=FinalCorpusPackingConfig)

    @property
    def cleaned_directory(self) -> Path:
        return self.output_directory / "cleaned"

    @property
    def packed_directory(self) -> Path:
        return self.output_directory / "packed"

    @property
    def metadata_directory(self) -> Path:
        return self.output_directory / "metadata"

    @property
    def statistics_directory(self) -> Path:
        return self.output_directory / "statistics"


@dataclass(frozen=True)
class FinalSourceFile:
    """One supported source file discovered under an input corpus directory."""

    path: Path
    relative_path: PurePosixPath
    source_section: str
    repository: str | None
    extension: str


@dataclass(frozen=True)
class FinalCorpusDocument:
    """One accepted, cleaned final-corpus document."""

    source_path: Path
    relative_path: PurePosixPath
    cleaned_path: Path
    source_section: str
    repository: str | None
    extension: str
    content_type: str
    size_bytes: int
    character_count: int
    word_count: int
    line_count: int
    token_count: int
    content_sha256: str
    normalized_sha256: str


@dataclass(frozen=True)
class FinalCorpusRejection:
    """One source file rejected by validation or deduplication."""

    path: Path
    relative_path: str
    reason: str
    size_bytes: int = 0


@dataclass(frozen=True)
class FinalCorpusBuildResult:
    """Artifacts produced by a final corpus build."""

    total_input_files: int
    processed_files: int
    skipped_files: int
    duplicates_removed: int
    total_tokens: int
    packed_sequences: int
    statistics_path: Path
    manifest_path: Path
    shard_index_path: Path
    output_directory: Path


def load_final_corpus_config(path: Path | str = DEFAULT_CONFIG_PATH) -> FinalCorpusConfig:
    """Load and validate YAML configuration for the final local corpus builder."""

    config_path = Path(path)
    default_project_root = (
        config_path.resolve().parents[1]
        if config_path.parent.name == "configs"
        else Path.cwd()
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise FinalCorpusError("Final corpus config must be a YAML mapping.")
    section = _mapping(payload.get("final_corpus", payload), "final_corpus")
    root = (
        _resolve(config_path.parent.resolve(), Path(payload["project_root"]))
        if "project_root" in payload
        else default_project_root.resolve()
    )
    corpus_root = _resolve(root, Path(section.get("corpus_root", "python_corpus")))
    output_directory = _resolve(
        corpus_root,
        Path(section.get("output_directory", "final_corpus")),
    )
    source_directories = tuple(
        _resolve(corpus_root, Path(value))
        for value in _strings(
            section.get("source_directories", DEFAULT_SOURCE_DIRECTORIES),
            "source_directories",
        )
    )
    packing_section = _mapping(section.get("packing", {}), "final_corpus.packing")
    sequence_length = _sequence_length(packing_section)
    overlap = _non_negative_int(packing_section.get("overlap", 0), "overlap")
    if overlap >= sequence_length:
        raise FinalCorpusError("packing.overlap must be smaller than sequence_length.")
    strategy = str(packing_section.get("packing_strategy", "packed"))
    if strategy not in PACKING_STRATEGIES:
        raise FinalCorpusError(
            f"packing.packing_strategy must be one of {sorted(PACKING_STRATEGIES)}."
        )
    if strategy == "packed" and overlap:
        raise FinalCorpusError("packing.overlap requires packing_strategy: sliding_window.")

    return FinalCorpusConfig(
        project_root=root,
        corpus_root=corpus_root,
        output_directory=output_directory,
        source_directories=source_directories,
        tokenizer_path=_resolve(
            root,
            Path(section.get("tokenizer", "data/tokenizer/tokenizer.json")),
        ),
        min_file_size=_non_negative_int(
            section.get("min_file_size", section.get("minimum_size", 80)),
            "min_file_size",
        ),
        max_file_size=_positive_int(
            section.get("max_file_size", section.get("maximum_size", 2_000_000)),
            "max_file_size",
        ),
        allowed_extensions=_extensions(
            section.get("allowed_extensions", DEFAULT_ALLOWED_EXTENSIONS)
        ),
        ignored_directories=_strings(
            section.get("ignored_directories", DEFAULT_IGNORED_DIRECTORIES),
            "ignored_directories",
        ),
        ignored_suffixes=_extensions(section.get("ignored_suffixes", DEFAULT_IGNORED_SUFFIXES)),
        deduplication=bool(section.get("deduplication", True)),
        comment_removal=bool(section.get("comment_removal", False)),
        target_training_tokens=_optional_positive_int(
            section.get("target_training_tokens"),
            "target_training_tokens",
        ),
        packing=FinalCorpusPackingConfig(
            sequence_length=sequence_length,
            overlap=overlap,
            packing_strategy=strategy,
            max_tokens_per_shard=_positive_int(
                packing_section.get("max_tokens_per_shard", 10_000_000),
                "max_tokens_per_shard",
            ),
            shard_prefix=_filename(packing_section.get("shard_prefix", "final_corpus")),
            add_bos=bool(packing_section.get("add_bos", False)),
            add_eos=bool(packing_section.get("add_eos", True)),
            document_boundary=str(packing_section.get("document_boundary", "eos")),
            pad_final_sequence=bool(packing_section.get("pad_final_sequence", False)),
        ),
    )


def ensure_final_corpus_folders(config: FinalCorpusConfig) -> None:
    """Create final corpus output folders."""

    for directory in (
        config.output_directory,
        config.cleaned_directory,
        config.packed_directory,
        config.metadata_directory,
        config.statistics_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def scan_final_corpus_sources(config: FinalCorpusConfig) -> list[FinalSourceFile]:
    """Recursively scan all configured existing corpus folders."""

    ignored_dirs = set(config.ignored_directories)
    ignored_suffixes = set(config.ignored_suffixes)
    allowed_extensions = set(config.allowed_extensions)
    sources: list[FinalSourceFile] = []
    seen_paths: set[Path] = set()
    output_root = config.output_directory.resolve()
    for source_directory in config.source_directories:
        if not source_directory.exists():
            continue
        if not source_directory.is_dir():
            continue
        for root, dirnames, filenames in os.walk(source_directory):
            current = Path(root)
            try:
                current_resolved = current.resolve()
            except OSError:
                continue
            if current_resolved == output_root or output_root in current_resolved.parents:
                dirnames[:] = []
                continue
            dirnames[:] = sorted(name for name in dirnames if name not in ignored_dirs)
            for filename in sorted(filenames):
                path = current / filename
                suffix = path.suffix.lower()
                if suffix not in allowed_extensions or suffix in ignored_suffixes:
                    continue
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                relative = PurePosixPath(path.relative_to(config.corpus_root).as_posix())
                if any(part in ignored_dirs for part in relative.parts):
                    continue
                sources.append(
                    FinalSourceFile(
                        path=path,
                        relative_path=relative,
                        source_section=relative.parts[0] if relative.parts else "",
                        repository=_repository_name(relative),
                        extension=suffix,
                    )
                )
    return sources


def process_final_corpus(
    sources: list[FinalSourceFile],
    config: FinalCorpusConfig,
    tokenizer: CodeTokenizer,
) -> tuple[list[FinalCorpusDocument], list[FinalCorpusRejection]]:
    """Validate, clean, deduplicate, tokenize, and write cleaned final corpus files."""

    documents: list[FinalCorpusDocument] = []
    rejected: list[FinalCorpusRejection] = []
    seen_hashes: set[str] = set()
    config.cleaned_directory.mkdir(parents=True, exist_ok=True)

    for source in sources:
        try:
            size = source.path.stat().st_size
        except OSError:
            rejected.append(_reject(source, "stat_failed"))
            continue
        if size == 0:
            rejected.append(_reject(source, "empty_file", size))
            continue
        if size < config.min_file_size:
            rejected.append(_reject(source, "too_small", size))
            continue
        if size > config.max_file_size:
            rejected.append(_reject(source, "too_large", size))
            continue
        try:
            raw_text = source.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            rejected.append(_reject(source, "invalid_utf8", size))
            continue
        except OSError:
            rejected.append(_reject(source, "read_failed", size))
            continue

        cleaned = clean_final_text(raw_text, source.extension, config)
        if not cleaned.strip():
            rejected.append(_reject(source, "empty_after_cleaning", size))
            continue
        cleaned_size = len(cleaned.encode("utf-8"))
        if cleaned_size < config.min_file_size:
            rejected.append(_reject(source, "too_small_after_cleaning", cleaned_size))
            continue
        normalized_sha = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if config.deduplication and normalized_sha in seen_hashes:
            rejected.append(_reject(source, "duplicate", size))
            continue
        seen_hashes.add(normalized_sha)
        token_ids = tokenizer.encode(cleaned)
        cleaned_path = _cleaned_path(config, source.relative_path, source.extension)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        with cleaned_path.open("w", encoding="utf-8", newline="\n") as cleaned_file:
            cleaned_file.write(cleaned)
        documents.append(
            FinalCorpusDocument(
                source_path=source.path,
                relative_path=source.relative_path,
                cleaned_path=cleaned_path,
                source_section=source.source_section,
                repository=source.repository,
                extension=source.extension,
                content_type=_content_type(source.extension),
                size_bytes=cleaned_size,
                character_count=len(cleaned),
                word_count=count_words(cleaned),
                line_count=cleaned.count("\n") + (1 if cleaned else 0),
                token_count=len(token_ids),
                content_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                normalized_sha256=normalized_sha,
            )
        )
    return documents, rejected


def clean_final_text(text: str, extension: str, config: FinalCorpusConfig) -> str:
    """Normalize one source document while preserving training content."""

    text = unicodedata.normalize("NFC", text)
    if extension in PYTHON_EXTENSIONS:
        return normalize_text(
            text,
            preserve_comments=not config.comment_removal,
            extension=extension,
        )
    if extension in DOCUMENTATION_EXTENSIONS:
        return clean_documentation_text(text, extension)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip() + "\n"


def write_final_metadata(
    documents: list[FinalCorpusDocument],
    rejected: list[FinalCorpusRejection],
    config: FinalCorpusConfig,
) -> Path:
    """Write accepted and rejected final corpus metadata."""

    config.metadata_directory.mkdir(parents=True, exist_ok=True)
    manifest = config.metadata_directory / "final_manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as file:
        for document in documents:
            file.write(json.dumps(_document_record(document), sort_keys=True) + "\n")
    rejected_path = config.metadata_directory / "final_rejected.jsonl"
    with rejected_path.open("w", encoding="utf-8", newline="\n") as file:
        for item in rejected:
            file.write(json.dumps(_rejection_record(item), sort_keys=True) + "\n")
    return manifest


def pack_final_corpus(
    documents: list[FinalCorpusDocument],
    tokenizer: CodeTokenizer,
    config: FinalCorpusConfig,
    manifest_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Pack final corpus documents into train-ready fixed-length uint16 shards."""

    index_path = config.packed_directory / "index.json"
    if force:
        prepare_sequence_output(
            config.packed_directory,
            config.packing.shard_prefix,
            [index_path],
        )
    config.packed_directory.mkdir(parents=True, exist_ok=True)
    packing_config = SequencePackingConfig(
        context_length=config.packing.context_length,
        add_bos=config.packing.add_bos,
        add_eos=config.packing.add_eos,
        document_boundary=config.packing.document_boundary,
        pad_final_sequence=config.packing.pad_final_sequence,
    )
    writer = SequenceShardWriter(
        config.packed_directory,
        max_tokens_per_shard=config.packing.max_tokens_per_shard,
        context_length=config.packing.context_length,
        prefix=config.packing.shard_prefix,
    )
    try:
        if config.packing.packing_strategy == "sliding_window":
            shard_stats = _write_sliding_window_sequences(
                documents,
                tokenizer,
                config,
                packing_config,
                writer,
            )
        else:
            shard_stats = _write_packed_sequences(documents, tokenizer, packing_config, writer)
    except Exception:
        writer.abort()
        raise
    write_sequence_shard_index(
        index_path,
        shard_stats,
        tokenizer_path=config.tokenizer_path,
        tokenizer_sha256=tokenizer_file_hash(config.tokenizer_path),
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        context_length=config.packing.context_length,
        source_manifest=manifest_path,
        creation_timestamp=datetime.now(UTC).isoformat(),
        build_fingerprint=_build_fingerprint(documents, config),
    )
    return index_path


def build_final_statistics(
    total_input_files: int,
    documents: list[FinalCorpusDocument],
    rejected: list[FinalCorpusRejection],
    config: FinalCorpusConfig,
    shard_index: dict[str, Any],
) -> dict[str, Any]:
    """Build the requested final corpus statistics report."""

    token_counts = [document.token_count for document in documents]
    sizes = [document.size_bytes for document in documents]
    reasons = Counter(item.reason for item in rejected)
    packed_token_count = int(shard_index.get("token_count", 0))
    packed_sequences = int(shard_index.get("sequence_count", 0))
    total_tokens = sum(token_counts)
    largest = sorted(documents, key=lambda document: document.size_bytes, reverse=True)[:10]
    smallest = sorted(documents, key=lambda document: document.size_bytes)[:10]
    estimated_epochs = (
        round(config.target_training_tokens / total_tokens, 6)
        if config.target_training_tokens and total_tokens
        else (1.0 if total_tokens else 0.0)
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_root": str(config.corpus_root),
        "output_directory": str(config.output_directory),
        "source_directories": [str(path) for path in config.source_directories if path.exists()],
        "total_input_files": total_input_files,
        "processed_files": len(documents),
        "skipped_files": len(rejected),
        "duplicates_removed": reasons.get("duplicate", 0),
        "total_code_files": sum(document.content_type == "python_code" for document in documents),
        "total_documentation_files": sum(
            document.content_type == "documentation" for document in documents
        ),
        "total_characters": sum(document.character_count for document in documents),
        "total_words": sum(document.word_count for document in documents),
        "total_tokens": total_tokens,
        "packed_sequences": packed_sequences,
        "packed_tokens": packed_token_count,
        "average_sequence_length": round(packed_token_count / packed_sequences, 6)
        if packed_sequences
        else 0.0,
        "average_file_size": round(sum(sizes) / len(sizes), 6) if sizes else 0.0,
        "median_file_size": int(median(sizes)) if sizes else 0,
        "average_tokens_per_document": round(total_tokens / len(documents), 6)
        if documents
        else 0.0,
        "largest_files": [_file_summary(document) for document in largest],
        "smallest_files": [_file_summary(document) for document in smallest],
        "estimated_epochs": estimated_epochs,
        "source_sections": dict(Counter(document.source_section for document in documents)),
        "extensions": dict(Counter(document.extension for document in documents)),
        "skip_reasons": dict(sorted(reasons.items())),
        "tokenizer": str(config.tokenizer_path),
        "tokenizer_sha256": tokenizer_file_hash(config.tokenizer_path),
        "packing_strategy": config.packing.packing_strategy,
        "sequence_length": config.packing.sequence_length,
        "overlap": config.packing.overlap,
        "packing": shard_index,
    }


def write_final_statistics(statistics: dict[str, Any], config: FinalCorpusConfig) -> Path:
    """Write JSON and Markdown final corpus statistics reports."""

    config.statistics_directory.mkdir(parents=True, exist_ok=True)
    json_path = config.statistics_directory / "final_statistics.json"
    json_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = config.statistics_directory / "final_statistics.md"
    markdown_path.write_text(_statistics_markdown(statistics), encoding="utf-8")
    return json_path


def run_final_corpus_pipeline(
    config: FinalCorpusConfig,
    *,
    force: bool = False,
) -> FinalCorpusBuildResult:
    """Run the full local-only final corpus pipeline."""

    if force and config.output_directory.exists():
        shutil.rmtree(config.output_directory)
    ensure_final_corpus_folders(config)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)
    sources = scan_final_corpus_sources(config)
    documents, rejected = process_final_corpus(sources, config, tokenizer)
    manifest_path = write_final_metadata(documents, rejected, config)
    shard_index_path = pack_final_corpus(documents, tokenizer, config, manifest_path, force=force)
    shard_index = json.loads(shard_index_path.read_text(encoding="utf-8"))
    statistics = build_final_statistics(len(sources), documents, rejected, config, shard_index)
    statistics_path = write_final_statistics(statistics, config)
    return FinalCorpusBuildResult(
        total_input_files=len(sources),
        processed_files=statistics["processed_files"],
        skipped_files=statistics["skipped_files"],
        duplicates_removed=statistics["duplicates_removed"],
        total_tokens=statistics["total_tokens"],
        packed_sequences=statistics["packed_sequences"],
        statistics_path=statistics_path,
        manifest_path=manifest_path,
        shard_index_path=shard_index_path,
        output_directory=config.output_directory,
    )


def run_final_corpus_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for final corpus building."""

    parser = argparse.ArgumentParser(description="Build the final local GenPy corpus.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = run_final_corpus_pipeline(load_final_corpus_config(args.config), force=args.force)
    print(f"Total input files: {result.total_input_files}")
    print(f"Processed files: {result.processed_files}")
    print(f"Skipped files: {result.skipped_files}")
    print(f"Duplicates removed: {result.duplicates_removed}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Packed sequences: {result.packed_sequences}")
    print(f"Statistics: {result.statistics_path}")
    print(f"Packed index: {result.shard_index_path}")
    return 0


def _write_packed_sequences(
    documents: list[FinalCorpusDocument],
    tokenizer: CodeTokenizer,
    packing_config: SequencePackingConfig,
    writer: SequenceShardWriter,
) -> SequenceShardStatistics:
    packer = SequencePacker(packing_config, tokenizer.pad_token_id)
    for document in documents:
        prepared = _prepared_document(document, tokenizer, packing_config)
        for sequence in packer.add_document(prepared, _document_record(document)):
            writer.write_sequence(sequence)
    for sequence in packer.finish():
        writer.write_sequence(sequence)
    return writer.close()


def _write_sliding_window_sequences(
    documents: list[FinalCorpusDocument],
    tokenizer: CodeTokenizer,
    config: FinalCorpusConfig,
    packing_config: SequencePackingConfig,
    writer: SequenceShardWriter,
) -> SequenceShardStatistics:
    sequence_index = 0
    sequence_length = config.packing.sequence_length
    step = sequence_length - config.packing.overlap
    for document in documents:
        prepared = _prepared_document(document, tokenizer, packing_config)
        if len(prepared) < sequence_length:
            if not packing_config.pad_final_sequence:
                continue
            padding = sequence_length - len(prepared)
            writer.write_sequence(
                PackedSequence(
                    token_ids=[*prepared, *([tokenizer.pad_token_id] * padding)],
                    sequence_index=sequence_index,
                    document_offsets=[
                        _document_record(document)
                        | {"sequence_token_start": 0, "sequence_token_end": len(prepared)}
                    ],
                    padding_tokens=padding,
                )
            )
            sequence_index += 1
            continue
        for start in range(0, len(prepared) - sequence_length + 1, step):
            writer.write_sequence(
                PackedSequence(
                    token_ids=prepared[start : start + sequence_length],
                    sequence_index=sequence_index,
                    document_offsets=[
                        _document_record(document)
                        | {
                            "sequence_token_start": 0,
                            "sequence_token_end": sequence_length,
                            "document_token_start": start,
                            "document_token_end": start + sequence_length,
                        }
                    ],
                    padding_tokens=0,
                )
            )
            sequence_index += 1
    return writer.close()


def _prepared_document(
    document: FinalCorpusDocument,
    tokenizer: CodeTokenizer,
    packing_config: SequencePackingConfig,
) -> list[int]:
    token_ids = tokenizer.encode(document.cleaned_path.read_text(encoding="utf-8"))
    return prepare_document_tokens(
        token_ids,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        config=packing_config,
    )


def _cleaned_path(
    config: FinalCorpusConfig,
    relative_path: PurePosixPath,
    extension: str,
) -> Path:
    if extension == ".html":
        target = relative_path.with_name(f"{relative_path.name}.txt")
    else:
        target = relative_path
    return config.cleaned_directory / target


def _reject(
    source: FinalSourceFile,
    reason: str,
    size_bytes: int = 0,
) -> FinalCorpusRejection:
    return FinalCorpusRejection(
        path=source.path,
        relative_path=source.relative_path.as_posix(),
        reason=reason,
        size_bytes=size_bytes,
    )


def _document_record(document: FinalCorpusDocument) -> dict[str, Any]:
    return {
        "source_path": str(document.source_path),
        "stored_path": document.relative_path.as_posix(),
        "cleaned_path": str(document.cleaned_path),
        "source_section": document.source_section,
        "source_type": document.content_type,
        "source_id": document.source_section,
        "repository": document.repository,
        "extension": document.extension,
        "size_bytes": document.size_bytes,
        "character_count": document.character_count,
        "word_count": document.word_count,
        "line_count": document.line_count,
        "token_count": document.token_count,
        "content_sha256": document.content_sha256,
        "normalized_sha256": document.normalized_sha256,
    }


def _rejection_record(item: FinalCorpusRejection) -> dict[str, Any]:
    return {
        "source_path": str(item.path),
        "stored_path": item.relative_path,
        "reason": item.reason,
        "size_bytes": item.size_bytes,
    }


def _file_summary(document: FinalCorpusDocument) -> dict[str, Any]:
    return {
        "path": document.relative_path.as_posix(),
        "bytes": document.size_bytes,
        "tokens": document.token_count,
        "type": document.content_type,
    }


def _statistics_markdown(statistics: dict[str, Any]) -> str:
    largest = "\n".join(
        f"- `{item['path']}`: {item['bytes']} bytes, {item['tokens']} tokens"
        for item in statistics["largest_files"]
    )
    largest = largest or "- None"
    smallest = "\n".join(
        f"- `{item['path']}`: {item['bytes']} bytes, {item['tokens']} tokens"
        for item in statistics["smallest_files"]
    )
    smallest = smallest or "- None"
    return (
        "# Final Corpus Statistics\n\n"
        f"- Total input files: {statistics['total_input_files']}\n"
        f"- Processed files: {statistics['processed_files']}\n"
        f"- Skipped files: {statistics['skipped_files']}\n"
        f"- Duplicates removed: {statistics['duplicates_removed']}\n"
        f"- Code files: {statistics['total_code_files']}\n"
        f"- Documentation files: {statistics['total_documentation_files']}\n"
        f"- Total characters: {statistics['total_characters']}\n"
        f"- Total words: {statistics['total_words']}\n"
        f"- Total tokens: {statistics['total_tokens']}\n"
        f"- Packed sequences: {statistics['packed_sequences']}\n"
        f"- Average sequence length: {statistics['average_sequence_length']}\n"
        f"- Estimated epochs: {statistics['estimated_epochs']}\n\n"
        "## Largest Files\n\n"
        f"{largest}\n\n"
        "## Smallest Files\n\n"
        f"{smallest}\n"
    )


def _build_fingerprint(documents: list[FinalCorpusDocument], config: FinalCorpusConfig) -> str:
    digest = hashlib.sha256()
    digest.update(str(config.corpus_root).encode("utf-8"))
    digest.update(str(config.output_directory).encode("utf-8"))
    digest.update(str(config.packing.sequence_length).encode("utf-8"))
    digest.update(str(config.packing.overlap).encode("utf-8"))
    digest.update(config.packing.packing_strategy.encode("utf-8"))
    for document in documents:
        digest.update(document.relative_path.as_posix().encode("utf-8"))
        digest.update(document.normalized_sha256.encode("utf-8"))
    return digest.hexdigest()


def _repository_name(relative: PurePosixPath) -> str | None:
    if len(relative.parts) >= 2 and relative.parts[0] == "github":
        return relative.parts[1]
    return None


def _content_type(extension: str) -> str:
    return "python_code" if extension in PYTHON_EXTENSIONS else "documentation"


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise FinalCorpusError(f"{name} must be a mapping.")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise FinalCorpusError(f"{name} must be a list.")
    result = tuple(str(item) for item in value if str(item))
    if not result:
        raise FinalCorpusError(f"{name} must not be empty.")
    return result


def _extensions(value: Any) -> tuple[str, ...]:
    extensions: list[str] = []
    for item in _strings(value, "extensions"):
        extension = item.lower()
        extensions.append(extension if extension.startswith(".") else f".{extension}")
    return tuple(extensions)


def _sequence_length(packing: dict[str, Any]) -> int:
    if "sequence_length" in packing:
        return _positive_int(packing["sequence_length"], "sequence_length")
    if "context_length" in packing:
        return _positive_int(packing["context_length"], "context_length") + 1
    return 1025


def _positive_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise FinalCorpusError(f"{name} must be greater than zero.")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    number = int(value)
    if number < 0:
        raise FinalCorpusError(f"{name} must be non-negative.")
    return number


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _filename(value: Any) -> str:
    text = str(value)
    if not text or Path(text).name != text:
        raise FinalCorpusError("shard_prefix must be a filename component.")
    return text


__all__ = [
    "FinalCorpusBuildResult",
    "FinalCorpusConfig",
    "FinalCorpusDocument",
    "FinalCorpusError",
    "FinalCorpusPackingConfig",
    "FinalCorpusRejection",
    "FinalSourceFile",
    "build_final_statistics",
    "clean_final_text",
    "ensure_final_corpus_folders",
    "load_final_corpus_config",
    "pack_final_corpus",
    "process_final_corpus",
    "run_final_corpus_cli",
    "run_final_corpus_pipeline",
    "scan_final_corpus_sources",
    "write_final_metadata",
    "write_final_statistics",
]
