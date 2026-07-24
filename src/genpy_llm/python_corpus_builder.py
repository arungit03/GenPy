"""Local-only Python corpus builder for GenPy pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tokenize
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

import yaml

from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash
from genpy_llm.sequence_packer import (
    SequencePacker,
    SequencePackingConfig,
    prepare_document_tokens,
)
from genpy_llm.shard_builder import (
    SequenceShardWriter,
    prepare_sequence_output,
    write_sequence_shard_index,
)

UTC = timezone.utc

DEFAULT_CONFIG_PATH = Path("configs/python_corpus.yaml")
DEFAULT_ALLOWED_EXTENSIONS = (".py", ".pyi", ".md", ".rst", ".txt")
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
DOCUMENTATION_EXTENSIONS = {".md", ".rst", ".txt"}


class PythonCorpusBuilderError(RuntimeError):
    """Raised when the local corpus builder cannot continue safely."""


@dataclass(frozen=True)
class PythonCorpusPackingConfig:
    """Packed-shard generation settings."""

    context_length: int = 1024
    max_tokens_per_shard: int = 10_000_000
    shard_prefix: str = "python_corpus"
    add_bos: bool = False
    add_eos: bool = True
    document_boundary: str = "eos"
    pad_final_sequence: bool = False


@dataclass(frozen=True)
class PythonCorpusConfig:
    """Validated local Python corpus configuration."""

    project_root: Path
    input_directory: Path
    output_directory: Path
    tokenizer_path: Path
    min_file_size: int = 80
    max_file_size: int = 2_000_000
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    ignored_directories: tuple[str, ...] = DEFAULT_IGNORED_DIRECTORIES
    ignored_suffixes: tuple[str, ...] = DEFAULT_IGNORED_SUFFIXES
    deduplication: bool = True
    preserve_comments: bool = True
    packing: PythonCorpusPackingConfig = field(default_factory=PythonCorpusPackingConfig)

    @property
    def cleaned_directory(self) -> Path:
        return self.output_directory / "cleaned"

    @property
    def packed_directory(self) -> Path:
        return self.output_directory / "packed"

    @property
    def statistics_directory(self) -> Path:
        return self.output_directory / "statistics"

    @property
    def metadata_directory(self) -> Path:
        return self.output_directory / "metadata"


@dataclass(frozen=True)
class ScannedFile:
    """One supported source file discovered under the input root."""

    path: Path
    relative_path: PurePosixPath
    source_section: str
    repository: str | None
    extension: str


@dataclass(frozen=True)
class CorpusDocument:
    """One cleaned, accepted corpus document."""

    source_path: Path
    relative_path: PurePosixPath
    cleaned_path: Path
    source_section: str
    repository: str | None
    extension: str
    content_type: str
    text: str
    size_bytes: int
    character_count: int
    word_count: int
    line_count: int
    content_sha256: str
    normalized_sha256: str
    token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RejectedFile:
    """One source file rejected by validation or deduplication."""

    path: Path
    relative_path: str
    reason: str
    size_bytes: int = 0


@dataclass(frozen=True)
class PythonCorpusBuildResult:
    """Artifacts produced by a corpus build."""

    accepted_files: int
    rejected_files: int
    duplicate_count: int
    total_tokens: int
    cleaned_directory: Path
    packed_directory: Path
    statistics_path: Path
    metadata_path: Path
    shard_index_path: Path


def load_python_corpus_config(path: Path | str = DEFAULT_CONFIG_PATH) -> PythonCorpusConfig:
    """Load and validate YAML configuration for the local corpus builder."""

    config_path = Path(path)
    project_root = (
        config_path.resolve().parents[1]
        if config_path.parent.name == "configs"
        else Path.cwd()
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise PythonCorpusBuilderError("Python corpus config must be a YAML mapping.")
    section = _mapping(payload.get("python_corpus", payload), "python_corpus")
    root = _resolve(project_root, Path(payload.get("project_root", project_root)))
    output = _resolve(root, Path(section.get("output_directory", "data/python_corpus")))
    packing = _mapping(section.get("packing", {}), "python_corpus.packing")
    return PythonCorpusConfig(
        project_root=root,
        input_directory=_resolve(root, Path(section.get("input_directory", "python_corpus"))),
        output_directory=output,
        tokenizer_path=_resolve(
            root,
            Path(section.get("tokenizer", "data/tokenizer/tokenizer.json")),
        ),
        min_file_size=_non_negative_int(section.get("min_file_size", 80), "min_file_size"),
        max_file_size=_positive_int(section.get("max_file_size", 2_000_000), "max_file_size"),
        allowed_extensions=_extensions(
            section.get("allowed_extensions", DEFAULT_ALLOWED_EXTENSIONS)
        ),
        ignored_directories=_strings(
            section.get("ignored_directories", DEFAULT_IGNORED_DIRECTORIES),
            "ignored_directories",
        ),
        ignored_suffixes=_extensions(section.get("ignored_suffixes", DEFAULT_IGNORED_SUFFIXES)),
        deduplication=bool(section.get("deduplication", True)),
        preserve_comments=bool(section.get("preserve_comments", True)),
        packing=PythonCorpusPackingConfig(
            context_length=_positive_int(packing.get("context_length", 1024), "context_length"),
            max_tokens_per_shard=_positive_int(
                packing.get("max_tokens_per_shard", 10_000_000),
                "max_tokens_per_shard",
            ),
            shard_prefix=_filename(packing.get("shard_prefix", "python_corpus")),
            add_bos=bool(packing.get("add_bos", False)),
            add_eos=bool(packing.get("add_eos", True)),
            document_boundary=str(packing.get("document_boundary", "eos")),
            pad_final_sequence=bool(packing.get("pad_final_sequence", False)),
        ),
    )


def scan_python_corpus(config: PythonCorpusConfig) -> list[ScannedFile]:
    """Recursively scan the configured input directory for supported local files."""

    if not config.input_directory.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_directory}")
    if not config.input_directory.is_dir():
        raise PythonCorpusBuilderError(f"Input path is not a directory: {config.input_directory}")

    ignored_dirs = set(config.ignored_directories)
    allowed = set(config.allowed_extensions)
    ignored_suffixes = set(config.ignored_suffixes)
    scanned: list[ScannedFile] = []
    for root, dirnames, filenames in os.walk(config.input_directory):
        dirnames[:] = sorted(name for name in dirnames if name not in ignored_dirs)
        current = Path(root)
        for filename in sorted(filenames):
            path = current / filename
            suffix = path.suffix.lower()
            if suffix not in allowed or suffix in ignored_suffixes:
                continue
            relative = PurePosixPath(path.relative_to(config.input_directory).as_posix())
            if any(part in ignored_dirs for part in relative.parts):
                continue
            scanned.append(
                ScannedFile(
                    path=path,
                    relative_path=relative,
                    source_section=relative.parts[0] if relative.parts else "",
                    repository=_repository_name(relative),
                    extension=suffix,
                )
            )
    return scanned


def clean_and_filter_files(
    files: list[ScannedFile],
    config: PythonCorpusConfig,
) -> tuple[list[CorpusDocument], list[RejectedFile]]:
    """Validate, normalize, deduplicate, and write cleaned corpus files."""

    documents: list[CorpusDocument] = []
    rejected: list[RejectedFile] = []
    seen_hashes: dict[str, PurePosixPath] = {}
    config.cleaned_directory.mkdir(parents=True, exist_ok=True)

    for item in files:
        try:
            size = item.path.stat().st_size
        except OSError:
            rejected.append(RejectedFile(item.path, item.relative_path.as_posix(), "stat_failed"))
            continue
        if size == 0:
            rejected.append(
                RejectedFile(item.path, item.relative_path.as_posix(), "empty_file", size)
            )
            continue
        if size < config.min_file_size:
            rejected.append(
                RejectedFile(item.path, item.relative_path.as_posix(), "too_small", size)
            )
            continue
        if size > config.max_file_size:
            rejected.append(
                RejectedFile(item.path, item.relative_path.as_posix(), "too_large", size)
            )
            continue
        try:
            raw_text = item.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            rejected.append(
                RejectedFile(item.path, item.relative_path.as_posix(), "invalid_utf8", size)
            )
            continue
        except OSError:
            rejected.append(
                RejectedFile(item.path, item.relative_path.as_posix(), "read_failed", size)
            )
            continue

        text = normalize_text(
            raw_text,
            preserve_comments=config.preserve_comments,
            extension=item.extension,
        )
        if not text.strip():
            rejected.append(
                RejectedFile(
                    item.path,
                    item.relative_path.as_posix(),
                    "empty_after_cleaning",
                    size,
                )
            )
            continue
        normalized_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if config.deduplication and normalized_sha in seen_hashes:
            rejected.append(
                RejectedFile(item.path, item.relative_path.as_posix(), "duplicate", size)
            )
            continue
        seen_hashes[normalized_sha] = item.relative_path

        cleaned_path = config.cleaned_directory / item.relative_path
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.write_text(text, encoding="utf-8")
        documents.append(
            CorpusDocument(
                source_path=item.path,
                relative_path=item.relative_path,
                cleaned_path=cleaned_path,
                source_section=item.source_section,
                repository=item.repository,
                extension=item.extension,
                content_type=_content_type(item.extension),
                text=text,
                size_bytes=len(text.encode("utf-8")),
                character_count=len(text),
                word_count=count_words(text),
                line_count=text.count("\n") + (1 if text else 0),
                content_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                normalized_sha256=normalized_sha,
            )
        )
    return documents, rejected


def normalize_text(text: str, *, preserve_comments: bool, extension: str) -> str:
    """Normalize text line endings, trailing whitespace, and optional Python comments."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = normalized.strip() + "\n"
    if preserve_comments or extension not in PYTHON_EXTENSIONS:
        return normalized
    return _strip_python_comments(normalized)


def count_words(text: str) -> int:
    """Return a lightweight word count for code and documentation text."""

    return sum(1 for part in text.replace("_", " ").split() if any(char.isalnum() for char in part))


def add_token_counts(
    documents: list[CorpusDocument],
    tokenizer: CodeTokenizer,
) -> list[CorpusDocument]:
    """Tokenize accepted documents with the existing GenPy tokenizer."""

    tokenized: list[CorpusDocument] = []
    for document in documents:
        tokenized.append(
            CorpusDocument(
                **{
                    **asdict(document),
                    "source_path": document.source_path,
                    "relative_path": document.relative_path,
                    "cleaned_path": document.cleaned_path,
                    "token_ids": tuple(tokenizer.encode(document.text)),
                }
            )
        )
    return tokenized


def write_metadata(
    documents: list[CorpusDocument],
    rejected: list[RejectedFile],
    config: PythonCorpusConfig,
) -> Path:
    """Write accepted and rejected file metadata."""

    config.metadata_directory.mkdir(parents=True, exist_ok=True)
    manifest = config.metadata_directory / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(_document_metadata(document), sort_keys=True) + "\n")
    rejected_path = config.metadata_directory / "rejected.jsonl"
    with rejected_path.open("w", encoding="utf-8") as file:
        for item in rejected:
            file.write(json.dumps(_rejected_metadata(item), sort_keys=True) + "\n")
    return manifest


def build_statistics(
    documents: list[CorpusDocument],
    rejected: list[RejectedFile],
    config: PythonCorpusConfig,
    shard_index: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the requested corpus statistics report."""

    sizes = [document.size_bytes for document in documents]
    token_counts = [len(document.token_ids) for document in documents]
    repositories = {
        document.repository
        for document in documents
        if document.source_section == "github" and document.repository
    }
    rejected_reasons = Counter(item.reason for item in rejected)
    duplicate_count = rejected_reasons.get("duplicate", 0)
    largest = sorted(documents, key=lambda document: document.size_bytes, reverse=True)[:10]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "input_directory": str(config.input_directory),
        "output_directory": str(config.output_directory),
        "number_of_repositories": len(repositories),
        "repositories": sorted(repositories),
        "number_of_files": len(documents),
        "python_files": sum(document.extension in PYTHON_EXTENSIONS for document in documents),
        "documentation_files": sum(
            document.extension in DOCUMENTATION_EXTENSIONS for document in documents
        ),
        "total_characters": sum(document.character_count for document in documents),
        "total_words": sum(document.word_count for document in documents),
        "total_tokens": sum(token_counts),
        "average_file_size": round(sum(sizes) / len(sizes), 6) if sizes else 0.0,
        "median_file_size": int(median(sizes)) if sizes else 0,
        "largest_files": [
            {
                "path": document.relative_path.as_posix(),
                "bytes": document.size_bytes,
                "tokens": len(document.token_ids),
            }
            for document in largest
        ],
        "duplicate_count": duplicate_count,
        "rejected_files": len(rejected),
        "rejection_reasons": dict(sorted(rejected_reasons.items())),
        "tokenizer": str(config.tokenizer_path),
        "tokenizer_sha256": tokenizer_file_hash(config.tokenizer_path),
        "packing": shard_index or {},
    }


def write_statistics(statistics: dict[str, Any], config: PythonCorpusConfig) -> Path:
    """Write JSON and Markdown statistics reports."""

    config.statistics_directory.mkdir(parents=True, exist_ok=True)
    json_path = config.statistics_directory / "statistics.json"
    json_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = config.statistics_directory / "statistics.md"
    markdown_path.write_text(_statistics_markdown(statistics), encoding="utf-8")
    return json_path


def pack_documents(
    documents: list[CorpusDocument],
    tokenizer: CodeTokenizer,
    config: PythonCorpusConfig,
    manifest_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Generate train-ready fixed-length uint16 shards."""

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
    packer = SequencePacker(packing_config, pad_token_id=tokenizer.pad_token_id)
    writer = SequenceShardWriter(
        config.packed_directory,
        max_tokens_per_shard=config.packing.max_tokens_per_shard,
        context_length=config.packing.context_length,
        prefix=config.packing.shard_prefix,
    )
    try:
        for document in documents:
            prepared = prepare_document_tokens(
                list(document.token_ids),
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                config=packing_config,
            )
            for sequence in packer.add_document(prepared, _document_metadata(document)):
                writer.write_sequence(sequence)
        for sequence in packer.finish():
            writer.write_sequence(sequence)
        shard_stats = writer.close()
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


def run_python_corpus_builder(
    config: PythonCorpusConfig,
    *,
    force: bool = False,
) -> PythonCorpusBuildResult:
    """Run the full local-only corpus build pipeline."""

    if force and config.output_directory.exists():
        shutil.rmtree(config.output_directory)
    config.cleaned_directory.mkdir(parents=True, exist_ok=True)
    config.packed_directory.mkdir(parents=True, exist_ok=True)
    config.statistics_directory.mkdir(parents=True, exist_ok=True)
    config.metadata_directory.mkdir(parents=True, exist_ok=True)

    scanned = scan_python_corpus(config)
    documents, rejected = clean_and_filter_files(scanned, config)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)
    documents = add_token_counts(documents, tokenizer)
    manifest_path = write_metadata(documents, rejected, config)
    shard_index_path = pack_documents(documents, tokenizer, config, manifest_path, force=force)
    shard_index = json.loads(shard_index_path.read_text(encoding="utf-8"))
    statistics = build_statistics(documents, rejected, config, shard_index)
    statistics_path = write_statistics(statistics, config)
    return PythonCorpusBuildResult(
        accepted_files=len(documents),
        rejected_files=len(rejected),
        duplicate_count=statistics["duplicate_count"],
        total_tokens=statistics["total_tokens"],
        cleaned_directory=config.cleaned_directory,
        packed_directory=config.packed_directory,
        statistics_path=statistics_path,
        metadata_path=manifest_path,
        shard_index_path=shard_index_path,
    )


def run_python_corpus_builder_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the local-only Python corpus builder."""

    parser = argparse.ArgumentParser(description="Build a local GenPy Python corpus.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = run_python_corpus_builder(load_python_corpus_config(args.config), force=args.force)
    print(f"Accepted files: {result.accepted_files}")
    print(f"Rejected files: {result.rejected_files}")
    print(f"Duplicate files: {result.duplicate_count}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Statistics: {result.statistics_path}")
    print(f"Packed index: {result.shard_index_path}")
    return 0


def _strip_python_comments(text: str) -> str:
    output: list[str] = []
    previous_line = 1
    previous_column = 0
    try:
        tokens = tokenize.generate_tokens(StringIO(text).readline)
        for token in tokens:
            token_type, token_text, start, end, _line = token
            if token_type == tokenize.COMMENT:
                continue
            start_line, start_column = start
            end_line, end_column = end
            if start_line > previous_line:
                output.append("\n" * (start_line - previous_line))
                previous_column = 0
            if start_column > previous_column:
                output.append(" " * (start_column - previous_column))
            output.append(token_text)
            previous_line = end_line
            previous_column = end_column
    except tokenize.TokenError:
        return text
    return "".join(output).strip() + "\n"


def _document_metadata(document: CorpusDocument) -> dict[str, Any]:
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
        "content_sha256": document.content_sha256,
        "normalized_sha256": document.normalized_sha256,
        "token_count": len(document.token_ids),
    }


def _rejected_metadata(item: RejectedFile) -> dict[str, Any]:
    return {
        "source_path": str(item.path),
        "stored_path": item.relative_path,
        "reason": item.reason,
        "size_bytes": item.size_bytes,
    }


def _statistics_markdown(statistics: dict[str, Any]) -> str:
    largest = "\n".join(
        f"- `{item['path']}`: {item['bytes']} bytes, {item['tokens']} tokens"
        for item in statistics["largest_files"]
    )
    largest = largest or "- None"
    return (
        "# Python Corpus Statistics\n\n"
        f"- Repositories: {statistics['number_of_repositories']}\n"
        f"- Files: {statistics['number_of_files']}\n"
        f"- Python files: {statistics['python_files']}\n"
        f"- Documentation files: {statistics['documentation_files']}\n"
        f"- Total characters: {statistics['total_characters']}\n"
        f"- Total words: {statistics['total_words']}\n"
        f"- Total tokens: {statistics['total_tokens']}\n"
        f"- Average file size: {statistics['average_file_size']}\n"
        f"- Duplicate count: {statistics['duplicate_count']}\n\n"
        "## Largest Files\n\n"
        f"{largest}\n"
    )


def _build_fingerprint(documents: list[CorpusDocument], config: PythonCorpusConfig) -> str:
    digest = hashlib.sha256()
    digest.update(str(config.input_directory).encode("utf-8"))
    digest.update(str(config.output_directory).encode("utf-8"))
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
        raise PythonCorpusBuilderError(f"{name} must be a mapping.")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PythonCorpusBuilderError(f"{name} must be a list.")
    result = tuple(str(item) for item in value if str(item))
    if not result:
        raise PythonCorpusBuilderError(f"{name} must not be empty.")
    return result


def _extensions(value: Any) -> tuple[str, ...]:
    extensions = []
    for item in _strings(value, "extensions"):
        extension = item.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.append(extension)
    return tuple(extensions)


def _positive_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise PythonCorpusBuilderError(f"{name} must be greater than zero.")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    number = int(value)
    if number < 0:
        raise PythonCorpusBuilderError(f"{name} must be non-negative.")
    return number


def _filename(value: Any) -> str:
    text = str(value)
    if not text or Path(text).name != text:
        raise PythonCorpusBuilderError("shard_prefix must be a filename component.")
    return text


__all__ = [
    "CorpusDocument",
    "PythonCorpusBuildResult",
    "PythonCorpusBuilderError",
    "PythonCorpusConfig",
    "PythonCorpusPackingConfig",
    "RejectedFile",
    "ScannedFile",
    "add_token_counts",
    "build_statistics",
    "clean_and_filter_files",
    "count_words",
    "load_python_corpus_config",
    "normalize_text",
    "pack_documents",
    "run_python_corpus_builder",
    "run_python_corpus_builder_cli",
    "scan_python_corpus",
    "write_metadata",
    "write_statistics",
]
