"""Local approved-source collection for Corpus V2."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath

from genpy_llm.corpus_v2.manifest import CollectedDocument, SourceSpec


class CorpusV2CollectionError(RuntimeError):
    """Raised when source collection cannot continue."""


SUPPORTED_EXTENSIONS = {
    ".py": "python_code",
    ".md": "technical_text",
    ".rst": "technical_text",
    ".txt": "technical_text",
}
ARCHIVE_EXTENSIONS = {
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".whl",
}
DEFAULT_SKIP_DIRECTORIES = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "ENV",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
)


def collect_documents(
    sources: Iterable[SourceSpec],
    *,
    skip_directories: Iterable[str] = DEFAULT_SKIP_DIRECTORIES,
    max_file_bytes: int = 2_000_000,
) -> Iterator[CollectedDocument]:
    """Yield supported files from approved local roots in deterministic order."""

    for source in sorted(sources, key=lambda item: item.source_id):
        if source.source_type not in {
            "local_python",
            "local_markdown",
            "local_rst",
            "local_txt",
            "local_git",
            "local_dataset",
        }:
            raise CorpusV2CollectionError(f"Unsupported source type: {source.source_type}")
        if not source.path.exists():
            raise FileNotFoundError(f"Corpus source not found: {source.path}")
        if source.path.is_file():
            yield from _file_source(source, max_file_bytes=max_file_bytes)
        elif source.path.is_dir():
            yield from _directory_source(
                source,
                skip_directories=tuple(skip_directories),
                max_file_bytes=max_file_bytes,
            )
        else:
            raise CorpusV2CollectionError(f"Unsupported source path: {source.path}")


def _file_source(source: SourceSpec, *, max_file_bytes: int) -> Iterator[CollectedDocument]:
    relative = PurePosixPath(source.path.name)
    if not _selected(relative, source.include, source.exclude):
        return
    document = _read_candidate(source, source.path, relative, max_file_bytes=max_file_bytes)
    if document is not None:
        yield document


def _directory_source(
    source: SourceSpec,
    *,
    skip_directories: tuple[str, ...],
    max_file_bytes: int,
) -> Iterator[CollectedDocument]:
    ignored = tuple(item.casefold() for item in skip_directories)
    for directory, dir_names, file_names in os.walk(source.path, topdown=True):
        dir_names[:] = sorted(
            name
            for name in dir_names
            if not _skip_part(name, ignored) and not (Path(directory) / name).is_symlink()
        )
        for filename in sorted(file_names):
            path = Path(directory) / filename
            if path.is_symlink() or not path.is_file():
                continue
            relative = PurePosixPath(path.relative_to(source.path).as_posix())
            if not _selected(relative, source.include, source.exclude):
                continue
            document = _read_candidate(source, path, relative, max_file_bytes=max_file_bytes)
            if document is not None:
                yield document


def _read_candidate(
    source: SourceSpec,
    path: Path,
    relative: PurePosixPath,
    *,
    max_file_bytes: int,
) -> CollectedDocument | None:
    suffix = path.suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS or suffix in ARCHIVE_EXTENSIONS:
        return None
    if path.stat().st_size > max_file_bytes:
        return None
    content = path.read_bytes()
    if _looks_binary(content):
        return None
    return CollectedDocument(
        source=source,
        path=path,
        relative_path=relative.as_posix(),
        content=content,
        content_type=SUPPORTED_EXTENSIONS[suffix],
    )


def _selected(path: PurePosixPath, include: Iterable[str], exclude: Iterable[str]) -> bool:
    value = path.as_posix()
    return _matches_any(value, include) and not _matches_any(value, exclude)


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(
        fnmatch.fnmatch(path, pattern)
        or PurePosixPath(path).match(pattern)
        or (pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]))
        for pattern in patterns
    )


def _skip_part(name: str, patterns: Iterable[str]) -> bool:
    folded = name.casefold()
    return any(fnmatch.fnmatch(folded, pattern) for pattern in patterns)


def _looks_binary(content: bytes) -> bool:
    if b"\x00" in content:
        return True
    sample = content[:4096]
    if not sample:
        return False
    textish = sum(byte in {9, 10, 13} or 32 <= byte <= 126 or byte >= 128 for byte in sample)
    return textish / len(sample) < 0.85


__all__ = [
    "CorpusV2CollectionError",
    "DEFAULT_SKIP_DIRECTORIES",
    "SUPPORTED_EXTENSIONS",
    "collect_documents",
]
