"""Classification and indexing framework for expanding the approved Python corpus."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from genpy_llm.python_corpus_collector import (
    CollectionResult,
    CorpusCollectorConfig,
    collect_python_corpus,
    configure_collector_logging,
    load_corpus_collector_config,
)

LOGGER = logging.getLogger("genpy_llm.python_corpus_expansion")
EXPANSION_VERSION = 1
CATEGORY_NAMES = (
    "Core Python",
    "OOP",
    "Algorithms",
    "Data Structures",
    "File Handling",
    "Exception Handling",
    "Standard Library",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Pytest",
)
_PRIMARY_CATEGORY_PRIORITY = (
    "Pytest",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Algorithms",
    "Data Structures",
    "File Handling",
    "Exception Handling",
    "OOP",
    "Standard Library",
    "Core Python",
)
_ALGORITHM_TERMS = {
    "algorithm",
    "algorithms",
    "binary_search",
    "breadth_first",
    "bfs",
    "depth_first",
    "dfs",
    "dijkstra",
    "dynamic_programming",
    "fibonacci",
    "knapsack",
    "merge_sort",
    "quicksort",
    "quick_sort",
    "search",
    "sort",
    "traverse",
}
_DATA_STRUCTURE_TERMS = {
    "binary_tree",
    "deque",
    "graph",
    "hash_map",
    "heap",
    "linked_list",
    "node",
    "priority_queue",
    "queue",
    "stack",
    "tree",
    "trie",
}
_FILE_MODULES = {"csv", "io", "os", "pathlib", "pickle", "shutil", "tempfile"}
_FILE_METHODS = {
    "open",
    "read",
    "read_bytes",
    "read_text",
    "write",
    "write_bytes",
    "write_text",
}


class CorpusExpansionError(RuntimeError):
    """Raised when a corpus expansion index cannot be built safely."""


@dataclass(frozen=True)
class CorpusExpansionConfig:
    """Configuration for collection, classification, and indexing."""

    collector: CorpusCollectorConfig
    index_path: Path
    report_path: Path
    collect_before_index: bool = True


@dataclass(frozen=True)
class CorpusExpansionResult:
    """Summary and artifact paths from one expansion run."""

    index_path: Path
    report_path: Path
    total_repositories: int
    total_python_files: int
    total_functions: int
    total_classes: int
    estimated_instruction_pairs: int
    rejected_index_records: int
    category_files: dict[str, int]
    collection: CollectionResult | None = None


@dataclass(frozen=True)
class SymbolRecord:
    """One function, async function, or class extracted from validated source."""

    kind: str
    qualified_name: str
    line_start: int
    line_end: int
    definition_sha256: str


@dataclass(frozen=True)
class FileAnalysis:
    """Classification and symbol counts derived from a parsed Python file."""

    categories: tuple[str, ...]
    primary_category: str
    functions: int
    classes: int
    symbols: tuple[SymbolRecord, ...]


def load_corpus_expansion_config(
    path: Path | str = "configs/dataset_pipeline.yaml",
) -> CorpusExpansionConfig:
    """Load expansion settings alongside the existing collector configuration."""

    collector = load_corpus_collector_config(path)
    try:
        raw = yaml.safe_load(collector.config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - collector already parses this file
        raise CorpusExpansionError(f"Invalid YAML in {collector.config_path}: {exc}") from exc
    section = raw.get("corpus_expansion", {})
    if not isinstance(section, dict):
        raise CorpusExpansionError("corpus_expansion must be a YAML mapping.")
    index_path = _resolve(
        collector.project_root,
        section.get("index", "data/raw/corpus_index.sqlite3"),
    )
    report_path = _resolve(
        collector.project_root,
        section.get("report", "data/raw/corpus_expansion_report.json"),
    )
    for artifact in (index_path, report_path):
        try:
            artifact.resolve().relative_to(collector.output_directory.resolve())
        except ValueError as exc:
            raise CorpusExpansionError(
                f"Corpus expansion artifacts must be under {collector.output_directory}: "
                f"{artifact}"
            ) from exc
    return CorpusExpansionConfig(
        collector=collector,
        index_path=index_path,
        report_path=report_path,
        collect_before_index=bool(section.get("collect_before_index", True)),
    )


def expand_python_corpus(
    config: CorpusExpansionConfig,
    *,
    collect: bool | None = None,
) -> CorpusExpansionResult:
    """Import approved sources, validate the corpus, and atomically rebuild its index."""

    should_collect = config.collect_before_index if collect is None else collect
    collection = collect_python_corpus(config.collector) if should_collect else None
    manifest_records = tuple(_iter_manifest(config.collector.provenance_manifest))
    started_at = _timestamp()
    partial_index = Path(f"{config.index_path}.partial")
    partial_index.parent.mkdir(parents=True, exist_ok=True)
    partial_index.unlink(missing_ok=True)
    database = sqlite3.connect(partial_index)
    rejection_reasons: Counter[str] = Counter()
    category_files: Counter[str] = Counter()
    primary_categories: Counter[str] = Counter()
    repositories: set[str] = set()
    total_files = total_functions = total_classes = duplicate_symbols = 0
    try:
        _initialize_schema(database)
        for provenance in manifest_records:
            reason, path, content, tree = _validate_index_candidate(config, provenance)
            if reason is not None:
                rejection_reasons[reason] += 1
                continue
            assert path is not None and content is not None and tree is not None
            if database.execute(
                "SELECT 1 FROM files WHERE stored_path = ?",
                (provenance["stored_path"],),
            ).fetchone():
                rejection_reasons["duplicate_stored_path"] += 1
                continue
            if database.execute(
                "SELECT 1 FROM files WHERE content_sha256 = ?",
                (provenance["content_sha256"],),
            ).fetchone():
                rejection_reasons["duplicate_content"] += 1
                continue
            analysis = analyze_python_file(content, tree, path)
            file_id = _insert_file(database, provenance, path, analysis)
            for category in analysis.categories:
                database.execute(
                    "INSERT INTO file_categories (file_id, category) VALUES (?, ?)",
                    (file_id, category),
                )
                category_files[category] += 1
            primary_categories[analysis.primary_category] += 1
            duplicate_symbols += _insert_symbols(database, file_id, analysis.symbols)
            source = provenance.get("source")
            if isinstance(source, dict) and isinstance(source.get("id"), str):
                repositories.add(str(source["id"]))
            total_files += 1
            total_functions += analysis.functions
            total_classes += analysis.classes
        _write_index_metadata(
            database,
            total_files=total_files,
            total_functions=total_functions,
            total_classes=total_classes,
            repositories=len(repositories),
            duplicate_symbols=duplicate_symbols,
        )
        database.commit()
    except Exception:
        database.close()
        partial_index.unlink(missing_ok=True)
        raise
    database.close()
    os.replace(partial_index, config.index_path)

    estimated_pairs = total_functions + total_classes
    report = {
        "expansion_version": EXPANSION_VERSION,
        "started_at": started_at,
        "completed_at": _timestamp(),
        "configuration": str(config.collector.config_path),
        "index": str(config.index_path),
        "provenance_manifest": str(config.collector.provenance_manifest),
        "total_repositories": len(repositories),
        "total_python_files": total_files,
        "total_functions": total_functions,
        "total_classes": total_classes,
        "estimated_instruction_pairs": estimated_pairs,
        "duplicate_symbols": duplicate_symbols,
        "rejected_index_records": sum(rejection_reasons.values()),
        "index_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "category_files": {
            category: category_files.get(category, 0) for category in CATEGORY_NAMES
        },
        "primary_category_files": {
            category: primary_categories.get(category, 0) for category in CATEGORY_NAMES
        },
        "collection": asdict(collection) if collection is not None else None,
    }
    _atomic_json_dump(report, config.report_path)
    LOGGER.info(
        "Corpus expansion ready: repositories=%s files=%s functions=%s classes=%s "
        "estimated_pairs=%s",
        len(repositories),
        total_files,
        total_functions,
        total_classes,
        estimated_pairs,
    )
    return CorpusExpansionResult(
        index_path=config.index_path,
        report_path=config.report_path,
        total_repositories=len(repositories),
        total_python_files=total_files,
        total_functions=total_functions,
        total_classes=total_classes,
        estimated_instruction_pairs=estimated_pairs,
        rejected_index_records=sum(rejection_reasons.values()),
        category_files={
            category: category_files.get(category, 0) for category in CATEGORY_NAMES
        },
        collection=collection,
    )


def analyze_python_file(content: str, tree: ast.Module, path: Path) -> FileAnalysis:
    """Classify parsed Python and extract stable symbol metadata."""

    classifier = _CategoryClassifier(path)
    classifier.visit(tree)
    categories = classifier.categories()
    symbols = tuple(_iter_symbols(content, tree))
    functions = sum(symbol.kind in {"function", "async_function"} for symbol in symbols)
    classes = sum(symbol.kind == "class" for symbol in symbols)
    primary = next(category for category in _PRIMARY_CATEGORY_PRIORITY if category in categories)
    return FileAnalysis(
        categories=categories,
        primary_category=primary,
        functions=functions,
        classes=classes,
        symbols=symbols,
    )


def run_corpus_expansion_cli(argv: Sequence[str] | None = None) -> int:
    """Run corpus import, classification, indexing, and reporting."""

    parser = argparse.ArgumentParser(
        description="Import and classify approved Python corpus collections."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_pipeline.yaml"),
        help="Dataset pipeline YAML containing corpus collection/expansion settings.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Rebuild from the current provenance manifest without importing sources.",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    try:
        config = load_corpus_expansion_config(args.config)
        configure_collector_logging(config.collector, level=args.log_level)
        result = expand_python_corpus(config, collect=not args.index_only)
    except Exception as exc:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Corpus expansion failed")
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Python corpus expansion complete")
    print(f"Index: {result.index_path}")
    print(f"Report: {result.report_path}")
    print(
        f"Repositories={result.total_repositories} files={result.total_python_files} "
        f"functions={result.total_functions} classes={result.total_classes} "
        f"estimated_pairs={result.estimated_instruction_pairs}"
    )
    return 0


class _CategoryClassifier(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.imports: set[str] = set()
        self.names: set[str] = set()
        self.has_class = False
        self.has_exception_handling = False
        self.has_file_handling = False

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.update(alias.name.split(".")[0] for alias in node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split(".")[0])
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.has_class = True
        self.names.add(_normalized_name(node.name))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(_normalized_name(node.name))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(_normalized_name(node.name))
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.has_exception_handling = True
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.has_exception_handling = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self.has_file_handling = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _FILE_METHODS:
            self.has_file_handling = True
        self.generic_visit(node)

    def categories(self) -> tuple[str, ...]:
        categories: set[str] = set()
        path_text = self.path.as_posix().casefold()
        name_terms = set(self.names)
        name_terms.update(_normalized_name(part) for part in self.path.parts)
        if self.has_class:
            categories.add("OOP")
        if any(_term_matches(name, _ALGORITHM_TERMS) for name in name_terms):
            categories.add("Algorithms")
        if (
            self.imports.intersection({"collections", "heapq", "queue"})
            or any(_term_matches(name, _DATA_STRUCTURE_TERMS) for name in name_terms)
        ):
            categories.add("Data Structures")
        if self.has_file_handling or self.imports.intersection(_FILE_MODULES):
            categories.add("File Handling")
        if self.has_exception_handling:
            categories.add("Exception Handling")
        if self.imports.intersection(sys.stdlib_module_names):
            categories.add("Standard Library")
        if "numpy" in self.imports:
            categories.add("NumPy")
        if "pandas" in self.imports:
            categories.add("Pandas")
        if "matplotlib" in self.imports:
            categories.add("Matplotlib")
        if "pytest" in self.imports or "/tests/" in f"/{path_text}/" or self.path.name.startswith(
            "test_"
        ):
            categories.add("Pytest")
        if not categories:
            categories.add("Core Python")
        return tuple(category for category in CATEGORY_NAMES if category in categories)


class _SymbolCollector(ast.NodeVisitor):
    def __init__(self, content: str) -> None:
        self.content = content
        self.scope: list[str] = []
        self.symbols: list[SymbolRecord] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node, "async_function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node, "class")

    def _visit_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        kind: str,
    ) -> None:
        qualified_name = ".".join([*self.scope, node.name])
        definition = ast.get_source_segment(self.content, node) or ast.dump(
            node,
            include_attributes=False,
        )
        normalized = "\n".join(line.rstrip() for line in definition.strip().splitlines())
        self.symbols.append(
            SymbolRecord(
                kind=kind,
                qualified_name=qualified_name,
                line_start=int(node.lineno),
                line_end=int(getattr(node, "end_lineno", node.lineno)),
                definition_sha256=_text_hash(normalized),
            )
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def _iter_symbols(content: str, tree: ast.Module) -> Iterator[SymbolRecord]:
    collector = _SymbolCollector(content)
    collector.visit(tree)
    yield from collector.symbols


def _validate_index_candidate(
    config: CorpusExpansionConfig,
    provenance: Mapping[str, Any],
) -> tuple[str | None, Path | None, str | None, ast.Module | None]:
    stored_path = provenance.get("stored_path")
    if not isinstance(stored_path, str):
        return "invalid_provenance", None, None, None
    source = provenance.get("source")
    if not isinstance(source, dict) or any(
        not isinstance(source.get(field), str) or not source[field]
        for field in ("id", "type", "location")
    ):
        return "invalid_provenance", None, None, None
    if not isinstance(provenance.get("source_path"), str):
        return "invalid_provenance", None, None, None
    expected_hash = provenance.get("content_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return "invalid_provenance", None, None, None
    if "license" not in provenance:
        return "invalid_provenance", None, None, None
    license_value = provenance.get("license")
    if license_value is not None and not isinstance(license_value, str):
        return "invalid_provenance", None, None, None
    if not isinstance(provenance.get("collection_timestamp"), str):
        return "invalid_provenance", None, None, None
    path = (config.collector.output_directory / stored_path).resolve()
    try:
        path.relative_to(config.collector.output_directory.resolve())
    except ValueError:
        return "unsafe_stored_path", None, None, None
    if not path.is_file():
        return "missing_file", None, None, None
    try:
        content_bytes = path.read_bytes()
    except OSError:
        return "file_read_error", None, None, None
    if _bytes_hash(content_bytes) != expected_hash:
        return "hash_mismatch", None, None, None
    if not (
        config.collector.minimum_file_bytes
        <= len(content_bytes)
        <= config.collector.maximum_file_bytes
    ):
        return "size_out_of_bounds", None, None, None
    try:
        content = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "invalid_utf8", None, None, None
    try:
        tree = ast.parse(content, filename=stored_path)
    except (SyntaxError, ValueError, TypeError):
        return "invalid_python_syntax", None, None, None
    return None, path, content, tree


def _initialize_schema(database: sqlite3.Connection) -> None:
    database.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE files (
            file_id INTEGER PRIMARY KEY,
            stored_path TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_location TEXT NOT NULL,
            source_revision TEXT,
            source_approval TEXT,
            repository_url TEXT,
            repository_stars INTEGER,
            repository_created_at TEXT,
            repository_updated_at TEXT,
            repository_pushed_at TEXT,
            default_branch TEXT,
            package_name TEXT,
            package_version TEXT,
            release_date TEXT,
            homepage TEXT,
            project_url TEXT,
            author TEXT,
            summary TEXT,
            keywords TEXT,
            source_archive TEXT,
            download_url TEXT,
            archive_sha256 TEXT,
            license TEXT,
            language TEXT NOT NULL CHECK (language = 'Python'),
            content_sha256 TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL,
            collection_timestamp TEXT NOT NULL,
            indexed_timestamp TEXT NOT NULL,
            primary_category TEXT NOT NULL,
            function_count INTEGER NOT NULL,
            class_count INTEGER NOT NULL,
            estimated_instruction_pairs INTEGER NOT NULL
        );
        CREATE TABLE file_categories (
            file_id INTEGER NOT NULL REFERENCES files(file_id),
            category TEXT NOT NULL,
            PRIMARY KEY (file_id, category)
        ) WITHOUT ROWID;
        CREATE INDEX file_categories_category ON file_categories(category, file_id);
        CREATE INDEX files_source_id ON files(source_id);
        CREATE INDEX files_package_name ON files(package_name, package_version);
        CREATE INDEX files_primary_category ON files(primary_category);
        CREATE TABLE symbols (
            symbol_id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL REFERENCES files(file_id),
            kind TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            definition_sha256 TEXT NOT NULL,
            duplicate_of_symbol_id INTEGER REFERENCES symbols(symbol_id),
            UNIQUE (file_id, kind, qualified_name, line_start)
        );
        CREATE INDEX symbols_definition_hash ON symbols(kind, definition_sha256);
        CREATE INDEX symbols_qualified_name ON symbols(qualified_name);
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _insert_file(
    database: sqlite3.Connection,
    provenance: Mapping[str, Any],
    path: Path,
    analysis: FileAnalysis,
) -> int:
    source = provenance["source"]
    assert isinstance(source, dict)
    cursor = database.execute(
        """
        INSERT INTO files (
            stored_path, source_path, source_id, source_type, source_location,
            source_revision, source_approval, repository_url, repository_stars,
            repository_created_at, repository_updated_at, repository_pushed_at,
            default_branch, package_name, package_version, release_date, homepage,
            project_url, author, summary, keywords, source_archive, download_url,
            archive_sha256, license, language, content_sha256, size_bytes,
            collection_timestamp, indexed_timestamp, primary_category,
            function_count, class_count, estimated_instruction_pairs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  'Python', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            provenance["stored_path"],
            provenance.get("source_path", ""),
            source["id"],
            source.get("type", "unknown"),
            source.get("location", ""),
            source.get("revision"),
            source.get("approval"),
            source.get("repository_url"),
            source.get("stars"),
            source.get("repository_created_at"),
            source.get("repository_updated_at"),
            source.get("repository_pushed_at"),
            source.get("default_branch"),
            source.get("package"),
            source.get("version"),
            source.get("release_date"),
            source.get("homepage"),
            source.get("project_url"),
            source.get("author"),
            source.get("summary"),
            source.get("keywords"),
            source.get("source_archive"),
            source.get("download_url"),
            source.get("archive_sha256"),
            provenance.get("license"),
            provenance["content_sha256"],
            path.stat().st_size,
            provenance["collection_timestamp"],
            _timestamp(),
            analysis.primary_category,
            analysis.functions,
            analysis.classes,
            analysis.functions + analysis.classes,
        ),
    )
    return int(cursor.lastrowid)


def _insert_symbols(
    database: sqlite3.Connection,
    file_id: int,
    symbols: Sequence[SymbolRecord],
) -> int:
    duplicates = 0
    for symbol in symbols:
        prior = database.execute(
            "SELECT symbol_id FROM symbols WHERE kind = ? AND definition_sha256 = ? "
            "ORDER BY symbol_id LIMIT 1",
            (symbol.kind, symbol.definition_sha256),
        ).fetchone()
        duplicate_of = int(prior[0]) if prior is not None else None
        duplicates += duplicate_of is not None
        database.execute(
            """
            INSERT INTO symbols (
                file_id, kind, qualified_name, line_start, line_end,
                definition_sha256, duplicate_of_symbol_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                symbol.kind,
                symbol.qualified_name,
                symbol.line_start,
                symbol.line_end,
                symbol.definition_sha256,
                duplicate_of,
            ),
        )
    return duplicates


def _write_index_metadata(
    database: sqlite3.Connection,
    *,
    total_files: int,
    total_functions: int,
    total_classes: int,
    repositories: int,
    duplicate_symbols: int,
) -> None:
    values = {
        "expansion_version": str(EXPANSION_VERSION),
        "created_at": _timestamp(),
        "total_repositories": str(repositories),
        "total_python_files": str(total_files),
        "total_functions": str(total_functions),
        "total_classes": str(total_classes),
        "estimated_instruction_pairs": str(total_functions + total_classes),
        "duplicate_symbols": str(duplicate_symbols),
    }
    database.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        sorted(values.items()),
    )


def _iter_manifest(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise CorpusExpansionError(
            f"Corpus provenance manifest not found: {path}. Run the collector first."
        )
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusExpansionError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise CorpusExpansionError(
                    f"Manifest record {path}:{line_number} is not an object."
                )
            yield record


def _term_matches(value: str, terms: set[str]) -> bool:
    padded = f"_{value}_"
    return any(f"_{term}_" in padded for term in terms)


def _normalized_name(value: str) -> str:
    return "_".join(part for part in _split_identifier(value) if part)


def _split_identifier(value: str) -> list[str]:
    expanded = ""
    for index, character in enumerate(value):
        if index and character.isupper() and value[index - 1].islower():
            expanded += "_"
        expanded += character.lower() if character.isalnum() else "_"
    return expanded.split("_")


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CorpusExpansionError("Corpus expansion path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _bytes_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _text_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CATEGORY_NAMES",
    "CorpusExpansionConfig",
    "CorpusExpansionError",
    "CorpusExpansionResult",
    "FileAnalysis",
    "analyze_python_file",
    "expand_python_corpus",
    "load_corpus_expansion_config",
    "run_corpus_expansion_cli",
]
