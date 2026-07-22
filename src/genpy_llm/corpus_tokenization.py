"""Shared provenance-manifest to binary-token-shard pipeline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from genpy_llm.binary_sharding import (
    BinaryShardStatistics,
    BinaryTokenShardWriter,
    write_binary_shard_index,
)
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.python_dataset_pipeline import ProgressBar

_WORKER_TOKENIZER: CodeTokenizer | None = None


class CorpusTokenizationError(RuntimeError):
    """Raised when validated corpus files cannot be encoded safely."""


@dataclass(frozen=True)
class CorpusTokenShardConfig:
    tokenizer_path: Path
    output_directory: Path
    shard_index_path: Path
    statistics_path: Path
    max_tokens_per_shard: int
    workers: int
    max_pending_tasks_per_worker: int
    shard_prefix: str
    document_index_filename: str = "document_index.jsonl"


MetadataBuilder = Callable[[Mapping[str, Any]], dict[str, Any]]


def build_manifest_token_shards(
    *,
    manifest_path: Path,
    corpus_root: Path,
    source_types: set[str],
    config: CorpusTokenShardConfig,
    tokenizer_sha256: str,
    manifest_fingerprint: str,
    build_fingerprint: str,
    metadata_builder: MetadataBuilder | None = None,
    progress: bool = True,
) -> tuple[BinaryShardStatistics, dict[str, Any]]:
    """Encode selected manifest records in deterministic order using bounded workers."""

    _validate_config(config)
    tokenizer = CodeTokenizer.from_file(config.tokenizer_path)
    if tokenizer.vocab_size > 65_536:
        raise CorpusTokenizationError("uint16 shards require a vocabulary <= 65,536.")
    count = count_manifest_records(manifest_path, source_types)
    writer = BinaryTokenShardWriter(
        config.output_directory,
        max_tokens_per_shard=config.max_tokens_per_shard,
        prefix=config.shard_prefix,
        document_index_filename=config.document_index_filename,
    )
    bar = ProgressBar("tokenize", count, enabled=progress)
    rejected: Counter[str] = Counter()
    source_bytes = source_characters = source_lines = 0
    records = iter_manifest_records(manifest_path, source_types)
    tasks = (
        _task_from_record(corpus_root, record, metadata_builder)
        for record in records
    )
    try:
        for completed, result in enumerate(
            _ordered_results(tasks, tokenizer, config),
            start=1,
        ):
            error = result.get("error")
            if error is not None:
                rejected[str(error)] += 1
            else:
                token_ids = result["token_ids"]
                metadata = result["metadata"]
                writer.write_document(token_ids, metadata)
                source_bytes += int(result["source_bytes"])
                source_characters += int(result["source_characters"])
                source_lines += int(result["source_lines"])
            bar.update(completed)
        shard_statistics = writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        bar.close()
    created_at = _timestamp()
    write_binary_shard_index(
        config.shard_index_path,
        shard_statistics,
        tokenizer_path=config.tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        source_manifest=manifest_path,
        creation_timestamp=created_at,
    )
    index = json.loads(config.shard_index_path.read_text(encoding="utf-8"))
    index.update(
        build_fingerprint=build_fingerprint,
        source_manifest_sha256=manifest_fingerprint,
    )
    atomic_json(config.shard_index_path, index)
    statistics = {
        "format_version": 1,
        "creation_timestamp": created_at,
        "build_fingerprint": build_fingerprint,
        "tokenizer_sha256": tokenizer_sha256,
        "source_manifest_sha256": manifest_fingerprint,
        "documents": shard_statistics.documents,
        "rejected_documents": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "token_count": shard_statistics.token_count,
        "byte_count": shard_statistics.byte_count,
        "shard_count": len(shard_statistics.shards),
        "source_bytes": source_bytes,
        "source_characters": source_characters,
        "source_lines": source_lines,
        "average_tokens_per_document": _ratio(
            shard_statistics.token_count, shard_statistics.documents
        ),
        "average_characters_per_token": _ratio(
            source_characters, shard_statistics.token_count
        ),
        "tokens_per_source_byte": _ratio(shard_statistics.token_count, source_bytes),
        "shards": [asdict(shard) for shard in shard_statistics.shards],
    }
    atomic_json(config.statistics_path, statistics)
    return shard_statistics, statistics


def iter_manifest_records(
    manifest_path: Path,
    source_types: set[str],
) -> Iterator[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusTokenizationError(
                    f"Invalid corpus manifest JSON at line {line_number}: {exc}"
                ) from exc
            source = record.get("source") if isinstance(record, dict) else None
            if isinstance(source, dict) and source.get("type") in source_types:
                yield record


def count_manifest_records(manifest_path: Path, source_types: set[str]) -> int:
    return sum(1 for _ in iter_manifest_records(manifest_path, source_types))


def stable_manifest_fingerprint(manifest_path: Path, source_types: set[str]) -> str:
    """Hash stable corpus identity without mutable filesystem timestamps."""

    digest = hashlib.sha256()
    for record in iter_manifest_records(manifest_path, source_types):
        source = record.get("source")
        stable = {
            "stored_path": record.get("stored_path"),
            "content_sha256": record.get("content_sha256"),
            "license": record.get("license"),
            "source_id": source.get("id") if isinstance(source, dict) else None,
            "revision": source.get("revision") if isinstance(source, dict) else None,
        }
        digest.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_binary_output(config: CorpusTokenShardConfig) -> None:
    """Remove only the configured shard generation before an intentional rebuild."""

    config.output_directory.mkdir(parents=True, exist_ok=True)
    targets = list(config.output_directory.glob(f"{config.shard_prefix}_*.bin"))
    targets.extend(config.output_directory.glob(f"{config.shard_prefix}_*.bin.partial"))
    targets.extend(
        [
            config.output_directory / config.document_index_filename,
            Path(f"{config.output_directory / config.document_index_filename}.partial"),
            config.shard_index_path,
            Path(f"{config.shard_index_path}.partial"),
            config.statistics_path,
            Path(f"{config.statistics_path}.partial"),
        ]
    )
    for path in targets:
        path.unlink(missing_ok=True)


def binary_outputs_valid(config: CorpusTokenShardConfig, fingerprint: str) -> bool:
    try:
        index: Any = json.loads(config.shard_index_path.read_text(encoding="utf-8"))
        statistics: Any = json.loads(config.statistics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(index, dict) or not isinstance(statistics, dict):
        return False
    if index.get("build_fingerprint") != fingerprint:
        return False
    if statistics.get("build_fingerprint") != fingerprint:
        return False
    shards = index.get("shards")
    if not isinstance(shards, list):
        return False
    for item in shards:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            return False
        path = config.output_directory / item["filename"]
        if not path.is_file() or _file_hash(path) != item.get("sha256"):
            return False
    return (config.output_directory / config.document_index_filename).is_file()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _ordered_results(
    tasks: Iterable[dict[str, Any]],
    tokenizer: CodeTokenizer,
    config: CorpusTokenShardConfig,
) -> Iterator[dict[str, Any]]:
    if config.workers == 1:
        for task in tasks:
            yield _tokenize(task, tokenizer)
        return
    iterator = iter(tasks)
    limit = config.workers * config.max_pending_tasks_per_worker
    pending: deque[Future[dict[str, Any]]] = deque()
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_initialize_worker,
        initargs=(str(config.tokenizer_path),),
    ) as executor:
        for _ in range(limit):
            task = next(iterator, None)
            if task is None:
                break
            pending.append(executor.submit(_worker_tokenize, task))
        while pending:
            yield pending.popleft().result()
            task = next(iterator, None)
            if task is not None:
                pending.append(executor.submit(_worker_tokenize, task))


def _task_from_record(
    corpus_root: Path,
    record: Mapping[str, Any],
    metadata_builder: MetadataBuilder | None,
) -> dict[str, Any]:
    source = record.get("source")
    metadata = (
        metadata_builder(record)
        if metadata_builder is not None
        else {
            "stored_path": record.get("stored_path"),
            "source_path": record.get("source_path"),
            "content_sha256": record.get("content_sha256"),
            "repository": source.get("repository_url")
            if isinstance(source, dict)
            else None,
            "source_id": source.get("id") if isinstance(source, dict) else None,
            "revision": source.get("revision") if isinstance(source, dict) else None,
            "license": record.get("license"),
            "byte_size": record.get("size_bytes"),
            "import_timestamp": record.get("collection_timestamp"),
        }
    )
    return {
        "path": str(corpus_root / str(record["stored_path"])),
        "expected_sha256": record["content_sha256"],
        "metadata": metadata,
    }


def _initialize_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = CodeTokenizer.from_file(tokenizer_path)


def _worker_tokenize(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_TOKENIZER is None:  # pragma: no cover
        raise CorpusTokenizationError("Tokenizer worker was not initialized.")
    return _tokenize(task, _WORKER_TOKENIZER)


def _tokenize(task: dict[str, Any], tokenizer: CodeTokenizer) -> dict[str, Any]:
    path = Path(task["path"])
    try:
        content = path.read_bytes()
    except OSError:
        return {"error": "read_error", "metadata": task["metadata"]}
    if hashlib.sha256(content).hexdigest() != task["expected_sha256"]:
        return {"error": "hash_mismatch", "metadata": task["metadata"]}
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {"error": "invalid_utf8", "metadata": task["metadata"]}
    token_ids = tokenizer.encode(text)
    token_ids.append(tokenizer.eos_token_id)
    metadata = dict(task["metadata"])
    metadata["token_count"] = len(token_ids)
    return {
        "error": None,
        "metadata": metadata,
        "token_ids": token_ids,
        "source_bytes": len(content),
        "source_characters": len(text),
        "source_lines": len(text.splitlines()),
    }


def _validate_config(config: CorpusTokenShardConfig) -> None:
    if not config.tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {config.tokenizer_path}")
    if config.max_tokens_per_shard <= 0 or config.workers <= 0:
        raise CorpusTokenizationError("Shard size and worker count must be positive.")
    if config.max_pending_tasks_per_worker <= 0:
        raise CorpusTokenizationError("max_pending_tasks_per_worker must be positive.")


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _timestamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CorpusTokenShardConfig",
    "CorpusTokenizationError",
    "atomic_json",
    "binary_outputs_valid",
    "build_manifest_token_shards",
    "count_manifest_records",
    "iter_manifest_records",
    "prepare_binary_output",
    "stable_manifest_fingerprint",
]
