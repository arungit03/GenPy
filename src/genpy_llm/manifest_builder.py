"""Manifest builders for the final GenPy pretraining corpus."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL atomically with deterministic key ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                json.dump(
                    record,
                    file,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def build_training_manifest(
    *,
    corpus_version: int,
    creation_date: str,
    tokenizer_path: Path,
    tokenizer_hash: str,
    repositories: int,
    packages: int,
    accepted_files: int,
    rejected_files: int,
    duplicates_removed: int,
    total_files: int,
    total_tokens: int,
    context_length: int,
    shard_index: Mapping[str, Any],
    source_manifest: Path,
    merged_manifest: Path,
    build_fingerprint: str,
) -> dict[str, Any]:
    """Create the Phase 6 training manifest."""

    shards = shard_index.get("shards")
    shards = shards if isinstance(shards, list) else []
    return {
        "corpus_version": corpus_version,
        "creation_date": creation_date,
        "tokenizer": str(tokenizer_path),
        "tokenizer_version": _tokenizer_version(tokenizer_path),
        "tokenizer_hash": tokenizer_hash,
        "number_of_repositories": repositories,
        "number_of_packages": packages,
        "accepted_files": accepted_files,
        "rejected_files": rejected_files,
        "duplicates_removed": duplicates_removed,
        "total_files": total_files,
        "total_tokens": total_tokens,
        "context_length": context_length,
        "sequence_length": context_length + 1,
        "training_sequences": int(shard_index.get("sequence_count") or 0),
        "shard_count": len(shards),
        "checksums": {
            str(item.get("filename")): item.get("sha256")
            for item in shards
            if isinstance(item, Mapping)
        },
        "source_manifest": str(source_manifest),
        "merged_manifest": str(merged_manifest),
        "build_fingerprint": build_fingerprint,
    }


def manifest_fingerprint(records: Iterable[Mapping[str, Any]], options: Mapping[str, Any]) -> str:
    """Hash stable final corpus identity and build options."""

    digest = hashlib.sha256()
    digest.update(json.dumps(options, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\n")
    for record in records:
        stable = {
            "stored_path": record.get("stored_path"),
            "content_sha256": record.get("content_sha256"),
            "source": record.get("source"),
            "license": record.get("license"),
        }
        digest.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _tokenizer_version(tokenizer_path: Path) -> str:
    metadata = tokenizer_path.with_name("tokenizer_metadata.json")
    if not metadata.is_file():
        return "unknown"
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    value = payload.get("format_version")
    return str(value) if value is not None else "unknown"


__all__ = [
    "build_training_manifest",
    "manifest_fingerprint",
    "write_json",
    "write_jsonl",
]
