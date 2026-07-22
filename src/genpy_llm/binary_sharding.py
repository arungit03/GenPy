"""Binary token shards and document indexes for scalable GPT pre-training."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class BinaryShardingError(RuntimeError):
    """Raised when binary token shards cannot be written safely."""


@dataclass(frozen=True)
class BinaryShardInfo:
    """Metadata for one finalized little-endian uint16 shard."""

    filename: str
    shard_index: int
    documents: int
    token_count: int
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class BinaryShardStatistics:
    """Summary returned after all binary shards are finalized."""

    documents: int
    token_count: int
    byte_count: int
    shards: tuple[BinaryShardInfo, ...]
    document_index: Path


@dataclass
class _ActiveBinaryShard:
    path: Path
    partial_path: Path
    file: Any
    shard_index: int
    documents: int = 0
    token_count: int = 0


@dataclass
class BinaryTokenShardWriter:
    """Write document-aligned, atomic uint16 token shards and a JSONL index."""

    output_directory: Path
    max_tokens_per_shard: int
    prefix: str = "github_tokens"
    document_index_filename: str = "document_index.jsonl"
    _active: _ActiveBinaryShard | None = field(default=None, init=False)
    _shards: list[BinaryShardInfo] = field(default_factory=list, init=False)
    _documents: int = field(default=0, init=False)
    _tokens: int = field(default=0, init=False)
    _document_index_file: Any = field(default=None, init=False)
    _document_index_partial: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.max_tokens_per_shard <= 0:
            raise BinaryShardingError("max_tokens_per_shard must be positive.")
        if not self.prefix or Path(self.prefix).name != self.prefix:
            raise BinaryShardingError("prefix must be a plain filename component.")
        if Path(self.document_index_filename).name != self.document_index_filename:
            raise BinaryShardingError("document_index_filename must be a filename.")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.output_directory.glob(f"{self.prefix}_*.bin"))
        if existing:
            raise BinaryShardingError(
                f"Refusing to overwrite existing binary shard: {existing[0]}"
            )
        document_index = self.output_directory / self.document_index_filename
        if document_index.exists():
            raise BinaryShardingError(f"Refusing to overwrite index: {document_index}")
        self._document_index_partial = Path(f"{document_index}.partial")
        self._document_index_partial.unlink(missing_ok=True)
        self._document_index_file = self._document_index_partial.open(
            "w", encoding="utf-8", newline="\n"
        )

    def write_document(self, token_ids: list[int], metadata: dict[str, Any]) -> None:
        """Write one complete document without crossing shard boundaries."""

        if not token_ids:
            raise BinaryShardingError("A binary-shard document must contain tokens.")
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id > 65_535
            for token_id in token_ids
        ):
            raise BinaryShardingError("Token IDs must be uint16-compatible integers.")
        if (
            self._active is not None
            and self._active.documents > 0
            and self._active.token_count + len(token_ids) > self.max_tokens_per_shard
        ):
            self._close_active()
        if self._active is None:
            self._open_next()
        assert self._active is not None
        offset = self._active.token_count
        values = array("H", token_ids)
        if sys.byteorder != "little":  # pragma: no cover - CI platforms are little endian
            values.byteswap()
        values.tofile(self._active.file)
        self._active.documents += 1
        self._active.token_count += len(token_ids)
        self._documents += 1
        self._tokens += len(token_ids)
        record = {
            **metadata,
            "shard": self._active.path.name,
            "shard_index": self._active.shard_index,
            "token_offset": offset,
            "token_count": len(token_ids),
        }
        json.dump(
            record,
            self._document_index_file,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._document_index_file.write("\n")

    def close(self) -> BinaryShardStatistics:
        """Finalize all partial files and return cumulative statistics."""

        if self._active is not None:
            self._close_active()
        document_index = self.output_directory / self.document_index_filename
        try:
            self._document_index_file.close()
            os.replace(self._document_index_partial, document_index)
        except Exception:
            self._document_index_partial.unlink(missing_ok=True)
            raise
        return BinaryShardStatistics(
            documents=self._documents,
            token_count=self._tokens,
            byte_count=self._tokens * 2,
            shards=tuple(self._shards),
            document_index=document_index,
        )

    def abort(self) -> None:
        """Close and remove partial output after a failed build."""

        if self._active is not None:
            try:
                self._active.file.close()
            finally:
                self._active.partial_path.unlink(missing_ok=True)
                self._active = None
        if self._document_index_file is not None:
            self._document_index_file.close()
        self._document_index_partial.unlink(missing_ok=True)

    def _open_next(self) -> None:
        shard_index = len(self._shards)
        path = self.output_directory / f"{self.prefix}_{shard_index:05d}.bin"
        partial = Path(f"{path}.partial")
        if path.exists() or partial.exists():
            raise BinaryShardingError(f"Refusing to overwrite binary shard: {path}")
        self._active = _ActiveBinaryShard(
            path=path,
            partial_path=partial,
            file=partial.open("wb"),
            shard_index=shard_index,
        )

    def _close_active(self) -> None:
        active = self._active
        self._active = None
        if active is None:
            return
        try:
            active.file.close()
            os.replace(active.partial_path, active.path)
            self._shards.append(
                BinaryShardInfo(
                    filename=active.path.name,
                    shard_index=active.shard_index,
                    documents=active.documents,
                    token_count=active.token_count,
                    byte_count=active.token_count * 2,
                    sha256=_file_hash(active.path),
                )
            )
        except Exception:
            active.partial_path.unlink(missing_ok=True)
            raise


def write_binary_shard_index(
    path: Path,
    statistics: BinaryShardStatistics,
    *,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    vocab_size: int,
    eos_token_id: int,
    source_manifest: Path,
    creation_timestamp: str,
) -> None:
    """Write the binary format contract consumed by GenPy."""

    payload = {
        "format_version": 1,
        "format": "genpy_uint16_token_shards",
        "dtype": "uint16",
        "byte_order": "little",
        "document_boundaries": "eos_token",
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "source_manifest": str(source_manifest),
        "creation_timestamp": creation_timestamp,
        "documents": statistics.documents,
        "token_count": statistics.token_count,
        "byte_count": statistics.byte_count,
        "document_index": statistics.document_index.name,
        "shards": [asdict(shard) for shard in statistics.shards],
    }
    _atomic_json(path, payload)


def read_binary_tokens(path: Path) -> list[int]:
    """Read a small uint16 shard for verification and tests."""

    if path.stat().st_size % 2:
        raise BinaryShardingError(f"Binary token shard has an odd byte count: {path}")
    values = array("H")
    with path.open("rb") as file:
        values.fromfile(file, path.stat().st_size // 2)
    if sys.byteorder != "little":  # pragma: no cover
        values.byteswap()
    return list(values)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BinaryShardInfo",
    "BinaryShardStatistics",
    "BinaryShardingError",
    "BinaryTokenShardWriter",
    "read_binary_tokens",
    "write_binary_shard_index",
]
