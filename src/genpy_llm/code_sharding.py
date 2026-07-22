"""Compressed JSONL shard writing for Python code records."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CodeShardingError(RuntimeError):
    """Raised when code shards cannot be written safely."""


@dataclass(frozen=True)
class ShardWriteStats:
    """Summary of shard writer activity."""

    records: int
    uncompressed_bytes: int
    shard_paths: tuple[Path, ...]


@dataclass
class _ActiveShard:
    path: Path
    partial_path: Path
    file: Any
    records: int = 0
    uncompressed_bytes: int = 0


@dataclass
class CompressedShardWriter:
    """Write records to gzip JSONL shards with atomic finalization."""

    output_dir: Path
    split: str
    shard_mb: int = 200
    prefix: str = "python"
    start_index: int | None = None
    _active: _ActiveShard | None = field(default=None, init=False)
    _next_index: int = field(default=0, init=False)
    _records: int = field(default=0, init=False)
    _bytes: int = field(default=0, init=False)
    _paths: list[Path] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.shard_mb <= 0:
            raise CodeShardingError("shard_mb must be greater than zero.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._next_index = (
            self.start_index if self.start_index is not None else self._discover_next()
        )

    @property
    def shard_bytes(self) -> int:
        """Configured approximate uncompressed bytes per shard."""

        return int(self.shard_mb * 1024 * 1024)

    def write(self, record: dict[str, Any]) -> None:
        """Write one JSON record line, rotating shards as needed."""

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        encoded = line.encode("utf-8")
        if self._active is None:
            self._open_next()
        assert self._active is not None
        would_exceed_shard = self._active.uncompressed_bytes + len(encoded) > self.shard_bytes
        if self._active.records > 0 and would_exceed_shard:
            self._close_active()
            self._open_next()
        assert self._active is not None
        self._active.file.write(line)
        self._active.records += 1
        self._active.uncompressed_bytes += len(encoded)
        self._records += 1
        self._bytes += len(encoded)

    def close(self) -> ShardWriteStats:
        """Finalize active shard and return cumulative stats."""

        if self._active is not None:
            self._close_active()
        return ShardWriteStats(
            records=self._records,
            uncompressed_bytes=self._bytes,
            shard_paths=tuple(self._paths),
        )

    def abort(self) -> None:
        """Close and remove the active partial shard after failure."""

        active = self._active
        self._active = None
        if active is None:
            return
        try:
            active.file.close()
        finally:
            active.partial_path.unlink(missing_ok=True)

    def _discover_next(self) -> int:
        pattern = f"{self.prefix}_{self.split}_*.jsonl.gz"
        indexes: list[int] = []
        for path in self.output_dir.glob(pattern):
            stem = path.name.removesuffix(".jsonl.gz")
            try:
                indexes.append(int(stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(indexes, default=-1) + 1

    def _open_next(self) -> None:
        while True:
            path = self.output_dir / f"{self.prefix}_{self.split}_{self._next_index:05d}.jsonl.gz"
            self._next_index += 1
            if not path.exists() and not path.with_suffix(path.suffix + ".partial").exists():
                break
        partial_path = Path(str(path) + ".partial")
        file = gzip.open(partial_path, "wt", encoding="utf-8", newline="\n")
        self._active = _ActiveShard(path=path, partial_path=partial_path, file=file)

    def _close_active(self) -> None:
        active = self._active
        self._active = None
        if active is None:
            return
        try:
            active.file.close()
            if active.path.exists():
                raise CodeShardingError(f"Refusing to overwrite existing shard: {active.path}")
            os.replace(active.partial_path, active.path)
            if active.records > 0:
                self._paths.append(active.path)
        except Exception:
            active.partial_path.unlink(missing_ok=True)
            raise


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a tiny gzip JSONL shard for tests and inspections."""

    records: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CodeShardingError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise CodeShardingError(f"Record in {path}:{line_number} must be an object.")
                records.append(record)
    except OSError as exc:
        raise CodeShardingError(f"Could not read gzip shard {path}: {exc}") from exc
    return records


__all__ = ["CodeShardingError", "CompressedShardWriter", "ShardWriteStats", "read_gzip_jsonl"]
