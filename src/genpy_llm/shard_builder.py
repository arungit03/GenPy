"""Binary fixed-sequence shard writer for Phase 5.5C."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from genpy_llm.sequence_packer import PackedSequence


class SequenceShardBuilderError(RuntimeError):
    """Raised when final sequence shards cannot be written."""


@dataclass(frozen=True)
class SequenceShardInfo:
    """Metadata for one finalized sequence shard."""

    filename: str
    metadata_filename: str
    shard_index: int
    sequence_count: int
    token_count: int
    byte_count: int
    sha256: str
    metadata_sha256: str


@dataclass(frozen=True)
class SequenceShardStatistics:
    """Aggregate final shard statistics."""

    sequence_count: int
    token_count: int
    byte_count: int
    shards: tuple[SequenceShardInfo, ...]


@dataclass
class _ActiveSequenceShard:
    path: Path
    partial_path: Path
    metadata_path: Path
    metadata_partial_path: Path
    file: Any
    shard_index: int
    sequence_count: int = 0
    token_count: int = 0
    metadata: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SequenceShardWriter:
    """Write fixed-length uint16 token sequences plus gzipped sidecar metadata."""

    output_directory: Path
    max_tokens_per_shard: int
    context_length: int
    prefix: str = "shard"
    _active: _ActiveSequenceShard | None = field(default=None, init=False)
    _shards: list[SequenceShardInfo] = field(default_factory=list, init=False)
    _sequence_count: int = 0
    _token_count: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens_per_shard <= 0:
            raise SequenceShardBuilderError("max_tokens_per_shard must be positive.")
        if self.context_length <= 0:
            raise SequenceShardBuilderError("context_length must be positive.")
        if not self.prefix or Path(self.prefix).name != self.prefix:
            raise SequenceShardBuilderError("prefix must be a plain filename component.")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.output_directory.glob(f"{self.prefix}_*.bin"))
        if existing:
            raise SequenceShardBuilderError(f"Refusing to overwrite existing shard: {existing[0]}")

    def write_sequence(self, sequence: PackedSequence) -> None:
        """Append a fixed-length packed sequence."""

        if len(sequence.token_ids) != self.context_length + 1:
            raise SequenceShardBuilderError("Packed sequence length must be context_length + 1.")
        _validate_uint16(sequence.token_ids)
        if (
            self._active is not None
            and self._active.sequence_count > 0
            and self._active.token_count + len(sequence.token_ids) > self.max_tokens_per_shard
        ):
            self._close_active()
        if self._active is None:
            self._open_next()
        assert self._active is not None
        offset = self._active.token_count
        values = array("H", sequence.token_ids)
        if sys.byteorder != "little":  # pragma: no cover
            values.byteswap()
        values.tofile(self._active.file)
        self._active.metadata.append(
            {
                "sequence_index": sequence.sequence_index,
                "shard_token_offset": offset,
                "token_count": len(sequence.token_ids),
                "padding_tokens": sequence.padding_tokens,
                "documents": sequence.document_offsets,
            }
        )
        self._active.sequence_count += 1
        self._active.token_count += len(sequence.token_ids)
        self._sequence_count += 1
        self._token_count += len(sequence.token_ids)

    def close(self) -> SequenceShardStatistics:
        """Finalize all shards."""

        if self._active is not None:
            self._close_active()
        return SequenceShardStatistics(
            sequence_count=self._sequence_count,
            token_count=self._token_count,
            byte_count=self._token_count * 2,
            shards=tuple(self._shards),
        )

    def abort(self) -> None:
        """Remove partial output after a failed build."""

        if self._active is not None:
            try:
                self._active.file.close()
            finally:
                self._active.partial_path.unlink(missing_ok=True)
                self._active.metadata_partial_path.unlink(missing_ok=True)
                self._active = None

    def _open_next(self) -> None:
        shard_index = len(self._shards)
        path = self.output_directory / f"{self.prefix}_{shard_index:05d}.bin"
        metadata_path = self.output_directory / f"{self.prefix}_{shard_index:05d}.metadata.json.gz"
        partial = Path(f"{path}.partial")
        metadata_partial = Path(f"{metadata_path}.partial")
        if path.exists() or partial.exists() or metadata_path.exists() or metadata_partial.exists():
            raise SequenceShardBuilderError(f"Refusing to overwrite shard output: {path}")
        self._active = _ActiveSequenceShard(
            path=path,
            partial_path=partial,
            metadata_path=metadata_path,
            metadata_partial_path=metadata_partial,
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
            with gzip.open(active.metadata_partial_path, "wt", encoding="utf-8") as file:
                json.dump(
                    {
                        "format": "genpy_sequence_shard_metadata",
                        "shard": active.path.name,
                        "shard_index": active.shard_index,
                        "context_length": self.context_length,
                        "sequence_count": active.sequence_count,
                        "token_count": active.token_count,
                        "sequences": active.metadata,
                    },
                    file,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            os.replace(active.partial_path, active.path)
            os.replace(active.metadata_partial_path, active.metadata_path)
            self._shards.append(
                SequenceShardInfo(
                    filename=active.path.name,
                    metadata_filename=active.metadata_path.name,
                    shard_index=active.shard_index,
                    sequence_count=active.sequence_count,
                    token_count=active.token_count,
                    byte_count=active.token_count * 2,
                    sha256=_file_hash(active.path),
                    metadata_sha256=_file_hash(active.metadata_path),
                )
            )
        except Exception:
            active.partial_path.unlink(missing_ok=True)
            active.metadata_partial_path.unlink(missing_ok=True)
            raise


def write_sequence_shard_index(
    path: Path,
    statistics: SequenceShardStatistics,
    *,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    context_length: int,
    source_manifest: Path,
    creation_timestamp: str,
    build_fingerprint: str,
) -> dict[str, Any]:
    """Write the final binary shard contract."""

    payload = {
        "format_version": 1,
        "format": "genpy_uint16_packed_sequence_shards",
        "dtype": "uint16",
        "byte_order": "little",
        "sequence_layout": "contiguous_fixed_length",
        "context_length": context_length,
        "sequence_length": context_length + 1,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": vocab_size,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "source_manifest": str(source_manifest),
        "creation_timestamp": creation_timestamp,
        "build_fingerprint": build_fingerprint,
        "sequence_count": statistics.sequence_count,
        "token_count": statistics.token_count,
        "byte_count": statistics.byte_count,
        "shards": [asdict(shard) for shard in statistics.shards],
    }
    _atomic_json(path, payload)
    return payload


def prepare_sequence_output(output_directory: Path, prefix: str, filenames: list[Path]) -> None:
    """Remove only final pretraining artifacts controlled by Phase 5.5C."""

    output_directory.mkdir(parents=True, exist_ok=True)
    targets = list(output_directory.glob(f"{prefix}_*.bin"))
    targets.extend(output_directory.glob(f"{prefix}_*.bin.partial"))
    targets.extend(output_directory.glob(f"{prefix}_*.metadata.json.gz"))
    targets.extend(output_directory.glob(f"{prefix}_*.metadata.json.gz.partial"))
    targets.extend(filenames)
    targets.extend(Path(f"{path}.partial") for path in filenames)
    for path in targets:
        path.unlink(missing_ok=True)


def final_outputs_valid(index_path: Path, statistics_path: Path, fingerprint: str) -> bool:
    """Validate resumable final binary outputs."""

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if index.get("build_fingerprint") != fingerprint:
        return False
    if statistics.get("build_fingerprint") != fingerprint:
        return False
    output_directory = index_path.parent
    shards = index.get("shards")
    if not isinstance(shards, list):
        return False
    for shard in shards:
        if not isinstance(shard, dict):
            return False
        data_path = output_directory / str(shard.get("filename"))
        metadata_path = output_directory / str(shard.get("metadata_filename"))
        if not data_path.is_file() or not metadata_path.is_file():
            return False
        if _file_hash(data_path) != shard.get("sha256"):
            return False
        if _file_hash(metadata_path) != shard.get("metadata_sha256"):
            return False
    return True


def _validate_uint16(token_ids: list[int]) -> None:
    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        or token_id > 65_535
        for token_id in token_ids
    ):
        raise SequenceShardBuilderError("Token IDs must be uint16-compatible integers.")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SequenceShardBuilderError",
    "SequenceShardInfo",
    "SequenceShardStatistics",
    "SequenceShardWriter",
    "final_outputs_valid",
    "prepare_sequence_output",
    "write_sequence_shard_index",
]
