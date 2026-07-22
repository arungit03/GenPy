"""Packed binary shard dataset for Phase 6 GPT pretraining."""

from __future__ import annotations

import glob
import json
import math
import random
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash


class PretrainingDatasetError(RuntimeError):
    """Raised when packed pretraining shards cannot be read safely."""


@dataclass(frozen=True)
class PackedShardDatasetConfig:
    """Configuration for packed binary shard loading."""

    shard_pattern: str
    manifest_path: Path
    batch_seed: int = 42
    shuffle: bool = True
    mmap: bool = True


class PackedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Map-style dataset over fixed-length packed uint16 token shards."""

    def __init__(
        self,
        shard_pattern: str | Path,
        *,
        tokenizer: CodeTokenizer,
        manifest_path: Path,
        sequence_length: int | None = None,
        mmap: bool = True,
    ) -> None:
        if not isinstance(tokenizer, CodeTokenizer):
            raise PretrainingDatasetError("tokenizer must be a CodeTokenizer.")
        self.shard_pattern = str(shard_pattern)
        self.tokenizer = tokenizer
        self.manifest_path = Path(manifest_path)
        self.index_path = self._resolve_index_path()
        self.index = self._load_index()
        self.sequence_length = int(sequence_length or self.index["sequence_length"])
        self.context_length = self.sequence_length - 1
        self.mmap = bool(mmap)
        self.shards = self._load_shards()
        self._prefix_counts: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["sequence_count"])
            self._prefix_counts.append(total)
        if total <= 0:
            raise PretrainingDatasetError("Packed pretraining shards contain no sequences.")

    def __len__(self) -> int:
        return self._prefix_counts[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index must be an integer.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = self._find_shard(index)
        previous = 0 if shard_index == 0 else self._prefix_counts[shard_index - 1]
        local_index = index - previous
        shard = self.shards[shard_index]
        token_ids = self._read_sequence(shard, local_index)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.long)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }

    def _resolve_index_path(self) -> Path:
        if self.manifest_path.is_file():
            return self.manifest_path
        candidate = Path(self.shard_pattern).parent / "index.json"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Packed shard manifest not found: {self.manifest_path}")

    def _load_index(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PretrainingDatasetError(f"Invalid shard index JSON: {self.index_path}") from exc
        if not isinstance(payload, dict):
            raise PretrainingDatasetError("Packed shard index must be an object.")
        if payload.get("format") != "genpy_uint16_packed_sequence_shards":
            raise PretrainingDatasetError("Unsupported packed shard format.")
        if payload.get("dtype") != "uint16" or payload.get("byte_order") != "little":
            raise PretrainingDatasetError("Packed shards must be little-endian uint16.")
        if payload.get("vocab_size") != self.tokenizer.vocab_size:
            raise PretrainingDatasetError("Shard vocabulary size does not match tokenizer.")
        tokenizer_path = getattr(self.tokenizer, "source_path", None)
        if tokenizer_path is not None and payload.get("tokenizer_sha256") != tokenizer_file_hash(
            tokenizer_path
        ):
            raise PretrainingDatasetError("Shard tokenizer hash does not match tokenizer.")
        sequence_length = payload.get("sequence_length")
        if not isinstance(sequence_length, int) or sequence_length <= 1:
            raise PretrainingDatasetError("Packed shard sequence_length must be greater than one.")
        return payload

    def _load_shards(self) -> list[dict[str, Any]]:
        matched = {Path(path).name for path in glob.glob(self.shard_pattern)}
        if not matched:
            raise FileNotFoundError(f"No packed shards match pattern: {self.shard_pattern}")
        shards = []
        for item in self.index.get("shards", []):
            if not isinstance(item, dict):
                raise PretrainingDatasetError("Shard index entries must be objects.")
            filename = item.get("filename")
            if not isinstance(filename, str) or filename not in matched:
                continue
            path = self.index_path.parent / filename
            if not path.is_file():
                raise FileNotFoundError(f"Packed shard missing: {path}")
            expected_bytes = int(item["sequence_count"]) * self.sequence_length * 2
            if path.stat().st_size != expected_bytes:
                raise PretrainingDatasetError(f"Packed shard has unexpected size: {path}")
            shards.append({**item, "path": path})
        if not shards:
            raise PretrainingDatasetError("No matched shards are present in the shard index.")
        return sorted(shards, key=lambda item: int(item["shard_index"]))

    def _find_shard(self, index: int) -> int:
        low = 0
        high = len(self._prefix_counts) - 1
        while low < high:
            mid = (low + high) // 2
            if index < self._prefix_counts[mid]:
                high = mid
            else:
                low = mid + 1
        return low

    def _read_sequence(self, shard: dict[str, Any], local_index: int) -> list[int]:
        offset = int(local_index) * self.sequence_length * 2
        path = Path(shard["path"])
        if self.mmap:
            if sys.byteorder != "little":  # pragma: no cover
                raise PretrainingDatasetError("Big-endian mmap loading is unsupported.")
            values = np.memmap(
                path,
                dtype="<u2",
                mode="r",
                offset=offset,
                shape=(self.sequence_length,),
            )
            return [int(value) for value in values.tolist()]
        with path.open("rb") as file:
            file.seek(offset)
            values = array("H")
            values.fromfile(file, self.sequence_length)
        if sys.byteorder != "little":  # pragma: no cover
            values.byteswap()
        return list(values)


class DeterministicSequenceSampler(torch.utils.data.Sampler[int]):
    """Epoch-aware deterministic sampler, ready for distributed partitioning."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        start_index: int = 0,
    ) -> None:
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise PretrainingDatasetError("Invalid distributed sampler rank/world_size.")
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_index = int(start_index)
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        indices = indices[self.rank :: self.world_size]
        yield from indices[self.start_index :]

    def __len__(self) -> int:
        total = math.ceil(len(self.dataset) / self.world_size)
        return max(0, total - self.start_index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.start_index = 0


__all__ = [
    "DeterministicSequenceSampler",
    "PackedSequenceDataset",
    "PackedShardDatasetConfig",
    "PretrainingDatasetError",
]
