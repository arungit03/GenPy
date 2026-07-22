"""Streaming PyTorch dataset for gzip JSONL Python code shards."""

from __future__ import annotations

import glob
import gzip
import json
import random
import sys
from array import array
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash


class StreamingDatasetError(RuntimeError):
    """Raised when streaming code data cannot be read safely."""


@dataclass(frozen=True)
class StreamingDatasetConfig:
    """Streaming code dataset settings."""

    shard_pattern: str
    text_field: str = "text"
    context_length: int = 512
    stride: int = 512
    append_eos: bool = True
    pack_across_files: bool = True
    shuffle_shards: bool = False
    shuffle_buffer_records: int = 0
    seed: int = 42
    incomplete_window_policy: str = "drop"
    ignore_index: int = 0


class StreamingGPTDataset(IterableDataset):
    """Read gzip JSONL shards incrementally and yield GPT training windows."""

    def __init__(
        self,
        shard_pattern: str | Path,
        tokenizer: CodeTokenizer,
        *,
        text_field: str = "text",
        context_length: int = 512,
        stride: int = 512,
        append_eos: bool = True,
        pack_across_files: bool = True,
        shuffle_shards: bool = False,
        shuffle_buffer_records: int = 0,
        seed: int = 42,
        incomplete_window_policy: str = "drop",
        ignore_index: int | None = None,
    ) -> None:
        if not isinstance(tokenizer, CodeTokenizer):
            raise StreamingDatasetError("tokenizer must be a CodeTokenizer.")
        if tokenizer.vocab_size <= 0:
            raise StreamingDatasetError("tokenizer vocabulary must not be empty.")
        if context_length <= 0:
            raise StreamingDatasetError("context_length must be greater than zero.")
        if stride <= 0:
            raise StreamingDatasetError("stride must be greater than zero.")
        if incomplete_window_policy not in {"drop", "pad"}:
            raise StreamingDatasetError("incomplete_window_policy must be 'drop' or 'pad'.")
        if shuffle_buffer_records < 0:
            raise StreamingDatasetError("shuffle_buffer_records must be non-negative.")
        self.shard_pattern = str(shard_pattern)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.context_length = int(context_length)
        self.stride = int(stride)
        self.append_eos = bool(append_eos)
        self.pack_across_files = bool(pack_across_files)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_buffer_records = int(shuffle_buffer_records)
        self.seed = int(seed)
        self.incomplete_window_policy = incomplete_window_policy
        self.ignore_index = tokenizer.pad_token_id if ignore_index is None else int(ignore_index)
        self._shards = tuple(sorted(Path(path) for path in glob.glob(self.shard_pattern)))
        if not self._shards:
            raise FileNotFoundError(f"No shards match pattern: {self.shard_pattern}")
        suffixes = {path.suffix for path in self._shards}
        if ".bin" in suffixes and suffixes != {".bin"}:
            raise StreamingDatasetError("Binary and JSONL shards cannot be mixed.")
        self._binary = suffixes == {".bin"}
        if self._binary:
            self._validate_binary_contract()

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        """Resolved shard paths in deterministic order."""

        return self._shards

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield shifted GPT examples."""

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        worker_count = worker_info.num_workers if worker_info else 1
        shards = list(self._shards)
        if self.shuffle_shards:
            random.Random(self.seed).shuffle(shards)
        shards = shards[worker_id::worker_count]
        if not shards:
            return iter(())
        return self._iter_shards(shards, worker_id)

    def _iter_shards(
        self,
        shards: list[Path],
        worker_id: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if self._binary:
            yield from self._iter_binary_shards(shards)
            return
        buffer: list[int] = []
        for text in self._iter_texts(shards, worker_id):
            token_ids = self.tokenizer.encode(text)
            if self.append_eos:
                token_ids.append(self.tokenizer.eos_token_id)
            self._validate_token_ids(token_ids)
            if self.pack_across_files:
                buffer.extend(token_ids)
                yield from self._emit_available(buffer)
            else:
                local = list(token_ids)
                yield from self._emit_available(local)
                yield from self._emit_incomplete(local)
        if self.pack_across_files:
            yield from self._emit_incomplete(buffer)

    def _iter_binary_shards(
        self,
        shards: list[Path],
    ) -> Iterator[dict[str, torch.Tensor]]:
        buffer: deque[int] = deque()
        for shard in shards:
            if not self.pack_across_files:
                buffer.clear()
            if shard.stat().st_size % 2:
                raise StreamingDatasetError(f"Binary shard has an odd byte count: {shard}")
            try:
                with shard.open("rb") as file:
                    while chunk := file.read(2 * 65_536):
                        values = array("H")
                        values.frombytes(chunk)
                        if sys.byteorder != "little":  # pragma: no cover
                            values.byteswap()
                        ids = list(values)
                        self._validate_token_ids(ids)
                        buffer.extend(ids)
                        while len(buffer) >= self.context_length + 1:
                            window = list(islice(buffer, 0, self.context_length + 1))
                            yield _make_sample(
                                window[:-1],
                                window[1:],
                                [1] * self.context_length,
                            )
                            for _ in range(min(self.stride, len(buffer))):
                                buffer.popleft()
            except OSError as exc:
                raise StreamingDatasetError(f"Could not read binary shard {shard}: {exc}") from exc
            if not self.pack_across_files:
                local = list(buffer)
                buffer.clear()
                yield from self._emit_incomplete(local)
        if self.pack_across_files:
            local = list(buffer)
            buffer.clear()
            yield from self._emit_incomplete(local)

    def _validate_binary_contract(self) -> None:
        directory = self._shards[0].parent
        candidates = (directory / "index.json", directory / "shard_index.json")
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            names = ", ".join(path.name for path in candidates)
            raise StreamingDatasetError(
                f"Binary shard index not found in {directory}; expected one of: {names}"
            )
        contracts: list[tuple[Path, object]] = []
        for path in existing:
            try:
                contracts.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError) as exc:
                if len(existing) == 1:
                    raise StreamingDatasetError(
                        f"Invalid binary shard index {path}: {exc}"
                    ) from exc
        shard_names = {path.name for path in self._shards}
        matching = [
            (path, contract)
            for path, contract in contracts
            if isinstance(contract, dict)
            and isinstance(contract.get("shards"), list)
            and shard_names.issubset(
                {
                    item.get("filename")
                    for item in contract["shards"]
                    if isinstance(item, dict)
                    and isinstance(item.get("filename"), str)
                }
            )
        ]
        if matching:
            index_path, contract = matching[0]
        elif len(existing) == 1 and len(contracts) == 1:
            index_path, contract = contracts[0]
        else:
            raise StreamingDatasetError(
                f"No binary shard index in {directory} matches the selected shards."
            )
        if not isinstance(contract, dict) or contract.get("format") != (
            "genpy_uint16_token_shards"
        ):
            raise StreamingDatasetError("Unsupported binary shard index format.")
        if contract.get("dtype") != "uint16" or contract.get("byte_order") != "little":
            raise StreamingDatasetError("Binary shards must use little-endian uint16 tokens.")
        if contract.get("vocab_size") != self.tokenizer.vocab_size:
            raise StreamingDatasetError("Binary shard vocabulary does not match tokenizer.")
        if contract.get("eos_token_id") != self.tokenizer.eos_token_id:
            raise StreamingDatasetError("Binary shard EOS token does not match tokenizer.")
        tokenizer_hash = contract.get("tokenizer_sha256")
        tokenizer_path = getattr(self.tokenizer, "source_path", None)
        if tokenizer_path is not None and tokenizer_hash != tokenizer_file_hash(tokenizer_path):
            raise StreamingDatasetError("Binary shards were built with a different tokenizer.")

    def _iter_texts(self, shards: list[Path], worker_id: int) -> Iterator[str]:
        rng = random.Random(self.seed + worker_id)
        pending: deque[str] = deque()
        for shard in shards:
            try:
                with gzip.open(shard, "rt", encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise StreamingDatasetError(
                                f"Invalid JSON in {shard}:{line_number}: {exc}"
                            ) from exc
                        if not isinstance(record, dict):
                            raise StreamingDatasetError(
                                f"Record in {shard}:{line_number} must be an object."
                            )
                        text = record.get(self.text_field)
                        if not isinstance(text, str):
                            raise StreamingDatasetError(
                                f"Record in {shard}:{line_number} is missing text field."
                            )
                        if self.shuffle_buffer_records <= 1:
                            yield text
                        else:
                            pending.append(text)
                            if len(pending) >= self.shuffle_buffer_records:
                                index = rng.randrange(len(pending))
                                yield pending[index]
                                del pending[index]
            except OSError as exc:
                raise StreamingDatasetError(f"Could not read gzip shard {shard}: {exc}") from exc
        while pending:
            index = rng.randrange(len(pending))
            yield pending[index]
            del pending[index]

    def _emit_available(self, buffer: list[int]) -> Iterator[dict[str, torch.Tensor]]:
        while len(buffer) >= self.context_length + 1:
            window = buffer[: self.context_length + 1]
            yield _make_sample(
                window[:-1],
                window[1:],
                [1] * self.context_length,
            )
            del buffer[: self.stride]

    def _emit_incomplete(self, buffer: list[int]) -> Iterator[dict[str, torch.Tensor]]:
        if not buffer or self.incomplete_window_policy == "drop":
            buffer.clear()
            return
        tokens = buffer[: self.context_length + 1]
        if len(tokens) < 2:
            buffer.clear()
            return
        inputs = tokens[:-1][: self.context_length]
        targets = tokens[1:][: self.context_length]
        mask = [1] * len(inputs)
        while len(inputs) < self.context_length:
            inputs.append(self.tokenizer.pad_token_id)
            targets.append(self.ignore_index)
            mask.append(0)
        buffer.clear()
        yield _make_sample(inputs, targets, mask)

    def _validate_token_ids(self, token_ids: list[int]) -> None:
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.tokenizer.vocab_size:
                raise StreamingDatasetError("Tokenizer produced an invalid token ID.")


def _make_sample(
    input_ids: list[int],
    target_ids: list[int],
    attention_mask: list[int],
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
    }


__all__ = ["StreamingDatasetConfig", "StreamingDatasetError", "StreamingGPTDataset"]
