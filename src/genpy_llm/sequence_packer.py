"""Deterministic fixed-length sequence packing for GPT pretraining."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class SequencePackingError(RuntimeError):
    """Raised when token sequences cannot be packed safely."""


@dataclass(frozen=True)
class SequencePackingConfig:
    """Configuration for final pretraining sequence packing."""

    context_length: int
    add_bos: bool = False
    add_eos: bool = True
    document_boundary: str = "eos"
    pad_final_sequence: bool = False

    @property
    def sequence_length(self) -> int:
        """Number of token IDs stored per training sequence."""

        return self.context_length + 1


@dataclass(frozen=True)
class PackedSequence:
    """One fixed-length packed training sequence."""

    token_ids: list[int]
    sequence_index: int
    document_offsets: list[dict[str, Any]]
    padding_tokens: int = 0


@dataclass
class SequencePacker:
    """Pack document token streams into deterministic non-overlapping sequences."""

    config: SequencePackingConfig
    pad_token_id: int
    _buffer: list[int] = field(default_factory=list, init=False)
    _documents: list[dict[str, Any]] = field(default_factory=list, init=False)
    _sequence_index: int = 0

    def __post_init__(self) -> None:
        if self.config.context_length <= 0:
            raise SequencePackingError("context_length must be positive.")
        if self.config.document_boundary not in {"none", "eos", "eos_bos"}:
            raise SequencePackingError("document_boundary must be none, eos, or eos_bos.")
        if self.pad_token_id < 0 or self.pad_token_id > 65_535:
            raise SequencePackingError("pad_token_id must be uint16-compatible.")

    def add_document(
        self,
        token_ids: list[int],
        metadata: Mapping[str, Any],
    ) -> list[PackedSequence]:
        """Add one tokenized document and return all newly completed sequences."""

        if not token_ids:
            return []
        _validate_uint16_ids(token_ids)
        start = len(self._buffer)
        self._buffer.extend(token_ids)
        self._documents.append(
            {
                "stored_path": metadata.get("stored_path"),
                "content_sha256": metadata.get("content_sha256"),
                "source_type": metadata.get("source_type"),
                "source_id": metadata.get("source_id"),
                "repository": metadata.get("repository"),
                "package": metadata.get("package"),
                "token_start": start,
                "token_end": len(self._buffer),
            }
        )
        return self._emit_complete()

    def finish(self) -> list[PackedSequence]:
        """Return the optional final padded sequence and reset the buffer."""

        if not self._buffer:
            return []
        if not self.config.pad_final_sequence:
            self._buffer.clear()
            self._documents.clear()
            return []
        sequence_length = self.config.sequence_length
        padding = sequence_length - len(self._buffer)
        if padding <= 0:
            return self._emit_complete()
        token_ids = [*self._buffer, *([self.pad_token_id] * padding)]
        sequence = PackedSequence(
            token_ids=token_ids,
            sequence_index=self._sequence_index,
            document_offsets=self._clip_documents(0, len(self._buffer)),
            padding_tokens=padding,
        )
        self._sequence_index += 1
        self._buffer.clear()
        self._documents.clear()
        return [sequence]

    def _emit_complete(self) -> list[PackedSequence]:
        sequences: list[PackedSequence] = []
        sequence_length = self.config.sequence_length
        while len(self._buffer) >= sequence_length:
            sequences.append(
                PackedSequence(
                    token_ids=self._buffer[:sequence_length],
                    sequence_index=self._sequence_index,
                    document_offsets=self._clip_documents(0, sequence_length),
                    padding_tokens=0,
                )
            )
            self._sequence_index += 1
            del self._buffer[:sequence_length]
            self._advance_documents(sequence_length)
        return sequences

    def _clip_documents(self, start: int, end: int) -> list[dict[str, Any]]:
        clipped: list[dict[str, Any]] = []
        for document in self._documents:
            token_start = int(document["token_start"])
            token_end = int(document["token_end"])
            if token_end <= start or token_start >= end:
                continue
            clipped.append(
                {
                    key: value
                    for key, value in document.items()
                    if key not in {"token_start", "token_end"}
                }
                | {
                    "sequence_token_start": max(token_start, start) - start,
                    "sequence_token_end": min(token_end, end) - start,
                }
            )
        return clipped

    def _advance_documents(self, amount: int) -> None:
        remaining: list[dict[str, Any]] = []
        for document in self._documents:
            token_start = int(document["token_start"]) - amount
            token_end = int(document["token_end"]) - amount
            if token_end <= 0:
                continue
            item = dict(document)
            item["token_start"] = max(0, token_start)
            item["token_end"] = token_end
            remaining.append(item)
        self._documents = remaining


def prepare_document_tokens(
    token_ids: list[int],
    *,
    bos_token_id: int,
    eos_token_id: int,
    config: SequencePackingConfig,
) -> list[int]:
    """Apply configured BOS/EOS document boundaries."""

    prepared = list(token_ids)
    if config.add_bos or config.document_boundary == "eos_bos":
        prepared.insert(0, bos_token_id)
    if config.add_eos or config.document_boundary in {"eos", "eos_bos"}:
        prepared.append(eos_token_id)
    return prepared


def _validate_uint16_ids(token_ids: list[int]) -> None:
    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        or token_id > 65_535
        for token_id in token_ids
    ):
        raise SequencePackingError("Token IDs must be uint16-compatible integers.")


__all__ = [
    "PackedSequence",
    "SequencePacker",
    "SequencePackingConfig",
    "SequencePackingError",
    "prepare_document_tokens",
]
