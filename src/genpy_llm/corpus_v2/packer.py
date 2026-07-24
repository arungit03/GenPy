"""Fixed-length packing and sharding for Corpus V2."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.corpus_v2.manifest import TokenizedDocument, timestamp, write_json
from genpy_llm.sequence_packer import (
    SequencePacker,
    SequencePackingConfig,
    prepare_document_tokens,
)
from genpy_llm.shard_builder import (
    SequenceShardWriter,
    final_outputs_valid,
    prepare_sequence_output,
    write_sequence_shard_index,
)


@dataclass(frozen=True)
class PackingSettings:
    """Packing and shard output settings."""

    output_directory: Path
    shard_prefix: str = "corpus_v2"
    context_length: int = 1024
    max_tokens_per_shard: int = 10_000_000
    add_bos: bool = False
    add_eos: bool = True
    pad_final_sequence: bool = False

    @property
    def index_path(self) -> Path:
        return self.output_directory / "index.json"

    @property
    def statistics_path(self) -> Path:
        return self.output_directory / "statistics.json"


@dataclass(frozen=True)
class PackingResult:
    """Packing output summary."""

    shard_index: dict[str, Any]
    resumed: bool


def pack_documents(
    documents: Iterable[TokenizedDocument],
    *,
    tokenizer: CodeTokenizer,
    tokenizer_path: Path,
    tokenizer_hash: str,
    source_manifest: Path,
    settings: PackingSettings,
    build_fingerprint: str,
    force: bool = False,
) -> PackingResult:
    """Pack tokenized documents into Phase-6-compatible binary shards."""

    if (
        not force
        and final_outputs_valid(settings.index_path, settings.statistics_path, build_fingerprint)
    ):
        import json

        return PackingResult(
            json.loads(settings.index_path.read_text(encoding="utf-8")),
            resumed=True,
        )
    prepare_sequence_output(
        settings.output_directory,
        settings.shard_prefix,
        [settings.index_path, settings.statistics_path],
    )
    packing_config = SequencePackingConfig(
        context_length=settings.context_length,
        add_bos=settings.add_bos,
        add_eos=settings.add_eos,
        document_boundary="eos" if settings.add_eos else "none",
        pad_final_sequence=settings.pad_final_sequence,
    )
    packer = SequencePacker(packing_config, pad_token_id=tokenizer.pad_token_id)
    writer = SequenceShardWriter(
        settings.output_directory,
        max_tokens_per_shard=settings.max_tokens_per_shard,
        context_length=settings.context_length,
        prefix=settings.shard_prefix,
    )
    try:
        for document in documents:
            prepared = prepare_document_tokens(
                document.token_ids,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                config=packing_config,
            )
            for sequence in packer.add_document(prepared, _sequence_metadata(document)):
                writer.write_sequence(sequence)
        for sequence in packer.finish():
            writer.write_sequence(sequence)
        shard_stats = writer.close()
    except Exception:
        writer.abort()
        raise
    shard_index = write_sequence_shard_index(
        settings.index_path,
        shard_stats,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_hash,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        context_length=settings.context_length,
        source_manifest=source_manifest,
        creation_timestamp=timestamp(),
        build_fingerprint=build_fingerprint,
    )
    write_json(
        settings.statistics_path,
        {
            "build_fingerprint": build_fingerprint,
            "sequence_count": shard_index["sequence_count"],
            "token_count": shard_index["token_count"],
            "byte_count": shard_index["byte_count"],
            "shard_count": len(shard_index["shards"]),
        },
    )
    return PackingResult(shard_index, resumed=False)


def _sequence_metadata(document: TokenizedDocument) -> dict[str, Any]:
    return {
        "stored_path": document.stored_path,
        "content_sha256": document.sha256,
        "source_type": document.source_type,
        "source_id": document.source_id,
        "repository": None,
        "package": None,
    }


__all__ = [
    "PackingResult",
    "PackingSettings",
    "pack_documents",
]
