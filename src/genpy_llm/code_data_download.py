"""Streaming download orchestration for Python code training data."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genpy_llm.code_filtering import (
    CodeFilterSettings,
    filter_code_record,
)
from genpy_llm.code_sharding import CompressedShardWriter

UTC = timezone.utc

DATASET_SOURCE = "codeparrot/codeparrot-clean"
INSTRUCTION_SOURCE = "sahil2801/CodeAlpaca-20k"
MANIFEST_VERSION = 1


class CodeDataDownloadError(RuntimeError):
    """Raised when code training data cannot be downloaded or converted."""


@dataclass(frozen=True)
class DownloadSummary:
    """Final downloader counters."""

    accepted_files: int
    rejected_files: int
    duplicate_files: int
    unknown_license_files: int
    train_records: int
    validation_records: int
    accepted_bytes: int
    train_shards: tuple[Path, ...]
    validation_shards: tuple[Path, ...]
    resumed: bool

    @property
    def accepted_gb(self) -> float:
        """Accepted uncompressed source size in decimal GB."""

        return self.accepted_bytes / 1_000_000_000


@dataclass(frozen=True)
class InstructionSummary:
    """Instruction conversion counters."""

    written_records: int
    skipped_records: int
    duplicate_records: int
    output_path: Path


def load_existing_hashes(path: Path, *, resume: bool) -> set[str]:
    """Load accepted hashes for resume-safe deduplication."""

    if not resume or not path.exists():
        return set()
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if value:
                hashes.add(value)
    return hashes


def stream_code_records(
    records: Iterable[dict[str, Any]],
    *,
    target_bytes: int,
    shard_mb: int,
    train_output: Path,
    validation_output: Path,
    hash_path: Path,
    manifest_path: Path,
    settings: CodeFilterSettings,
    validation_percent: int | float,
    seed: int,
    resume: bool,
    max_files: int | None = None,
) -> DownloadSummary:
    """Filter, deduplicate, split, and shard streamed code records."""

    if target_bytes <= 0:
        raise CodeDataDownloadError("target_bytes must be greater than zero.")
    manifest = _load_manifest(manifest_path) if resume and manifest_path.exists() else None
    if manifest is not None:
        _validate_resume_manifest(
            manifest,
            target_bytes=target_bytes,
            shard_mb=shard_mb,
            validation_percent=validation_percent,
            seed=seed,
            settings=settings,
        )
    train_writer = CompressedShardWriter(train_output, split="train", shard_mb=shard_mb)
    validation_writer = CompressedShardWriter(
        validation_output,
        split="validation",
        shard_mb=shard_mb,
    )
    seen_hashes = load_existing_hashes(hash_path, resume=resume)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = rejected = duplicate = unknown_license = 0
    train_records = validation_records = 0
    accepted_bytes = int(manifest.get("accepted_bytes", 0)) if manifest else 0
    try:
        with hash_path.open("a", encoding="utf-8") as hash_file:
            for index, raw_record in enumerate(records):
                if max_files is not None and index >= max_files:
                    break
                result = filter_code_record(
                    raw_record,
                    settings=settings,
                    validation_percent=validation_percent,
                )
                if not result.accepted:
                    rejected += 1
                    if result.reason == "unknown_license":
                        unknown_license += 1
                    continue
                assert result.record is not None
                if result.record.content_hash in seen_hashes:
                    duplicate += 1
                    continue
                seen_hashes.add(result.record.content_hash)
                hash_file.write(result.record.content_hash + "\n")
                if accepted % 100 == 0:
                    hash_file.flush()
                json_record = result.record.to_json_record()
                if result.record.split == "validation":
                    validation_writer.write(json_record)
                    validation_records += 1
                else:
                    train_writer.write(json_record)
                    train_records += 1
                accepted += 1
                accepted_bytes += result.record.byte_count
                if accepted % 1000 == 0:
                    print(f"accepted={accepted} bytes={accepted_bytes}")
                if accepted_bytes >= target_bytes:
                    break
        train_stats = train_writer.close()
        validation_stats = validation_writer.close()
    except Exception:
        train_writer.abort()
        validation_writer.abort()
        raise
    summary = DownloadSummary(
        accepted_files=accepted,
        rejected_files=rejected,
        duplicate_files=duplicate,
        unknown_license_files=unknown_license,
        train_records=train_records,
        validation_records=validation_records,
        accepted_bytes=accepted_bytes,
        train_shards=train_stats.shard_paths,
        validation_shards=validation_stats.shard_paths,
        resumed=resume,
    )
    write_manifest(
        manifest_path=manifest_path,
        summary=summary,
        target_bytes=target_bytes,
        shard_mb=shard_mb,
        validation_percent=validation_percent,
        seed=seed,
        settings=settings,
        completed=accepted_bytes >= target_bytes,
    )
    return summary


def write_manifest(
    *,
    manifest_path: Path,
    summary: DownloadSummary,
    target_bytes: int,
    shard_mb: int,
    validation_percent: int | float,
    seed: int,
    settings: CodeFilterSettings,
    completed: bool,
) -> None:
    """Atomically write the code download manifest."""

    now = datetime.now(UTC).isoformat()
    payload = {
        "format_version": MANIFEST_VERSION,
        "dataset_source": DATASET_SOURCE,
        "creation_timestamp": now,
        "last_update_timestamp": now,
        "target_bytes": target_bytes,
        "accepted_bytes": summary.accepted_bytes,
        "shard_size_mb": shard_mb,
        "accepted_file_count": summary.accepted_files,
        "rejected_file_count": summary.rejected_files,
        "duplicate_count": summary.duplicate_files,
        "license_skip_count": summary.unknown_license_files,
        "train_record_count": summary.train_records,
        "validation_record_count": summary.validation_records,
        "train_shard_names": [path.name for path in summary.train_shards],
        "validation_shard_names": [path.name for path in summary.validation_shards],
        "validation_percent": validation_percent,
        "seed": seed,
        "filter_settings": asdict(settings),
        "accepted_licenses": list(settings.accepted_licenses),
        "completed": completed,
        "resumed": summary.resumed,
    }
    _atomic_json_dump(payload, manifest_path)


def convert_instruction_records(
    records: Iterable[dict[str, Any]],
    output_path: Path,
) -> InstructionSummary:
    """Convert instruction-code records into GenPy fine-tuning JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(str(output_path) + ".partial")
    seen: set[tuple[str, str, str]] = set()
    written = skipped = duplicate = 0
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as file:
            for raw in records:
                instruction = _clean_instruction_text(raw.get("instruction"))
                response = _clean_instruction_text(raw.get("output", raw.get("response")))
                input_text = _clean_instruction_text(raw.get("input", ""))
                if not instruction or not response:
                    skipped += 1
                    continue
                key = (instruction, input_text, response)
                if key in seen:
                    duplicate += 1
                    continue
                seen.add(key)
                json.dump(
                    {
                        "instruction": instruction,
                        "input": input_text,
                        "response": response,
                        "source": "CodeAlpaca-20k",
                    },
                    file,
                    ensure_ascii=False,
                )
                file.write("\n")
                written += 1
        os.replace(partial_path, output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return InstructionSummary(written, skipped, duplicate, output_path)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise CodeDataDownloadError(f"Could not read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CodeDataDownloadError("Manifest must be a JSON object.")
    if payload.get("format_version") != MANIFEST_VERSION:
        raise CodeDataDownloadError("Unsupported manifest format version.")
    return payload


def _validate_resume_manifest(
    manifest: dict[str, Any],
    *,
    target_bytes: int,
    shard_mb: int,
    validation_percent: int | float,
    seed: int,
    settings: CodeFilterSettings,
) -> None:
    filter_settings = asdict(settings)
    filter_settings["accepted_licenses"] = list(filter_settings["accepted_licenses"])
    expected = {
        "target_bytes": target_bytes,
        "shard_size_mb": shard_mb,
        "validation_percent": validation_percent,
        "seed": seed,
        "filter_settings": filter_settings,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CodeDataDownloadError(f"Resume manifest conflicts on {key}.")


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(str(path) + ".partial")
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(partial_path, path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def _clean_instruction_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


__all__ = [
    "CodeDataDownloadError",
    "DownloadSummary",
    "InstructionSummary",
    "convert_instruction_records",
    "stream_code_records",
    "write_manifest",
]
