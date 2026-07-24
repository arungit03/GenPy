"""Validation and exact-duplicate cleaning for Alpaca-style SFT JSONL datasets.

Operates directly on ``data/fine_tuning/*.jsonl`` records (``instruction``,
``input``, ``output``, ``category``, ...). Unlike
:func:`genpy_llm.instruction_dataset.load_instruction_records` (which is strict
and used at training time), loading here is lenient: malformed lines are
counted and skipped rather than raising, so a single pass can produce a
complete validation report instead of stopping at the first defect.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SFTDatasetCleaningError(ValueError):
    """Raised when SFT dataset cleaning cannot proceed safely."""


@dataclass(frozen=True)
class SFTRecord:
    """One loaded Alpaca-style SFT record, plus its original line number."""

    line_number: int
    instruction: str
    input: str
    output: str
    category: str | None
    raw: dict[str, Any]

    @property
    def pair_key(self) -> tuple[str, str, str]:
        """Exact-match key used for instruction/output/pair duplicate detection."""

        return (self.instruction, self.input, self.output)


@dataclass(frozen=True)
class SFTLengthStatistics:
    """Character-length statistics for instructions and outputs."""

    count: int
    instruction_mean: float
    instruction_median: float
    instruction_min: int
    instruction_max: int
    output_mean: float
    output_median: float
    output_min: int
    output_max: int


@dataclass(frozen=True)
class SFTValidationReport:
    """Full validation report for one SFT JSONL file."""

    path: str
    total_lines: int
    usable_records: int
    broken_json_lines: tuple[int, ...]
    empty_instruction_records: int
    empty_output_records: int
    malformed_records: int
    category_counts: dict[str, int]
    length_statistics: SFTLengthStatistics | None


@dataclass(frozen=True)
class SFTDeduplicationReport:
    """Before/after statistics for exact-duplicate removal."""

    original_size: int
    duplicate_instruction_records: int
    duplicate_output_records: int
    duplicate_pair_records_removed: int
    new_size: int
    category_counts_before: dict[str, int]
    category_counts_after: dict[str, int]


def load_sft_records_lenient(path: Path | str) -> tuple[list[SFTRecord], tuple[int, ...]]:
    """Load Alpaca-style JSONL records, skipping and reporting broken lines."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"SFT dataset not found: {input_path}")
    records: list[SFTRecord] = []
    broken_lines: list[int] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                broken_lines.append(line_number)
                continue
            if not isinstance(payload, dict):
                broken_lines.append(line_number)
                continue
            instruction = payload.get("instruction")
            output = payload.get("output")
            input_text = payload.get("input", "")
            category = payload.get("category")
            records.append(
                SFTRecord(
                    line_number=line_number,
                    instruction=instruction if isinstance(instruction, str) else "",
                    input=input_text if isinstance(input_text, str) else "",
                    output=output if isinstance(output, str) else "",
                    category=category if isinstance(category, str) else None,
                    raw=payload,
                )
            )
    return records, tuple(broken_lines)


def analyze_sft_dataset(path: Path | str) -> SFTValidationReport:
    """Produce a full validation report for one SFT JSONL file."""

    input_path = Path(path)
    records, broken_lines = load_sft_records_lenient(input_path)
    # Split on literal "\n" only (matching JSONL's one-record-per-physical-line
    # format and how the file is iterated for loading). str.splitlines() also
    # breaks on Unicode line separators (e.g. U+2028/U+2029/U+0085), which are
    # legal, unescaped characters inside a JSON string value and would
    # otherwise inflate this count past the real number of records.
    all_lines = input_path.read_text(encoding="utf-8").split("\n")
    total_lines = sum(1 for line in all_lines if line.strip())

    empty_instruction = sum(1 for record in records if not record.instruction.strip())
    empty_output = sum(1 for record in records if not record.output.strip())
    malformed = sum(
        1
        for record in records
        if not isinstance(record.raw.get("instruction"), str)
        or not isinstance(record.raw.get("output"), str)
    )
    category_counts: dict[str, int] = {}
    for record in records:
        key = record.category or "unknown"
        category_counts[key] = category_counts.get(key, 0) + 1

    usable = [
        record for record in records if record.instruction.strip() and record.output.strip()
    ]
    length_statistics = _length_statistics(usable) if usable else None

    return SFTValidationReport(
        path=str(input_path),
        total_lines=total_lines,
        usable_records=len(usable),
        broken_json_lines=broken_lines,
        empty_instruction_records=empty_instruction,
        empty_output_records=empty_output,
        malformed_records=malformed,
        category_counts=dict(sorted(category_counts.items(), key=lambda item: -item[1])),
        length_statistics=length_statistics,
    )


def find_duplicate_instructions(records: Sequence[SFTRecord]) -> dict[str, int]:
    """Return {instruction_text: occurrence_count} for instructions seen more than once."""

    return _duplicate_counts(record.instruction for record in records)


def find_duplicate_outputs(records: Sequence[SFTRecord]) -> dict[str, int]:
    """Return {output_text: occurrence_count} for outputs seen more than once."""

    return _duplicate_counts(record.output for record in records)


def find_duplicate_pairs(records: Sequence[SFTRecord]) -> dict[tuple[str, str, str], int]:
    """Return {(instruction, input, output): occurrence_count} seen more than once."""

    return _duplicate_counts(record.pair_key for record in records)


def deduplicate_sft_records(
    records: Sequence[SFTRecord],
) -> tuple[list[SFTRecord], SFTDeduplicationReport]:
    """Remove exact (instruction, input, output) duplicates, keeping first occurrence.

    Preserves original order and never alters the content of a kept record, so
    no semantic meaning changes — only exact repeats are collapsed to one.
    """

    category_counts_before = _category_counts(records)
    duplicate_instructions = find_duplicate_instructions(records)
    duplicate_outputs = find_duplicate_outputs(records)

    seen_pairs: set[tuple[str, str, str]] = set()
    deduplicated: list[SFTRecord] = []
    for record in records:
        key = record.pair_key
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduplicated.append(record)

    duplicate_instruction_records = sum(
        count - 1 for count in duplicate_instructions.values()
    )
    duplicate_output_records = sum(count - 1 for count in duplicate_outputs.values())
    report = SFTDeduplicationReport(
        original_size=len(records),
        duplicate_instruction_records=duplicate_instruction_records,
        duplicate_output_records=duplicate_output_records,
        duplicate_pair_records_removed=len(records) - len(deduplicated),
        new_size=len(deduplicated),
        category_counts_before=category_counts_before,
        category_counts_after=_category_counts(deduplicated),
    )
    return deduplicated, report


def write_sft_records(records: Sequence[SFTRecord], path: Path | str) -> None:
    """Write records back to JSONL, preserving every original field."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record.raw, ensure_ascii=False, sort_keys=True) + "\n")


def _length_statistics(records: Sequence[SFTRecord]) -> SFTLengthStatistics:
    instruction_lengths = [len(record.instruction) for record in records]
    output_lengths = [len(record.output) for record in records]
    return SFTLengthStatistics(
        count=len(records),
        instruction_mean=statistics.mean(instruction_lengths),
        instruction_median=statistics.median(instruction_lengths),
        instruction_min=min(instruction_lengths),
        instruction_max=max(instruction_lengths),
        output_mean=statistics.mean(output_lengths),
        output_median=statistics.median(output_lengths),
        output_min=min(output_lengths),
        output_max=max(output_lengths),
    )


def _duplicate_counts(values: Any) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: count for key, count in counts.items() if count > 1}


def _category_counts(records: Sequence[SFTRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = record.category or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


__all__ = [
    "SFTDatasetCleaningError",
    "SFTDeduplicationReport",
    "SFTLengthStatistics",
    "SFTRecord",
    "SFTValidationReport",
    "analyze_sft_dataset",
    "deduplicate_sft_records",
    "find_duplicate_instructions",
    "find_duplicate_outputs",
    "find_duplicate_pairs",
    "load_sft_records_lenient",
    "write_sft_records",
]
