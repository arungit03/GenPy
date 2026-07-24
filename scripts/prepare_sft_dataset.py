#!/usr/bin/env python3
"""Validate and deduplicate the SFT training dataset (no retraining performed).

Reads data/fine_tuning/train.jsonl, reports validation statistics, removes
exact (instruction, input, output) duplicates, and writes the result to
data/fine_tuning/train.deduplicated.jsonl. The original file is left
untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.sft_dataset_cleaning import (  # noqa: E402
    analyze_sft_dataset,
    deduplicate_sft_records,
    find_duplicate_pairs,
    load_sft_records_lenient,
    write_sft_records,
)


def main() -> int:
    source_path = PROJECT_ROOT / "data" / "fine_tuning" / "train.jsonl"
    output_path = PROJECT_ROOT / "data" / "fine_tuning" / "train.deduplicated.jsonl"

    report = analyze_sft_dataset(source_path)
    print(f"Source: {source_path}")
    print(f"Total non-blank lines: {report.total_lines}")
    print(f"Usable records: {report.usable_records}")
    print(f"Broken JSON lines: {len(report.broken_json_lines)}")
    print(f"Empty instructions: {report.empty_instruction_records}")
    print(f"Empty outputs: {report.empty_output_records}")
    print(f"Malformed records: {report.malformed_records}")

    records, _broken = load_sft_records_lenient(source_path)
    duplicate_pairs = find_duplicate_pairs(records)
    deduplicated, dedup_report = deduplicate_sft_records(records)
    write_sft_records(deduplicated, output_path)

    print(f"\nOriginal size: {dedup_report.original_size}")
    print(f"Duplicate (instruction, input, output) groups: {len(duplicate_pairs)}")
    print(f"Records removed (exact duplicates): {dedup_report.duplicate_pair_records_removed}")
    print(f"New size: {dedup_report.new_size}")
    print(f"Deduplicated dataset written to: {output_path}")

    summary_path = PROJECT_ROOT / "reports" / "training_pipeline" / "dedup_run.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "source": str(source_path),
                "output": str(output_path),
                "validation": {
                    "total_lines": report.total_lines,
                    "usable_records": report.usable_records,
                    "broken_json_lines": len(report.broken_json_lines),
                    "empty_instruction_records": report.empty_instruction_records,
                    "empty_output_records": report.empty_output_records,
                    "malformed_records": report.malformed_records,
                    "category_counts": report.category_counts,
                },
                "deduplication": {
                    "original_size": dedup_report.original_size,
                    "duplicate_instruction_records": dedup_report.duplicate_instruction_records,
                    "duplicate_output_records": dedup_report.duplicate_output_records,
                    "duplicate_pair_groups": len(duplicate_pairs),
                    "duplicate_pair_records_removed": dedup_report.duplicate_pair_records_removed,
                    "new_size": dedup_report.new_size,
                    "category_counts_before": dedup_report.category_counts_before,
                    "category_counts_after": dedup_report.category_counts_after,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Run summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
