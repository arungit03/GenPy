"""Download and shard Python code training data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.code_data_download import (
    CodeDataDownloadError,
    convert_instruction_records,
    stream_code_records,
)
from genpy_llm.code_filtering import CodeFilterSettings


def main() -> int:
    args = _parse_args()
    try:
        if not args.skip_code:
            records = _load_code_dataset(args)
            summary = stream_code_records(
                records,
                target_bytes=int(args.code_gb * 1_000_000_000),
                shard_mb=args.shard_mb,
                train_output=_resolve(args.train_output),
                validation_output=_resolve(args.validation_output),
                hash_path=PROJECT_ROOT / "data" / "metadata" / "accepted_hashes.txt",
                manifest_path=PROJECT_ROOT / "data" / "metadata" / "code_download_manifest.json",
                settings=CodeFilterSettings(),
                validation_percent=args.validation_percent,
                seed=args.seed,
                resume=args.resume,
                max_files=args.max_files,
            )
            print("Code data download summary")
            print("==========================")
            print(f"Accepted files: {summary.accepted_files}")
            print(f"Rejected files: {summary.rejected_files}")
            print(f"Duplicate files: {summary.duplicate_files}")
            print(f"Unknown-licence files: {summary.unknown_license_files}")
            print(f"Train records: {summary.train_records}")
            print(f"Validation records: {summary.validation_records}")
            print(f"Accepted uncompressed GB: {summary.accepted_gb:.4f}")
            compressed_mb = _directory_size(_resolve(args.train_output).parent)
            print(f"Approximate compressed disk size: {compressed_mb:.2f} MB")
            print(f"Number of train shards: {len(summary.train_shards)}")
            print(f"Number of validation shards: {len(summary.validation_shards)}")
            print(f"Train output: {_resolve(args.train_output)}")
            print(f"Validation output: {_resolve(args.validation_output)}")
            print(f"Resumed: {summary.resumed}")
        if not args.skip_instructions:
            instruction_records = _load_instruction_dataset()
            instruction_summary = convert_instruction_records(
                instruction_records,
                _resolve(args.instruction_output),
            )
            print()
            print("Instruction data summary")
            print("========================")
            print(f"Total records: {instruction_summary.written_records}")
            print(f"Skipped records: {instruction_summary.skipped_records}")
            print(f"Duplicate records: {instruction_summary.duplicate_records}")
            print(f"Output: {instruction_summary.output_path}")
    except Exception as exc:  # noqa: BLE001
        _report_error(exc, debug=args.debug)
        return 1
    return 0


def _load_code_dataset(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise CodeDataDownloadError("datasets package is required.") from exc
    try:
        return load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    except Exception as exc:
        raise CodeDataDownloadError(f"Could not stream code dataset: {exc}") from exc


def _load_instruction_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise CodeDataDownloadError("datasets package is required.") from exc
    try:
        return load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    except Exception as exc:
        raise CodeDataDownloadError(f"Could not load instruction dataset: {exc}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Python code training data.")
    parser.add_argument("--code-gb", type=float, default=2)
    parser.add_argument("--shard-mb", type=int, default=200)
    parser.add_argument("--train-output", type=Path, default=Path("data/code_shards/train"))
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("data/code_shards/validation"),
    )
    parser.add_argument(
        "--instruction-output",
        type=Path,
        default=Path("data/fine_tuning/code_instructions.jsonl"),
    )
    parser.add_argument("--validation-percent", type=float, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--skip-instructions", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _directory_size(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / (1024 * 1024)


def _report_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        logging.exception("Download failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
