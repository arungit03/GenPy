"""Prepare GPT-style train, validation, and test datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import ConfigError, DatasetConfig, load_config
from genpy_llm.dataset import (
    DataLoaders,
    DatasetOutputPaths,
    DatasetPreparationError,
    DatasetPreparationStats,
    DatasetSplits,
    create_dataloaders,
    prepare_dataset,
    save_dataset_splits,
)
from genpy_llm.logging_utils import setup_logging
from genpy_llm.vocabulary import Vocabulary, VocabularyError


def main() -> int:
    """Parse arguments and prepare datasets."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        app_config = load_config(_resolve_optional_path(args.config))
        logger = setup_logging(
            log_dir=app_config.paths.logs_dir,
            log_file=app_config.logging.log_file,
            level="DEBUG" if args.debug else app_config.logging.level,
        )
        dataset_config = _override_dataset_config(app_config.dataset, args)

        input_path = _resolve_path(args.input, app_config.data.encoded_file)
        vocabulary_path = _resolve_path(args.vocabulary, app_config.data.vocabulary_file)
        metadata_path = _resolve_path(args.metadata_output, app_config.data.dataset_metadata_file)
        output_paths = _resolve_output_paths(args.dataset_dir, app_config)

        if not args.no_save:
            _validate_outputs(input_path, vocabulary_path, output_paths, metadata_path, args.force)

        splits, stats = prepare_dataset(
            input_path=input_path,
            vocabulary_path=vocabulary_path,
            config=dataset_config,
            encoding=app_config.data.encoding,
        )
        if not args.no_save and dataset_config.save_prepared_tensors:
            save_dataset_splits(
                splits=splits,
                output_paths=output_paths,
                metadata_path=metadata_path,
                stats=stats,
                project_root=app_config.project_root,
                encoding=app_config.data.encoding,
            )

        vocabulary = Vocabulary.load(vocabulary_path, config=app_config.vocabulary)
        loaders = create_dataloaders(splits, dataset_config) if args.show_batch else None
    except (
        ConfigError,
        DatasetPreparationError,
        FileExistsError,
        FileNotFoundError,
        IsADirectoryError,
        LookupError,
        OSError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Dataset preparation completed successfully.")
    _print_summary(stats, output_paths, metadata_path, saved=not args.no_save)
    if args.show_sample > 0:
        _print_samples(splits, vocabulary, args.show_sample)
    if loaders is not None:
        _print_batch(loaders)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GenPy LLM GPT datasets.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--input", type=str, default=None, help="Encoded JSONL input file.")
    parser.add_argument("--vocabulary", type=str, default=None, help="Vocabulary JSON file.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset output directory.")
    parser.add_argument("--metadata-output", type=str, default=None, help="Dataset metadata JSON.")
    parser.add_argument("--context-length", type=_positive_int, default=None)
    parser.add_argument("--stride", type=_positive_int, default=None)
    parser.add_argument("--sequence-mode", choices=["continuous", "per_sequence"], default=None)
    parser.add_argument("--short-sequence-policy", choices=["skip", "pad"], default=None)
    parser.add_argument("--split-unit", choices=["sequence", "sample"], default=None)
    parser.add_argument("--train-ratio", type=_ratio, default=None)
    parser.add_argument("--validation-ratio", type=_ratio, default=None)
    parser.add_argument("--test-ratio", type=_ratio, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--show-sample", type=_non_negative_int, default=0)
    parser.add_argument("--show-batch", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _override_dataset_config(config: DatasetConfig, args: argparse.Namespace) -> DatasetConfig:
    updates = {}
    for argument_name, field_name in {
        "context_length": "context_length",
        "stride": "stride",
        "sequence_mode": "sequence_mode",
        "short_sequence_policy": "short_sequence_policy",
        "split_unit": "split_unit",
        "train_ratio": "train_ratio",
        "validation_ratio": "validation_ratio",
        "test_ratio": "test_ratio",
        "split_seed": "split_seed",
        "batch_size": "batch_size",
    }.items():
        value = getattr(args, argument_name)
        if value is not None:
            updates[field_name] = value
    config = replace(config, **updates)
    _validate_dataset_config(config)
    if args.no_save:
        config = replace(config, save_prepared_tensors=False)
    return config


def _validate_dataset_config(config: DatasetConfig) -> None:
    if config.context_length <= 0:
        raise ConfigError("context_length must be greater than zero.")
    if config.stride <= 0:
        raise ConfigError("stride must be greater than zero.")
    ratio_sum = config.train_ratio + config.validation_ratio + config.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ConfigError("split ratios must sum to 1.0.")
    if config.batch_size <= 0:
        raise ConfigError("batch_size must be greater than zero.")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return number


def _ratio(value: str) -> float:
    number = float(value)
    if number < 0 or number > 1:
        raise argparse.ArgumentTypeError("ratio must be between 0 and 1.")
    return number


def _resolve_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return _resolve_against_project_root(Path(value))


def _resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    return _resolve_against_project_root(Path(value))


def _resolve_output_paths(dataset_dir_value: str | None, app_config: object) -> DatasetOutputPaths:
    if dataset_dir_value is not None:
        dataset_dir = _resolve_against_project_root(Path(dataset_dir_value))
        return DatasetOutputPaths(
            train=dataset_dir / "train.pt",
            validation=dataset_dir / "validation.pt",
            test=dataset_dir / "test.pt",
        )
    return DatasetOutputPaths(
        train=app_config.data.train_dataset_file,
        validation=app_config.data.validation_dataset_file,
        test=app_config.data.test_dataset_file,
    )


def _resolve_against_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_outputs(
    input_path: Path,
    vocabulary_path: Path,
    output_paths: DatasetOutputPaths,
    metadata_path: Path,
    force: bool,
) -> None:
    del force
    protected = {input_path.resolve(), vocabulary_path.resolve()}
    for path in [output_paths.train, output_paths.validation, output_paths.test, metadata_path]:
        if path.resolve() in protected:
            raise ValueError("Dataset outputs must not overwrite input or vocabulary files.")
        if path.exists() and path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {path}")


def _print_summary(
    stats: DatasetPreparationStats,
    output_paths: DatasetOutputPaths,
    metadata_path: Path,
    saved: bool,
) -> None:
    print("GenPy LLM Dataset Preparation Complete")
    print("======================================")
    print(stats.summary())
    print()
    print(f"Saved tensors: {saved}")
    print(f"Train file: {output_paths.train}")
    print(f"Validation file: {output_paths.validation}")
    print(f"Test file: {output_paths.test}")
    print(f"Metadata file: {metadata_path}")


def _print_samples(splits: DatasetSplits, vocabulary: Vocabulary, count: int) -> None:
    shown = 0
    for split_name, dataset in [
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ]:
        for index in range(len(dataset)):
            if shown >= count:
                return
            sample = dataset[index]
            input_ids = sample["input_ids"].tolist()
            target_ids = sample["target_ids"].tolist()
            print()
            print(f"Sample {shown} ({split_name})")
            print("Input IDs:")
            print(_shorten(input_ids))
            print("Target IDs:")
            print(_shorten(target_ids))
            print("Input tokens:")
            print(_shorten(vocabulary.decode(input_ids)))
            print("Target tokens:")
            print(_shorten(vocabulary.decode(target_ids)))
            print("Attention mask:")
            print(_shorten(sample["attention_mask"].tolist()))
            shown += 1


def _print_batch(loaders: DataLoaders) -> None:
    print()
    print("Training batch")
    print("==============")
    if len(loaders.train.dataset) == 0:
        print("Training split is empty.")
        return
    batch = next(iter(loaders.train))
    print(f"input_ids shape:      {batch['input_ids'].shape}")
    print(f"target_ids shape:     {batch['target_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"dtype: {batch['input_ids'].dtype}")
    print(f"device: {batch['input_ids'].device}")


def _shorten(values: list[object], limit: int = 32) -> list[object]:
    if len(values) <= limit:
        return values
    return [*values[:limit], "..."]


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Dataset preparation failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
