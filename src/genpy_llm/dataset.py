"""GPT-style dataset preparation for next-token prediction."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from genpy_llm.compat import zip_strict
from genpy_llm.config import DatasetConfig
from genpy_llm.vocabulary import Vocabulary

DATASET_FORMAT_VERSION = 1


class DatasetPreparationError(ValueError):
    """Raised when encoded data or dataset configuration is invalid."""


@dataclass(frozen=True)
class EncodedSequence:
    """One encoded sequence from Step 4 JSONL."""

    sequence_id: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrainingSample:
    """One prepared input-target pair before tensor conversion."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    source_sequence_id: int | None


@dataclass(frozen=True)
class DatasetSplits:
    """Prepared train, validation, and test datasets."""

    train: GPTDataset
    validation: GPTDataset
    test: GPTDataset


@dataclass(frozen=True)
class DataLoaders:
    """PyTorch DataLoaders for each split."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


@dataclass(frozen=True)
class DatasetOutputPaths:
    """Output paths for saved dataset splits."""

    train: Path
    validation: Path
    test: Path


@dataclass(frozen=True)
class DatasetPreparationStats:
    """Summary of dataset preparation."""

    input_file: Path
    vocabulary_file: Path
    source_sequences: int
    empty_sequences_skipped: int
    short_sequences_skipped: int
    source_token_count: int
    total_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    context_length: int
    stride: int
    padded_samples: int
    padding_tokens_added: int
    minimum_token_id: int | None
    maximum_token_id: int | None
    sequence_mode: str
    split_unit: str
    split_seed: int

    def summary(self) -> str:
        """Return a readable multi-line summary."""

        return "\n".join(
            [
                "Dataset preparation summary",
                "===========================",
                f"Input file: {self.input_file}",
                f"Vocabulary file: {self.vocabulary_file}",
                f"Source sequences: {self.source_sequences}",
                f"Source tokens: {self.source_token_count}",
                f"Empty sequences skipped: {self.empty_sequences_skipped}",
                f"Short sequences skipped: {self.short_sequences_skipped}",
                f"Total samples: {self.total_samples}",
                f"Train samples: {self.train_samples}",
                f"Validation samples: {self.validation_samples}",
                f"Test samples: {self.test_samples}",
                f"Context length: {self.context_length}",
                f"Stride: {self.stride}",
                f"Padded samples: {self.padded_samples}",
                f"Padding tokens added: {self.padding_tokens_added}",
                f"Minimum token ID: {self.minimum_token_id}",
                f"Maximum token ID: {self.maximum_token_id}",
                f"Sequence mode: {self.sequence_mode}",
                f"Split unit: {self.split_unit}",
                f"Split seed: {self.split_seed}",
            ]
        )

    def __str__(self) -> str:
        """Return the same readable summary used by the CLI."""

        return self.summary()


@dataclass
class _PreparationCounters:
    empty_sequences_skipped: int = 0
    short_sequences_skipped: int = 0
    padded_samples: int = 0
    padding_tokens_added: int = 0


class GPTDataset(Dataset[dict[str, torch.Tensor]]):
    """A small PyTorch Dataset for GPT next-token prediction."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.attention_mask = attention_mask
        self._validate_tensors()

    def __len__(self) -> int:
        """Return the number of samples."""

        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample dictionary."""

        if not isinstance(index, int):
            raise IndexError("Dataset index must be an integer.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")
        item = {
            "input_ids": self.input_ids[index],
            "target_ids": self.target_ids[index],
        }
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[index]
        return item

    @property
    def context_length(self) -> int:
        """Return the second tensor dimension."""

        return int(self.input_ids.shape[1])

    def _validate_tensors(self) -> None:
        for name, tensor in {"input_ids": self.input_ids, "target_ids": self.target_ids}.items():
            if not isinstance(tensor, torch.Tensor):
                raise DatasetPreparationError(f"{name} must be a torch.Tensor.")
            if tensor.ndim != 2:
                raise DatasetPreparationError(f"{name} must be two-dimensional.")
            if tensor.dtype != torch.long:
                raise DatasetPreparationError(f"{name} must use torch.long dtype.")
        if self.input_ids.shape != self.target_ids.shape:
            raise DatasetPreparationError("input_ids and target_ids must have the same shape.")
        if self.attention_mask is not None:
            if self.attention_mask.shape != self.input_ids.shape:
                raise DatasetPreparationError("attention_mask shape must match input_ids.")
            if self.attention_mask.dtype != torch.long:
                raise DatasetPreparationError("attention_mask must use torch.long dtype.")


def read_encoded_sequences(
    input_path: Path,
    vocabulary_size: int,
    encoding: str = "utf-8",
) -> Iterator[EncodedSequence]:
    """Yield validated encoded sequences from Step 4 JSONL."""

    input_path = input_path.resolve()
    _validate_input_path(input_path, "Input")
    with input_path.open("r", encoding=encoding) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            yield _parse_encoded_record(line, line_number, vocabulary_size)


def create_training_windows(
    token_ids: Sequence[int],
    context_length: int,
    stride: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Create shifted input and target windows from a token ID stream."""

    _validate_window_args(context_length, stride)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    required_length = context_length + 1
    for start in range(0, max(len(token_ids) - required_length + 1, 0), stride):
        window = list(token_ids[start : start + required_length])
        if len(window) == required_length:
            inputs.append(window[:context_length])
            targets.append(window[1:])
    return inputs, targets


def prepare_dataset(
    input_path: Path,
    vocabulary_path: Path,
    config: DatasetConfig,
    encoding: str = "utf-8",
) -> tuple[DatasetSplits, DatasetPreparationStats]:
    """Prepare train, validation, and test GPT datasets from encoded JSONL."""

    vocabulary = Vocabulary.load(vocabulary_path, encoding=encoding)
    sequences = list(read_encoded_sequences(input_path, len(vocabulary), encoding=encoding))
    if not vocabulary_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_path}")

    counters = _PreparationCounters(
        empty_sequences_skipped=sum(1 for sequence in sequences if not sequence.token_ids)
    )
    non_empty_sequences = [sequence for sequence in sequences if sequence.token_ids]
    source_token_count = sum(len(sequence.token_ids) for sequence in non_empty_sequences)
    all_token_ids = [
        token_id for sequence in non_empty_sequences for token_id in sequence.token_ids
    ]

    if config.split_unit == "sequence":
        train_sequences, validation_sequences, test_sequences = _split_items(
            non_empty_sequences,
            config,
        )
        train_samples = _samples_from_sequences(train_sequences, vocabulary, config, counters)
        validation_samples = _samples_from_sequences(
            validation_sequences, vocabulary, config, counters
        )
        test_samples = _samples_from_sequences(test_sequences, vocabulary, config, counters)
    else:
        all_samples = _samples_from_sequences(non_empty_sequences, vocabulary, config, counters)
        train_samples, validation_samples, test_samples = _split_items(all_samples, config)

    total_samples = len(train_samples) + len(validation_samples) + len(test_samples)
    if total_samples == 0:
        raise DatasetPreparationError("Dataset preparation produced zero samples.")

    splits = DatasetSplits(
        train=_dataset_from_samples(train_samples, config.context_length),
        validation=_dataset_from_samples(validation_samples, config.context_length),
        test=_dataset_from_samples(test_samples, config.context_length),
    )
    stats = DatasetPreparationStats(
        input_file=input_path.resolve(),
        vocabulary_file=vocabulary_path.resolve(),
        source_sequences=len(sequences),
        empty_sequences_skipped=counters.empty_sequences_skipped,
        short_sequences_skipped=counters.short_sequences_skipped,
        source_token_count=source_token_count,
        total_samples=total_samples,
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        test_samples=len(test_samples),
        context_length=config.context_length,
        stride=config.stride,
        padded_samples=counters.padded_samples,
        padding_tokens_added=counters.padding_tokens_added,
        minimum_token_id=min(all_token_ids) if all_token_ids else None,
        maximum_token_id=max(all_token_ids) if all_token_ids else None,
        sequence_mode=config.sequence_mode,
        split_unit=config.split_unit,
        split_seed=config.split_seed,
    )
    return splits, stats


def create_dataloaders(
    splits: DatasetSplits,
    config: DatasetConfig,
) -> DataLoaders:
    """Create configured PyTorch DataLoaders."""

    generator = torch.Generator()
    generator.manual_seed(config.split_seed)
    return DataLoaders(
        train=DataLoader(
            splits.train,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last_train,
            generator=generator if config.shuffle_train else None,
        ),
        validation=DataLoader(
            splits.validation,
            batch_size=config.batch_size,
            shuffle=config.shuffle_validation,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last_validation,
        ),
        test=DataLoader(
            splits.test,
            batch_size=config.batch_size,
            shuffle=config.shuffle_test,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last_test,
        ),
    )


def save_dataset_splits(
    splits: DatasetSplits,
    output_paths: DatasetOutputPaths,
    metadata_path: Path,
    stats: DatasetPreparationStats,
    project_root: Path | None = None,
    encoding: str = "utf-8",
) -> None:
    """Save prepared tensors and metadata atomically."""

    _save_dataset_split(output_paths.train, "train", splits.train)
    _save_dataset_split(output_paths.validation, "validation", splits.validation)
    _save_dataset_split(output_paths.test, "test", splits.test)
    save_dataset_metadata(metadata_path, stats, output_paths, project_root, encoding)


def load_dataset_split(path: Path) -> GPTDataset:
    """Load and validate one saved dataset split."""

    path = path.resolve()
    _validate_input_path(path, "Dataset split")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise DatasetPreparationError("Saved dataset split must contain a dictionary.")
    if data.get("format_version") != DATASET_FORMAT_VERSION:
        raise DatasetPreparationError(
            f"Unsupported dataset format version: {data.get('format_version')}"
        )
    split = data.get("split")
    if split not in {"train", "validation", "test"}:
        raise DatasetPreparationError("Saved dataset split name is invalid.")
    context_length = data.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        raise DatasetPreparationError("Saved context_length is invalid.")
    dataset = GPTDataset(
        input_ids=data.get("input_ids"),
        target_ids=data.get("target_ids"),
        attention_mask=data.get("attention_mask"),
    )
    if dataset.context_length != context_length:
        raise DatasetPreparationError("Saved context_length does not match tensor shape.")
    return dataset


def save_dataset_metadata(
    metadata_path: Path,
    stats: DatasetPreparationStats,
    output_paths: DatasetOutputPaths,
    project_root: Path | None = None,
    encoding: str = "utf-8",
) -> None:
    """Save dataset preparation metadata as JSON."""

    project_root = project_root or Path.cwd()
    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "source_file": _portable_path(stats.input_file, project_root),
        "vocabulary_file": _portable_path(stats.vocabulary_file, project_root),
        "train_dataset_file": _portable_path(output_paths.train, project_root),
        "validation_dataset_file": _portable_path(output_paths.validation, project_root),
        "test_dataset_file": _portable_path(output_paths.test, project_root),
        "sequence_mode": stats.sequence_mode,
        "split_unit": stats.split_unit,
        "context_length": stats.context_length,
        "stride": stats.stride,
        "source_sequences": stats.source_sequences,
        "source_tokens": stats.source_token_count,
        "total_samples": stats.total_samples,
        "train_samples": stats.train_samples,
        "validation_samples": stats.validation_samples,
        "test_samples": stats.test_samples,
        "padded_samples": stats.padded_samples,
        "padding_tokens_added": stats.padding_tokens_added,
        "split_seed": stats.split_seed,
        "minimum_token_id": stats.minimum_token_id,
        "maximum_token_id": stats.maximum_token_id,
    }
    _write_json_atomic(metadata_path.resolve(), metadata, encoding)


def _samples_from_sequences(
    sequences: Sequence[EncodedSequence],
    vocabulary: Vocabulary,
    config: DatasetConfig,
    counters: _PreparationCounters,
) -> list[TrainingSample]:
    if config.sequence_mode == "continuous":
        stream = _combined_stream(sequences, vocabulary, config)
        return _samples_from_token_ids(stream, None, vocabulary.pad_id, config, counters)

    samples: list[TrainingSample] = []
    for sequence in sequences:
        samples.extend(
            _samples_from_token_ids(
                sequence.token_ids,
                sequence.sequence_id,
                vocabulary.pad_id,
                config,
                counters,
            )
        )
    return samples


def _combined_stream(
    sequences: Sequence[EncodedSequence],
    vocabulary: Vocabulary,
    config: DatasetConfig,
) -> list[int]:
    stream: list[int] = []
    for sequence in sequences:
        if not sequence.token_ids:
            continue
        if stream and config.add_eos_between_sequences and stream[-1] != vocabulary.eos_id:
            stream.append(vocabulary.eos_id)
        stream.extend(sequence.token_ids)
    return stream


def _samples_from_token_ids(
    token_ids: Sequence[int],
    source_sequence_id: int | None,
    pad_id: int,
    config: DatasetConfig,
    counters: _PreparationCounters,
) -> list[TrainingSample]:
    if not token_ids:
        return []
    required_length = config.context_length + 1
    if len(token_ids) < required_length:
        if config.short_sequence_policy == "skip":
            counters.short_sequences_skipped += 1
            return []
        padding_needed = required_length - len(token_ids)
        padded = list(token_ids) + [pad_id] * padding_needed
        counters.padded_samples += 1
        counters.padding_tokens_added += padding_needed
        return [
            TrainingSample(
                input_ids=tuple(padded[: config.context_length]),
                target_ids=tuple(padded[1:]),
                attention_mask=tuple(
                    1 if position < len(token_ids) else 0
                    for position in range(config.context_length)
                ),
                source_sequence_id=source_sequence_id,
            )
        ]

    inputs, targets = create_training_windows(token_ids, config.context_length, config.stride)
    return [
        TrainingSample(
            input_ids=tuple(input_ids),
            target_ids=tuple(target_ids),
            attention_mask=(1,) * config.context_length,
            source_sequence_id=source_sequence_id,
        )
        for input_ids, target_ids in zip_strict(inputs, targets)
    ]


def _split_items(
    items: Sequence[Any], config: DatasetConfig
) -> tuple[list[Any], list[Any], list[Any]]:
    indices = list(range(len(items)))
    if config.shuffle_before_split:
        rng = random.Random(config.split_seed)
        rng.shuffle(indices)

    train_count, validation_count, _test_count = _split_counts(len(indices), config)
    train_indices = set(indices[:train_count])
    validation_indices = set(indices[train_count : train_count + validation_count])
    test_indices = set(indices[train_count + validation_count :])

    train = [item for index, item in enumerate(items) if index in train_indices]
    validation = [item for index, item in enumerate(items) if index in validation_indices]
    test = [item for index, item in enumerate(items) if index in test_indices]
    return train, validation, test


def _split_counts(total: int, config: DatasetConfig) -> tuple[int, int, int]:
    desired = [
        total * config.train_ratio,
        total * config.validation_ratio,
        total * config.test_ratio,
    ]
    counts = [int(value) for value in desired]
    remainder = total - sum(counts)
    remainders = sorted(
        enumerate(value - int(value) for value in desired),
        key=lambda item: (-item[1], item[0]),
    )
    for index, _fraction in remainders[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _dataset_from_samples(samples: Sequence[TrainingSample], context_length: int) -> GPTDataset:
    if not samples:
        empty = torch.empty((0, context_length), dtype=torch.long)
        return GPTDataset(empty, empty.clone(), empty.clone())

    input_tensor = torch.tensor([sample.input_ids for sample in samples], dtype=torch.long)
    target_tensor = torch.tensor([sample.target_ids for sample in samples], dtype=torch.long)
    mask_tensor = torch.tensor([sample.attention_mask for sample in samples], dtype=torch.long)
    return GPTDataset(input_tensor, target_tensor, mask_tensor)


def _parse_encoded_record(line: str, line_number: int, vocabulary_size: int) -> EncodedSequence:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetPreparationError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise DatasetPreparationError(f"JSONL line {line_number} must contain a JSON object.")
    if "sequence_id" not in record:
        raise DatasetPreparationError(f"JSONL line {line_number} is missing sequence_id.")
    if not isinstance(record["sequence_id"], int) or isinstance(record["sequence_id"], bool):
        raise DatasetPreparationError(f"JSONL line {line_number} sequence_id must be an integer.")
    if "token_ids" not in record:
        raise DatasetPreparationError(f"JSONL line {line_number} is missing token_ids.")
    token_ids = record["token_ids"]
    if not isinstance(token_ids, list):
        raise DatasetPreparationError(f"JSONL line {line_number} token_ids must be a list.")
    for position, token_id in enumerate(token_ids):
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise DatasetPreparationError(
                f"JSONL line {line_number} token ID at position {position} must be an integer."
            )
        if token_id < 0:
            raise DatasetPreparationError(
                f"JSONL line {line_number} token ID at position {position} is negative."
            )
        if token_id >= vocabulary_size:
            raise DatasetPreparationError(
                f"JSONL line {line_number} token ID {token_id} is outside vocabulary size."
            )
    if "token_count" in record and record["token_count"] != len(token_ids):
        raise DatasetPreparationError(
            f"JSONL line {line_number} token_count does not match token_ids length."
        )
    if (
        "tokens" in record
        and isinstance(record["tokens"], list)
        and len(record["tokens"]) != len(token_ids)
    ):
        raise DatasetPreparationError(
            f"JSONL line {line_number} tokens length does not match token_ids length."
        )
    return EncodedSequence(sequence_id=record["sequence_id"], token_ids=tuple(token_ids))


def _save_dataset_split(path: Path, split: str, dataset: GPTDataset) -> None:
    path = path.resolve()
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Dataset output path is a directory: {path}")
    payload = {
        "format_version": DATASET_FORMAT_VERSION,
        "split": split,
        "context_length": dataset.context_length,
        "input_ids": dataset.input_ids.cpu(),
        "target_ids": dataset.target_ids.cpu(),
        "attention_mask": dataset.attention_mask.cpu()
        if dataset.attention_mask is not None
        else None,
    }
    _torch_save_atomic(path, payload)


def _torch_save_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        temp_path = _create_temp_path(path)
        torch.save(payload, temp_path)
        temp_path.replace(path)
        temp_path = None
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _write_json_atomic(path: Path, data: dict[str, Any], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        temp_path = _create_temp_path(path)
        with temp_path.open("w", encoding=encoding, newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temp_path.replace(path)
        temp_path = None
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _validate_input_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} path is not a file: {path}")


def _validate_window_args(context_length: int, stride: int) -> None:
    if context_length <= 0:
        raise DatasetPreparationError("context_length must be greater than zero.")
    if stride <= 0:
        raise DatasetPreparationError("stride must be greater than zero.")


def _create_temp_path(output_path: Path) -> Path:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    os.close(file_descriptor)
    return Path(temp_name)


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name
