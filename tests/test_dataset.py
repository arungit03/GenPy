from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from genpy_llm.config import DatasetConfig
from genpy_llm.dataset import (
    DATASET_FORMAT_VERSION,
    DatasetOutputPaths,
    DatasetPreparationError,
    GPTDataset,
    create_dataloaders,
    create_training_windows,
    load_dataset_split,
    prepare_dataset,
    read_encoded_sequences,
    save_dataset_splits,
)
from genpy_llm.preprocessing import TextPreprocessor
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.vocabulary import Vocabulary
from tests.test_preprocessing import make_config as make_preprocessing_config
from tests.test_tokenization import make_config as make_tokenization_config
from tests.test_vocabulary import make_config as make_vocabulary_config


def make_config(**overrides: object) -> DatasetConfig:
    values = {
        "context_length": 4,
        "stride": 4,
        "sequence_mode": "continuous",
        "short_sequence_policy": "pad",
        "add_eos_between_sequences": False,
        "split_unit": "sequence",
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "split_seed": 42,
        "shuffle_before_split": True,
        "batch_size": 2,
        "shuffle_train": True,
        "shuffle_validation": False,
        "shuffle_test": False,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last_train": False,
        "drop_last_validation": False,
        "drop_last_test": False,
        "save_prepared_tensors": True,
    }
    values.update(overrides)
    return DatasetConfig(**values)


def write_encoded_jsonl(path: Path, sequences: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sequence_id, token_ids in enumerate(sequences):
            file.write(
                json.dumps(
                    {
                        "sequence_id": sequence_id,
                        "tokens": [str(token_id) for token_id in token_ids],
                        "token_ids": token_ids,
                        "token_count": len(token_ids),
                    }
                )
            )
            file.write("\n")


def save_vocab(path: Path, extra_tokens: list[str] | None = None) -> Vocabulary:
    tokens = [["token", "<EOS>", *(extra_tokens or [])]]
    vocabulary = Vocabulary.build(tokens, make_vocabulary_config())
    vocabulary.save(path)
    return vocabulary


def prepare_files(tmp_path: Path, sequences: list[list[int]]) -> tuple[Path, Path]:
    input_path = tmp_path / "encoded.jsonl"
    vocabulary_path = tmp_path / "vocab.json"
    write_encoded_jsonl(input_path, sequences)
    save_vocab(vocabulary_path, [str(i) for i in range(50)])
    return input_path, vocabulary_path


def test_basic_sliding_window_creation() -> None:
    inputs, targets = create_training_windows([0, 1, 2, 3, 4, 5], 4, 1)

    assert inputs[0] == [0, 1, 2, 3]
    assert targets[0] == [1, 2, 3, 4]


def test_context_length_one() -> None:
    assert create_training_windows([1, 2, 3], 1, 1) == ([[1], [2]], [[2], [3]])


def test_stride_equal_to_context_length() -> None:
    inputs, _targets = create_training_windows([0, 1, 2, 3, 4, 5, 6], 3, 3)

    assert inputs == [[0, 1, 2], [3, 4, 5]]


def test_stride_greater_than_context_length() -> None:
    inputs, _targets = create_training_windows([0, 1, 2, 3, 4, 5, 6, 7], 2, 3)

    assert inputs == [[0, 1], [3, 4]]


def test_exact_context_plus_one_input() -> None:
    inputs, targets = create_training_windows([1, 2, 3], 2, 2)

    assert inputs == [[1, 2]]
    assert targets == [[2, 3]]


def test_input_shorter_than_required() -> None:
    assert create_training_windows([1, 2], 2, 1) == ([], [])


def test_invalid_context_length() -> None:
    with pytest.raises(DatasetPreparationError):
        create_training_windows([1, 2], 0, 1)


def test_invalid_stride() -> None:
    with pytest.raises(DatasetPreparationError):
        create_training_windows([1, 2], 1, 0)


def test_continuous_sequence_mode(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6, 7], [8, 9, 10]])

    splits, stats = prepare_dataset(
        input_path, vocabulary_path, make_config(shuffle_before_split=False)
    )

    assert stats.sequence_mode == "continuous"
    assert stats.total_samples >= 1
    assert splits.train[0]["input_ids"].shape == torch.Size([4])


def test_per_sequence_mode(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6, 7, 8, 9]])

    _splits, stats = prepare_dataset(
        input_path,
        vocabulary_path,
        make_config(sequence_mode="per_sequence", shuffle_before_split=False),
    )

    assert stats.total_samples == 1


def test_eos_insertion_between_sequences(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6], [7, 8, 9]])
    vocabulary = Vocabulary.load(vocabulary_path, make_vocabulary_config())

    splits, _stats = prepare_dataset(
        input_path,
        vocabulary_path,
        make_config(add_eos_between_sequences=True, shuffle_before_split=False),
    )

    assert vocabulary.eos_id in splits.train[0]["input_ids"].tolist()


def test_avoiding_duplicate_eos_insertion(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 3], [7, 8, 9]])

    splits, _stats = prepare_dataset(
        input_path,
        vocabulary_path,
        make_config(add_eos_between_sequences=True, shuffle_before_split=False),
    )

    assert splits.train[0]["input_ids"].tolist().count(3) == 1


def test_short_sequence_skip_policy(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6]])

    with pytest.raises(DatasetPreparationError):
        prepare_dataset(input_path, vocabulary_path, make_config(short_sequence_policy="skip"))


def test_short_sequence_pad_policy(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6]])

    splits, stats = prepare_dataset(
        input_path, vocabulary_path, make_config(shuffle_before_split=False)
    )

    assert stats.padded_samples == 1
    assert splits.train[0]["input_ids"].shape == torch.Size([4])


def test_correct_pad_token_id_usage(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6]])
    vocabulary = Vocabulary.load(vocabulary_path, make_vocabulary_config())

    splits, _stats = prepare_dataset(
        input_path, vocabulary_path, make_config(shuffle_before_split=False)
    )

    assert vocabulary.pad_id in splits.train[0]["input_ids"].tolist()


def test_target_padding_and_mask(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6]])

    splits, _stats = prepare_dataset(
        input_path, vocabulary_path, make_config(shuffle_before_split=False)
    )
    sample = splits.train[0]

    assert sample["target_ids"].tolist()[-1] == 0
    assert sample["attention_mask"].tolist() == [1, 1, 0, 0]


def test_empty_sequence_skipping(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[]])

    with pytest.raises(DatasetPreparationError):
        prepare_dataset(input_path, vocabulary_path, make_config())


def test_deterministic_sample_splitting(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(
        tmp_path, [[i, i + 1, i + 2, i + 3, i + 4] for i in range(5, 15)]
    )
    config = make_config(split_unit="sample", split_seed=7)

    first, _stats = prepare_dataset(input_path, vocabulary_path, config)
    second, _stats = prepare_dataset(input_path, vocabulary_path, config)

    assert torch.equal(first.train.input_ids, second.train.input_ids)


def test_different_seeds_producing_different_splits(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(
        tmp_path, [[i, i + 1, i + 2, i + 3, i + 4] for i in range(5, 20)]
    )

    first, _stats = prepare_dataset(
        input_path, vocabulary_path, make_config(split_unit="sample", split_seed=1)
    )
    second, _stats = prepare_dataset(
        input_path, vocabulary_path, make_config(split_unit="sample", split_seed=2)
    )

    assert not torch.equal(first.train.input_ids, second.train.input_ids)


def test_no_sample_loss_across_splits(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(
        tmp_path, [[i, i + 1, i + 2, i + 3, i + 4] for i in range(5, 15)]
    )

    _splits, stats = prepare_dataset(input_path, vocabulary_path, make_config(split_unit="sample"))

    assert (
        stats.train_samples + stats.validation_samples + stats.test_samples == stats.total_samples
    )


def test_zero_sized_validation_split(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6, 7, 8, 9]])

    _splits, stats = prepare_dataset(
        input_path, vocabulary_path, make_config(validation_ratio=0.0, test_ratio=0.2)
    )

    assert stats.validation_samples == 0


def test_zero_sized_test_split(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6, 7, 8, 9]])

    _splits, stats = prepare_dataset(
        input_path, vocabulary_path, make_config(validation_ratio=0.2, test_ratio=0.0)
    )

    assert stats.test_samples == 0


def test_gptdataset_length_and_item() -> None:
    dataset = GPTDataset(torch.ones((2, 4), dtype=torch.long), torch.ones((2, 4), dtype=torch.long))

    assert len(dataset) == 2
    assert set(dataset[0]) == {"input_ids", "target_ids"}


def test_gptdataset_with_attention_mask_keys() -> None:
    dataset = GPTDataset(
        torch.ones((1, 4), dtype=torch.long),
        torch.ones((1, 4), dtype=torch.long),
        torch.ones((1, 4), dtype=torch.long),
    )

    assert set(dataset[0]) == {"input_ids", "target_ids", "attention_mask"}


def test_tensor_shapes_and_dtype() -> None:
    dataset = GPTDataset(torch.ones((1, 4), dtype=torch.long), torch.ones((1, 4), dtype=torch.long))

    assert dataset.input_ids.shape == torch.Size([1, 4])
    assert dataset.input_ids.dtype == torch.long


def test_rejecting_mismatched_shapes() -> None:
    with pytest.raises(DatasetPreparationError):
        GPTDataset(torch.ones((1, 4), dtype=torch.long), torch.ones((2, 4), dtype=torch.long))


def test_rejecting_non_2d_tensors() -> None:
    with pytest.raises(DatasetPreparationError):
        GPTDataset(torch.ones((4,), dtype=torch.long), torch.ones((4,), dtype=torch.long))


def test_rejecting_non_long_tensors() -> None:
    with pytest.raises(DatasetPreparationError):
        GPTDataset(torch.ones((1, 4)), torch.ones((1, 4), dtype=torch.long))


def test_attention_mask_validation() -> None:
    with pytest.raises(DatasetPreparationError):
        GPTDataset(
            torch.ones((1, 4), dtype=torch.long),
            torch.ones((1, 4), dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
        )


def test_empty_split_tensor_shape() -> None:
    dataset = GPTDataset(
        torch.empty((0, 4), dtype=torch.long), torch.empty((0, 4), dtype=torch.long)
    )

    assert dataset.input_ids.shape == torch.Size([0, 4])


def test_dataloader_batch_shape() -> None:
    dataset = GPTDataset(
        torch.ones((3, 4), dtype=torch.long),
        torch.ones((3, 4), dtype=torch.long),
        torch.ones((3, 4), dtype=torch.long),
    )
    loaders = create_dataloaders(
        splits=type("Splits", (), {"train": dataset, "validation": dataset, "test": dataset})(),
        config=make_config(batch_size=2, shuffle_train=False),
    )

    batch = next(iter(loaders.train))
    assert batch["input_ids"].shape == torch.Size([2, 4])


def test_validation_and_test_dataloaders_not_shuffled() -> None:
    dataset = GPTDataset(
        torch.arange(12).view(3, 4),
        torch.arange(12).view(3, 4),
        torch.ones((3, 4), dtype=torch.long),
    )
    splits = type("Splits", (), {"train": dataset, "validation": dataset, "test": dataset})()
    loaders = create_dataloaders(splits, make_config(batch_size=1, shuffle_train=False))

    assert next(iter(loaders.validation))["input_ids"][0, 0].item() == 0
    assert next(iter(loaders.test))["input_ids"][0, 0].item() == 0


def test_jsonl_reading_success(tmp_path: Path) -> None:
    input_path = tmp_path / "encoded.jsonl"
    write_encoded_jsonl(input_path, [[1, 2]])

    assert list(read_encoded_sequences(input_path, vocabulary_size=10))[0].token_ids == (1, 2)


def test_malformed_jsonl_handling(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text("{bad}\n", encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="line 1"):
        list(read_encoded_sequences(input_path, 10))


@pytest.mark.parametrize(
    "record,error",
    [
        ({"token_ids": [1]}, "sequence_id"),
        ({"sequence_id": "0", "token_ids": [1]}, "sequence_id"),
        ({"sequence_id": 0}, "token_ids"),
        ({"sequence_id": 0, "token_ids": "1"}, "list"),
        ({"sequence_id": 0, "token_ids": [True]}, "integer"),
        ({"sequence_id": 0, "token_ids": [-1]}, "negative"),
        ({"sequence_id": 0, "token_ids": [99]}, "outside"),
        ({"sequence_id": 0, "token_ids": [1], "token_count": 2}, "token_count"),
        ({"sequence_id": 0, "token_ids": [1], "tokens": ["a", "b"]}, "tokens length"),
    ],
)
def test_jsonl_validation_errors(tmp_path: Path, record: dict[str, object], error: str) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match=error):
        list(read_encoded_sequences(input_path, 10))


def test_missing_input_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(read_encoded_sequences(tmp_path / "missing.jsonl", 10))


def test_input_path_being_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        list(read_encoded_sequences(tmp_path, 10))


def test_missing_vocabulary_file(tmp_path: Path) -> None:
    input_path = tmp_path / "encoded.jsonl"
    write_encoded_jsonl(input_path, [[1, 2]])

    with pytest.raises(FileNotFoundError):
        prepare_dataset(input_path, tmp_path / "missing_vocab.json", make_config())


def test_dataset_statistics_accuracy(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6]])

    _splits, stats = prepare_dataset(
        input_path, vocabulary_path, make_config(shuffle_before_split=False)
    )

    assert stats.source_sequences == 1
    assert stats.source_token_count == 2
    assert stats.padded_samples == 1


def test_saving_and_loading_splits(tmp_path: Path) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6]])
    splits, stats = prepare_dataset(
        input_path, vocabulary_path, make_config(shuffle_before_split=False)
    )
    output_paths = DatasetOutputPaths(
        tmp_path / "train.pt", tmp_path / "validation.pt", tmp_path / "test.pt"
    )

    save_dataset_splits(splits, output_paths, tmp_path / "metadata.json", stats, tmp_path)

    assert output_paths.train.exists()
    assert output_paths.validation.exists()
    assert output_paths.test.exists()
    assert (tmp_path / "metadata.json").exists()
    loaded = load_dataset_split(output_paths.train)
    assert torch.equal(loaded.input_ids, splits.train.input_ids)


def test_unsupported_format_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format_version": DATASET_FORMAT_VERSION + 1}, path)

    with pytest.raises(DatasetPreparationError, match="Unsupported"):
        load_dataset_split(path)


def test_corrupted_saved_dataset_detection(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {"format_version": DATASET_FORMAT_VERSION, "split": "train", "context_length": 4}, path
    )

    with pytest.raises(DatasetPreparationError):
        load_dataset_split(path)


def test_temporary_file_cleanup_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, vocabulary_path = prepare_files(tmp_path, [[5, 6]])
    splits, stats = prepare_dataset(
        input_path, vocabulary_path, make_config(shuffle_before_split=False)
    )
    output_paths = DatasetOutputPaths(
        tmp_path / "train.pt", tmp_path / "validation.pt", tmp_path / "test.pt"
    )

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise OSError("save failed")

    monkeypatch.setattr("genpy_llm.dataset.torch.save", fail_save)

    with pytest.raises(OSError):
        save_dataset_splits(splits, output_paths, tmp_path / "metadata.json", stats, tmp_path)

    assert not list(tmp_path.glob(".train.pt.*.tmp"))


def test_existing_steps_1_to_4_remain_functional() -> None:
    preprocessor = TextPreprocessor(make_preprocessing_config())
    tokenizer = TextTokenizer(make_tokenization_config(add_eos_token=True))
    vocabulary = Vocabulary.build([["Hello", "World", "!", "<EOS>"]], make_vocabulary_config())

    tokens = tokenizer.tokenize(preprocessor.clean_text("Hello     World!"))
    ids = vocabulary.encode(tokens)

    assert vocabulary.decode(ids) == tokens
