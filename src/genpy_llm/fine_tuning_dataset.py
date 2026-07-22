"""Fine-tuning dataset loading and preprocessing for GenPy Code LLM."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from genpy_llm.code_tokenizer import CodeTokenizer


class FineTuningDatasetError(ValueError):
    """Raised when fine-tuning data cannot be loaded or encoded."""


@dataclass(frozen=True)
class FineTuningExample:
    """One instruction/input/output fine-tuning example."""

    instruction: str = ""
    input: str = ""
    output: str = ""


@dataclass(frozen=True)
class FineTuningSplit:
    """Deterministic train/validation split."""

    train_examples: tuple[FineTuningExample, ...]
    validation_examples: tuple[FineTuningExample, ...]


@dataclass(frozen=True)
class FineTuningDatasetStats:
    """Summary of dataset preprocessing decisions."""

    examples: int
    max_length: int
    truncated_examples: int
    padded_examples: int


class FineTuningDataset(Dataset):
    """Map-style dataset for code instruction fine-tuning."""

    def __init__(
        self,
        examples: list[FineTuningExample] | tuple[FineTuningExample, ...],
        tokenizer: CodeTokenizer,
        *,
        max_length: int,
        response_only_loss: bool = True,
        ignore_index: int | None = None,
        pad_to_max_length: bool = True,
    ) -> None:
        if not examples:
            raise FineTuningDatasetError("Fine-tuning dataset is empty.")
        if max_length <= 1:
            raise FineTuningDatasetError("max_length must be greater than one.")
        self.examples = tuple(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.response_only_loss = bool(response_only_loss)
        self.ignore_index = tokenizer.pad_token_id if ignore_index is None else int(ignore_index)
        self.pad_to_max_length = bool(pad_to_max_length)
        self.stats = self._build_stats()

    def __len__(self) -> int:
        """Return the number of examples."""

        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one tokenized, shifted language-modeling batch item."""

        example = self.examples[index]
        prompt, token_ids, prompt_token_count = self._tokenize_example(example)
        token_ids = token_ids[: self.max_length + 1]
        if len(token_ids) < 2:
            raise FineTuningDatasetError("Fine-tuning example produced too few tokens.")

        input_ids = token_ids[:-1][: self.max_length]
        target_ids = token_ids[1:][: self.max_length]
        attention_mask = [1] * len(input_ids)

        if self.response_only_loss and prompt:
            prompt_target_cutoff = min(prompt_token_count, len(target_ids))
            for position in range(prompt_target_cutoff):
                target_ids[position] = self.ignore_index

        if self.pad_to_max_length:
            while len(input_ids) < self.max_length:
                input_ids.append(self.tokenizer.pad_token_id)
                target_ids.append(self.ignore_index)
                attention_mask.append(0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }

    def _build_stats(self) -> FineTuningDatasetStats:
        truncated = 0
        padded = 0
        for example in self.examples:
            token_ids = self._raw_token_ids(example)
            if len(token_ids) > self.max_length + 1:
                truncated += 1
            if len(token_ids) <= self.max_length:
                padded += 1
        return FineTuningDatasetStats(
            examples=len(self.examples),
            max_length=self.max_length,
            truncated_examples=truncated,
            padded_examples=padded if self.pad_to_max_length else 0,
        )

    def _tokenize_example(self, example: FineTuningExample) -> tuple[str, list[int], int]:
        prompt = format_fine_tuning_prompt(example)
        if not prompt:
            token_ids = self._raw_token_ids(example)[: self.max_length + 1]
            return prompt, token_ids, 0
        prompt_ids = self.tokenizer.encode(prompt)
        output_ids = self.tokenizer.encode(_clean_required(example.output, "output"))
        if not output_ids or output_ids[-1] != self.tokenizer.eos_token_id:
            output_ids.append(self.tokenizer.eos_token_id)
        max_tokens = self.max_length + 1
        if len(output_ids) >= max_tokens:
            return prompt, output_ids[:max_tokens], 0
        available_prompt_tokens = max_tokens - len(output_ids)
        prompt_ids = prompt_ids[-available_prompt_tokens:]
        return prompt, prompt_ids + output_ids, len(prompt_ids)

    def _raw_token_ids(self, example: FineTuningExample) -> list[int]:
        prompt = format_fine_tuning_prompt(example)
        token_ids = self.tokenizer.encode(f"{prompt}{_clean_required(example.output, 'output')}")
        if not token_ids or token_ids[-1] != self.tokenizer.eos_token_id:
            token_ids.append(self.tokenizer.eos_token_id)
        return token_ids


def load_fine_tuning_examples(
    path: Path | str,
    *,
    encoding: str = "utf-8",
) -> list[FineTuningExample]:
    """Load fine-tuning examples from JSONL, JSON, or TXT."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Fine-tuning dataset not found: {dataset_path}")
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        examples = _load_jsonl_examples(dataset_path, encoding=encoding)
    elif suffix == ".json":
        examples = _load_json_examples(dataset_path, encoding=encoding)
    elif suffix == ".txt":
        examples = _load_txt_examples(dataset_path, encoding=encoding)
    else:
        raise FineTuningDatasetError(
            f"Unsupported fine-tuning dataset format: {dataset_path.suffix}"
        )
    if not examples:
        raise FineTuningDatasetError("Fine-tuning dataset is empty.")
    return examples


def split_fine_tuning_examples(
    examples: list[FineTuningExample] | tuple[FineTuningExample, ...],
    *,
    validation_split: float,
    seed: int,
    shuffle: bool = True,
) -> FineTuningSplit:
    """Split examples deterministically into train and validation sets."""

    if not 0 <= validation_split < 1:
        raise FineTuningDatasetError("validation_split must be at least 0 and less than 1.")
    ordered = list(examples)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    validation_count = max(1, int(len(ordered) * validation_split)) if len(ordered) > 1 else 0
    validation = tuple(ordered[:validation_count])
    train = tuple(ordered[validation_count:])
    if not train:
        train = validation
    return FineTuningSplit(train_examples=train, validation_examples=validation)


def format_fine_tuning_prompt(example: FineTuningExample) -> str:
    """Return the supervised prompt prefix for one example."""

    instruction = _clean_text(example.instruction)
    input_text = _clean_text(example.input)
    if not instruction:
        return ""
    if input_text:
        return f"<instruction>\n{instruction}\n<input>\n{input_text}\n<output>\n"
    return f"<instruction>\n{instruction}\n<output>\n"


def format_fine_tuning_sequence(example: FineTuningExample) -> str:
    """Return the full text sequence used for fine-tuning."""

    output = _clean_required(example.output, "output")
    return f"{format_fine_tuning_prompt(example)}{output}"


def _load_jsonl_examples(path: Path, *, encoding: str) -> list[FineTuningExample]:
    examples: list[FineTuningExample] = []
    with path.open("r", encoding=encoding) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"Invalid JSON in {path}:{line_number}: {exc}"
                raise FineTuningDatasetError(message) from exc
            examples.append(_example_from_record(record, source=f"{path}:{line_number}"))
    return examples


def _load_json_examples(path: Path, *, encoding: str) -> list[FineTuningExample]:
    try:
        raw = json.loads(path.read_text(encoding=encoding))
    except json.JSONDecodeError as exc:
        raise FineTuningDatasetError(f"Invalid JSON in {path}: {exc}") from exc
    if isinstance(raw, dict):
        for key in ("examples", "data", "records"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = [raw]
    if not isinstance(raw, list):
        raise FineTuningDatasetError("JSON fine-tuning data must be an object or list.")
    return [
        _example_from_record(record, source=f"{path}[{index}]")
        for index, record in enumerate(raw)
    ]


def _load_txt_examples(path: Path, *, encoding: str) -> list[FineTuningExample]:
    text = path.read_text(encoding=encoding).replace("\r\n", "\n").replace("\r", "\n")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not chunks and text.strip():
        chunks = [text.strip()]
    return [FineTuningExample(output=chunk) for chunk in chunks]


def _example_from_record(record: Any, *, source: str) -> FineTuningExample:
    if not isinstance(record, dict):
        raise FineTuningDatasetError(f"Fine-tuning record must be an object: {source}")
    output = _clean_text(record.get("output"))
    if not output:
        output = _clean_text(record.get("response"))
    if not output:
        output = _clean_text(record.get("text"))
    output = _clean_required(output, f"output in {source}")
    return FineTuningExample(
        instruction=_clean_text(record.get("instruction")),
        input=_clean_text(record.get("input")),
        output=output,
    )


def _clean_required(value: object, name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise FineTuningDatasetError(f"{name} must not be empty.")
    return text


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


__all__ = [
    "FineTuningDataset",
    "FineTuningDatasetError",
    "FineTuningDatasetStats",
    "FineTuningExample",
    "FineTuningSplit",
    "format_fine_tuning_prompt",
    "format_fine_tuning_sequence",
    "load_fine_tuning_examples",
    "split_fine_tuning_examples",
]
