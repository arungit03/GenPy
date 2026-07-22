"""Instruction-to-code fine-tuning helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from genpy_llm.code_tokenizer import CodeTokenizer


class CodeFineTuningError(RuntimeError):
    """Raised when code instruction fine-tuning data is invalid."""


@dataclass(frozen=True)
class CodeInstructionExample:
    """One instruction-code example."""

    instruction: str
    input: str
    response: str


class CodeInstructionDataset(Dataset):
    """Map-style dataset for formatted instruction-to-code examples."""

    def __init__(
        self,
        examples: list[CodeInstructionExample],
        tokenizer: CodeTokenizer,
        *,
        max_sequence_length: int,
        response_only_loss: bool,
        ignore_index: int,
    ) -> None:
        if not examples:
            raise CodeFineTuningError("Instruction dataset is empty.")
        if max_sequence_length <= 1:
            raise CodeFineTuningError("max_sequence_length must be greater than one.")
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_sequence_length = int(max_sequence_length)
        self.response_only_loss = bool(response_only_loss)
        self.ignore_index = int(ignore_index)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        prompt = format_instruction_prompt(example.instruction, example.input)
        full_text = f"{prompt}{example.response}"
        prompt_ids = self.tokenizer.encode(prompt)
        token_ids = self.tokenizer.encode(full_text)
        eos_id = self.tokenizer.eos_token_id
        if not token_ids or token_ids[-1] != eos_id:
            token_ids.append(eos_id)
        token_ids = token_ids[: self.max_sequence_length + 1]
        if len(token_ids) < 2:
            raise CodeFineTuningError("Instruction example produced too few tokens.")
        input_ids = token_ids[:-1][: self.max_sequence_length]
        target_ids = token_ids[1:][: self.max_sequence_length]
        attention_mask = [1] * len(input_ids)
        if self.response_only_loss:
            prompt_cutoff = min(len(prompt_ids), len(target_ids))
            for position in range(prompt_cutoff):
                target_ids[position] = self.ignore_index
        while len(input_ids) < self.max_sequence_length:
            input_ids.append(self.tokenizer.pad_token_id)
            target_ids.append(self.ignore_index)
            attention_mask.append(0)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }


def format_instruction_prompt(instruction: str, input_text: str = "") -> str:
    """Format instruction/input fields before the response."""

    instruction = _clean_required(instruction, "instruction")
    input_text = input_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if input_text:
        return f"<instruction>\n{instruction}\n<input>\n{input_text}\n<output>\n"
    return f"<instruction>\n{instruction}\n<output>\n"


def load_code_instruction_examples(
    path: Path,
    *,
    encoding: str = "utf-8",
) -> list[CodeInstructionExample]:
    """Load instruction records from JSONL."""

    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")
    examples: list[CodeInstructionExample] = []
    with path.open("r", encoding=encoding) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CodeFineTuningError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise CodeFineTuningError(f"Record in {path}:{line_number} must be an object.")
            instruction = _clean_text(record.get("instruction"))
            response = _clean_text(record.get("response"))
            input_text = _clean_text(record.get("input", ""))
            if not instruction or not response:
                continue
            examples.append(CodeInstructionExample(instruction, input_text, response))
    if not examples:
        raise CodeFineTuningError("Instruction dataset is empty.")
    return examples


def split_instruction_examples(
    examples: list[CodeInstructionExample],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[CodeInstructionExample], list[CodeInstructionExample]]:
    """Deterministically split instruction examples."""

    if not 0 <= validation_ratio < 1:
        raise CodeFineTuningError("validation_ratio must be at least 0 and less than 1.")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_ratio)) if len(shuffled) > 1 else 0
    return shuffled[validation_count:], shuffled[:validation_count]


def _clean_required(value: str, name: str) -> str:
    value = _clean_text(value)
    if not value:
        raise CodeFineTuningError(f"{name} must not be empty.")
    return value


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


__all__ = [
    "CodeFineTuningError",
    "CodeInstructionDataset",
    "CodeInstructionExample",
    "format_instruction_prompt",
    "load_code_instruction_examples",
    "split_instruction_examples",
]
