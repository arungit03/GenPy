"""Alpaca-style JSONL instruction dataset for Phase 7 SFT."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.conversation_formatter import ConversationTemplate


class InstructionDatasetError(ValueError):
    """Raised when instruction data cannot be loaded or tokenized."""


@dataclass(frozen=True)
class InstructionRecord:
    """One Alpaca-style instruction record."""

    instruction: str
    input: str
    output: str


@dataclass(frozen=True)
class InstructionDatasetStats:
    """Tokenization statistics for an instruction dataset."""

    source_records: int
    usable_records: int
    skipped_records: int
    truncated_records: int
    mask_prompt_tokens: bool
    context_length: int


class InstructionDataset(Dataset[dict[str, torch.Tensor]]):
    """Map-style tokenized SFT dataset with optional prompt-loss masking."""

    def __init__(
        self,
        records: Sequence[InstructionRecord],
        *,
        tokenizer: CodeTokenizer,
        template: ConversationTemplate,
        context_length: int,
        mask_prompt_tokens: bool,
    ) -> None:
        if not isinstance(tokenizer, CodeTokenizer):
            raise InstructionDatasetError("tokenizer must be a CodeTokenizer.")
        if context_length <= 1:
            raise InstructionDatasetError("context_length must be greater than one.")
        self.tokenizer = tokenizer
        self.template = template
        self.context_length = int(context_length)
        self.mask_prompt_tokens = bool(mask_prompt_tokens)
        self.examples: list[dict[str, torch.Tensor]] = []
        truncated = 0
        skipped = 0
        for record in records:
            example, was_truncated = self._encode_record(record)
            if example is None:
                skipped += 1
                continue
            truncated += int(was_truncated)
            self.examples.append(example)
        if not self.examples:
            raise InstructionDatasetError("instruction dataset produced no usable examples.")
        self.stats = InstructionDatasetStats(
            source_records=len(records),
            usable_records=len(self.examples),
            skipped_records=skipped,
            truncated_records=truncated,
            mask_prompt_tokens=self.mask_prompt_tokens,
            context_length=self.context_length,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        *,
        tokenizer: CodeTokenizer,
        template: ConversationTemplate,
        context_length: int,
        mask_prompt_tokens: bool,
    ) -> InstructionDataset:
        """Load and tokenize an Alpaca-style JSONL dataset."""

        return cls(
            load_instruction_records(path),
            tokenizer=tokenizer,
            template=template,
            context_length=context_length,
            mask_prompt_tokens=mask_prompt_tokens,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]

    def _encode_record(
        self,
        record: InstructionRecord,
    ) -> tuple[dict[str, torch.Tensor] | None, bool]:
        prompt = self.template.format_prompt(record.instruction, record.input)
        full = self.template.format_conversation(record.instruction, record.input, record.output)
        prompt_ids = [self.tokenizer.bos_token_id, *self.tokenizer.encode(prompt)]
        token_ids = [self.tokenizer.bos_token_id, *self.tokenizer.encode(full)]
        if token_ids[-1] != self.tokenizer.eos_token_id:
            token_ids.append(self.tokenizer.eos_token_id)
        if len(token_ids) < 2:
            return None, False
        truncated = len(token_ids) > self.context_length + 1
        token_ids = token_ids[: self.context_length + 1]
        if token_ids[-1] != self.tokenizer.eos_token_id and truncated:
            token_ids[-1] = self.tokenizer.eos_token_id
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        attention = [1] * len(input_ids)
        labels = list(target_ids)
        if self.mask_prompt_tokens:
            ignore_until = min(len(labels), max(0, len(prompt_ids) - 1))
            labels[:ignore_until] = [-100] * ignore_until
        pad = self.context_length - len(input_ids)
        if pad > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * pad)
            labels.extend([-100] * pad)
            attention.extend([0] * pad)
        if all(label == -100 for label in labels):
            return None, truncated
        return (
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "target_ids": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            },
            truncated,
        )


def load_instruction_records(path: Path | str) -> list[InstructionRecord]:
    """Load Alpaca-style records from JSONL."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Instruction dataset not found: {input_path}")
    records: list[InstructionRecord] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise InstructionDatasetError(
                    f"Malformed JSONL at line {line_number}: {exc}"
                ) from exc
            records.append(_record_from_payload(payload, line_number))
    if not records:
        raise InstructionDatasetError(f"Instruction dataset is empty: {input_path}")
    return records


def _record_from_payload(payload: Any, line_number: int) -> InstructionRecord:
    if not isinstance(payload, dict):
        raise InstructionDatasetError(f"Line {line_number} must be a JSON object.")
    instruction = payload.get("instruction")
    input_text = payload.get("input", "")
    output = payload.get("output")
    if not isinstance(instruction, str) or not instruction.strip():
        raise InstructionDatasetError(f"Line {line_number} has an empty instruction.")
    if input_text is None:
        input_text = ""
    if not isinstance(input_text, str):
        raise InstructionDatasetError(f"Line {line_number} input must be a string.")
    if not isinstance(output, str) or not output.strip():
        raise InstructionDatasetError(f"Line {line_number} has an empty output.")
    return InstructionRecord(
        instruction=instruction.strip(),
        input=input_text.strip(),
        output=output.strip(),
    )


__all__ = [
    "InstructionDataset",
    "InstructionDatasetError",
    "InstructionDatasetStats",
    "InstructionRecord",
    "load_instruction_records",
]
