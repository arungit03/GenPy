from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from genpy_llm.code_tokenizer import CodeTokenizer, train_byte_level_bpe_tokenizer
from genpy_llm.pretraining_generation import (
    ASSISTANT_MARKER,
    USER_MARKER,
    CodeGenerationSettings,
    clean_assistant_response,
    generate_code_sample,
)


def test_assistant_extraction_uses_last_assistant_marker() -> None:
    text = (
        "<|system|>\n"
        "You are GenPy.\n"
        "<|user|>\n"
        "Write bubble sort.\n"
        "<|assistant|>\n"
        "draft\n"
        "<|assistant|>\n"
        "def hello():\n"
        "    print(\"hi\")\n"
        "<|user|>\n"
        "Next task"
    )

    assert clean_assistant_response(text) == 'def hello():\n    print("hi")'


def test_generation_decodes_only_new_tokens_and_stops_before_user_turn(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    prompt = "<|system|>\nYou are GenPy.\n\n<|user|>\nWrite bubble sort.\n\n<|assistant|>\n"
    reply = "def bubble_sort(items):\n    return items\n"
    generated_ids = tokenizer.encode(f"{reply}{USER_MARKER}\nWrite insertion sort.")
    model = _SequenceModel(generated_ids, tokenizer.vocab_size)

    result = generate_code_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=torch.device("cpu"),
        context_length=128,
        settings=_settings(),
    )

    assert result.text == reply.strip()
    assert result.stopped is True
    assert prompt not in result.text
    assert "<|system|>" not in result.text
    assert "<|user|>" not in result.text
    assert "<|assistant|>" not in result.text
    assert tokenizer.decode(result.generated_token_ids, skip_special_tokens=True) == reply


def test_generation_reports_leading_conversation_marker_behavior(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    prompt = "<|system|>\nYou are GenPy.\n\n<|user|>\nWrite bubble sort.\n\n<|assistant|>\n"
    reply = "def bubble_sort(items):\n    return items\n"
    generated_ids = tokenizer.encode(f"{ASSISTANT_MARKER}\n{reply}") + [tokenizer.eos_token_id]
    model = _SequenceModel(generated_ids, tokenizer.vocab_size)

    result = generate_code_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=torch.device("cpu"),
        context_length=128,
        settings=_settings(),
    )

    assert result.text == reply.strip()
    assert result.diagnostic_report is not None
    assert result.diagnostic_report["issue"] == "E. checkpoint behavior"
    assert result.diagnostic_report["prompt_sent_to_model"] == prompt
    assert result.diagnostic_report["generated_token_ids"] == result.raw_generated_token_ids
    assert ASSISTANT_MARKER in result.diagnostic_report["decoded_generated_text"]


def test_generation_stops_before_new_assistant_turn_after_content(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    prompt = "<|system|>\nYou are GenPy.\n\n<|user|>\nWrite bubble sort.\n\n<|assistant|>\n"
    reply = "def bubble_sort(items):\n    return items\n"
    generated_ids = tokenizer.encode(f"{reply}{ASSISTANT_MARKER}\ndef other():\n    pass")
    model = _SequenceModel(generated_ids, tokenizer.vocab_size)

    result = generate_code_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=torch.device("cpu"),
        context_length=128,
        settings=_settings(),
    )

    assert result.text == reply.strip()
    assert ASSISTANT_MARKER not in result.text


def test_min_new_tokens_suppresses_early_eos_until_floor(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    content_id = tokenizer.encode("pass")[0]
    model = _EosPreferringModel(tokenizer.eos_token_id, content_id, tokenizer.vocab_size)

    immediate = generate_code_sample(
        model=model,
        tokenizer=tokenizer,
        prompt="<|assistant|>\n",
        device=torch.device("cpu"),
        context_length=128,
        settings=_settings(),
    )
    assert immediate.generated_token_ids == ()
    assert immediate.stopped is True

    model = _EosPreferringModel(tokenizer.eos_token_id, content_id, tokenizer.vocab_size)
    floored = generate_code_sample(
        model=model,
        tokenizer=tokenizer,
        prompt="<|assistant|>\n",
        device=torch.device("cpu"),
        context_length=128,
        settings=replace(_settings(), min_new_tokens=5),
    )
    assert len(floored.generated_token_ids) == 5
    assert all(token == content_id for token in floored.generated_token_ids)
    assert floored.stopped is True


class _EosPreferringModel(torch.nn.Module):
    """Always ranks <eos> first and one content token second."""

    def __init__(self, eos_id: int, content_id: int, vocabulary_size: int) -> None:
        super().__init__()
        self.eos_id = eos_id
        self.content_id = content_id
        self.vocabulary_size = vocabulary_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocabulary_size),
            -1_000_000.0,
            device=input_ids.device,
        )
        logits[:, -1, self.eos_id] = 1_000_000.0
        logits[:, -1, self.content_id] = 999_999.0
        return logits


class _SequenceModel(torch.nn.Module):
    def __init__(self, token_ids: list[int], vocabulary_size: int) -> None:
        super().__init__()
        self.token_ids = token_ids
        self.vocabulary_size = vocabulary_size
        self.index = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_id = self.token_ids[min(self.index, len(self.token_ids) - 1)]
        self.index += 1
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocabulary_size),
            -1_000_000.0,
            device=input_ids.device,
        )
        logits[:, -1, token_id] = 1_000_000.0
        return logits


def _settings() -> CodeGenerationSettings:
    return CodeGenerationSettings(
        prompts=(),
        max_new_tokens=128,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=False,
        repetition_penalty=1.0,
        stop_tokens=("<eos>",),
    )


def _tokenizer(tmp_path: Path) -> CodeTokenizer:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "instruction": "Write bubble sort.",
                "output": (
                    "<|system|>\n"
                    "You are GenPy.\n"
                    "<|user|>\n"
                    "Write bubble sort.\n"
                    "<|assistant|>\n"
                    "def bubble_sort(items):\n"
                    "    return items\n"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    train_byte_level_bpe_tokenizer(
        [corpus],
        output_path=tokenizer_path,
        metadata_path=tmp_path / "tokenizer_metadata.json",
        vocab_size=320,
        min_frequency=1,
        show_progress=False,
    )
    return CodeTokenizer.from_file(tokenizer_path)
