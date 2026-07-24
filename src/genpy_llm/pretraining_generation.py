"""Code-token generation helpers for Phase 6 pretraining samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.generation import (
    GenerationError,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)

SYSTEM_MARKER = "<|system|>"
USER_MARKER = "<|user|>"
ASSISTANT_MARKER = "<|assistant|>"
CONVERSATION_MARKERS = (SYSTEM_MARKER, USER_MARKER, ASSISTANT_MARKER)


@dataclass(frozen=True)
class CodeGenerationSettings:
    """Autoregressive generation settings for code-token models."""

    prompts: tuple[str, ...]
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    do_sample: bool
    repetition_penalty: float
    stop_tokens: tuple[str, ...]
    # Suppress end-of-sequence and stop tokens until this many content tokens
    # have been emitted. 0 keeps the original stop-at-first-EOS behavior.
    min_new_tokens: int = 0


@dataclass(frozen=True)
class CodeGenerationResult:
    """Generated code sample and token metadata."""

    prompt: str
    text: str
    generated_token_ids: tuple[int, ...]
    stopped: bool
    raw_generated_token_ids: tuple[int, ...] = ()
    diagnostic_report: dict[str, Any] | None = None


@torch.no_grad()
def generate_code_sample(
    *,
    model: nn.Module,
    tokenizer: CodeTokenizer,
    prompt: str,
    device: torch.device,
    context_length: int,
    settings: CodeGenerationSettings,
) -> CodeGenerationResult:
    """Generate Python code from a Phase 5 tokenizer-backed GPT model."""

    if not isinstance(model, nn.Module):
        raise GenerationError("model must be a torch.nn.Module.")
    if not isinstance(tokenizer, CodeTokenizer):
        raise GenerationError("tokenizer must be a CodeTokenizer.")
    if not isinstance(prompt, str) or not prompt:
        raise GenerationError("prompt must be a non-empty string.")
    if settings.min_new_tokens < 0:
        raise GenerationError("min_new_tokens must be non-negative.")
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [tokenizer.bos_token_id]
    stop_sequences = _build_stop_sequences(tokenizer, settings.stop_tokens)
    single_token_stop_ids = [
        sequence["ids"][0] for sequence in stop_sequences if len(sequence["ids"]) == 1
    ]
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    raw_generated: list[int] = []
    emitted: list[int] = []
    stopped = False
    marker_generated_first = False
    model.eval()
    for _ in range(settings.max_new_tokens):
        logits = model(tokens[:, -context_length:])
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise GenerationError("model must return logits with shape batch x time x vocab.")
        next_logits = logits[0, -1, :]
        if len(emitted) < settings.min_new_tokens and single_token_stop_ids:
            next_logits = next_logits.clone()
            next_logits[single_token_stop_ids] = float("-inf")
        next_logits = apply_repetition_penalty(
            next_logits,
            tokens.squeeze(0),
            settings.repetition_penalty,
        )
        next_logits = apply_temperature(next_logits, settings.temperature)
        next_logits = apply_top_k(next_logits, settings.top_k)
        next_logits = apply_top_p(next_logits, settings.top_p)
        if settings.do_sample:
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
        else:
            next_id = int(torch.argmax(next_logits).item())
        raw_generated.append(next_id)
        emitted.append(next_id)
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        stop_match = _matched_stop_sequence(emitted, stop_sequences)
        if stop_match is None:
            continue

        del emitted[-len(stop_match["ids"]) :]
        stopped = True
        if (
            stop_match["marker"] == ASSISTANT_MARKER
            and not _has_assistant_content(emitted, tokenizer)
        ):
            marker_generated_first = True
            stopped = False
            continue
        if len(emitted) < settings.min_new_tokens:
            # Multi-token stop sequence completed before the requested floor:
            # drop it and keep generating until min_new_tokens is reached.
            stopped = False
            continue
        if not _has_assistant_content(emitted, tokenizer):
            marker_generated_first = True
        break

    decoded_generated_text = tokenizer.decode(raw_generated, skip_special_tokens=True)
    text = clean_assistant_response(
        tokenizer.decode(emitted, skip_special_tokens=True),
        stop_markers=settings.stop_tokens,
    )
    diagnostic_report = None
    if marker_generated_first:
        diagnostic_report = _diagnostic_report(
            prompt=prompt,
            prompt_token_ids=input_ids,
            raw_generated_token_ids=raw_generated,
            decoded_generated_text=decoded_generated_text,
            tokenizer=tokenizer,
            stop_sequences=stop_sequences,
        )
    return CodeGenerationResult(
        prompt=prompt,
        text=text,
        generated_token_ids=tuple(emitted),
        stopped=stopped,
        raw_generated_token_ids=tuple(raw_generated),
        diagnostic_report=diagnostic_report,
    )


def clean_assistant_response(
    text: str,
    *,
    stop_markers: tuple[str, ...] = CONVERSATION_MARKERS,
) -> str:
    """Return only assistant content from generated text."""

    if not isinstance(text, str):
        raise GenerationError("generated text must be a string.")
    cleaned = text.strip()
    last_assistant_marker = cleaned.rfind(ASSISTANT_MARKER)
    if last_assistant_marker >= 0:
        cleaned = cleaned[last_assistant_marker + len(ASSISTANT_MARKER) :]

    marker_index = _first_terminal_marker_index(cleaned, stop_markers)
    if marker_index is not None:
        cleaned = cleaned[:marker_index]

    for marker in CONVERSATION_MARKERS:
        cleaned = cleaned.replace(marker, "")
    for marker in stop_markers:
        if marker not in CONVERSATION_MARKERS:
            cleaned = cleaned.replace(marker, "")

    lines = [line.rstrip() for line in cleaned.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _first_terminal_marker_index(text: str, stop_markers: tuple[str, ...]) -> int | None:
    candidates: list[int] = []
    for marker in (*stop_markers, SYSTEM_MARKER, USER_MARKER, ASSISTANT_MARKER):
        if not marker:
            continue
        index = text.find(marker)
        if index >= 0:
            candidates.append(index)
    return min(candidates) if candidates else None


def _build_stop_sequences(
    tokenizer: CodeTokenizer,
    configured_stop_tokens: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    sequences: list[dict[str, Any]] = [
        {"marker": "<eos>", "ids": (tokenizer.eos_token_id,), "kind": "eos"},
    ]
    for marker in (USER_MARKER, SYSTEM_MARKER, ASSISTANT_MARKER):
        sequences.append(
            {
                "marker": marker,
                "ids": tuple(tokenizer.encode(marker)),
                "kind": "conversation_marker",
            }
        )
    for marker in configured_stop_tokens:
        ids = _stop_token_ids(tokenizer, marker)
        if ids:
            sequences.append({"marker": marker, "ids": ids, "kind": "configured_stop_token"})

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for sequence in sequences:
        ids = tuple(sequence["ids"])
        if ids and ids not in seen:
            deduplicated.append(sequence)
            seen.add(ids)
    return tuple(deduplicated)


def _stop_token_ids(tokenizer: CodeTokenizer, token: str) -> tuple[int, ...]:
    token_id = tokenizer.token_to_id(token)
    if token_id is not None:
        return (int(token_id),)
    return tuple(tokenizer.encode(token))


def _matched_stop_sequence(
    generated_token_ids: list[int],
    stop_sequences: tuple[dict[str, Any], ...],
) -> dict[str, Any] | None:
    for sequence in sorted(stop_sequences, key=lambda item: len(item["ids"]), reverse=True):
        ids = sequence["ids"]
        if len(generated_token_ids) >= len(ids) and tuple(generated_token_ids[-len(ids) :]) == ids:
            return sequence
    return None


def _has_assistant_content(token_ids: list[int], tokenizer: CodeTokenizer) -> bool:
    return bool(tokenizer.decode(token_ids, skip_special_tokens=True).strip())


def _diagnostic_report(
    *,
    prompt: str,
    prompt_token_ids: list[int],
    raw_generated_token_ids: list[int],
    decoded_generated_text: str,
    tokenizer: CodeTokenizer,
    stop_sequences: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "issue": "E. checkpoint behavior",
        "reason": "The model generated a conversation marker before assistant content.",
        "prompt_sent_to_model": prompt,
        "prompt_token_ids": tuple(prompt_token_ids),
        "generated_token_ids": tuple(raw_generated_token_ids),
        "decoded_generated_text": decoded_generated_text,
        "first_50_generated_tokens": tuple(
            tokenizer.id_to_token(token_id) or f"<unknown:{token_id}>"
            for token_id in raw_generated_token_ids[:50]
        ),
        "tokenizer_special_tokens": tokenizer.special_tokens,
        "stopping_criteria": tuple(
            {
                "marker": sequence["marker"],
                "ids": sequence["ids"],
                "kind": sequence["kind"],
            }
            for sequence in stop_sequences
        ),
    }


__all__ = [
    "clean_assistant_response",
    "CodeGenerationResult",
    "CodeGenerationSettings",
    "generate_code_sample",
]
