"""Code-token generation helpers for Phase 6 pretraining samples."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class CodeGenerationResult:
    """Generated code sample and token metadata."""

    prompt: str
    text: str
    generated_token_ids: tuple[int, ...]
    stopped: bool


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
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [tokenizer.bos_token_id]
    stop_ids = {
        token_id
        for token in settings.stop_tokens
        if (token_id := tokenizer.token_to_id(token)) is not None
    }
    stop_ids.add(tokenizer.eos_token_id)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    stopped = False
    model.eval()
    for _ in range(settings.max_new_tokens):
        logits = model(tokens[:, -context_length:])
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise GenerationError("model must return logits with shape batch x time x vocab.")
        next_logits = logits[0, -1, :]
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
        generated.append(next_id)
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        if next_id in stop_ids:
            stopped = True
            break
    text = tokenizer.decode([*input_ids, *generated], skip_special_tokens=True)
    return CodeGenerationResult(
        prompt=prompt,
        text=text,
        generated_token_ids=tuple(generated),
        stopped=stopped,
    )


__all__ = [
    "CodeGenerationResult",
    "CodeGenerationSettings",
    "generate_code_sample",
]
