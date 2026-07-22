"""Autoregressive code generation for GenPy Code LLM."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from genpy_llm.checkpointing import load_checkpoint
from genpy_llm.code_fine_tuning import format_instruction_prompt
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.code_training import CodeConfig, create_code_model
from genpy_llm.generation import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)


class CodeGenerationError(ValueError):
    """Raised when code generation cannot continue safely."""


@dataclass(frozen=True)
class CodeGenerationResult:
    """Generated code text and throughput metadata."""

    text: str
    generated_token_ids: tuple[int, ...]
    stopped_on_eos: bool
    elapsed_seconds: float
    tokens_per_second: float


def load_code_model_for_generation(
    *,
    config: CodeConfig,
    tokenizer: CodeTokenizer,
    checkpoint_path: Path,
    device: torch.device,
):
    """Create a code GPT model and load a checkpoint."""

    model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device, restore_rng=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_code_text(
    *,
    model,
    tokenizer: CodeTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    do_sample: bool,
    stop_on_eos: bool,
    instruction_mode: bool,
    code_only: bool,
    context_length: int,
) -> CodeGenerationResult:
    """Generate Python code or instruction-tuned response text."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise CodeGenerationError("prompt must not be empty.")
    if max_new_tokens <= 0:
        raise CodeGenerationError("max_new_tokens must be greater than zero.")
    formatted_prompt = format_instruction_prompt(prompt) if instruction_mode else prompt
    prompt_ids = tokenizer.encode(formatted_prompt)
    if not prompt_ids:
        raise CodeGenerationError("prompt produced no tokens.")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    stopped = False
    model.eval()
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        context = input_ids[:, -context_length:]
        logits = model(context)
        next_logits = logits[:, -1, :].squeeze(0)
        next_logits = apply_repetition_penalty(
            next_logits,
            input_ids.squeeze(0),
            repetition_penalty,
        )
        next_logits = _suppress_special(next_logits, tokenizer)
        next_logits = apply_temperature(next_logits, temperature)
        next_logits = apply_top_k(next_logits, top_k)
        next_logits = apply_top_p(next_logits, top_p)
        if do_sample:
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probabilities / probabilities.sum(), 1).item())
        else:
            next_id = int(torch.argmax(next_logits).item())
        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        if stop_on_eos and next_id == tokenizer.eos_token_id:
            stopped = True
            break
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated)
    if instruction_mode and code_only:
        text = extract_code_only(text)
    return CodeGenerationResult(
        text=text,
        generated_token_ids=tuple(generated),
        stopped_on_eos=stopped,
        elapsed_seconds=elapsed,
        tokens_per_second=len(generated) / elapsed if elapsed > 0 else 0.0,
    )


def extract_code_only(text: str) -> str:
    """Extract response/code from instruction-style generated text."""

    if "<output>" in text:
        text = text.split("<output>", 1)[1]
    elif "### Response:" in text:
        text = text.split("### Response:", 1)[1]
    if "<instruction>" in text:
        text = text.split("<instruction>", 1)[0]
    elif "### Instruction:" in text:
        text = text.split("### Instruction:", 1)[0]
    return text.strip()


def _suppress_special(logits: torch.Tensor, tokenizer: CodeTokenizer) -> torch.Tensor:
    output = logits.clone()
    for token_id in (tokenizer.pad_token_id, tokenizer.unknown_token_id, tokenizer.bos_token_id):
        if 0 <= token_id < output.numel():
            output[token_id] = torch.finfo(output.dtype).min
    return output


__all__ = [
    "CodeGenerationError",
    "CodeGenerationResult",
    "extract_code_only",
    "generate_code_text",
    "load_code_model_for_generation",
]
