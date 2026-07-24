"""Autoregressive text generation for GenPy LLM."""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from genpy_llm.checkpointing import CheckpointError, load_checkpoint
from genpy_llm.config import load_config
from genpy_llm.gpt import create_gpt_model
from genpy_llm.preprocessing import TextPreprocessor
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.vocabulary import Vocabulary


class GenerationError(ValueError):
    """Raised when text generation cannot continue safely."""


@dataclass(frozen=True)
class GenerationResult:
    """Text and token details returned by autoregressive generation."""

    prompt: str
    generated_text: str
    prompt_tokens: tuple[str, ...]
    generated_tokens: tuple[str, ...]
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stopped_on_eos: bool
    total_tokens: int


class TextGenerator:
    """Generate text from a GPT model, tokenizer, and vocabulary."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TextTokenizer,
        vocabulary: Vocabulary,
        device: torch.device,
        context_length: int,
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise GenerationError("model must be a torch.nn.Module.")
        if not isinstance(tokenizer, TextTokenizer):
            raise GenerationError("tokenizer must be a TextTokenizer.")
        if not isinstance(vocabulary, Vocabulary):
            raise GenerationError("vocabulary must be a Vocabulary.")
        if not isinstance(device, torch.device):
            raise GenerationError("device must be a torch.device.")
        if (
            not isinstance(context_length, int)
            or isinstance(context_length, bool)
            or context_length <= 0
        ):
            raise GenerationError("context_length must be an integer greater than zero.")

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.device = device
        self.context_length = context_length
        self.preprocessor = preprocessor

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        """Generate text autoregressively from a prompt."""

        _validate_generation_args(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            stop_on_eos=stop_on_eos,
        )
        cleaned_prompt = self._clean_prompt(prompt)
        prompt_tokens = self._prompt_tokens(cleaned_prompt)
        prompt_token_ids = tuple(self.vocabulary.encode(prompt_tokens))
        if not prompt_token_ids:
            raise GenerationError("prompt must produce at least one token.")

        self.model.eval()
        generated_ids: list[int] = []
        stopped_on_eos = False
        input_ids = torch.tensor(
            [prompt_token_ids],
            dtype=torch.long,
            device=self.device,
        )
        for _step in range(max_new_tokens):
            context_ids = input_ids[:, -self.context_length :]
            logits = self.model(context_ids)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise GenerationError("model must return logits with shape batch x time x vocab.")
            next_logits = logits[:, -1, :].squeeze(0)
            next_logits = apply_repetition_penalty(
                next_logits,
                input_ids.squeeze(0),
                repetition_penalty,
            )
            next_logits = _suppress_non_generation_tokens(next_logits, self.vocabulary)
            next_logits = apply_temperature(next_logits, temperature)
            next_logits = apply_top_k(next_logits, top_k)
            next_logits = apply_top_p(next_logits, top_p)
            next_id = _select_next_token(next_logits, do_sample=do_sample)
            generated_ids.append(next_id)
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[next_id]], dtype=torch.long, device=self.device),
                ],
                dim=1,
            )
            if stop_on_eos and next_id == self.vocabulary.eos_id:
                stopped_on_eos = True
                break

        generated_tokens = tuple(self.vocabulary.decode(generated_ids))
        generated_text = self._decode_text(
            original_prompt=prompt,
            cleaned_prompt=cleaned_prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )
        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            prompt_tokens=tuple(prompt_tokens),
            generated_tokens=generated_tokens,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=tuple(generated_ids),
            stopped_on_eos=stopped_on_eos,
            total_tokens=len(prompt_token_ids) + len(generated_ids),
        )

    def _clean_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise GenerationError("prompt must be a string.")
        cleaned_prompt = (
            self.preprocessor.clean_text(prompt)
            if self.preprocessor is not None
            else prompt.strip()
        )
        if not cleaned_prompt:
            raise GenerationError("prompt must not be empty.")
        return cleaned_prompt

    def _prompt_tokens(self, prompt: str) -> list[str]:
        tokens = self.tokenizer.tokenize(prompt)
        if tokens and tokens[-1] == self.vocabulary.config.eos_token:
            tokens = tokens[:-1]
        if (
            tokens
            and tokens[0] == self.vocabulary.config.bos_token
            and not self.tokenizer.config.add_bos_token
        ):
            tokens = tokens[1:]
        if not tokens:
            raise GenerationError("prompt must produce at least one non-EOS token.")
        return tokens

    def _decode_text(
        self,
        *,
        original_prompt: str,
        cleaned_prompt: str,
        prompt_tokens: list[str],
        generated_tokens: tuple[str, ...],
    ) -> str:
        decoded = self.tokenizer.detokenize([*prompt_tokens, *generated_tokens])
        if decoded.startswith(cleaned_prompt):
            return decoded
        if decoded.startswith(original_prompt):
            return decoded
        suffix = self.tokenizer.detokenize(generated_tokens)
        if not suffix:
            return original_prompt
        no_space_before = (".", ",", "!", "?", ":", ";")
        if original_prompt.endswith((" ", "\n")) or suffix.startswith(no_space_before):
            return f"{original_prompt}{suffix}"
        return f"{original_prompt} {suffix}"


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Scale logits by temperature."""

    _validate_logits(logits)
    _validate_positive_float("temperature", temperature)
    return logits / float(temperature)


def apply_top_k(
    logits: torch.Tensor,
    top_k: int | None,
) -> torch.Tensor:
    """Keep only the top-k logits, preserving at least one token."""

    _validate_logits(logits)
    if top_k is None:
        return logits
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise GenerationError("top_k must be None or an integer greater than zero.")
    if top_k >= logits.numel():
        return logits
    values, _indices = torch.topk(logits, k=top_k)
    threshold = values[-1]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def apply_top_p(
    logits: torch.Tensor,
    top_p: float | None,
) -> torch.Tensor:
    """Keep the smallest sorted set whose probability mass reaches top_p."""

    _validate_logits(logits)
    if top_p is None:
        return logits
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not 0 < top_p <= 1:
        raise GenerationError("top_p must be None or greater than zero and at most one.")
    if top_p >= 1:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    sorted_remove = cumulative > float(top_p)
    sorted_remove[1:] = sorted_remove[:-1].clone()
    sorted_remove[0] = False
    remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
    remove_mask.scatter_(0, sorted_indices, sorted_remove)
    filtered = logits.masked_fill(remove_mask, torch.finfo(logits.dtype).min)
    return _ensure_any_finite(filtered, logits)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to token IDs already present in context."""

    _validate_logits(logits)
    _validate_positive_float("repetition_penalty", penalty)
    if not isinstance(generated_ids, torch.Tensor):
        raise GenerationError("generated_ids must be a torch.Tensor.")
    if generated_ids.numel() == 0 or float(penalty) == 1.0:
        return logits
    output = logits.clone()
    repeated_ids = torch.unique(generated_ids.to(device=logits.device, dtype=torch.long))
    repeated_ids = repeated_ids[(repeated_ids >= 0) & (repeated_ids < logits.numel())]
    if repeated_ids.numel() == 0:
        return output
    selected = output[repeated_ids]
    output[repeated_ids] = torch.where(
        selected < 0,
        selected * float(penalty),
        selected / float(penalty),
    )
    return output


def create_generator_from_checkpoint(
    checkpoint_path: Path,
    config_path: Path | None,
    vocabulary_path: Path | None,
    device: torch.device,
) -> TextGenerator:
    """Load config, vocabulary, GPT model, checkpoint state, and return a ready generator."""

    if not isinstance(device, torch.device):
        raise GenerationError("device must be a torch.device.")
    config = load_config(config_path)
    resolved_vocabulary_path = vocabulary_path or config.data.vocabulary_file
    checkpoint_metadata = _read_checkpoint_metadata(checkpoint_path)
    vocabulary = Vocabulary.load(resolved_vocabulary_path, encoding=config.data.encoding)
    model, metadata = create_gpt_model(resolved_vocabulary_path, config)
    if checkpoint_metadata.get("vocabulary_size") != len(vocabulary):
        raise GenerationError(
            "Checkpoint vocabulary size does not match the loaded vocabulary. "
            f"checkpoint={checkpoint_metadata.get('vocabulary_size')} loaded={len(vocabulary)}"
        )
    if checkpoint_metadata.get("context_length") != config.model.context_length:
        raise GenerationError(
            "Checkpoint context length does not match the loaded configuration. "
            f"checkpoint={checkpoint_metadata.get('context_length')} "
            f"loaded={config.model.context_length}"
        )
    if metadata.vocab_size != len(vocabulary):
        raise GenerationError("Model vocabulary size does not match loaded vocabulary.")
    if metadata.context_length != config.model.context_length:
        raise GenerationError("Model context length does not match loaded configuration.")

    load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
        restore_rng=False,
    )
    model.to(device)
    model.eval()
    tokenizer = TextTokenizer(config.tokenization)
    preprocessor = TextPreprocessor(config.preprocessing)
    return TextGenerator(
        model=model,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        device=device,
        context_length=config.model.context_length,
        preprocessor=preprocessor,
    )


def _validate_generation_args(
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    do_sample: bool,
    repetition_penalty: float,
    stop_on_eos: bool,
) -> None:
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise GenerationError("max_new_tokens must be an integer greater than zero.")
    _validate_positive_float("temperature", temperature)
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise GenerationError("top_k must be None or an integer greater than zero.")
    if top_p is not None and (
        not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not 0 < top_p <= 1
    ):
        raise GenerationError("top_p must be None or greater than zero and at most one.")
    if not isinstance(do_sample, bool):
        raise GenerationError("do_sample must be a boolean.")
    _validate_positive_float("repetition_penalty", repetition_penalty)
    if not isinstance(stop_on_eos, bool):
        raise GenerationError("stop_on_eos must be a boolean.")


def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise GenerationError("logits must be a torch.Tensor.")
    if logits.ndim != 1:
        raise GenerationError("logits must be a one-dimensional tensor.")
    if logits.numel() <= 0:
        raise GenerationError("logits must contain at least one token.")
    if not logits.dtype.is_floating_point:
        raise GenerationError("logits must use a floating-point dtype.")
    if not bool(torch.isfinite(logits).any().item()):
        raise GenerationError("logits must contain at least one finite value.")


def _validate_positive_float(name: str, value: float) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
        or not math.isfinite(float(value))
    ):
        raise GenerationError(f"{name} must be a finite number greater than zero.")


def _suppress_non_generation_tokens(logits: torch.Tensor, vocabulary: Vocabulary) -> torch.Tensor:
    output = logits.clone()
    suppressed_ids = {vocabulary.pad_id, vocabulary.bos_id, vocabulary.unknown_id}
    for token_id in suppressed_ids:
        if token_id != vocabulary.eos_id and 0 <= token_id < output.numel():
            output[token_id] = torch.finfo(output.dtype).min
    if bool(torch.isfinite(output).any().item()):
        return output
    output[vocabulary.eos_id] = torch.zeros((), dtype=output.dtype, device=output.device)
    return output


def _select_next_token(logits: torch.Tensor, *, do_sample: bool) -> int:
    _validate_logits(logits)
    logits = _ensure_any_finite(logits, logits)
    if not do_sample:
        return int(torch.argmax(logits).item())
    probabilities = torch.softmax(logits, dim=-1)
    if (
        not bool(torch.isfinite(probabilities).all().item())
        or float(probabilities.sum().item()) <= 0
    ):
        raise GenerationError("sampling probabilities became invalid.")
    probabilities = probabilities / probabilities.sum()
    return int(torch.multinomial(probabilities, num_samples=1).item())


def _ensure_any_finite(filtered: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if bool(torch.isfinite(filtered).any().item()):
        return filtered
    output = torch.full_like(filtered, torch.finfo(filtered.dtype).min)
    output[int(torch.argmax(fallback).item())] = fallback.max()
    return output


def _read_checkpoint_metadata(checkpoint_path: Path | str) -> Mapping[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    if not path.is_file():
        raise GenerationError(f"Checkpoint path is not a file: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GenerationError("Checkpoint payload must be a mapping.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise GenerationError("Checkpoint metadata must be a mapping.")
    return metadata


__all__ = [
    "GenerationError",
    "GenerationResult",
    "TextGenerator",
    "apply_repetition_penalty",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "create_generator_from_checkpoint",
]
