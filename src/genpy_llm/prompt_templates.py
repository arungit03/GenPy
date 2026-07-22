"""Prompt templates for GenPy instruction fine-tuning."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = "You are GenPy, a Python coding assistant."

DEFAULT_TEMPLATE = {
    "system_prefix": "<|system|>",
    "user_prefix": "<|user|>",
    "assistant_prefix": "<|assistant|>",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
}

DEFAULT_GENERATION_PROMPTS = (
    "Write bubble sort.",
    "Reverse linked list.",
    "Binary search.",
    "Read CSV with pandas.",
)


__all__ = ["DEFAULT_GENERATION_PROMPTS", "DEFAULT_SYSTEM_PROMPT", "DEFAULT_TEMPLATE"]
