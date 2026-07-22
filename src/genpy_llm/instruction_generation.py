"""Instruction-generation helpers for Phase 7 sample prompts."""

from __future__ import annotations

from genpy_llm.conversation_formatter import ConversationTemplate
from genpy_llm.prompt_templates import DEFAULT_GENERATION_PROMPTS


def generation_prompts(configured: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Return configured prompts or GenPy's default Python-coding prompts."""

    if configured is None:
        return DEFAULT_GENERATION_PROMPTS
    prompts = tuple(
        prompt.strip() for prompt in configured if isinstance(prompt, str) and prompt.strip()
    )
    return prompts or DEFAULT_GENERATION_PROMPTS


def format_generation_prompt(
    instruction: str,
    *,
    template: ConversationTemplate,
    input_text: str = "",
) -> str:
    """Format an instruction as a generation prompt without an assistant answer."""

    return template.format_prompt(instruction, input_text)


__all__ = ["format_generation_prompt", "generation_prompts"]
