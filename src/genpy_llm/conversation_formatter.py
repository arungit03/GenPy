"""Conversation formatting for Phase 7 supervised instruction fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genpy_llm.prompt_templates import DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPLATE


class ConversationFormatError(ValueError):
    """Raised when an instruction conversation cannot be formatted."""


@dataclass(frozen=True)
class ConversationTemplate:
    """Configurable system/user/assistant text template."""

    system_prefix: str = "<|system|>"
    user_prefix: str = "<|user|>"
    assistant_prefix: str = "<|assistant|>"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> ConversationTemplate:
        """Build a template from YAML-compatible data."""

        payload = dict(DEFAULT_TEMPLATE)
        if data:
            payload.update(data)
        return cls(
            system_prefix=_required_text(payload, "system_prefix"),
            user_prefix=_required_text(payload, "user_prefix"),
            assistant_prefix=_required_text(payload, "assistant_prefix"),
            system_prompt=_required_text(payload, "system_prompt"),
        )

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Return the prompt up to the assistant turn."""

        instruction = _clean_text(instruction, "instruction")
        input_text = _optional_text(input_text, "input")
        user_body = instruction if not input_text else f"{instruction}\n\n{input_text}"
        return (
            f"{self.system_prefix}\n"
            f"{self.system_prompt}\n\n"
            f"{self.user_prefix}\n"
            f"{user_body}\n\n"
            f"{self.assistant_prefix}\n"
        )

    def format_conversation(self, instruction: str, input_text: str, output: str) -> str:
        """Return a complete supervised conversation."""

        output = _clean_text(output, "output")
        return f"{self.format_prompt(instruction, input_text)}{output}"


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConversationFormatError(f"template.{key} must be a non-empty string.")
    return value.strip()


def _clean_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationFormatError(f"{field} must be a non-empty string.")
    return value.strip()


def _optional_text(value: str, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConversationFormatError(f"{field} must be a string.")
    return value.strip()


__all__ = ["ConversationFormatError", "ConversationTemplate"]
