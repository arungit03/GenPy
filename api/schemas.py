"""Pydantic schemas for the GenPy offline API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    """Readiness response for the local server."""

    status: Literal["healthy"]
    device: str
    model_loaded: bool


class ModelResponse(BaseModel):
    """Metadata for the model loaded into the API process."""

    model_name: str
    parameter_count: int = Field(ge=0)
    checkpoint_path: str
    quantization: Optional[str]
    lora_enabled: bool
    lora_adapter: Optional[str]
    device: str
    tokenizer_path: str
    context_length: int = Field(gt=0)
    vocabulary_size: int = Field(gt=0)
    loaded_at: str


class GenerationRequest(BaseModel):
    """Prompt-completion request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=65536)
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, gt=0.0, le=5.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        """Reject prompts that contain only whitespace."""

        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class ChatMessage(BaseModel):
    """One chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=65536)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        """Reject empty chat turns."""

        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ChatRequest(BaseModel):
    """Chat-completion request."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, gt=0.0, le=5.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def require_user_message(self) -> ChatRequest:
        """Require at least one user turn."""

        if not any(message.role == "user" for message in self.messages):
            raise ValueError("chat messages must include at least one user message")
        return self


class GenerationResponse(BaseModel):
    """Clean assistant-only generation response."""

    generated_text: str
