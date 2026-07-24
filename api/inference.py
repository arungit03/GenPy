"""Single-load inference service for the GenPy offline API."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from api.config import APIConfig, detect_api_device
from api.schemas import (
    ChatMessage,
    ChatRequest,
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    ModelResponse,
)
from genpy_llm.checkpointing import load_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer, tokenizer_file_hash
from genpy_llm.fine_tuning import Phase7Config, load_phase7_config
from genpy_llm.instruction_generation import format_generation_prompt
from genpy_llm.lora import load_lora_adapters
from genpy_llm.pretraining import create_phase6_model
from genpy_llm.pretraining_generation import (
    CodeGenerationSettings,
    clean_assistant_response,
    generate_code_sample,
)
from genpy_llm.quantization import load_quantized_checkpoint

UTC = timezone.utc

LOGGER = logging.getLogger("genpy_api")


@dataclass(frozen=True)
class LoadedModelInfo:
    """Metadata for the process-local model."""

    model_name: str
    checkpoint_path: Path
    quantization: str | None
    lora_adapter: Path | None
    device: torch.device
    tokenizer_path: Path
    context_length: int
    vocabulary_size: int
    parameter_count: int
    loaded_at: str


class GenPyInferenceService:
    """Serve one tokenizer and model instance for the lifetime of the process."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: CodeTokenizer,
        phase7_config: Phase7Config,
        api_config: APIConfig,
        info: LoadedModelInfo,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.phase7_config = phase7_config
        self.api_config = api_config
        self.info = info
        self._generation_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: APIConfig) -> GenPyInferenceService:
        """Load tokenizer, model, checkpoint, optional quantization, and optional LoRA once."""

        LOGGER.info("api_startup_started")
        device = detect_api_device(config.device)
        LOGGER.info("api_device_selected device=%s", device)

        phase7_path = _required_path(config.resolve_path(config.phase7_config), "Phase 7 config")
        phase7_config = load_phase7_config(phase7_path)
        tokenizer = CodeTokenizer.from_file(phase7_config.data.tokenizer)
        LOGGER.info(
            "api_tokenizer_loaded path=%s vocab_size=%d",
            tokenizer.source_path,
            tokenizer.vocab_size,
        )

        model = create_phase6_model(phase7_config.model, tokenizer)
        quantized_path = config.resolve_path(config.quantized_checkpoint)
        primary_checkpoint_path = _required_path(
            config.resolve_path(config.checkpoint),
            "checkpoint",
        )

        quantization: str | None = None
        if quantized_path is not None and _quantized_checkpoint_matches(
            quantized_path,
            primary_checkpoint_path,
        ):
            checkpoint_path = quantized_path
            loaded_quantized = load_quantized_checkpoint(checkpoint_path, model, map_location="cpu")
            model = loaded_quantized.model
            quantization = loaded_quantized.method
            LOGGER.info(
                "api_quantized_checkpoint_loaded path=%s method=%s",
                checkpoint_path,
                quantization,
            )
        else:
            checkpoint_path = primary_checkpoint_path
            _validate_checkpoint_tokenizer(checkpoint_path, tokenizer)
            load_checkpoint(
                checkpoint_path,
                model,
                optimizer=None,
                map_location="cpu",
                restore_rng=False,
            )
            LOGGER.info("api_checkpoint_loaded path=%s", checkpoint_path)

        lora_path = config.resolve_path(config.lora_adapter)
        if lora_path is not None:
            if quantization == "dynamic_int8":
                raise ValueError("LoRA adapters cannot be applied to dynamic INT8 checkpoints.")
            loaded_lora = load_lora_adapters(model, lora_path, map_location="cpu")
            LOGGER.info(
                "api_lora_adapter_loaded path=%s adapters=%d",
                loaded_lora.path,
                loaded_lora.adapter_count,
            )

        serving_device = torch.device("cpu") if quantization == "dynamic_int8" else device
        if serving_device.type != "cpu":
            model.to(serving_device)
        model.eval()

        info = LoadedModelInfo(
            model_name="GenPy GPT",
            checkpoint_path=checkpoint_path.resolve(),
            quantization=quantization,
            lora_adapter=None if lora_path is None else lora_path.resolve(),
            device=serving_device,
            tokenizer_path=phase7_config.data.tokenizer.resolve(),
            context_length=phase7_config.model.context_length,
            vocabulary_size=tokenizer.vocab_size,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            loaded_at=datetime.now(UTC).isoformat(),
        )
        LOGGER.info(
            "api_startup_completed checkpoint=%s quantization=%s lora=%s device=%s parameters=%d",
            info.checkpoint_path,
            info.quantization,
            bool(info.lora_adapter),
            info.device,
            info.parameter_count,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            phase7_config=phase7_config,
            api_config=config,
            info=info,
        )

    def health(self) -> HealthResponse:
        """Return process readiness."""

        return HealthResponse(status="healthy", device=str(self.info.device), model_loaded=True)

    def model_info(self) -> ModelResponse:
        """Return loaded model metadata."""

        return ModelResponse(
            model_name=self.info.model_name,
            parameter_count=self.info.parameter_count,
            checkpoint_path=str(self.info.checkpoint_path),
            quantization=self.info.quantization,
            lora_enabled=self.info.lora_adapter is not None,
            lora_adapter=None if self.info.lora_adapter is None else str(self.info.lora_adapter),
            device=str(self.info.device),
            tokenizer_path=str(self.info.tokenizer_path),
            context_length=self.info.context_length,
            vocabulary_size=self.info.vocabulary_size,
            loaded_at=self.info.loaded_at,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate from a single instruction prompt."""

        prompt = format_generation_prompt(
            request.prompt,
            template=self.phase7_config.template,
            input_text="",
        )
        return self._generate(prompt, _settings_from_generation_request(request, self.api_config))

    def chat(self, request: ChatRequest) -> GenerationResponse:
        """Generate from chat messages."""

        prompt = _chat_prompt(request.messages, self.phase7_config.template)
        return self._generate(prompt, _settings_from_chat_request(request, self.api_config))

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        settings: CodeGenerationSettings,
    ) -> GenerationResponse:
        started = time.perf_counter()
        prompt_sent_to_model = _prompt_with_bos(prompt, self.tokenizer)
        input_token_ids = tuple(self.tokenizer.encode(prompt_sent_to_model))
        _log_generation_request(self.info, prompt, prompt_sent_to_model, input_token_ids, settings)
        try:
            with self._generation_lock:
                generated = generate_code_sample(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt_sent_to_model,
                    device=self.info.device,
                    context_length=self.info.context_length,
                    settings=settings,
                )
        except Exception:
            LOGGER.exception("api_generation_failed")
            raise
        elapsed = time.perf_counter() - started
        token_count = len(generated.generated_token_ids)
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0.0
        LOGGER.info(
            "api_generation_completed tokens=%d elapsed_seconds=%.6f tokens_per_second=%.3f",
            token_count,
            elapsed,
            tokens_per_second,
        )
        final_response = _clean_generated_text(
            generated.text,
            prompt_sent_to_model,
            settings.stop_tokens,
        )
        _log_generation_result(self.tokenizer, generated, final_response)
        if generated.diagnostic_report is not None:
            LOGGER.warning(
                "api_generation_diagnostic report=%s",
                json.dumps(
                    {
                        **generated.diagnostic_report,
                        "checkpoint_metadata": _checkpoint_metadata(self.info),
                    },
                    default=str,
                    sort_keys=True,
                ),
            )
        return GenerationResponse(generated_text=final_response)


def _settings_from_generation_request(
    request: GenerationRequest,
    config: APIConfig,
) -> CodeGenerationSettings:
    generation = config.generation
    return CodeGenerationSettings(
        prompts=(),
        max_new_tokens=request.max_new_tokens or generation.max_new_tokens,
        temperature=request.temperature or generation.temperature,
        top_k=None,
        top_p=generation.top_p if request.top_p is None else request.top_p,
        do_sample=generation.do_sample,
        repetition_penalty=generation.repetition_penalty,
        stop_tokens=generation.stop_tokens,
        min_new_tokens=generation.min_new_tokens,
    )


def _settings_from_chat_request(request: ChatRequest, config: APIConfig) -> CodeGenerationSettings:
    generation = config.generation
    return CodeGenerationSettings(
        prompts=(),
        max_new_tokens=request.max_new_tokens or generation.max_new_tokens,
        temperature=request.temperature or generation.temperature,
        top_k=None,
        top_p=generation.top_p if request.top_p is None else request.top_p,
        do_sample=generation.do_sample,
        repetition_penalty=generation.repetition_penalty,
        stop_tokens=generation.stop_tokens,
        min_new_tokens=generation.min_new_tokens,
    )


def _chat_prompt(messages: list[ChatMessage], template) -> str:
    """Format chat as the latest user instruction, matching single-turn SFT."""

    system_prompt = template.system_prompt
    latest_user_content: str | None = None
    for message in messages:
        content = message.content.strip()
        if message.role == "system":
            system_prompt = content
        elif message.role == "user":
            latest_user_content = content
    if latest_user_content is None:
        raise ValueError("chat messages must include at least one user message.")
    return (
        f"{template.system_prefix}\n"
        f"{system_prompt}\n\n"
        f"{template.user_prefix}\n"
        f"{latest_user_content}\n\n"
        f"{template.assistant_prefix}\n"
    )


def _clean_generated_text(text: str, prompt: str, stop_tokens: tuple[str, ...]) -> str:
    """Remove prompt echo and return assistant-only content."""

    generated = text
    if generated.startswith(prompt):
        generated = generated[len(prompt) :]
    return clean_assistant_response(generated, stop_markers=stop_tokens)


def _checkpoint_metadata(info: LoadedModelInfo) -> dict[str, object]:
    return {
        "checkpoint_path": str(info.checkpoint_path),
        "quantization": info.quantization,
        "lora_adapter": None if info.lora_adapter is None else str(info.lora_adapter),
        "device": str(info.device),
        "context_length": info.context_length,
        "vocabulary_size": info.vocabulary_size,
        "parameter_count": info.parameter_count,
        "loaded_at": info.loaded_at,
    }


def _required_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} path is required.")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _quantized_checkpoint_matches(quantized_path: Path, primary_checkpoint_path: Path) -> bool:
    """Return whether a quantized checkpoint was derived from the configured checkpoint."""

    if not quantized_path.is_file():
        raise FileNotFoundError(f"quantized checkpoint not found: {quantized_path}")
    source_checkpoint = _quantized_source_checkpoint(quantized_path)
    if source_checkpoint is None:
        LOGGER.warning(
            "api_quantized_checkpoint_ignored path=%s reason=missing_source_checkpoint",
            quantized_path,
        )
        return False
    if source_checkpoint.resolve() == primary_checkpoint_path.resolve():
        return True
    LOGGER.warning(
        "api_quantized_checkpoint_ignored path=%s source_checkpoint=%s configured_checkpoint=%s",
        quantized_path,
        source_checkpoint,
        primary_checkpoint_path,
    )
    return False


def _quantized_source_checkpoint(path: Path) -> Path | None:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    quantization = payload.get("quantization")
    if not isinstance(quantization, dict):
        return None
    source = quantization.get("source_checkpoint")
    if not isinstance(source, str) or not source.strip():
        return None
    return Path(source)


def _validate_checkpoint_tokenizer(checkpoint_path: Path, tokenizer: CodeTokenizer) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {checkpoint_path}")
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        checkpoint_vocab_size = metadata.get("vocabulary_size")
        if checkpoint_vocab_size is not None and int(checkpoint_vocab_size) != tokenizer.vocab_size:
            raise ValueError(
                "Checkpoint vocabulary size does not match tokenizer: "
                f"checkpoint={checkpoint_vocab_size} tokenizer={tokenizer.vocab_size}"
            )
    vocabulary_metadata = payload.get("vocabulary_metadata")
    expected_hash = None
    if isinstance(vocabulary_metadata, dict):
        expected_hash = vocabulary_metadata.get("tokenizer_sha256")
    actual_hash = tokenizer_file_hash(tokenizer.source_path) if tokenizer.source_path else None
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(
            "Checkpoint tokenizer hash does not match loaded tokenizer: "
            f"checkpoint={expected_hash} tokenizer={actual_hash}"
        )
    LOGGER.info(
        "api_checkpoint_tokenizer_match checkpoint=%s tokenizer=%s tokenizer_sha256=%s "
        "vocab_size=%d",
        checkpoint_path.resolve(),
        tokenizer.source_path,
        actual_hash,
        tokenizer.vocab_size,
    )


def _prompt_with_bos(prompt: str, tokenizer: CodeTokenizer) -> str:
    """Match Phase 7 SFT inputs, which prepend BOS before the formatted prompt."""

    bos_token = tokenizer.special_tokens[2]
    encoded = tokenizer.encode(prompt)
    if encoded and encoded[0] == tokenizer.bos_token_id:
        return prompt
    return f"{bos_token}{prompt}"


def _log_generation_request(
    info: LoadedModelInfo,
    prompt_before_tokenization: str,
    prompt_sent_to_model: str,
    input_token_ids: tuple[int, ...],
    settings: CodeGenerationSettings,
) -> None:
    LOGGER.info(
        "api_inference_trace_request checkpoint_path=%s tokenizer_path=%s "
        "prompt_before_tokenization=%r prompt_sent_to_model=%r input_token_ids=%s "
        "max_new_tokens=%d temperature=%s top_k=%s top_p=%s do_sample=%s "
        "repetition_penalty=%s stop_tokens=%s eos_token_handling=token_id",
        info.checkpoint_path,
        info.tokenizer_path,
        prompt_before_tokenization,
        prompt_sent_to_model,
        input_token_ids,
        settings.max_new_tokens,
        settings.temperature,
        settings.top_k,
        settings.top_p,
        settings.do_sample,
        settings.repetition_penalty,
        settings.stop_tokens,
    )


def _log_generation_result(tokenizer: CodeTokenizer, generated, final_response: str) -> None:
    LOGGER.info(
        "api_inference_trace_response generated_token_ids=%s emitted_token_ids=%s "
        "decoded_output_before_postprocessing=%r "
        "decoded_output_before_postprocessing_with_specials=%r stopped=%s "
        "final_response_returned_to_frontend=%r",
        tuple(generated.raw_generated_token_ids),
        tuple(generated.generated_token_ids),
        tokenizer.decode(generated.raw_generated_token_ids, skip_special_tokens=True),
        tokenizer.decode(generated.raw_generated_token_ids, skip_special_tokens=False),
        generated.stopped,
        final_response,
    )


__all__ = ["GenPyInferenceService", "LoadedModelInfo"]
