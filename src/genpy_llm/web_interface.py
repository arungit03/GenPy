"""Local Gradio interface for GenPy LLM text generation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from genpy_llm.config import AppConfig
from genpy_llm.device import select_device
from genpy_llm.generation import GenerationError, TextGenerator, create_generator_from_checkpoint
from genpy_llm.performance import compile_model
from genpy_llm.quantization import QuantizationError, quantize_dynamic_int8

LOGGER = logging.getLogger("genpy_llm")


class WebInterfaceError(ValueError):
    """Raised when the local web interface cannot be constructed."""


@dataclass(frozen=True)
class WebInterfaceState:
    """Loaded model details displayed in the local interface."""

    checkpoint_path: Path
    device: str
    quantization: str
    torch_compile: bool
    compile_mode: str

    @property
    def summary(self) -> str:
        """Return a compact startup summary."""

        return (
            f"Checkpoint: {self.checkpoint_path}\n"
            f"Device: {self.device}\n"
            f"Quantization: {self.quantization}\n"
            f"torch.compile: {self.torch_compile} ({self.compile_mode})"
        )


class GenPyWebInterface:
    """Build and serve a Gradio UI around an already loaded TextGenerator."""

    def __init__(
        self,
        *,
        generator: TextGenerator,
        config: AppConfig,
        state: WebInterfaceState,
    ) -> None:
        if not isinstance(generator, TextGenerator):
            raise WebInterfaceError("generator must be a TextGenerator.")
        self.generator = generator
        self.config = config
        self.state = state

    def build(self):
        """Construct the Gradio Blocks object without launching a server."""

        gradio = _import_gradio()
        with gradio.Blocks(title=self.config.web_interface.title, analytics_enabled=False) as app:
            gradio.Markdown(f"# {self.config.web_interface.title}")
            gradio.Markdown(self.config.web_interface.description)
            with gradio.Row():
                with gradio.Column(scale=2):
                    prompt = gradio.Textbox(
                        label="Prompt",
                        value=self.config.web_interface.default_prompt,
                        lines=7,
                        max_lines=14,
                    )
                    with gradio.Row():
                        generate_button = gradio.Button("Generate", variant="primary")
                        clear_button = gradio.Button("Clear")
                    generated_text = gradio.Textbox(
                        label="Generated text",
                        lines=10,
                        interactive=False,
                    )
                with gradio.Column(scale=1):
                    max_new_tokens = gradio.Slider(
                        label="Max new tokens",
                        minimum=1,
                        maximum=max(1, self.config.generation.max_new_tokens * 4),
                        step=1,
                        value=self.config.generation.max_new_tokens,
                    )
                    temperature = gradio.Slider(
                        label="Temperature",
                        minimum=0.05,
                        maximum=2.0,
                        step=0.05,
                        value=self.config.generation.temperature,
                    )
                    top_k = gradio.Number(
                        label="Top-k",
                        value=self.config.generation.top_k or 0,
                        precision=0,
                    )
                    top_p = gradio.Slider(
                        label="Top-p",
                        minimum=0.01,
                        maximum=1.0,
                        step=0.01,
                        value=self.config.generation.top_p or 1.0,
                    )
                    repetition_penalty = gradio.Slider(
                        label="Repetition penalty",
                        minimum=0.1,
                        maximum=3.0,
                        step=0.05,
                        value=self.config.generation.repetition_penalty,
                    )
                    greedy = gradio.Checkbox(
                        label="Greedy",
                        value=not self.config.generation.do_sample,
                    )
                    status = gradio.Textbox(label="Status", value="Ready", interactive=False)
                    gradio.Textbox(
                        label="Checkpoint and device",
                        value=self.state.summary,
                        lines=4,
                        interactive=False,
                    )
                    generation_time = gradio.Number(
                        label="Generation time",
                        value=0.0,
                        precision=4,
                        interactive=False,
                    )
                    generated_tokens = gradio.Number(
                        label="Generated token count",
                        value=0,
                        precision=0,
                        interactive=False,
                    )
                    tokens_per_second = gradio.Number(
                        label="Tokens per second",
                        value=0.0,
                        precision=2,
                        interactive=False,
                    )
                    eos_status = gradio.Textbox(label="EOS status", value="No", interactive=False)

            outputs = [
                generated_text,
                status,
                generation_time,
                generated_tokens,
                tokens_per_second,
                eos_status,
            ]
            generate_button.click(
                fn=self.generate,
                inputs=[
                    prompt,
                    max_new_tokens,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    greedy,
                ],
                outputs=outputs,
            )
            clear_button.click(fn=self.clear, inputs=None, outputs=[prompt, *outputs])
        return app

    def launch(self, *, host: str, port: int, share: bool) -> None:
        """Launch the local Gradio server."""

        self.build().launch(server_name=host, server_port=port, share=share)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | float,
        temperature: int | float,
        top_k: int | float | None,
        top_p: int | float,
        repetition_penalty: int | float,
        greedy: bool,
    ) -> tuple[str, str, float, int, float, str]:
        """Generate text and return UI field values."""

        try:
            cleaned_prompt = _validate_prompt(
                prompt,
                self.config.web_interface.max_prompt_characters,
            )
            normalized_top_k = _normalize_optional_int(top_k)
            start = time.perf_counter()
            result = self.generator.generate(
                prompt=cleaned_prompt,
                max_new_tokens=_positive_int(max_new_tokens, "max_new_tokens"),
                temperature=_positive_float(temperature, "temperature"),
                top_k=normalized_top_k,
                top_p=_top_p(top_p),
                do_sample=not bool(greedy),
                repetition_penalty=_positive_float(repetition_penalty, "repetition_penalty"),
                stop_on_eos=self.config.generation.stop_on_eos,
            )
            elapsed = time.perf_counter() - start
            token_count = len(result.generated_token_ids)
            tokens_per_second = token_count / elapsed if elapsed > 0 else 0.0
            eos_status = "Yes" if result.stopped_on_eos else "No"
            return (
                result.generated_text,
                "Generation complete",
                float(elapsed),
                token_count,
                float(tokens_per_second),
                eos_status,
            )
        except (GenerationError, ValueError) as exc:
            return ("", f"Generation failed: {exc}", 0.0, 0, 0.0, "No")
        except Exception as exc:  # noqa: BLE001 - hide tracebacks from UI users.
            LOGGER.exception("Unexpected web generation failure.")
            return ("", f"Generation failed: {type(exc).__name__}", 0.0, 0, 0.0, "No")

    def clear(self) -> tuple[str, str, str, float, int, float, str]:
        """Clear prompt, output, and metrics."""

        return (
            "",
            "",
            "Ready",
            0.0,
            0,
            0.0,
            "No",
        )


class GenPyCodeWebInterface:
    """Build a Gradio UI around a loaded code model and CodeTokenizer."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        config,
        state: WebInterfaceState,
        instruction_mode: bool,
        code_only: bool,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.state = state
        self.instruction_mode = instruction_mode
        self.code_only = code_only

    def build(self):
        """Construct the code-focused Gradio Blocks object."""

        gradio = _import_gradio()
        with gradio.Blocks(title="GenPy Code LLM", analytics_enabled=False) as app:
            gradio.Markdown("# GenPy Code LLM")
            gradio.Markdown("Generate Python code from a local GenPy code checkpoint.")
            with gradio.Row():
                with gradio.Column(scale=2):
                    prompt = gradio.Textbox(
                        label="Prompt",
                        value="Write Python code to reverse a string",
                        lines=6,
                    )
                    with gradio.Row():
                        generate_button = gradio.Button("Generate", variant="primary")
                        clear_button = gradio.Button("Clear")
                    generated_code = gradio.Code(label="Generated Python", language="python")
                with gradio.Column(scale=1):
                    max_new_tokens = gradio.Slider(
                        label="Max new tokens",
                        minimum=1,
                        maximum=max(1, self.config.generation.max_new_tokens * 4),
                        step=1,
                        value=self.config.generation.max_new_tokens,
                    )
                    temperature = gradio.Slider(
                        label="Temperature",
                        minimum=0.05,
                        maximum=2.0,
                        step=0.05,
                        value=self.config.generation.temperature,
                    )
                    top_k = gradio.Number(
                        label="Top-k",
                        value=self.config.generation.top_k or 0,
                        precision=0,
                    )
                    top_p = gradio.Slider(
                        label="Top-p",
                        minimum=0.01,
                        maximum=1.0,
                        step=0.01,
                        value=self.config.generation.top_p or 1.0,
                    )
                    repetition_penalty = gradio.Slider(
                        label="Repetition penalty",
                        minimum=0.1,
                        maximum=3.0,
                        step=0.05,
                        value=self.config.generation.repetition_penalty,
                    )
                    greedy = gradio.Checkbox(label="Greedy", value=False)
                    status = gradio.Textbox(label="Status", value="Ready", interactive=False)
                    checkpoint_info = gradio.Textbox(
                        label="Checkpoint type and vocabulary",
                        value=f"{self.state.summary}\nVocabulary size: {self.tokenizer.vocab_size}",
                        lines=5,
                        interactive=False,
                    )
                    del checkpoint_info
                    generation_time = gradio.Number(
                        label="Generation time",
                        value=0.0,
                        precision=4,
                        interactive=False,
                    )
                    generated_tokens = gradio.Number(
                        label="Generated token count",
                        value=0,
                        precision=0,
                        interactive=False,
                    )
                    tokens_per_second = gradio.Number(
                        label="Tokens per second",
                        value=0.0,
                        precision=2,
                        interactive=False,
                    )
                    eos_status = gradio.Textbox(label="EOS status", value="No", interactive=False)
            outputs = [
                generated_code,
                status,
                generation_time,
                generated_tokens,
                tokens_per_second,
                eos_status,
            ]
            generate_button.click(
                fn=self.generate,
                inputs=[
                    prompt,
                    max_new_tokens,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    greedy,
                ],
                outputs=outputs,
            )
            clear_button.click(fn=self.clear, inputs=None, outputs=[prompt, *outputs])
        return app

    def launch(self, *, host: str, port: int, share: bool) -> None:
        """Launch the local code UI."""

        self.build().launch(server_name=host, server_port=port, share=share)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | float,
        temperature: int | float,
        top_k: int | float | None,
        top_p: int | float,
        repetition_penalty: int | float,
        greedy: bool,
    ) -> tuple[str, str, float, int, float, str]:
        """Generate code and return UI field values."""

        try:
            from genpy_llm.code_generation import generate_code_text

            result = generate_code_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=_validate_prompt(prompt, 4000),
                device=next(self.model.parameters()).device,
                max_new_tokens=_positive_int(max_new_tokens, "max_new_tokens"),
                temperature=_positive_float(temperature, "temperature"),
                top_k=_normalize_optional_int(top_k),
                top_p=_top_p(top_p),
                repetition_penalty=_positive_float(repetition_penalty, "repetition_penalty"),
                do_sample=not bool(greedy),
                stop_on_eos=self.config.generation.stop_on_eos,
                instruction_mode=self.instruction_mode,
                code_only=self.code_only,
                context_length=self.config.model.context_length,
            )
            return (
                result.text,
                "Generation complete",
                result.elapsed_seconds,
                len(result.generated_token_ids),
                result.tokens_per_second,
                "Yes" if result.stopped_on_eos else "No",
            )
        except Exception as exc:  # noqa: BLE001 - hide tracebacks from UI users.
            LOGGER.exception("Unexpected code web generation failure.")
            return ("", f"Generation failed: {type(exc).__name__}", 0.0, 0, 0.0, "No")

    def clear(self) -> tuple[str, str, str, float, int, float, str]:
        """Clear code UI fields."""

        return ("", "", "Ready", 0.0, 0, 0.0, "No")


def create_web_interface(
    *,
    config: AppConfig,
    checkpoint_path: Path,
    config_path: Path | None,
    vocabulary_path: Path | None,
    device_name: str,
    quantization: str,
    torch_compile: bool,
    compile_mode: str,
) -> GenPyWebInterface:
    """Load the generator once and return a ready local web interface."""

    device = select_device(device_name)
    generator = create_generator_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        vocabulary_path=vocabulary_path,
        device=device,
    )
    if quantization == "dynamic_int8":
        if device.type != "cpu":
            raise QuantizationError("dynamic_int8 quantization is supported only on CPU.")
        generator.model = quantize_dynamic_int8(generator.model)
        torch_compile = False
    elif quantization != "none":
        raise QuantizationError("quantization must be 'none' or 'dynamic_int8'.")
    if torch_compile:
        generator.model = compile_model(generator.model, enabled=True, mode=compile_mode)
    return GenPyWebInterface(
        generator=generator,
        config=config,
        state=WebInterfaceState(
            checkpoint_path=checkpoint_path,
            device=str(device),
            quantization=quantization,
            torch_compile=torch_compile,
            compile_mode=compile_mode,
        ),
    )


def create_code_web_interface(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device_name: str,
    instruction_mode: bool,
    code_only: bool,
) -> GenPyCodeWebInterface:
    """Load a code model once and return a ready local code web interface."""

    from genpy_llm.code_generation import load_code_model_for_generation
    from genpy_llm.code_tokenizer import CodeTokenizer
    from genpy_llm.code_training import load_code_config

    config = load_code_config(config_path)
    tokenizer = CodeTokenizer.from_file(config.tokenizer.path)
    device = select_device(device_name)
    model = load_code_model_for_generation(
        config=config,
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return GenPyCodeWebInterface(
        model=model,
        tokenizer=tokenizer,
        config=config,
        state=WebInterfaceState(
            checkpoint_path=checkpoint_path,
            device=str(device),
            quantization="none",
            torch_compile=False,
            compile_mode="default",
        ),
        instruction_mode=instruction_mode,
        code_only=code_only,
    )


def _import_gradio():
    try:
        import gradio as gradio
    except ImportError as exc:
        raise WebInterfaceError(
            "Gradio is required for the local web interface. Install requirements.txt first."
        ) from exc
    return gradio


def _validate_prompt(prompt: str, max_characters: int) -> str:
    if not isinstance(prompt, str):
        raise WebInterfaceError("prompt must be a string.")
    if len(prompt) > max_characters:
        raise WebInterfaceError(f"prompt must be at most {max_characters} characters.")
    cleaned = prompt.strip()
    if not cleaned:
        raise WebInterfaceError("prompt must not be empty.")
    return prompt


def _positive_int(value: int | float, name: str) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WebInterfaceError(f"{name} must be greater than 0.")
    number = int(value)
    if number <= 0 or number != float(value):
        raise WebInterfaceError(f"{name} must be an integer greater than 0.")
    return number


def _positive_float(value: int | float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
        raise WebInterfaceError(f"{name} must be greater than 0.")
    return float(value)


def _top_p(value: int | float) -> float:
    number = _positive_float(value, "top_p")
    if number > 1:
        raise WebInterfaceError("top_p must be at most 1.")
    return number


def _normalize_optional_int(value: int | float | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WebInterfaceError("top_k must be an integer greater than or equal to 0.")
    if float(value) == 0:
        return None
    number = int(value)
    if number <= 0 or number != float(value):
        raise WebInterfaceError("top_k must be an integer greater than or equal to 0.")
    return number


__all__ = [
    "GenPyWebInterface",
    "GenPyCodeWebInterface",
    "WebInterfaceError",
    "WebInterfaceState",
    "create_code_web_interface",
    "create_web_interface",
]
