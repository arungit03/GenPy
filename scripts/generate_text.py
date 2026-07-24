"""Generate text from a trained GenPy GPT checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.checkpointing import CheckpointError
from genpy_llm.compat import zip_strict
from genpy_llm.config import ConfigError, load_config
from genpy_llm.device import select_device
from genpy_llm.generation import GenerationError, create_generator_from_checkpoint
from genpy_llm.gpt import GPTModelError
from genpy_llm.logging_utils import setup_logging
from genpy_llm.performance import PerformanceError, compile_model
from genpy_llm.quantization import QuantizationError, quantize_dynamic_int8
from genpy_llm.utils import set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Parse arguments and generate text."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        config_path = _resolve_optional_path(args.config)
        config = load_config(config_path)
        setup_logging(
            log_dir=config.paths.logs_dir,
            log_file=config.logging.log_file,
            level="DEBUG" if args.debug else config.logging.level,
        )
        set_seed(args.seed if args.seed is not None else config.training.seed)
        device = select_device(args.device or config.training.device)
        compile_enabled = args.compile or config.optimization.torch_compile
        compile_mode = args.compile_mode or config.optimization.compile_mode
        quantization = args.quantization or config.optimization.quantization
        checkpoint_path = _resolve_path(args.checkpoint)
        vocabulary_path = _resolve_optional_path(args.vocabulary)
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
            compile_enabled = False
        elif quantization != "none":
            raise QuantizationError("quantization must be 'none' or 'dynamic_int8'.")
        if compile_enabled:
            generator.model = compile_model(generator.model, enabled=True, mode=compile_mode)
        result = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens
            if args.max_new_tokens is not None
            else config.generation.max_new_tokens,
            temperature=args.temperature
            if args.temperature is not None
            else config.generation.temperature,
            top_k=args.top_k if args.top_k is not None else config.generation.top_k,
            top_p=args.top_p if args.top_p is not None else config.generation.top_p,
            do_sample=not args.greedy if args.greedy else config.generation.do_sample,
            repetition_penalty=args.repetition_penalty
            if args.repetition_penalty is not None
            else config.generation.repetition_penalty,
            stop_on_eos=config.generation.stop_on_eos,
        )
    except (
        CheckpointError,
        ConfigError,
        FileNotFoundError,
        GenerationError,
        GPTModelError,
        IsADirectoryError,
        OSError,
        PerformanceError,
        QuantizationError,
        RuntimeError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    print("GenPy LLM Text Generation")
    print("=========================")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"torch.compile: {compile_enabled} ({compile_mode})")
    print(f"Quantization: {quantization}")
    unknown_prompt_tokens = _unknown_prompt_tokens(result, generator.vocabulary.unknown_id)
    if unknown_prompt_tokens:
        print(f"Unknown prompt tokens: {', '.join(unknown_prompt_tokens)}")
    print()
    print(result.generated_text)
    if args.show_tokens:
        print()
        print("Prompt tokens:", result.prompt_tokens)
        print("Prompt token IDs:", result.prompt_token_ids)
        print("Generated tokens:", result.generated_tokens)
        print("Generated token IDs:", result.generated_token_ids)
        print(f"Stopped on EOS: {result.stopped_on_eos}")
        print(f"Total tokens: {result.total_tokens}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a GPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--vocabulary", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=None)
    parser.add_argument("--temperature", type=_positive_float, default=None)
    parser.add_argument("--top-k", type=_positive_int, default=None)
    parser.add_argument("--top-p", type=_top_p, default=None)
    parser.add_argument("--repetition-penalty", type=_positive_float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--quantization", choices=["none", "dynamic_int8"], default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-tokens", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _unknown_prompt_tokens(result, unknown_id: int) -> tuple[str, ...]:
    return tuple(
        token
        for token, token_id in zip_strict(result.prompt_tokens, result.prompt_token_ids)
        if token_id == unknown_id and token != "<UNK>"
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _top_p(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("value must be greater than zero and at most one.")
    return number


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return _resolve_path(path)


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("Text generation failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
