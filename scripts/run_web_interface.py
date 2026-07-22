"""Run the local GenPy LLM Gradio interface."""

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
from genpy_llm.code_training import load_code_config
from genpy_llm.config import ConfigError, load_config
from genpy_llm.generation import GenerationError
from genpy_llm.gpt import GPTModelError
from genpy_llm.logging_utils import setup_logging
from genpy_llm.performance import PerformanceError
from genpy_llm.quantization import QuantizationError
from genpy_llm.utils import set_seed
from genpy_llm.vocabulary import VocabularyError
from genpy_llm.web_interface import (
    WebInterfaceError,
    create_code_web_interface,
    create_web_interface,
)


def main() -> int:
    """Load a checkpoint once and launch the local UI."""

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
        if args.checkpoint is not None:
            checkpoint_path = _resolve_path(args.checkpoint)
        elif args.code_mode:
            code_config = load_code_config(_resolve_path(args.code_config))
            checkpoint_path = (
                code_config.fine_tuning.output_directory / code_config.fine_tuning.best_filename
            )
        else:
            checkpoint_path = config.web_interface.default_checkpoint
        vocabulary_path = _resolve_optional_path(args.vocabulary)
        device_name = args.device or config.training.device
        quantization = args.quantization or config.optimization.quantization
        compile_enabled = args.compile or config.optimization.torch_compile
        compile_mode = args.compile_mode or config.optimization.compile_mode
        host = args.host or config.web_interface.host
        port = args.port if args.port is not None else config.web_interface.port
        share = args.share if args.share else config.web_interface.share

        if args.code_mode:
            app = create_code_web_interface(
                config_path=_resolve_path(args.code_config),
                checkpoint_path=checkpoint_path,
                device_name=device_name,
                instruction_mode=args.instruction_mode,
                code_only=args.code_only,
            )
        else:
            app = create_web_interface(
                config=config,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                vocabulary_path=vocabulary_path,
                device_name=device_name,
                quantization=quantization,
                torch_compile=compile_enabled,
                compile_mode=compile_mode,
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
        WebInterfaceError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    print("GenPy LLM Web Interface")
    print("=======================")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {app.state.device}")
    print(f"Quantization: {app.state.quantization}")
    print(f"torch.compile: {app.state.torch_compile} ({app.state.compile_mode})")
    print(f"URL: http://{host}:{port}")
    app.launch(host=host, port=port, share=share)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local GenPy LLM Gradio interface.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--code-config", type=Path, default=Path("configs/code_small.yaml"))
    parser.add_argument("--vocabulary", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=_port, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--quantization", choices=["none", "dynamic_int8"], default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--code-mode", action="store_true")
    parser.add_argument("--instruction-mode", action="store_true")
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535.")
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
        logger.exception("Web interface startup failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
