"""Benchmark GenPy GPT inference and optional training steps."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.checkpointing import CheckpointError, load_checkpoint
from genpy_llm.config import ConfigError, load_config
from genpy_llm.device import select_device
from genpy_llm.gpt import GPTModelError, create_gpt_model
from genpy_llm.logging_utils import setup_logging
from genpy_llm.losses import LossError
from genpy_llm.optimizers import OptimizerError
from genpy_llm.performance import (
    PerformanceError,
    autocast_context,
    build_performance_metrics,
    compile_model,
    reset_peak_memory,
    synchronize_if_cuda,
)
from genpy_llm.quantization import QuantizationError, quantize_dynamic_int8
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Run a short model benchmark."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        config = load_config(_resolve_optional_path(args.config))
        setup_logging(
            log_dir=config.paths.logs_dir,
            log_file=config.logging.log_file,
            level="DEBUG" if args.debug else config.logging.level,
        )
        device = select_device(args.device or config.training.device)
        mixed_precision = args.mixed_precision or config.optimization.mixed_precision
        compile_enabled = args.compile or config.optimization.torch_compile
        compile_mode = args.compile_mode or config.optimization.compile_mode
        quantization = args.quantization or config.optimization.quantization
        warmup_steps = args.warmup_steps or config.optimization.benchmark_warmup_steps
        steps = args.steps or config.optimization.benchmark_steps
        sequence_length = args.sequence_length or min(32, config.model.context_length)
        if sequence_length > config.model.context_length:
            raise ValueError("sequence_length must be less than or equal to model.context_length.")

        model, metadata = create_gpt_model(config.data.vocabulary_file, config)
        load_checkpoint(
            _resolve_path(args.checkpoint),
            model,
            optimizer=None,
            map_location=device,
            restore_rng=False,
        )
        model.to(device)
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        if quantization == "dynamic_int8":
            if device.type != "cpu":
                raise QuantizationError("dynamic_int8 quantization is supported only on CPU.")
            model = quantize_dynamic_int8(model.cpu())
            compile_enabled = False
        elif quantization != "none":
            raise QuantizationError("quantization must be 'none' or 'dynamic_int8'.")
        model = compile_model(model, enabled=compile_enabled, mode=compile_mode)
        model.eval()
        input_ids = torch.randint(
            low=0,
            high=metadata.vocab_size,
            size=(args.batch_size, sequence_length),
            dtype=torch.long,
            device=device,
        )
        output_shape, inference_metrics = _benchmark_inference(
            model=model,
            input_ids=input_ids,
            device=device,
            warmup_steps=warmup_steps,
            steps=steps,
            mixed_precision=mixed_precision,
        )
    except (
        CheckpointError,
        ConfigError,
        FileNotFoundError,
        GPTModelError,
        IsADirectoryError,
        LossError,
        OptimizerError,
        OSError,
        PerformanceError,
        QuantizationError,
        RuntimeError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    print("GenPy LLM Benchmark")
    print("===================")
    print(f"Checkpoint: {_resolve_path(args.checkpoint)}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Model parameters: {metadata.total_parameters}")
    print(f"Output shape: {output_shape}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"torch.compile: {compile_enabled} ({compile_mode})")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"Quantization: {quantization}")
    print(f"Average step time: {inference_metrics.elapsed_seconds / steps:.6f}s")
    print(f"Tokens per second: {inference_metrics.tokens_per_second:.2f}")
    print(f"Peak memory MB: {inference_metrics.peak_memory_mb}")
    return 0


def _benchmark_inference(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    steps: int,
    mixed_precision: str,
) -> tuple[tuple[int, ...], object]:
    with torch.no_grad():
        for _ in range(warmup_steps):
            with autocast_context(mixed_precision, device):
                output = model(input_ids)
        reset_peak_memory(device)
        synchronize_if_cuda(device)
        start = time.perf_counter()
        for _ in range(steps):
            with autocast_context(mixed_precision, device):
                output = model(input_ids)
        synchronize_if_cuda(device)
    elapsed = time.perf_counter() - start
    metrics = build_performance_metrics(
        elapsed_seconds=elapsed,
        processed_tokens=int(input_ids.numel()) * steps,
        device=device,
    )
    return tuple(output.shape), metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GenPy GPT.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--sequence-length", type=_positive_int, default=None)
    parser.add_argument("--warmup-steps", type=_positive_int, default=None)
    parser.add_argument("--steps", type=_positive_int, default=None)
    parser.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--quantization", choices=["none", "dynamic_int8"], default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
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
        logger.exception("Benchmark failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
