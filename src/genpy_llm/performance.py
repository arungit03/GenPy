"""Performance helpers for GenPy LLM training and inference."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

SUPPORTED_MIXED_PRECISION = {"none", "fp16", "bf16"}
SUPPORTED_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune"}
PRECISION_ALIASES = {
    "fp32": "none",
    "float32": "none",
    "full": "none",
    "full_precision": "none",
}
LOGGER = logging.getLogger("genpy_llm.performance")


class PerformanceError(ValueError):
    """Raised when an optimization option is invalid or unsupported."""


@dataclass(frozen=True)
class PerformanceMetrics:
    """Lightweight throughput and memory metrics."""

    elapsed_seconds: float
    processed_tokens: int
    tokens_per_second: float
    peak_memory_mb: float | None
    device: str


def validate_mixed_precision(mixed_precision: str, device: torch.device) -> None:
    """Validate an AMP mode for a device."""

    mixed_precision = normalize_mixed_precision(mixed_precision)
    if mixed_precision not in SUPPORTED_MIXED_PRECISION:
        raise PerformanceError(_precision_error_message(mixed_precision))
    if mixed_precision == "none":
        return
    if mixed_precision == "fp16" and device.type == "mps":
        if _mps_fp16_autocast_supported():
            return
        raise PerformanceError(
            "fp16 mixed precision is not supported by this PyTorch/MPS installation."
        )
    if mixed_precision == "fp16" and device.type != "cuda":
        raise PerformanceError("fp16 mixed precision requires a CUDA device.")
    if mixed_precision == "bf16":
        if device.type == "cpu":
            return
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return
        raise PerformanceError("bf16 mixed precision is not supported on this device.")


def normalize_mixed_precision(value: str, *, allow_fp32_alias: bool = False) -> str:
    """Normalize precision strings into GenPy's canonical mixed-precision values."""

    if not isinstance(value, str) or not value.strip():
        raise PerformanceError("mixed_precision must be one of: none, fp16, bf16.")
    normalized = value.strip().lower().replace("-", "_")
    if allow_fp32_alias and normalized in PRECISION_ALIASES:
        return PRECISION_ALIASES[normalized]
    if normalized not in SUPPORTED_MIXED_PRECISION:
        raise PerformanceError(_precision_error_message(value))
    return normalized


def resolve_mixed_precision(
    mixed_precision: str,
    device: torch.device,
    *,
    logger: logging.Logger | None = None,
) -> str:
    """Return an effective AMP mode for a device, with safe MPS fallbacks."""

    selected = normalize_mixed_precision(mixed_precision)
    active_logger = logger or LOGGER
    if device.type != "mps":
        validate_mixed_precision(selected, device)
        return selected
    if selected == "none":
        return "none"
    if selected == "bf16":
        active_logger.warning(
            "bf16 mixed precision is not supported on Apple MPS; using full precision."
        )
        return "none"
    if selected == "fp16":
        if _mps_fp16_autocast_supported():
            return "fp16"
        active_logger.warning(
            "fp16 mixed precision is not supported by this PyTorch/MPS installation; "
            "using full precision."
        )
        return "none"
    raise PerformanceError(_precision_error_message(selected))


def autocast_context(mixed_precision: str, device: torch.device):
    """Return an autocast context for the requested mode."""

    mixed_precision = normalize_mixed_precision(mixed_precision)
    validate_mixed_precision(mixed_precision, device)
    if mixed_precision == "none":
        return contextlib.nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def create_grad_scaler(mixed_precision: str, device: torch.device):
    """Create a GradScaler only for CUDA fp16 training."""

    mixed_precision = normalize_mixed_precision(mixed_precision)
    validate_mixed_precision(mixed_precision, device)
    if mixed_precision != "fp16" or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler(device="cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def compile_model(
    model: nn.Module,
    enabled: bool,
    mode: str = "default",
) -> nn.Module:
    """Optionally compile a model with torch.compile."""

    if not isinstance(model, nn.Module):
        raise PerformanceError("model must be a torch.nn.Module.")
    if not isinstance(enabled, bool):
        raise PerformanceError("enabled must be true or false.")
    if mode not in SUPPORTED_COMPILE_MODES:
        raise PerformanceError("compile mode must be default, reduce-overhead, or max-autotune.")
    if not enabled:
        return model
    if hasattr(model, "_orig_mod"):
        return model
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        raise PerformanceError("torch.compile is not available in this PyTorch version.")
    try:
        return compiler(model, mode=mode)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PerformanceError(f"torch.compile failed: {exc}") from exc


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the underlying module for torch.compile wrappers."""

    wrapped = getattr(model, "_orig_mod", None)
    if isinstance(wrapped, nn.Module):
        return wrapped
    return model


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA before/after timing."""

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak memory stats when available."""

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    """Return CUDA peak memory in MiB, or None on CPU/unsupported devices."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def _mps_fp16_autocast_supported() -> bool:
    if not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return False
    try:
        with torch.autocast(device_type="mps", dtype=torch.float16):
            return True
    except (RuntimeError, TypeError, AttributeError):
        return False


def _precision_error_message(value: object) -> str:
    return (
        f"Unsupported mixed precision value {value!r}. "
        "Use mixed_precision: none, fp16, or bf16. "
        "Legacy training.precision: fp32 is accepted and mapped to mixed_precision: none."
    )


def build_performance_metrics(
    *,
    elapsed_seconds: float,
    processed_tokens: int,
    device: torch.device,
) -> PerformanceMetrics:
    """Build throughput metrics from elapsed time and token count."""

    if elapsed_seconds <= 0:
        tokens_per_second = 0.0
    else:
        tokens_per_second = processed_tokens / elapsed_seconds
    return PerformanceMetrics(
        elapsed_seconds=float(elapsed_seconds),
        processed_tokens=int(processed_tokens),
        tokens_per_second=float(tokens_per_second),
        peak_memory_mb=peak_memory_mb(device),
        device=str(device),
    )


class StepTimer:
    """Small CUDA-aware timer context."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.elapsed_seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> StepTimer:
        synchronize_if_cuda(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        synchronize_if_cuda(self.device)
        self.elapsed_seconds = time.perf_counter() - self._start


__all__ = [
    "PerformanceError",
    "PerformanceMetrics",
    "StepTimer",
    "autocast_context",
    "build_performance_metrics",
    "compile_model",
    "create_grad_scaler",
    "normalize_mixed_precision",
    "peak_memory_mb",
    "reset_peak_memory",
    "resolve_mixed_precision",
    "synchronize_if_cuda",
    "unwrap_compiled_model",
    "validate_mixed_precision",
]
