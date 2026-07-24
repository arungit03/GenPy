"""Inference quantization helpers for GenPy LLM."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

UTC = timezone.utc

QUANTIZATION_FORMAT_VERSION = 1
SUPPORTED_QUANTIZATION_METHODS = {"fp16", "bf16", "dynamic_int8"}


class QuantizationError(ValueError):
    """Raised when quantization is unsupported or invalid."""


@dataclass(frozen=True)
class BackendCapabilities:
    """Quantization support detected for one runtime device."""

    device: str
    fp16: bool
    bf16: bool
    dynamic_int8: bool
    skip_reasons: Mapping[str, str]


@dataclass(frozen=True)
class QuantizedCheckpointInfo:
    """Metadata written alongside a quantized inference checkpoint."""

    method: str
    dtype: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    created_at: str
    parameter_count: int
    model_state_bytes: int


@dataclass(frozen=True)
class LoadedQuantizedCheckpoint:
    """Summary returned after loading a quantized inference checkpoint."""

    model: nn.Module
    method: str
    checkpoint_path: Path
    source_checkpoint: str
    model_state_bytes: int
    metadata: Mapping[str, Any]


class QuantizedInferenceModel(nn.Module):
    """Inference-only wrapper around a dynamically quantized model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        self.vocab_size = model.vocab_size
        self.context_length = model.context_length

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        if mode:
            raise QuantizationError("Quantized models are for inference only.")
        self.model.eval()
        return super().train(False)


def detect_backend_capabilities(device: torch.device | str) -> BackendCapabilities:
    """Return quantization methods that can run inference on the selected backend."""

    resolved = device if isinstance(device, torch.device) else torch.device(str(device))
    skip_reasons: dict[str, str] = {}
    fp16 = resolved.type in {"cuda", "mps"}
    if not fp16:
        skip_reasons["fp16"] = "fp16 inference is enabled only for CUDA or Apple MPS."

    bf16 = resolved.type == "cpu" or (
        resolved.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    if not bf16:
        skip_reasons["bf16"] = "bf16 inference is supported on CPU or CUDA with bf16 support."

    dynamic_int8 = resolved.type == "cpu" and _dynamic_int8_engine() is not None
    if not dynamic_int8:
        skip_reasons["dynamic_int8"] = (
            "dynamic INT8 quantization requires CPU and an available quantized engine."
        )

    return BackendCapabilities(
        device=str(resolved),
        fp16=fp16,
        bf16=bf16,
        dynamic_int8=dynamic_int8,
        skip_reasons=skip_reasons,
    )


def is_quantization_supported(method: str, device: torch.device | str) -> bool:
    """Return whether a quantization method can run on a device."""

    method = normalize_quantization_method(method)
    capabilities = detect_backend_capabilities(device)
    return bool(getattr(capabilities, method))


def normalize_quantization_method(method: str) -> str:
    """Normalize and validate a Phase 10 quantization method."""

    if not isinstance(method, str) or not method.strip():
        raise QuantizationError("quantization method must be a non-empty string.")
    normalized = method.strip().lower().replace("-", "_")
    if normalized in {"int8", "dynamic"}:
        normalized = "dynamic_int8"
    if normalized not in SUPPORTED_QUANTIZATION_METHODS:
        methods = ", ".join(sorted(SUPPORTED_QUANTIZATION_METHODS))
        raise QuantizationError(f"quantization method must be one of: {methods}.")
    return normalized


def convert_model_precision(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Return an inference-only deep copy converted to a floating-point dtype."""

    if not isinstance(model, nn.Module):
        raise QuantizationError("model must be a torch.nn.Module.")
    if dtype not in {torch.float16, torch.bfloat16}:
        raise QuantizationError("dtype must be torch.float16 or torch.bfloat16.")
    converted = copy.deepcopy(model).eval()
    for parameter in converted.parameters():
        parameter.requires_grad = False
    return converted.to(dtype=dtype)


def quantize_fp16(model: nn.Module) -> nn.Module:
    """Return an FP16 inference copy of a model."""

    return convert_model_precision(model, torch.float16)


def quantize_bf16(model: nn.Module) -> nn.Module:
    """Return a BF16 inference copy of a model."""

    return convert_model_precision(model, torch.bfloat16)


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Return a CPU dynamic-INT8 quantized copy of a model."""

    if not isinstance(model, nn.Module):
        raise QuantizationError("model must be a torch.nn.Module.")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if device.type != "cpu":
        raise QuantizationError("dynamic_int8 quantization is supported only on CPU.")
    engine = _dynamic_int8_engine()
    if engine is None:
        raise QuantizationError("dynamic_int8 quantization requires a PyTorch quantized engine.")
    torch.backends.quantized.engine = engine
    quantized_source = copy.deepcopy(model).cpu().eval()
    for parameter in quantized_source.parameters():
        parameter.requires_grad = False
    quantized = torch.ao.quantization.quantize_dynamic(
        quantized_source,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    return QuantizedInferenceModel(quantized)


def quantize_model(model: nn.Module, method: str) -> nn.Module:
    """Return an inference copy quantized with a supported Phase 10 method."""

    method = normalize_quantization_method(method)
    if method == "fp16":
        return quantize_fp16(model)
    if method == "bf16":
        return quantize_bf16(model)
    return quantize_dynamic_int8(model)


def save_quantized_checkpoint(
    *,
    model: nn.Module,
    output_path: Path | str,
    method: str,
    source_checkpoint: Path | str,
    source_metadata: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
    vocabulary_metadata: Mapping[str, Any] | None = None,
) -> QuantizedCheckpointInfo:
    """Save a model-only quantized checkpoint without modifying the source checkpoint."""

    if not isinstance(model, nn.Module):
        raise QuantizationError("model must be a torch.nn.Module.")
    method = normalize_quantization_method(method)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_checkpoint)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    state_dict = _cpu_tensor_tree(model.state_dict())
    info = QuantizedCheckpointInfo(
        method=method,
        dtype=_method_dtype_name(method),
        source_checkpoint=str(source_path.resolve()),
        source_checkpoint_sha256=file_sha256(source_path),
        created_at=datetime.now(UTC).isoformat(),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        model_state_bytes=tensor_tree_nbytes(state_dict),
    )
    payload = {
        "format_version": QUANTIZATION_FORMAT_VERSION,
        "quantization": asdict(info),
        "source_metadata": dict(source_metadata or {}),
        "model_config": dict(model_config or {}),
        "vocabulary_metadata": dict(vocabulary_metadata or {}),
        "model_state_dict": state_dict,
    }
    torch.save(payload, destination)
    return info


def load_quantized_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> LoadedQuantizedCheckpoint:
    """Load a Phase 10 quantized inference checkpoint into a matching model instance."""

    input_path = Path(checkpoint_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Quantized checkpoint not found: {input_path}")
    payload = _torch_load(input_path, map_location=map_location)
    _validate_quantized_payload(payload, input_path)
    quantization = payload["quantization"]
    method = normalize_quantization_method(str(quantization["method"]))
    target = _prepare_load_target(model, method)
    try:
        target.load_state_dict(payload["model_state_dict"], strict=strict)
    except RuntimeError as exc:
        raise QuantizationError(f"Could not load quantized checkpoint {input_path}: {exc}") from exc
    return LoadedQuantizedCheckpoint(
        model=target,
        method=method,
        checkpoint_path=input_path.resolve(),
        source_checkpoint=str(quantization["source_checkpoint"]),
        model_state_bytes=int(quantization.get("model_state_bytes", 0)),
        metadata=payload,
    )


def model_state_nbytes(model: nn.Module) -> int:
    """Return serialized tensor storage bytes for a model state dict."""

    if not isinstance(model, nn.Module):
        raise QuantizationError("model must be a torch.nn.Module.")
    return tensor_tree_nbytes(model.state_dict())


def tensor_tree_nbytes(value: Any) -> int:
    """Return total tensor storage bytes for nested state-dict-like values."""

    if isinstance(value, torch.Tensor):
        return _tensor_nbytes(value)
    if isinstance(value, Mapping):
        return sum(tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in value)
    return 0


def file_sha256(path: Path | str) -> str:
    """Return the SHA256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_load_target(model: nn.Module, method: str) -> nn.Module:
    if method == "fp16":
        model.to(dtype=torch.float16)
        return model
    if method == "bf16":
        model.to(dtype=torch.bfloat16)
        return model
    cpu_model = model.cpu().eval()
    return quantize_dynamic_int8(cpu_model)


def _method_dtype_name(method: str) -> str:
    if method == "fp16":
        return "float16"
    if method == "bf16":
        return "bfloat16"
    return "qint8_dynamic"


def _dynamic_int8_engine() -> str | None:
    supported = tuple(getattr(torch.backends.quantized, "supported_engines", ()))
    current = getattr(torch.backends.quantized, "engine", "none")
    if current and current != "none":
        return str(current)
    for candidate in ("x86", "fbgemm", "qnnpack"):
        if candidate in supported:
            return candidate
    return None


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    if tensor.is_quantized:
        return tensor.numel() * tensor.element_size()
    return tensor.numel() * tensor.element_size()


def _cpu_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tensor_tree(item) for item in value)
    return value


def _torch_load(checkpoint_path: Path, map_location: str | torch.device) -> Mapping[str, Any]:
    try:
        loaded = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(loaded, Mapping):
        raise QuantizationError(
            f"Quantized checkpoint payload must be a mapping: {checkpoint_path}"
        )
    return loaded


def _validate_quantized_payload(payload: Mapping[str, Any], checkpoint_path: Path) -> None:
    required = {"format_version", "quantization", "model_state_dict"}
    missing = required - payload.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise QuantizationError(f"Quantized checkpoint {checkpoint_path} is missing: {names}.")
    if payload["format_version"] != QUANTIZATION_FORMAT_VERSION:
        raise QuantizationError(
            f"Unsupported quantized checkpoint version: {payload['format_version']}"
        )
    if not isinstance(payload["quantization"], Mapping):
        raise QuantizationError("quantization metadata must be a mapping.")


__all__ = [
    "BackendCapabilities",
    "LoadedQuantizedCheckpoint",
    "QUANTIZATION_FORMAT_VERSION",
    "QuantizationError",
    "QuantizedCheckpointInfo",
    "QuantizedInferenceModel",
    "SUPPORTED_QUANTIZATION_METHODS",
    "convert_model_precision",
    "detect_backend_capabilities",
    "file_sha256",
    "is_quantization_supported",
    "load_quantized_checkpoint",
    "model_state_nbytes",
    "normalize_quantization_method",
    "quantize_bf16",
    "quantize_dynamic_int8",
    "quantize_fp16",
    "quantize_model",
    "save_quantized_checkpoint",
    "tensor_tree_nbytes",
]
