"""CPU dynamic quantization helpers for GenPy LLM."""

from __future__ import annotations

import copy

import torch
from torch import nn


class QuantizationError(ValueError):
    """Raised when quantization is unsupported or invalid."""


class QuantizedInferenceModel(nn.Module):
    """Inference-only wrapper around a dynamically quantized model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.vocab_size = model.vocab_size
        self.context_length = model.context_length

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        if mode:
            raise QuantizationError("Quantized models are for inference only.")
        return super().train(False)


def quantize_dynamic_int8(
    model: nn.Module,
) -> nn.Module:
    """Return a CPU dynamic-INT8 quantized copy of a model."""

    if not isinstance(model, nn.Module):
        raise QuantizationError("model must be a torch.nn.Module.")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if device.type != "cpu":
        raise QuantizationError("dynamic_int8 quantization is supported only on CPU.")
    quantized_source = copy.deepcopy(model).cpu().eval()
    quantized = torch.quantization.quantize_dynamic(
        quantized_source,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    return QuantizedInferenceModel(quantized)


__all__ = ["QuantizationError", "QuantizedInferenceModel", "quantize_dynamic_int8"]
