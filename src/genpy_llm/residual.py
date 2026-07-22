"""Residual connection helpers for GenPy LLM."""

from __future__ import annotations

import torch
from torch import nn

from genpy_llm.normalization import GPTLayerNorm, NormalizationError


class ResidualError(ValueError):
    """Raised when residual connection inputs are invalid."""


class ResidualConnection(nn.Module):
    """Add a residual path to a sublayer output."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        if (
            not isinstance(dropout, int | float)
            or isinstance(dropout, bool)
            or not 0.0 <= dropout < 1.0
        ):
            raise ResidualError("dropout must be at least 0.0 and less than 1.0.")
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        residual: torch.Tensor,
        sublayer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Return residual plus dropout-applied sublayer output."""

        _validate_residual_inputs(residual, sublayer_output)
        return residual + self.dropout(sublayer_output)


class PreNormResidual(nn.Module):
    """Apply LayerNorm before a sublayer, then add a residual connection."""

    def __init__(
        self,
        embedding_dim: int,
        sublayer: nn.Module,
        normalization_epsilon: float = 1e-5,
        residual_dropout: float = 0.1,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(sublayer, nn.Module):
            raise ResidualError("sublayer must be a torch.nn.Module.")
        try:
            self.layer_norm = GPTLayerNorm(
                embedding_dim=embedding_dim,
                epsilon=normalization_epsilon,
                elementwise_affine=elementwise_affine,
            )
        except NormalizationError as exc:
            raise ResidualError(str(exc)) from exc
        self.sublayer = sublayer
        self.residual = ResidualConnection(dropout=residual_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Normalize, call the wrapped sublayer, and add the residual path."""

        hidden_states = cast_residual_stream_for_autocast(hidden_states)
        normalized = self.layer_norm(hidden_states)
        sublayer_output = self.sublayer(normalized, *args, **kwargs)
        if isinstance(sublayer_output, tuple):
            raise ResidualError(
                "PreNormResidual expects the wrapped sublayer to return a tensor, not a tuple. "
                "Call attention sublayers with return_attention=False."
            )
        if not isinstance(sublayer_output, torch.Tensor):
            raise ResidualError("Wrapped sublayer must return a torch.Tensor.")
        return self.residual(hidden_states, sublayer_output)


def cast_residual_stream_for_autocast(hidden_states: torch.Tensor) -> torch.Tensor:
    """Use the active AMP dtype for float32 residual streams."""

    if not isinstance(hidden_states, torch.Tensor):
        return hidden_states
    if hidden_states.dtype != torch.float32:
        return hidden_states
    device_type = hidden_states.device.type
    if not _is_autocast_enabled(device_type):
        return hidden_states
    dtype = _get_autocast_dtype(device_type)
    if dtype not in {torch.float16, torch.bfloat16}:
        return hidden_states
    return hidden_states.to(dtype=dtype)


def _validate_residual_inputs(residual: torch.Tensor, sublayer_output: torch.Tensor) -> None:
    if not isinstance(residual, torch.Tensor):
        raise ResidualError("residual must be a torch.Tensor.")
    if not isinstance(sublayer_output, torch.Tensor):
        raise ResidualError("sublayer_output must be a torch.Tensor.")
    if not residual.is_floating_point() or not sublayer_output.is_floating_point():
        raise ResidualError("residual and sublayer_output must be floating-point tensors.")
    if residual.shape != sublayer_output.shape:
        raise ResidualError(
            "residual and sublayer_output shapes must match exactly. "
            f"Received {tuple(residual.shape)} and {tuple(sublayer_output.shape)}."
        )
    if residual.dtype != sublayer_output.dtype:
        raise ResidualError("residual and sublayer_output dtypes must match.")
    if residual.device != sublayer_output.device:
        raise ResidualError("residual and sublayer_output devices must match.")


def _is_autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        if device_type == "cuda":
            return bool(torch.is_autocast_enabled())
        if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            return bool(torch.is_autocast_cpu_enabled())
        return False


def _get_autocast_dtype(device_type: str) -> torch.dtype | None:
    getter = getattr(torch, "get_autocast_dtype", None)
    if getter is not None:
        try:
            return getter(device_type)
        except TypeError:
            pass
    if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
        return torch.get_autocast_gpu_dtype()
    if device_type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
        return torch.get_autocast_cpu_dtype()
    return None


__all__ = [
    "PreNormResidual",
    "ResidualConnection",
    "ResidualError",
    "cast_residual_stream_for_autocast",
]
