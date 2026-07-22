"""Device selection helpers for PyTorch."""

from __future__ import annotations

import torch

VALID_DEVICE_OPTIONS = {"auto", "cpu", "cuda", "mps"}


def select_device(requested_device: str = "auto") -> torch.device:
    """Return a torch.device based on availability and user preference."""

    normalized = requested_device.strip().lower()
    if normalized not in VALID_DEVICE_OPTIONS:
        options = ", ".join(sorted(VALID_DEVICE_OPTIONS))
        raise ValueError(f"Invalid device '{requested_device}'. Expected one of: {options}.")

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available on this system.")

    if normalized == "mps" and not _mps_is_available():
        raise RuntimeError("MPS was requested, but it is not available on this system.")

    return torch.device(normalized)


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
