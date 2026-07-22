from __future__ import annotations

import pytest
import torch

from genpy_llm.device import select_device


def test_auto_device_selection_returns_torch_device() -> None:
    device = select_device("auto")

    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}


def test_explicit_cpu_selection() -> None:
    device = select_device("cpu")

    assert device == torch.device("cpu")


def test_invalid_device_name() -> None:
    with pytest.raises(ValueError, match="Invalid device"):
        select_device("quantum")
