"""General utilities used across GenPy LLM."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def get_project_root() -> Path:
    """Return the repository root from this source file."""

    return Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directories(paths: list[Path] | tuple[Path, ...]) -> None:
    """Create required directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch module."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
