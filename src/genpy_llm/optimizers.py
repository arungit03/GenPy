"""Optimizer creation helpers for GenPy LLM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from genpy_llm.config import OptimizerConfig


class OptimizerError(ValueError):
    """Raised when an optimizer cannot be created safely."""


@dataclass(frozen=True)
class OptimizerMetadata:
    """Summary of optimizer settings and parameter groups."""

    optimizer_type: str
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    decayed_parameter_count: int
    non_decayed_parameter_count: int
    trainable_tensor_count: int


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer for trainable model parameters."""

    optimizer, _metadata = create_optimizer_with_metadata(model, config)
    return optimizer


def create_optimizer_with_metadata(
    model: nn.Module,
    config: OptimizerConfig,
) -> tuple[torch.optim.Optimizer, OptimizerMetadata]:
    """Create an AdamW optimizer and return grouping metadata."""

    _validate_optimizer_config(config)
    if not isinstance(model, nn.Module):
        raise OptimizerError("model must be a torch.nn.Module.")
    grouped = _group_parameters(model, separate_weight_decay=config.separate_weight_decay)
    if not grouped.decayed and not grouped.non_decayed:
        raise OptimizerError("model has no trainable parameters.")

    if config.separate_weight_decay:
        parameter_groups = [
            {"params": grouped.decayed, "weight_decay": float(config.weight_decay)},
            {"params": grouped.non_decayed, "weight_decay": 0.0},
        ]
    else:
        parameter_groups = [
            {
                "params": grouped.decayed + grouped.non_decayed,
                "weight_decay": float(config.weight_decay),
            }
        ]

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(config.learning_rate),
        betas=(float(config.beta1), float(config.beta2)),
        eps=float(config.epsilon),
    )
    metadata = OptimizerMetadata(
        optimizer_type=config.type,
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        beta1=float(config.beta1),
        beta2=float(config.beta2),
        epsilon=float(config.epsilon),
        decayed_parameter_count=sum(parameter.numel() for parameter in grouped.decayed),
        non_decayed_parameter_count=sum(parameter.numel() for parameter in grouped.non_decayed),
        trainable_tensor_count=len(grouped.decayed) + len(grouped.non_decayed),
    )
    return optimizer, metadata


@dataclass(frozen=True)
class _GroupedParameters:
    decayed: list[nn.Parameter]
    non_decayed: list[nn.Parameter]


def _group_parameters(model: nn.Module, separate_weight_decay: bool) -> _GroupedParameters:
    parameter_by_id: dict[int, nn.Parameter] = {}
    should_decay_by_id: dict[int, bool] = {}
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            parameter_by_id[parameter_id] = parameter
            decay_this_owner = _owner_should_decay(module, name, parameter)
            should_decay_by_id[parameter_id] = should_decay_by_id.get(parameter_id, False) or (
                separate_weight_decay and decay_this_owner
            )

    decayed: list[nn.Parameter] = []
    non_decayed: list[nn.Parameter] = []
    for parameter_id, parameter in parameter_by_id.items():
        if should_decay_by_id.get(parameter_id, False):
            decayed.append(parameter)
        else:
            non_decayed.append(parameter)
    return _GroupedParameters(decayed=decayed, non_decayed=non_decayed)


def _owner_should_decay(module: nn.Module, name: str, parameter: nn.Parameter) -> bool:
    if name.endswith("bias"):
        return False
    if parameter.ndim <= 1:
        return False
    return isinstance(module, nn.Linear) and name == "weight"


def _validate_optimizer_config(config: OptimizerConfig) -> None:
    if config.type != "adamw":
        raise OptimizerError("Only adamw optimizer is supported.")
    if config.learning_rate <= 0:
        raise OptimizerError("learning_rate must be greater than zero.")
    if config.weight_decay < 0:
        raise OptimizerError("weight_decay must be greater than or equal to zero.")
    if not 0 <= config.beta1 < 1:
        raise OptimizerError("beta1 must be at least 0.0 and less than 1.0.")
    if not 0 <= config.beta2 < 1:
        raise OptimizerError("beta2 must be at least 0.0 and less than 1.0.")
    if config.epsilon <= 0:
        raise OptimizerError("epsilon must be greater than zero.")
    if not isinstance(config.separate_weight_decay, bool):
        raise OptimizerError("separate_weight_decay must be true or false.")


__all__ = [
    "OptimizerError",
    "OptimizerMetadata",
    "create_optimizer",
    "create_optimizer_with_metadata",
]
