"""Weight-parametrized LoRA adapters for GenPy attention projections."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

LORA_FORMAT_VERSION = 1
DEFAULT_LORA_TARGETS = (
    "attention.qkv_projection",
    "attention.output_projection",
)


class LoRAError(RuntimeError):
    """Raised when LoRA adapters cannot be configured or restored safely."""


@dataclass(frozen=True)
class LoRAAdapterInfo:
    """Dimensions and hyperparameters for one attached adapter."""

    module_name: str
    in_features: int
    out_features: int
    rank: int
    alpha: float
    dropout: float
    parameters: int
    merged: bool


@dataclass(frozen=True)
class LoRAStats:
    """Model-level LoRA parameter summary."""

    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    adapter_parameters: int
    adapter_count: int
    adapters: tuple[LoRAAdapterInfo, ...]

    @property
    def trainable_percentage(self) -> float:
        """Return the percentage of parameters trained by LoRA."""

        if self.total_parameters == 0:
            return 0.0
        return 100.0 * self.trainable_parameters / self.total_parameters


@dataclass(frozen=True)
class LoadedLoRAAdapters:
    """Metadata returned after loading an adapter-only checkpoint."""

    path: Path
    adapter_count: int
    metadata: Mapping[str, Any]
    merged: bool


class LoRAWeightParametrization(nn.Module):
    """Expose ``base_weight + scaled(B @ A)`` through a parameterized weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        _validate_adapter_values(rank, alpha, dropout)
        if in_features <= 0 or out_features <= 0:
            raise LoRAError("LoRA projection dimensions must be greater than zero.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer("merged", torch.tensor(False), persistent=False)
        self.register_buffer(
            "merged_delta",
            torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype),
            persistent=False,
        )

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Return the effective weight consumed by ``F.linear``."""

        if bool(self.merged.item()):
            return base_weight
        return base_weight + self.delta_weight(apply_dropout=self.training)

    def delta_weight(self, *, apply_dropout: bool = False) -> torch.Tensor:
        """Return the scaled low-rank weight update."""

        lora_A = F.dropout(
            self.lora_A,
            p=self.dropout_p,
            training=apply_dropout and self.dropout_p > 0,
        )
        return (self.lora_B @ lora_A) * self.scaling

    @torch.no_grad()
    def merge_into(self, original_weight: nn.Parameter) -> None:
        """Materialize the adapter update into the frozen original weight."""

        if bool(self.merged.item()):
            return
        delta = self.delta_weight(apply_dropout=False).to(
            device=original_weight.device,
            dtype=original_weight.dtype,
        )
        original_weight.add_(delta)
        self.merged_delta.copy_(delta)
        self.merged.fill_(True)

    @torch.no_grad()
    def unmerge_from(self, original_weight: nn.Parameter) -> None:
        """Remove the exact update previously merged into the original weight."""

        if not bool(self.merged.item()):
            return
        original_weight.sub_(
            self.merged_delta.to(device=original_weight.device, dtype=original_weight.dtype)
        )
        self.merged_delta.zero_()
        self.merged.fill_(False)


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
) -> LoRAStats:
    """Freeze a model and attach LoRA to every matching attention projection."""

    if not isinstance(model, nn.Module):
        raise LoRAError("model must be a torch.nn.Module.")
    _validate_adapter_values(rank, alpha, dropout)
    targets = _validate_target_modules(target_modules)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(target) for target in targets):
            matches.append((name, module))
    if not matches:
        names = ", ".join(targets)
        raise LoRAError(f"No nn.Linear modules matched LoRA targets: {names}")
    for name, layer in matches:
        _validate_projection_shape(name, layer)
        if parametrize.is_parametrized(layer, "weight"):
            if _adapter_for_layer(layer) is not None:
                raise LoRAError(f"Linear layer already has a GenPy LoRA adapter: {name}")
            raise LoRAError(f"Linear weight already has a non-LoRA parametrization: {name}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for _name, layer in matches:
        _register_adapter(
            layer,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    return lora_stats(model)


def lora_stats(model: nn.Module) -> LoRAStats:
    """Return adapter and trainable-parameter statistics."""

    adapters = tuple(
        LoRAAdapterInfo(
            module_name=name,
            in_features=adapter.in_features,
            out_features=adapter.out_features,
            rank=adapter.rank,
            alpha=adapter.alpha,
            dropout=adapter.dropout_p,
            parameters=adapter.lora_A.numel() + adapter.lora_B.numel(),
            merged=bool(adapter.merged.item()),
        )
        for name, _layer, adapter in iter_lora_adapters(model)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    adapter_parameters = sum(item.parameters for item in adapters)
    return LoRAStats(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
        adapter_parameters=adapter_parameters,
        adapter_count=len(adapters),
        adapters=adapters,
    )


def iter_lora_adapters(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Linear, LoRAWeightParametrization]]:
    """Yield module paths, Linear layers, and their GenPy LoRA parametrizations."""

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not parametrize.is_parametrized(module, "weight"):
            continue
        parametrizations = module.parametrizations.weight
        adapters = [
            item for item in parametrizations if isinstance(item, LoRAWeightParametrization)
        ]
        if len(adapters) > 1:
            raise LoRAError(f"Multiple GenPy LoRA adapters are registered on {name}.")
        if adapters:
            yield name, module, adapters[0]


def merge_lora_weights(model: nn.Module) -> LoRAStats:
    """Merge all attached adapters into their frozen original weights."""

    found = False
    for _name, layer, adapter in iter_lora_adapters(model):
        adapter.merge_into(layer.parametrizations.weight.original)
        found = True
    if not found:
        raise LoRAError("Model has no LoRA adapters to merge.")
    return lora_stats(model)


def unmerge_lora_weights(model: nn.Module) -> LoRAStats:
    """Restore all attached adapters to unmerged parameterized form."""

    found = False
    for _name, layer, adapter in iter_lora_adapters(model):
        adapter.unmerge_from(layer.parametrizations.weight.original)
        found = True
    if not found:
        raise LoRAError("Model has no LoRA adapters to unmerge.")
    return lora_stats(model)


def save_lora_adapters(
    model: nn.Module,
    path: Path | str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save only LoRA tensors and reconstruction metadata."""

    adapters: dict[str, dict[str, Any]] = {}
    for name, _layer, adapter in iter_lora_adapters(model):
        adapters[name] = {
            "in_features": adapter.in_features,
            "out_features": adapter.out_features,
            "rank": adapter.rank,
            "alpha": adapter.alpha,
            "dropout": adapter.dropout_p,
            "lora_A": adapter.lora_A.detach().cpu().clone(),
            "lora_B": adapter.lora_B.detach().cpu().clone(),
        }
    if not adapters:
        raise LoRAError("Model has no LoRA adapters to save.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": LORA_FORMAT_VERSION,
        "adapter_type": "lora_weight_parametrization",
        "metadata": _safe_metadata(metadata or {}),
        "adapters": adapters,
    }
    partial = Path(f"{output_path}.partial")
    try:
        torch.save(payload, partial)
        partial.replace(output_path)
    finally:
        partial.unlink(missing_ok=True)
    return output_path.resolve()


def load_lora_adapters(
    model: nn.Module,
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    merge: bool = False,
) -> LoadedLoRAAdapters:
    """Load adapter-only tensors, registering weight parametrizations as needed."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"LoRA adapter checkpoint not found: {input_path}")
    payload = _torch_load(input_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise LoRAError("LoRA adapter checkpoint must contain a mapping.")
    if payload.get("format_version") != LORA_FORMAT_VERSION:
        raise LoRAError("Unsupported LoRA adapter checkpoint format version.")
    raw_adapters = payload.get("adapters")
    if not isinstance(raw_adapters, Mapping) or not raw_adapters:
        raise LoRAError("LoRA adapter checkpoint contains no adapters.")

    existing = {name for name, _layer, _adapter in iter_lora_adapters(model)}
    requested = {str(name) for name in raw_adapters}
    if existing and strict and existing != requested:
        missing = sorted(requested - existing)
        unexpected = sorted(existing - requested)
        raise LoRAError(f"LoRA target mismatch; missing={missing}, unexpected={unexpected}.")
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module_name, raw in raw_adapters.items():
        if not isinstance(module_name, str) or not isinstance(raw, Mapping):
            raise LoRAError("Invalid LoRA adapter entry.")
        try:
            layer = model.get_submodule(module_name)
        except AttributeError as exc:
            if strict:
                raise LoRAError(f"LoRA target module not found: {module_name}") from exc
            continue
        if not isinstance(layer, nn.Linear):
            raise LoRAError(f"LoRA target is not nn.Linear: {module_name}")
        rank = _checkpoint_int(raw, "rank", module_name)
        alpha = _checkpoint_float(raw, "alpha", module_name)
        dropout = _checkpoint_float(raw, "dropout", module_name)
        _validate_checkpoint_dimensions(layer, raw, module_name)
        adapter = _adapter_for_layer(layer)
        if adapter is None:
            adapter = _register_adapter(
                layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        elif (
            adapter.rank != rank
            or adapter.alpha != alpha
            or adapter.dropout_p != dropout
        ):
            raise LoRAError(f"LoRA hyperparameters do not match for {module_name}.")
        if bool(adapter.merged.item()):
            adapter.unmerge_from(layer.parametrizations.weight.original)
        _copy_adapter_tensor(adapter.lora_A, raw.get("lora_A"), module_name, "lora_A")
        _copy_adapter_tensor(adapter.lora_B, raw.get("lora_B"), module_name, "lora_B")
        adapter.lora_A.requires_grad = True
        adapter.lora_B.requires_grad = True

    loaded_names = {name for name, _layer, _adapter in iter_lora_adapters(model)}
    if strict and loaded_names != requested:
        raise LoRAError("Not all LoRA adapters could be restored.")
    if merge:
        merge_lora_weights(model)
    raw_metadata = payload.get("metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    return LoadedLoRAAdapters(
        path=input_path.resolve(),
        adapter_count=len(requested),
        metadata=metadata,
        merged=merge,
    )


def _register_adapter(
    layer: nn.Linear,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRAWeightParametrization:
    if parametrize.is_parametrized(layer, "weight"):
        existing = _adapter_for_layer(layer)
        if existing is not None:
            raise LoRAError("Linear layer already has a GenPy LoRA adapter.")
        raise LoRAError("Linear weight already has a non-LoRA parametrization.")
    adapter = LoRAWeightParametrization(
        layer.in_features,
        layer.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    parametrize.register_parametrization(layer, "weight", adapter)
    layer.parametrizations.weight.original.requires_grad = False
    if layer.bias is not None:
        layer.bias.requires_grad = False
    adapter.lora_A.requires_grad = True
    adapter.lora_B.requires_grad = True
    return adapter


def _adapter_for_layer(layer: nn.Linear) -> LoRAWeightParametrization | None:
    if not parametrize.is_parametrized(layer, "weight"):
        return None
    matches = [
        item
        for item in layer.parametrizations.weight
        if isinstance(item, LoRAWeightParametrization)
    ]
    if len(matches) > 1:
        raise LoRAError("Linear layer has multiple GenPy LoRA adapters.")
    return matches[0] if matches else None


def _validate_projection_shape(name: str, layer: nn.Linear) -> None:
    if name.endswith("attention.qkv_projection"):
        if layer.out_features != 3 * layer.in_features:
            raise LoRAError(f"Fused QKV projection has invalid dimensions: {name}")
    elif name.endswith("attention.output_projection"):
        if layer.out_features != layer.in_features:
            raise LoRAError(f"Attention output projection has invalid dimensions: {name}")


def _validate_adapter_values(rank: int, alpha: float, dropout: float) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise LoRAError("LoRA rank must be an integer greater than zero.")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha <= 0:
        raise LoRAError("LoRA alpha must be greater than zero.")
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0.0 <= dropout < 1.0
    ):
        raise LoRAError("LoRA dropout must be at least zero and less than one.")


def _validate_target_modules(target_modules: Sequence[str]) -> tuple[str, ...]:
    if isinstance(target_modules, str) or not target_modules:
        raise LoRAError("target_modules must be a non-empty sequence of module suffixes.")
    values = tuple(str(item).strip() for item in target_modules)
    if any(not item for item in values):
        raise LoRAError("LoRA target module suffixes must not be empty.")
    supported = set(DEFAULT_LORA_TARGETS)
    unsupported = sorted(set(values) - supported)
    if unsupported:
        raise LoRAError(f"Unsupported LoRA targets: {', '.join(unsupported)}")
    return values


def _validate_checkpoint_dimensions(
    layer: nn.Linear,
    raw: Mapping[str, Any],
    module_name: str,
) -> None:
    in_features = _checkpoint_int(raw, "in_features", module_name)
    out_features = _checkpoint_int(raw, "out_features", module_name)
    if (layer.in_features, layer.out_features) != (in_features, out_features):
        raise LoRAError(
            f"LoRA dimensions do not match {module_name}: checkpoint "
            f"{in_features}->{out_features}, model {layer.in_features}->{layer.out_features}."
        )


def _checkpoint_int(raw: Mapping[str, Any], key: str, module_name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LoRAError(f"Invalid {key} for LoRA target {module_name}.")
    return value


def _checkpoint_float(raw: Mapping[str, Any], key: str, module_name: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LoRAError(f"Invalid {key} for LoRA target {module_name}.")
    return float(value)


@torch.no_grad()
def _copy_adapter_tensor(
    destination: nn.Parameter,
    source: Any,
    module_name: str,
    tensor_name: str,
) -> None:
    if not isinstance(source, torch.Tensor) or source.shape != destination.shape:
        raise LoRAError(f"Invalid {tensor_name} tensor for LoRA target {module_name}.")
    destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def _safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(metadata), default=str))
    except (TypeError, ValueError) as exc:
        raise LoRAError(f"LoRA metadata is not serializable: {exc}") from exc


def _torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "LORA_FORMAT_VERSION",
    "LoadedLoRAAdapters",
    "LoRAAdapterInfo",
    "LoRAError",
    "LoRAStats",
    "LoRAWeightParametrization",
    "apply_lora",
    "iter_lora_adapters",
    "load_lora_adapters",
    "lora_stats",
    "merge_lora_weights",
    "save_lora_adapters",
    "unmerge_lora_weights",
]
