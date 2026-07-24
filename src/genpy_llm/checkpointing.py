"""Checkpoint saving and loading helpers for GenPy LLM."""

from __future__ import annotations

import math
import os
import random
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from genpy_llm.performance import unwrap_compiled_model

CHECKPOINT_FORMAT_VERSION = 1
SUPPORTED_MONITORS = {"training_loss", "validation_loss"}
SUPPORTED_MODES = {"min", "max"}


class CheckpointError(RuntimeError):
    """Raised when checkpoint saving or loading cannot continue safely."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata stored alongside model and optimizer state."""

    format_version: int
    epoch: int
    global_step: int
    training_loss: float | None
    validation_loss: float | None
    best_metric: float | None
    vocabulary_size: int
    context_length: int
    model_parameter_count: int


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Summary returned after restoring a checkpoint."""

    epoch: int
    global_step: int
    best_metric: float | None
    training_loss: float | None
    validation_loss: float | None
    checkpoint_path: Path
    extra_state: Mapping[str, Any]


@dataclass(frozen=True)
class ManagedCheckpointResult:
    """Result of periodic checkpoint saving with optional best and rotation handling."""

    checkpoint_path: Path
    best_checkpoint_path: Path | None
    best_metric: float | None
    removed_paths: tuple[Path, ...]


def checkpoint_filename(filename_prefix: str, epoch: int, global_step: int) -> str:
    """Return the managed checkpoint filename for an epoch and global step."""

    _validate_filename_prefix(filename_prefix)
    _validate_non_negative_int("epoch", epoch)
    _validate_non_negative_int("global_step", global_step)
    return f"{filename_prefix}_epoch_{epoch:04d}_step_{global_step:08d}.pt"


def save_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    training_loss: float | None = None,
    validation_loss: float | None = None,
    best_metric: float | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    model_config: Mapping[str, Any] | None = None,
    vocabulary_metadata: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> CheckpointMetadata:
    """Atomically save model, optimizer, optional scheduler, and RNG state."""

    output_path = _prepare_output_path(checkpoint_path)
    _validate_save_inputs(model, optimizer, epoch, global_step)
    checkpoint_model = unwrap_compiled_model(model)
    training_loss = _validate_optional_metric("training_loss", training_loss)
    validation_loss = _validate_optional_metric("validation_loss", validation_loss)
    best_metric = _validate_optional_metric("best_metric", best_metric)
    metadata = CheckpointMetadata(
        format_version=CHECKPOINT_FORMAT_VERSION,
        epoch=epoch,
        global_step=global_step,
        training_loss=training_loss,
        validation_loss=validation_loss,
        best_metric=best_metric,
        vocabulary_size=_model_int_attr(checkpoint_model, "vocab_size"),
        context_length=_model_int_attr(checkpoint_model, "context_length"),
        model_parameter_count=sum(parameter.numel() for parameter in checkpoint_model.parameters()),
    )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": asdict(metadata),
        "model_state_dict": _cpu_tensor_tree(checkpoint_model.state_dict()),
        "optimizer_state_dict": _cpu_tensor_tree(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tensor_tree(_scheduler_state_dict(scheduler)),
        "scaler_state_dict": _cpu_tensor_tree(_scaler_state_dict(scaler)),
        "rng_state": _rng_state(),
        "model_config": dict(model_config or {}),
        "vocabulary_metadata": dict(vocabulary_metadata or {}),
        "extra_state": dict(extra_state or {}),
    }
    _atomic_torch_save(payload, output_path)
    return metadata


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: object | None = None,
    scaler: object | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> LoadedCheckpoint:
    """Load a checkpoint into model, optional optimizer/scheduler, and optional RNG state."""

    input_path = Path(checkpoint_path)
    _validate_input_path(input_path)
    checkpoint_model = unwrap_compiled_model(model)
    if not isinstance(checkpoint_model, nn.Module):
        raise CheckpointError("model must be a torch.nn.Module.")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a torch.optim.Optimizer or None.")

    payload = _torch_load(input_path, map_location=map_location)
    _validate_payload(payload, input_path)
    metadata = _metadata_from_payload(payload, input_path)
    try:
        checkpoint_model.load_state_dict(payload["model_state_dict"], strict=strict)
    except RuntimeError as exc:
        raise CheckpointError(f"Could not restore model state from {input_path}: {exc}") from exc
    if optimizer is not None:
        try:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except (RuntimeError, ValueError) as exc:
            message = f"Could not restore optimizer state from {input_path}: {exc}"
            raise CheckpointError(message) from exc
        _move_optimizer_state_to_model_device(optimizer, checkpoint_model)
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(payload["scaler_state_dict"])
        except AttributeError as exc:
            raise CheckpointError("scaler must provide a load_state_dict() method.") from exc
    if scheduler is not None:
        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler_state is None:
            raise CheckpointError("Checkpoint does not contain scheduler state.")
        try:
            scheduler.load_state_dict(scheduler_state)
        except AttributeError as exc:
            raise CheckpointError("scheduler must provide a load_state_dict() method.") from exc
    if restore_rng:
        _restore_rng_state(payload.get("rng_state", {}))

    return LoadedCheckpoint(
        epoch=metadata.epoch,
        global_step=metadata.global_step,
        best_metric=metadata.best_metric,
        training_loss=metadata.training_loss,
        validation_loss=metadata.validation_loss,
        checkpoint_path=input_path.resolve(),
        extra_state=payload.get("extra_state", {})
        if isinstance(payload.get("extra_state", {}), Mapping)
        else {},
    )


def find_latest_checkpoint(
    directory: Path | str,
    filename_prefix: str = "genpy",
) -> Path | None:
    """Return the latest managed checkpoint path by epoch, then global step."""

    checkpoint_dir = Path(directory)
    if not checkpoint_dir.exists():
        return None
    if not checkpoint_dir.is_dir():
        raise CheckpointError(f"Checkpoint directory is not a directory: {checkpoint_dir}")
    matches = _managed_checkpoints(checkpoint_dir, filename_prefix)
    if not matches:
        return None
    _epoch, _step, path = max(matches, key=lambda item: (item[0], item[1], item[2].name))
    return path


def rotate_checkpoints(
    directory: Path | str,
    *,
    keep_last: int,
    filename_prefix: str = "genpy",
) -> tuple[Path, ...]:
    """Remove older managed checkpoints and return the deleted paths."""

    _validate_positive_int("keep_last", keep_last)
    checkpoint_dir = Path(directory)
    if not checkpoint_dir.exists():
        return ()
    if not checkpoint_dir.is_dir():
        raise CheckpointError(f"Checkpoint directory is not a directory: {checkpoint_dir}")
    matches = sorted(
        _managed_checkpoints(checkpoint_dir, filename_prefix),
        key=lambda item: (item[0], item[1], item[2].name),
        reverse=True,
    )
    removed: list[Path] = []
    for _epoch, _step, path in matches[keep_last:]:
        path.unlink(missing_ok=True)
        removed.append(path)
    return tuple(removed)


def is_better_metric(metric: float, best_metric: float | None, mode: str) -> bool:
    """Return whether a metric improves on the current best value."""

    metric = _validate_optional_metric("metric", metric)
    if metric is None:
        return False
    if mode not in SUPPORTED_MODES:
        raise CheckpointError("mode must be 'min' or 'max'.")
    if best_metric is None:
        return True
    best_metric = _validate_optional_metric("best_metric", best_metric)
    return metric < best_metric if mode == "min" else metric > best_metric


def save_managed_checkpoint(
    directory: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    filename_prefix: str,
    epoch: int,
    global_step: int,
    training_loss: float | None,
    validation_loss: float | None,
    best_metric: float | None,
    keep_last: int,
    save_best: bool,
    monitor: str,
    mode: str,
    scheduler: object | None = None,
    scaler: object | None = None,
    model_config: Mapping[str, Any] | None = None,
    vocabulary_metadata: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> ManagedCheckpointResult:
    """Save a periodic checkpoint, rotate old files, and optionally update best."""

    _validate_positive_int("keep_last", keep_last)
    _validate_monitor_and_mode(monitor, mode)
    checkpoint_dir = Path(directory)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_filename(filename_prefix, epoch, global_step)
    current_metric = training_loss if monitor == "training_loss" else validation_loss
    next_best_metric = best_metric

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=epoch,
        global_step=global_step,
        training_loss=training_loss,
        validation_loss=validation_loss,
        best_metric=best_metric,
        scheduler=scheduler,
        scaler=scaler,
        model_config=model_config,
        vocabulary_metadata=vocabulary_metadata,
        extra_state=extra_state,
    )
    best_checkpoint_path = None
    if (
        save_best
        and current_metric is not None
        and is_better_metric(current_metric, best_metric, mode)
    ):
        next_best_metric = current_metric
        best_checkpoint_path = checkpoint_dir / f"{filename_prefix}_best.pt"
        save_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            training_loss=training_loss,
            validation_loss=validation_loss,
            best_metric=next_best_metric,
            scheduler=scheduler,
            scaler=scaler,
            model_config=model_config,
            vocabulary_metadata=vocabulary_metadata,
            extra_state=extra_state,
        )
    removed_paths = rotate_checkpoints(
        checkpoint_dir,
        keep_last=keep_last,
        filename_prefix=filename_prefix,
    )
    return ManagedCheckpointResult(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        best_metric=next_best_metric,
        removed_paths=removed_paths,
    )


def _prepare_output_path(checkpoint_path: Path | str) -> Path:
    output_path = Path(checkpoint_path)
    if output_path.name.strip() == "":
        raise CheckpointError("checkpoint_path must include a filename.")
    if output_path.exists() and output_path.is_dir():
        raise CheckpointError(f"Checkpoint path is a directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _validate_input_path(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint path is not a file: {checkpoint_path}")


def _validate_save_inputs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
) -> None:
    if not isinstance(model, nn.Module):
        raise CheckpointError("model must be a torch.nn.Module.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a torch.optim.Optimizer.")
    _validate_non_negative_int("epoch", epoch)
    _validate_non_negative_int("global_step", global_step)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CheckpointError(f"{name} must be an integer greater than zero.")


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CheckpointError(f"{name} must be a non-negative integer.")


def _validate_filename_prefix(filename_prefix: str) -> None:
    if not isinstance(filename_prefix, str) or filename_prefix.strip() == "":
        raise CheckpointError("filename_prefix must be a non-empty string.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename_prefix):
        raise CheckpointError(
            "filename_prefix may only contain letters, numbers, dots, dashes, and underscores."
        )


def _validate_monitor_and_mode(monitor: str, mode: str) -> None:
    if monitor not in SUPPORTED_MONITORS:
        raise CheckpointError("monitor must be 'training_loss' or 'validation_loss'.")
    if mode not in SUPPORTED_MODES:
        raise CheckpointError("mode must be 'min' or 'max'.")


def _validate_optional_metric(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CheckpointError(f"{name} must be a finite number or None.")
    number = float(value)
    if not math.isfinite(number):
        raise CheckpointError(f"{name} must be finite.")
    return number


def _model_int_attr(model: nn.Module, name: str) -> int:
    value = getattr(model, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CheckpointError(f"model must expose a positive integer {name} attribute.")
    return value


def _scheduler_state_dict(scheduler: object | None) -> object | None:
    if scheduler is None:
        return None
    try:
        return scheduler.state_dict()
    except AttributeError as exc:
        raise CheckpointError("scheduler must provide a state_dict() method.") from exc


def _scaler_state_dict(scaler: object | None) -> object | None:
    if scaler is None:
        return None
    try:
        return scaler.state_dict()
    except AttributeError as exc:
        raise CheckpointError("scaler must provide a state_dict() method.") from exc


def _cpu_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tensor_tree(item) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    if not isinstance(rng_state, Mapping):
        raise CheckpointError("Checkpoint RNG state must be a mapping.")
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])
    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _atomic_torch_save(payload: Mapping[str, Any], output_path: Path) -> None:
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        temp_path.replace(output_path)
    except (OSError, RuntimeError, ValueError) as exc:
        temp_path.unlink(missing_ok=True)
        raise CheckpointError(f"Could not save checkpoint to {output_path}: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def _torch_load(checkpoint_path: Path, map_location: str | torch.device) -> Mapping[str, Any]:
    try:
        loaded = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=map_location)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise CheckpointError(f"Checkpoint payload must be a mapping: {checkpoint_path}")
    return loaded


def _validate_payload(payload: Mapping[str, Any], checkpoint_path: Path) -> None:
    required = {
        "format_version",
        "metadata",
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
    }
    missing = required - payload.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise CheckpointError(f"Checkpoint {checkpoint_path} is missing key(s): {names}.")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(f"Unsupported checkpoint format version: {payload['format_version']}")


def _metadata_from_payload(
    payload: Mapping[str, Any],
    checkpoint_path: Path,
) -> CheckpointMetadata:
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise CheckpointError(f"Checkpoint metadata must be a mapping: {checkpoint_path}")
    try:
        return CheckpointMetadata(**raw_metadata)
    except TypeError as exc:
        raise CheckpointError(f"Checkpoint metadata is invalid: {checkpoint_path}") from exc


def _move_optimizer_state_to_model_device(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> None:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _managed_checkpoints(
    directory: Path,
    filename_prefix: str,
) -> list[tuple[int, int, Path]]:
    _validate_filename_prefix(filename_prefix)
    pattern = re.compile(
        rf"^{re.escape(filename_prefix)}_epoch_(?P<epoch>\d{{4,}})_step_(?P<step>\d{{8,}})\.pt$"
    )
    matches: list[tuple[int, int, Path]] = []
    for path in directory.glob(f"{filename_prefix}_epoch_*_step_*.pt"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        matches.append((int(match.group("epoch")), int(match.group("step")), path))
    return matches


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "CheckpointMetadata",
    "LoadedCheckpoint",
    "ManagedCheckpointResult",
    "checkpoint_filename",
    "find_latest_checkpoint",
    "is_better_metric",
    "load_checkpoint",
    "rotate_checkpoints",
    "save_checkpoint",
    "save_managed_checkpoint",
]
