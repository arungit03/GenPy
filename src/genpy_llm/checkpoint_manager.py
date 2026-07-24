"""Phase 6.3 checkpoint resolution, validation, and saving."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.code_tokenizer import tokenizer_file_hash


class Phase63CheckpointError(RuntimeError):
    """Raised when Phase 6.3 checkpoint handling cannot continue."""


def resolve_latest_phase6_checkpoint(
    checkpoint: Path | str | None,
    *,
    project_root: Path,
    search_directory: Path,
) -> Path:
    """Resolve an explicit checkpoint or the latest Phase 6 checkpoint."""

    if checkpoint is not None:
        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Phase 6 checkpoint not found: {candidate}")
        return candidate.resolve()
    canonical = search_directory / "last_checkpoint.pt"
    if canonical.is_file():
        return canonical.resolve()
    candidates = list(search_directory.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No Phase 6 checkpoints found in {search_directory}")
    return max(candidates, key=_checkpoint_sort_key).resolve()


def validate_checkpoint_tokenizer(
    checkpoint_path: Path,
    *,
    tokenizer_path: Path,
) -> None:
    """Verify checkpoint vocabulary metadata does not conflict with the tokenizer."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise Phase63CheckpointError(f"Checkpoint is not a mapping: {checkpoint_path}")
    vocabulary = payload.get("vocabulary_metadata")
    if isinstance(vocabulary, dict):
        expected = tokenizer_file_hash(tokenizer_path)
        actual = vocabulary.get("tokenizer_sha256")
        if actual is not None and actual != expected:
            raise Phase63CheckpointError(
                "Checkpoint tokenizer hash does not match Phase 6.3 tokenizer."
            )


def save_phase63_checkpoint(
    *,
    output_dir: Path,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    scaler: object | None,
    model_config: dict[str, Any],
    tokenizer_path: Path,
    training_loss: float | None,
    validation_loss: float | None,
    best_metric: float | None,
    save_best: bool,
    extra_state: dict[str, Any],
) -> dict[str, Any]:
    """Save Phase 6.3 epoch, last, and optional best checkpoints."""

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = output_dir / f"epoch_{epoch:03d}.pt"
    tokenizer_hash = tokenizer_file_hash(tokenizer_path)
    metadata = save_checkpoint(
        epoch_path,
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
        vocabulary_metadata={
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": tokenizer_hash,
        },
        extra_state=extra_state,
    )
    last_path = output_dir / "last_checkpoint.pt"
    shutil.copy2(epoch_path, last_path)
    best_path = None
    if save_best:
        best_path = output_dir / "best_checkpoint.pt"
        shutil.copy2(epoch_path, best_path)
    return {
        "epoch": epoch,
        "global_step": global_step,
        "checkpoint": str(epoch_path),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path is not None else None,
        "checkpoint_size_bytes": epoch_path.stat().st_size,
        "metadata": asdict(metadata),
    }


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Load a checkpoint payload for diagnostics."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise Phase63CheckpointError(f"Checkpoint payload must be a mapping: {path}")
    return payload


def ensure_checkpoint_loads(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load a checkpoint into supplied objects to detect corruption."""

    load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=map_location,
        restore_rng=False,
    )


def append_checkpoint_history(path: Path, record: dict[str, Any]) -> None:
    """Append a checkpoint history record to a JSON array artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]]
    if path.is_file():
        history = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            history = []
    else:
        history = []
    history.append(record)
    path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return (int(stem.rsplit("_", 1)[-1]), path.name)
    except ValueError:
        return (-1, path.name)


__all__ = [
    "Phase63CheckpointError",
    "append_checkpoint_history",
    "ensure_checkpoint_loads",
    "load_checkpoint_payload",
    "resolve_latest_phase6_checkpoint",
    "save_phase63_checkpoint",
    "validate_checkpoint_tokenizer",
]
