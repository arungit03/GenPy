"""Configuration loading for the GenPy offline API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_CONFIG_PATH = PROJECT_ROOT / "configs" / "api.yaml"


class APIGenerationConfig(BaseModel):
    """Default generation settings for API requests."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.7, gt=0.0, le=5.0)
    top_p: Optional[float] = Field(default=0.95, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    min_new_tokens: int = Field(default=0, ge=0, le=4096)
    do_sample: bool = True
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=10.0)
    stop_tokens: tuple[str, ...] = ("<eos>",)


class APIConfig(BaseModel):
    """Runtime configuration for local model serving."""

    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    phase7_config: Path = Path("configs/finetuning.yaml")
    checkpoint: Path = Path("checkpoints/fine_tuned/best_checkpoint.pt")
    quantized_checkpoint: Optional[Path] = None
    lora_adapter: Optional[Path] = None
    generation: APIGenerationConfig = Field(default_factory=APIGenerationConfig)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Validate supported device names."""

        normalized = value.strip().lower()
        if normalized not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        return normalized

    def resolve_path(self, path: Path | None) -> Path | None:
        """Resolve a config path relative to the project root."""

        if path is None:
            return None
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_api_config(path: Path | str = DEFAULT_API_CONFIG_PATH) -> APIConfig:
    """Load and validate API YAML configuration."""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"API config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("API config must be a YAML mapping.")
    return APIConfig.model_validate(_normalize_empty_paths(payload))


def detect_api_device(requested_device: str) -> torch.device:
    """Select the API device, preferring Apple MPS before CUDA for auto mode."""

    requested = requested_device.strip().lower()
    if requested == "auto":
        if _mps_is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not _mps_is_available():
        raise RuntimeError("MPS was requested, but it is not available on this system.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available on this system.")
    if requested != "cpu":
        APIConfig(device=requested)
    return torch.device(requested)


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _normalize_empty_paths(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for key in ("checkpoint", "quantized_checkpoint", "lora_adapter", "phase7_config"):
        if normalized.get(key) == "":
            normalized[key] = None
    return normalized
