"""Phase 6.3 training metrics, reports, and early stopping."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping settings."""

    enabled: bool = True
    patience: int = 3
    min_delta: float = 0.0
    monitor: str = "validation_loss"
    mode: str = "min"


@dataclass
class EarlyStoppingState:
    """Mutable early-stopping state."""

    config: EarlyStoppingConfig
    best_metric: float | None = None
    bad_epochs: int = 0

    def update(self, metric: float | None) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` after observing a metric."""

        if not self.config.enabled or metric is None:
            return False, False
        improved = self._improved(metric)
        if improved:
            self.best_metric = metric
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.config.patience

    def _improved(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.config.mode == "min":
            return metric < self.best_metric - self.config.min_delta
        return metric > self.best_metric + self.config.min_delta


@dataclass
class TrainingMonitor:
    """Write Phase 6.3 logs and summaries."""

    report_dir: Path
    csv_path: Path = field(init=False)
    json_path: Path = field(init=False)
    curves_path: Path = field(init=False)
    summary_path: Path = field(init=False)
    checkpoint_history_path: Path = field(init=False)
    records: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.report_dir / "training_log.csv"
        self.json_path = self.report_dir / "training_log.json"
        self.curves_path = self.report_dir / "training_curves.json"
        self.summary_path = self.report_dir / "summary.md"
        self.checkpoint_history_path = self.report_dir / "checkpoint_history.json"

    def log(self, record: dict[str, Any]) -> None:
        """Append one metric record and update CSV/JSON logs."""

        payload = {"timestamp": _timestamp(), **record}
        self.records.append(payload)
        self._write_csv()
        self.json_path.write_text(
            json.dumps(self.records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.curves_path.write_text(
            json.dumps(_curves(self.records), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def log_checkpoint(self, record: dict[str, Any]) -> None:
        """Append checkpoint history."""

        payload = {"timestamp": _timestamp(), **record}
        self.checkpoint_history.append(payload)
        self.checkpoint_history_path.write_text(
            json.dumps(self.checkpoint_history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_summary(
        self,
        *,
        status: str,
        source_checkpoint: Path,
        last_checkpoint: Path | None,
        best_checkpoint: Path | None,
        global_step: int,
        best_metric: float | None,
        reason: str | None = None,
    ) -> None:
        """Write the Markdown training summary."""

        last_record = self.records[-1] if self.records else {}
        lines = [
            "# GenPy Phase 6.3 Continued Pretraining Summary",
            "",
            f"- Status: {status}",
            f"- Reason: {reason or 'none'}",
            f"- Source checkpoint: `{source_checkpoint}`",
            f"- Last checkpoint: `{last_checkpoint}`",
            f"- Best checkpoint: `{best_checkpoint}`",
            f"- Global step: {global_step}",
            f"- Best metric: {_fmt(best_metric)}",
            f"- Last training loss: {_fmt(last_record.get('training_loss'))}",
            f"- Last validation loss: {_fmt(last_record.get('validation_loss'))}",
            f"- Last perplexity: {_fmt(last_record.get('perplexity'))}",
            f"- Tokens processed: {int(last_record.get('tokens_processed') or 0):,}",
            f"- Tokens/sec: {_fmt(last_record.get('tokens_per_second'))}",
            f"- Checkpoints saved: {len(self.checkpoint_history)}",
            "",
            "This phase resumes from an existing Phase 6 checkpoint and never starts "
            "from random initialization.",
        ]
        self.summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_csv(self) -> None:
        fields = sorted({key for record in self.records for key in record})
        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: record.get(key) for key in fields})


def _curves(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "step": [record.get("step") for record in records],
        "training_loss": [record.get("training_loss") for record in records],
        "validation_loss": [record.get("validation_loss") for record in records],
        "perplexity": [record.get("perplexity") for record in records],
        "learning_rate": [record.get("learning_rate") for record in records],
        "tokens_per_second": [record.get("tokens_per_second") for record in records],
        "gradient_norm": [record.get("gradient_norm") for record in records],
    }


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "EarlyStoppingConfig",
    "EarlyStoppingState",
    "TrainingMonitor",
]
