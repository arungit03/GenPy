from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from genpy_llm.checkpointing import save_checkpoint
from genpy_llm.code_evaluation import (
    TrainingMetricsRow,
    append_training_metrics_csv,
    build_loss_history,
    discover_code_checkpoints,
    loss_history_from_training_metrics,
    perplexity_from_loss,
    read_training_metrics_csv,
    resolve_code_checkpoint,
    write_loss_curve_png,
    write_loss_history_csv,
)


class TinyCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = 8
        self.context_length = 4
        self.embedding = nn.Embedding(self.vocab_size, 4)
        self.output = nn.Linear(4, self.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        return self.output(self.embedding(input_ids))


def test_checkpoint_summary_resolves_latest_best_and_loss_artifacts(tmp_path: Path) -> None:
    _save(tmp_path / "genpy_code_step_00000010.pt", global_step=10, validation_loss=2.0)
    _save(tmp_path / "genpy_code_step_00000020.pt", global_step=20, validation_loss=1.8)
    _save(tmp_path / "genpy_code_best.pt", global_step=20, validation_loss=1.7)

    summary = discover_code_checkpoints(
        tmp_path,
        filename_prefix="genpy_code",
        best_filename="genpy_code_best.pt",
    )
    rows = build_loss_history(summary)
    csv_path = tmp_path / "loss_history.csv"
    png_path = tmp_path / "loss_curve.png"

    write_loss_history_csv(rows, csv_path)
    write_loss_curve_png(rows, png_path)

    assert summary.total_checkpoints == 3
    assert summary.latest_checkpoint is not None
    assert summary.latest_checkpoint.path.name == "genpy_code_step_00000020.pt"
    assert summary.best_checkpoint is not None
    assert summary.best_checkpoint.path.name == "genpy_code_best.pt"
    assert resolve_code_checkpoint(
        "latest",
        checkpoint_directory=tmp_path,
        filename_prefix="genpy_code",
        best_filename="genpy_code_best.pt",
        project_root=tmp_path,
    ).name == "genpy_code_step_00000020.pt"
    assert resolve_code_checkpoint(
        "best",
        checkpoint_directory=tmp_path,
        filename_prefix="genpy_code",
        best_filename="genpy_code_best.pt",
        project_root=tmp_path,
    ).name == "genpy_code_best.pt"
    assert csv_path.read_text(encoding="utf-8").startswith("global_step,training_loss")
    assert png_path.read_bytes().startswith(b"\x89PNG")
    assert perplexity_from_loss(0.0) == pytest.approx(1.0)


def test_training_metrics_csv_round_trip_and_curve(tmp_path: Path) -> None:
    metrics_path = tmp_path / "training_metrics.csv"
    append_training_metrics_csv(
        TrainingMetricsRow(
            global_step=1,
            training_loss=2.5,
            validation_loss=2.25,
            perplexity=9.49,
            learning_rate=0.001,
            gradient_norm=0.75,
            tokens_per_second=123.0,
            tokens_processed=512,
            elapsed_seconds=4.0,
            eta_seconds=40.0,
        ),
        metrics_path,
    )
    rows = read_training_metrics_csv(metrics_path)
    curve_rows = loss_history_from_training_metrics(rows)
    curve_path = tmp_path / "loss_curve.png"

    write_loss_curve_png(curve_rows, curve_path)

    assert len(rows) == 1
    assert rows[0].gradient_norm == pytest.approx(0.75)
    assert curve_rows[0].training_loss == pytest.approx(2.5)
    assert curve_path.read_bytes().startswith(b"\x89PNG")


def _save(path: Path, *, global_step: int, validation_loss: float) -> None:
    model = TinyCheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=0,
        global_step=global_step,
        training_loss=validation_loss + 0.1,
        validation_loss=validation_loss,
        best_metric=validation_loss,
    )
