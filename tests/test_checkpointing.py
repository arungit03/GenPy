from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from genpy_llm.checkpointing import (
    CheckpointError,
    checkpoint_filename,
    find_latest_checkpoint,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    save_managed_checkpoint,
)
from genpy_llm.config import ConfigError, load_config
from genpy_llm.training import GPTTrainer


class TinyGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = 11
        self.context_length = 4
        self.embedding = nn.Embedding(self.vocab_size, 8)
        self.output = nn.Linear(8, self.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        return self.output(self.embedding(input_ids))


@dataclass(frozen=True)
class TinyCheckpointConfig:
    directory: Path
    save_every_epochs: int = 1
    keep_last: int = 2
    save_best: bool = True
    monitor: str = "validation_loss"
    mode: str = "min"
    filename_prefix: str = "tiny"


def test_save_and_load_checkpoint_restores_model_optimizer_and_scheduler(tmp_path: Path) -> None:
    model = TinyGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    _optimizer_step(model, optimizer)
    scheduler.step()
    expected_state = _state_clone(model)

    checkpoint_path = tmp_path / "checkpoint.pt"
    metadata = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=5,
        training_loss=1.25,
        validation_loss=1.1,
        best_metric=1.1,
        extra_state={"tokens_processed": 128},
    )
    _damage_model(model)
    new_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    loaded = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        scheduler=new_scheduler,
    )

    assert metadata.epoch == 2
    assert loaded.epoch == 2
    assert loaded.global_step == 5
    assert loaded.validation_loss == pytest.approx(1.1)
    assert loaded.extra_state["tokens_processed"] == 128
    assert new_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert _state_matches(model, expected_state)


def test_checkpoint_rng_state_can_be_restored(tmp_path: Path) -> None:
    model = TinyGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    checkpoint_path = tmp_path / "rng.pt"

    save_checkpoint(checkpoint_path, model, optimizer, epoch=1, global_step=0)
    expected_python = random.random()
    expected_numpy = np.random.rand(3)
    expected_torch = torch.rand(3)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    load_checkpoint(checkpoint_path, model, optimizer=optimizer)

    assert random.random() == expected_python
    assert np.array_equal(np.random.rand(3), expected_numpy)
    assert torch.equal(torch.rand(3), expected_torch)


def test_find_latest_checkpoint_and_rotation_keep_newest_files(tmp_path: Path) -> None:
    model = TinyGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for epoch in range(1, 5):
        save_checkpoint(
            tmp_path / checkpoint_filename("tiny", epoch, epoch * 10),
            model,
            optimizer,
            epoch=epoch,
            global_step=epoch * 10,
        )
    save_checkpoint(tmp_path / "tiny_best.pt", model, optimizer, epoch=4, global_step=40)

    removed = rotate_checkpoints(tmp_path, keep_last=2, filename_prefix="tiny")
    latest = find_latest_checkpoint(tmp_path, filename_prefix="tiny")

    assert latest is not None
    assert latest.name == "tiny_epoch_0004_step_00000040.pt"
    assert {path.name for path in removed} == {
        "tiny_epoch_0001_step_00000010.pt",
        "tiny_epoch_0002_step_00000020.pt",
    }
    assert (tmp_path / "tiny_best.pt").exists()


def test_save_managed_checkpoint_updates_best_and_rotates(tmp_path: Path) -> None:
    model = TinyGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    best_metric = None
    for epoch, validation_loss in enumerate([2.0, 1.5, 1.8, 1.0], start=1):
        result = save_managed_checkpoint(
            tmp_path,
            model,
            optimizer,
            filename_prefix="tiny",
            epoch=epoch,
            global_step=epoch,
            training_loss=validation_loss + 0.1,
            validation_loss=validation_loss,
            best_metric=best_metric,
            keep_last=2,
            save_best=True,
            monitor="validation_loss",
            mode="min",
        )
        best_metric = result.best_metric

    regular_checkpoints = sorted(tmp_path.glob("tiny_epoch_*_step_*.pt"))
    best = load_checkpoint(tmp_path / "tiny_best.pt", model, optimizer=optimizer, restore_rng=False)

    assert [path.name for path in regular_checkpoints] == [
        "tiny_epoch_0003_step_00000003.pt",
        "tiny_epoch_0004_step_00000004.pt",
    ]
    assert best.best_metric == pytest.approx(1.0)
    assert result.best_checkpoint_path == tmp_path / "tiny_best.pt"


def test_trainer_fit_can_save_checkpoints_and_continue_epoch_numbering(tmp_path: Path) -> None:
    model = TinyGPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer = GPTTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )
    train_loader = [_batch(), _batch()]
    validation_loader = [_batch()]

    result = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=2,
        start_epoch=3,
        checkpoint_config=TinyCheckpointConfig(directory=tmp_path),
    )

    assert [epoch.epoch for epoch in result.epochs] == [3, 4]
    assert result.total_optimizer_steps == 4
    assert (tmp_path / "tiny_epoch_0003_step_00000002.pt").exists()
    assert (tmp_path / "tiny_epoch_0004_step_00000004.pt").exists()
    assert (tmp_path / "tiny_best.pt").exists()
    assert find_latest_checkpoint(tmp_path, filename_prefix="tiny") is not None


def test_invalid_checkpoint_config_values_are_rejected(tmp_path: Path) -> None:
    base_config = Path("configs/base.yaml").read_text(encoding="utf-8")
    invalid_config = base_config.replace("keep_last: 3", "keep_last: 0")
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(invalid_config, encoding="utf-8")

    with pytest.raises(ConfigError, match="checkpoint.keep_last"):
        load_config(config_path)


def test_load_rejects_missing_required_checkpoint_keys(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.pt"
    torch.save({"format_version": 1}, bad_path)

    with pytest.raises(CheckpointError, match="missing key"):
        load_checkpoint(bad_path, TinyGPT())


def _optimizer_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    batch = _batch()
    loss = nn.CrossEntropyLoss()(
        model(batch["input_ids"]).reshape(-1, 11),
        batch["target_ids"].reshape(-1),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long),
        "target_ids": torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
    }


def _state_clone(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _damage_model(model: nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3.0)


def _state_matches(model: nn.Module, expected_state: dict[str, torch.Tensor]) -> bool:
    return all(
        torch.equal(model.state_dict()[name].detach().cpu(), expected_state[name])
        for name in expected_state
    )
