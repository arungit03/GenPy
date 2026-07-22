from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from genpy_llm.dataset import GPTDataset
from genpy_llm.gpt import GPTModel
from genpy_llm.training import GPTTrainer, TrainingError


class CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _model(dropout: float = 0.0) -> GPTModel:
    return GPTModel(
        vocab_size=13,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        context_length=4,
        feed_forward_hidden_dim=16,
        padding_idx=0,
        dropout=dropout,
    )


def _batch(mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long),
        "target_ids": torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]], dtype=torch.long),
    }
    if mask is not None:
        batch["attention_mask"] = mask
    return batch


def _loader(num_batches: int = 2, mask: torch.Tensor | None = None) -> DataLoader:
    batch = _batch(mask)
    input_ids = batch["input_ids"].repeat(num_batches, 1)
    target_ids = batch["target_ids"].repeat(num_batches, 1)
    attention_mask = None
    if mask is not None:
        attention_mask = mask.repeat(num_batches, 1)
    dataset = GPTDataset(input_ids, target_ids, attention_mask)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def _trainer(
    model: GPTModel | None = None,
    accumulation: int = 1,
    max_grad_norm: float | None = 1.0,
    scheduler: object | None = None,
) -> GPTTrainer:
    model = model or _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return GPTTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        gradient_accumulation_steps=accumulation,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
    )


def test_one_training_batch() -> None:
    trainer = _trainer()

    metrics = trainer.train_batch(_batch(), batch_index=0)

    assert metrics.loss > 0
    assert metrics.tokens == 8
    assert trainer.total_optimizer_steps == 1


def test_parameter_updates_after_optimizer_step() -> None:
    model = _model()
    trainer = _trainer(model)
    before = model.token_embedding.weight.detach().clone()

    trainer.train_batch(_batch(), batch_index=0)

    assert not torch.equal(before, model.token_embedding.weight.detach())


def test_no_update_during_validation() -> None:
    model = _model()
    trainer = _trainer(model)
    before = model.token_embedding.weight.detach().clone()

    metrics = trainer.evaluate(_loader(1))

    assert metrics.loss > 0
    assert torch.equal(before, model.token_embedding.weight.detach())
    assert trainer.total_optimizer_steps == 0


def test_correct_train_eval_modes() -> None:
    model = _model()
    trainer = _trainer(model)
    model.train()
    trainer.evaluate(_loader(1))
    assert model.training

    model.eval()
    trainer.train_batch(_batch(), batch_index=0)
    assert model.training


def test_correct_gradient_accumulation() -> None:
    trainer = _trainer(accumulation=2)

    trainer.train_batch(_batch(), batch_index=0)
    assert trainer.total_optimizer_steps == 0
    trainer.train_batch(_batch(), batch_index=1)
    assert trainer.total_optimizer_steps == 1


def test_final_partial_accumulation_step() -> None:
    trainer = _trainer(accumulation=2)

    metrics = trainer.train_epoch(_loader(num_batches=3), epoch=1)

    assert metrics.optimizer_steps == 2
    assert trainer.total_optimizer_steps == 2


def test_scheduler_steps_only_with_optimizer() -> None:
    scheduler = CountingScheduler()
    trainer = _trainer(accumulation=2, scheduler=scheduler)

    trainer.train_batch(_batch(), batch_index=0)
    assert scheduler.steps == 0
    trainer.train_batch(_batch(), batch_index=1)
    assert scheduler.steps == 1


def test_gradient_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_clip(parameters, max_norm):
        calls["count"] += 1
        assert max_norm == 0.5
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)
    trainer = _trainer(max_grad_norm=0.5)

    trainer.train_batch(_batch(), batch_index=0)

    assert calls["count"] == 1


def test_clipping_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_clip(parameters, max_norm):
        calls["count"] += 1
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)
    trainer = _trainer(max_grad_norm=None)

    trainer.train_batch(_batch(), batch_index=0)

    assert calls["count"] == 0


def test_gradient_clearing() -> None:
    model = _model()
    trainer = _trainer(model)

    trainer.train_batch(_batch(), batch_index=0)

    assert all(parameter.grad is None for parameter in model.parameters())


def test_token_weighted_loss_averaging() -> None:
    trainer = _trainer()

    metrics = trainer.train_epoch(_loader(num_batches=2), epoch=1)

    assert metrics.training_loss > 0
    assert metrics.training_tokens == 16


def test_padding_mask_token_counting() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    trainer = _trainer()

    metrics = trainer.train_batch(_batch(mask), batch_index=0)

    assert metrics.tokens == 5


def test_missing_batch_keys() -> None:
    trainer = _trainer()

    with pytest.raises(TrainingError, match="missing"):
        trainer.train_batch({"input_ids": torch.ones(1, 2, dtype=torch.long)}, batch_index=0)


def test_invalid_shapes_and_dtypes() -> None:
    trainer = _trainer()

    with pytest.raises(TrainingError, match="torch.long"):
        trainer.train_batch(
            {
                "input_ids": torch.ones(1, 2, dtype=torch.float32),
                "target_ids": torch.ones(1, 2, dtype=torch.long),
            },
            batch_index=0,
        )
    with pytest.raises(TrainingError, match="two-dimensional"):
        trainer.train_batch(
            {
                "input_ids": torch.ones(1, 2, 1, dtype=torch.long),
                "target_ids": torch.ones(1, 2, dtype=torch.long),
            },
            batch_index=0,
        )


def test_non_scalar_loss_rejection() -> None:
    trainer = GPTTrainer(
        model=_model(),
        optimizer=torch.optim.SGD(_model().parameters(), lr=0.1),
        loss_fn=lambda logits, targets: torch.ones(2),
        device=torch.device("cpu"),
    )

    with pytest.raises(TrainingError, match="scalar"):
        trainer.train_batch(_batch(), batch_index=0)


def test_nan_and_infinite_loss_rejection() -> None:
    for bad_loss in [torch.tensor(float("nan")), torch.tensor(float("inf"))]:
        model = _model()
        trainer = GPTTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=lambda logits, targets, value=bad_loss: value,
            device=torch.device("cpu"),
        )
        with pytest.raises(TrainingError, match="NaN or infinite"):
            trainer.train_batch(_batch(), batch_index=0)


def test_empty_training_loader() -> None:
    trainer = _trainer()

    with pytest.raises(TrainingError, match="empty"):
        trainer.train_epoch([], epoch=1)


def test_empty_validation_loader() -> None:
    trainer = _trainer()

    metrics = trainer.evaluate([])

    assert metrics.loss == 0.0
    assert metrics.tokens == 0


def test_cpu_behavior() -> None:
    trainer = _trainer()

    assert trainer.train_batch(_batch(), batch_index=0).batch_size == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    model = _model()
    trainer = GPTTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cuda"),
    )

    assert trainer.train_batch(_batch(), batch_index=0).tokens == 8


def test_deterministic_smoke_training_with_fixed_seed() -> None:
    torch.manual_seed(7)
    first_model = _model()
    first = _trainer(first_model).fit(_loader(1), None, epochs=1)
    torch.manual_seed(7)
    second_model = _model()
    second = _trainer(second_model).fit(_loader(1), None, epochs=1)

    assert first.epochs[0].training_loss == pytest.approx(second.epochs[0].training_loss)


def test_fit_validation_and_result() -> None:
    trainer = _trainer()

    result = trainer.fit(_loader(2), _loader(1), epochs=2, validate_every_epochs=1)

    assert result.completed_epochs == 2
    assert result.total_optimizer_steps == 4
    assert result.epochs[-1].validation_loss is not None


def test_steps_1_to_13_remain_functional() -> None:
    model = _model()
    trainer = _trainer(model)

    logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    metrics = trainer.train_batch(_batch(), batch_index=0)

    assert logits.shape == (1, 3, 13)
    assert metrics.loss > 0
