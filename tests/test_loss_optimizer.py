from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from genpy_llm.config import LossConfig, OptimizerConfig, load_config
from genpy_llm.gpt import GPTModel
from genpy_llm.losses import GPTCrossEntropyLoss, LossError, create_loss_function
from genpy_llm.optimizers import OptimizerError, create_optimizer, create_optimizer_with_metadata
from genpy_llm.training import GPTTrainer


def _model(tie_embeddings: bool = True) -> GPTModel:
    return GPTModel(
        vocab_size=7,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        context_length=4,
        feed_forward_hidden_dim=16,
        padding_idx=5,
        dropout=0.0,
        tie_embeddings=tie_embeddings,
    )


def _optimizer_config(separate_weight_decay: bool = True) -> OptimizerConfig:
    return OptimizerConfig(
        type="adamw",
        learning_rate=0.003,
        weight_decay=0.2,
        beta1=0.8,
        beta2=0.9,
        epsilon=1e-7,
        separate_weight_decay=separate_weight_decay,
    )


def test_basic_cross_entropy_loss() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    targets = torch.tensor([[0, 1]], dtype=torch.long)

    loss = GPTCrossEntropyLoss(ignore_padding=False)(logits, targets)

    assert loss.item() < 0.2


def test_scalar_loss_output() -> None:
    loss = GPTCrossEntropyLoss(ignore_padding=False)(
        torch.randn(2, 3, 5),
        torch.randint(0, 5, (2, 3), dtype=torch.long),
    )

    assert loss.ndim == 0


def test_batched_and_flattened_inputs_match() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3), dtype=torch.long)
    loss_fn = GPTCrossEntropyLoss(ignore_padding=False)

    assert torch.allclose(
        loss_fn(logits, targets),
        loss_fn(logits.reshape(-1, 5), targets.reshape(-1)),
    )


def test_gradient_flow() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.randint(0, 5, (2, 3), dtype=torch.long)

    GPTCrossEntropyLoss(ignore_padding=False)(logits, targets).backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_padding_ignored_correctly_non_zero_padding_id() -> None:
    logits = torch.randn(1, 3, 6)
    targets = torch.tensor([[1, 5, 2]], dtype=torch.long)
    loss_fn = GPTCrossEntropyLoss(padding_idx=5, ignore_padding=True)

    expected = torch.nn.functional.cross_entropy(
        logits[:, [0, 2]].reshape(-1, 6),
        torch.tensor([1, 2]),
    )

    assert torch.allclose(loss_fn(logits, targets), expected)


def test_loss_changes_when_padding_is_ignored() -> None:
    logits = torch.randn(1, 3, 6)
    targets = torch.tensor([[1, 5, 2]], dtype=torch.long)

    ignored = GPTCrossEntropyLoss(padding_idx=5, ignore_padding=True)(logits, targets)
    included = GPTCrossEntropyLoss(padding_idx=5, ignore_padding=False)(logits, targets)

    assert not torch.allclose(ignored, included)


def test_label_smoothing() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3), dtype=torch.long)

    first = GPTCrossEntropyLoss(ignore_padding=False, label_smoothing=0.0)(logits, targets)
    second = GPTCrossEntropyLoss(ignore_padding=False, label_smoothing=0.2)(logits, targets)

    assert not torch.allclose(first, second)


def test_invalid_logits_rank() -> None:
    with pytest.raises(LossError, match="rank"):
        GPTCrossEntropyLoss()(torch.randn(2, 3, 4, 5), torch.ones(2, 3, dtype=torch.long))


def test_invalid_target_rank() -> None:
    with pytest.raises(LossError, match="rank"):
        GPTCrossEntropyLoss()(torch.randn(2, 3, 5), torch.ones(2, 3, 1, dtype=torch.long))


def test_shape_mismatch() -> None:
    with pytest.raises(LossError, match="match"):
        GPTCrossEntropyLoss()(torch.randn(2, 3, 5), torch.ones(2, 2, dtype=torch.long))


def test_invalid_target_dtype() -> None:
    with pytest.raises(LossError, match="torch.long"):
        GPTCrossEntropyLoss()(torch.randn(2, 3, 5), torch.ones(2, 3, dtype=torch.float32))


def test_out_of_range_targets() -> None:
    with pytest.raises(LossError, match="below"):
        GPTCrossEntropyLoss()(torch.randn(1, 2, 5), torch.tensor([[1, 5]], dtype=torch.long))


def test_nan_and_infinite_logits() -> None:
    for value in [float("nan"), float("inf")]:
        logits = torch.randn(1, 2, 5)
        logits[0, 0, 0] = value
        with pytest.raises(LossError, match="NaN or infinite"):
            GPTCrossEntropyLoss()(logits, torch.tensor([[1, 2]], dtype=torch.long))


def test_all_padding_batch_behavior() -> None:
    with pytest.raises(LossError, match="All target tokens"):
        GPTCrossEntropyLoss(padding_idx=5)(
            torch.randn(1, 2, 6),
            torch.tensor([[5, 5]], dtype=torch.long),
        )


def test_create_loss_function_uses_vocabulary_padding(tmp_path: Path) -> None:
    path = _write_vocab(tmp_path, pad_token="<PADX>")
    config = LossConfig(type="cross_entropy", ignore_padding=True, label_smoothing=0.1)

    loss_fn = create_loss_function(path, config)

    assert loss_fn.padding_idx == 0
    assert loss_fn.label_smoothing == 0.1


def test_adamw_creation_and_hyperparameters() -> None:
    optimizer = create_optimizer(_model(), _optimizer_config())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.003)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1e-7)


def test_weight_decay_grouping() -> None:
    optimizer, metadata = create_optimizer_with_metadata(_model(), _optimizer_config())

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)
    assert metadata.decayed_parameter_count > 0
    assert metadata.non_decayed_parameter_count > 0


def test_bias_and_layernorm_excluded_linear_weights_included() -> None:
    model = _model()
    optimizer, _metadata = create_optimizer_with_metadata(model, _optimizer_config())
    decayed = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    non_decayed = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}

    assert id(model.blocks[0].attention.qkv_projection.weight) in decayed
    assert id(model.blocks[0].attention.qkv_projection.bias) in non_decayed
    assert id(model.final_norm.layer_norm.weight) in non_decayed


def test_frozen_parameters_excluded() -> None:
    model = _model()
    model.final_norm.layer_norm.weight.requires_grad_(False)
    optimizer, _metadata = create_optimizer_with_metadata(model, _optimizer_config())
    grouped = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert id(model.final_norm.layer_norm.weight) not in grouped


def test_no_duplicate_parameters_and_every_trainable_included() -> None:
    model = _model(tie_embeddings=True)
    optimizer, metadata = create_optimizer_with_metadata(model, _optimizer_config())
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = list(model.parameters())

    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}
    assert metadata.trainable_tensor_count == len(trainable)


def test_tied_weights_handled_once() -> None:
    model = _model(tie_embeddings=True)
    optimizer, _metadata = create_optimizer_with_metadata(model, _optimizer_config())
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    assert [id(parameter) for parameter in grouped].count(id(model.token_embedding.weight)) == 1


def test_single_optimizer_group_when_separate_disabled() -> None:
    optimizer, _metadata = create_optimizer_with_metadata(
        _model(),
        _optimizer_config(separate_weight_decay=False),
    )

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)


def test_no_trainable_parameters_error() -> None:
    model = _model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(OptimizerError, match="no trainable"):
        create_optimizer(model, _optimizer_config())


def test_one_optimizer_step_updates_parameters_and_clear_gradients() -> None:
    model = _model()
    optimizer = create_optimizer(model, _optimizer_config())
    before = model.token_embedding.weight.detach().clone()
    logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    loss = GPTCrossEntropyLoss(padding_idx=5)(logits, torch.tensor([[2, 3, 4]], dtype=torch.long))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    assert not torch.equal(before, model.token_embedding.weight.detach())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_trainer_integration() -> None:
    model = _model()
    loss_fn = GPTCrossEntropyLoss(padding_idx=5)
    optimizer = create_optimizer(model, _optimizer_config())
    trainer = GPTTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )

    metrics = trainer.train_batch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "target_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
        },
        batch_index=0,
    )

    assert metrics.loss > 0


def test_cpu_behavior() -> None:
    assert GPTCrossEntropyLoss()(torch.randn(1, 2, 5), torch.tensor([[1, 2]])).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_behavior_when_available() -> None:
    logits = torch.randn(1, 2, 5, device="cuda")
    targets = torch.tensor([[1, 2]], dtype=torch.long, device="cuda")

    assert GPTCrossEntropyLoss()(logits, targets).device.type == "cuda"


def test_steps_1_to_14_remain_functional() -> None:
    config = load_config()
    model = _model()
    loss_fn = GPTCrossEntropyLoss(padding_idx=5)
    optimizer = create_optimizer(model, _optimizer_config())

    assert config.loss.type == "cross_entropy"
    loss = loss_fn(
        model(torch.tensor([[1, 2, 3]], dtype=torch.long)),
        torch.tensor([[2, 3, 4]]),
    )

    assert loss.ndim == 0
    assert isinstance(optimizer, torch.optim.AdamW)


def _write_vocab(tmp_path: Path, pad_token: str = "<PAD>") -> Path:
    tokens = [pad_token, "<UNK>", "<BOS>", "<EOS>", "<NL>", "a"]
    payload = {
        "format_version": 1,
        "vocab_size": len(tokens),
        "special_tokens": {
            "pad_token": pad_token,
            "unknown_token": "<UNK>",
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "newline_token": "<NL>",
        },
        "token_to_id": {token: index for index, token in enumerate(tokens)},
        "id_to_token": tokens,
        "frequencies": {token: 1 for token in tokens},
    }
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
