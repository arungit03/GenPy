from __future__ import annotations

from pathlib import Path

import pytest
import torch

from genpy_llm.gpt import GPTModel
from genpy_llm.lora import (
    LoRAError,
    apply_lora,
    iter_lora_adapters,
    load_lora_adapters,
    merge_lora_weights,
    save_lora_adapters,
    unmerge_lora_weights,
)
from genpy_llm.lora_evaluation import ComparisonMethodResult, write_lora_comparison
from genpy_llm.lora_training import load_phase9_config


def test_lora_targets_active_attention_and_freezes_base_model() -> None:
    model = _model()

    stats = apply_lora(model, rank=2, alpha=4.0, dropout=0.1)

    assert stats.adapter_count == 4
    assert {item.module_name for item in stats.adapters} == {
        "blocks.0.attention.qkv_projection",
        "blocks.0.attention.output_projection",
        "blocks.1.attention.qkv_projection",
        "blocks.1.attention.output_projection",
    }
    assert stats.trainable_parameters == stats.adapter_parameters
    assert stats.trainable_parameters > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is ("lora_A" in name or "lora_B" in name)


def test_effective_weight_is_visible_to_attention_f_linear_path() -> None:
    torch.manual_seed(5)
    model = _model(num_layers=1)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    baseline = model(input_ids).detach()
    apply_lora(
        model,
        rank=2,
        alpha=2.0,
        target_modules=("attention.qkv_projection",),
    )
    name, layer, adapter = next(iter_lora_adapters(model))
    original = layer.parametrizations.weight.original.detach().clone()
    with torch.no_grad():
        adapter.lora_A.normal_(mean=0.0, std=0.5)
        adapter.lora_B.normal_(mean=0.0, std=0.5)
    expected = original + adapter.delta_weight()

    updated = model(input_ids).detach()

    assert name == "blocks.0.attention.qkv_projection"
    assert torch.allclose(layer.weight, expected)
    assert not torch.allclose(updated, baseline)


def test_lora_merge_and_unmerge_preserve_effective_output() -> None:
    torch.manual_seed(7)
    model = _model(num_layers=1)
    apply_lora(model, rank=2, alpha=4.0)
    model.eval()
    for _name, _layer, adapter in iter_lora_adapters(model):
        with torch.no_grad():
            adapter.lora_B.normal_(mean=0.0, std=0.1)
    input_ids = torch.tensor([[2, 4, 6]], dtype=torch.long)
    unmerged_output = model(input_ids).detach()

    merged = merge_lora_weights(model)
    merged_output = model(input_ids).detach()
    restored = unmerge_lora_weights(model)
    restored_output = model(input_ids).detach()

    assert all(item.merged for item in merged.adapters)
    assert all(not item.merged for item in restored.adapters)
    assert torch.allclose(merged_output, unmerged_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(restored_output, unmerged_output, atol=1e-5, rtol=1e-5)


def test_lora_dropout_is_training_only() -> None:
    model = _model(num_layers=1)
    apply_lora(
        model,
        rank=4,
        alpha=4.0,
        dropout=0.5,
        target_modules=("attention.output_projection",),
    )
    _name, layer, adapter = next(iter_lora_adapters(model))
    with torch.no_grad():
        adapter.lora_A.fill_(1.0)
        adapter.lora_B.fill_(1.0)
    model.train()
    training_weight_one = layer.weight.detach().clone()
    training_weight_two = layer.weight.detach().clone()
    model.eval()
    evaluation_weight_one = layer.weight.detach().clone()
    evaluation_weight_two = layer.weight.detach().clone()

    assert not torch.equal(training_weight_one, training_weight_two)
    assert torch.equal(evaluation_weight_one, evaluation_weight_two)


def test_adapter_only_save_and_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(11)
    source = _model(num_layers=1)
    target = _model(num_layers=1)
    target.load_state_dict(source.state_dict())
    apply_lora(source, rank=2, alpha=8.0, dropout=0.0)
    for _name, _layer, adapter in iter_lora_adapters(source):
        with torch.no_grad():
            adapter.lora_B.uniform_(-0.2, 0.2)
    path = save_lora_adapters(source, tmp_path / "adapter.pt", metadata={"step": 12})

    loaded = load_lora_adapters(target, path)

    source.eval()
    target.eval()
    input_ids = torch.tensor([[1, 3, 5]], dtype=torch.long)
    assert loaded.adapter_count == 2
    assert loaded.metadata["step"] == 12
    assert torch.allclose(source(input_ids), target(input_ids))
    assert path.stat().st_size < 100_000


def test_loading_into_existing_adapters_keeps_only_lora_trainable(tmp_path: Path) -> None:
    source = _model(num_layers=1)
    target = _model(num_layers=1)
    target.load_state_dict(source.state_dict())
    apply_lora(source, rank=2, alpha=2.0)
    apply_lora(target, rank=2, alpha=2.0)
    path = save_lora_adapters(source, tmp_path / "resume.pt")

    load_lora_adapters(target, path)

    for name, parameter in target.named_parameters():
        assert parameter.requires_grad is ("lora_A" in name or "lora_B" in name)


def test_optimizer_step_changes_only_lora_parameters() -> None:
    torch.manual_seed(13)
    model = _model(num_layers=1)
    apply_lora(model, rank=2, alpha=2.0)
    originals = {
        name: layer.parametrizations.weight.original.detach().clone()
        for name, layer, _adapter in iter_lora_adapters(model)
    }
    adapter_before = {
        name: adapter.lora_B.detach().clone()
        for name, _layer, adapter in iter_lora_adapters(model)
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss = model(torch.tensor([[1, 2, 3]], dtype=torch.long)).sum()
    loss.backward()
    optimizer.step()

    for name, layer, adapter in iter_lora_adapters(model):
        assert torch.equal(layer.parametrizations.weight.original, originals[name])
        assert not torch.equal(adapter.lora_B, adapter_before[name])


def test_invalid_or_duplicate_lora_configuration_is_rejected() -> None:
    model = _model(num_layers=1)

    with pytest.raises(LoRAError, match="rank"):
        apply_lora(model, rank=0, alpha=1.0)
    apply_lora(model, rank=2, alpha=2.0)
    with pytest.raises(LoRAError, match="already"):
        apply_lora(model, rank=2, alpha=2.0)


def test_phase9_config_targets_active_attention_projections() -> None:
    config = load_phase9_config()

    assert config.adapter.rank == 8
    assert config.adapter.target_modules == (
        "attention.qkv_projection",
        "attention.output_projection",
    )
    assert config.training.base_checkpoint.name == "last_checkpoint.pt"


def test_full_vs_lora_comparison_artifacts(tmp_path: Path) -> None:
    full_checkpoint = tmp_path / "full.pt"
    adapter_checkpoint = tmp_path / "adapter.pt"
    full_checkpoint.write_bytes(b"x" * 1000)
    adapter_checkpoint.write_bytes(b"x" * 100)
    full = ComparisonMethodResult(
        method="Full fine-tuning",
        checkpoint=str(full_checkpoint),
        checkpoint_size_bytes=1000,
        trainable_parameters=1000,
        validation_loss=2.0,
        perplexity=7.3,
        generation_tokens_per_second=10.0,
        automatic_checks_passed=3,
        prompt_count=20,
    )
    lora = ComparisonMethodResult(
        method="LoRA",
        checkpoint=str(adapter_checkpoint),
        checkpoint_size_bytes=100,
        trainable_parameters=100,
        validation_loss=2.1,
        perplexity=8.1,
        generation_tokens_per_second=11.0,
        automatic_checks_passed=2,
        prompt_count=20,
    )

    json_path, csv_path, report_path = write_lora_comparison(full, lora, tmp_path / "eval")

    assert json_path.is_file()
    assert csv_path.read_text(encoding="utf-8").count("\n") == 3
    report = report_path.read_text(encoding="utf-8")
    assert "Full Fine-Tuning vs LoRA" in report
    assert "90.0000%" in report


def _model(*, num_layers: int = 2) -> GPTModel:
    return GPTModel(
        vocab_size=32,
        embedding_dim=8,
        num_heads=2,
        num_layers=num_layers,
        context_length=8,
        feed_forward_hidden_dim=16,
        padding_idx=0,
        dropout=0.0,
        attention_dropout=0.0,
        feed_forward_dropout=0.0,
        residual_dropout=0.0,
    )
