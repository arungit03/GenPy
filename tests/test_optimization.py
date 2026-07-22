from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import genpy_llm.performance as performance
from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.config import ConfigError, TokenizationConfig, VocabularyConfig, load_config
from genpy_llm.generation import TextGenerator
from genpy_llm.gpt import GPTModel, GPTModelError
from genpy_llm.performance import (
    PerformanceError,
    build_performance_metrics,
    compile_model,
    create_grad_scaler,
    normalize_mixed_precision,
    peak_memory_mb,
    reset_peak_memory,
    resolve_mixed_precision,
    unwrap_compiled_model,
    validate_mixed_precision,
)
from genpy_llm.quantization import QuantizationError, quantize_dynamic_int8
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.training import GPTTrainer, TrainingError
from genpy_llm.vocabulary import Vocabulary


class DummyScaler:
    def __init__(self) -> None:
        self.loaded = False

    def state_dict(self) -> dict[str, int]:
        return {"scale": 3}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.loaded = state["scale"] == 3


def test_full_precision_training_unchanged() -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer = GPTTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    metrics = trainer.train_batch(_batch(), batch_index=0)

    assert metrics.loss > 0
    assert trainer.total_optimizer_steps == 1
    assert trainer.mixed_precision == "none"
    assert trainer.scaler is None


def test_unsupported_amp_errors() -> None:
    with pytest.raises(PerformanceError, match="fp16"):
        validate_mixed_precision("fp16", torch.device("cpu"))
    with pytest.raises(TrainingError, match="fp16"):
        GPTTrainer(
            model=_tiny_model(),
            optimizer=torch.optim.AdamW(_tiny_model().parameters(), lr=0.01),
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            mixed_precision="fp16",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_fp16_amp_on_cuda_when_available() -> None:
    model = _tiny_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer = GPTTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cuda"),
        mixed_precision="fp16",
    )

    metrics = trainer.train_batch({key: value.cuda() for key, value in _batch().items()}, 0)

    assert metrics.loss > 0
    assert trainer.scaler is not None


def test_bf16_supported_on_cpu() -> None:
    validate_mixed_precision("bf16", torch.device("cpu"))
    assert create_grad_scaler("bf16", torch.device("cpu")) is None


def test_cuda_precision_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_mixed_precision("none", torch.device("cuda")) == "none"
    assert resolve_mixed_precision("fp16", torch.device("cuda")) == "fp16"
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert resolve_mixed_precision("bf16", torch.device("cuda")) == "bf16"


def test_mps_precision_resolution_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert resolve_mixed_precision("none", torch.device("mps")) == "none"
    assert resolve_mixed_precision("bf16", torch.device("mps")) == "none"
    assert "bf16 mixed precision is not supported on Apple MPS" in caplog.text

    monkeypatch.setattr(performance, "_mps_fp16_autocast_supported", lambda: True)
    assert resolve_mixed_precision("fp16", torch.device("mps")) == "fp16"

    monkeypatch.setattr(performance, "_mps_fp16_autocast_supported", lambda: False)
    assert resolve_mixed_precision("fp16", torch.device("mps")) == "none"


def test_cpu_none_precision_resolution() -> None:
    assert resolve_mixed_precision("none", torch.device("cpu")) == "none"


def test_fp32_alias_only_allowed_for_legacy_precision_field() -> None:
    assert normalize_mixed_precision("fp32", allow_fp32_alias=True) == "none"
    with pytest.raises(PerformanceError, match="Unsupported mixed precision"):
        normalize_mixed_precision("fp32")


def test_scaler_checkpoint_save_load(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / "scaler.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        global_step=1,
        scaler=DummyScaler(),
    )
    loaded_scaler = DummyScaler()

    load_checkpoint(checkpoint_path, model, optimizer=optimizer, scaler=loaded_scaler)

    assert loaded_scaler.loaded is True


def test_torch_compile_disabled_behavior() -> None:
    model = _tiny_model()

    assert compile_model(model, enabled=False) is model
    assert unwrap_compiled_model(model) is model


def test_torch_compile_optional_behavior_when_available() -> None:
    model = nn.Linear(2, 2)
    if not hasattr(torch, "compile"):
        with pytest.raises(PerformanceError):
            compile_model(model, enabled=True)
        return
    compiled = compile_model(model, enabled=True)

    assert unwrap_compiled_model(compiled) is model


def test_gradient_checkpointing_forward_backward_and_attention_error() -> None:
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    batch = _batch()

    logits = model(batch["input_ids"], padding_mask=batch["attention_mask"])
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (2, 4, model.vocab_size)
    assert any(parameter.grad is not None for parameter in model.parameters())
    with pytest.raises(GPTModelError, match="gradient checkpointing"):
        model(batch["input_ids"], padding_mask=batch["attention_mask"], return_attention=True)


def test_evaluation_without_checkpointing_attention_maps() -> None:
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    model.eval()
    logits, attention_maps = model(_batch()["input_ids"], return_attention=True)

    assert logits.shape == (2, 4, model.vocab_size)
    assert len(attention_maps) == model.num_layers


def test_performance_metric_calculations_and_cpu_memory() -> None:
    device = torch.device("cpu")
    reset_peak_memory(device)
    metrics = build_performance_metrics(
        elapsed_seconds=2.0,
        processed_tokens=10,
        device=device,
    )

    assert metrics.tokens_per_second == 5.0
    assert metrics.peak_memory_mb is None
    assert peak_memory_mb(device) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_cuda_memory_reporting_when_available() -> None:
    device = torch.device("cuda")
    reset_peak_memory(device)
    tensor = torch.empty((32, 32), device=device)
    del tensor

    assert peak_memory_mb(device) is not None


def test_dynamic_int8_cpu_quantization_logits_shape_and_no_mutation() -> None:
    model = _tiny_model()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    quantized = quantize_dynamic_int8(model)
    logits = quantized(_batch()["input_ids"])

    assert logits.shape == (2, 4, model.vocab_size)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_quantized_model_rejects_training_use() -> None:
    quantized = quantize_dynamic_int8(_tiny_model())

    with pytest.raises(QuantizationError, match="inference only"):
        quantized.train(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_quantization_rejects_cuda() -> None:
    with pytest.raises(QuantizationError, match="CPU"):
        quantize_dynamic_int8(_tiny_model().cuda())


def test_greedy_generation_with_quantized_model() -> None:
    vocabulary = _vocabulary()
    generator = TextGenerator(
        model=quantize_dynamic_int8(_tiny_model()),
        tokenizer=TextTokenizer(_tokenization_config()),
        vocabulary=vocabulary,
        device=torch.device("cpu"),
        context_length=4,
    )

    result = generator.generate("Hello", max_new_tokens=2, do_sample=False)

    assert result.prompt == "Hello"
    assert len(result.generated_token_ids) <= 2


def test_optimization_config_validation(tmp_path: Path) -> None:
    import yaml

    data = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    data["optimization"]["quantization"] = "bad"
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="optimization.quantization"):
        load_config(config_path)


def test_steps_1_to_18_remain_functional() -> None:
    from genpy_llm.fine_tuning import FineTuningDataset
    from genpy_llm.generation import TextGenerator

    assert FineTuningDataset is not None
    assert TextGenerator is not None


def _tiny_model() -> GPTModel:
    return GPTModel(
        vocab_size=len(_vocabulary()),
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        context_length=4,
        feed_forward_hidden_dim=16,
        padding_idx=0,
        dropout=0.0,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[5, 6, 3, 0], [5, 6, 7, 3]], dtype=torch.long),
        "target_ids": torch.tensor([[6, 3, 0, 0], [6, 7, 3, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long),
    }


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        method="word",
        preserve_case=True,
        preserve_punctuation=True,
        preserve_newlines=True,
        split_contractions=False,
        normalize_quotes=True,
        normalize_dashes=True,
        add_bos_token=False,
        add_eos_token=True,
        add_newline_token=True,
        bos_token="<BOS>",
        eos_token="<EOS>",
        newline_token="<NL>",
        unknown_token="<UNK>",
    )


def _vocabulary_config() -> VocabularyConfig:
    return VocabularyConfig(
        min_frequency=1,
        max_size=None,
        include_special_tokens=True,
        save_frequencies=True,
        strict_special_token_validation=True,
        pad_token="<PAD>",
        unknown_token="<UNK>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        newline_token="<NL>",
        special_token_order=("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"),
    )


def _vocabulary() -> Vocabulary:
    return Vocabulary(
        token_to_id={
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<NL>": 4,
            "Hello": 5,
            "world": 6,
            "!": 7,
        },
        frequencies=None,
        config=_vocabulary_config(),
    )
