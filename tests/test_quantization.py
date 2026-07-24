from __future__ import annotations

import json
from pathlib import Path

import torch

from genpy_llm.gpt import GPTModel
from genpy_llm.quantization import (
    QuantizedInferenceModel,
    detect_backend_capabilities,
    load_quantized_checkpoint,
    model_state_nbytes,
    normalize_quantization_method,
    quantize_bf16,
    quantize_dynamic_int8,
    quantize_fp16,
    save_quantized_checkpoint,
    tensor_tree_nbytes,
)
from genpy_llm.quantization_benchmark import (
    QuantizationBenchmarkSummary,
    QuantizationMethodResult,
    quantized_checkpoint_path,
    write_quantization_artifacts,
)


def test_fp16_and_bf16_conversion_do_not_mutate_source() -> None:
    model = _model()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    fp16 = quantize_fp16(model)
    bf16 = quantize_bf16(model)

    assert next(fp16.parameters()).dtype == torch.float16
    assert next(bf16.parameters()).dtype == torch.bfloat16
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_dynamic_int8_quantizes_linear_layers_and_keeps_wrapper_contract() -> None:
    model = _model()

    quantized = quantize_dynamic_int8(model)

    assert isinstance(quantized, QuantizedInferenceModel)
    assert any("quantized.dynamic" in type(module).__module__ for module in quantized.modules())
    logits = quantized(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert logits.shape == (1, 3, model.vocab_size)


def test_backend_capabilities_skip_dynamic_int8_on_mps() -> None:
    capabilities = detect_backend_capabilities(torch.device("mps"))

    assert capabilities.fp16 is True
    assert capabilities.dynamic_int8 is False
    assert "requires CPU" in capabilities.skip_reasons["dynamic_int8"]


def test_quantized_checkpoint_round_trip_for_fp16(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source")
    model = quantize_fp16(_model())
    output_path = tmp_path / "model_fp16.pt"

    info = save_quantized_checkpoint(
        model=model,
        output_path=output_path,
        method="fp16",
        source_checkpoint=source_checkpoint,
        source_metadata={"global_step": 7},
    )
    target = _model()
    loaded = load_quantized_checkpoint(output_path, target)

    assert output_path.is_file()
    assert info.method == "fp16"
    assert loaded.method == "fp16"
    assert next(loaded.model.parameters()).dtype == torch.float16
    assert loaded.metadata["source_metadata"]["global_step"] == 7


def test_tensor_tree_nbytes_counts_nested_tensors() -> None:
    payload = {
        "a": torch.zeros(2, 3, dtype=torch.float32),
        "b": [torch.zeros(4, dtype=torch.int64)],
    }

    assert tensor_tree_nbytes(payload) == 56
    assert model_state_nbytes(_model()) > 0


def test_quantization_artifacts_are_written(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"x" * 100)
    result = QuantizationMethodResult(
        method="fp16",
        status="ok",
        reason="",
        checkpoint=str(checkpoint),
        checkpoint_size_bytes=100,
        model_state_bytes=80,
        load_time_seconds=0.01,
        device="cpu",
        model_memory_mb=0.1,
        peak_device_memory_mb=None,
        inference_time_seconds=0.5,
        generated_tokens=10,
        tokens_per_second=20.0,
        validation_loss=1.5,
        perplexity=4.48,
    )
    summary = QuantizationBenchmarkSummary(
        source_checkpoint=str(checkpoint),
        evaluated_at="2026-01-01T00:00:00+00:00",
        device="cpu",
        capabilities=detect_backend_capabilities(torch.device("cpu")),
        results=(result,),
    )

    artifacts = write_quantization_artifacts(summary, tmp_path / "evaluation")

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["method"] == "fp16"
    assert "Phase 10 Quantization" in artifacts.report_path.read_text(encoding="utf-8")
    assert artifacts.csv_path.read_text(encoding="utf-8").count("\n") == 2


def test_method_normalization_and_checkpoint_naming(tmp_path: Path) -> None:
    assert normalize_quantization_method("int8") == "dynamic_int8"
    assert quantized_checkpoint_path(
        tmp_path / "last_checkpoint.pt",
        "fp16",
        tmp_path / "quantized",
    ) == tmp_path / "quantized" / "last_checkpoint_fp16.pt"


def _model() -> GPTModel:
    return GPTModel(
        vocab_size=32,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        context_length=8,
        feed_forward_hidden_dim=16,
        padding_idx=0,
        dropout=0.0,
        attention_dropout=0.0,
        feed_forward_dropout=0.0,
        residual_dropout=0.0,
        tie_embeddings=False,
    )
