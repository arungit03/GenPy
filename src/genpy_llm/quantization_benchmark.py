"""Phase 10 quantization benchmarking for GenPy."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from genpy_llm.checkpointing import LoadedCheckpoint
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.device import select_device
from genpy_llm.evaluation_benchmark import (
    EvaluationPrompt,
    calculate_validation_metrics,
    evaluate_prompts,
    load_evaluation_prompts,
    load_model_for_evaluation,
    resolve_evaluation_checkpoint,
)
from genpy_llm.fine_tuning import Phase7Config, load_phase7_config
from genpy_llm.performance import peak_memory_mb, reset_peak_memory
from genpy_llm.pretraining import create_phase6_model
from genpy_llm.quantization import (
    BackendCapabilities,
    QuantizationError,
    detect_backend_capabilities,
    is_quantization_supported,
    load_quantized_checkpoint,
    model_state_nbytes,
    normalize_quantization_method,
    quantize_model,
    save_quantized_checkpoint,
)

UTC = timezone.utc

QUANTIZATION_RESULTS_JSON = "quantization_results.json"
QUANTIZATION_RESULTS_CSV = "quantization_results.csv"
QUANTIZATION_REPORT = "quantization_report.md"


class QuantizationBenchmarkError(RuntimeError):
    """Raised when Phase 10 quantization benchmarking cannot continue."""


@dataclass(frozen=True)
class Phase10Config:
    """Complete Phase 10 quantization configuration."""

    project_root: Path
    phase7_config_path: Path
    phase7: Phase7Config
    source_checkpoint: str
    methods: tuple[str, ...]
    checkpoint_output_dir: Path
    evaluation_output_dir: Path
    prompt_dataset: Path
    max_new_tokens: int
    validation_batches: int | None
    benchmark_prompt_count: int
    warmup_runs: int
    timed_runs: int
    device: str
    log_level: str


@dataclass(frozen=True)
class QuantizationMethodResult:
    """Size, memory, speed, and quality metrics for one method."""

    method: str
    status: str
    reason: str
    checkpoint: str
    checkpoint_size_bytes: int
    model_state_bytes: int
    load_time_seconds: float | None
    device: str
    model_memory_mb: float | None
    peak_device_memory_mb: float | None
    inference_time_seconds: float | None
    generated_tokens: int
    tokens_per_second: float | None
    validation_loss: float | None
    perplexity: float | None


@dataclass(frozen=True)
class QuantizationBenchmarkSummary:
    """Complete Phase 10 benchmark result."""

    source_checkpoint: str
    evaluated_at: str
    device: str
    capabilities: BackendCapabilities
    results: tuple[QuantizationMethodResult, ...]


@dataclass(frozen=True)
class QuantizationArtifacts:
    """Paths written by the Phase 10 benchmark."""

    json_path: Path
    csv_path: Path
    report_path: Path


def load_phase10_config(path: Path | str = "configs/quantization.yaml") -> Phase10Config:
    """Load Phase 10 YAML configuration."""

    root = Path(__file__).resolve().parents[2]
    config_path = _resolve(root, path)
    raw = _yaml_mapping(config_path, "phase10")
    section = _as_mapping(raw.get("phase10", {}), "phase10")
    phase7_config_path = _resolve(root, section.get("phase7_config", "configs/finetuning.yaml"))
    phase7 = load_phase7_config(phase7_config_path)
    checkpoints = _as_mapping(section.get("checkpoints", {}), "phase10.checkpoints")
    evaluation = _as_mapping(section.get("evaluation", {}), "phase10.evaluation")
    runtime = _as_mapping(section.get("runtime", {}), "phase10.runtime")
    methods = tuple(
        normalize_quantization_method(str(method))
        for method in section.get("methods", ("fp16", "bf16", "dynamic_int8"))
    )
    if not methods:
        raise QuantizationBenchmarkError("phase10.methods must not be empty.")
    return Phase10Config(
        project_root=root,
        phase7_config_path=phase7_config_path,
        phase7=phase7,
        source_checkpoint=str(checkpoints.get("source_checkpoint", "latest")),
        methods=methods,
        checkpoint_output_dir=_resolve(
            root,
            checkpoints.get("output_dir", "checkpoints/quantized"),
        ),
        evaluation_output_dir=_resolve(root, evaluation.get("output_dir", "evaluation")),
        prompt_dataset=_resolve(
            root,
            evaluation.get("prompt_dataset", "data/evaluation/prompts.json"),
        ),
        max_new_tokens=int(evaluation.get("max_new_tokens", 16)),
        validation_batches=_optional_int(evaluation.get("validation_batches", 1)),
        benchmark_prompt_count=max(1, int(evaluation.get("benchmark_prompt_count", 2))),
        warmup_runs=max(0, int(evaluation.get("warmup_runs", 1))),
        timed_runs=max(1, int(evaluation.get("timed_runs", 1))),
        device=str(runtime.get("device", "auto")),
        log_level=str(section.get("log_level", "INFO")).upper(),
    )


def quantize_checkpoint_variants(
    config: Phase10Config,
    *,
    checkpoint_path: Path | None = None,
    methods: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Create separate model-only quantized checkpoints for requested methods."""

    source_checkpoint = checkpoint_path or resolve_evaluation_checkpoint(
        config.source_checkpoint,
        config=config.phase7,
    )
    created: dict[str, Path] = {}
    model, _tokenizer, loaded = load_model_for_evaluation(
        config.phase7,
        source_checkpoint,
        torch.device("cpu"),
    )
    source_payload = _loaded_checkpoint_payload(loaded)
    for method in methods or config.methods:
        normalized = normalize_quantization_method(method)
        quantized = quantize_model(model, normalized)
        output_path = quantized_checkpoint_path(
            source_checkpoint,
            normalized,
            config.checkpoint_output_dir,
        )
        save_quantized_checkpoint(
            model=quantized,
            output_path=output_path,
            method=normalized,
            source_checkpoint=source_checkpoint,
            source_metadata=source_payload,
        )
        created[normalized] = output_path
        del quantized
    return created


@torch.no_grad()
def benchmark_quantization(
    config: Phase10Config,
    *,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
) -> QuantizationBenchmarkSummary:
    """Run a smoke benchmark across baseline and supported quantized variants."""

    selected_device = device or select_device(config.device)
    capabilities = detect_backend_capabilities(selected_device)
    source_checkpoint = checkpoint_path or resolve_evaluation_checkpoint(
        config.source_checkpoint,
        config=config.phase7,
    )
    prompts = load_evaluation_prompts(config.prompt_dataset)[: config.benchmark_prompt_count]
    artifacts = quantize_checkpoint_variants(config, checkpoint_path=source_checkpoint)

    results: list[QuantizationMethodResult] = [
        _benchmark_fp32(
            config=config,
            checkpoint_path=source_checkpoint,
            prompts=prompts,
            device=selected_device,
        )
    ]
    for method in config.methods:
        artifact_path = artifacts[method]
        if not is_quantization_supported(method, selected_device):
            results.append(
                _skipped_result(
                    method=method,
                    path=artifact_path,
                    device=selected_device,
                    reason=capabilities.skip_reasons.get(method, "unsupported backend"),
                )
            )
            continue
        results.append(
            _benchmark_quantized_method(
                config=config,
                method=method,
                checkpoint_path=artifact_path,
                prompts=prompts,
                device=selected_device,
            )
        )

    return QuantizationBenchmarkSummary(
        source_checkpoint=str(source_checkpoint.resolve()),
        evaluated_at=datetime.now(UTC).isoformat(),
        device=str(selected_device),
        capabilities=capabilities,
        results=tuple(results),
    )


def write_quantization_artifacts(
    summary: QuantizationBenchmarkSummary,
    output_dir: Path | str,
) -> QuantizationArtifacts:
    """Write Phase 10 JSON, CSV, and Markdown benchmark artifacts."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / QUANTIZATION_RESULTS_JSON
    csv_path = directory / QUANTIZATION_RESULTS_CSV
    report_path = directory / QUANTIZATION_REPORT
    payload = {
        "metadata": {
            "source_checkpoint": summary.source_checkpoint,
            "evaluated_at": summary.evaluated_at,
            "device": summary.device,
            "capabilities": asdict(summary.capabilities),
        },
        "results": [asdict(item) for item in summary.results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(summary.results, csv_path)
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return QuantizationArtifacts(json_path=json_path, csv_path=csv_path, report_path=report_path)


def quantized_checkpoint_path(
    source_checkpoint: Path,
    method: str,
    output_dir: Path,
) -> Path:
    """Return the canonical Phase 10 artifact path for a source checkpoint and method."""

    return output_dir / f"{source_checkpoint.stem}_{normalize_quantization_method(method)}.pt"


def _benchmark_fp32(
    *,
    config: Phase10Config,
    checkpoint_path: Path,
    prompts: Sequence[EvaluationPrompt],
    device: torch.device,
) -> QuantizationMethodResult:
    started = time.perf_counter()
    model, tokenizer, _loaded = load_model_for_evaluation(config.phase7, checkpoint_path, device)
    load_time = time.perf_counter() - started
    return _run_metrics(
        config=config,
        method="fp32",
        checkpoint_path=checkpoint_path,
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        load_time=load_time,
        model_state_bytes=model_state_nbytes(model),
    )


def _benchmark_quantized_method(
    *,
    config: Phase10Config,
    method: str,
    checkpoint_path: Path,
    prompts: Sequence[EvaluationPrompt],
    device: torch.device,
) -> QuantizationMethodResult:
    tokenizer = CodeTokenizer.from_file(config.phase7.data.tokenizer)
    model = create_phase6_model(config.phase7.model, tokenizer)
    started = time.perf_counter()
    loaded = load_quantized_checkpoint(checkpoint_path, model, map_location="cpu")
    quantized_model = loaded.model
    if method != "dynamic_int8":
        quantized_model.to(device)
    load_time = time.perf_counter() - started
    active_device = torch.device("cpu") if method == "dynamic_int8" else device
    return _run_metrics(
        config=config,
        method=method,
        checkpoint_path=checkpoint_path,
        model=quantized_model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=active_device,
        load_time=load_time,
        model_state_bytes=loaded.model_state_bytes,
    )


def _run_metrics(
    *,
    config: Phase10Config,
    method: str,
    checkpoint_path: Path,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    prompts: Sequence[EvaluationPrompt],
    device: torch.device,
    load_time: float,
    model_state_bytes: int,
) -> QuantizationMethodResult:
    try:
        validation = calculate_validation_metrics(
            model=model,
            tokenizer=tokenizer,
            config=config.phase7,
            device=device,
            max_batches=config.validation_batches,
        )
        _warmup_generation(config, model, tokenizer, prompts, device)
        reset_peak_memory(device)
        started = time.perf_counter()
        generated_tokens = 0
        for _ in range(config.timed_runs):
            generation_results = evaluate_prompts(
                model=model,
                tokenizer=tokenizer,
                config=config.phase7,
                prompts=prompts,
                device=device,
                max_new_tokens=config.max_new_tokens,
            )
            generated_tokens += sum(item.generated_tokens for item in generation_results)
        inference_time = time.perf_counter() - started
    except (RuntimeError, QuantizationError, ValueError) as exc:
        return QuantizationMethodResult(
            method=method,
            status="failed",
            reason=str(exc),
            checkpoint=str(checkpoint_path.resolve()),
            checkpoint_size_bytes=checkpoint_path.stat().st_size,
            model_state_bytes=model_state_bytes,
            load_time_seconds=load_time,
            device=str(device),
            model_memory_mb=_mib(model_state_bytes),
            peak_device_memory_mb=peak_memory_mb(device),
            inference_time_seconds=None,
            generated_tokens=0,
            tokens_per_second=None,
            validation_loss=None,
            perplexity=None,
        )
    return QuantizationMethodResult(
        method=method,
        status="ok",
        reason="",
        checkpoint=str(checkpoint_path.resolve()),
        checkpoint_size_bytes=checkpoint_path.stat().st_size,
        model_state_bytes=model_state_bytes,
        load_time_seconds=load_time,
        device=str(device),
        model_memory_mb=_mib(model_state_bytes),
        peak_device_memory_mb=peak_memory_mb(device),
        inference_time_seconds=inference_time,
        generated_tokens=generated_tokens,
        tokens_per_second=generated_tokens / inference_time if inference_time > 0 else 0.0,
        validation_loss=None if validation is None else validation.loss,
        perplexity=None if validation is None else validation.perplexity,
    )


def _warmup_generation(
    config: Phase10Config,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    prompts: Sequence[EvaluationPrompt],
    device: torch.device,
) -> None:
    if not prompts:
        return
    for _ in range(config.warmup_runs):
        evaluate_prompts(
            model=model,
            tokenizer=tokenizer,
            config=config.phase7,
            prompts=prompts[:1],
            device=device,
            max_new_tokens=1,
        )


def _skipped_result(
    *,
    method: str,
    path: Path,
    device: torch.device,
    reason: str,
) -> QuantizationMethodResult:
    return QuantizationMethodResult(
        method=method,
        status="skipped",
        reason=reason,
        checkpoint=str(path.resolve()),
        checkpoint_size_bytes=path.stat().st_size if path.is_file() else 0,
        model_state_bytes=0,
        load_time_seconds=None,
        device=str(device),
        model_memory_mb=None,
        peak_device_memory_mb=None,
        inference_time_seconds=None,
        generated_tokens=0,
        tokens_per_second=None,
        validation_loss=None,
        perplexity=None,
    )


def _loaded_checkpoint_payload(loaded: LoadedCheckpoint) -> dict[str, Any]:
    return {
        "epoch": loaded.epoch,
        "global_step": loaded.global_step,
        "best_metric": loaded.best_metric,
        "training_loss": loaded.training_loss,
        "validation_loss": loaded.validation_loss,
        "checkpoint_path": str(loaded.checkpoint_path),
        "extra_state": dict(loaded.extra_state),
    }


def _write_csv(results: Sequence[QuantizationMethodResult], output_path: Path) -> None:
    fields = tuple(asdict(results[0])) if results else tuple(asdict(_empty_result()))
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _markdown_report(summary: QuantizationBenchmarkSummary) -> str:
    lines = [
        "# GenPy Phase 10 Quantization Report",
        "",
        f"- Source checkpoint: `{summary.source_checkpoint}`",
        f"- Device: `{summary.device}`",
        f"- Evaluated at: {summary.evaluated_at}",
        "",
        "| Method | Status | Size MiB | Load s | Memory MiB | Tokens/sec | Loss | Perplexity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary.results:
        lines.append(
            "| "
            f"{result.method} | {result.status} | {_mib(result.checkpoint_size_bytes):.2f} | "
            f"{_optional(result.load_time_seconds)} | {_optional(result.model_memory_mb)} | "
            f"{_optional(result.tokens_per_second)} | {_optional(result.validation_loss)} | "
            f"{_optional(result.perplexity)} |"
        )
    skipped = [item for item in summary.results if item.status != "ok"]
    if skipped:
        lines.extend(["", "## Skipped or Failed Methods", ""])
        for item in skipped:
            lines.append(f"- `{item.method}`: {item.reason}")
    lines.extend(
        [
            "",
            "Original checkpoints are read-only inputs. Phase 10 writes separate model-only "
            "quantized checkpoints under the configured quantized checkpoint directory.",
        ]
    )
    return "\n".join(lines) + "\n"


def _empty_result() -> QuantizationMethodResult:
    return QuantizationMethodResult(
        method="",
        status="",
        reason="",
        checkpoint="",
        checkpoint_size_bytes=0,
        model_state_bytes=0,
        load_time_seconds=None,
        device="",
        model_memory_mb=None,
        peak_device_memory_mb=None,
        inference_time_seconds=None,
        generated_tokens=0,
        tokens_per_second=None,
        validation_loss=None,
        perplexity=None,
    )


def _optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _mib(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _as_mapping(payload, label)


def _as_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise QuantizationBenchmarkError(f"{label} must be a mapping.")
    return payload


def _resolve(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


__all__ = [
    "QUANTIZATION_REPORT",
    "QUANTIZATION_RESULTS_CSV",
    "QUANTIZATION_RESULTS_JSON",
    "Phase10Config",
    "QuantizationArtifacts",
    "QuantizationBenchmarkError",
    "QuantizationBenchmarkSummary",
    "QuantizationMethodResult",
    "benchmark_quantization",
    "load_phase10_config",
    "quantize_checkpoint_variants",
    "quantized_checkpoint_path",
    "write_quantization_artifacts",
]
