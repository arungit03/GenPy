"""Full fine-tuning versus LoRA evaluation for GenPy Phase 9."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from genpy_llm.checkpointing import LoadedCheckpoint, load_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.evaluation_benchmark import (
    EvaluationArtifacts,
    EvaluationSummary,
    build_evaluation_summary,
    calculate_validation_metrics,
    evaluate_prompts,
    load_evaluation_prompts,
    load_model_for_evaluation,
    write_evaluation_artifacts,
)
from genpy_llm.lora import load_lora_adapters, lora_stats
from genpy_llm.lora_training import Phase9Config
from genpy_llm.pretraining import create_phase6_model

COMPARISON_JSON = "comparison_results.json"
COMPARISON_CSV = "comparison_results.csv"
COMPARISON_REPORT = "comparison_report.md"


@dataclass(frozen=True)
class ComparisonMethodResult:
    """Comparable quality, speed, parameter, and storage metrics."""

    method: str
    checkpoint: str
    checkpoint_size_bytes: int
    trainable_parameters: int
    validation_loss: float | None
    perplexity: float | None
    generation_tokens_per_second: float
    automatic_checks_passed: int
    prompt_count: int


@dataclass(frozen=True)
class LoRAComparisonResult:
    """Evaluation summaries and comparison artifact paths."""

    full_fine_tuning: ComparisonMethodResult
    lora: ComparisonMethodResult
    full_artifacts: EvaluationArtifacts
    lora_artifacts: EvaluationArtifacts
    comparison_json: Path
    comparison_csv: Path
    comparison_report: Path


@torch.no_grad()
def evaluate_full_vs_lora(
    config: Phase9Config,
    *,
    adapter_path: Path,
    device: torch.device,
    output_dir: Path | None = None,
    max_new_tokens: int | None = None,
    validation_batches: int | None = None,
) -> LoRAComparisonResult:
    """Evaluate full fine-tuning and LoRA on identical validation data and prompts."""

    destination = output_dir or config.evaluation.output_dir
    generation_length = max_new_tokens or config.evaluation.max_new_tokens
    batch_limit = (
        validation_batches
        if validation_batches is not None
        else config.evaluation.validation_batches
    )
    prompts = load_evaluation_prompts(config.evaluation.prompt_dataset)

    full_checkpoint = config.evaluation.full_fine_tuned_checkpoint
    full_model, full_tokenizer, full_loaded = load_model_for_evaluation(
        config.phase7,
        full_checkpoint,
        device,
    )
    full_parameter_count = sum(parameter.numel() for parameter in full_model.parameters())
    full_validation = calculate_validation_metrics(
        model=full_model,
        tokenizer=full_tokenizer,
        config=config.phase7,
        device=device,
        max_batches=batch_limit,
    )
    full_results = evaluate_prompts(
        model=full_model,
        tokenizer=full_tokenizer,
        config=config.phase7,
        prompts=prompts,
        device=device,
        max_new_tokens=generation_length,
    )
    full_summary = build_evaluation_summary(
        checkpoint_path=full_checkpoint,
        loaded_checkpoint=full_loaded,
        device=device,
        validation=full_validation,
        results=full_results,
    )
    full_artifacts = write_evaluation_artifacts(full_summary, destination / "full_fine_tuning")
    del full_model
    _empty_device_cache(device)

    tokenizer = CodeTokenizer.from_file(config.phase7.data.tokenizer)
    lora_model = create_phase6_model(config.phase7.model, tokenizer)
    base_loaded = load_checkpoint(
        config.training.base_checkpoint,
        lora_model,
        optimizer=None,
        map_location="cpu",
        restore_rng=False,
    )
    lora_model.to(device)
    loaded_adapter = load_lora_adapters(lora_model, adapter_path, map_location="cpu")
    lora_model.eval()
    adapter_stats = lora_stats(lora_model)
    lora_validation = calculate_validation_metrics(
        model=lora_model,
        tokenizer=tokenizer,
        config=config.phase7,
        device=device,
        max_batches=batch_limit,
    )
    lora_results = evaluate_prompts(
        model=lora_model,
        tokenizer=tokenizer,
        config=config.phase7,
        prompts=prompts,
        device=device,
        max_new_tokens=generation_length,
    )
    adapter_step = loaded_adapter.metadata.get("global_step", 0)
    adapter_loaded_checkpoint = LoadedCheckpoint(
        epoch=base_loaded.epoch,
        global_step=int(adapter_step) if isinstance(adapter_step, (int, float)) else 0,
        best_metric=None,
        training_loss=None,
        validation_loss=None,
        checkpoint_path=adapter_path.resolve(),
        extra_state=loaded_adapter.metadata,
    )
    lora_summary = build_evaluation_summary(
        checkpoint_path=adapter_path,
        loaded_checkpoint=adapter_loaded_checkpoint,
        device=device,
        validation=lora_validation,
        results=lora_results,
    )
    lora_artifacts = write_evaluation_artifacts(lora_summary, destination / "lora")

    full_method = _method_result(
        "Full fine-tuning",
        full_checkpoint,
        full_parameter_count,
        full_summary,
    )
    lora_method = _method_result(
        "LoRA",
        adapter_path,
        adapter_stats.trainable_parameters,
        lora_summary,
    )
    comparison_json, comparison_csv, comparison_report = write_lora_comparison(
        full_method,
        lora_method,
        destination,
    )
    return LoRAComparisonResult(
        full_fine_tuning=full_method,
        lora=lora_method,
        full_artifacts=full_artifacts,
        lora_artifacts=lora_artifacts,
        comparison_json=comparison_json,
        comparison_csv=comparison_csv,
        comparison_report=comparison_report,
    )


def _method_result(
    method: str,
    checkpoint: Path,
    trainable_parameters: int,
    summary: EvaluationSummary,
) -> ComparisonMethodResult:
    return ComparisonMethodResult(
        method=method,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        trainable_parameters=trainable_parameters,
        validation_loss=None if summary.validation is None else summary.validation.loss,
        perplexity=None if summary.validation is None else summary.validation.perplexity,
        generation_tokens_per_second=summary.tokens_per_second,
        automatic_checks_passed=sum(item.passed is True for item in summary.results),
        prompt_count=len(summary.results),
    )


def write_lora_comparison(
    full: ComparisonMethodResult,
    lora: ComparisonMethodResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / COMPARISON_JSON
    csv_path = output_dir / COMPARISON_CSV
    report_path = output_dir / COMPARISON_REPORT
    parameter_reduction = _reduction(full.trainable_parameters, lora.trainable_parameters)
    storage_reduction = _reduction(full.checkpoint_size_bytes, lora.checkpoint_size_bytes)
    payload: dict[str, Any] = {
        "full_fine_tuning": asdict(full),
        "lora": asdict(lora),
        "comparison": {
            "trainable_parameter_reduction_percent": parameter_reduction,
            "checkpoint_size_reduction_percent": storage_reduction,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = tuple(asdict(full))
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(asdict(full))
        writer.writerow(asdict(lora))
    report_path.write_text(
        _comparison_markdown(full, lora, parameter_reduction, storage_reduction),
        encoding="utf-8",
    )
    return json_path, csv_path, report_path


def _comparison_markdown(
    full: ComparisonMethodResult,
    lora: ComparisonMethodResult,
    parameter_reduction: float,
    storage_reduction: float,
) -> str:
    lines = [
        "# GenPy Phase 9: Full Fine-Tuning vs LoRA",
        "",
        "| Metric | Full fine-tuning | LoRA |",
        "|---|---:|---:|",
        f"| Trainable parameters | {full.trainable_parameters:,} | {lora.trainable_parameters:,} |",
        (
            f"| Checkpoint size | {_mib(full.checkpoint_size_bytes):.2f} MiB | "
            f"{_mib(lora.checkpoint_size_bytes):.2f} MiB |"
        ),
        (
            f"| Validation loss | {_optional(full.validation_loss)} | "
            f"{_optional(lora.validation_loss)} |"
        ),
        f"| Perplexity | {_optional(full.perplexity)} | {_optional(lora.perplexity)} |",
        (
            "| Generation speed | "
            f"{full.generation_tokens_per_second:.3f} tokens/sec | "
            f"{lora.generation_tokens_per_second:.3f} tokens/sec |"
        ),
        (
            "| Automatic checks | "
            f"{full.automatic_checks_passed}/{full.prompt_count} | "
            f"{lora.automatic_checks_passed}/{lora.prompt_count} |"
        ),
        "",
        f"- Trainable-parameter reduction: {parameter_reduction:.4f}%",
        f"- Checkpoint-size reduction: {storage_reduction:.4f}%",
        "",
        "Both methods use the same prompts, generation settings, and validation batch limit.",
    ]
    return "\n".join(lines) + "\n"


def _reduction(baseline: int, reduced: int) -> float:
    return 0.0 if baseline <= 0 else 100.0 * (baseline - reduced) / baseline


def _optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _mib(value: int) -> float:
    return value / (1024 * 1024)


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


__all__ = [
    "COMPARISON_CSV",
    "COMPARISON_JSON",
    "COMPARISON_REPORT",
    "ComparisonMethodResult",
    "LoRAComparisonResult",
    "evaluate_full_vs_lora",
    "write_lora_comparison",
]
