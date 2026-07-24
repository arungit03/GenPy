"""Benchmark and evaluation framework comparing GenPy checkpoints.

Compares the base pretraining checkpoint against the continued-pretraining
checkpoint without retraining or modifying anything: validation loss and
perplexity on the Final Corpus, next-token accuracy, inference speed and
latency, memory and checkpoint size, a Python coding benchmark, documentation
QA, and text-generation quality. Reuses the existing model, tokenizer, dataset
loader, generation utilities, and the dependency-free PNG plotting canvas.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import logging
import math
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from genpy_llm.benchmark_prompts import (
    DOCUMENTATION_QA,
    PYTHON_BENCHMARK_PROMPTS,
    TEXT_GENERATION_TASKS,
    DocumentationQuestion,
    PythonPrompt,
    TextGenerationTask,
)
from genpy_llm.checkpointing import load_checkpoint
from genpy_llm.code_evaluation import _Canvas, _plot_series, format_size
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.device import select_device
from genpy_llm.evaluation_benchmark import extract_python_code
from genpy_llm.losses import GPTCrossEntropyLoss
from genpy_llm.pretraining import create_phase6_model, load_phase6_config
from genpy_llm.pretraining_dataset import PackedSequenceDataset
from genpy_llm.pretraining_generation import CodeGenerationSettings, generate_code_sample

UTC = timezone.utc

LOGGER = logging.getLogger("genpy_llm.benchmark_suite")
CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint_step_(?P<step>\d{5,})$")
MARKDOWN_MARKERS = ("#", "- ", "* ", "1.", "```")


class BenchmarkError(RuntimeError):
    """Raised when the benchmark suite cannot proceed."""


@dataclass(frozen=True)
class BenchmarkPaths:
    """Input and output locations for the benchmark suite."""

    model_config: Path
    training_config: Path
    optimizer_config: Path
    generation_config: Path
    tokenizer: Path
    corpus_directory: Path
    base_search_dir: Path
    continued_search_dir: Path
    output_dir: Path
    training_metrics: tuple[Path, ...]
    continued_training_log: Path


@dataclass(frozen=True)
class BenchmarkEvaluationConfig:
    """Evaluation workload sizes and validation settings."""

    batch_size: int
    validation_batches: int
    validation_fraction: float
    seed: int
    python_prompt_limit: int | None
    doc_qa_limit: int | None
    text_task_limit: int | None
    latency_runs: int
    latency_tokens: int
    execution_timeout_seconds: float
    run_generated_code: bool


@dataclass(frozen=True)
class BenchmarkGenerationConfig:
    """Sampling settings used for every generation task."""

    temperature: float
    top_p: float | None
    top_k: int | None
    max_new_tokens: int
    do_sample: bool
    repetition_penalty: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete benchmark suite configuration."""

    config_path: Path
    project_root: Path
    paths: BenchmarkPaths
    evaluation: BenchmarkEvaluationConfig
    generation: BenchmarkGenerationConfig
    device: str


@dataclass(frozen=True)
class CheckpointProfile:
    """Load-time, size, and memory profile for one checkpoint."""

    role: str
    path: str
    global_step: int
    epoch: int
    checkpoint_size_bytes: int
    load_seconds: float
    parameter_count: int
    parameter_memory_mb: float
    device_memory_mb: float
    rss_after_load_mb: float


@dataclass(frozen=True)
class ValidationResult:
    """Validation loss, perplexity, and next-token accuracy."""

    loss: float
    perplexity: float
    next_token_accuracy: float
    tokens: int
    batches: int
    tokens_per_second: float


@dataclass(frozen=True)
class LatencyResult:
    """Short-generation latency profile."""

    runs: int
    tokens_per_run: int
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    tokens_per_second: float
    per_run_seconds: tuple[float, ...]


@dataclass(frozen=True)
class PythonBenchmarkResult:
    """Aggregate Python-coding benchmark result."""

    prompt_count: int
    syntax_rate: float
    compile_rate: float
    execution_rate: float
    pass_rate: float
    average_response_chars: float
    average_generated_tokens: float
    tokens_per_second: float
    category_pass_rates: dict[str, float]


@dataclass(frozen=True)
class DocumentationQAResult:
    """Aggregate documentation QA result (keyword coverage scoring)."""

    question_count: int
    average_score: float
    source_scores: dict[str, float]
    tokens_per_second: float


@dataclass(frozen=True)
class TextGenerationResult:
    """Instruction-following and formatting quality result."""

    task_count: int
    instruction_following: float
    markdown_rate: float
    code_fence_rate: float
    formatting_score: float
    repetition_rate: float
    coherence_score: float
    average_response_chars: float


@dataclass(frozen=True)
class CheckpointBenchmark:
    """All benchmark results for one checkpoint."""

    profile: CheckpointProfile
    validation: ValidationResult
    latency: LatencyResult
    python_benchmark: PythonBenchmarkResult
    documentation_qa: DocumentationQAResult
    text_generation: TextGenerationResult


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Paths produced by a full benchmark run."""

    metrics_path: Path
    summary_path: Path
    comparison_path: Path
    plots_dir: Path
    overall_improvement_percent: float


def load_benchmark_config(path: Path | str = "configs/benchmark.yaml") -> BenchmarkConfig:
    """Load and validate the benchmark YAML configuration."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise BenchmarkError("Benchmark config must be a mapping.")
    root = _resolve(config_path.parent, raw.get("project_root", ".."))
    section = _mapping(raw.get("benchmark", {}), "benchmark")
    paths = _mapping(section.get("paths", {}), "benchmark.paths")
    evaluation = _mapping(section.get("evaluation", {}), "benchmark.evaluation")
    generation = _mapping(section.get("generation", {}), "benchmark.generation")
    metrics_files = tuple(
        _resolve(root, item)
        for item in paths.get("training_metrics", ["metrics/training_metrics.jsonl"])
    )
    config = BenchmarkConfig(
        config_path=config_path,
        project_root=root,
        paths=BenchmarkPaths(
            model_config=_resolve(root, paths.get("model_config", "configs/model.yaml")),
            training_config=_resolve(root, paths.get("training_config", "configs/training.yaml")),
            optimizer_config=_resolve(
                root, paths.get("optimizer_config", "configs/optimizer.yaml")
            ),
            generation_config=_resolve(
                root, paths.get("generation_config", "configs/generation.yaml")
            ),
            tokenizer=_resolve(root, paths.get("tokenizer", "data/tokenizer/tokenizer.json")),
            corpus_directory=_resolve(
                root, paths.get("corpus_directory", "python_corpus/final_corpus/packed")
            ),
            base_search_dir=_resolve(root, paths.get("base_search_dir", "checkpoints")),
            continued_search_dir=_resolve(
                root, paths.get("continued_search_dir", "checkpoints/continued_pretraining")
            ),
            output_dir=_resolve(root, paths.get("output_dir", "reports/benchmark")),
            training_metrics=metrics_files,
            continued_training_log=_resolve(
                root,
                paths.get(
                    "continued_training_log",
                    "reports/continued_pretraining/training_log.json",
                ),
            ),
        ),
        evaluation=BenchmarkEvaluationConfig(
            batch_size=int(evaluation.get("batch_size", 2)),
            validation_batches=int(evaluation.get("validation_batches", 20)),
            validation_fraction=float(evaluation.get("validation_fraction", 0.001)),
            seed=int(evaluation.get("seed", 42)),
            python_prompt_limit=_optional_int(evaluation.get("python_prompt_limit")),
            doc_qa_limit=_optional_int(evaluation.get("doc_qa_limit")),
            text_task_limit=_optional_int(evaluation.get("text_task_limit")),
            latency_runs=int(evaluation.get("latency_runs", 5)),
            latency_tokens=int(evaluation.get("latency_tokens", 16)),
            execution_timeout_seconds=float(evaluation.get("execution_timeout_seconds", 5.0)),
            run_generated_code=bool(evaluation.get("run_generated_code", True)),
        ),
        generation=BenchmarkGenerationConfig(
            temperature=float(generation.get("temperature", 0.7)),
            top_p=_optional_float(generation.get("top_p")),
            top_k=_optional_int(generation.get("top_k")),
            max_new_tokens=int(generation.get("max_new_tokens", 48)),
            do_sample=bool(generation.get("do_sample", False)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
        ),
        device=str(section.get("device", "auto")),
    )
    if config.evaluation.batch_size <= 0:
        raise BenchmarkError("benchmark.evaluation.batch_size must be positive.")
    if config.generation.max_new_tokens <= 0:
        raise BenchmarkError("benchmark.generation.max_new_tokens must be positive.")
    if config.evaluation.latency_runs <= 0 or config.evaluation.latency_tokens <= 0:
        raise BenchmarkError("latency_runs and latency_tokens must be positive.")
    return config


def resolve_benchmark_checkpoint(spec: str | Path, role: str, config: BenchmarkConfig) -> Path:
    """Resolve ``latest_base``/``latest`` keywords or explicit paths to a checkpoint."""

    text = str(spec).strip()
    if text == "latest_base":
        canonical = config.paths.base_search_dir / "last_checkpoint.pt"
        if canonical.is_file():
            return canonical.resolve()
        best = config.paths.base_search_dir / "best_model.pt"
        if best.is_file():
            return best.resolve()
        candidates = sorted(config.paths.base_search_dir.glob("step_*.pt"))
        if candidates:
            return candidates[-1].resolve()
        raise FileNotFoundError(f"No base checkpoint found in {config.paths.base_search_dir}")
    if text == "latest":
        directory = config.paths.continued_search_dir
        step_dirs: list[tuple[int, Path]] = []
        if directory.is_dir():
            for item in directory.iterdir():
                match = CHECKPOINT_DIR_PATTERN.fullmatch(item.name)
                if match is not None and (item / "model.pt").is_file():
                    step_dirs.append((int(match.group("step")), item / "model.pt"))
        if step_dirs:
            return max(step_dirs, key=lambda pair: pair[0])[1].resolve()
        canonical = directory / "last_checkpoint.pt"
        if canonical.is_file():
            return canonical.resolve()
        raise FileNotFoundError(f"No continued checkpoint found in {directory}")
    candidate = Path(spec)
    if not candidate.is_absolute():
        candidate = config.project_root / candidate
    if candidate.is_dir():
        candidate = candidate / "model.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"{role} checkpoint not found: {candidate}")
    return candidate.resolve()


def load_benchmark_model(
    config: BenchmarkConfig,
    checkpoint_path: Path,
    device: torch.device,
    role: str,
) -> tuple[torch.nn.Module, CodeTokenizer, CheckpointProfile]:
    """Load tokenizer, model, and checkpoint; profile size, time, and memory."""

    phase6 = load_phase6_config(
        config.paths.training_config,
        model_config=config.paths.model_config,
        optimizer_config=config.paths.optimizer_config,
        generation_config=config.paths.generation_config,
    )
    tokenizer = CodeTokenizer.from_file(config.paths.tokenizer)
    model = create_phase6_model(phase6.model, tokenizer)
    started = time.perf_counter()
    loaded = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        map_location="cpu",
        restore_rng=False,
    )
    model.to(device)
    model.eval()
    _synchronize(device)
    load_seconds = time.perf_counter() - started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_memory = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    profile = CheckpointProfile(
        role=role,
        path=str(checkpoint_path),
        global_step=loaded.global_step,
        epoch=loaded.epoch,
        checkpoint_size_bytes=checkpoint_path.stat().st_size,
        load_seconds=load_seconds,
        parameter_count=parameter_count,
        parameter_memory_mb=parameter_memory / (1024 * 1024),
        device_memory_mb=_device_memory_mb(device, parameter_memory),
        rss_after_load_mb=_rss_mb(),
    )
    return model, tokenizer, profile


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: BenchmarkConfig,
    device: torch.device,
) -> ValidationResult:
    """Validation loss, perplexity, and next-token top-1 accuracy on the Final Corpus."""

    dataset = PackedSequenceDataset(
        str(config.paths.corpus_directory / "*.bin"),
        tokenizer=tokenizer,
        manifest_path=config.paths.corpus_directory / "index.json",
        mmap=True,
    )
    indices = list(range(len(dataset)))
    random.Random(config.evaluation.seed).shuffle(indices)
    validation_count = max(1, int(len(indices) * config.evaluation.validation_fraction))
    subset = Subset(dataset, indices[:validation_count])
    loader = DataLoader(
        subset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=0,
    )
    loss_fn = GPTCrossEntropyLoss(padding_idx=tokenizer.pad_token_id, ignore_padding=True)
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    batches = 0
    started = time.perf_counter()
    for batch in loader:
        if batches >= config.evaluation.validation_batches:
            break
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, padding_mask=attention_mask)
        loss = loss_fn(logits, target_ids)
        predictions = logits.argmax(dim=-1)
        mask = attention_mask.bool()
        correct_tokens += int((predictions.eq(target_ids) & mask).sum().item())
        tokens = int(mask.sum().item())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
        batches += 1
    _synchronize(device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    if not total_tokens:
        raise BenchmarkError("Validation produced no tokens; check the corpus configuration.")
    loss_value = total_loss / total_tokens
    return ValidationResult(
        loss=loss_value,
        perplexity=math.exp(min(20.0, loss_value)),
        next_token_accuracy=correct_tokens / total_tokens,
        tokens=total_tokens,
        batches=batches,
        tokens_per_second=total_tokens / elapsed,
    )


@torch.no_grad()
def measure_generation_latency(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: BenchmarkConfig,
    device: torch.device,
) -> LatencyResult:
    """Repeated short greedy generations measuring per-request latency."""

    settings = CodeGenerationSettings(
        prompts=("def main():",),
        max_new_tokens=config.evaluation.latency_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=False,
        repetition_penalty=1.0,
        stop_tokens=(),
    )
    durations: list[float] = []
    generated_tokens = 0
    for _run in range(config.evaluation.latency_runs):
        _synchronize(device)
        started = time.perf_counter()
        result = generate_code_sample(
            model=model,
            tokenizer=tokenizer,
            prompt="def main():",
            device=device,
            context_length=model.context_length,
            settings=settings,
        )
        _synchronize(device)
        durations.append(time.perf_counter() - started)
        generated_tokens += len(result.generated_token_ids)
    total = sum(durations)
    return LatencyResult(
        runs=len(durations),
        tokens_per_run=config.evaluation.latency_tokens,
        mean_seconds=total / len(durations),
        min_seconds=min(durations),
        max_seconds=max(durations),
        tokens_per_second=generated_tokens / total if total > 0 else 0.0,
        per_run_seconds=tuple(durations),
    )


def evaluate_python_benchmark(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: BenchmarkConfig,
    device: torch.device,
    prompts: Sequence[PythonPrompt] | None = None,
) -> PythonBenchmarkResult:
    """Generate code for every Python prompt and score syntax/compile/execution."""

    selected = list(prompts if prompts is not None else PYTHON_BENCHMARK_PROMPTS)
    if config.evaluation.python_prompt_limit is not None:
        selected = selected[: config.evaluation.python_prompt_limit]
    if not selected:
        raise BenchmarkError("At least one Python benchmark prompt is required.")
    settings = _generation_settings(config)
    syntax_ok = 0
    compile_ok = 0
    execution_ok = 0
    response_chars = 0
    generated_tokens = 0
    total_seconds = 0.0
    category_totals: dict[str, list[int]] = {}
    for item in selected:
        answer, tokens, seconds = _generate(model, tokenizer, item.prompt, device, settings)
        generated_tokens += tokens
        total_seconds += seconds
        response_chars += len(answer)
        code = extract_python_code(answer)
        item_syntax = bool(code) and _parses(code)
        item_compile = item_syntax and _compiles(code)
        item_execute = (
            item_compile
            and config.evaluation.run_generated_code
            and _executes(code, config.evaluation.execution_timeout_seconds)
        )
        syntax_ok += int(item_syntax)
        compile_ok += int(item_compile)
        execution_ok += int(item_execute)
        bucket = category_totals.setdefault(item.category, [0, 0])
        bucket[0] += int(item_syntax)
        bucket[1] += 1
    count = len(selected)
    return PythonBenchmarkResult(
        prompt_count=count,
        syntax_rate=syntax_ok / count,
        compile_rate=compile_ok / count,
        execution_rate=execution_ok / count,
        pass_rate=syntax_ok / count,
        average_response_chars=response_chars / count,
        average_generated_tokens=generated_tokens / count,
        tokens_per_second=generated_tokens / total_seconds if total_seconds > 0 else 0.0,
        category_pass_rates={
            category: passed / total
            for category, (passed, total) in sorted(category_totals.items())
        },
    )


def evaluate_documentation_qa(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: BenchmarkConfig,
    device: torch.device,
    questions: Sequence[DocumentationQuestion] | None = None,
) -> DocumentationQAResult:
    """Answer documentation questions and score expected-keyword coverage."""

    selected = list(questions if questions is not None else DOCUMENTATION_QA)
    if config.evaluation.doc_qa_limit is not None:
        selected = selected[: config.evaluation.doc_qa_limit]
    if not selected:
        raise BenchmarkError("At least one documentation question is required.")
    settings = _generation_settings(config)
    total_score = 0.0
    generated_tokens = 0
    total_seconds = 0.0
    source_totals: dict[str, list[float]] = {}
    for item in selected:
        answer, tokens, seconds = _generate(model, tokenizer, item.question, device, settings)
        generated_tokens += tokens
        total_seconds += seconds
        score = keyword_coverage(answer, item.keywords)
        total_score += score
        bucket = source_totals.setdefault(item.source, [0.0, 0.0])
        bucket[0] += score
        bucket[1] += 1.0
    count = len(selected)
    return DocumentationQAResult(
        question_count=count,
        average_score=total_score / count,
        source_scores={
            source: scored / total for source, (scored, total) in sorted(source_totals.items())
        },
        tokens_per_second=generated_tokens / total_seconds if total_seconds > 0 else 0.0,
    )


def evaluate_text_generation(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: BenchmarkConfig,
    device: torch.device,
    tasks: Sequence[TextGenerationTask] | None = None,
) -> TextGenerationResult:
    """Score instruction following, Markdown formatting, repetition, and coherence."""

    selected = list(tasks if tasks is not None else TEXT_GENERATION_TASKS)
    if config.evaluation.text_task_limit is not None:
        selected = selected[: config.evaluation.text_task_limit]
    if not selected:
        raise BenchmarkError("At least one text-generation task is required.")
    settings = _generation_settings(config)
    following_total = 0.0
    markdown_wanted = markdown_present = 0
    fence_wanted = fence_present = 0
    repetition_total = 0.0
    coherence_total = 0.0
    response_chars = 0
    token_ids_by_task: list[list[int]] = []
    for task in selected:
        answer, _tokens, _seconds = _generate(
            model, tokenizer, task.instruction, device, settings, collect_ids=token_ids_by_task
        )
        response_chars += len(answer)
        following_total += keyword_coverage(answer, task.required_terms)
        if task.wants_markdown:
            markdown_wanted += 1
            markdown_present += int(any(marker in answer for marker in MARKDOWN_MARKERS))
        if task.wants_code_fence:
            fence_wanted += 1
            fence_present += int("```" in answer)
        ids = token_ids_by_task[-1] if token_ids_by_task else []
        repetition_total += repetition_rate(ids)
        coherence_total += coherence_score(ids)
    count = len(selected)
    markdown_rate = markdown_present / markdown_wanted if markdown_wanted else 1.0
    fence_rate = fence_present / fence_wanted if fence_wanted else 1.0
    return TextGenerationResult(
        task_count=count,
        instruction_following=following_total / count,
        markdown_rate=markdown_rate,
        code_fence_rate=fence_rate,
        formatting_score=(markdown_rate + fence_rate) / 2,
        repetition_rate=repetition_total / count,
        coherence_score=coherence_total / count,
        average_response_chars=response_chars / count,
    )


def keyword_coverage(answer: str, keywords: Sequence[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""

    if not keywords:
        return 1.0
    normalized = answer.casefold()
    return sum(1 for keyword in keywords if keyword.casefold() in normalized) / len(keywords)


def repetition_rate(token_ids: Sequence[int]) -> float:
    """1 - unique/total token ratio; higher means more repetitive output."""

    if not token_ids:
        return 0.0
    return 1.0 - len(set(token_ids)) / len(token_ids)


def coherence_score(token_ids: Sequence[int]) -> float:
    """Distinct-bigram ratio as a lightweight coherence/degeneracy proxy."""

    if len(token_ids) < 2:
        return 0.0
    bigrams = list(zip(token_ids, token_ids[1:]))
    return len(set(bigrams)) / len(bigrams)


def run_checkpoint_benchmark(
    config: BenchmarkConfig,
    checkpoint_path: Path,
    role: str,
    device: torch.device,
) -> CheckpointBenchmark:
    """Run every evaluation for one checkpoint and free the model afterwards."""

    LOGGER.info("benchmark_checkpoint_started role=%s path=%s", role, checkpoint_path)
    model, tokenizer, profile = load_benchmark_model(config, checkpoint_path, device, role)
    try:
        validation = evaluate_validation(model, tokenizer, config, device)
        LOGGER.info(
            "benchmark_validation role=%s loss=%.6f perplexity=%.3f accuracy=%.4f",
            role,
            validation.loss,
            validation.perplexity,
            validation.next_token_accuracy,
        )
        latency = measure_generation_latency(model, tokenizer, config, device)
        python_benchmark = evaluate_python_benchmark(model, tokenizer, config, device)
        LOGGER.info(
            "benchmark_python role=%s prompts=%d syntax=%.3f execution=%.3f",
            role,
            python_benchmark.prompt_count,
            python_benchmark.syntax_rate,
            python_benchmark.execution_rate,
        )
        documentation_qa = evaluate_documentation_qa(model, tokenizer, config, device)
        text_generation = evaluate_text_generation(model, tokenizer, config, device)
    finally:
        del model
        gc.collect()
        _empty_device_cache(device)
    LOGGER.info("benchmark_checkpoint_completed role=%s", role)
    return CheckpointBenchmark(
        profile=profile,
        validation=validation,
        latency=latency,
        python_benchmark=python_benchmark,
        documentation_qa=documentation_qa,
        text_generation=text_generation,
    )


def build_comparison(
    base: CheckpointBenchmark,
    continued: CheckpointBenchmark,
) -> dict[str, Any]:
    """Build base-vs-continued deltas and the overall improvement figure."""

    loss_improvement = _relative_improvement(
        base.validation.loss, continued.validation.loss, lower_is_better=True
    )
    perplexity_improvement = _relative_improvement(
        base.validation.perplexity, continued.validation.perplexity, lower_is_better=True
    )
    accuracy_delta = (
        continued.validation.next_token_accuracy - base.validation.next_token_accuracy
    ) * 100
    python_delta = (continued.python_benchmark.pass_rate - base.python_benchmark.pass_rate) * 100
    doc_delta = (
        continued.documentation_qa.average_score - base.documentation_qa.average_score
    ) * 100
    instruction_delta = (
        continued.text_generation.instruction_following - base.text_generation.instruction_following
    ) * 100
    speed_improvement = _relative_improvement(
        base.latency.tokens_per_second, continued.latency.tokens_per_second, lower_is_better=False
    )
    latency_improvement = _relative_improvement(
        base.latency.mean_seconds, continued.latency.mean_seconds, lower_is_better=True
    )
    components = {
        "validation_loss_improvement_percent": loss_improvement,
        "perplexity_improvement_percent": perplexity_improvement,
        "next_token_accuracy_delta_pp": accuracy_delta,
        "python_pass_rate_delta_pp": python_delta,
        "documentation_qa_delta_pp": doc_delta,
        "instruction_following_delta_pp": instruction_delta,
    }
    overall = sum(components.values()) / len(components)
    return {
        "validation_loss": {
            "base": base.validation.loss,
            "continued": continued.validation.loss,
            "delta": continued.validation.loss - base.validation.loss,
            "improvement_percent": loss_improvement,
        },
        "perplexity": {
            "base": base.validation.perplexity,
            "continued": continued.validation.perplexity,
            "delta": continued.validation.perplexity - base.validation.perplexity,
            "improvement_percent": perplexity_improvement,
        },
        "next_token_accuracy": {
            "base": base.validation.next_token_accuracy,
            "continued": continued.validation.next_token_accuracy,
            "delta_pp": accuracy_delta,
        },
        "speed": {
            "base_generation_tokens_per_second": base.latency.tokens_per_second,
            "continued_generation_tokens_per_second": continued.latency.tokens_per_second,
            "base_validation_tokens_per_second": base.validation.tokens_per_second,
            "continued_validation_tokens_per_second": continued.validation.tokens_per_second,
            "improvement_percent": speed_improvement,
        },
        "latency": {
            "base_mean_seconds": base.latency.mean_seconds,
            "continued_mean_seconds": continued.latency.mean_seconds,
            "improvement_percent": latency_improvement,
        },
        "memory": {
            "base_parameter_memory_mb": base.profile.parameter_memory_mb,
            "continued_parameter_memory_mb": continued.profile.parameter_memory_mb,
            "base_device_memory_mb": base.profile.device_memory_mb,
            "continued_device_memory_mb": continued.profile.device_memory_mb,
            "base_rss_mb": base.profile.rss_after_load_mb,
            "continued_rss_mb": continued.profile.rss_after_load_mb,
        },
        "checkpoint_size": {
            "base_bytes": base.profile.checkpoint_size_bytes,
            "continued_bytes": continued.profile.checkpoint_size_bytes,
            "base_human": format_size(base.profile.checkpoint_size_bytes),
            "continued_human": format_size(continued.profile.checkpoint_size_bytes),
        },
        "load_time": {
            "base_seconds": base.profile.load_seconds,
            "continued_seconds": continued.profile.load_seconds,
        },
        "python_benchmark": {
            "base_pass_rate": base.python_benchmark.pass_rate,
            "continued_pass_rate": continued.python_benchmark.pass_rate,
            "base_execution_rate": base.python_benchmark.execution_rate,
            "continued_execution_rate": continued.python_benchmark.execution_rate,
            "delta_pp": python_delta,
        },
        "documentation_qa": {
            "base_score": base.documentation_qa.average_score,
            "continued_score": continued.documentation_qa.average_score,
            "delta_pp": doc_delta,
        },
        "text_generation": {
            "base_instruction_following": base.text_generation.instruction_following,
            "continued_instruction_following": continued.text_generation.instruction_following,
            "base_repetition_rate": base.text_generation.repetition_rate,
            "continued_repetition_rate": continued.text_generation.repetition_rate,
            "delta_pp": instruction_delta,
        },
        "overall_improvement_components": components,
        "overall_improvement_percent": overall,
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    base: str | Path = "latest_base",
    continued: str | Path = "latest",
) -> BenchmarkRunResult:
    """Run the full benchmark for both checkpoints and write all reports."""

    device = select_device(config.device)
    LOGGER.info("benchmark_started device=%s", device)
    base_path = resolve_benchmark_checkpoint(base, "base", config)
    continued_path = resolve_benchmark_checkpoint(continued, "continued", config)
    if base_path == continued_path:
        raise BenchmarkError("Base and continued checkpoints resolve to the same file.")
    base_result = run_checkpoint_benchmark(config, base_path, "base", device)
    continued_result = run_checkpoint_benchmark(config, continued_path, "continued", device)
    comparison = build_comparison(base_result, continued_result)
    output_dir = config.paths.output_dir
    plots_dir = output_dir / "plots"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    comparison_path = output_dir / "comparison.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "device": str(device),
        "config": str(config.config_path),
        "generation": asdict(config.generation),
        "base": _benchmark_payload(base_result),
        "continued": _benchmark_payload(continued_result),
        "comparison": comparison,
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(_summary_markdown(base_result, continued_result, comparison))
    comparison_path.write_text(_comparison_markdown(base_result, continued_result, comparison))
    render_benchmark_plots(config, base_result, continued_result, plots_dir)
    LOGGER.info(
        "benchmark_completed overall_improvement=%.3f%% output=%s",
        comparison["overall_improvement_percent"],
        output_dir,
    )
    return BenchmarkRunResult(
        metrics_path=metrics_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
        plots_dir=plots_dir,
        overall_improvement_percent=comparison["overall_improvement_percent"],
    )


def render_benchmark_plots(
    config: BenchmarkConfig,
    base: CheckpointBenchmark,
    continued: CheckpointBenchmark,
    plots_dir: Path,
) -> None:
    """Render loss, perplexity, speed, memory, and latency PNGs plus a legend file."""

    plots_dir.mkdir(parents=True, exist_ok=True)
    train_series, validation_series = _training_history(config)
    loss_points = validation_series or train_series
    _write_series_png(
        plots_dir / "loss.png",
        [((36, 112, 194), train_series), ((210, 86, 59), validation_series)],
    )
    perplexity_series = [
        (step, math.exp(min(20.0, value))) for step, value in loss_points
    ]
    validation_bars = [
        ("base", base.validation.perplexity),
        ("continued", continued.validation.perplexity),
    ]
    _write_series_png(
        plots_dir / "perplexity.png",
        [((36, 112, 194), perplexity_series)],
        bars=validation_bars,
    )
    _write_bars_png(
        plots_dir / "speed.png",
        [
            ("base generation tok/s", base.latency.tokens_per_second, (36, 112, 194)),
            ("continued generation tok/s", continued.latency.tokens_per_second, (46, 158, 96)),
            ("base validation tok/s", base.validation.tokens_per_second, (120, 160, 210)),
            ("continued validation tok/s", continued.validation.tokens_per_second, (130, 200, 160)),
        ],
    )
    _write_bars_png(
        plots_dir / "memory.png",
        [
            ("base checkpoint MB", base.profile.checkpoint_size_bytes / 1e6, (36, 112, 194)),
            (
                "continued checkpoint MB",
                continued.profile.checkpoint_size_bytes / 1e6,
                (46, 158, 96),
            ),
            ("base parameter MB", base.profile.parameter_memory_mb, (120, 160, 210)),
            ("continued parameter MB", continued.profile.parameter_memory_mb, (130, 200, 160)),
        ],
    )
    _write_series_png(
        plots_dir / "latency.png",
        [
            (
                (36, 112, 194),
                [(index, value) for index, value in enumerate(base.latency.per_run_seconds, 1)],
            ),
            (
                (46, 158, 96),
                [
                    (index, value)
                    for index, value in enumerate(continued.latency.per_run_seconds, 1)
                ],
            ),
        ],
    )
    legend = {
        "loss.png": "Training loss (blue) and validation loss (red) over global steps.",
        "perplexity.png": (
            "Perplexity from the discovered loss history (blue line) with final "
            "base/continued validation perplexity bars."
        ),
        "speed.png": (
            "Bars, left to right: base generation tok/s, continued generation tok/s, "
            "base validation tok/s, continued validation tok/s."
        ),
        "memory.png": (
            "Bars, left to right: base checkpoint MB, continued checkpoint MB, "
            "base parameter MB, continued parameter MB."
        ),
        "latency.png": (
            "Per-run generation latency in seconds: base (blue) vs continued (green)."
        ),
        "note": "PNGs are rendered with the dependency-free canvas and carry no text labels.",
    }
    (plots_dir / "plots.json").write_text(
        json.dumps(legend, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_benchmark_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the benchmark suite."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Benchmark GenPy checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--base", default="latest_base")
    parser.add_argument("--continued", default="latest")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args(argv)
    try:
        config = load_benchmark_config(args.config)
        if args.device is not None:
            config = replace(config, device=args.device)
        if args.max_new_tokens is not None:
            config = replace(
                config,
                generation=replace(config.generation, max_new_tokens=args.max_new_tokens),
            )
        result = run_benchmark(config, base=args.base, continued=args.continued)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Benchmark complete")
    print(f"Overall improvement: {result.overall_improvement_percent:+.3f}%")
    print(f"Metrics: {result.metrics_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Comparison: {result.comparison_path}")
    print(f"Plots: {result.plots_dir}")
    return 0


def _generation_settings(config: BenchmarkConfig) -> CodeGenerationSettings:
    return CodeGenerationSettings(
        prompts=(),
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        do_sample=config.generation.do_sample,
        repetition_penalty=config.generation.repetition_penalty,
        stop_tokens=("<eos>",),
    )


@torch.no_grad()
def _generate(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    prompt: str,
    device: torch.device,
    settings: CodeGenerationSettings,
    collect_ids: list[list[int]] | None = None,
) -> tuple[str, int, float]:
    _synchronize(device)
    started = time.perf_counter()
    result = generate_code_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        context_length=model.context_length,
        settings=settings,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if collect_ids is not None:
        collect_ids.append(list(result.generated_token_ids))
    answer = tokenizer.decode(result.generated_token_ids, skip_special_tokens=True).strip()
    return answer, len(result.generated_token_ids), elapsed


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _compiles(code: str) -> bool:
    try:
        compile(code, "<benchmark>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _executes(code: str, timeout_seconds: float) -> bool:
    with tempfile.TemporaryDirectory(prefix="genpy_benchmark_") as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=workdir,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
    return completed.returncode == 0


def _training_history(
    config: BenchmarkConfig,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train: list[tuple[int, float]] = []
    validation: list[tuple[int, float]] = []
    for path in config.paths.training_metrics:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            step = record.get("step")
            if not isinstance(step, int):
                continue
            if record.get("type") == "train" and isinstance(record.get("loss"), (int, float)):
                train.append((step, float(record["loss"])))
            if record.get("type") == "validation" and isinstance(
                record.get("validation_loss"), (int, float)
            ):
                validation.append((step, float(record["validation_loss"])))
    log_path = config.paths.continued_training_log
    if log_path.is_file():
        try:
            records = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            records = []
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get("step"), int):
                    continue
                if isinstance(record.get("training_loss"), (int, float)):
                    train.append((record["step"], float(record["training_loss"])))
                if isinstance(record.get("validation_loss"), (int, float)):
                    validation.append((record["step"], float(record["validation_loss"])))
    return sorted(train), sorted(validation)


def _write_series_png(
    output_path: Path,
    series: list[tuple[tuple[int, int, int], list[tuple[int, float]]]],
    *,
    bars: list[tuple[str, float]] | None = None,
) -> None:
    width, height = 900, 520
    canvas = _Canvas(width, height, background=(255, 255, 255))
    margin_left, margin_right, margin_top, margin_bottom = 70, 28, 36, 62
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    canvas.line(margin_left, margin_top, margin_left, margin_top + plot_height, (40, 40, 40))
    canvas.line(
        margin_left,
        margin_top + plot_height,
        margin_left + plot_width,
        margin_top + plot_height,
        (40, 40, 40),
    )
    for fraction in (0.25, 0.5, 0.75):
        y = int(margin_top + plot_height * fraction)
        canvas.line(margin_left, y, margin_left + plot_width, y, (230, 230, 230))
    all_points = [point for _color, points in series for point in points]
    if all_points:
        min_step = min(step for step, _value in all_points)
        max_step = max(step for step, _value in all_points)
        min_value = min(value for _step, value in all_points)
        max_value = max(value for _step, value in all_points)
        for color, points in series:
            _plot_series(
                canvas,
                points,
                bounds=(min_step, max_step, min_value, max_value),
                area=(margin_left, margin_top, plot_width, plot_height),
                color=color,
            )
    if bars:
        _draw_bars(
            canvas,
            [(label, value, (46, 158, 96)) for label, value in bars],
            area=(margin_left, margin_top, plot_width, plot_height),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canvas.to_png_bytes())


def _write_bars_png(
    output_path: Path,
    bars: list[tuple[str, float, tuple[int, int, int]]],
) -> None:
    width, height = 900, 520
    canvas = _Canvas(width, height, background=(255, 255, 255))
    margin_left, margin_right, margin_top, margin_bottom = 70, 28, 36, 62
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    canvas.line(margin_left, margin_top, margin_left, margin_top + plot_height, (40, 40, 40))
    canvas.line(
        margin_left,
        margin_top + plot_height,
        margin_left + plot_width,
        margin_top + plot_height,
        (40, 40, 40),
    )
    _draw_bars(canvas, bars, area=(margin_left, margin_top, plot_width, plot_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canvas.to_png_bytes())


def _draw_bars(
    canvas: _Canvas,
    bars: list[tuple[str, float, tuple[int, int, int]]],
    *,
    area: tuple[int, int, int, int],
) -> None:
    if not bars:
        return
    x0, y0, width, height = area
    max_value = max((value for _label, value, _color in bars), default=0.0)
    if max_value <= 0:
        max_value = 1.0
    slot = width // max(1, len(bars))
    bar_width = max(8, int(slot * 0.5))
    for index, (_label, value, color) in enumerate(bars):
        bar_height = int(height * min(1.0, max(0.0, value / max_value)))
        left = x0 + index * slot + (slot - bar_width) // 2
        bottom = y0 + height
        top = bottom - max(2, bar_height)
        for x in range(left, left + bar_width):
            canvas.line(x, top, x, bottom, color)


def _benchmark_payload(result: CheckpointBenchmark) -> dict[str, Any]:
    return {
        "profile": asdict(result.profile),
        "validation": asdict(result.validation),
        "latency": asdict(result.latency),
        "python_benchmark": asdict(result.python_benchmark),
        "documentation_qa": asdict(result.documentation_qa),
        "text_generation": asdict(result.text_generation),
    }


def _summary_markdown(
    base: CheckpointBenchmark,
    continued: CheckpointBenchmark,
    comparison: dict[str, Any],
) -> str:
    lines = [
        "# GenPy Benchmark Summary",
        "",
        f"- Generated at: {datetime.now(UTC).isoformat()}",
        f"- Base checkpoint: `{base.profile.path}` (step {base.profile.global_step})",
        (
            f"- Continued checkpoint: `{continued.profile.path}` "
            f"(step {continued.profile.global_step})"
        ),
        "",
        "| Metric | Base | Continued |",
        "| --- | --- | --- |",
        f"| Validation loss | {base.validation.loss:.6f} | {continued.validation.loss:.6f} |",
        (
            f"| Perplexity | {base.validation.perplexity:.3f} "
            f"| {continued.validation.perplexity:.3f} |"
        ),
        (
            f"| Next-token accuracy | {base.validation.next_token_accuracy:.4f} "
            f"| {continued.validation.next_token_accuracy:.4f} |"
        ),
        (
            f"| Generation speed (tok/s) | {base.latency.tokens_per_second:.2f} "
            f"| {continued.latency.tokens_per_second:.2f} |"
        ),
        (
            f"| Validation throughput (tok/s) | {base.validation.tokens_per_second:.1f} "
            f"| {continued.validation.tokens_per_second:.1f} |"
        ),
        (
            f"| Mean generation latency (s) | {base.latency.mean_seconds:.4f} "
            f"| {continued.latency.mean_seconds:.4f} |"
        ),
        (
            f"| Parameter memory | {base.profile.parameter_memory_mb:.1f} MB "
            f"| {continued.profile.parameter_memory_mb:.1f} MB |"
        ),
        (
            f"| Checkpoint size | {format_size(base.profile.checkpoint_size_bytes)} "
            f"| {format_size(continued.profile.checkpoint_size_bytes)} |"
        ),
        (
            f"| Checkpoint load time (s) | {base.profile.load_seconds:.2f} "
            f"| {continued.profile.load_seconds:.2f} |"
        ),
        (
            f"| Python benchmark pass rate | {base.python_benchmark.pass_rate:.3f} "
            f"| {continued.python_benchmark.pass_rate:.3f} |"
        ),
        (
            f"| Python execution rate | {base.python_benchmark.execution_rate:.3f} "
            f"| {continued.python_benchmark.execution_rate:.3f} |"
        ),
        (
            f"| Documentation QA score | {base.documentation_qa.average_score:.3f} "
            f"| {continued.documentation_qa.average_score:.3f} |"
        ),
        (
            f"| Instruction following | {base.text_generation.instruction_following:.3f} "
            f"| {continued.text_generation.instruction_following:.3f} |"
        ),
        (
            f"| Repetition rate | {base.text_generation.repetition_rate:.3f} "
            f"| {continued.text_generation.repetition_rate:.3f} |"
        ),
        "",
        f"**Overall improvement: {comparison['overall_improvement_percent']:+.3f}%**",
        "",
        (
            "Overall improvement averages relative loss/perplexity improvements with "
            "percentage-point deltas for next-token accuracy, Python pass rate, "
            "documentation QA, and instruction following."
        ),
        "",
        "Plots: `plots/loss.png`, `plots/perplexity.png`, `plots/speed.png`, "
        "`plots/memory.png`, `plots/latency.png` (series legend in `plots/plots.json`).",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _comparison_markdown(
    base: CheckpointBenchmark,
    continued: CheckpointBenchmark,
    comparison: dict[str, Any],
) -> str:
    def row(metric: str, base_value: str, continued_value: str, delta: str) -> str:
        return f"| {metric} | {base_value} | {continued_value} | {delta} |"

    validation = comparison["validation_loss"]
    perplexity = comparison["perplexity"]
    speed = comparison["speed"]
    latency = comparison["latency"]
    python_cmp = comparison["python_benchmark"]
    doc_cmp = comparison["documentation_qa"]
    text_cmp = comparison["text_generation"]
    lines = [
        "# Base vs Continued Checkpoint Comparison",
        "",
        f"- Base: `{base.profile.path}` (step {base.profile.global_step})",
        f"- Continued: `{continued.profile.path}` (step {continued.profile.global_step})",
        "",
        "| Metric | Base | Continued | Delta |",
        "| --- | --- | --- | --- |",
        row(
            "Validation loss",
            f"{validation['base']:.6f}",
            f"{validation['continued']:.6f}",
            f"{validation['delta']:+.6f} ({validation['improvement_percent']:+.3f}%)",
        ),
        row(
            "Perplexity",
            f"{perplexity['base']:.3f}",
            f"{perplexity['continued']:.3f}",
            f"{perplexity['delta']:+.3f} ({perplexity['improvement_percent']:+.3f}%)",
        ),
        row(
            "Next-token accuracy",
            f"{base.validation.next_token_accuracy:.4f}",
            f"{continued.validation.next_token_accuracy:.4f}",
            f"{comparison['next_token_accuracy']['delta_pp']:+.2f} pp",
        ),
        row(
            "Generation speed (tok/s)",
            f"{speed['base_generation_tokens_per_second']:.2f}",
            f"{speed['continued_generation_tokens_per_second']:.2f}",
            f"{speed['improvement_percent']:+.3f}%",
        ),
        row(
            "Mean latency (s)",
            f"{latency['base_mean_seconds']:.4f}",
            f"{latency['continued_mean_seconds']:.4f}",
            f"{latency['improvement_percent']:+.3f}%",
        ),
        row(
            "Parameter memory (MB)",
            f"{base.profile.parameter_memory_mb:.1f}",
            f"{continued.profile.parameter_memory_mb:.1f}",
            "n/a",
        ),
        row(
            "Checkpoint size",
            comparison["checkpoint_size"]["base_human"],
            comparison["checkpoint_size"]["continued_human"],
            "n/a",
        ),
        row(
            "Load time (s)",
            f"{comparison['load_time']['base_seconds']:.2f}",
            f"{comparison['load_time']['continued_seconds']:.2f}",
            "n/a",
        ),
        row(
            "Python pass rate",
            f"{python_cmp['base_pass_rate']:.3f}",
            f"{python_cmp['continued_pass_rate']:.3f}",
            f"{python_cmp['delta_pp']:+.2f} pp",
        ),
        row(
            "Python execution rate",
            f"{python_cmp['base_execution_rate']:.3f}",
            f"{python_cmp['continued_execution_rate']:.3f}",
            "n/a",
        ),
        row(
            "Documentation QA score",
            f"{doc_cmp['base_score']:.3f}",
            f"{doc_cmp['continued_score']:.3f}",
            f"{doc_cmp['delta_pp']:+.2f} pp",
        ),
        row(
            "Instruction following",
            f"{text_cmp['base_instruction_following']:.3f}",
            f"{text_cmp['continued_instruction_following']:.3f}",
            f"{text_cmp['delta_pp']:+.2f} pp",
        ),
        row(
            "Repetition rate",
            f"{text_cmp['base_repetition_rate']:.3f}",
            f"{text_cmp['continued_repetition_rate']:.3f}",
            "lower is better",
        ),
        "",
        "## Python benchmark pass rate by category",
        "",
        "| Category | Base | Continued |",
        "| --- | --- | --- |",
    ]
    categories = sorted(
        set(base.python_benchmark.category_pass_rates)
        | set(continued.python_benchmark.category_pass_rates)
    )
    for category in categories:
        base_rate = base.python_benchmark.category_pass_rates.get(category, 0.0)
        continued_rate = continued.python_benchmark.category_pass_rates.get(category, 0.0)
        lines.append(f"| {category} | {base_rate:.3f} | {continued_rate:.3f} |")
    lines.extend(
        [
            "",
            "## Documentation QA score by source",
            "",
            "| Source | Base | Continued |",
            "| --- | --- | --- |",
        ]
    )
    sources = sorted(
        set(base.documentation_qa.source_scores) | set(continued.documentation_qa.source_scores)
    )
    for source in sources:
        base_score = base.documentation_qa.source_scores.get(source, 0.0)
        continued_score = continued.documentation_qa.source_scores.get(source, 0.0)
        lines.append(f"| {source} | {base_score:.3f} | {continued_score:.3f} |")
    lines.extend(
        [
            "",
            f"**Overall improvement: {comparison['overall_improvement_percent']:+.3f}%**",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _relative_improvement(base: float, continued: float, *, lower_is_better: bool) -> float:
    if base == 0:
        return 0.0
    change = (continued - base) / abs(base) * 100
    return -change if lower_is_better else change


def _device_memory_mb(device: torch.device, parameter_memory_bytes: int) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    if device.type == "mps" and hasattr(torch, "mps"):
        current = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current):
            return current() / (1024 * 1024)
    return parameter_memory_bytes / (1024 * 1024)


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return usage / divisor


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        empty = getattr(torch.mps, "empty_cache", None)
        if callable(empty):
            empty()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BenchmarkError(f"{name} must be a mapping.")
    return value


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise BenchmarkError("Configured path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


__all__ = [
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkRunResult",
    "CheckpointBenchmark",
    "CheckpointProfile",
    "DocumentationQAResult",
    "LatencyResult",
    "PythonBenchmarkResult",
    "TextGenerationResult",
    "ValidationResult",
    "build_comparison",
    "coherence_score",
    "evaluate_documentation_qa",
    "evaluate_python_benchmark",
    "evaluate_text_generation",
    "evaluate_validation",
    "keyword_coverage",
    "load_benchmark_config",
    "load_benchmark_model",
    "measure_generation_latency",
    "render_benchmark_plots",
    "repetition_rate",
    "resolve_benchmark_checkpoint",
    "run_benchmark",
    "run_benchmark_cli",
    "run_checkpoint_benchmark",
]
