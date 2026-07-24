"""Phase 6.3 checkpoint benchmark comparison."""

from __future__ import annotations

import ast
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from genpy_llm.checkpointing import load_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.losses import GPTCrossEntropyLoss
from genpy_llm.pretraining import Phase6Config, create_phase6_model
from genpy_llm.pretraining_dataset import PackedSequenceDataset
from genpy_llm.pretraining_generation import generate_code_sample

UTC = timezone.utc


@dataclass(frozen=True)
class BenchmarkSettings:
    """Phase 6.3 benchmark settings."""

    enabled: bool = True
    validation_batches: int = 1
    prompt_count: int = 3
    max_new_tokens: int = 16


@dataclass(frozen=True)
class CheckpointBenchmark:
    """Benchmark result for one checkpoint."""

    checkpoint: str
    validation_loss: float | None
    perplexity: float | None
    validation_tokens: int
    syntax_correctness: float
    python_benchmark_score: float
    repetition_rate: float
    generation_tokens_per_second: float


@dataclass(frozen=True)
class BenchmarkComparison:
    """Before/after Phase 6.3 benchmark comparison."""

    previous: CheckpointBenchmark
    continued: CheckpointBenchmark
    validation_loss_delta: float | None
    perplexity_delta: float | None
    generated_at: str


def benchmark_phase63_checkpoints(
    *,
    config: Phase6Config,
    previous_checkpoint: Path,
    continued_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    settings: BenchmarkSettings,
) -> BenchmarkComparison:
    """Benchmark previous and continued checkpoints and write comparison reports."""

    previous = _benchmark_checkpoint(
        config=config,
        checkpoint=previous_checkpoint,
        device=device,
        settings=settings,
    )
    continued = _benchmark_checkpoint(
        config=config,
        checkpoint=continued_checkpoint,
        device=device,
        settings=settings,
    )
    comparison = BenchmarkComparison(
        previous=previous,
        continued=continued,
        validation_loss_delta=_delta(previous.validation_loss, continued.validation_loss),
        perplexity_delta=_delta(previous.perplexity, continued.perplexity),
        generated_at=datetime.now(UTC).isoformat(),
    )
    write_benchmark_comparison(comparison, output_dir)
    return comparison


def write_benchmark_comparison(comparison: BenchmarkComparison, output_dir: Path) -> None:
    """Write JSON and Markdown benchmark comparison artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(comparison)
    (output_dir / "comparison_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison_report.md").write_text(
        _markdown(comparison),
        encoding="utf-8",
    )


def _benchmark_checkpoint(
    *,
    config: Phase6Config,
    checkpoint: Path,
    device: torch.device,
    settings: BenchmarkSettings,
) -> CheckpointBenchmark:
    tokenizer = CodeTokenizer.from_file(config.data.tokenizer)
    model = create_phase6_model(config.model, tokenizer)
    load_checkpoint(checkpoint, model, optimizer=None, map_location="cpu", restore_rng=False)
    model.to(device)
    model.eval()
    validation_loss, validation_tokens = _validation_loss(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        max_batches=settings.validation_batches,
    )
    generation = _generation_checks(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        max_new_tokens=settings.max_new_tokens,
        prompt_count=settings.prompt_count,
    )
    return CheckpointBenchmark(
        checkpoint=str(checkpoint.resolve()),
        validation_loss=validation_loss,
        perplexity=math.exp(min(20.0, validation_loss)) if validation_loss is not None else None,
        validation_tokens=validation_tokens,
        syntax_correctness=generation["syntax_correctness"],
        python_benchmark_score=generation["python_benchmark_score"],
        repetition_rate=generation["repetition_rate"],
        generation_tokens_per_second=generation["generation_tokens_per_second"],
    )


@torch.no_grad()
def _validation_loss(
    *,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: Phase6Config,
    device: torch.device,
    max_batches: int,
) -> tuple[float | None, int]:
    dataset = PackedSequenceDataset(
        config.data.shard_pattern,
        tokenizer=tokenizer,
        manifest_path=config.data.shard_index,
        sequence_length=config.model.context_length + 1,
        mmap=config.data.mmap,
    )
    if len(dataset) == 0:
        return None, 0
    validation_count = max(1, int(len(dataset) * config.data.validation_fraction))
    subset = Subset(dataset, list(range(min(validation_count, len(dataset)))))
    loader = DataLoader(subset, batch_size=config.data.batch_size, shuffle=False, num_workers=0)
    loss_fn = GPTCrossEntropyLoss(padding_idx=tokenizer.pad_token_id, ignore_padding=True)
    total_loss = 0.0
    total_tokens = 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, padding_mask=attention_mask)
        loss = loss_fn(logits, target_ids)
        tokens = int(attention_mask.sum().item())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    return (total_loss / total_tokens if total_tokens else None, total_tokens)


@torch.no_grad()
def _generation_checks(
    *,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: Phase6Config,
    device: torch.device,
    max_new_tokens: int,
    prompt_count: int,
) -> dict[str, float]:
    prompts = config.generation.prompts[: max(1, prompt_count)]
    generation_settings = replace(config.generation, max_new_tokens=max_new_tokens)
    syntax_passes = 0
    repetition_rates: list[float] = []
    generated_tokens = 0
    started = time.perf_counter()
    for prompt in prompts:
        generated = generate_code_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            context_length=config.model.context_length,
            settings=generation_settings,
        )
        token_ids = generated.generated_token_ids
        generated_tokens += len(token_ids)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        syntax_passes += int(_has_valid_python(text))
        repetition_rates.append(_repetition_rate(token_ids))
    elapsed = max(time.perf_counter() - started, 1e-12)
    syntax = syntax_passes / len(prompts) if prompts else 0.0
    repetition = sum(repetition_rates) / len(repetition_rates) if repetition_rates else 0.0
    return {
        "syntax_correctness": round(syntax, 6),
        "python_benchmark_score": round(max(0.0, syntax * (1.0 - repetition)), 6),
        "repetition_rate": round(repetition, 6),
        "generation_tokens_per_second": generated_tokens / elapsed,
    }


def _has_valid_python(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        ast.parse(stripped)
    except SyntaxError:
        return False
    return True


def _repetition_rate(token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    unique = len(set(token_ids))
    return 1.0 - (unique / len(token_ids))


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _markdown(comparison: BenchmarkComparison) -> str:
    lines = [
        "# GenPy Phase 6.3 Benchmark Comparison",
        "",
        f"- Generated at: {comparison.generated_at}",
        f"- Previous checkpoint: `{comparison.previous.checkpoint}`",
        f"- Continued checkpoint: `{comparison.continued.checkpoint}`",
        f"- Validation loss before: {_fmt(comparison.previous.validation_loss)}",
        f"- Validation loss after: {_fmt(comparison.continued.validation_loss)}",
        f"- Validation loss delta: {_fmt(comparison.validation_loss_delta)}",
        f"- Perplexity before: {_fmt(comparison.previous.perplexity)}",
        f"- Perplexity after: {_fmt(comparison.continued.perplexity)}",
        f"- Perplexity delta: {_fmt(comparison.perplexity_delta)}",
        f"- Syntax correctness before: {comparison.previous.syntax_correctness:.2%}",
        f"- Syntax correctness after: {comparison.continued.syntax_correctness:.2%}",
        f"- Python benchmark score before: {comparison.previous.python_benchmark_score:.6f}",
        f"- Python benchmark score after: {comparison.continued.python_benchmark_score:.6f}",
        f"- Repetition before: {comparison.previous.repetition_rate:.2%}",
        f"- Repetition after: {comparison.continued.repetition_rate:.2%}",
        f"- Generation speed before: {comparison.previous.generation_tokens_per_second:.3f}",
        f"- Generation speed after: {comparison.continued.generation_tokens_per_second:.3f}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


__all__ = [
    "BenchmarkComparison",
    "BenchmarkSettings",
    "CheckpointBenchmark",
    "benchmark_phase63_checkpoints",
    "write_benchmark_comparison",
]
