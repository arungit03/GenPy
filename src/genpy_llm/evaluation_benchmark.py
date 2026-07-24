"""Phase 8 checkpoint evaluation and benchmarking for GenPy."""

from __future__ import annotations

import ast
import csv
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from genpy_llm.checkpointing import LoadedCheckpoint, load_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.evaluation import EvaluationMetrics, evaluation_metrics
from genpy_llm.fine_tuning import Phase7Config
from genpy_llm.instruction_dataset import InstructionDataset
from genpy_llm.instruction_generation import format_generation_prompt
from genpy_llm.pretraining import create_phase6_model
from genpy_llm.pretraining_generation import generate_code_sample

UTC = timezone.utc

DEFAULT_EVALUATION_DATASET = Path("data/evaluation/prompts.json")
RESULTS_JSON = "evaluation_results.json"
RESULTS_CSV = "evaluation_results.csv"
REPORT_MARKDOWN = "evaluation_report.md"


class EvaluationBenchmarkError(RuntimeError):
    """Raised when a Phase 8 evaluation cannot be completed."""


@dataclass(frozen=True)
class AutomaticCheck:
    """Safe, static checks that may be applied to a generated answer."""

    python_syntax: bool = False
    required_terms: tuple[str, ...] = ()
    required_any: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class EvaluationPrompt:
    """One benchmark prompt and its optional automatic check."""

    id: str
    prompt: str
    check: AutomaticCheck | None = None


@dataclass(frozen=True)
class AutomaticCheckResult:
    """Outcome of a non-executing generated-answer check."""

    passed: bool | None
    details: str


@dataclass(frozen=True)
class PromptEvaluationResult:
    """Generation, timing, throughput, and check data for one prompt."""

    id: str
    prompt: str
    generated_answer: str
    generation_time_seconds: float
    generated_tokens: int
    tokens_per_second: float
    passed: bool | None
    check_details: str

    @property
    def pass_fail(self) -> str:
        """Return a report-friendly automatic-check status."""

        if self.passed is None:
            return "N/A"
        return "Pass" if self.passed else "Fail"


@dataclass(frozen=True)
class EvaluationSummary:
    """Complete Phase 8 evaluation result."""

    checkpoint: str
    checkpoint_step: int
    checkpoint_epoch: int
    device: str
    evaluated_at: str
    validation: EvaluationMetrics | None
    total_generation_time_seconds: float
    total_generated_tokens: int
    tokens_per_second: float
    results: tuple[PromptEvaluationResult, ...]


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths written by the evaluation report generator."""

    json_path: Path
    csv_path: Path
    report_path: Path


def load_evaluation_prompts(path: Path | str) -> tuple[EvaluationPrompt, ...]:
    """Load and validate the Phase 8 JSON prompt dataset."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationBenchmarkError(f"Invalid evaluation dataset: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise EvaluationBenchmarkError("Evaluation dataset must be a non-empty JSON array.")

    prompts: list[EvaluationPrompt] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, Mapping):
            raise EvaluationBenchmarkError(f"Evaluation record {index} must be an object.")
        prompt_id = _required_string(item.get("id"), f"record {index} id")
        prompt = _required_string(item.get("prompt"), f"record {index} prompt")
        if prompt_id in seen_ids:
            raise EvaluationBenchmarkError(f"Duplicate evaluation id: {prompt_id}")
        seen_ids.add(prompt_id)
        prompts.append(
            EvaluationPrompt(
                id=prompt_id,
                prompt=prompt,
                check=_parse_check(item.get("check"), index),
            )
        )
    return tuple(prompts)


def resolve_evaluation_checkpoint(
    checkpoint: Path | str | None,
    *,
    config: Phase7Config,
) -> Path:
    """Resolve an explicit checkpoint or Phase 7's canonical latest checkpoint."""

    if checkpoint is not None and str(checkpoint).strip().lower() != "latest":
        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            candidate = config.project_root / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate.resolve()

    directory = config.checkpoint.output_dir
    canonical = directory / config.checkpoint.last_filename
    if canonical.is_file():
        return canonical.resolve()
    candidates = list(directory.glob(f"{config.checkpoint.step_prefix}_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No fine-tuned checkpoints found in {directory}")
    return max(candidates, key=_checkpoint_sort_key).resolve()


def load_model_for_evaluation(
    config: Phase7Config,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, CodeTokenizer, LoadedCheckpoint]:
    """Load Phase 7's tokenizer, model configuration, and checkpoint for inference."""

    tokenizer = CodeTokenizer.from_file(config.data.tokenizer)
    model = create_phase6_model(config.model, tokenizer)
    loaded = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        map_location="cpu",
        restore_rng=False,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, loaded


@torch.no_grad()
def calculate_validation_metrics(
    *,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: Phase7Config,
    device: torch.device,
    max_batches: int | None = None,
) -> EvaluationMetrics | None:
    """Calculate token-weighted Phase 7 validation loss and perplexity when available."""

    if config.data.validation_path is None or not config.data.validation_path.is_file():
        return None
    context_length = min(
        config.data.context_length or config.model.context_length,
        config.model.context_length,
    )
    dataset = InstructionDataset.from_jsonl(
        config.data.validation_path,
        tokenizer=tokenizer,
        template=config.template,
        context_length=context_length,
        mask_prompt_tokens=config.data.mask_prompt_tokens,
    )
    loader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=0)
    batch_limit = config.training.evaluation_steps if max_batches is None else max_batches
    if batch_limit is not None and batch_limit <= 0:
        return None

    total_weighted_loss = 0.0
    total_tokens = 0
    batches = 0
    model.eval()
    for batch in loader:
        if batch_limit is not None and batches >= batch_limit:
            break
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, padding_mask=attention_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        token_count = int((targets != -100).sum().item())
        total_weighted_loss += float(loss.item()) * token_count
        total_tokens += token_count
        batches += 1
    if not total_tokens:
        return None
    return evaluation_metrics(total_weighted_loss / total_tokens, total_tokens, batches)


@torch.no_grad()
def evaluate_prompts(
    *,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: Phase7Config,
    prompts: Sequence[EvaluationPrompt],
    device: torch.device,
    max_new_tokens: int | None = None,
) -> tuple[PromptEvaluationResult, ...]:
    """Run timed inference and static checks for every supplied prompt."""

    if not prompts:
        raise EvaluationBenchmarkError("At least one evaluation prompt is required.")
    settings = config.generation
    if max_new_tokens is not None:
        if max_new_tokens <= 0:
            raise EvaluationBenchmarkError("max_new_tokens must be greater than zero.")
        settings = replace(settings, max_new_tokens=max_new_tokens)

    results: list[PromptEvaluationResult] = []
    for item in prompts:
        formatted_prompt = format_generation_prompt(item.prompt, template=config.template)
        _synchronize(device)
        started = time.perf_counter()
        generated = generate_code_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            device=device,
            context_length=config.model.context_length,
            settings=settings,
        )
        _synchronize(device)
        elapsed = time.perf_counter() - started
        answer = tokenizer.decode(generated.generated_token_ids, skip_special_tokens=True).strip()
        check = run_automatic_check(answer, item.check)
        token_count = len(generated.generated_token_ids)
        results.append(
            PromptEvaluationResult(
                id=item.id,
                prompt=item.prompt,
                generated_answer=answer,
                generation_time_seconds=elapsed,
                generated_tokens=token_count,
                tokens_per_second=token_count / elapsed if elapsed > 0 else 0.0,
                passed=check.passed,
                check_details=check.details,
            )
        )
    return tuple(results)


def build_evaluation_summary(
    *,
    checkpoint_path: Path,
    loaded_checkpoint: LoadedCheckpoint,
    device: torch.device,
    validation: EvaluationMetrics | None,
    results: Sequence[PromptEvaluationResult],
) -> EvaluationSummary:
    """Combine checkpoint, validation, and generation results."""

    total_time = sum(item.generation_time_seconds for item in results)
    total_tokens = sum(item.generated_tokens for item in results)
    return EvaluationSummary(
        checkpoint=str(checkpoint_path.resolve()),
        checkpoint_step=loaded_checkpoint.global_step,
        checkpoint_epoch=loaded_checkpoint.epoch,
        device=str(device),
        evaluated_at=datetime.now(UTC).isoformat(),
        validation=validation,
        total_generation_time_seconds=total_time,
        total_generated_tokens=total_tokens,
        tokens_per_second=total_tokens / total_time if total_time > 0 else 0.0,
        results=tuple(results),
    )


def write_evaluation_artifacts(
    summary: EvaluationSummary,
    output_dir: Path | str,
) -> EvaluationArtifacts:
    """Write the required JSON, CSV, and Markdown evaluation artifacts."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / RESULTS_JSON
    csv_path = directory / RESULTS_CSV
    report_path = directory / REPORT_MARKDOWN

    json_path.write_text(
        json.dumps(_summary_payload(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(summary.results, csv_path)
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return EvaluationArtifacts(json_path=json_path, csv_path=csv_path, report_path=report_path)


def run_automatic_check(
    answer: str,
    check: AutomaticCheck | None,
) -> AutomaticCheckResult:
    """Run non-executing syntax and term checks on a generated answer."""

    if check is None:
        return AutomaticCheckResult(None, "No automatic check configured.")
    failures: list[str] = []
    normalized = _normalize_for_check(answer)
    missing = [
        term for term in check.required_terms if _normalize_for_check(term) not in normalized
    ]
    if missing:
        failures.append(f"missing required terms: {', '.join(missing)}")
    for alternatives in check.required_any:
        if not any(_normalize_for_check(term) in normalized for term in alternatives):
            failures.append(f"missing one of: {', '.join(alternatives)}")
    if check.python_syntax:
        code = extract_python_code(answer)
        if not code:
            failures.append("no Python code found")
        else:
            try:
                ast.parse(code)
            except SyntaxError as exc:
                failures.append(f"Python syntax error at line {exc.lineno}")
    if failures:
        return AutomaticCheckResult(False, "; ".join(failures))
    return AutomaticCheckResult(True, "All configured static checks passed.")


def extract_python_code(answer: str) -> str:
    """Extract a Python code candidate from Markdown or a plain generated answer."""

    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    for candidate in fences:
        if candidate.strip():
            return candidate.strip()
    stripped = answer.strip()
    if not stripped:
        return ""
    try:
        ast.parse(stripped)
    except SyntaxError:
        pass
    else:
        return stripped
    lines = stripped.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:from\s+\S+\s+import|import\s+|def\s+|class\s+|@\w+)", line):
            return "\n".join(lines[index:]).strip()
    return ""


def _summary_payload(summary: EvaluationSummary) -> dict[str, Any]:
    validation = None if summary.validation is None else asdict(summary.validation)
    return {
        "metadata": {
            "checkpoint": summary.checkpoint,
            "checkpoint_step": summary.checkpoint_step,
            "checkpoint_epoch": summary.checkpoint_epoch,
            "device": summary.device,
            "evaluated_at": summary.evaluated_at,
            "prompt_count": len(summary.results),
            "validation": validation,
            "total_generation_time_seconds": summary.total_generation_time_seconds,
            "total_generated_tokens": summary.total_generated_tokens,
            "tokens_per_second": summary.tokens_per_second,
            "automatic_checks": (
                "Static syntax and task-term heuristics; generated code is not executed."
            ),
        },
        "results": [
            {
                **asdict(item),
                "pass_fail": item.pass_fail,
            }
            for item in summary.results
        ],
    }


def _write_csv(results: Sequence[PromptEvaluationResult], output_path: Path) -> None:
    fields = (
        "id",
        "prompt",
        "generated_answer",
        "generation_time_seconds",
        "generated_tokens",
        "tokens_per_second",
        "pass_fail",
        "check_details",
    )
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "id": result.id,
                    "prompt": result.prompt,
                    "generated_answer": result.generated_answer,
                    "generation_time_seconds": f"{result.generation_time_seconds:.6f}",
                    "generated_tokens": result.generated_tokens,
                    "tokens_per_second": f"{result.tokens_per_second:.6f}",
                    "pass_fail": result.pass_fail,
                    "check_details": result.check_details,
                }
            )


def _markdown_report(summary: EvaluationSummary) -> str:
    validation_loss = "N/A"
    perplexity = "N/A"
    validation_tokens = "N/A"
    if summary.validation is not None:
        validation_loss = f"{summary.validation.loss:.6f}"
        perplexity = f"{summary.validation.perplexity:.6f}"
        validation_tokens = str(summary.validation.tokens)
    lines = [
        "# GenPy Phase 8 Evaluation Report",
        "",
        f"- Checkpoint: `{summary.checkpoint}`",
        f"- Checkpoint step: {summary.checkpoint_step}",
        f"- Device: `{summary.device}`",
        f"- Evaluated at: {summary.evaluated_at}",
        f"- Validation loss: {validation_loss}",
        f"- Perplexity: {perplexity}",
        f"- Validation tokens: {validation_tokens}",
        f"- Generated tokens: {summary.total_generated_tokens}",
        f"- Aggregate generation speed: {summary.tokens_per_second:.3f} tokens/sec",
        "",
        (
            "> Pass/fail uses static syntax and task-term heuristics only. "
            "Generated code is not executed."
        ),
        "",
    ]
    for index, result in enumerate(summary.results, start=1):
        lines.extend(
            [
                f"## {index}. {_markdown_inline(result.prompt.splitlines()[0])}",
                "",
                "**Prompt**",
                "",
                _code_block(result.prompt, "text"),
                "",
                "**Generated answer**",
                "",
                _code_block(result.generated_answer or "(empty output)", "text"),
                "",
                f"- Generation time: {result.generation_time_seconds:.6f} seconds",
                f"- Generated tokens: {result.generated_tokens}",
                f"- Tokens/sec: {result.tokens_per_second:.3f}",
                f"- Pass/Fail: **{result.pass_fail}**",
                f"- Check: {result.check_details}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _parse_check(payload: Any, index: int) -> AutomaticCheck | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise EvaluationBenchmarkError(f"Record {index} check must be an object.")
    required_terms = _string_tuple(payload.get("required_terms", ()), index, "required_terms")
    raw_groups = payload.get("required_any", ())
    if not isinstance(raw_groups, (list, tuple)):
        raise EvaluationBenchmarkError(f"Record {index} required_any must be an array.")
    groups = tuple(
        _string_tuple(group, index, "required_any group")
        for group in raw_groups
    )
    python_syntax = payload.get("python_syntax", False)
    if not isinstance(python_syntax, bool):
        raise EvaluationBenchmarkError(f"Record {index} python_syntax must be a boolean.")
    return AutomaticCheck(
        python_syntax=python_syntax,
        required_terms=required_terms,
        required_any=groups,
    )


def _string_tuple(payload: Any, index: int, label: str) -> tuple[str, ...]:
    if not isinstance(payload, (list, tuple)):
        raise EvaluationBenchmarkError(f"Record {index} {label} must be an array.")
    values = tuple(_required_string(value, f"record {index} {label}") for value in payload)
    if label == "required_any group" and not values:
        raise EvaluationBenchmarkError(f"Record {index} required_any groups cannot be empty.")
    return values


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationBenchmarkError(f"{label} must be a non-empty string.")
    return value.strip()


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    matches = re.findall(r"\d+", path.stem)
    step = int(matches[-1]) if matches else -1
    return step, path.stat().st_mtime_ns, path.name


def _normalize_for_check(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _markdown_inline(text: str) -> str:
    return text.replace("\\", "\\\\").replace("#", "\\#").replace("`", "\\`")


def _code_block(text: str, language: str) -> str:
    longest = max((len(match) for match in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


__all__ = [
    "AutomaticCheck",
    "AutomaticCheckResult",
    "DEFAULT_EVALUATION_DATASET",
    "EvaluationArtifacts",
    "EvaluationBenchmarkError",
    "EvaluationPrompt",
    "EvaluationSummary",
    "PromptEvaluationResult",
    "build_evaluation_summary",
    "calculate_validation_metrics",
    "evaluate_prompts",
    "extract_python_code",
    "load_evaluation_prompts",
    "load_model_for_evaluation",
    "resolve_evaluation_checkpoint",
    "run_automatic_check",
    "write_evaluation_artifacts",
]
