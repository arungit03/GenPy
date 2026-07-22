"""Evaluation and checkpoint reporting helpers for GenPy Code LLM."""

from __future__ import annotations

import csv
import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from genpy_llm.code_generation import CodeGenerationResult, generate_code_text
from genpy_llm.code_tokenizer import CodeTokenizer

DEFAULT_CODE_PROMPTS: tuple[str, ...] = (
    "def factorial(n):",
    "class Student:",
    "for i in range(10):",
    "import numpy as np",
    "def quicksort(arr):",
    "class LinkedList:",
    "def fibonacci(n):",
    "try:",
    'with open("file.txt"):',
)


class CodeEvaluationError(RuntimeError):
    """Raised when code-model evaluation artifacts cannot be produced."""


@dataclass(frozen=True)
class CodeCheckpointInfo:
    """Metadata and file information for one code checkpoint."""

    path: Path
    global_step: int | None
    training_loss: float | None
    validation_loss: float | None
    best_metric: float | None
    size_bytes: int

    @property
    def size_mb(self) -> float:
        """Return checkpoint size in MiB."""

        return self.size_bytes / (1024 * 1024)


@dataclass(frozen=True)
class CodeCheckpointSummary:
    """Discovered checkpoint collection with latest and best selections."""

    latest_checkpoint: CodeCheckpointInfo | None
    best_checkpoint: CodeCheckpointInfo | None
    checkpoints: tuple[CodeCheckpointInfo, ...]

    @property
    def total_checkpoints(self) -> int:
        """Return the number of discovered checkpoint files."""

        return len(self.checkpoints)

    @property
    def total_size_bytes(self) -> int:
        """Return the total size of all discovered checkpoint files."""

        return sum(checkpoint.size_bytes for checkpoint in self.checkpoints)


@dataclass(frozen=True)
class LossHistoryRow:
    """One row of checkpoint-backed loss history."""

    checkpoint: str
    global_step: int
    training_loss: float | None
    validation_loss: float | None
    best_metric: float | None


@dataclass(frozen=True)
class TrainingMetricsRow:
    """One recorded training or validation monitoring row."""

    global_step: int
    training_loss: float
    validation_loss: float | None
    perplexity: float | None
    learning_rate: float
    gradient_norm: float | None
    tokens_per_second: float
    tokens_processed: int
    elapsed_seconds: float
    eta_seconds: float


@dataclass(frozen=True)
class GenerationExample:
    """One generated example and timing result."""

    prompt: str
    text: str
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class GenerationBenchmark:
    """Aggregate generation benchmark results."""

    examples: tuple[GenerationExample, ...]

    @property
    def average_generation_length(self) -> float:
        """Return average generated-token count."""

        if not self.examples:
            return 0.0
        return sum(example.generated_tokens for example in self.examples) / len(self.examples)

    @property
    def tokens_per_second(self) -> float:
        """Return aggregate generated-token throughput."""

        total_tokens = sum(example.generated_tokens for example in self.examples)
        total_seconds = sum(example.elapsed_seconds for example in self.examples)
        return total_tokens / total_seconds if total_seconds > 0 else 0.0


def discover_code_checkpoints(
    checkpoint_directory: Path,
    *,
    filename_prefix: str,
    best_filename: str,
) -> CodeCheckpointSummary:
    """Discover code checkpoints and identify latest and best files."""

    directory = Path(checkpoint_directory)
    if not directory.exists():
        return CodeCheckpointSummary(None, None, ())
    if not directory.is_dir():
        raise CodeEvaluationError(f"Checkpoint path is not a directory: {directory}")
    checkpoints = tuple(
        sorted(
            (load_code_checkpoint_info(path) for path in directory.glob("*.pt")),
            key=lambda item: (
                item.global_step if item.global_step is not None else -1,
                item.path.name,
            ),
        )
    )
    latest_path = _latest_code_checkpoint_path(directory, filename_prefix)
    best_path = directory / best_filename
    latest = _find_info(checkpoints, latest_path)
    best = (
        _find_info(checkpoints, best_path)
        if best_path.exists()
        else _best_by_validation(checkpoints)
    )
    return CodeCheckpointSummary(latest, best, checkpoints)


def resolve_code_checkpoint(
    spec: str | Path | None,
    *,
    checkpoint_directory: Path,
    filename_prefix: str,
    best_filename: str,
    project_root: Path,
) -> Path:
    """Resolve 'latest', 'best', or an explicit checkpoint path."""

    summary = discover_code_checkpoints(
        checkpoint_directory,
        filename_prefix=filename_prefix,
        best_filename=best_filename,
    )
    if spec is None or str(spec).strip().lower() == "best":
        if summary.best_checkpoint is None:
            raise FileNotFoundError(f"No best checkpoint found in {checkpoint_directory}")
        return summary.best_checkpoint.path
    normalized = str(spec).strip()
    if normalized.lower() == "latest":
        if summary.latest_checkpoint is None:
            raise FileNotFoundError(f"No latest checkpoint found in {checkpoint_directory}")
        return summary.latest_checkpoint.path
    path = Path(normalized)
    path = path if path.is_absolute() else project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path


def load_code_checkpoint_info(path: Path) -> CodeCheckpointInfo:
    """Load metadata for one checkpoint file."""

    checkpoint_path = Path(path)
    payload = _safe_torch_load_metadata(checkpoint_path)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return CodeCheckpointInfo(
        path=checkpoint_path.resolve(),
        global_step=_optional_int(metadata.get("global_step")),
        training_loss=_optional_float(metadata.get("training_loss")),
        validation_loss=_optional_float(metadata.get("validation_loss")),
        best_metric=_optional_float(metadata.get("best_metric")),
        size_bytes=checkpoint_path.stat().st_size,
    )


def build_loss_history(summary: CodeCheckpointSummary) -> tuple[LossHistoryRow, ...]:
    """Build loss-history rows from checkpoint metadata."""

    rows: list[LossHistoryRow] = []
    seen: set[tuple[int, str]] = set()
    for checkpoint in summary.checkpoints:
        if checkpoint.global_step is None:
            continue
        if checkpoint.training_loss is None and checkpoint.validation_loss is None:
            continue
        key = (checkpoint.global_step, checkpoint.path.name)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            LossHistoryRow(
                checkpoint=checkpoint.path.name,
                global_step=checkpoint.global_step,
                training_loss=checkpoint.training_loss,
                validation_loss=checkpoint.validation_loss,
                best_metric=checkpoint.best_metric,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.global_step, row.checkpoint)))


def write_loss_history_csv(rows: tuple[LossHistoryRow, ...], output_path: Path) -> None:
    """Write loss history to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["global_step", "training_loss", "validation_loss", "best_metric", "checkpoint"]
        )
        for row in rows:
            writer.writerow(
                [
                    row.global_step,
                    "" if row.training_loss is None else f"{row.training_loss:.8f}",
                    "" if row.validation_loss is None else f"{row.validation_loss:.8f}",
                    "" if row.best_metric is None else f"{row.best_metric:.8f}",
                    row.checkpoint,
                ]
            )


def append_training_metrics_csv(row: TrainingMetricsRow, output_path: Path) -> None:
    """Append a monitoring row to a training metrics CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if needs_header:
            writer.writerow(
                [
                    "global_step",
                    "training_loss",
                    "validation_loss",
                    "perplexity",
                    "learning_rate",
                    "gradient_norm",
                    "tokens_per_second",
                    "tokens_processed",
                    "elapsed_seconds",
                    "eta_seconds",
                ]
            )
        writer.writerow(
            [
                row.global_step,
                f"{row.training_loss:.8f}",
                "" if row.validation_loss is None else f"{row.validation_loss:.8f}",
                "" if row.perplexity is None else f"{row.perplexity:.8f}",
                f"{row.learning_rate:.10f}",
                "" if row.gradient_norm is None else f"{row.gradient_norm:.8f}",
                f"{row.tokens_per_second:.4f}",
                row.tokens_processed,
                f"{row.elapsed_seconds:.4f}",
                f"{row.eta_seconds:.4f}",
            ]
        )


def read_training_metrics_csv(path: Path) -> tuple[TrainingMetricsRow, ...]:
    """Read recorded training metrics from CSV."""

    if not path.exists():
        return ()
    rows: list[TrainingMetricsRow] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            step = _optional_int(_parse_number(raw.get("global_step")))
            training_loss = _optional_float(_parse_number(raw.get("training_loss")))
            if step is None or training_loss is None:
                continue
            rows.append(
                TrainingMetricsRow(
                    global_step=step,
                    training_loss=training_loss,
                    validation_loss=_optional_float(_parse_number(raw.get("validation_loss"))),
                    perplexity=_optional_float(_parse_number(raw.get("perplexity"))),
                    learning_rate=_optional_float(_parse_number(raw.get("learning_rate"))) or 0.0,
                    gradient_norm=_optional_float(_parse_number(raw.get("gradient_norm"))),
                    tokens_per_second=(
                        _optional_float(_parse_number(raw.get("tokens_per_second"))) or 0.0
                    ),
                    tokens_processed=_optional_int(_parse_number(raw.get("tokens_processed"))) or 0,
                    elapsed_seconds=(
                        _optional_float(_parse_number(raw.get("elapsed_seconds"))) or 0.0
                    ),
                    eta_seconds=_optional_float(_parse_number(raw.get("eta_seconds"))) or 0.0,
                )
            )
    return tuple(rows)


def loss_history_from_training_metrics(
    rows: tuple[TrainingMetricsRow, ...],
) -> tuple[LossHistoryRow, ...]:
    """Convert recorded training metrics into loss-history rows."""

    return tuple(
        LossHistoryRow(
            checkpoint=f"step_{row.global_step}",
            global_step=row.global_step,
            training_loss=row.training_loss,
            validation_loss=row.validation_loss,
            best_metric=None,
        )
        for row in rows
    )


def write_loss_curve_png(rows: tuple[LossHistoryRow, ...], output_path: Path) -> None:
    """Write a simple dependency-free PNG loss curve."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    train_points = [
        (row.global_step, row.training_loss) for row in rows if row.training_loss is not None
    ]
    validation_points = [
        (row.global_step, row.validation_loss) for row in rows if row.validation_loss is not None
    ]
    all_values = [value for _step, value in train_points + validation_points]
    if all_values:
        min_step = min(row.global_step for row in rows)
        max_step = max(row.global_step for row in rows)
        min_loss = min(all_values)
        max_loss = max(all_values)
        _plot_series(
            canvas,
            train_points,
            bounds=(min_step, max_step, min_loss, max_loss),
            area=(margin_left, margin_top, plot_width, plot_height),
            color=(36, 112, 194),
        )
        _plot_series(
            canvas,
            validation_points,
            bounds=(min_step, max_step, min_loss, max_loss),
            area=(margin_left, margin_top, plot_width, plot_height),
            color=(210, 86, 59),
        )
    output_path.write_bytes(canvas.to_png_bytes())


def run_generation_benchmark(
    *,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    prompts: tuple[str, ...] = DEFAULT_CODE_PROMPTS,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    do_sample: bool,
    stop_on_eos: bool,
    context_length: int,
) -> GenerationBenchmark:
    """Generate code for benchmark prompts and return aggregate metrics."""

    examples: list[GenerationExample] = []
    for prompt in prompts:
        result = generate_code_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_on_eos=stop_on_eos,
            instruction_mode=False,
            code_only=False,
            context_length=context_length,
        )
        examples.append(_generation_example(prompt, result))
    return GenerationBenchmark(examples=tuple(examples))


def write_generation_examples(
    benchmark: GenerationBenchmark,
    output_path: Path,
    *,
    checkpoint_path: Path | None = None,
    step: int | None = None,
) -> None:
    """Write generated examples to a text artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["GenPy Code Generation Benchmark", "==============================", ""]
    if checkpoint_path is not None:
        lines.append(f"Checkpoint: {checkpoint_path}")
    if step is not None:
        lines.append(f"Step: {step}")
    if checkpoint_path is not None or step is not None:
        lines.append("")
    for index, example in enumerate(benchmark.examples, start=1):
        lines.extend(
            [
                f"Prompt {index}: {example.prompt}",
                f"Generated tokens: {example.generated_tokens}",
                f"Tokens/sec: {example.tokens_per_second:.2f}",
                "Output:",
                example.text.rstrip(),
                "",
                "-" * 72,
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def perplexity_from_loss(loss: float | None) -> float | None:
    """Return perplexity for a cross-entropy loss."""

    if loss is None:
        return None
    if loss > 80:
        return math.inf
    return math.exp(loss)


def format_size(size_bytes: int) -> str:
    """Return a human-readable file size."""

    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"


def _generation_example(prompt: str, result: CodeGenerationResult) -> GenerationExample:
    return GenerationExample(
        prompt=prompt,
        text=result.text,
        generated_tokens=len(result.generated_token_ids),
        elapsed_seconds=result.elapsed_seconds,
        tokens_per_second=result.tokens_per_second,
    )


def _latest_code_checkpoint_path(directory: Path, filename_prefix: str) -> Path | None:
    pattern = re.compile(rf"^{re.escape(filename_prefix)}_step_(?P<step>\d+)\.pt$")
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"{filename_prefix}_step_*.pt"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            candidates.append((int(match.group("step")), path))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1].name))[1].resolve()
    fallbacks = sorted(directory.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return fallbacks[0].resolve() if fallbacks else None


def _find_info(
    checkpoints: tuple[CodeCheckpointInfo, ...],
    path: Path | None,
) -> CodeCheckpointInfo | None:
    if path is None:
        return None
    resolved = path.resolve()
    for checkpoint in checkpoints:
        if checkpoint.path == resolved:
            return checkpoint
    return None


def _best_by_validation(
    checkpoints: tuple[CodeCheckpointInfo, ...],
) -> CodeCheckpointInfo | None:
    candidates = [
        checkpoint for checkpoint in checkpoints if checkpoint.validation_loss is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda checkpoint: checkpoint.validation_loss or math.inf)


def _safe_torch_load_metadata(path: Path) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise CodeEvaluationError(f"Could not read checkpoint metadata {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CodeEvaluationError(f"Checkpoint payload must be a mapping: {path}")
    return loaded


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _parse_number(value: str | None) -> int | float | None:
    if value is None or value.strip() == "":
        return None
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return None


class _Canvas:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        background: tuple[int, int, int],
    ) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(color)

    def line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            for x_offset in (-1, 0, 1):
                for y_offset in (-1, 0, 1):
                    self.set_pixel(x0 + x_offset, y0 + y_offset, color)
            if x0 == x1 and y0 == y1:
                break
            error2 = 2 * error
            if error2 >= dy:
                error += dy
                x0 += sx
            if error2 <= dx:
                error += dx
                y0 += sy

    def to_png_bytes(self) -> bytes:
        """Encode the canvas as PNG bytes."""

        raw = bytearray()
        row_stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * row_stride
            raw.extend(self.pixels[start : start + row_stride])
        return b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                _png_chunk(
                    b"IHDR",
                    struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
                ),
                _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)),
                _png_chunk(b"IEND", b""),
            ]
        )


def _plot_series(
    canvas: _Canvas,
    points: list[tuple[int, float]],
    *,
    bounds: tuple[int, int, float, float],
    area: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    if not points:
        return
    min_step, max_step, min_loss, max_loss = bounds
    x0, y0, width, height = area
    step_span = max(max_step - min_step, 1)
    loss_span = max(max_loss - min_loss, 1e-9)

    def transform(step: int, loss: float) -> tuple[int, int]:
        x = x0 + int((step - min_step) / step_span * width)
        y = y0 + height - int((loss - min_loss) / loss_span * height)
        return x, y

    transformed = [transform(step, loss) for step, loss in sorted(points)]
    for x, y in transformed:
        canvas.line(x - 3, y, x + 3, y, color)
        canvas.line(x, y - 3, x, y + 3, color)
    for first, second in zip(transformed, transformed[1:], strict=False):
        canvas.line(first[0], first[1], second[0], second[1], color)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


__all__ = [
    "CodeCheckpointInfo",
    "CodeCheckpointSummary",
    "CodeEvaluationError",
    "DEFAULT_CODE_PROMPTS",
    "GenerationBenchmark",
    "GenerationExample",
    "LossHistoryRow",
    "TrainingMetricsRow",
    "append_training_metrics_csv",
    "build_loss_history",
    "discover_code_checkpoints",
    "format_size",
    "load_code_checkpoint_info",
    "perplexity_from_loss",
    "loss_history_from_training_metrics",
    "read_training_metrics_csv",
    "resolve_code_checkpoint",
    "run_generation_benchmark",
    "write_generation_examples",
    "write_loss_curve_png",
    "write_loss_history_csv",
]
