"""Fine-tune a GenPy Code LLM on instruction, JSON, JSONL, or TXT data."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.checkpointing import CheckpointError, load_checkpoint, save_checkpoint
from genpy_llm.code_evaluation import (
    TrainingMetricsRow,
    append_training_metrics_csv,
    loss_history_from_training_metrics,
    perplexity_from_loss,
    read_training_metrics_csv,
    run_generation_benchmark,
    write_generation_examples,
    write_loss_curve_png,
)
from genpy_llm.code_generation import generate_code_text
from genpy_llm.code_tokenizer import CodeTokenizer, ensure_code_tokenizer
from genpy_llm.code_training import CodeConfig, create_code_model, load_code_config, select_device
from genpy_llm.fine_tuning_dataset import (
    FineTuningDataset,
    load_fine_tuning_examples,
    split_fine_tuning_examples,
)
from genpy_llm.losses import GPTCrossEntropyLoss
from genpy_llm.optimizers import create_optimizer
from genpy_llm.performance import autocast_context, create_grad_scaler
from genpy_llm.utils import set_seed

LOGGER = logging.getLogger("genpy_llm.fine_tuning")
FINE_TUNE_PROMPTS: tuple[str, ...] = (
    "def factorial(n):",
    "class Student:",
    "def fibonacci(n):",
    "class LinkedList:",
    "import numpy as np",
)


class FineTuningError(RuntimeError):
    """Raised when fine-tuning cannot continue."""


@dataclass(frozen=True)
class FineTuneSettings:
    """Runtime settings for supervised code fine-tuning."""

    base_config: Path
    checkpoint: Path | None
    dataset: Path
    output_dir: Path
    log_dir: Path
    evaluation_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_length: int
    gradient_accumulation: int
    save_every: int
    eval_every: int
    seed: int
    device: str
    resume: str | None
    validation_split: float
    response_only_loss: bool
    shuffle: bool
    mixed_precision: str
    max_grad_norm: float | None
    early_stopping_patience: int
    keep_last: int
    generation_max_new_tokens: int
    max_steps: int | None = None
    max_train_batches: int | None = None
    eval_batches: int | None = None
    debug: bool = False


@dataclass(frozen=True)
class FineTuneResult:
    """Summary returned by a fine-tuning run."""

    global_step: int
    best_validation_loss: float | None
    latest_checkpoint: Path
    best_checkpoint: Path | None
    metrics_csv: Path
    metrics_jsonl: Path


def main() -> int:
    """CLI entrypoint for supervised fine-tuning."""

    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = _settings_from_args(args)
        result = run_fine_tuning(settings)
    except Exception as exc:  # noqa: BLE001
        _report_error(exc, debug=args.debug)
        return 1

    print("GenPy Code LLM Fine-Tuning")
    print("==========================")
    print(f"Global step: {result.global_step}")
    print(f"Best validation loss: {_format_optional(result.best_validation_loss)}")
    print(f"Latest checkpoint: {result.latest_checkpoint}")
    if result.best_checkpoint is not None:
        print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Metrics CSV: {result.metrics_csv}")
    print(f"Metrics JSONL: {result.metrics_jsonl}")
    return 0


def run_fine_tuning(settings: FineTuneSettings) -> FineTuneResult:
    """Run supervised fine-tuning with resume, monitoring, and evaluation."""

    _validate_settings(settings)
    set_seed(settings.seed)
    code_config = load_code_config(_resolve(settings.base_config))
    if settings.max_length > code_config.model.context_length:
        raise FineTuningError(
            "max_length cannot exceed the model context length "
            f"({settings.max_length} > {code_config.model.context_length})."
        )
    tokenizer = ensure_code_tokenizer(
        tokenizer_path=code_config.tokenizer.path,
        metadata_path=code_config.tokenizer.metadata_path,
        project_root=PROJECT_ROOT,
        vocab_size=code_config.tokenizer.vocab_size,
        preferred_corpus_paths=(_resolve(settings.dataset),),
        train_pattern=code_config.streaming_dataset.train_pattern,
    )
    device = select_device(settings.device)
    mixed_precision = _effective_mixed_precision(settings.mixed_precision, device)

    examples = load_fine_tuning_examples(_resolve(settings.dataset))
    split = split_fine_tuning_examples(
        examples,
        validation_split=settings.validation_split,
        seed=settings.seed,
        shuffle=settings.shuffle,
    )
    train_dataset = FineTuningDataset(
        split.train_examples,
        tokenizer,
        max_length=settings.max_length,
        response_only_loss=settings.response_only_loss,
        ignore_index=tokenizer.pad_token_id,
    )
    validation_examples = split.validation_examples or split.train_examples[:1]
    validation_dataset = FineTuningDataset(
        validation_examples,
        tokenizer,
        max_length=settings.max_length,
        response_only_loss=settings.response_only_loss,
        ignore_index=tokenizer.pad_token_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=settings.shuffle,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=settings.batch_size)

    model = create_code_model(code_config, tokenizer.vocab_size, tokenizer.pad_token_id).to(device)
    optimizer = create_optimizer(
        model,
        replace(
            code_config.optimizer,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
        ),
    )
    total_steps = _total_optimizer_steps(settings, train_loader)
    scheduler = _build_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=settings.warmup_ratio,
        minimum_ratio=code_config.scheduler.minimum_learning_rate_ratio,
    )
    scaler = create_grad_scaler(mixed_precision, device)
    loss_fn = GPTCrossEntropyLoss(tokenizer.pad_token_id, True, code_config.loss.label_smoothing)

    start_epoch, global_step, best_validation_loss = _load_starting_state(
        settings=settings,
        code_config=code_config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = settings.log_dir / "metrics.csv"
    metrics_jsonl = settings.log_dir / "metrics.jsonl"
    writer = _create_summary_writer(settings.log_dir)
    start_time = time.perf_counter()
    tokens_seen = 0
    stale_evaluations = 0
    latest_checkpoint = settings.output_dir / "latest.pt"
    best_checkpoint = settings.output_dir / "best.pt" if best_validation_loss is not None else None

    print("Entering fine-tuning loop")
    print(
        f"Device={device} AMP={mixed_precision} train={len(train_dataset)} "
        f"validation={len(validation_dataset)} total_steps={total_steps}"
    )
    optimizer.zero_grad(set_to_none=True)

    for epoch_index in range(start_epoch, settings.epochs):
        epoch_number = epoch_index + 1
        batch_limit = _epoch_batch_limit(settings, train_loader)
        running_loss = 0.0
        running_tokens = 0
        accumulated_batches = 0
        for batch_index, batch in enumerate(train_loader):
            if settings.max_train_batches is not None and batch_index >= settings.max_train_batches:
                break
            model.train()
            input_ids, target_ids, attention_mask = _move_batch(batch, device)
            with autocast_context(mixed_precision, device):
                logits = model(input_ids, padding_mask=attention_mask)
                loss = loss_fn(logits, target_ids)
            scaled_loss = loss / settings.gradient_accumulation
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            batch_tokens = int((target_ids != tokenizer.pad_token_id).sum().item())
            tokens_seen += batch_tokens
            running_loss += float(loss.detach().item()) * max(batch_tokens, 1)
            running_tokens += max(batch_tokens, 1)
            accumulated_batches += 1

            last_batch = batch_index + 1 >= batch_limit
            should_step = accumulated_batches >= settings.gradient_accumulation or last_batch
            if not should_step:
                continue

            gradient_norm = _optimizer_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=settings.max_grad_norm,
            )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_batches = 0
            global_step += 1

            train_loss = running_loss / max(running_tokens, 1)
            should_evaluate = global_step % settings.eval_every == 0
            should_save = global_step % settings.save_every == 0
            if should_evaluate or should_save:
                validation_loss = None
                perplexity = None
                save_best_checkpoint = False
                if should_evaluate:
                    validation_loss = evaluate_loss(
                        model=model,
                        loader=validation_loader,
                        loss_fn=loss_fn,
                        device=device,
                        mixed_precision=mixed_precision,
                        max_batches=settings.eval_batches,
                    )
                    perplexity = perplexity_from_loss(validation_loss)
                    _write_generation_snapshot(
                        model=model,
                        tokenizer=tokenizer,
                        config=code_config,
                        settings=settings,
                        device=device,
                        global_step=global_step,
                    )
                    improved = (
                        best_validation_loss is None
                        or validation_loss < best_validation_loss
                    )
                    if improved:
                        best_validation_loss = validation_loss
                        best_checkpoint = settings.output_dir / "best.pt"
                        stale_evaluations = 0
                        save_best_checkpoint = True
                    else:
                        stale_evaluations += 1
                elapsed = time.perf_counter() - start_time
                eta = _eta_seconds(global_step, total_steps, elapsed)
                tokens_per_second = tokens_seen / elapsed if elapsed > 0 else 0.0
                row = TrainingMetricsRow(
                    global_step=global_step,
                    training_loss=train_loss,
                    validation_loss=validation_loss,
                    perplexity=perplexity,
                    learning_rate=_current_learning_rate(optimizer),
                    gradient_norm=gradient_norm,
                    tokens_per_second=tokens_per_second,
                    tokens_processed=tokens_seen,
                    elapsed_seconds=elapsed,
                    eta_seconds=eta,
                )
                _record_metrics(row, metrics_csv, metrics_jsonl, writer)
                _print_metrics(epoch_number, settings.epochs, row)
                latest_checkpoint = _save_fine_tune_checkpoint(
                    path=settings.output_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    tokenizer=tokenizer,
                    config=code_config,
                    settings=settings,
                    epoch=epoch_number,
                    global_step=global_step,
                    training_loss=train_loss,
                    validation_loss=validation_loss,
                    best_validation_loss=best_validation_loss,
                )
                if should_save:
                    _save_fine_tune_checkpoint(
                        path=settings.output_dir / f"step_{global_step:08d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        tokenizer=tokenizer,
                        config=code_config,
                        settings=settings,
                        epoch=epoch_number,
                        global_step=global_step,
                        training_loss=train_loss,
                        validation_loss=validation_loss,
                        best_validation_loss=best_validation_loss,
                    )
                    _rotate_periodic_checkpoints(settings.output_dir, settings.keep_last)
                if save_best_checkpoint and best_checkpoint is not None:
                    _save_fine_tune_checkpoint(
                        path=best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        tokenizer=tokenizer,
                        config=code_config,
                        settings=settings,
                        epoch=epoch_number,
                        global_step=global_step,
                        training_loss=train_loss,
                        validation_loss=validation_loss,
                        best_validation_loss=best_validation_loss,
                    )
                running_loss = 0.0
                running_tokens = 0
                if _should_stop_early(settings, stale_evaluations):
                    LOGGER.info("Early stopping after %s stale evaluations.", stale_evaluations)
                    _write_loss_curve(metrics_csv, settings.evaluation_dir / "loss_curve.png")
                    _close_summary_writer(writer)
                    return FineTuneResult(
                        global_step=global_step,
                        best_validation_loss=best_validation_loss,
                        latest_checkpoint=latest_checkpoint,
                        best_checkpoint=best_checkpoint,
                        metrics_csv=metrics_csv,
                        metrics_jsonl=metrics_jsonl,
                    )
            if settings.max_steps is not None and global_step >= settings.max_steps:
                break
        epoch_loss = running_loss / max(running_tokens, 1) if running_tokens else None
        validation_loss = evaluate_loss(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=mixed_precision,
            max_batches=settings.eval_batches,
        )
        save_best_epoch_checkpoint = (
            best_validation_loss is None or validation_loss < best_validation_loss
        )
        if save_best_epoch_checkpoint:
            best_validation_loss = validation_loss
            best_checkpoint = settings.output_dir / "best.pt"
        latest_checkpoint = _save_fine_tune_checkpoint(
            path=settings.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=tokenizer,
            config=code_config,
            settings=settings,
            epoch=epoch_number,
            global_step=global_step,
            training_loss=epoch_loss,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss,
        )
        _save_fine_tune_checkpoint(
            path=settings.output_dir / f"epoch_{epoch_number}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=tokenizer,
            config=code_config,
            settings=settings,
            epoch=epoch_number,
            global_step=global_step,
            training_loss=epoch_loss,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss,
        )
        if save_best_epoch_checkpoint and best_checkpoint is not None:
            _save_fine_tune_checkpoint(
                path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                tokenizer=tokenizer,
                config=code_config,
                settings=settings,
                epoch=epoch_number,
                global_step=global_step,
                training_loss=epoch_loss,
                validation_loss=validation_loss,
                best_validation_loss=best_validation_loss,
            )
        if settings.max_steps is not None and global_step >= settings.max_steps:
            break

    _write_loss_curve(metrics_csv, settings.evaluation_dir / "loss_curve.png")
    _close_summary_writer(writer)
    return FineTuneResult(
        global_step=global_step,
        best_validation_loss=best_validation_loss,
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=best_checkpoint,
        metrics_csv=metrics_csv,
        metrics_jsonl=metrics_jsonl,
    )


@torch.no_grad()
def evaluate_loss(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: GPTCrossEntropyLoss,
    device: torch.device,
    mixed_precision: str,
    max_batches: int | None = None,
) -> float:
    """Compute validation loss for a fine-tuning dataset."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids, target_ids, attention_mask = _move_batch(batch, device)
        with autocast_context(mixed_precision, device):
            logits = model(input_ids, padding_mask=attention_mask)
            loss = loss_fn(logits, target_ids)
        tokens = int((target_ids != loss_fn.padding_idx).sum().item())
        total_loss += float(loss.detach().item()) * max(tokens, 1)
        total_tokens += max(tokens, 1)
    return total_loss / max(total_tokens, 1)


def _settings_from_args(args: argparse.Namespace) -> FineTuneSettings:
    settings = load_fine_tune_settings(args.config)
    checkpoint = args.checkpoint or args.base_checkpoint
    overrides = {
        "checkpoint": _optional_resolved(checkpoint),
        "dataset": _optional_resolved(args.dataset),
        "output_dir": _optional_resolved(args.output_dir),
        "log_dir": _optional_resolved(args.log_dir),
        "evaluation_dir": _optional_resolved(args.evaluation_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "gradient_accumulation": args.gradient_accumulation,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device": args.device,
        "resume": args.resume,
        "max_steps": args.max_steps,
        "max_train_batches": args.max_train_batches or args.max_batches,
        "eval_batches": args.eval_batches or args.validation_batches,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "debug": args.debug,
    }
    for name, value in overrides.items():
        if value is not None:
            settings = replace(settings, **{name: value})
    if settings.checkpoint is None and settings.resume is None:
        code_config = load_code_config(_resolve(settings.base_config))
        settings = replace(
            settings,
            checkpoint=code_config.checkpoint.directory / code_config.checkpoint.best_filename,
        )
    return settings


def load_fine_tune_settings(config_path: Path | str) -> FineTuneSettings:
    """Load fine-tuning settings from configs/fine_tune.yaml or code_small.yaml."""

    path = _resolve(Path(config_path))
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuning config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise FineTuningError("Fine-tuning config must be a YAML mapping.")
    fine_tuning = raw.get("fine_tuning", {})
    if not isinstance(fine_tuning, dict):
        raise FineTuningError("fine_tuning config must be a YAML mapping.")

    default_base_config = (
        path if "model" in raw and "tokenizer" in raw else Path("configs/code_small.yaml")
    )
    output_dir = fine_tuning.get("output_dir", fine_tuning.get("output_directory"))
    dataset = fine_tuning.get("dataset", fine_tuning.get("dataset_file"))
    learning_rate = fine_tuning.get("learning_rate", 5e-5)
    return FineTuneSettings(
        base_config=_resolve_path_value(fine_tuning.get("base_config", default_base_config)),
        checkpoint=_optional_path_value(
            fine_tuning.get("checkpoint", fine_tuning.get("base_checkpoint"))
        ),
        dataset=_resolve_path_value(dataset or "data/fine_tuning/code_instructions.jsonl"),
        output_dir=_resolve_path_value(output_dir or "checkpoints/code_fine_tune"),
        log_dir=_resolve_path_value(fine_tuning.get("log_dir", "logs/fine_tune")),
        evaluation_dir=_resolve_path_value(
            fine_tuning.get("evaluation_dir", "evaluation/fine_tune")
        ),
        epochs=int(fine_tuning.get("epochs", 3)),
        batch_size=int(fine_tuning.get("batch_size", 4)),
        learning_rate=float(learning_rate),
        weight_decay=float(fine_tuning.get("weight_decay", 0.01)),
        warmup_ratio=float(fine_tuning.get("warmup_ratio", 0.03)),
        max_length=int(
            fine_tuning.get("max_length", fine_tuning.get("max_sequence_length", 512))
        ),
        gradient_accumulation=int(
            fine_tuning.get(
                "gradient_accumulation",
                fine_tuning.get("gradient_accumulation_steps", 8),
            )
        ),
        save_every=int(fine_tuning.get("save_every", fine_tuning.get("save_every_steps", 500))),
        eval_every=int(
            fine_tuning.get("eval_every", fine_tuning.get("validate_every_steps", 500))
        ),
        seed=int(raw.get("project", {}).get("seed", fine_tuning.get("seed", 42))),
        device=str(fine_tuning.get("device", raw.get("training", {}).get("device", "auto"))),
        resume=_optional_str(fine_tuning.get("resume")),
        validation_split=float(
            fine_tuning.get("validation_split", fine_tuning.get("validation_ratio", 0.05))
        ),
        response_only_loss=bool(fine_tuning.get("response_only_loss", True)),
        shuffle=bool(fine_tuning.get("shuffle", True)),
        mixed_precision=str(
            fine_tuning.get(
                "mixed_precision",
                raw.get("training", {}).get("mixed_precision", "fp16"),
            )
        ),
        max_grad_norm=_optional_float(
            fine_tuning.get("max_grad_norm", raw.get("training", {}).get("max_grad_norm", 1.0))
        ),
        early_stopping_patience=int(fine_tuning.get("early_stopping_patience", 5)),
        keep_last=int(fine_tuning.get("keep_last", raw.get("training", {}).get("keep_last", 3))),
        generation_max_new_tokens=int(fine_tuning.get("generation_max_new_tokens", 96)),
        max_steps=_optional_int(fine_tuning.get("max_steps")),
        eval_batches=_optional_int(
            fine_tuning.get("eval_batches", fine_tuning.get("validation_steps"))
        ),
    )


def _load_starting_state(
    *,
    settings: FineTuneSettings,
    code_config: CodeConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object | None,
    device: torch.device,
) -> tuple[int, int, float | None]:
    if settings.resume is not None:
        resume_path = _resolve_resume_checkpoint(settings.resume, settings.output_dir)
        LOGGER.info("Resuming fine-tuning from %s", resume_path)
        loaded = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
            restore_rng=True,
        )
        return loaded.epoch, loaded.global_step, loaded.best_metric

    if settings.checkpoint is None:
        raise FineTuningError("A base --checkpoint is required unless --resume is used.")
    base_checkpoint = _resolve(settings.checkpoint)
    LOGGER.info("Loading base checkpoint from %s", base_checkpoint)
    load_checkpoint(base_checkpoint, model, optimizer=None, map_location=device, restore_rng=False)
    return 0, 0, None


def _save_fine_tune_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: object | None,
    tokenizer: CodeTokenizer,
    config: CodeConfig,
    settings: FineTuneSettings,
    epoch: int,
    global_step: int,
    training_loss: float | None,
    validation_loss: float | None,
    best_validation_loss: float | None,
) -> Path:
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=epoch,
        global_step=global_step,
        training_loss=training_loss,
        validation_loss=validation_loss,
        best_metric=best_validation_loss,
        scheduler=scheduler,
        scaler=scaler,
        model_config={
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": config.model.embedding_dim,
            "num_heads": config.model.num_heads,
            "num_layers": config.model.num_layers,
            "context_length": config.model.context_length,
            "dropout": config.model.dropout,
            "tie_embeddings": config.model.tie_embeddings,
        },
        vocabulary_metadata={
            "vocabulary_size": tokenizer.vocab_size,
            "tokenizer_type": config.tokenizer.type,
        },
        extra_state={
            "checkpoint_type": "code_fine_tune",
            "format_version": 1,
            "output_dir": str(settings.output_dir),
            "dataset": str(settings.dataset),
        },
    )
    LOGGER.info("Saved checkpoint: %s", path)
    return path


def _write_generation_snapshot(
    *,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    config: CodeConfig,
    settings: FineTuneSettings,
    device: torch.device,
    global_step: int,
) -> None:
    benchmark = run_generation_benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=FINE_TUNE_PROMPTS,
        device=device,
        max_new_tokens=settings.generation_max_new_tokens,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        repetition_penalty=config.generation.repetition_penalty,
        do_sample=config.generation.do_sample,
        stop_on_eos=config.generation.stop_on_eos,
        context_length=config.model.context_length,
    )
    output_path = settings.evaluation_dir / f"step_{global_step:08d}_generation.txt"
    write_generation_examples(benchmark, output_path, step=global_step)


def generate_fine_tuned_text(
    *,
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    prompt: str,
    config: CodeConfig,
    device: torch.device,
) -> str:
    """Generate text from a fine-tuned model using instruction mode."""

    result = generate_code_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        repetition_penalty=config.generation.repetition_penalty,
        do_sample=config.generation.do_sample,
        stop_on_eos=config.generation.stop_on_eos,
        instruction_mode=True,
        code_only=True,
        context_length=config.model.context_length,
    )
    return result.text


def _optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: object | None,
    max_grad_norm: float | None,
) -> float:
    if scaler is not None:
        scaler.unscale_(optimizer)
    gradient_norm = _compute_gradient_norm(model)
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return gradient_norm


def _compute_gradient_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm(2)
        total += float(norm.item()) ** 2
    return math.sqrt(total)


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["input_ids"].to(device),
        batch["target_ids"].to(device),
        batch["attention_mask"].to(device),
    )


def _build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _total_optimizer_steps(settings: FineTuneSettings, train_loader: DataLoader) -> int:
    if settings.max_steps is not None:
        return max(settings.max_steps, 1)
    batches = _epoch_batch_limit(settings, train_loader)
    updates_per_epoch = math.ceil(batches / settings.gradient_accumulation)
    return max(updates_per_epoch * settings.epochs, 1)


def _epoch_batch_limit(settings: FineTuneSettings, train_loader: DataLoader) -> int:
    batches = len(train_loader)
    if settings.max_train_batches is not None:
        batches = min(batches, settings.max_train_batches)
    return max(batches, 1)


def _record_metrics(
    row: TrainingMetricsRow,
    metrics_csv: Path,
    metrics_jsonl: Path,
    writer: Any,
) -> None:
    append_training_metrics_csv(row, metrics_csv)
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with metrics_jsonl.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row.__dict__) + "\n")
    if writer is not None:
        writer.add_scalar("train/loss", row.training_loss, row.global_step)
        writer.add_scalar("train/learning_rate", row.learning_rate, row.global_step)
        writer.add_scalar("train/gradient_norm", row.gradient_norm or 0.0, row.global_step)
        writer.add_scalar("train/tokens_per_second", row.tokens_per_second, row.global_step)
        if row.validation_loss is not None:
            writer.add_scalar("validation/loss", row.validation_loss, row.global_step)
        if row.perplexity is not None:
            writer.add_scalar("validation/perplexity", row.perplexity, row.global_step)


def _print_metrics(epoch: int, total_epochs: int, row: TrainingMetricsRow) -> None:
    print(
        " | ".join(
            [
                f"Epoch {epoch}/{total_epochs}",
                f"Step {row.global_step}",
                f"Loss {row.training_loss:.6f}",
                f"Validation {_format_optional(row.validation_loss)}",
                f"Perplexity {_format_optional(row.perplexity)}",
                f"LR {row.learning_rate:.8f}",
                f"Grad {row.gradient_norm or 0.0:.4f}",
                f"Tokens/sec {row.tokens_per_second:.2f}",
                f"ETA {_format_duration(row.eta_seconds)}",
            ]
        )
    )


def _write_loss_curve(metrics_csv: Path, output_path: Path) -> None:
    rows = read_training_metrics_csv(metrics_csv)
    write_loss_curve_png(loss_history_from_training_metrics(rows), output_path)


def _create_summary_writer(log_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(log_dir / "tensorboard"))


def _close_summary_writer(writer: Any) -> None:
    if writer is not None:
        writer.flush()
        writer.close()


def _rotate_periodic_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(
        output_dir.glob("step_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_path in checkpoints[keep_last:]:
        old_path.unlink(missing_ok=True)


def _resolve_resume_checkpoint(resume: str, output_dir: Path) -> Path:
    normalized = resume.strip()
    if normalized.lower() == "latest":
        return _existing_checkpoint(output_dir / "latest.pt")
    if normalized.lower() == "best":
        return _existing_checkpoint(output_dir / "best.pt")
    return _existing_checkpoint(_resolve(Path(normalized)))


def _existing_checkpoint(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path


def _effective_mixed_precision(mixed_precision: str, device: torch.device) -> str:
    if mixed_precision == "fp16" and device.type != "cuda":
        LOGGER.warning("fp16 requires CUDA; using full precision on %s.", device)
        return "none"
    if mixed_precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        LOGGER.warning("bf16 is not supported on this CUDA device; using full precision.")
        return "none"
    return mixed_precision


def _should_stop_early(settings: FineTuneSettings, stale_evaluations: int) -> bool:
    return (
        settings.early_stopping_patience > 0
        and stale_evaluations >= settings.early_stopping_patience
    )


def _eta_seconds(global_step: int, total_steps: int, elapsed_seconds: float) -> float:
    if global_step <= 0:
        return 0.0
    remaining = max(total_steps - global_step, 0)
    return remaining * (elapsed_seconds / global_step)


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _validate_settings(settings: FineTuneSettings) -> None:
    if settings.epochs <= 0:
        raise FineTuningError("epochs must be greater than zero.")
    if settings.batch_size <= 0:
        raise FineTuningError("batch_size must be greater than zero.")
    if settings.learning_rate <= 0:
        raise FineTuningError("learning_rate must be greater than zero.")
    if settings.weight_decay < 0:
        raise FineTuningError("weight_decay must be non-negative.")
    if not 0 <= settings.warmup_ratio < 1:
        raise FineTuningError("warmup_ratio must be at least 0 and less than 1.")
    if settings.max_length <= 1:
        raise FineTuningError("max_length must be greater than one.")
    if settings.gradient_accumulation <= 0:
        raise FineTuningError("gradient_accumulation must be greater than zero.")
    if settings.save_every <= 0:
        raise FineTuningError("save_every must be greater than zero.")
    if settings.eval_every <= 0:
        raise FineTuningError("eval_every must be greater than zero.")
    if settings.keep_last <= 0:
        raise FineTuningError("keep_last must be greater than zero.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a GenPy Code LLM.")
    parser.add_argument("--config", type=Path, default=Path("configs/fine_tune.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--validation-batches", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_path_value(value: object) -> Path:
    if not isinstance(value, str | Path):
        raise FineTuningError(f"Expected path value, found {type(value).__name__}.")
    return _resolve(Path(value))


def _optional_path_value(value: object) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _resolve_path_value(value)


def _optional_resolved(value: Path | None) -> Path | None:
    return None if value is None else _resolve(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, whole_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {whole_seconds}s"
    return f"{whole_seconds}s"


def _report_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        logging.exception("Fine-tuning failed.")
    elif isinstance(exc, (CheckpointError, FineTuningError, FileNotFoundError, ValueError)):
        print(f"Error: {exc}", file=sys.stderr)
    else:
        raise exc


if __name__ == "__main__":
    raise SystemExit(main())
