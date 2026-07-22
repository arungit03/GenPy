"""Train a GenPy Code LLM base model from streaming code shards."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.code_evaluation import resolve_code_checkpoint
from genpy_llm.code_tokenizer import ensure_code_tokenizer
from genpy_llm.code_training import (
    create_code_dataloader,
    create_code_model,
    load_code_config,
    select_device,
    train_code_steps,
    validate_code_training_artifacts,
)
from genpy_llm.utils import set_seed

LOGGER = logging.getLogger("genpy_llm.train_code_model")


def main() -> int:
    args = _parse_args()
    _configure_logging(args.debug)
    try:
        LOGGER.debug("loading config: %s", _resolve(args.config))
        config = load_code_config(_resolve(args.config))
        LOGGER.debug(
            "config loaded: project=%s model_layers=%s embedding_dim=%s context_length=%s "
            "batch_size=%s accumulation=%s mixed_precision=%s",
            config.project_name,
            config.model.num_layers,
            config.model.embedding_dim,
            config.model.context_length,
            config.training.batch_size,
            config.training.gradient_accumulation_steps,
            config.training.mixed_precision,
        )
        if args.checkpoint_dir is not None:
            config = _with_checkpoint_directory(config, _resolve(args.checkpoint_dir))
            LOGGER.debug("checkpoint directory overridden: %s", config.checkpoint.directory)
        tokenizer = ensure_code_tokenizer(
            tokenizer_path=config.tokenizer.path,
            metadata_path=config.tokenizer.metadata_path,
            project_root=PROJECT_ROOT,
            vocab_size=config.tokenizer.vocab_size,
            train_pattern=config.streaming_dataset.train_pattern,
        )
        artifacts = validate_code_training_artifacts(config)
        LOGGER.debug(
            "training artifacts verified: tokenizer=%s metadata=%s train_shards=%s "
            "validation_shards=%s checkpoint_dir=%s mixed_precision=%s",
            artifacts.tokenizer_path,
            artifacts.tokenizer_metadata_path,
            len(artifacts.train_shards),
            len(artifacts.validation_shards),
            artifacts.checkpoint_directory,
            artifacts.mixed_precision,
        )
        set_seed(config.seed)
        LOGGER.debug("seed set: %s", config.seed)
        LOGGER.debug("loading tokenizer: %s", config.tokenizer.path)
        LOGGER.debug(
            "tokenizer loaded: vocab_size=%s pad_token_id=%s eos_token_id=%s",
            tokenizer.vocab_size,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
        )
        LOGGER.debug(
            "selecting device: requested=%s configured=%s",
            args.device,
            config.training.device,
        )
        device = select_device(args.device or config.training.device)
        LOGGER.debug("device selected: %s", device)
        if config.training.mixed_precision == "fp16" and device.type != "cuda":
            print("Warning: fp16 requires CUDA; using full precision on this device.")
            config = _with_mixed_precision(config, "none")
            LOGGER.debug("mixed precision adjusted for device: %s", config.training.mixed_precision)
        model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
        LOGGER.debug(
            "model created: parameters=%s trainable=%s gradient_checkpointing=%s",
            model.parameter_count,
            model.trainable_parameter_count,
            model.gradient_checkpointing,
        )
        train_loader = create_code_dataloader(
            config,
            tokenizer,
            split="train",
            batch_size=config.training.batch_size,
        )
        LOGGER.debug(
            "train dataloader created: batch_size=%s num_workers=%s pin_memory=%s shards=%s",
            config.training.batch_size,
            config.streaming_dataset.num_workers,
            config.streaming_dataset.pin_memory,
            len(train_loader.dataset.shard_paths),
        )
        validation_loader = create_code_dataloader(
            config,
            tokenizer,
            split="validation",
            batch_size=config.training.batch_size,
        )
        LOGGER.debug(
            "validation dataloader created: batch_size=%s num_workers=%s pin_memory=%s shards=%s",
            config.training.batch_size,
            config.streaming_dataset.num_workers,
            config.streaming_dataset.pin_memory,
            len(validation_loader.dataset.shard_paths),
        )
        resume_checkpoint = _resolve_resume_checkpoint(args, config)
        if resume_checkpoint and resume_checkpoint.name == "genpy_best.pt":
            print("Warning: old general-text checkpoints are incompatible with code tokenizers.")
        if resume_checkpoint is not None:
            LOGGER.debug("resume checkpoint resolved: %s", resume_checkpoint)
        max_batches = args.max_batches
        validation_batches = args.validation_batches
        if args.debug and args.max_steps is not None and validation_batches is None:
            validation_batches = 1
            LOGGER.debug(
                "debug validation batch cap enabled: validation_batches=%s "
                "(pass --validation-batches to override)",
                validation_batches,
            )
        LOGGER.debug("calling train_code_steps")
        result = train_code_steps(
            model=model,
            tokenizer=tokenizer,
            config=config,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            max_steps=args.max_steps or config.training.max_steps,
            max_batches=max_batches,
            validation_batches=validation_batches,
            checkpoint_path=resume_checkpoint,
            evaluation_dir=_resolve(args.evaluation_dir),
            logger=LOGGER,
        )
        LOGGER.debug("train_code_steps returned: global_step=%s", result.global_step)
    except KeyboardInterrupt:
        print("Interrupted before completion.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        _report_error(exc, debug=args.debug)
        return 1
    print("GenPy Code LLM Base Training")
    print("============================")
    print(f"Device: {device}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Global step: {result.global_step}")
    print(f"Training loss: {result.training_loss:.6f}")
    print(f"Validation loss: {result.validation_loss}")
    print(f"Tokens processed: {result.tokens_processed}")
    print(f"Tokens per second: {result.tokens_per_second:.2f}")
    print(f"Resume checkpoint: {resume_checkpoint}")
    print(f"Latest checkpoint: {result.latest_checkpoint}")
    print(f"Best checkpoint: {result.best_checkpoint}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a code GPT model from streaming shards.")
    parser.add_argument("--config", type=Path, default=Path("configs/code_small.yaml"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from latest, best, or a checkpoint path.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--evaluation-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--validation-batches", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _with_mixed_precision(config, value: str):
    from dataclasses import replace

    return replace(config, training=replace(config.training, mixed_precision=value))


def _with_checkpoint_directory(config, directory: Path):
    from dataclasses import replace

    return replace(config, checkpoint=replace(config.checkpoint, directory=directory))


def _resolve_resume_checkpoint(args: argparse.Namespace, config) -> Path | None:
    if args.resume is None:
        return None
    if args.checkpoint is not None and str(args.resume).lower() == "latest":
        legacy_path = _resolve(args.checkpoint)
        if not legacy_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {legacy_path}")
        return legacy_path
    return resolve_code_checkpoint(
        args.resume,
        checkpoint_directory=config.checkpoint.directory,
        filename_prefix=config.checkpoint.filename_prefix,
        best_filename=config.checkpoint.best_filename,
        project_root=PROJECT_ROOT,
    )


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_optional(path: Path | None) -> Path | None:
    if path is None:
        return None
    return _resolve(path)


def _report_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        print("Code model training failed with traceback:", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    else:
        print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
