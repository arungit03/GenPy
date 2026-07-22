"""Inspect the complete untrained GPT decoder model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import ConfigError, load_config
from genpy_llm.dataset import DatasetPreparationError, load_dataset_split
from genpy_llm.device import select_device
from genpy_llm.gpt import GPTModel, GPTModelError, create_gpt_model
from genpy_llm.logging_utils import setup_logging
from genpy_llm.utils import count_trainable_parameters, set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Parse arguments and inspect one untrained GPT decoder forward pass."""

    _configure_console()
    args = _parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        app_config = load_config(_resolve_optional_path(args.config))
        logger = setup_logging(
            log_dir=app_config.paths.logs_dir,
            log_file=app_config.logging.log_file,
            level="DEBUG" if args.debug else app_config.logging.level,
        )
        set_seed(args.seed if args.seed is not None else app_config.training.seed)
        device = select_device(args.device or app_config.training.device)
        model, metadata = create_gpt_model(
            vocabulary_path=app_config.data.vocabulary_file,
            config=app_config,
        )
        model = model.to(device)
        model.eval()

        input_ids, padding_mask, dataset_path = _load_input_ids(args, app_config, model, device)
        with torch.no_grad():
            if args.show_attention:
                logits, attention_maps = model(
                    input_ids,
                    padding_mask=padding_mask,
                    return_attention=True,
                )
            else:
                logits = model(input_ids, padding_mask=padding_mask)
                attention_maps = []
    except (
        ConfigError,
        DatasetPreparationError,
        FileNotFoundError,
        GPTModelError,
        IsADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("GPT decoder inspection completed successfully.")
    _print_summary(
        model=model,
        input_ids=input_ids,
        logits=logits,
        attention_maps=attention_maps,
        dataset_path=dataset_path,
        device=device,
        show_attention=args.show_attention,
        show_layer=args.show_layer,
        show_head=args.show_head,
    )
    print()
    print(metadata.summary())
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GenPy LLM GPT decoder.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--dataset", type=str, default=None, help="Prepared dataset split file.")
    parser.add_argument("--show-batch", action="store_true")
    parser.add_argument("--show-attention", action="store_true")
    parser.add_argument("--show-layer", type=int, default=0)
    parser.add_argument("--show-head", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _load_input_ids(
    args: argparse.Namespace,
    app_config,
    model: GPTModel,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, Path | None]:
    if not args.show_batch:
        sequence_length = min(4, model.context_length)
        input_ids = torch.randint(0, model.vocab_size, (2, sequence_length), device=device)
        return input_ids, None, None

    dataset_path = _resolve_path(args.dataset, app_config.data.train_dataset_file)
    dataset = load_dataset_split(dataset_path)
    if len(dataset) == 0:
        raise DatasetPreparationError("Dataset split is empty.")
    loader = DataLoader(dataset, batch_size=app_config.dataset.batch_size, shuffle=False)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    padding_mask = batch.get("attention_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)
    return input_ids, padding_mask, dataset_path


def _print_summary(
    model: GPTModel,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    attention_maps: list[torch.Tensor],
    dataset_path: Path | None,
    device: torch.device,
    show_attention: bool,
    show_layer: int,
    show_head: int,
) -> None:
    predictions = logits.argmax(dim=-1)
    print("GenPy LLM GPT Decoder")
    print("=====================")
    print(f"Dataset file: {dataset_path if dataset_path is not None else 'synthetic sample'}")
    print(f"Device: {device}")
    print(f"Input shape: {list(input_ids.shape)}")
    print(f"Logits shape: {list(logits.shape)}")
    print(f"Predicted token IDs shape: {list(predictions.shape)}")
    print(f"First predicted token IDs: {predictions[0, : min(12, predictions.shape[1])].tolist()}")
    print(f"Layer count: {model.num_layers}")
    print(f"Total parameters: {model.parameter_count}")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
    print(f"Weight tying enabled: {model.tie_embeddings}")
    print(f"Embedding/head share parameter: {model.embeddings_are_tied()}")
    print(f"Finite logits: {bool(torch.isfinite(logits).all().item())}")
    if show_attention:
        _validate_attention_selection(show_layer, show_head, model, attention_maps)
        selected = attention_maps[show_layer]
        print(f"Attention map count: {len(attention_maps)}")
        print(f"Selected attention shape: {list(selected.shape)}")
        print(f"Causal masking confirmed: {_future_probabilities_are_zero(selected)}")
        _print_attention_matrix(selected[0, show_head], show_layer, show_head)
    print()
    print("Predictions and attention patterns are random because the model is untrained.")


def _validate_attention_selection(
    show_layer: int,
    show_head: int,
    model: GPTModel,
    attention_maps: list[torch.Tensor],
) -> None:
    if show_layer < 0 or show_layer >= len(attention_maps):
        raise GPTModelError(
            f"show_layer must be between 0 and {len(attention_maps) - 1}. Received {show_layer}."
        )
    if show_head < 0 or show_head >= model.num_heads:
        raise GPTModelError(
            f"show_head must be between 0 and {model.num_heads - 1}. Received {show_head}."
        )


def _future_probabilities_are_zero(weights: torch.Tensor) -> bool:
    sequence_length = weights.shape[-1]
    future_mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=weights.device),
        diagonal=1,
    )
    future_values = weights[..., future_mask]
    return bool(torch.allclose(future_values, torch.zeros_like(future_values), atol=1e-7))


def _print_attention_matrix(
    matrix: torch.Tensor,
    layer_index: int,
    head_index: int,
    max_rows: int = 8,
) -> None:
    shown = min(matrix.shape[0], max_rows)
    print()
    print(f"Attention matrix preview for layer {layer_index}, head {head_index}")
    print("================================================")
    for row_index in range(shown):
        values = [f"{value:.4f}" for value in matrix[row_index, :shown].tolist()]
        print(f"{row_index:02d}: {' '.join(values)}")
    if matrix.shape[0] > shown:
        print(f"... truncated to first {shown} rows/columns")


def _resolve_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return _resolve_against_project_root(Path(value))


def _resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    return _resolve_against_project_root(Path(value))


def _resolve_against_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report_error(exc: Exception, debug: bool) -> None:
    logger = logging.getLogger("genpy_llm")
    if debug:
        logger.exception("GPT decoder inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
