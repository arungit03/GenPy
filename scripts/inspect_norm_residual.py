"""Inspect GPT-style layer normalization and residual connections."""

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

from genpy_llm.attention import AttentionError, MultiHeadCausalSelfAttention
from genpy_llm.config import ConfigError, load_config
from genpy_llm.dataset import DatasetPreparationError, load_dataset_split
from genpy_llm.device import select_device
from genpy_llm.embeddings import EmbeddingError, create_token_embedding
from genpy_llm.feed_forward import (
    FeedForwardError,
    FeedForwardNetwork,
    resolve_feed_forward_hidden_dim,
)
from genpy_llm.logging_utils import setup_logging
from genpy_llm.normalization import GPTLayerNorm, NormalizationError
from genpy_llm.positional_encoding import (
    GPTInputEmbedding,
    PositionalEncoding,
    PositionalEncodingError,
)
from genpy_llm.residual import PreNormResidual, ResidualConnection, ResidualError
from genpy_llm.utils import count_trainable_parameters, set_seed
from genpy_llm.vocabulary import VocabularyError


def main() -> int:
    """Parse arguments and inspect normalization/residual behavior."""

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

        epsilon = args.epsilon if args.epsilon is not None else app_config.normalization.epsilon
        dropout = args.dropout if args.dropout is not None else app_config.residual.dropout
        embedding_dim = app_config.model.embedding_dim

        layer_norm = GPTLayerNorm(
            embedding_dim=embedding_dim,
            epsilon=epsilon,
            elementwise_affine=app_config.normalization.elementwise_affine,
        ).to(device)
        residual = ResidualConnection(dropout=dropout).to(device)
        layer_norm.eval()
        residual.eval()

        input_embedding = _build_input_embedding(app_config).to(device)
        input_embedding.eval()
        attention = _build_attention(app_config).to(device) if args.use_attention else None
        ffn = _build_ffn(app_config).to(device) if args.use_ffn else None
        if attention is not None:
            attention.eval()
        if ffn is not None:
            ffn.eval()

        hidden_states, padding_mask, dataset_path = _load_hidden_states(
            args=args,
            app_config=app_config,
            input_embedding=input_embedding,
            device=device,
        )
    except (
        AttentionError,
        ConfigError,
        DatasetPreparationError,
        EmbeddingError,
        FeedForwardError,
        FileNotFoundError,
        IsADirectoryError,
        NormalizationError,
        OSError,
        PositionalEncodingError,
        ResidualError,
        RuntimeError,
        ValueError,
        VocabularyError,
    ) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    logger.info("Normalization and residual inspection completed successfully.")

    try:
        with torch.no_grad():
            normalized = layer_norm(hidden_states)
            residual_output = residual(hidden_states, torch.zeros_like(hidden_states))
            current_states = hidden_states
            attention_output = None
            ffn_output = None
            if attention is not None:
                attention_block = PreNormResidual(
                    embedding_dim=embedding_dim,
                    sublayer=attention,
                    normalization_epsilon=epsilon,
                    residual_dropout=dropout,
                    elementwise_affine=app_config.normalization.elementwise_affine,
                ).to(device)
                attention_block.eval()
                current_states = attention_block(current_states, padding_mask=padding_mask)
                attention_output = current_states
            if ffn is not None:
                ffn_block = PreNormResidual(
                    embedding_dim=embedding_dim,
                    sublayer=ffn,
                    normalization_epsilon=epsilon,
                    residual_dropout=dropout,
                    elementwise_affine=app_config.normalization.elementwise_affine,
                ).to(device)
                ffn_block.eval()
                current_states = ffn_block(current_states)
                ffn_output = current_states
    except (AttentionError, FeedForwardError, NormalizationError, ResidualError) as exc:
        _report_error(exc, debug=args.debug)
        return 1

    _print_summary(
        hidden_states=hidden_states,
        normalized=normalized,
        residual_output=residual_output,
        layer_norm=layer_norm,
        residual=residual,
        device=device,
        dataset_path=dataset_path,
        attention=attention,
        ffn=ffn,
        attention_output=attention_output,
        ffn_output=ffn_output,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect GenPy LLM layer normalization and residual connections."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--dataset", type=str, default=None, help="Prepared dataset split file.")
    parser.add_argument("--show-batch", action="store_true")
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--use-ffn", action="store_true")
    parser.add_argument("--dropout", type=_dropout, default=None)
    parser.add_argument("--epsilon", type=_positive_float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _build_input_embedding(app_config) -> GPTInputEmbedding:
    token_embedding, _token_metadata = create_token_embedding(
        vocabulary_path=app_config.data.vocabulary_file,
        embedding_config=app_config.embeddings,
        expected_vocab_size=app_config.model.vocab_size,
        encoding=app_config.data.encoding,
    )
    positional_encoding = PositionalEncoding(
        embedding_dim=app_config.embeddings.embedding_dim,
        max_sequence_length=app_config.positional_encoding.max_sequence_length,
        encoding_type=app_config.positional_encoding.type,
        dropout=app_config.positional_encoding.dropout,
        initialization_std=app_config.positional_encoding.initialization_std,
    )
    return GPTInputEmbedding(token_embedding, positional_encoding)


def _build_attention(app_config) -> MultiHeadCausalSelfAttention:
    return MultiHeadCausalSelfAttention(
        embedding_dim=app_config.model.embedding_dim,
        num_heads=app_config.model.num_heads,
        max_sequence_length=app_config.positional_encoding.max_sequence_length,
        dropout=app_config.attention.dropout,
        use_bias=app_config.attention.use_bias,
    )


def _build_ffn(app_config) -> FeedForwardNetwork:
    hidden_dim = resolve_feed_forward_hidden_dim(
        embedding_dim=app_config.model.embedding_dim,
        hidden_multiplier=app_config.feed_forward.hidden_multiplier,
        hidden_dim=app_config.feed_forward.hidden_dim,
    )
    return FeedForwardNetwork(
        embedding_dim=app_config.model.embedding_dim,
        hidden_dim=hidden_dim,
        activation=app_config.feed_forward.activation,
        dropout=app_config.feed_forward.dropout,
        use_bias=app_config.feed_forward.use_bias,
        initialization_std=app_config.feed_forward.initialization_std,
    )


def _load_hidden_states(
    args: argparse.Namespace,
    app_config,
    input_embedding: GPTInputEmbedding,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, Path | None]:
    if not args.show_batch:
        hidden_states = torch.randn(2, 4, app_config.model.embedding_dim, device=device)
        return hidden_states, None, None

    dataset_path = _resolve_path(args.dataset, app_config.data.train_dataset_file)
    dataset = load_dataset_split(dataset_path)
    if len(dataset) == 0:
        raise DatasetPreparationError("Dataset split is empty.")
    loader = DataLoader(dataset, batch_size=app_config.dataset.batch_size, shuffle=False)
    batch = next(iter(loader))
    token_ids = batch["input_ids"].to(device)
    padding_mask = batch.get("attention_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)
    with torch.no_grad():
        hidden_states = input_embedding(token_ids)
    return hidden_states, padding_mask, dataset_path


def _print_summary(
    hidden_states: torch.Tensor,
    normalized: torch.Tensor,
    residual_output: torch.Tensor,
    layer_norm: GPTLayerNorm,
    residual: ResidualConnection,
    device: torch.device,
    dataset_path: Path | None,
    attention: MultiHeadCausalSelfAttention | None,
    ffn: FeedForwardNetwork | None,
    attention_output: torch.Tensor | None,
    ffn_output: torch.Tensor | None,
) -> None:
    before_mean = hidden_states.mean(dim=-1)
    after_mean = normalized.mean(dim=-1)
    after_variance = normalized.var(dim=-1, unbiased=False)
    total_parameters = layer_norm.parameter_count
    total_parameters += attention.parameter_count if attention is not None else 0
    total_parameters += ffn.parameter_count if ffn is not None else 0

    print("GenPy LLM LayerNorm and Residual Connections")
    print("============================================")
    print(f"Dataset file: {dataset_path if dataset_path is not None else 'synthetic sample'}")
    print(f"Device: {device}")
    print(f"LayerNorm epsilon: {layer_norm.epsilon}")
    print(f"LayerNorm affine: {layer_norm.elementwise_affine}")
    print(f"Residual dropout: {residual.dropout.p}")
    print(f"LayerNorm parameters: {layer_norm.parameter_count}")
    print(f"Trainable LayerNorm parameters: {count_trainable_parameters(layer_norm)}")
    print(f"Total inspected parameters: {total_parameters}")
    print()
    print(f"Input shape:              {list(hidden_states.shape)}")
    print(f"Normalized shape:         {list(normalized.shape)}")
    print(f"Residual output shape:    {list(residual_output.shape)}")
    if attention_output is not None:
        print(f"Attention pre-norm shape: {list(attention_output.shape)}")
    if ffn_output is not None:
        print(f"FFN pre-norm shape:       {list(ffn_output.shape)}")
    print(f"Mean before normalization:     {_tensor_summary(before_mean)}")
    print(f"Mean after normalization:      {_tensor_summary(after_mean)}")
    print(f"Variance after normalization:  {_tensor_summary(after_variance)}")
    print(f"Finite normalized output:      {bool(torch.isfinite(normalized).all().item())}")
    print()
    print("All weights are randomly initialized and untrained.")


def _tensor_summary(tensor: torch.Tensor) -> str:
    if tensor.numel() == 0:
        return "empty tensor"
    return (
        f"mean={tensor.mean().item():.8f}, "
        f"min={tensor.min().item():.8f}, "
        f"max={tensor.max().item():.8f}"
    )


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return number


def _dropout(value: str) -> float:
    number = float(value)
    if not 0.0 <= number < 1.0:
        raise argparse.ArgumentTypeError("dropout must be at least 0.0 and less than 1.0.")
    return number


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
        logger.exception("Normalization/residual inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
