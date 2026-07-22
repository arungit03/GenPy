"""Main entry point for GenPy LLM."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import load_config
from genpy_llm.device import select_device
from genpy_llm.logging_utils import setup_logging
from genpy_llm.utils import ensure_directories, set_seed


def main() -> None:
    """Initialize project setup components without training a model."""

    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()

    ensure_directories(
        [
            config.paths.data_dir,
            config.paths.raw_data_dir,
            config.paths.processed_data_dir,
            config.paths.tokenized_data_dir,
            config.data.vocabulary_dir,
            config.data.dataset_dir,
            config.paths.checkpoints_dir,
            config.paths.logs_dir,
            config.paths.notebooks_dir,
        ]
    )

    logger = setup_logging(
        log_dir=config.paths.logs_dir,
        log_file=config.logging.log_file,
        level=config.logging.level,
    )
    set_seed(config.training.seed)
    device = select_device(config.training.device)

    logger.info("GenPy LLM Step 20 is ready.")
    logger.info("Selected device: %s", device)
    logger.info("Text preprocessing is available through scripts/preprocess_text.py.")
    logger.info("Text tokenization is available through scripts/tokenize_text.py.")
    logger.info("Vocabulary building is available through scripts/build_vocabulary.py.")
    logger.info("Dataset preparation is available through scripts/prepare_dataset.py.")
    logger.info("Token embeddings are available through scripts/inspect_embeddings.py.")
    logger.info("Positional encoding is available through scripts/inspect_positional_encoding.py.")
    logger.info("Causal self-attention is available through scripts/inspect_self_attention.py.")
    logger.info(
        "Multi-head causal self-attention is available through "
        "scripts/inspect_multi_head_attention.py."
    )
    logger.info("Feed-forward network is available through scripts/inspect_feed_forward.py.")
    logger.info(
        "Layer normalization and residual connections are available through "
        "scripts/inspect_norm_residual.py."
    )
    logger.info(
        "Transformer block inspection is available through scripts/inspect_transformer_block.py."
    )
    logger.info("GPT decoder inspection is available through scripts/inspect_gpt_model.py.")
    logger.info("Training loop smoke test is available through scripts/run_training_loop.py.")
    logger.info(
        "Loss and optimizer inspection is available through scripts/inspect_loss_optimizer.py."
    )
    logger.info("Checkpoint inspection is available through scripts/inspect_checkpoint.py.")
    logger.info("Checkpointed GPT training is available through scripts/train_gpt.py.")
    logger.info("Text generation is available through scripts/generate_text.py.")
    logger.info("Supervised fine-tuning is available through scripts/fine_tune_gpt.py.")
    logger.info("Model benchmarking is available through scripts/benchmark_model.py.")
    logger.info("Local web interface is available through scripts/run_web_interface.py.")
    logger.info("Code data download is available through scripts/download_code_training_data.py.")
    logger.info("Code tokenizer training is available through scripts/train_code_tokenizer.py.")
    logger.info("Code model training is available through scripts/train_code_model.py.")
    logger.info(
        "Code instruction fine-tuning is available through scripts/fine_tune_code_model.py."
    )
    logger.info("Code generation is available through scripts/generate_code.py.")
    logger.info("Configured raw input: %s", config.data.input_file)
    logger.info("Configured processed output: %s", config.data.output_file)
    logger.info("Configured tokenization method: %s", config.tokenization.method)
    logger.info("Configured tokenized output: %s", config.data.tokenized_file)
    logger.info("Configured vocabulary max size: %s", config.vocabulary.max_size)
    logger.info("Configured vocabulary min frequency: %s", config.vocabulary.min_frequency)
    logger.info("Configured vocabulary output: %s", config.data.vocabulary_file)
    logger.info("Configured dataset context length: %s", config.dataset.context_length)
    logger.info("Configured dataset stride: %s", config.dataset.stride)
    logger.info("Configured dataset batch size: %s", config.dataset.batch_size)
    logger.info("Configured dataset sequence mode: %s", config.dataset.sequence_mode)
    logger.info("Configured encoded input: %s", config.data.encoded_file)
    logger.info("Configured dataset output directory: %s", config.data.dataset_dir)
    logger.info("Configured embedding dimension: %s", config.embeddings.embedding_dim)
    logger.info("Configured embedding initialization: %s", config.embeddings.initialization)
    logger.info("Configured embedding scaling: %s", config.embeddings.scale_embeddings)
    logger.info("Configured embedding freeze: %s", config.embeddings.freeze_embeddings)
    logger.info("Configured embedding vocabulary: %s", config.data.vocabulary_file)
    logger.info("Configured positional encoding type: %s", config.positional_encoding.type)
    logger.info(
        "Configured positional max sequence length: %s",
        config.positional_encoding.max_sequence_length,
    )
    logger.info("Configured positional dropout: %s", config.positional_encoding.dropout)
    logger.info("Configured attention dropout: %s", config.attention.dropout)
    logger.info("Configured attention use bias: %s", config.attention.use_bias)
    logger.info("Configured attention causal mode: %s", config.attention.causal)
    logger.info("Configured attention heads: %s", config.model.num_heads)
    logger.info("Configured model layers: %s", config.model.num_layers)
    logger.info("Configured model dropout: %s", config.model.dropout)
    logger.info("Configured model use bias: %s", config.model.use_bias)
    logger.info("Configured model tied embeddings: %s", config.model.tie_embeddings)
    logger.info("Configured model initialization std: %s", config.model.initialization_std)
    logger.info("Configured training epochs: %s", config.training.epochs)
    logger.info(
        "Configured gradient accumulation steps: %s",
        config.training.gradient_accumulation_steps,
    )
    logger.info("Configured max gradient norm: %s", config.training.max_grad_norm)
    logger.info("Configured train log interval: %s", config.training.log_every_steps)
    logger.info(
        "Configured validation interval: %s",
        config.training.validate_every_epochs,
    )
    logger.info("Configured loss type: %s", config.loss.type)
    logger.info("Configured loss ignore padding: %s", config.loss.ignore_padding)
    logger.info("Configured loss label smoothing: %s", config.loss.label_smoothing)
    logger.info("Configured optimizer type: %s", config.optimizer.type)
    logger.info("Configured optimizer learning rate: %s", config.optimizer.learning_rate)
    logger.info("Configured optimizer weight decay: %s", config.optimizer.weight_decay)
    logger.info("Configured optimizer beta1: %s", config.optimizer.beta1)
    logger.info("Configured optimizer beta2: %s", config.optimizer.beta2)
    logger.info("Configured optimizer epsilon: %s", config.optimizer.epsilon)
    logger.info(
        "Configured optimizer separate weight decay: %s",
        config.optimizer.separate_weight_decay,
    )
    logger.info("Configured checkpoint directory: %s", config.checkpoint.directory)
    logger.info("Configured checkpoint interval: %s", config.checkpoint.save_every_epochs)
    logger.info("Configured checkpoint retention count: %s", config.checkpoint.keep_last)
    logger.info("Configured best checkpoint saving: %s", config.checkpoint.save_best)
    logger.info("Configured checkpoint monitor: %s", config.checkpoint.monitor)
    logger.info("Configured checkpoint mode: %s", config.checkpoint.mode)
    logger.info("Configured checkpoint filename prefix: %s", config.checkpoint.filename_prefix)
    logger.info("Configured generation max new tokens: %s", config.generation.max_new_tokens)
    logger.info("Configured generation temperature: %s", config.generation.temperature)
    logger.info("Configured generation top-k: %s", config.generation.top_k)
    logger.info("Configured generation top-p: %s", config.generation.top_p)
    logger.info("Configured generation sampling enabled: %s", config.generation.do_sample)
    logger.info(
        "Configured generation repetition penalty: %s",
        config.generation.repetition_penalty,
    )
    logger.info("Configured generation stop on EOS: %s", config.generation.stop_on_eos)
    logger.info("Configured fine-tuning dataset: %s", config.fine_tuning.dataset_file)
    logger.info("Configured fine-tuning output directory: %s", config.fine_tuning.output_directory)
    logger.info("Configured fine-tuning epochs: %s", config.fine_tuning.epochs)
    logger.info("Configured fine-tuning batch size: %s", config.fine_tuning.batch_size)
    logger.info("Configured fine-tuning learning rate: %s", config.fine_tuning.learning_rate)
    logger.info("Configured fine-tuning weight decay: %s", config.fine_tuning.weight_decay)
    logger.info("Configured fine-tuning max grad norm: %s", config.fine_tuning.max_grad_norm)
    logger.info(
        "Configured fine-tuning gradient accumulation steps: %s",
        config.fine_tuning.gradient_accumulation_steps,
    )
    logger.info(
        "Configured fine-tuning freeze embeddings: %s",
        config.fine_tuning.freeze_embeddings,
    )
    logger.info(
        "Configured fine-tuning frozen first layers: %s",
        config.fine_tuning.freeze_first_n_layers,
    )
    logger.info(
        "Configured fine-tuning train/validation ratio: %s",
        config.fine_tuning.train_validation_ratio,
    )
    logger.info("Configured mixed precision: %s", config.optimization.mixed_precision)
    logger.info("Configured torch.compile enabled: %s", config.optimization.torch_compile)
    logger.info("Configured torch.compile mode: %s", config.optimization.compile_mode)
    logger.info(
        "Configured gradient checkpointing: %s",
        config.optimization.gradient_checkpointing,
    )
    logger.info("Configured quantization: %s", config.optimization.quantization)
    logger.info(
        "Configured benchmark warmup steps: %s",
        config.optimization.benchmark_warmup_steps,
    )
    logger.info("Configured benchmark steps: %s", config.optimization.benchmark_steps)
    logger.info("Configured web interface title: %s", config.web_interface.title)
    logger.info("Configured web interface host: %s", config.web_interface.host)
    logger.info("Configured web interface port: %s", config.web_interface.port)
    logger.info("Configured web interface sharing: %s", config.web_interface.share)
    logger.info(
        "Configured web interface default checkpoint: %s",
        config.web_interface.default_checkpoint,
    )
    logger.info("Configured FFN hidden multiplier: %s", config.feed_forward.hidden_multiplier)
    logger.info("Configured FFN hidden dimension: %s", config.feed_forward.hidden_dim)
    logger.info("Configured FFN activation: %s", config.feed_forward.activation)
    logger.info("Configured FFN dropout: %s", config.feed_forward.dropout)
    logger.info("Configured normalization type: %s", config.normalization.type)
    logger.info("Configured normalization epsilon: %s", config.normalization.epsilon)
    logger.info(
        "Configured normalization affine parameters: %s",
        config.normalization.elementwise_affine,
    )
    logger.info("Configured residual dropout: %s", config.residual.dropout)
    logger.info(
        "Configured transformer block attention dropout: %s",
        config.transformer_block.attention_dropout,
    )
    logger.info(
        "Configured transformer block residual dropout: %s",
        config.transformer_block.residual_dropout,
    )
    logger.info(
        "Configured transformer block FFN dropout: %s",
        config.transformer_block.feed_forward_dropout,
    )


if __name__ == "__main__":
    main()
