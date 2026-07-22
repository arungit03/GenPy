"""Configuration loading for GenPy LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = {
    "project",
    "paths",
    "data",
    "preprocessing",
    "tokenizer",
    "model",
    "training",
    "generation",
    "logging",
}


@dataclass(frozen=True)
class ProjectConfig:
    """Project metadata."""

    name: str
    version: str
    description: str


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the project."""

    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    tokenized_data_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    notebooks_dir: Path


@dataclass(frozen=True)
class DataConfig:
    """Data file settings for preprocessing."""

    raw_dir: Path
    processed_dir: Path
    tokenized_dir: Path
    vocabulary_dir: Path
    dataset_dir: Path
    input_file: Path
    output_file: Path
    tokenized_file: Path
    vocabulary_file: Path
    vocabulary_metadata_file: Path
    encoded_file: Path
    train_dataset_file: Path
    validation_dataset_file: Path
    test_dataset_file: Path
    dataset_metadata_file: Path
    encoding: str

    @property
    def raw_file(self) -> Path:
        """Backward-compatible alias for the configured input file."""

        return self.input_file

    @property
    def processed_file(self) -> Path:
        """Backward-compatible alias for the configured output file."""

        return self.output_file


@dataclass(frozen=True)
class PreprocessingConfig:
    """Text cleaning options used before tokenization."""

    unicode_normalization: str
    lowercase: bool
    normalize_whitespace: bool
    preserve_newlines: bool
    remove_control_characters: bool
    remove_empty_lines: bool
    strip_lines: bool
    min_line_length: int
    max_line_length: int | None


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer settings reserved for future steps."""

    type: str
    vocab_size: int
    min_frequency: int
    lowercase: bool


@dataclass(frozen=True)
class TokenizationConfig:
    """Token string generation options used before vocabulary building."""

    method: str
    preserve_case: bool
    preserve_punctuation: bool
    preserve_newlines: bool
    split_contractions: bool
    normalize_quotes: bool
    normalize_dashes: bool
    add_bos_token: bool
    add_eos_token: bool
    add_newline_token: bool
    bos_token: str
    eos_token: str
    newline_token: str
    unknown_token: str


@dataclass(frozen=True)
class VocabularyConfig:
    """Vocabulary-building options used before dataset preparation."""

    min_frequency: int
    max_size: int | None
    include_special_tokens: bool
    save_frequencies: bool
    strict_special_token_validation: bool
    pad_token: str
    unknown_token: str
    bos_token: str
    eos_token: str
    newline_token: str
    special_token_order: tuple[str, ...]


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset preparation options for next-token prediction."""

    context_length: int
    stride: int
    sequence_mode: str
    short_sequence_policy: str
    add_eos_between_sequences: bool
    split_unit: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    split_seed: int
    shuffle_before_split: bool
    batch_size: int
    shuffle_train: bool
    shuffle_validation: bool
    shuffle_test: bool
    num_workers: int
    pin_memory: bool
    drop_last_train: bool
    drop_last_validation: bool
    drop_last_test: bool
    save_prepared_tensors: bool


@dataclass(frozen=True)
class EmbeddingConfig:
    """Token embedding options used before model architecture steps."""

    embedding_dim: int
    initialization: str
    initialization_std: float
    scale_embeddings: bool
    freeze_embeddings: bool
    zero_padding_embedding: bool


@dataclass(frozen=True)
class PositionalEncodingConfig:
    """Position encoding options used after token embeddings."""

    type: str
    max_sequence_length: int
    dropout: float
    initialization_std: float


@dataclass(frozen=True)
class AttentionConfig:
    """Single-head causal self-attention options."""

    dropout: float
    use_bias: bool
    causal: bool


@dataclass(frozen=True)
class FeedForwardConfig:
    """Position-wise feed-forward network options."""

    hidden_dim: int | None
    hidden_multiplier: int
    activation: str
    dropout: float
    use_bias: bool
    initialization_std: float


@dataclass(frozen=True)
class NormalizationConfig:
    """Layer normalization options."""

    type: str
    epsilon: float
    elementwise_affine: bool


@dataclass(frozen=True)
class ResidualConfig:
    """Residual connection options."""

    dropout: float


@dataclass(frozen=True)
class TransformerBlockConfig:
    """Single transformer block dropout options."""

    attention_dropout: float
    residual_dropout: float
    feed_forward_dropout: float


@dataclass(frozen=True)
class ModelConfig:
    """Small GPT-style model defaults reserved for future steps."""

    context_length: int
    vocab_size: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    use_bias: bool
    tie_embeddings: bool
    initialization_std: float


@dataclass(frozen=True)
class TrainingConfig:
    """Training defaults reserved for future steps."""

    batch_size: int
    learning_rate: float
    epochs: int
    gradient_accumulation_steps: int
    max_grad_norm: float | None
    log_every_steps: int
    validate_every_epochs: int
    seed: int
    device: str


@dataclass(frozen=True)
class LossConfig:
    """Loss function configuration."""

    type: str
    ignore_padding: bool
    label_smoothing: float


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration."""

    type: str
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    separate_weight_decay: bool


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint saving and loading configuration."""

    directory: Path
    save_every_epochs: int
    keep_last: int
    save_best: bool
    monitor: str
    mode: str
    filename_prefix: str


@dataclass(frozen=True)
class GenerationConfig:
    """Text generation defaults reserved for future steps."""

    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    do_sample: bool
    repetition_penalty: float
    stop_on_eos: bool


@dataclass(frozen=True)
class FineTuningConfig:
    """Supervised fine-tuning configuration."""

    dataset_file: Path
    output_directory: Path
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float | None
    gradient_accumulation_steps: int
    freeze_embeddings: bool
    freeze_first_n_layers: int
    train_validation_ratio: float
    seed: int


@dataclass(frozen=True)
class OptimizationConfig:
    """Optional performance optimization settings."""

    mixed_precision: str
    torch_compile: bool
    compile_mode: str
    gradient_checkpointing: bool
    quantization: str
    benchmark_warmup_steps: int
    benchmark_steps: int


@dataclass(frozen=True)
class WebInterfaceConfig:
    """Local Gradio interface configuration."""

    title: str
    description: str
    host: str
    port: int
    share: bool
    default_checkpoint: Path
    default_prompt: str
    max_prompt_characters: int


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str
    log_file: str


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    tokenizer: TokenizerConfig
    tokenization: TokenizationConfig
    vocabulary: VocabularyConfig
    dataset: DatasetConfig
    embeddings: EmbeddingConfig
    positional_encoding: PositionalEncodingConfig
    attention: AttentionConfig
    feed_forward: FeedForwardConfig
    normalization: NormalizationConfig
    residual: ResidualConfig
    transformer_block: TransformerBlockConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    generation: GenerationConfig
    fine_tuning: FineTuningConfig
    optimization: OptimizationConfig
    web_interface: WebInterfaceConfig
    logging: LoggingConfig
    project_root: Path


class ConfigError(ValueError):
    """Raised when the YAML configuration is missing or invalid."""


def get_default_config_path(project_root: Path | None = None) -> Path:
    """Return the default configuration path."""

    root = project_root or get_project_root()
    return root / "configs" / "base.yaml"


def get_project_root() -> Path:
    """Return the repository root from this source file."""

    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load, validate, and convert YAML configuration into dataclasses."""

    project_root = get_project_root()
    path = Path(config_path) if config_path is not None else get_default_config_path(project_root)
    path = path if path.is_absolute() else project_root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ConfigError(f"Configuration must be a YAML mapping: {path}")

    _validate_required_sections(raw_config)

    try:
        config = AppConfig(
            project=ProjectConfig(**raw_config["project"]),
            paths=_build_paths_config(raw_config["paths"], project_root),
            data=_build_data_config(raw_config["data"], project_root),
            preprocessing=PreprocessingConfig(**raw_config["preprocessing"]),
            tokenizer=TokenizerConfig(**raw_config["tokenizer"]),
            tokenization=TokenizationConfig(**_get_tokenization_values(raw_config)),
            vocabulary=_build_vocabulary_config(raw_config),
            dataset=DatasetConfig(**_get_dataset_values(raw_config)),
            model=ModelConfig(**_get_model_values(raw_config)),
            embeddings=EmbeddingConfig(**_get_embedding_values(raw_config)),
            positional_encoding=PositionalEncodingConfig(
                **_get_positional_encoding_values(raw_config)
            ),
            attention=AttentionConfig(**_get_attention_values(raw_config)),
            feed_forward=FeedForwardConfig(**_get_feed_forward_values(raw_config)),
            normalization=NormalizationConfig(**_get_normalization_values(raw_config)),
            residual=ResidualConfig(**_get_residual_values(raw_config)),
            transformer_block=TransformerBlockConfig(**_get_transformer_block_values(raw_config)),
            training=TrainingConfig(**_get_training_values(raw_config)),
            loss=LossConfig(**_get_loss_values(raw_config)),
            optimizer=OptimizerConfig(**_get_optimizer_values(raw_config)),
            checkpoint=CheckpointConfig(**_get_checkpoint_values(raw_config, project_root)),
            generation=GenerationConfig(**_get_generation_values(raw_config)),
            fine_tuning=FineTuningConfig(**_get_fine_tuning_values(raw_config, project_root)),
            optimization=OptimizationConfig(**_get_optimization_values(raw_config)),
            web_interface=WebInterfaceConfig(**_get_web_interface_values(raw_config, project_root)),
            logging=LoggingConfig(**raw_config["logging"]),
            project_root=project_root,
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Invalid configuration values in {path}: {exc}") from exc

    _validate_values(config)
    return config


def _validate_required_sections(raw_config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_SECTIONS - raw_config.keys())
    if missing:
        names = ", ".join(missing)
        raise ConfigError(f"Missing required configuration section(s): {names}")

    for section in REQUIRED_SECTIONS:
        if not isinstance(raw_config[section], dict):
            raise ConfigError(f"Configuration section '{section}' must be a mapping.")


def _build_paths_config(values: dict[str, Any], project_root: Path) -> PathsConfig:
    return PathsConfig(
        data_dir=_resolve_path(values["data_dir"], project_root),
        raw_data_dir=_resolve_path(values["raw_data_dir"], project_root),
        processed_data_dir=_resolve_path(values["processed_data_dir"], project_root),
        tokenized_data_dir=_resolve_path(
            values.get("tokenized_data_dir", "data/tokenized"),
            project_root,
        ),
        checkpoints_dir=_resolve_path(values["checkpoints_dir"], project_root),
        logs_dir=_resolve_path(values["logs_dir"], project_root),
        notebooks_dir=_resolve_path(values["notebooks_dir"], project_root),
    )


def _build_data_config(values: dict[str, Any], project_root: Path) -> DataConfig:
    return DataConfig(
        raw_dir=_resolve_path(values["raw_dir"], project_root),
        processed_dir=_resolve_path(values["processed_dir"], project_root),
        tokenized_dir=_resolve_path(values.get("tokenized_dir", "data/tokenized"), project_root),
        vocabulary_dir=_resolve_path(
            values.get("vocabulary_dir", "data/vocabulary"),
            project_root,
        ),
        dataset_dir=_resolve_path(values.get("dataset_dir", "data/datasets"), project_root),
        input_file=_resolve_path(values["input_file"], project_root),
        output_file=_resolve_path(values["output_file"], project_root),
        tokenized_file=_resolve_path(
            values.get("tokenized_file", "data/tokenized/tokens.jsonl"),
            project_root,
        ),
        vocabulary_file=_resolve_path(
            values.get("vocabulary_file", "data/vocabulary/vocab.json"),
            project_root,
        ),
        vocabulary_metadata_file=_resolve_path(
            values.get("vocabulary_metadata_file", "data/vocabulary/vocab_metadata.json"),
            project_root,
        ),
        encoded_file=_resolve_path(
            values.get("encoded_file", "data/vocabulary/encoded_tokens.jsonl"),
            project_root,
        ),
        train_dataset_file=_resolve_path(
            values.get("train_dataset_file", "data/datasets/train.pt"),
            project_root,
        ),
        validation_dataset_file=_resolve_path(
            values.get("validation_dataset_file", "data/datasets/validation.pt"),
            project_root,
        ),
        test_dataset_file=_resolve_path(
            values.get("test_dataset_file", "data/datasets/test.pt"),
            project_root,
        ),
        dataset_metadata_file=_resolve_path(
            values.get("dataset_metadata_file", "data/datasets/dataset_metadata.json"),
            project_root,
        ),
        encoding=str(values["encoding"]),
    )


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _get_model_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_model_values()
    user_values = raw_config.get("model", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'model' must be a mapping.")
    values.update(user_values)
    return values


def _default_model_values() -> dict[str, Any]:
    return {
        "context_length": 128,
        "vocab_size": 5000,
        "embedding_dim": 128,
        "num_heads": 4,
        "num_layers": 4,
        "dropout": 0.1,
        "use_bias": True,
        "tie_embeddings": True,
        "initialization_std": 0.02,
    }


def _validate_values(config: AppConfig) -> None:
    if not isinstance(config.model.context_length, int) or isinstance(
        config.model.context_length, bool
    ):
        raise ConfigError("model.context_length must be an integer.")
    if config.model.context_length <= 0:
        raise ConfigError("model.context_length must be greater than 0.")
    if not isinstance(config.model.vocab_size, int) or isinstance(config.model.vocab_size, bool):
        raise ConfigError("model.vocab_size must be an integer.")
    if config.model.vocab_size <= 0:
        raise ConfigError("model.vocab_size must be greater than 0.")
    if not isinstance(config.model.embedding_dim, int) or isinstance(
        config.model.embedding_dim, bool
    ):
        raise ConfigError("model.embedding_dim must be an integer.")
    if config.model.embedding_dim <= 0:
        raise ConfigError("model.embedding_dim must be greater than 0.")
    if not isinstance(config.model.num_heads, int) or isinstance(config.model.num_heads, bool):
        raise ConfigError("model.num_heads must be an integer.")
    if config.model.num_heads <= 0:
        raise ConfigError("model.num_heads must be greater than 0.")
    if not isinstance(config.model.num_layers, int) or isinstance(config.model.num_layers, bool):
        raise ConfigError("model.num_layers must be an integer.")
    if config.model.num_layers <= 0:
        raise ConfigError("model.num_layers must be greater than 0.")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ConfigError("model.dropout must be at least 0.0 and less than 1.0.")
    if not isinstance(config.model.use_bias, bool):
        raise ConfigError("model.use_bias must be true or false.")
    if not isinstance(config.model.tie_embeddings, bool):
        raise ConfigError("model.tie_embeddings must be true or false.")
    if (
        not isinstance(config.model.initialization_std, int | float)
        or isinstance(config.model.initialization_std, bool)
        or config.model.initialization_std <= 0
    ):
        raise ConfigError("model.initialization_std must be greater than 0.")
    if config.training.batch_size <= 0:
        raise ConfigError("training.batch_size must be greater than 0.")
    if config.training.learning_rate <= 0:
        raise ConfigError("training.learning_rate must be greater than 0.")
    if config.training.epochs <= 0:
        raise ConfigError("training.epochs must be greater than 0.")
    if (
        not isinstance(config.training.gradient_accumulation_steps, int)
        or isinstance(config.training.gradient_accumulation_steps, bool)
        or config.training.gradient_accumulation_steps <= 0
    ):
        raise ConfigError("training.gradient_accumulation_steps must be greater than 0.")
    if config.training.max_grad_norm is not None and (
        not isinstance(config.training.max_grad_norm, int | float)
        or isinstance(config.training.max_grad_norm, bool)
        or config.training.max_grad_norm <= 0
    ):
        raise ConfigError("training.max_grad_norm must be null or greater than 0.")
    if (
        not isinstance(config.training.log_every_steps, int)
        or isinstance(config.training.log_every_steps, bool)
        or config.training.log_every_steps <= 0
    ):
        raise ConfigError("training.log_every_steps must be greater than 0.")
    if (
        not isinstance(config.training.validate_every_epochs, int)
        or isinstance(config.training.validate_every_epochs, bool)
        or config.training.validate_every_epochs <= 0
    ):
        raise ConfigError("training.validate_every_epochs must be greater than 0.")
    _validate_preprocessing(config.preprocessing)
    _validate_tokenization(config.tokenization)
    _validate_vocabulary(config.vocabulary)
    _validate_dataset(config.dataset)
    _validate_embeddings(config.embeddings, config.model)
    _validate_positional_encoding(config.positional_encoding, config.dataset, config.model)
    _validate_attention(config.attention)
    _validate_feed_forward(config.feed_forward, config.model)
    _validate_normalization(config.normalization, config.model)
    _validate_residual(config.residual)
    _validate_transformer_block(config.transformer_block, config.model, config.positional_encoding)
    _validate_loss(config.loss)
    _validate_optimizer(config.optimizer)
    _validate_checkpoint(config.checkpoint)
    _validate_generation(config.generation)
    _validate_fine_tuning(config.fine_tuning, config.model)
    _validate_optimization(config.optimization)
    _validate_web_interface(config.web_interface)


def _validate_preprocessing(config: PreprocessingConfig) -> None:
    supported_normalization = {"NFC", "NFD", "NFKC", "NFKD", "none"}
    if config.unicode_normalization not in supported_normalization:
        options = ", ".join(sorted(supported_normalization))
        raise ConfigError(f"preprocessing.unicode_normalization must be one of: {options}.")

    bool_fields = {
        "lowercase": config.lowercase,
        "normalize_whitespace": config.normalize_whitespace,
        "preserve_newlines": config.preserve_newlines,
        "remove_control_characters": config.remove_control_characters,
        "remove_empty_lines": config.remove_empty_lines,
        "strip_lines": config.strip_lines,
    }
    for name, value in bool_fields.items():
        if not isinstance(value, bool):
            raise ConfigError(f"preprocessing.{name} must be true or false.")

    if not isinstance(config.min_line_length, int):
        raise ConfigError("preprocessing.min_line_length must be an integer.")
    if config.min_line_length < 0:
        raise ConfigError("preprocessing.min_line_length cannot be negative.")

    if config.max_line_length is not None:
        if not isinstance(config.max_line_length, int):
            raise ConfigError("preprocessing.max_line_length must be an integer or null.")
        if config.max_line_length < 0:
            raise ConfigError("preprocessing.max_line_length cannot be negative.")
        if config.max_line_length < config.min_line_length:
            raise ConfigError(
                "preprocessing.max_line_length must be greater than or equal to "
                "preprocessing.min_line_length."
            )


def _get_tokenization_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_tokenization_values()
    user_values = raw_config.get("tokenization", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'tokenization' must be a mapping.")
    values.update(user_values)
    return values


def _default_tokenization_values() -> dict[str, Any]:
    return {
        "method": "word",
        "preserve_case": True,
        "preserve_punctuation": True,
        "preserve_newlines": True,
        "split_contractions": False,
        "normalize_quotes": True,
        "normalize_dashes": True,
        "add_bos_token": False,
        "add_eos_token": True,
        "add_newline_token": True,
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "newline_token": "<NL>",
        "unknown_token": "<UNK>",
    }


def _validate_tokenization(config: TokenizationConfig) -> None:
    if config.method not in {"word", "character"}:
        raise ConfigError("tokenization.method must be either 'word' or 'character'.")

    bool_fields = {
        "preserve_case": config.preserve_case,
        "preserve_punctuation": config.preserve_punctuation,
        "preserve_newlines": config.preserve_newlines,
        "split_contractions": config.split_contractions,
        "normalize_quotes": config.normalize_quotes,
        "normalize_dashes": config.normalize_dashes,
        "add_bos_token": config.add_bos_token,
        "add_eos_token": config.add_eos_token,
        "add_newline_token": config.add_newline_token,
    }
    for name, value in bool_fields.items():
        if not isinstance(value, bool):
            raise ConfigError(f"tokenization.{name} must be true or false.")

    special_tokens = {
        "bos_token": config.bos_token,
        "eos_token": config.eos_token,
        "newline_token": config.newline_token,
        "unknown_token": config.unknown_token,
    }
    for name, value in special_tokens.items():
        if not isinstance(value, str) or not value:
            raise ConfigError(f"tokenization.{name} must be a non-empty string.")

    if len(set(special_tokens.values())) != len(special_tokens):
        raise ConfigError("tokenization special-token values must be unique.")


def _build_vocabulary_config(raw_config: dict[str, Any]) -> VocabularyConfig:
    values = _default_vocabulary_values()
    user_values = raw_config.get("vocabulary", {})
    if user_values is None:
        return VocabularyConfig(**values)
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'vocabulary' must be a mapping.")
    values.update(user_values)
    if not isinstance(values["special_token_order"], tuple):
        values["special_token_order"] = tuple(values["special_token_order"])
    return VocabularyConfig(**values)


def _default_vocabulary_values() -> dict[str, Any]:
    return {
        "min_frequency": 1,
        "max_size": 5000,
        "include_special_tokens": True,
        "save_frequencies": True,
        "strict_special_token_validation": True,
        "pad_token": "<PAD>",
        "unknown_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "newline_token": "<NL>",
        "special_token_order": ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"),
    }


def _validate_vocabulary(config: VocabularyConfig) -> None:
    if not isinstance(config.min_frequency, int):
        raise ConfigError("vocabulary.min_frequency must be an integer.")
    if config.min_frequency < 1:
        raise ConfigError("vocabulary.min_frequency must be greater than or equal to 1.")

    if config.max_size is not None:
        if not isinstance(config.max_size, int):
            raise ConfigError("vocabulary.max_size must be an integer or null.")
        if config.max_size <= 0:
            raise ConfigError("vocabulary.max_size must be greater than zero when provided.")

    bool_fields = {
        "include_special_tokens": config.include_special_tokens,
        "save_frequencies": config.save_frequencies,
        "strict_special_token_validation": config.strict_special_token_validation,
    }
    for name, value in bool_fields.items():
        if not isinstance(value, bool):
            raise ConfigError(f"vocabulary.{name} must be true or false.")

    special_tokens = {
        "pad_token": config.pad_token,
        "unknown_token": config.unknown_token,
        "bos_token": config.bos_token,
        "eos_token": config.eos_token,
        "newline_token": config.newline_token,
    }
    for name, value in special_tokens.items():
        if not isinstance(value, str) or not value:
            raise ConfigError(f"vocabulary.{name} must be a non-empty string.")

    if len(set(special_tokens.values())) != len(special_tokens):
        raise ConfigError("vocabulary special-token values must be unique.")

    if not isinstance(config.special_token_order, tuple):
        raise ConfigError("vocabulary.special_token_order must be a list of strings.")
    if any(not isinstance(token, str) or not token for token in config.special_token_order):
        raise ConfigError("vocabulary.special_token_order must contain non-empty strings.")
    if len(set(config.special_token_order)) != len(config.special_token_order):
        raise ConfigError("vocabulary.special_token_order must not contain duplicates.")

    expected_order = set(special_tokens.values())
    configured_order = set(config.special_token_order)
    missing = expected_order - configured_order
    unknown = configured_order - expected_order
    if missing:
        raise ConfigError(
            "vocabulary.special_token_order is missing required token(s): "
            f"{', '.join(sorted(missing))}."
        )
    if unknown:
        raise ConfigError(
            "vocabulary.special_token_order contains unknown token(s): "
            f"{', '.join(sorted(unknown))}."
        )

    if config.max_size is not None and config.max_size < len(config.special_token_order):
        raise ConfigError("vocabulary.max_size must be large enough to include all special tokens.")


def _get_dataset_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_dataset_values()
    user_values = raw_config.get("dataset", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'dataset' must be a mapping.")
    values.update(user_values)
    return values


def _default_dataset_values() -> dict[str, Any]:
    return {
        "context_length": 128,
        "stride": 128,
        "sequence_mode": "continuous",
        "short_sequence_policy": "pad",
        "add_eos_between_sequences": False,
        "split_unit": "sequence",
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "split_seed": 42,
        "shuffle_before_split": True,
        "batch_size": 16,
        "shuffle_train": True,
        "shuffle_validation": False,
        "shuffle_test": False,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last_train": False,
        "drop_last_validation": False,
        "drop_last_test": False,
        "save_prepared_tensors": True,
    }


def _validate_dataset(config: DatasetConfig) -> None:
    if not isinstance(config.context_length, int) or config.context_length <= 0:
        raise ConfigError("dataset.context_length must be an integer greater than zero.")
    if not isinstance(config.stride, int) or config.stride <= 0:
        raise ConfigError("dataset.stride must be an integer greater than zero.")
    if config.sequence_mode not in {"continuous", "per_sequence"}:
        raise ConfigError("dataset.sequence_mode must be 'continuous' or 'per_sequence'.")
    if config.short_sequence_policy not in {"skip", "pad"}:
        raise ConfigError("dataset.short_sequence_policy must be 'skip' or 'pad'.")
    if config.split_unit not in {"sequence", "sample"}:
        raise ConfigError("dataset.split_unit must be 'sequence' or 'sample'.")

    ratios = {
        "train_ratio": config.train_ratio,
        "validation_ratio": config.validation_ratio,
        "test_ratio": config.test_ratio,
    }
    for name, value in ratios.items():
        if not isinstance(value, int | float):
            raise ConfigError(f"dataset.{name} must be numeric.")
        if value < 0 or value > 1:
            raise ConfigError(f"dataset.{name} must be between 0 and 1.")
    ratio_sum = config.train_ratio + config.validation_ratio + config.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ConfigError("dataset split ratios must sum to 1.0.")
    if ratio_sum <= 0:
        raise ConfigError("At least one dataset split ratio must be greater than zero.")

    if not isinstance(config.split_seed, int):
        raise ConfigError("dataset.split_seed must be an integer.")
    if not isinstance(config.batch_size, int) or config.batch_size <= 0:
        raise ConfigError("dataset.batch_size must be an integer greater than zero.")
    if not isinstance(config.num_workers, int) or config.num_workers < 0:
        raise ConfigError("dataset.num_workers must be an integer greater than or equal to zero.")

    bool_fields = {
        "add_eos_between_sequences": config.add_eos_between_sequences,
        "shuffle_before_split": config.shuffle_before_split,
        "shuffle_train": config.shuffle_train,
        "shuffle_validation": config.shuffle_validation,
        "shuffle_test": config.shuffle_test,
        "pin_memory": config.pin_memory,
        "drop_last_train": config.drop_last_train,
        "drop_last_validation": config.drop_last_validation,
        "drop_last_test": config.drop_last_test,
        "save_prepared_tensors": config.save_prepared_tensors,
    }
    for name, value in bool_fields.items():
        if not isinstance(value, bool):
            raise ConfigError(f"dataset.{name} must be true or false.")


def _get_embedding_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_embedding_values()
    model_values = raw_config.get("model", {})
    if isinstance(model_values, dict):
        values["embedding_dim"] = model_values.get("embedding_dim", values["embedding_dim"])

    user_values = raw_config.get("embeddings", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'embeddings' must be a mapping.")
    values.update(user_values)
    return values


def _default_embedding_values() -> dict[str, Any]:
    return {
        "embedding_dim": 128,
        "initialization": "normal",
        "initialization_std": 0.02,
        "scale_embeddings": False,
        "freeze_embeddings": False,
        "zero_padding_embedding": True,
    }


def _validate_embeddings(config: EmbeddingConfig, model: ModelConfig) -> None:
    if (
        not isinstance(config.embedding_dim, int)
        or isinstance(config.embedding_dim, bool)
        or config.embedding_dim <= 0
    ):
        raise ConfigError("embeddings.embedding_dim must be an integer greater than zero.")
    if config.embedding_dim != model.embedding_dim:
        raise ConfigError("embeddings.embedding_dim must match model.embedding_dim.")

    if config.initialization not in {"normal", "uniform", "xavier_uniform"}:
        raise ConfigError(
            "embeddings.initialization must be one of: normal, uniform, xavier_uniform."
        )
    if (
        not isinstance(config.initialization_std, int | float)
        or isinstance(config.initialization_std, bool)
        or config.initialization_std <= 0
    ):
        raise ConfigError("embeddings.initialization_std must be greater than zero.")

    bool_fields = {
        "scale_embeddings": config.scale_embeddings,
        "freeze_embeddings": config.freeze_embeddings,
        "zero_padding_embedding": config.zero_padding_embedding,
    }
    for name, value in bool_fields.items():
        if not isinstance(value, bool):
            raise ConfigError(f"embeddings.{name} must be true or false.")

    if model.num_heads > 0 and config.embedding_dim % model.num_heads != 0:
        raise ConfigError(
            "embedding_dim must be divisible by num_heads. "
            f"Received embedding_dim={config.embedding_dim} and num_heads={model.num_heads}."
        )


def _get_positional_encoding_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_positional_encoding_values()
    dataset_values = raw_config.get("dataset", {})
    model_values = raw_config.get("model", {})
    context_lengths = [
        values["max_sequence_length"],
        dataset_values.get("context_length", 0) if isinstance(dataset_values, dict) else 0,
        model_values.get("context_length", 0) if isinstance(model_values, dict) else 0,
    ]
    values["max_sequence_length"] = max(
        value for value in context_lengths if isinstance(value, int) and not isinstance(value, bool)
    )

    user_values = raw_config.get("positional_encoding", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'positional_encoding' must be a mapping.")
    values.update(user_values)
    return values


def _default_positional_encoding_values() -> dict[str, Any]:
    return {
        "type": "learned",
        "max_sequence_length": 128,
        "dropout": 0.0,
        "initialization_std": 0.02,
    }


def _validate_positional_encoding(
    config: PositionalEncodingConfig,
    dataset: DatasetConfig,
    model: ModelConfig,
) -> None:
    if config.type not in {"learned", "sinusoidal"}:
        raise ConfigError("positional_encoding.type must be either 'learned' or 'sinusoidal'.")
    if (
        not isinstance(config.max_sequence_length, int)
        or isinstance(config.max_sequence_length, bool)
        or config.max_sequence_length <= 0
    ):
        raise ConfigError(
            "positional_encoding.max_sequence_length must be an integer greater than zero."
        )
    if (
        not isinstance(config.dropout, int | float)
        or isinstance(config.dropout, bool)
        or not 0.0 <= config.dropout < 1.0
    ):
        raise ConfigError("positional_encoding.dropout must be at least 0.0 and less than 1.0.")
    if (
        not isinstance(config.initialization_std, int | float)
        or isinstance(config.initialization_std, bool)
        or config.initialization_std <= 0
    ):
        raise ConfigError("positional_encoding.initialization_std must be greater than zero.")

    required_length = max(dataset.context_length, model.context_length)
    if config.max_sequence_length < required_length:
        raise ConfigError(
            "positional_encoding.max_sequence_length must be at least the configured "
            f"context length. Received max_sequence_length={config.max_sequence_length} "
            f"and required_length={required_length}."
        )


def _get_attention_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_attention_values()
    user_values = raw_config.get("attention", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'attention' must be a mapping.")
    values.update(user_values)
    return values


def _default_attention_values() -> dict[str, Any]:
    return {
        "dropout": 0.1,
        "use_bias": True,
        "causal": True,
    }


def _validate_attention(config: AttentionConfig) -> None:
    if (
        not isinstance(config.dropout, int | float)
        or isinstance(config.dropout, bool)
        or not 0.0 <= config.dropout < 1.0
    ):
        raise ConfigError("attention.dropout must be at least 0.0 and less than 1.0.")
    for name, value in {"use_bias": config.use_bias, "causal": config.causal}.items():
        if not isinstance(value, bool):
            raise ConfigError(f"attention.{name} must be true or false.")
    if not config.causal:
        raise ConfigError("GPT causal self-attention requires attention.causal to be true.")


def _get_feed_forward_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_feed_forward_values()
    user_values = raw_config.get("feed_forward", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'feed_forward' must be a mapping.")
    values.update(user_values)
    return values


def _default_feed_forward_values() -> dict[str, Any]:
    return {
        "hidden_dim": None,
        "hidden_multiplier": 4,
        "activation": "gelu",
        "dropout": 0.1,
        "use_bias": True,
        "initialization_std": 0.02,
    }


def _validate_feed_forward(config: FeedForwardConfig, model: ModelConfig) -> None:
    if config.hidden_dim is not None and (
        not isinstance(config.hidden_dim, int)
        or isinstance(config.hidden_dim, bool)
        or config.hidden_dim <= 0
    ):
        raise ConfigError("feed_forward.hidden_dim must be an integer greater than zero or null.")
    if (
        not isinstance(config.hidden_multiplier, int)
        or isinstance(config.hidden_multiplier, bool)
        or config.hidden_multiplier <= 0
    ):
        raise ConfigError("feed_forward.hidden_multiplier must be an integer greater than zero.")
    if config.activation not in {"gelu", "relu", "silu"}:
        raise ConfigError("feed_forward.activation must be one of: gelu, relu, silu.")
    if (
        not isinstance(config.dropout, int | float)
        or isinstance(config.dropout, bool)
        or not 0.0 <= config.dropout < 1.0
    ):
        raise ConfigError("feed_forward.dropout must be at least 0.0 and less than 1.0.")
    if not isinstance(config.use_bias, bool):
        raise ConfigError("feed_forward.use_bias must be true or false.")
    if (
        not isinstance(config.initialization_std, int | float)
        or isinstance(config.initialization_std, bool)
        or config.initialization_std <= 0
    ):
        raise ConfigError("feed_forward.initialization_std must be greater than zero.")
    if model.embedding_dim <= 0:
        raise ConfigError("model.embedding_dim must be greater than 0.")


def _get_normalization_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_normalization_values()
    user_values = raw_config.get("normalization", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'normalization' must be a mapping.")
    values.update(user_values)
    return values


def _default_normalization_values() -> dict[str, Any]:
    return {
        "type": "layer_norm",
        "epsilon": 1e-5,
        "elementwise_affine": True,
    }


def _validate_normalization(config: NormalizationConfig, model: ModelConfig) -> None:
    if config.type != "layer_norm":
        raise ConfigError("normalization.type must be 'layer_norm'.")
    if (
        not isinstance(config.epsilon, int | float)
        or isinstance(config.epsilon, bool)
        or config.epsilon <= 0
    ):
        raise ConfigError("normalization.epsilon must be greater than zero.")
    if not isinstance(config.elementwise_affine, bool):
        raise ConfigError("normalization.elementwise_affine must be true or false.")
    if model.embedding_dim <= 0:
        raise ConfigError("model.embedding_dim must be greater than 0.")


def _get_residual_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_residual_values()
    user_values = raw_config.get("residual", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'residual' must be a mapping.")
    values.update(user_values)
    return values


def _default_residual_values() -> dict[str, Any]:
    return {
        "dropout": 0.1,
    }


def _validate_residual(config: ResidualConfig) -> None:
    if (
        not isinstance(config.dropout, int | float)
        or isinstance(config.dropout, bool)
        or not 0.0 <= config.dropout < 1.0
    ):
        raise ConfigError("residual.dropout must be at least 0.0 and less than 1.0.")


def _get_transformer_block_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_transformer_block_values()
    user_values = raw_config.get("transformer_block", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'transformer_block' must be a mapping.")
    values.update(user_values)
    return values


def _default_transformer_block_values() -> dict[str, Any]:
    return {
        "attention_dropout": 0.1,
        "residual_dropout": 0.1,
        "feed_forward_dropout": 0.1,
    }


def _validate_transformer_block(
    config: TransformerBlockConfig,
    model: ModelConfig,
    positional_encoding: PositionalEncodingConfig,
) -> None:
    for name, value in {
        "attention_dropout": config.attention_dropout,
        "residual_dropout": config.residual_dropout,
        "feed_forward_dropout": config.feed_forward_dropout,
    }.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or not 0.0 <= value < 1.0:
            raise ConfigError(f"transformer_block.{name} must be at least 0.0 and less than 1.0.")
    if model.embedding_dim % model.num_heads != 0:
        raise ConfigError(
            "model.embedding_dim must be divisible by model.num_heads for transformer blocks."
        )
    if positional_encoding.max_sequence_length < model.context_length:
        raise ConfigError(
            "positional_encoding.max_sequence_length must be at least model.context_length."
        )


def _get_training_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_training_values()
    user_values = raw_config.get("training", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'training' must be a mapping.")
    values.update(user_values)
    return values


def _default_training_values() -> dict[str, Any]:
    return {
        "batch_size": 16,
        "learning_rate": 0.0003,
        "epochs": 10,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "log_every_steps": 10,
        "validate_every_epochs": 1,
        "seed": 42,
        "device": "auto",
    }


def _get_loss_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_loss_values()
    user_values = raw_config.get("loss", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'loss' must be a mapping.")
    values.update(user_values)
    return values


def _default_loss_values() -> dict[str, Any]:
    return {
        "type": "cross_entropy",
        "ignore_padding": True,
        "label_smoothing": 0.0,
    }


def _validate_loss(config: LossConfig) -> None:
    if config.type != "cross_entropy":
        raise ConfigError("loss.type must be 'cross_entropy'.")
    if not isinstance(config.ignore_padding, bool):
        raise ConfigError("loss.ignore_padding must be true or false.")
    if (
        not isinstance(config.label_smoothing, int | float)
        or isinstance(config.label_smoothing, bool)
        or not 0.0 <= config.label_smoothing < 1.0
    ):
        raise ConfigError("loss.label_smoothing must be at least 0.0 and less than 1.0.")


def _get_optimizer_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_optimizer_values()
    user_values = raw_config.get("optimizer", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'optimizer' must be a mapping.")
    values.update(user_values)
    return values


def _default_optimizer_values() -> dict[str, Any]:
    return {
        "type": "adamw",
        "learning_rate": 0.0003,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "separate_weight_decay": True,
    }


def _validate_optimizer(config: OptimizerConfig) -> None:
    if config.type != "adamw":
        raise ConfigError("optimizer.type must be 'adamw'.")
    if (
        not isinstance(config.learning_rate, int | float)
        or isinstance(config.learning_rate, bool)
        or config.learning_rate <= 0
    ):
        raise ConfigError("optimizer.learning_rate must be greater than 0.")
    if (
        not isinstance(config.weight_decay, int | float)
        or isinstance(config.weight_decay, bool)
        or config.weight_decay < 0
    ):
        raise ConfigError("optimizer.weight_decay must be greater than or equal to 0.")
    if (
        not isinstance(config.beta1, int | float)
        or isinstance(config.beta1, bool)
        or not 0 <= config.beta1 < 1
    ):
        raise ConfigError("optimizer.beta1 must be at least 0.0 and less than 1.0.")
    if (
        not isinstance(config.beta2, int | float)
        or isinstance(config.beta2, bool)
        or not 0 <= config.beta2 < 1
    ):
        raise ConfigError("optimizer.beta2 must be at least 0.0 and less than 1.0.")
    if (
        not isinstance(config.epsilon, int | float)
        or isinstance(config.epsilon, bool)
        or config.epsilon <= 0
    ):
        raise ConfigError("optimizer.epsilon must be greater than 0.")
    if not isinstance(config.separate_weight_decay, bool):
        raise ConfigError("optimizer.separate_weight_decay must be true or false.")


def _get_checkpoint_values(raw_config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    values = _default_checkpoint_values()
    user_values = raw_config.get("checkpoint", {})
    if user_values is None:
        user_values = {}
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'checkpoint' must be a mapping.")
    values.update(user_values)
    values["directory"] = _resolve_path(values["directory"], project_root)
    return values


def _default_checkpoint_values() -> dict[str, Any]:
    return {
        "directory": "checkpoints",
        "save_every_epochs": 1,
        "keep_last": 3,
        "save_best": True,
        "monitor": "validation_loss",
        "mode": "min",
        "filename_prefix": "genpy",
    }


def _validate_checkpoint(config: CheckpointConfig) -> None:
    if not isinstance(config.directory, Path) or str(config.directory).strip() == "":
        raise ConfigError("checkpoint.directory must be a non-empty path.")
    if config.directory.exists() and not config.directory.is_dir():
        raise ConfigError("checkpoint.directory must be a directory path.")
    if (
        not isinstance(config.save_every_epochs, int)
        or isinstance(config.save_every_epochs, bool)
        or config.save_every_epochs <= 0
    ):
        raise ConfigError("checkpoint.save_every_epochs must be greater than 0.")
    if (
        not isinstance(config.keep_last, int)
        or isinstance(config.keep_last, bool)
        or config.keep_last <= 0
    ):
        raise ConfigError("checkpoint.keep_last must be greater than 0.")
    if not isinstance(config.save_best, bool):
        raise ConfigError("checkpoint.save_best must be true or false.")
    if config.monitor not in {"training_loss", "validation_loss"}:
        raise ConfigError("checkpoint.monitor must be 'training_loss' or 'validation_loss'.")
    if config.mode not in {"min", "max"}:
        raise ConfigError("checkpoint.mode must be 'min' or 'max'.")
    if not isinstance(config.filename_prefix, str) or config.filename_prefix.strip() == "":
        raise ConfigError("checkpoint.filename_prefix must be a non-empty string.")


def _get_generation_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_generation_values()
    user_values = raw_config.get("generation", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'generation' must be a mapping.")
    values.update(user_values)
    return values


def _default_generation_values() -> dict[str, Any]:
    return {
        "max_new_tokens": 50,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "stop_on_eos": True,
    }


def _validate_generation(config: GenerationConfig) -> None:
    if (
        not isinstance(config.max_new_tokens, int)
        or isinstance(config.max_new_tokens, bool)
        or config.max_new_tokens <= 0
    ):
        raise ConfigError("generation.max_new_tokens must be greater than 0.")
    if (
        not isinstance(config.temperature, int | float)
        or isinstance(config.temperature, bool)
        or config.temperature <= 0
    ):
        raise ConfigError("generation.temperature must be greater than 0.")
    if config.top_k is not None and (
        not isinstance(config.top_k, int) or isinstance(config.top_k, bool) or config.top_k <= 0
    ):
        raise ConfigError("generation.top_k must be null or greater than 0.")
    if config.top_p is not None and (
        not isinstance(config.top_p, int | float)
        or isinstance(config.top_p, bool)
        or not 0 < config.top_p <= 1
    ):
        raise ConfigError("generation.top_p must be null or greater than 0 and at most 1.")
    if not isinstance(config.do_sample, bool):
        raise ConfigError("generation.do_sample must be true or false.")
    if (
        not isinstance(config.repetition_penalty, int | float)
        or isinstance(config.repetition_penalty, bool)
        or config.repetition_penalty <= 0
    ):
        raise ConfigError("generation.repetition_penalty must be greater than 0.")
    if not isinstance(config.stop_on_eos, bool):
        raise ConfigError("generation.stop_on_eos must be true or false.")


def _get_fine_tuning_values(raw_config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    values = _default_fine_tuning_values()
    user_values = raw_config.get("fine_tuning", {})
    if user_values is None:
        user_values = {}
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'fine_tuning' must be a mapping.")
    values.update(user_values)
    values["dataset_file"] = _resolve_path(values["dataset_file"], project_root)
    values["output_directory"] = _resolve_path(values["output_directory"], project_root)
    return values


def _default_fine_tuning_values() -> dict[str, Any]:
    return {
        "dataset_file": "data/fine_tuning/train.jsonl",
        "output_directory": "checkpoints/fine_tuned",
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "freeze_embeddings": False,
        "freeze_first_n_layers": 0,
        "train_validation_ratio": 0.9,
        "seed": 42,
    }


def _validate_fine_tuning(config: FineTuningConfig, model: ModelConfig) -> None:
    if not isinstance(config.dataset_file, Path) or str(config.dataset_file).strip() == "":
        raise ConfigError("fine_tuning.dataset_file must be a non-empty path.")
    if config.dataset_file.exists() and not config.dataset_file.is_file():
        raise ConfigError("fine_tuning.dataset_file must be a file path.")
    if not isinstance(config.output_directory, Path) or str(config.output_directory).strip() == "":
        raise ConfigError("fine_tuning.output_directory must be a non-empty path.")
    if config.output_directory.exists() and not config.output_directory.is_dir():
        raise ConfigError("fine_tuning.output_directory must be a directory path.")
    for name, value in {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"fine_tuning.{name} must be greater than 0.")
    for name, value in {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
    }.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"fine_tuning.{name} must be greater than or equal to 0.")
    if config.learning_rate <= 0:
        raise ConfigError("fine_tuning.learning_rate must be greater than 0.")
    if config.max_grad_norm is not None and (
        not isinstance(config.max_grad_norm, int | float)
        or isinstance(config.max_grad_norm, bool)
        or config.max_grad_norm <= 0
    ):
        raise ConfigError("fine_tuning.max_grad_norm must be null or greater than 0.")
    if not isinstance(config.freeze_embeddings, bool):
        raise ConfigError("fine_tuning.freeze_embeddings must be true or false.")
    if (
        not isinstance(config.freeze_first_n_layers, int)
        or isinstance(config.freeze_first_n_layers, bool)
        or config.freeze_first_n_layers < 0
        or config.freeze_first_n_layers > model.num_layers
    ):
        raise ConfigError(
            "fine_tuning.freeze_first_n_layers must be between 0 and model.num_layers."
        )
    if (
        not isinstance(config.train_validation_ratio, int | float)
        or isinstance(config.train_validation_ratio, bool)
        or not 0 < config.train_validation_ratio <= 1
    ):
        raise ConfigError(
            "fine_tuning.train_validation_ratio must be greater than 0 and at most 1."
        )
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise ConfigError("fine_tuning.seed must be an integer.")


def _get_optimization_values(raw_config: dict[str, Any]) -> dict[str, Any]:
    values = _default_optimization_values()
    user_values = raw_config.get("optimization", {})
    if user_values is None:
        return values
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'optimization' must be a mapping.")
    values.update(user_values)
    return values


def _default_optimization_values() -> dict[str, Any]:
    return {
        "mixed_precision": "none",
        "torch_compile": False,
        "compile_mode": "default",
        "gradient_checkpointing": False,
        "quantization": "none",
        "benchmark_warmup_steps": 2,
        "benchmark_steps": 10,
    }


def _validate_optimization(config: OptimizationConfig) -> None:
    if config.mixed_precision not in {"none", "fp16", "bf16"}:
        raise ConfigError("optimization.mixed_precision must be 'none', 'fp16', or 'bf16'.")
    if not isinstance(config.torch_compile, bool):
        raise ConfigError("optimization.torch_compile must be true or false.")
    if config.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ConfigError(
            "optimization.compile_mode must be 'default', 'reduce-overhead', or 'max-autotune'."
        )
    if not isinstance(config.gradient_checkpointing, bool):
        raise ConfigError("optimization.gradient_checkpointing must be true or false.")
    if config.quantization not in {"none", "dynamic_int8"}:
        raise ConfigError("optimization.quantization must be 'none' or 'dynamic_int8'.")
    for name, value in {
        "benchmark_warmup_steps": config.benchmark_warmup_steps,
        "benchmark_steps": config.benchmark_steps,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"optimization.{name} must be greater than 0.")


def _get_web_interface_values(raw_config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    values = _default_web_interface_values()
    user_values = raw_config.get("web_interface", {})
    if user_values is None:
        user_values = {}
    if not isinstance(user_values, dict):
        raise ConfigError("Configuration section 'web_interface' must be a mapping.")
    values.update(user_values)
    values["default_checkpoint"] = _resolve_path(values["default_checkpoint"], project_root)
    return values


def _default_web_interface_values() -> dict[str, Any]:
    return {
        "title": "GenPy LLM",
        "description": "Generate text using the trained GenPy model.",
        "host": "127.0.0.1",
        "port": 7860,
        "share": False,
        "default_checkpoint": "checkpoints/genpy_best.pt",
        "default_prompt": "Hello",
        "max_prompt_characters": 2000,
    }


def _validate_web_interface(config: WebInterfaceConfig) -> None:
    for name, value in {
        "title": config.title,
        "description": config.description,
        "host": config.host,
        "default_prompt": config.default_prompt,
    }.items():
        if not isinstance(value, str) or value.strip() == "":
            raise ConfigError(f"web_interface.{name} must be a non-empty string.")
    if (
        not isinstance(config.port, int)
        or isinstance(config.port, bool)
        or not 1 <= config.port <= 65535
    ):
        raise ConfigError("web_interface.port must be between 1 and 65535.")
    if not isinstance(config.share, bool):
        raise ConfigError("web_interface.share must be true or false.")
    if (
        not isinstance(config.default_checkpoint, Path)
        or str(config.default_checkpoint).strip() == ""
    ):
        raise ConfigError("web_interface.default_checkpoint must be a non-empty path.")
    if config.default_checkpoint.exists() and not config.default_checkpoint.is_file():
        raise ConfigError("web_interface.default_checkpoint must be a file path.")
    if (
        not isinstance(config.max_prompt_characters, int)
        or isinstance(config.max_prompt_characters, bool)
        or config.max_prompt_characters <= 0
    ):
        raise ConfigError("web_interface.max_prompt_characters must be greater than 0.")
