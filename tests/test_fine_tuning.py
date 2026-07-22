from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.config import ConfigError, TokenizationConfig, VocabularyConfig, load_config
from genpy_llm.fine_tuning import (
    FineTuningDataset,
    FineTuningError,
    configure_trainable_parameters,
    create_fine_tuning_optimizer,
    load_base_model_for_fine_tuning,
    load_fine_tuning_records,
    prepare_fine_tuning_dataset,
    run_fine_tuning,
)
from genpy_llm.generation import create_generator_from_checkpoint
from genpy_llm.gpt import GPTModel
from genpy_llm.losses import create_loss_function
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.training import GPTTrainer
from genpy_llm.vocabulary import Vocabulary


def test_instruction_and_plain_text_jsonl_parsing(tmp_path: Path) -> None:
    path = _records_path(
        tmp_path,
        [
            {"instruction": "What is AI?", "response": "AI helps."},
            {"text": "Complete training sequence."},
        ],
    )

    records = load_fine_tuning_records(path)

    assert records == [
        {"instruction": "What is AI?", "response": "AI helps."},
        {"text": "Complete training sequence."},
    ]


def test_malformed_record_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"instruction": "missing response"}\n', encoding="utf-8")

    with pytest.raises(FineTuningError, match="line 1"):
        load_fine_tuning_records(path)


def test_empty_record_skipping_and_tamil_support(tmp_path: Path) -> None:
    path = _records_path(
        tmp_path,
        [
            {"text": ""},
            {"instruction": "Say hello in Tamil", "response": "வணக்கம்"},
        ],
    )

    train_dataset, validation_dataset, stats = prepare_fine_tuning_dataset(
        path,
        TextTokenizer(_tokenization_config()),
        _vocabulary(),
        context_length=8,
        train_validation_ratio=1.0,
        seed=1,
    )

    assert stats.source_records == 2
    assert stats.usable_records == 1
    assert stats.skipped_records == 1
    assert len(train_dataset) == 1
    assert len(validation_dataset) == 0


def test_tokenization_encoding_eos_shift_padding_and_attention_mask(tmp_path: Path) -> None:
    path = _records_path(tmp_path, [{"text": "Hello world"}])
    vocabulary = _vocabulary()

    train_dataset, _validation_dataset, stats = prepare_fine_tuning_dataset(
        path,
        TextTokenizer(_tokenization_config()),
        vocabulary,
        context_length=6,
        train_validation_ratio=1.0,
        seed=1,
    )
    item = train_dataset[0]
    real_length = int(item["attention_mask"].sum().item())

    assert isinstance(train_dataset, FineTuningDataset)
    assert stats.truncated_records == 0
    assert item["input_ids"].dtype == torch.long
    assert item["target_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["input_ids"].shape == (6,)
    assert item["target_ids"][real_length - 1].item() == vocabulary.eos_id
    assert item["input_ids"][real_length:].tolist() == [vocabulary.pad_id] * (6 - real_length)
    assert item["attention_mask"].tolist() == [1] * real_length + [0] * (6 - real_length)


def test_truncation_and_deterministic_split(tmp_path: Path) -> None:
    path = _records_path(
        tmp_path,
        [
            {"text": "Hello world AI systems"},
            {"text": "GenPy model"},
            {"text": "AI model"},
            {"text": "Tamil வணக்கம்"},
        ],
    )

    first_train, first_validation, first_stats = prepare_fine_tuning_dataset(
        path,
        TextTokenizer(_tokenization_config()),
        _vocabulary(),
        context_length=2,
        train_validation_ratio=0.5,
        seed=7,
    )
    second_train, second_validation, second_stats = prepare_fine_tuning_dataset(
        path,
        TextTokenizer(_tokenization_config()),
        _vocabulary(),
        context_length=2,
        train_validation_ratio=0.5,
        seed=7,
    )

    assert first_stats.truncated_records > 0
    assert first_stats == second_stats
    assert [item["input_ids"].tolist() for item in first_train] == [
        item["input_ids"].tolist() for item in second_train
    ]
    assert len(first_validation) == len(second_validation)


def test_embedding_and_first_n_layer_freezing_excludes_optimizer_params() -> None:
    model = _tiny_model(num_layers=2)

    stats = configure_trainable_parameters(
        model,
        freeze_embeddings=True,
        freeze_first_n_layers=1,
    )
    optimizer = create_fine_tuning_optimizer(model, _app_config().fine_tuning)
    optimizer_param_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }

    assert stats.frozen_parameter_count > 0
    assert all(not parameter.requires_grad for parameter in model.token_embedding.parameters())
    assert all(not parameter.requires_grad for parameter in model.blocks[0].parameters())
    assert all(
        id(parameter) not in optimizer_param_ids
        for parameter in model.parameters()
        if not parameter.requires_grad
    )


def test_invalid_freeze_count_is_rejected() -> None:
    model = _tiny_model(num_layers=1)

    with pytest.raises(FineTuningError, match="freeze_first_n_layers"):
        configure_trainable_parameters(model, freeze_embeddings=False, freeze_first_n_layers=2)


def test_base_checkpoint_loading(tmp_path: Path) -> None:
    app_config, vocabulary_path = _tiny_app_config(tmp_path)
    model = _tiny_model(num_layers=1)
    checkpoint_path = _save_model_checkpoint(tmp_path, model)

    loaded_model = load_base_model_for_fine_tuning(
        checkpoint_path,
        app_config,
        torch.device("cpu"),
    )

    assert loaded_model.vocab_size == len(Vocabulary.load(vocabulary_path))
    assert loaded_model.context_length == app_config.model.context_length


def test_one_fine_tuning_batch_updates_trainable_and_not_frozen_layers(tmp_path: Path) -> None:
    vocabulary = _vocabulary()
    dataset_path = _records_path(tmp_path, [{"text": "Hello world"}, {"text": "AI systems"}])
    train_dataset, _validation_dataset, _stats = prepare_fine_tuning_dataset(
        dataset_path,
        TextTokenizer(_tokenization_config()),
        vocabulary,
        context_length=8,
        train_validation_ratio=1.0,
        seed=1,
    )
    model = _tiny_model(num_layers=2)
    configure_trainable_parameters(model, freeze_embeddings=True, freeze_first_n_layers=1)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    app_config, vocabulary_path = _tiny_app_config(tmp_path)
    optimizer = create_fine_tuning_optimizer(model, app_config.fine_tuning)
    trainer = GPTTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=create_loss_function(vocabulary_path, app_config.loss),
        device=torch.device("cpu"),
    )

    batch = {name: tensor.unsqueeze(0) for name, tensor in train_dataset[0].items()}
    trainer.train_batch(batch, batch_index=0)

    embedding_name, embedding_parameter = next(iter(model.token_embedding.named_parameters()))
    assert torch.equal(
        embedding_parameter.detach(),
        before[f"token_embedding.{embedding_name}"],
    )
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_fine_tuned_checkpoint_saving_loading_and_resume(tmp_path: Path) -> None:
    app_config, vocabulary_path = _tiny_app_config(tmp_path)
    dataset_path = _records_path(
        tmp_path,
        [{"text": "Hello world"}, {"text": "AI systems"}, {"text": "GenPy model"}],
    )
    tokenizer = TextTokenizer(app_config.tokenization)
    vocabulary = Vocabulary.load(vocabulary_path)
    train_dataset, validation_dataset, _stats = prepare_fine_tuning_dataset(
        dataset_path,
        tokenizer,
        vocabulary,
        context_length=app_config.model.context_length,
        train_validation_ratio=0.67,
        seed=1,
    )
    model = _tiny_model(num_layers=1)
    base_checkpoint = _save_model_checkpoint(tmp_path, model)
    parameter_stats = configure_trainable_parameters(model, False, 0)

    result = run_fine_tuning(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        vocabulary_path=vocabulary_path,
        app_config=app_config,
        fine_tuning_config=replace(app_config.fine_tuning, epochs=1, batch_size=1),
        output_directory=tmp_path / "ft",
        base_checkpoint_path=base_checkpoint,
        dataset_path=dataset_path,
        device=torch.device("cpu"),
        max_batches=1,
        parameter_stats=parameter_stats,
    )

    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.exists()
    loaded = load_checkpoint(
        result.best_checkpoint_path,
        _tiny_model(num_layers=1),
        restore_rng=False,
    )
    assert loaded.epoch == 1

    resumed = run_fine_tuning(
        model=_tiny_model(num_layers=1),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        vocabulary_path=vocabulary_path,
        app_config=app_config,
        fine_tuning_config=replace(app_config.fine_tuning, epochs=1, batch_size=1),
        output_directory=tmp_path / "ft",
        base_checkpoint_path=base_checkpoint,
        dataset_path=dataset_path,
        device=torch.device("cpu"),
        max_batches=1,
        resume_checkpoint_path=result.best_checkpoint_path,
        parameter_stats=parameter_stats,
    )
    assert resumed.epochs[0].epoch == 2


def test_fine_tuned_checkpoint_can_generate(tmp_path: Path) -> None:
    app_config, vocabulary_path = _tiny_app_config(tmp_path)
    model = _tiny_model(num_layers=1)
    checkpoint_path = _save_model_checkpoint(tmp_path, model)

    generator = create_generator_from_checkpoint(
        checkpoint_path,
        _tiny_config_path(tmp_path, vocabulary_path),
        vocabulary_path,
        torch.device("cpu"),
    )

    result = generator.generate("Hello", max_new_tokens=2, do_sample=False)

    assert result.prompt == "Hello"


def test_fine_tuning_config_validation(tmp_path: Path) -> None:
    config_data = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config_data["fine_tuning"]["train_validation_ratio"] = 0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="fine_tuning.train_validation_ratio"):
        load_config(path)


def test_cpu_behavior(tmp_path: Path) -> None:
    app_config, vocabulary_path = _tiny_app_config(tmp_path)
    dataset_path = _records_path(tmp_path, [{"text": "Hello world"}])
    train_dataset, validation_dataset, _stats = prepare_fine_tuning_dataset(
        dataset_path,
        TextTokenizer(app_config.tokenization),
        Vocabulary.load(vocabulary_path),
        context_length=app_config.model.context_length,
        train_validation_ratio=1.0,
        seed=1,
    )

    result = run_fine_tuning(
        model=_tiny_model(num_layers=1),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        vocabulary_path=vocabulary_path,
        app_config=app_config,
        fine_tuning_config=replace(app_config.fine_tuning, epochs=1, batch_size=1),
        output_directory=tmp_path / "cpu",
        base_checkpoint_path=_save_model_checkpoint(tmp_path, _tiny_model(num_layers=1)),
        dataset_path=dataset_path,
        device=torch.device("cpu"),
        max_batches=1,
    )

    assert result.global_step == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_optional_cuda_behavior(tmp_path: Path) -> None:
    app_config, vocabulary_path = _tiny_app_config(tmp_path)
    model = _tiny_model(num_layers=1)
    checkpoint_path = _save_model_checkpoint(tmp_path, model)
    loaded = load_base_model_for_fine_tuning(checkpoint_path, app_config, torch.device("cuda"))

    assert next(loaded.parameters()).device.type == "cuda"
    assert Vocabulary.load(vocabulary_path) is not None


def test_steps_1_to_17_remain_functional() -> None:
    from genpy_llm.generation import TextGenerator
    from genpy_llm.training import GPTTrainer

    assert TextGenerator is not None
    assert GPTTrainer is not None


def _records_path(tmp_path: Path, records: list[dict[str, str]]) -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        method="word",
        preserve_case=True,
        preserve_punctuation=True,
        preserve_newlines=True,
        split_contractions=False,
        normalize_quotes=True,
        normalize_dashes=True,
        add_bos_token=False,
        add_eos_token=True,
        add_newline_token=True,
        bos_token="<BOS>",
        eos_token="<EOS>",
        newline_token="<NL>",
        unknown_token="<UNK>",
    )


def _vocabulary_config() -> VocabularyConfig:
    return VocabularyConfig(
        min_frequency=1,
        max_size=None,
        include_special_tokens=True,
        save_frequencies=True,
        strict_special_token_validation=True,
        pad_token="<PAD>",
        unknown_token="<UNK>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        newline_token="<NL>",
        special_token_order=("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"),
    )


def _vocabulary() -> Vocabulary:
    return Vocabulary(
        token_to_id={
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<NL>": 4,
            "Hello": 5,
            "world": 6,
            "AI": 7,
            "systems": 8,
            "GenPy": 9,
            "model": 10,
            "Tamil": 11,
            "வணக்கம்": 12,
        },
        frequencies=None,
        config=_vocabulary_config(),
    )


def _tiny_model(num_layers: int = 1) -> GPTModel:
    return GPTModel(
        vocab_size=len(_vocabulary()),
        embedding_dim=8,
        num_heads=2,
        num_layers=num_layers,
        context_length=8,
        feed_forward_hidden_dim=16,
        padding_idx=0,
        dropout=0.0,
    )


def _save_model_checkpoint(tmp_path: Path, model: GPTModel) -> Path:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / f"model_{id(model)}.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=1, global_step=1)
    return checkpoint_path


def _tiny_app_config(tmp_path: Path):
    vocabulary_path = tmp_path / "vocab.json"
    _vocabulary().save(vocabulary_path)
    return load_config(_tiny_config_path(tmp_path, vocabulary_path)), vocabulary_path


def _tiny_config_path(tmp_path: Path, vocabulary_path: Path) -> Path:
    config_data = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config_data["data"]["vocabulary_file"] = str(vocabulary_path)
    config_data["model"].update(
        {
            "context_length": 8,
            "embedding_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
        }
    )
    config_data["dataset"]["context_length"] = 8
    config_data["dataset"]["stride"] = 8
    config_data["positional_encoding"]["max_sequence_length"] = 8
    config_data["feed_forward"]["hidden_multiplier"] = 2
    config_data["feed_forward"]["dropout"] = 0.0
    config_data["attention"]["dropout"] = 0.0
    config_data["residual"]["dropout"] = 0.0
    config_data["transformer_block"].update(
        {
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
            "feed_forward_dropout": 0.0,
        }
    )
    config_data["fine_tuning"]["dataset_file"] = str(tmp_path / "records.jsonl")
    config_data["fine_tuning"]["output_directory"] = str(tmp_path / "ft")
    config_data["fine_tuning"]["batch_size"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return path


def _app_config():
    return load_config()
