from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.code_tokenizer import (
    CodeTokenizer,
    tokenizer_file_hash,
    train_byte_level_bpe_tokenizer,
)
from genpy_llm.pretraining import (
    CosineWarmupScheduler,
    Phase6Trainer,
    create_phase6_model,
    load_phase6_config,
)
from genpy_llm.pretraining_dataset import (
    DeterministicSequenceSampler,
    PackedSequenceDataset,
)
from genpy_llm.pretraining_generation import generate_code_sample
from genpy_llm.sequence_packer import PackedSequence
from genpy_llm.shard_builder import SequenceShardWriter, write_sequence_shard_index


def test_packed_sequence_dataset_reads_mmap_and_plain_file(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    tokenizer = CodeTokenizer.from_file(paths["tokenizer"])

    mmap_dataset = PackedSequenceDataset(
        paths["shard_pattern"],
        tokenizer=tokenizer,
        manifest_path=paths["index"],
        mmap=True,
    )
    plain_dataset = PackedSequenceDataset(
        paths["shard_pattern"],
        tokenizer=tokenizer,
        manifest_path=paths["index"],
        mmap=False,
    )

    assert len(mmap_dataset) == 4
    assert torch.equal(mmap_dataset[0]["input_ids"], plain_dataset[0]["input_ids"])
    assert mmap_dataset[0]["input_ids"].shape == (4,)
    assert mmap_dataset[0]["target_ids"].shape == (4,)
    assert mmap_dataset[0]["attention_mask"].sum().item() > 0


def test_deterministic_sampler_repeats_order_and_changes_by_epoch(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    tokenizer = CodeTokenizer.from_file(paths["tokenizer"])
    dataset = PackedSequenceDataset(
        paths["shard_pattern"],
        tokenizer=tokenizer,
        manifest_path=paths["index"],
    )
    sampler = DeterministicSequenceSampler(dataset, shuffle=True, seed=7)

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(0)

    assert first == list(sampler)
    assert first != second


def test_scheduler_warms_up_and_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = CosineWarmupScheduler(
        optimizer,
        max_steps=10,
        warmup_steps=2,
        minimum_learning_rate_ratio=0.1,
    )

    assert optimizer.param_groups[0]["lr"] < 1.0
    scheduler.step()
    scheduler.step()
    warm_lr = optimizer.param_groups[0]["lr"]
    for _ in range(8):
        scheduler.step()

    assert warm_lr == pytest.approx(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_phase6_config_model_generation_and_training(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    config = _config(tmp_path, paths)
    tokenizer = CodeTokenizer.from_file(paths["tokenizer"])
    model = create_phase6_model(config.model, tokenizer)
    rotary_model = create_phase6_model(
        replace(config.model, positional_embedding="rotary"),
        tokenizer,
    )

    assert model.context_length == 4
    assert len(model.blocks) == 1
    assert model.embeddings_are_tied()
    assert rotary_model.blocks[0].attention.rotary_embeddings is True
    assert rotary_model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)).shape == (1, 4, 320)
    generated = generate_code_sample(
        model=model,
        tokenizer=tokenizer,
        prompt="def add",
        device=torch.device("cpu"),
        context_length=4,
        settings=config.generation,
    )
    assert generated.text

    result = Phase6Trainer(config).train()

    assert result.global_step == 2
    assert result.last_checkpoint is not None
    assert result.last_checkpoint.is_file()
    assert (config.checkpoint.directory / "step_00001.pt").is_file()
    assert (config.checkpoint.directory / "step_00002.pt").is_file()
    assert result.metrics_path.is_file()
    assert (config.outputs.samples_directory / "step_00001.json").is_file()

    resumed_config = _config(tmp_path, paths, max_steps=3, resume=True)
    resumed = Phase6Trainer(resumed_config).train()

    assert resumed.global_step == 3


def test_phase6_legacy_fp32_precision_maps_to_no_mixed_precision(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    config = _config(tmp_path, paths, legacy_precision=True)

    assert config.training.mixed_precision == "none"


def test_phase6_mps_dataloader_settings_are_safely_adjusted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    paths = _artifacts(tmp_path)
    config = _config(tmp_path, paths)
    config = replace(
        config,
        data=replace(
            config.data,
            dataloader_workers=2,
            prefetch_factor=4,
            pin_memory=True,
        ),
    )
    trainer = object.__new__(Phase6Trainer)
    trainer.config = config
    trainer.device = torch.device("mps")

    assert trainer._effective_dataloader_workers() == 0
    trainer.dataloader_workers = 0
    assert trainer._effective_prefetch_factor() is None
    assert trainer._effective_pin_memory() is False
    assert "forcing dataloader_workers=0" in caplog.text
    assert "prefetch_factor requires dataloader_workers > 0" in caplog.text
    assert "pin_memory is only useful for CUDA" in caplog.text


def _artifacts(root: Path) -> dict[str, Path | str]:
    tokenizer_path = _tokenizer(root)
    tokenizer = CodeTokenizer.from_file(tokenizer_path)
    output = root / "pretraining"
    writer = SequenceShardWriter(
        output,
        max_tokens_per_shard=10,
        context_length=4,
        prefix="shard",
    )
    sequences = [
        [tokenizer.bos_token_id, 10, 11, 12, tokenizer.eos_token_id],
        [tokenizer.bos_token_id, 13, 14, 15, tokenizer.eos_token_id],
        [tokenizer.bos_token_id, 16, 17, 18, tokenizer.eos_token_id],
        [tokenizer.bos_token_id, 19, 20, 21, tokenizer.eos_token_id],
    ]
    for index, token_ids in enumerate(sequences):
        writer.write_sequence(
            PackedSequence(
                token_ids=token_ids,
                sequence_index=index,
                document_offsets=[{"stored_path": f"{index}.py"}],
            )
        )
    statistics = writer.close()
    shard_index = write_sequence_shard_index(
        output / "index.json",
        statistics,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_file_hash(tokenizer_path),
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        context_length=4,
        source_manifest=output / "corpus_manifest.jsonl",
        creation_timestamp="2026-01-01T00:00:00+00:00",
        build_fingerprint="test",
    )
    manifest = {
        "tokenizer_hash": tokenizer_file_hash(tokenizer_path),
        "context_length": 4,
        "training_sequences": shard_index["sequence_count"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "tokenizer": tokenizer_path,
        "output": output,
        "index": output / "index.json",
        "manifest": output / "manifest.json",
        "shard_pattern": str(output / "shard_*.bin"),
    }


def _tokenizer(root: Path) -> Path:
    corpus = root / "tokenizer_corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "instruction": "Write Python.",
                "output": "def add(a, b):\n    return a + b\nclass Stack:\n    pass\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer_path = root / "tokenizer" / "tokenizer.json"
    train_byte_level_bpe_tokenizer(
        [corpus],
        output_path=tokenizer_path,
        metadata_path=tokenizer_path.with_name("tokenizer_metadata.json"),
        vocab_size=320,
        min_frequency=1,
        show_progress=False,
    )
    return tokenizer_path


def _config(
    root: Path,
    paths: dict[str, Path | str],
    *,
    max_steps: int = 2,
    resume: bool = False,
    legacy_precision: bool = False,
):
    model_path = root / "model.yaml"
    optimizer_path = root / "optimizer.yaml"
    generation_path = root / "generation.yaml"
    training_path = root / "training.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocabulary_size": 320,
                    "context_length": 4,
                    "hidden_size": 8,
                    "ffn_size": 16,
                    "decoder_layers": 1,
                    "attention_heads": 2,
                    "dropout": 0.0,
                    "attention_dropout": 0.0,
                    "residual_dropout": 0.0,
                    "activation": "swiglu",
                    "layer_norm_epsilon": 1e-5,
                    "positional_embedding": "learned",
                    "tied_embedding_weights": True,
                    "use_bias": True,
                    "initialization_std": 0.02,
                    "gradient_checkpointing": False,
                    "torch_compile": False,
                    "compile_mode": "default",
                    "flash_attention": "disabled",
                }
            }
        ),
        encoding="utf-8",
    )
    optimizer_path.write_text(
        yaml.safe_dump(
            {
                "optimizer": {
                    "type": "adamw",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "epsilon": 1e-8,
                    "separate_weight_decay": True,
                }
            }
        ),
        encoding="utf-8",
    )
    generation_path.write_text(
        yaml.safe_dump(
            {
                "generation": {
                    "prompts": ["def add"],
                    "max_new_tokens": 2,
                    "temperature": 1.0,
                    "top_k": None,
                    "top_p": None,
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "stop_tokens": ["<eos>"],
                }
            }
        ),
        encoding="utf-8",
    )
    training_settings = {
        "seed": 123,
        "max_steps": max_steps,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "device": "cpu",
        "log_every_steps": 1,
        "save_every_steps": 1,
        "validate_every_steps": 1,
        "validation_steps": 1,
        "keep_last": 3,
        "resume": resume,
        "resume_from": None,
    }
    if legacy_precision:
        training_settings["precision"] = "fp32"
    else:
        training_settings["mixed_precision"] = "none"

    training_path.write_text(
        yaml.safe_dump(
            {
                "pretraining": {
                    "data": {
                        "shard_pattern": paths["shard_pattern"],
                        "shard_index": str(paths["index"]),
                        "manifest": str(paths["manifest"]),
                        "tokenizer": str(paths["tokenizer"]),
                        "validation_fraction": 0.5,
                        "batch_size": 1,
                        "dataloader_workers": 0,
                        "pin_memory": False,
                        "prefetch_factor": None,
                        "mmap": True,
                        "shuffle": False,
                        "seed": 123,
                    },
                    "training": training_settings,
                    "scheduler": {
                        "warmup_steps": 0,
                        "minimum_learning_rate_ratio": 0.1,
                    },
                    "checkpoint": {
                        "directory": str(root / "checkpoints"),
                        "step_prefix": "step",
                        "best_filename": "best_model.pt",
                        "last_filename": "last_checkpoint.pt",
                        "monitor": "validation_loss",
                        "mode": "min",
                    },
                    "outputs": {
                        "metrics_directory": str(root / "metrics"),
                        "samples_directory": str(root / "generated_samples"),
                        "tensorboard_directory": str(root / "tensorboard"),
                        "log_file": str(root / "logs" / "pretraining.jsonl"),
                    },
                },
                "logging": {"level": "INFO"},
            }
        ),
        encoding="utf-8",
    )
    return load_phase6_config(
        training_path,
        model_config=model_path,
        optimizer_config=optimizer_path,
        generation_config=generation_path,
    )
