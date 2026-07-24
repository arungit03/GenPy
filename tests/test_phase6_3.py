from __future__ import annotations

import json
from pathlib import Path

import yaml

from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.code_tokenizer import (
    CodeTokenizer,
    tokenizer_file_hash,
    train_byte_level_bpe_tokenizer,
)
from genpy_llm.continued_training import (
    Phase63Error,
    check_phase63_readiness,
    load_phase63_config,
    run_phase63,
)
from genpy_llm.optimizers import create_optimizer_with_metadata
from genpy_llm.pretraining import CosineWarmupScheduler, Phase6Trainer, load_phase6_config
from genpy_llm.sequence_packer import PackedSequence
from genpy_llm.shard_builder import SequenceShardWriter, write_sequence_shard_index


def test_phase63_readiness_rejects_failed_corpus(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, readiness=False)
    config = load_phase63_config(paths["phase63"])

    try:
        check_phase63_readiness(config)
    except Phase63Error as exc:
        assert "readiness gate failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected readiness failure")


def test_phase63_resumes_trains_saves_logs_and_benchmarks(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, readiness=True)
    config = load_phase63_config(paths["phase63"])

    result = run_phase63(config)

    assert result.status == "completed"
    assert result.global_step == 1
    assert result.last_checkpoint is not None
    assert result.last_checkpoint.is_file()
    assert (config.paths.checkpoint_output_dir / "epoch_001.pt").is_file()
    assert (config.paths.checkpoint_output_dir / "last_checkpoint.pt").is_file()
    assert (config.paths.checkpoint_output_dir / "best_checkpoint.pt").is_file()
    assert (config.paths.report_dir / "training_log.csv").is_file()
    assert (config.paths.report_dir / "training_log.json").is_file()
    assert (config.paths.report_dir / "training_curves.json").is_file()
    assert (config.paths.report_dir / "checkpoint_history.json").is_file()
    assert result.benchmark_json is not None
    assert result.benchmark_json.is_file()
    assert result.benchmark_markdown is not None
    assert result.benchmark_markdown.is_file()

    phase6 = load_phase6_config(
        paths["training"],
        model_config=paths["model"],
        optimizer_config=paths["optimizer"],
        generation_config=paths["generation"],
    )
    tokenizer = CodeTokenizer.from_file(paths["tokenizer"])
    loaded_model = Phase6Trainer(phase6).model
    loaded_optimizer, _metadata = create_optimizer_with_metadata(loaded_model, phase6.optimizer)
    loaded = load_checkpoint(
        result.last_checkpoint,
        loaded_model,
        loaded_optimizer,
        map_location="cpu",
        restore_rng=False,
    )

    assert loaded.global_step == 1
    assert loaded.extra_state["phase"] == "6.3"

    comparison = json.loads(result.benchmark_json.read_text(encoding="utf-8"))
    assert comparison["previous"]["validation_loss"] is not None
    assert comparison["continued"]["perplexity"] is not None
    assert tokenizer.vocab_size == loaded_model.vocab_size


def test_phase63_early_stopping_state_is_configured(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, readiness=True, patience=1)
    config = load_phase63_config(paths["phase63"])

    assert config.early_stopping.enabled is True
    assert config.early_stopping.patience == 1
    assert config.training.max_steps == 1


def _fixture(root: Path, *, readiness: bool, patience: int = 2) -> dict[str, Path]:
    tokenizer_path = _tokenizer(root)
    artifacts = _packed_corpus(root, tokenizer_path)
    paths = _phase6_configs(root, artifacts)
    source_checkpoint = root / "checkpoints" / "last_checkpoint.pt"
    _corpus_v2_reports(
        root,
        tokenizer_path,
        artifacts,
        readiness=readiness,
        minimum_tokens=1,
    )
    _source_checkpoint(paths, source_checkpoint)
    phase63 = root / "phase6_3.yaml"
    phase63.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "project_root": ".",
                "phase6_3": {
                    "paths": {
                        "phase6_training_config": str(paths["training"]),
                        "model_config": str(paths["model"]),
                        "optimizer_config": str(paths["optimizer"]),
                        "generation_config": str(paths["generation"]),
                        "corpus_index": str(artifacts["index"]),
                        "corpus_manifest": str(
                            root / "data" / "corpus_v2" / "document_manifest.jsonl"
                        ),
                        "corpus_statistics": str(artifacts["statistics"]),
                        "corpus_report_manifest": str(
                            root / "reports" / "corpus_v2" / "manifest.json"
                        ),
                        "corpus_quality_report": str(
                            root / "reports" / "corpus_v2" / "quality_report.json"
                        ),
                        "checkpoint_search_dir": str(root / "checkpoints"),
                        "checkpoint_output_dir": str(root / "checkpoints" / "pretraining_v2"),
                        "report_dir": str(root / "reports" / "pretraining_v2"),
                        "log_file": str(root / "logs" / "phase6_3.jsonl"),
                    },
                    "training": {
                        "source_checkpoint": str(source_checkpoint),
                        "max_steps": 1,
                        "max_epochs": 1,
                        "batch_size": 1,
                        "device": "cpu",
                        "validation_interval_steps": 1,
                        "checkpoint_interval_steps": 1,
                        "max_grad_norm": 1.0,
                        "exploding_gradient_threshold": 1000.0,
                    },
                    "early_stopping": {
                        "enabled": True,
                        "patience": patience,
                        "min_delta": 0.0,
                        "monitor": "validation_loss",
                        "mode": "min",
                    },
                    "benchmark": {
                        "enabled": True,
                        "validation_batches": 1,
                        "prompt_count": 1,
                        "max_new_tokens": 1,
                    },
                },
                "logging": {"level": "INFO"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        **paths,
        **artifacts,
        "tokenizer": tokenizer_path,
        "phase63": phase63,
        "source_checkpoint": source_checkpoint,
    }


def _tokenizer(root: Path) -> Path:
    corpus = root / "tokenizer_corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "instruction": "Write Python.",
                "output": "def add(left, right):\n    return left + right\nclass Box:\n    pass\n",
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


def _packed_corpus(root: Path, tokenizer_path: Path) -> dict[str, Path]:
    tokenizer = CodeTokenizer.from_file(tokenizer_path)
    output = root / "data" / "corpus_v2"
    writer = SequenceShardWriter(
        output,
        max_tokens_per_shard=100,
        context_length=4,
        prefix="corpus_v2",
    )
    for index in range(4):
        writer.write_sequence(
            PackedSequence(
                token_ids=[
                    tokenizer.bos_token_id,
                    10 + index,
                    20 + index,
                    30 + index,
                    tokenizer.eos_token_id,
                ],
                sequence_index=index,
                document_offsets=[{"stored_path": f"doc_{index}.py"}],
            )
        )
    shard_stats = writer.close()
    fingerprint = "phase63-fixture"
    index = write_sequence_shard_index(
        output / "index.json",
        shard_stats,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_file_hash(tokenizer_path),
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        context_length=4,
        source_manifest=output / "document_manifest.jsonl",
        creation_timestamp="2026-01-01T00:00:00+00:00",
        build_fingerprint=fingerprint,
    )
    statistics = {
        "build_fingerprint": fingerprint,
        "sequence_count": index["sequence_count"],
        "token_count": index["token_count"],
        "byte_count": index["byte_count"],
        "shard_count": len(index["shards"]),
    }
    (output / "statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    (output / "document_manifest.jsonl").write_text(
        json.dumps({"stored_path": "doc_0.py", "token_count": 10}) + "\n",
        encoding="utf-8",
    )
    return {
        "corpus": output,
        "index": output / "index.json",
        "statistics": output / "statistics.json",
    }


def _phase6_configs(root: Path, artifacts: dict[str, Path]) -> dict[str, Path]:
    model = root / "model.yaml"
    optimizer = root / "optimizer.yaml"
    generation = root / "generation.yaml"
    training = root / "training.yaml"
    model.write_text(
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
                    "activation": "gelu",
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
    optimizer.write_text(
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
    generation.write_text(
        yaml.safe_dump(
            {
                "generation": {
                    "prompts": ["def add"],
                    "max_new_tokens": 1,
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
    training.write_text(
        yaml.safe_dump(
            {
                "pretraining": {
                    "data": {
                        "shard_pattern": str(artifacts["corpus"] / "corpus_v2_*.bin"),
                        "shard_index": str(artifacts["index"]),
                        "manifest": str(root / "reports" / "corpus_v2" / "manifest.json"),
                        "tokenizer": str(root / "tokenizer" / "tokenizer.json"),
                        "validation_fraction": 0.5,
                        "batch_size": 1,
                        "dataloader_workers": 0,
                        "pin_memory": False,
                        "prefetch_factor": None,
                        "mmap": True,
                        "shuffle": False,
                        "seed": 123,
                    },
                    "training": {
                        "seed": 123,
                        "max_steps": 10,
                        "gradient_accumulation_steps": 1,
                        "max_grad_norm": 1.0,
                        "device": "cpu",
                        "mixed_precision": "none",
                        "log_every_steps": 1,
                        "save_every_steps": 1,
                        "validate_every_steps": 1,
                        "validation_steps": 1,
                        "keep_last": 3,
                        "resume": False,
                        "resume_from": None,
                    },
                    "scheduler": {"warmup_steps": 0, "minimum_learning_rate_ratio": 0.1},
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
                        "samples_directory": str(root / "samples"),
                        "tensorboard_directory": str(root / "tensorboard"),
                        "log_file": str(root / "logs" / "pretraining.jsonl"),
                    },
                },
                "logging": {"level": "INFO"},
            }
        ),
        encoding="utf-8",
    )
    return {"model": model, "optimizer": optimizer, "generation": generation, "training": training}


def _source_checkpoint(paths: dict[str, Path], checkpoint_path: Path) -> None:
    phase6 = load_phase6_config(
        paths["training"],
        model_config=paths["model"],
        optimizer_config=paths["optimizer"],
        generation_config=paths["generation"],
    )
    trainer = Phase6Trainer(phase6)
    scheduler = CosineWarmupScheduler(
        trainer.optimizer,
        max_steps=10,
        warmup_steps=0,
        minimum_learning_rate_ratio=0.1,
    )
    save_checkpoint(
        checkpoint_path,
        trainer.model,
        trainer.optimizer,
        epoch=0,
        global_step=0,
        scheduler=scheduler,
        scaler=trainer.scaler,
        model_config=phase6.model.__dict__,
        vocabulary_metadata={
            "tokenizer": str(phase6.data.tokenizer),
            "tokenizer_sha256": tokenizer_file_hash(phase6.data.tokenizer),
        },
    )


def _corpus_v2_reports(
    root: Path,
    tokenizer_path: Path,
    artifacts: dict[str, Path],
    *,
    readiness: bool,
    minimum_tokens: int,
) -> None:
    report_dir = root / "reports" / "corpus_v2"
    report_dir.mkdir(parents=True)
    manifest = {
        "build_fingerprint": "phase63-fixture",
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_file_hash(tokenizer_path),
        "statistics": {"total_tokens": 20},
        "readiness": {"passed": readiness, "failures": [] if readiness else ["token_target"]},
        "shards": {"index": str(artifacts["index"]), "token_count": 20, "shard_count": 1},
    }
    quality = {
        "statistics": {"total_tokens": 20},
        "readiness": {
            "passed": readiness,
            "failures": [] if readiness else ["token_target"],
            "settings": {"minimum_tokens": minimum_tokens},
        },
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (report_dir / "quality_report.json").write_text(json.dumps(quality), encoding="utf-8")
