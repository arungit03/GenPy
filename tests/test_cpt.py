from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.code_tokenizer import (
    CodeTokenizer,
    tokenizer_file_hash,
    train_byte_level_bpe_tokenizer,
)
from genpy_llm.cpt import (
    CPT_PHASE,
    CPTCosineScheduler,
    CPTError,
    build_phase6_config,
    check_cpt_readiness,
    load_cpt_config,
    resolve_cpt_checkpoint,
    run_cpt,
)
from genpy_llm.optimizers import create_optimizer_with_metadata
from genpy_llm.pretraining import CosineWarmupScheduler, Phase6Trainer, load_phase6_config
from genpy_llm.pretraining_dataset import PackedSequenceDataset
from genpy_llm.sequence_packer import PackedSequence
from genpy_llm.shard_builder import SequenceShardWriter, write_sequence_shard_index


def test_cpt_dataset_loads_final_corpus_shards(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = load_cpt_config(paths["cpt"])
    source = resolve_cpt_checkpoint("latest", config)
    phase6 = build_phase6_config(config, source)

    tokenizer = CodeTokenizer.from_file(config.paths.tokenizer)
    dataset = PackedSequenceDataset(
        phase6.data.shard_pattern,
        tokenizer=tokenizer,
        manifest_path=phase6.data.shard_index,
        sequence_length=phase6.model.context_length + 1,
        mmap=True,
    )

    assert len(dataset) == 4
    item = dataset[0]
    assert tuple(item["input_ids"].shape) == (4,)
    assert tuple(item["target_ids"].shape) == (4,)
    assert tuple(item["attention_mask"].shape) == (4,)


def test_cpt_trains_saves_spec_layout_and_benchmarks(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = load_cpt_config(paths["cpt"])

    result = run_cpt(config, resume="latest")

    assert result.status == "completed"
    assert result.start_step == 0
    assert result.global_step == 2
    step_dir = config.paths.checkpoint_output_dir / "checkpoint_step_00002"
    assert result.checkpoint_directory == step_dir
    for name in ("model.pt", "optimizer.pt", "scheduler.pt", "trainer_state.json", "config.json"):
        assert (step_dir / name).is_file(), name
    assert (config.paths.checkpoint_output_dir / "last_checkpoint.pt").is_file()
    assert (config.paths.report_dir / "training_log.csv").is_file()
    assert (config.paths.report_dir / "training_curves.json").is_file()
    assert (config.paths.report_dir / "checkpoint_history.json").is_file()
    assert result.summary_path.is_file()
    summary_heading = result.summary_path.read_text(encoding="utf-8").splitlines()[0]
    assert summary_heading == "# GenPy Continued Pretraining (Final Corpus) Summary"
    assert result.benchmark_json is not None and result.benchmark_json.is_file()
    assert result.benchmark_markdown is not None and result.benchmark_markdown.is_file()

    trainer_state = json.loads((step_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert trainer_state["global_step"] == 2
    assert trainer_state["start_step"] == 0
    assert trainer_state["phase"] == CPT_PHASE
    saved_config = json.loads((step_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_config["training"]["max_steps"] == 2

    phase6 = load_phase6_config(
        paths["training"],
        model_config=paths["model"],
        optimizer_config=paths["optimizer"],
        generation_config=paths["generation"],
    )
    model = Phase6Trainer(phase6).model
    optimizer, _metadata = create_optimizer_with_metadata(model, phase6.optimizer)
    loaded = load_checkpoint(
        step_dir / "model.pt",
        model,
        optimizer,
        map_location="cpu",
        restore_rng=False,
    )
    assert loaded.global_step == 2
    assert loaded.extra_state["phase"] == CPT_PHASE
    assert loaded.extra_state["cpt"]["start_step"] == 0

    optimizer_state = torch.load(step_dir / "optimizer.pt", map_location="cpu")
    assert optimizer_state["state"], "optimizer momentum state must be saved"
    scheduler_state = torch.load(step_dir / "scheduler.pt", map_location="cpu")
    assert scheduler_state["step_count"] == 2
    assert scheduler_state["start_step"] == 0


def test_cpt_resume_latest_continues_exactly(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = load_cpt_config(paths["cpt"])
    first = run_cpt(config, resume="latest")
    assert first.global_step == 2

    extended = replace(config, training=replace(config.training, max_steps=4))
    second = run_cpt(extended, resume="latest")

    assert second.start_step == 0
    assert second.global_step == 4
    assert second.source_checkpoint.name == "model.pt"
    assert second.source_checkpoint.parent.name == "checkpoint_step_00002"
    step_dir = config.paths.checkpoint_output_dir / "checkpoint_step_00004"
    trainer_state = json.loads((step_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert trainer_state["global_step"] == 4
    assert trainer_state["start_step"] == 0
    scheduler_state = torch.load(step_dir / "scheduler.pt", map_location="cpu")
    assert scheduler_state["step_count"] == 4
    optimizer_state = torch.load(step_dir / "optimizer.pt", map_location="cpu")
    steps = [entry.get("step") for entry in optimizer_state["state"].values()]
    assert steps and all(
        (float(value) if not isinstance(value, torch.Tensor) else float(value.item())) >= 4
        for value in steps
    ), "optimizer per-parameter step counters must continue, not restart"


def test_resolve_cpt_checkpoint_prefers_step_directories(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = load_cpt_config(paths["cpt"])

    base = resolve_cpt_checkpoint("latest", config)
    assert base == paths["source_checkpoint"].resolve()

    output = config.paths.checkpoint_output_dir
    (output / "checkpoint_step_00005").mkdir(parents=True)
    (output / "checkpoint_step_00005" / "model.pt").write_bytes(b"x")
    (output / "checkpoint_step_00012").mkdir()
    (output / "checkpoint_step_00012" / "model.pt").write_bytes(b"x")
    (output / "last_checkpoint.pt").write_bytes(b"x")

    latest = resolve_cpt_checkpoint("latest", config)
    assert latest == (output / "checkpoint_step_00012" / "model.pt").resolve()

    explicit_dir = resolve_cpt_checkpoint(output / "checkpoint_step_00005", config)
    assert explicit_dir == (output / "checkpoint_step_00005" / "model.pt").resolve()

    with pytest.raises(FileNotFoundError):
        resolve_cpt_checkpoint(tmp_path / "missing.pt", config)


def test_cpt_readiness_rejects_sequence_length_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = load_cpt_config(paths["cpt"])
    broken = replace(config, training=replace(config.training, sequence_length=9))

    with pytest.raises(CPTError, match="sequence_length"):
        check_cpt_readiness(broken, "latest")


def test_cpt_scheduler_is_leg_relative(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = CPTCosineScheduler(
        optimizer,
        start_step=100,
        leg_steps=10,
        warmup_steps=2,
        minimum_learning_rate_ratio=0.1,
    )

    assert scheduler.get_last_lr()[0] == pytest.approx(0.01 * 1e-12)
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.005)
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.01)
    for _ in range(8):
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.001)

    state = scheduler.state_dict()
    assert state["start_step"] == 100
    assert state["step_count"] == 110
    reloaded = CPTCosineScheduler(
        optimizer,
        start_step=0,
        leg_steps=1,
        warmup_steps=0,
        minimum_learning_rate_ratio=0.1,
    )
    reloaded.load_state_dict(state)
    assert reloaded.start_step == 100
    assert reloaded.step_count == 110


def _fixture(root: Path) -> dict[str, Path]:
    tokenizer_path = _tokenizer(root)
    corpus_dir = _final_corpus(root, tokenizer_path)
    paths = _phase6_configs(root, corpus_dir, tokenizer_path)
    source_checkpoint = root / "checkpoints" / "last_checkpoint.pt"
    _source_checkpoint(paths, source_checkpoint)
    cpt = root / "continued_pretraining.yaml"
    cpt.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "project_root": ".",
                "continued_pretraining": {
                    "paths": {
                        "training_config": str(paths["training"]),
                        "model_config": str(paths["model"]),
                        "optimizer_config": str(paths["optimizer"]),
                        "generation_config": str(paths["generation"]),
                        "corpus_directory": str(corpus_dir),
                        "tokenizer": str(tokenizer_path),
                        "checkpoint_search_dir": str(root / "checkpoints"),
                        "checkpoint_output_dir": str(
                            root / "checkpoints" / "continued_pretraining"
                        ),
                        "report_dir": str(root / "reports" / "continued_pretraining"),
                        "log_file": str(root / "logs" / "continued_pretraining.jsonl"),
                    },
                    "training": {
                        "learning_rate": 0.0005,
                        "batch_size": 1,
                        "gradient_accumulation_steps": 1,
                        "epochs": 2,
                        "max_steps": 2,
                        "checkpoint_interval_steps": 2,
                        "validation_interval_steps": 1,
                        "log_interval_steps": 1,
                        "warmup_steps": 1,
                        "weight_decay": 0.0,
                        "sequence_length": 5,
                        "device": "cpu",
                        "precision": "fp32",
                        "max_grad_norm": 1.0,
                        "keep_last_checkpoints": 5,
                        "validation_fraction": 0.5,
                        "validation_steps": 1,
                        "shuffle": False,
                        "seed": 123,
                    },
                    "early_stopping": {
                        "enabled": False,
                        "patience": 3,
                        "min_improvement": 0.0,
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
        "tokenizer": tokenizer_path,
        "corpus": corpus_dir,
        "cpt": cpt,
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


def _final_corpus(root: Path, tokenizer_path: Path) -> Path:
    tokenizer = CodeTokenizer.from_file(tokenizer_path)
    output = root / "python_corpus" / "final_corpus" / "packed"
    writer = SequenceShardWriter(
        output,
        max_tokens_per_shard=100,
        context_length=4,
        prefix="final_corpus",
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
    write_sequence_shard_index(
        output / "index.json",
        shard_stats,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_file_hash(tokenizer_path),
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        context_length=4,
        source_manifest=root / "python_corpus" / "final_corpus" / "manifest.jsonl",
        creation_timestamp="2026-01-01T00:00:00+00:00",
        build_fingerprint="cpt-fixture",
    )
    return output


def _phase6_configs(root: Path, corpus_dir: Path, tokenizer_path: Path) -> dict[str, Path]:
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
                        "shard_pattern": str(corpus_dir / "final_corpus_*.bin"),
                        "shard_index": str(corpus_dir / "index.json"),
                        "manifest": str(corpus_dir / "index.json"),
                        "tokenizer": str(tokenizer_path),
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
