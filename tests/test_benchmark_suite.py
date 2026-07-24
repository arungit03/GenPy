from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from genpy_llm.benchmark_prompts import (
    DOCUMENTATION_QA,
    PYTHON_BENCHMARK_PROMPTS,
    TEXT_GENERATION_TASKS,
)
from genpy_llm.benchmark_suite import (
    coherence_score,
    keyword_coverage,
    load_benchmark_config,
    repetition_rate,
    resolve_benchmark_checkpoint,
    run_benchmark,
)
from genpy_llm.checkpointing import save_checkpoint
from genpy_llm.code_tokenizer import (
    CodeTokenizer,
    tokenizer_file_hash,
    train_byte_level_bpe_tokenizer,
)
from genpy_llm.optimizers import create_optimizer_with_metadata
from genpy_llm.pretraining import Phase6Trainer, load_phase6_config
from genpy_llm.sequence_packer import PackedSequence
from genpy_llm.shard_builder import SequenceShardWriter, write_sequence_shard_index

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_python_prompt_dataset_meets_requirements() -> None:
    assert len(PYTHON_BENCHMARK_PROMPTS) >= 100
    categories = Counter(prompt.category for prompt in PYTHON_BENCHMARK_PROMPTS)
    expected = {
        "algorithms",
        "data_structures",
        "oop",
        "typing",
        "decorators",
        "generators",
        "asyncio",
        "fastapi",
        "flask",
        "django",
        "numpy",
        "pandas",
        "pytorch",
        "regex",
        "file_handling",
        "cli",
        "testing",
        "logging",
        "json",
        "csv",
        "sql",
    }
    assert set(categories) == expected
    assert all(count >= 5 for count in categories.values())
    ids = [prompt.id for prompt in PYTHON_BENCHMARK_PROMPTS]
    assert len(ids) == len(set(ids))


def test_documentation_qa_dataset_meets_requirements() -> None:
    assert len(DOCUMENTATION_QA) >= 100
    sources = Counter(question.source for question in DOCUMENTATION_QA)
    assert set(sources) == {
        "python_docs",
        "peps",
        "fastapi",
        "numpy",
        "pandas",
        "django",
        "flask",
    }
    assert all(count >= 10 for count in sources.values())
    assert all(question.keywords for question in DOCUMENTATION_QA)
    ids = [question.id for question in DOCUMENTATION_QA]
    assert len(ids) == len(set(ids))
    assert len(TEXT_GENERATION_TASKS) >= 10


def test_scoring_helpers() -> None:
    answer = "The len builtin returns the length of items"
    assert keyword_coverage(answer, ("length", "items")) == 1.0
    assert keyword_coverage("no match here", ("missing",)) == 0.0
    assert keyword_coverage("anything", ()) == 1.0
    assert repetition_rate([]) == 0.0
    assert repetition_rate([7, 7, 7, 7]) == pytest.approx(0.75)
    assert repetition_rate([1, 2, 3, 4]) == 0.0
    assert coherence_score([1, 2, 3, 4]) == 1.0
    assert coherence_score([5, 5, 5, 5]) == pytest.approx(1 / 3)
    assert coherence_score([1]) == 0.0


def test_resolve_benchmark_checkpoint_orders(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = load_benchmark_config(paths["benchmark"])

    base = resolve_benchmark_checkpoint("latest_base", "base", config)
    assert base == (tmp_path / "checkpoints" / "last_checkpoint.pt").resolve()

    continued = resolve_benchmark_checkpoint("latest", "continued", config)
    assert continued.parent.name == "checkpoint_step_00002"

    explicit = resolve_benchmark_checkpoint(continued.parent, "continued", config)
    assert explicit == continued

    with pytest.raises(FileNotFoundError):
        resolve_benchmark_checkpoint(tmp_path / "missing.pt", "base", config)


def test_full_benchmark_pipeline_writes_reports_and_plots(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = load_benchmark_config(paths["benchmark"])

    result = run_benchmark(config, base="latest_base", continued="latest")

    output = config.paths.output_dir
    assert result.metrics_path == output / "metrics.json"
    assert result.summary_path.is_file()
    assert result.comparison_path.is_file()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    for side in ("base", "continued"):
        payload = metrics[side]
        assert payload["validation"]["loss"] > 0
        assert payload["validation"]["perplexity"] > 0
        assert 0 <= payload["validation"]["next_token_accuracy"] <= 1
        assert payload["latency"]["runs"] == 1
        assert payload["python_benchmark"]["prompt_count"] == 2
        assert payload["documentation_qa"]["question_count"] == 2
        assert payload["text_generation"]["task_count"] == 1
        assert payload["profile"]["checkpoint_size_bytes"] > 0
        assert payload["profile"]["load_seconds"] > 0
    comparison = metrics["comparison"]
    assert "overall_improvement_percent" in comparison
    assert isinstance(result.overall_improvement_percent, float)

    summary = result.summary_path.read_text(encoding="utf-8")
    assert "| Validation loss |" in summary
    assert "Overall improvement:" in summary
    comparison_text = result.comparison_path.read_text(encoding="utf-8")
    assert "| Metric | Base | Continued | Delta |" in comparison_text
    assert "pass rate by category" in comparison_text

    for name in ("loss.png", "perplexity.png", "speed.png", "memory.png", "latency.png"):
        png = (result.plots_dir / name).read_bytes()
        assert png.startswith(PNG_MAGIC), name
    legend = json.loads((result.plots_dir / "plots.json").read_text(encoding="utf-8"))
    assert "speed.png" in legend


def _fixture(root: Path) -> dict[str, Path]:
    tokenizer_path = _tokenizer(root)
    corpus_dir = _final_corpus(root, tokenizer_path)
    paths = _phase6_configs(root, corpus_dir, tokenizer_path)
    base_checkpoint = root / "checkpoints" / "last_checkpoint.pt"
    continued_dir = root / "checkpoints" / "continued_pretraining" / "checkpoint_step_00002"
    _write_checkpoint(paths, base_checkpoint, global_step=0)
    _write_checkpoint(paths, continued_dir / "model.pt", global_step=2)
    benchmark = root / "benchmark.yaml"
    benchmark.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "project_root": ".",
                "benchmark": {
                    "paths": {
                        "model_config": str(paths["model"]),
                        "training_config": str(paths["training"]),
                        "optimizer_config": str(paths["optimizer"]),
                        "generation_config": str(paths["generation"]),
                        "tokenizer": str(tokenizer_path),
                        "corpus_directory": str(corpus_dir),
                        "base_search_dir": str(root / "checkpoints"),
                        "continued_search_dir": str(
                            root / "checkpoints" / "continued_pretraining"
                        ),
                        "output_dir": str(root / "reports" / "benchmark"),
                        "training_metrics": [str(root / "missing_metrics.jsonl")],
                        "continued_training_log": str(root / "missing_log.json"),
                    },
                    "evaluation": {
                        "batch_size": 1,
                        "validation_batches": 2,
                        "validation_fraction": 0.5,
                        "seed": 42,
                        "python_prompt_limit": 2,
                        "doc_qa_limit": 2,
                        "text_task_limit": 1,
                        "latency_runs": 1,
                        "latency_tokens": 2,
                        "execution_timeout_seconds": 5.0,
                        "run_generated_code": True,
                    },
                    "generation": {
                        "temperature": 1.0,
                        "top_p": None,
                        "top_k": None,
                        "max_new_tokens": 4,
                        "do_sample": False,
                        "repetition_penalty": 1.0,
                    },
                    "device": "cpu",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {**paths, "tokenizer": tokenizer_path, "corpus": corpus_dir, "benchmark": benchmark}


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
        build_fingerprint="benchmark-fixture",
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


def _write_checkpoint(paths: dict[str, Path], checkpoint_path: Path, *, global_step: int) -> None:
    phase6 = load_phase6_config(
        paths["training"],
        model_config=paths["model"],
        optimizer_config=paths["optimizer"],
        generation_config=paths["generation"],
    )
    trainer = Phase6Trainer(phase6)
    optimizer, _metadata = create_optimizer_with_metadata(trainer.model, phase6.optimizer)
    save_checkpoint(
        checkpoint_path,
        trainer.model,
        optimizer,
        epoch=1,
        global_step=global_step,
        model_config=phase6.model.__dict__,
        vocabulary_metadata={
            "tokenizer": str(phase6.data.tokenizer),
            "tokenizer_sha256": tokenizer_file_hash(phase6.data.tokenizer),
        },
    )
