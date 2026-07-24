"""Tests for the training-pipeline remediation: scheduler, context length,
deduplication, and dataset validation (see reports/training_pipeline/)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.checkpointing import save_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer, train_byte_level_bpe_tokenizer
from genpy_llm.conversation_formatter import ConversationTemplate
from genpy_llm.fine_tuning import Phase7Trainer, load_phase7_config
from genpy_llm.gpt import GPTModel
from genpy_llm.instruction_dataset import InstructionDataset, load_instruction_records
from genpy_llm.pretraining import PretrainingError, compute_scheduler_total_steps
from genpy_llm.sft_dataset_cleaning import (
    SFTRecord,
    analyze_sft_dataset,
    deduplicate_sft_records,
    find_duplicate_instructions,
    find_duplicate_outputs,
    find_duplicate_pairs,
    load_sft_records_lenient,
    write_sft_records,
)

# --------------------------------------------------------------------------
# Task 1: scheduler
# --------------------------------------------------------------------------


def test_compute_scheduler_total_steps_reproduces_the_bug_scenario_correctly() -> None:
    # The exact configuration that previously collapsed to 1: epochs=1, max_steps unset.
    old_buggy_value = 1  # what `config.training.max_steps or max(1, epochs)` produced
    fixed_value = compute_scheduler_total_steps(
        dataset_size=45_572,
        batch_size=1,
        gradient_accumulation_steps=1,
        epochs=1,
        max_steps=None,
    )

    assert fixed_value == 45_572
    assert fixed_value != old_buggy_value


def test_compute_scheduler_total_steps_accounts_for_batch_and_accumulation() -> None:
    steps = compute_scheduler_total_steps(
        dataset_size=1_000,
        batch_size=4,
        gradient_accumulation_steps=2,
        epochs=3,
        max_steps=None,
    )

    assert steps == 3 * 1_000 // 4 // 2 == 375


def test_compute_scheduler_total_steps_explicit_override_wins() -> None:
    steps = compute_scheduler_total_steps(
        dataset_size=1_000,
        batch_size=4,
        gradient_accumulation_steps=2,
        epochs=3,
        max_steps=50,
    )

    assert steps == 50


def test_compute_scheduler_total_steps_floors_at_one() -> None:
    steps = compute_scheduler_total_steps(
        dataset_size=1,
        batch_size=8,
        gradient_accumulation_steps=4,
        epochs=1,
        max_steps=None,
    )

    assert steps == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_size": 0, "batch_size": 1, "gradient_accumulation_steps": 1, "epochs": 1},
        {"dataset_size": 1, "batch_size": 0, "gradient_accumulation_steps": 1, "epochs": 1},
        {"dataset_size": 1, "batch_size": 1, "gradient_accumulation_steps": 0, "epochs": 1},
        {"dataset_size": 1, "batch_size": 1, "gradient_accumulation_steps": 1, "epochs": 0},
    ],
)
def test_compute_scheduler_total_steps_rejects_non_positive_inputs(kwargs: dict) -> None:
    with pytest.raises(PretrainingError):
        compute_scheduler_total_steps(max_steps=None, **kwargs)


def test_phase7_trainer_derives_max_steps_from_dataset_when_unset(tmp_path: Path) -> None:
    """End-to-end: a real Phase7Trainer with max_steps unset must not collapse to 1 step."""

    config = _phase7_config(tmp_path, record_count=8, batch_size=2, epochs=1, max_steps=None)

    trainer = Phase7Trainer(config)

    # 8 records / batch_size 2 / grad_accum 1 * 1 epoch = 4 optimizer steps, not 1.
    assert trainer.scheduler.max_steps == 4
    assert trainer.scheduler.max_steps > 1


def test_phase7_trainer_respects_explicit_max_steps(tmp_path: Path) -> None:
    config = _phase7_config(tmp_path, record_count=8, batch_size=2, epochs=1, max_steps=3)

    trainer = Phase7Trainer(config)

    assert trainer.scheduler.max_steps == 3


# --------------------------------------------------------------------------
# Task 1 (continued): resume continuity with the fixed scheduler
# --------------------------------------------------------------------------


def test_phase7_scheduler_resume_continues_correct_trajectory(tmp_path: Path) -> None:
    config = _phase7_config(
        tmp_path,
        record_count=8,
        batch_size=1,
        epochs=1,
        max_steps=None,
        save_every_steps=1,
        eval_every_steps=100,
    )

    first_run = Phase7Trainer(config).train()
    assert first_run.global_step >= 1

    resumed_trainer = Phase7Trainer(replace(config, training=replace(config.training, resume=True)))
    # The resumed scheduler must still see the full (correct) horizon, not fall back to 1.
    assert resumed_trainer.scheduler.max_steps == 8
    assert resumed_trainer.scheduler.step_count == first_run.global_step

    resumed_result = resumed_trainer.train()
    assert resumed_result.global_step >= first_run.global_step


# --------------------------------------------------------------------------
# Task 2: context length
# --------------------------------------------------------------------------


def test_instruction_dataset_truncates_far_less_at_1024_than_256(tmp_path: Path) -> None:
    tokenizer_path = _tokenizer(tmp_path)
    tokenizer = CodeTokenizer.from_file(tokenizer_path)
    template = ConversationTemplate()
    long_output = "x = 1\n" * 150  # ~900 tokens with this toy tokenizer: exceeds 256, fits in 1024
    records = load_instruction_records(
        _write_jsonl(
            tmp_path / "long.jsonl",
            [
                {"instruction": "Write a long script.", "input": "", "output": long_output},
            ],
        )
    )

    short_context = InstructionDataset(
        records, tokenizer=tokenizer, template=template, context_length=256, mask_prompt_tokens=True
    )
    long_context = InstructionDataset(
        records,
        tokenizer=tokenizer,
        template=template,
        context_length=1024,
        mask_prompt_tokens=True,
    )

    assert short_context.stats.truncated_records == 1
    assert long_context.stats.truncated_records == 0


def test_phase7_config_loads_context_length_1024_from_finetuning_yaml(tmp_path: Path) -> None:
    config = _phase7_config(tmp_path, record_count=4, batch_size=1, epochs=1, max_steps=1)

    assert config.data.context_length == 1024


# --------------------------------------------------------------------------
# Task 3: deduplication
# --------------------------------------------------------------------------


def _record(
    instruction: str, output: str, category: str = "code_generation", input_text: str = ""
) -> SFTRecord:
    raw = {"instruction": instruction, "input": input_text, "output": output, "category": category}
    return SFTRecord(
        line_number=0,
        instruction=instruction,
        input=input_text,
        output=output,
        category=category,
        raw=raw,
    )


def test_deduplicate_sft_records_removes_only_exact_pairs() -> None:
    records = [
        _record("Write add.", "def add(a, b): return a + b"),
        _record("Write add.", "def add(a, b): return a + b"),  # exact duplicate pair
        _record("Write sub.", "def add(a, b): return a + b"),  # same output, different instruction
        _record(
            "Write add.", "def add(a, b):\n    return a + b"
        ),  # same instruction, different output
    ]

    deduplicated, report = deduplicate_sft_records(records)

    assert report.original_size == 4
    assert report.duplicate_pair_records_removed == 1
    assert report.new_size == 3
    assert len(deduplicated) == 3
    # First occurrence is kept, order preserved.
    assert deduplicated[0].instruction == "Write add."
    assert deduplicated[0].output == "def add(a, b): return a + b"


def test_deduplicate_sft_records_does_not_remove_shared_output_across_instructions() -> None:
    """Same output, different instruction: legitimate, must survive (no semantic-meaning change)."""

    records = [
        _record("Analyze complexity of `foo`.", "O(n) time.", category="complexity_analysis"),
        _record("Analyze complexity of `bar`.", "O(n) time.", category="complexity_analysis"),
        _record("Analyze complexity of `baz`.", "O(n) time.", category="complexity_analysis"),
    ]

    deduplicated, report = deduplicate_sft_records(records)

    assert report.duplicate_pair_records_removed == 0
    assert len(deduplicated) == 3


def test_find_duplicate_helpers_report_instructions_outputs_and_pairs_separately() -> None:
    records = [
        _record("Same instruction", "Output A"),
        _record("Same instruction", "Output B"),
        _record("Different instruction", "Output A"),
        _record("Same instruction", "Output A"),
    ]

    duplicate_instructions = find_duplicate_instructions(records)
    duplicate_outputs = find_duplicate_outputs(records)
    duplicate_pairs = find_duplicate_pairs(records)

    assert duplicate_instructions == {"Same instruction": 3}
    assert duplicate_outputs == {"Output A": 3}
    assert duplicate_pairs == {("Same instruction", "", "Output A"): 2}


def test_deduplicate_sft_records_preserves_category_counts_correctly() -> None:
    records = [
        _record("A", "1", category="code_generation"),
        _record("A", "1", category="code_generation"),  # exact duplicate, will be removed
        _record("B", "2", category="explanation"),
    ]

    deduplicated, report = deduplicate_sft_records(records)

    assert report.category_counts_before == {"code_generation": 2, "explanation": 1}
    assert report.category_counts_after == {"code_generation": 1, "explanation": 1}


def test_write_sft_records_roundtrips_and_dedup_script_is_reproducible(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "raw.jsonl",
        [
            {
                "instruction": "Write add.",
                "input": "",
                "output": "def add(a,b): return a+b",
                "category": "code_generation",
            },
            {
                "instruction": "Write add.",
                "input": "",
                "output": "def add(a,b): return a+b",
                "category": "code_generation",
            },
        ],
    )

    records, broken = load_sft_records_lenient(path)
    assert broken == ()
    deduplicated, report = deduplicate_sft_records(records)
    output_path = tmp_path / "deduplicated.jsonl"
    write_sft_records(deduplicated, output_path)

    reloaded, _ = load_sft_records_lenient(output_path)
    assert len(reloaded) == 1
    assert report.original_size == 2
    assert report.new_size == 1


# --------------------------------------------------------------------------
# Task 4: dataset validation
# --------------------------------------------------------------------------


def test_analyze_sft_dataset_reports_empty_broken_and_malformed(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instruction": "Write add.",
                        "output": "def add(): pass",
                        "category": "code_generation",
                    }
                ),
                "{not valid json",
                json.dumps(
                    {"instruction": "", "output": "def foo(): pass", "category": "code_generation"}
                ),
                json.dumps(
                    {"instruction": "Write bar.", "output": "", "category": "code_generation"}
                ),
                json.dumps(
                    {"instruction": 123, "output": "def foo(): pass", "category": "code_generation"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyze_sft_dataset(path)

    assert report.total_lines == 5
    assert len(report.broken_json_lines) == 1
    # The empty-string record and the non-string-instruction record (coerced
    # to "") both count as empty instructions by design.
    assert report.empty_instruction_records == 2
    assert report.empty_output_records == 1
    assert report.malformed_records == 1
    assert report.usable_records == 1


def test_analyze_sft_dataset_ignores_unicode_line_separators_in_line_count(tmp_path: Path) -> None:
    """A NEL/LS/PS character inside a JSON string is legal and must not inflate line counts."""

    path = tmp_path / "unicode.jsonl"
    record = {
        "instruction": "Write add.",
        "output": f"def add(a, b):{chr(0x2028)}    return a + b",
        "category": "code_generation",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = analyze_sft_dataset(path)

    assert report.total_lines == 1
    assert report.usable_records == 1


def test_analyze_sft_dataset_computes_length_statistics(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "lengths.jsonl",
        [
            {"instruction": "short", "output": "a" * 10, "category": "code_generation"},
            {
                "instruction": "a longer instruction here",
                "output": "a" * 100,
                "category": "code_generation",
            },
        ],
    )

    report = analyze_sft_dataset(path)
    stats = report.length_statistics

    assert stats is not None
    assert stats.count == 2
    assert stats.output_min == 10
    assert stats.output_max == 100
    assert stats.instruction_min == len("short")


# --------------------------------------------------------------------------
# Shared test helpers
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _tokenizer(root: Path) -> Path:
    corpus = root / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"instruction": "Write add.", "output": "def add(a, b): return a + b"})
            for _ in range(4)
        ),
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


def _phase7_config(
    root: Path,
    *,
    record_count: int,
    batch_size: int,
    epochs: int,
    max_steps: int | None,
    save_every_steps: int = 1,
    eval_every_steps: int = 1,
):
    tokenizer_path = _tokenizer(root)
    train_path = _write_jsonl(
        root / "train.jsonl",
        [
            {
                "instruction": f"Write function number {index}.",
                "input": "",
                "output": f"def f{index}():\n    return {index}",
            }
            for index in range(record_count)
        ],
    )
    model_path = root / "model.yaml"
    optimizer_path = root / "optimizer.yaml"
    config_path = root / "finetuning.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocabulary_size": 320,
                    "context_length": 128,
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
    tokenizer = CodeTokenizer.from_file(tokenizer_path)
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        context_length=128,
        feed_forward_hidden_dim=16,
        padding_idx=tokenizer.pad_token_id,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    base_checkpoint = root / "base.pt"
    save_checkpoint(
        base_checkpoint,
        model,
        optimizer,
        epoch=0,
        global_step=0,
        model_config={},
        vocabulary_metadata={"tokenizer": str(tokenizer_path)},
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "phase7": {
                    "model_config": str(model_path),
                    "optimizer_config": str(optimizer_path),
                    "learning_rate": 0.001,
                    "warmup_steps": 0,
                    "data": {
                        "train_path": str(train_path),
                        "validation_path": str(train_path),
                        "tokenizer": str(tokenizer_path),
                        "context_length": 1024,
                        "mask_prompt_tokens": True,
                        "batch_size": batch_size,
                        "dataloader_workers": 0,
                        "pin_memory": False,
                        "shuffle": False,
                    },
                    "training": {
                        "device": "cpu",
                        "mixed_precision": "none",
                        "epochs": epochs,
                        "max_steps": max_steps,
                        "gradient_accumulation_steps": 1,
                        "max_grad_norm": 1.0,
                        "log_every_steps": 1,
                        "save_every_steps": save_every_steps,
                        "eval_every_steps": eval_every_steps,
                        "evaluation_steps": 1,
                        "resume": False,
                    },
                    "checkpoint": {
                        "base_checkpoint": str(base_checkpoint),
                        "output_dir": str(root / "checkpoints"),
                        "keep_last": 2,
                    },
                    "generation": {
                        "prompts": ["Write add."],
                        "max_new_tokens": 2,
                        "temperature": 1.0,
                        "do_sample": False,
                        "repetition_penalty": 1.0,
                    },
                    "outputs": {
                        "metrics_dir": str(root / "metrics"),
                        "samples_dir": str(root / "samples"),
                        "log_file": str(root / "logs" / "phase7.jsonl"),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return load_phase7_config(config_path)
