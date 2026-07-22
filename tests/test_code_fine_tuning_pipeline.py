from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from genpy_llm.checkpointing import load_checkpoint, save_checkpoint
from genpy_llm.code_training import create_code_model
from genpy_llm.fine_tuning_dataset import (
    FineTuningDataset,
    FineTuningDatasetError,
    FineTuningExample,
    format_fine_tuning_prompt,
    load_fine_tuning_examples,
    split_fine_tuning_examples,
)
from genpy_llm.losses import GPTCrossEntropyLoss
from genpy_llm.optimizers import create_optimizer
from scripts.fine_tune_code_model import FineTuneSettings, evaluate_loss, run_fine_tuning
from tests.test_code_pipeline import _tiny_code_config, _tiny_tokenizer


def test_fine_tuning_loader_supports_jsonl_json_and_txt(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "instructions.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instruction": "Write add.",
                        "input": "two numbers",
                        "output": "def add(a, b):\n    return a + b",
                    }
                ),
                json.dumps({"instruction": "Legacy", "response": "print('legacy')"}),
            ]
        ),
        encoding="utf-8",
    )
    json_path = tmp_path / "instructions.json"
    json_path.write_text(
        json.dumps(
            {
                "examples": [
                    {"instruction": "Write sub.", "output": "def sub(a, b):\n    return a - b"}
                ]
            }
        ),
        encoding="utf-8",
    )
    txt_path = tmp_path / "plain.txt"
    txt_path.write_text("print('first')\n\nprint('second')\n", encoding="utf-8")

    jsonl_examples = load_fine_tuning_examples(jsonl_path)
    json_examples = load_fine_tuning_examples(json_path)
    txt_examples = load_fine_tuning_examples(txt_path)

    assert jsonl_examples[0].input == "two numbers"
    assert jsonl_examples[1].output == "print('legacy')"
    assert json_examples[0].instruction == "Write sub."
    assert [example.output for example in txt_examples] == ["print('first')", "print('second')"]


def test_fine_tuning_dataset_masks_prompt_and_pads(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    dataset = FineTuningDataset(
        [
            FineTuningExample(
                instruction="A",
                input="",
                output="x",
            )
        ],
        tokenizer,
        max_length=64,
        response_only_loss=True,
    )

    sample = dataset[0]

    assert format_fine_tuning_prompt(dataset.examples[0]).startswith("<instruction>")
    assert sample["input_ids"].shape == torch.Size([64])
    assert sample["attention_mask"].dtype == torch.bool
    assert (sample["target_ids"] == tokenizer.pad_token_id).any()
    assert (sample["target_ids"] != tokenizer.pad_token_id).any()
    assert dataset.stats.examples == 1


def test_fine_tuning_split_is_deterministic() -> None:
    examples = [FineTuningExample(output=f"print({index})") for index in range(10)]

    first = split_fine_tuning_examples(examples, validation_split=0.2, seed=123)
    second = split_fine_tuning_examples(examples, validation_split=0.2, seed=123)

    assert first == second
    assert len(first.validation_examples) == 2
    with pytest.raises(FineTuningDatasetError):
        split_fine_tuning_examples(examples, validation_split=1.0, seed=123)


def test_fine_tuning_training_step_checkpoint_resume_and_evaluation(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    config = _tiny_code_config(tmp_path, tokenizer)
    base_checkpoint = tmp_path / "base.pt"
    base_model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=0.001)
    save_checkpoint(base_checkpoint, base_model, base_optimizer, epoch=0, global_step=0)
    dataset_path = tmp_path / "fine_tune.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "Write one.", "output": "def one():\n    return 1"}),
                json.dumps({"instruction": "Write two.", "output": "def two():\n    return 2"}),
            ]
        ),
        encoding="utf-8",
    )
    settings = _tiny_settings(
        tmp_path,
        config_path=tmp_path / "code_small.yaml",
        checkpoint=base_checkpoint,
        dataset=dataset_path,
        epochs=1,
        max_steps=1,
    )

    first = run_fine_tuning(settings)
    resumed = run_fine_tuning(
        replace(settings, checkpoint=None, resume="latest", epochs=2, max_steps=2)
    )

    assert first.global_step == 1
    assert resumed.global_step == 2
    assert (settings.output_dir / "latest.pt").exists()
    assert (settings.output_dir / "best.pt").exists()
    assert (settings.output_dir / "epoch_1.pt").exists()
    assert (settings.evaluation_dir / "step_00000001_generation.txt").exists()
    assert (settings.evaluation_dir / "loss_curve.png").read_bytes().startswith(b"\x89PNG")
    loaded_model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    loaded_optimizer = create_optimizer(
        loaded_model,
        replace(config.optimizer, learning_rate=0.001, weight_decay=0.0),
    )
    loaded = load_checkpoint(
        settings.output_dir / "latest.pt",
        loaded_model,
        optimizer=loaded_optimizer,
        restore_rng=False,
    )
    assert loaded.global_step == 2


def test_fine_tuning_evaluate_loss_smoke(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    config = _tiny_code_config(tmp_path, tokenizer)
    model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    dataset = FineTuningDataset(
        [FineTuningExample(instruction="Write one.", output="def one():\n    return 1")],
        tokenizer,
        max_length=8,
    )
    loader = DataLoader(dataset, batch_size=1)
    loss = evaluate_loss(
        model=model,
        loader=loader,
        loss_fn=GPTCrossEntropyLoss(tokenizer.pad_token_id, True, 0.0),
        device=torch.device("cpu"),
        mixed_precision="none",
        max_batches=1,
    )

    assert loss > 0


def _tiny_settings(
    tmp_path: Path,
    *,
    config_path: Path,
    checkpoint: Path,
    dataset: Path,
    epochs: int,
    max_steps: int,
) -> FineTuneSettings:
    return FineTuneSettings(
        base_config=config_path,
        checkpoint=checkpoint,
        dataset=dataset,
        output_dir=tmp_path / "checkpoints" / "fine_tune",
        log_dir=tmp_path / "logs" / "fine_tune",
        evaluation_dir=tmp_path / "evaluation" / "fine_tune",
        epochs=epochs,
        batch_size=1,
        learning_rate=0.001,
        weight_decay=0.0,
        warmup_ratio=0.0,
        max_length=8,
        gradient_accumulation=1,
        save_every=1,
        eval_every=1,
        seed=123,
        device="cpu",
        resume=None,
        validation_split=0.5,
        response_only_loss=True,
        shuffle=False,
        mixed_precision="none",
        max_grad_norm=1.0,
        early_stopping_patience=0,
        keep_last=2,
        generation_max_new_tokens=1,
        max_steps=max_steps,
        max_train_batches=1,
        eval_batches=1,
    )
