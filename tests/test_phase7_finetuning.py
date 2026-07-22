from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from genpy_llm.checkpointing import save_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer, train_byte_level_bpe_tokenizer
from genpy_llm.conversation_formatter import ConversationTemplate
from genpy_llm.fine_tuning import Phase7Trainer, _sft_loss, load_phase7_config
from genpy_llm.gpt import GPTModel
from genpy_llm.instruction_dataset import InstructionDataset, load_instruction_records


def test_conversation_formatter_default_template() -> None:
    template = ConversationTemplate()

    text = template.format_conversation("Write add.", "", "def add(a, b): return a + b")

    assert "<|system|>" in text
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "You are GenPy, a Python coding assistant." in text


def test_instruction_dataset_and_prompt_loss_masking(tmp_path: Path) -> None:
    tokenizer_path = _tokenizer(tmp_path)
    tokenizer = CodeTokenizer.from_file(tokenizer_path)
    dataset_path = _dataset(tmp_path)
    records = load_instruction_records(dataset_path)

    dataset = InstructionDataset(
        records,
        tokenizer=tokenizer,
        template=ConversationTemplate(),
        context_length=128,
        mask_prompt_tokens=True,
    )
    unmasked = InstructionDataset(
        records,
        tokenizer=tokenizer,
        template=ConversationTemplate(),
        context_length=128,
        mask_prompt_tokens=False,
    )

    assert len(dataset) == 2
    assert dataset[0]["input_ids"].shape == (128,)
    assert dataset[0]["attention_mask"].sum() > 0
    assert int((dataset[0]["target_ids"] == -100).sum()) > 0
    assert int((unmasked[0]["target_ids"] == -100).sum()) < int(
        (dataset[0]["target_ids"] == -100).sum()
    )


def test_sft_loss_ignores_masked_targets() -> None:
    logits = torch.randn(1, 3, 10)
    targets = torch.tensor([[-100, 2, 3]], dtype=torch.long)

    loss = _sft_loss(logits, targets)

    assert loss.item() > 0


def test_phase7_trainer_checkpoint_resume_evaluation_and_generation(tmp_path: Path) -> None:
    config = _phase7_config(tmp_path)
    result = Phase7Trainer(config).train()

    assert result.global_step == 1
    assert result.last_checkpoint is not None
    assert result.last_checkpoint.is_file()
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.is_file()
    assert result.metrics_path.is_file()
    assert result.latest_sample_path is not None
    assert result.latest_sample_path.is_file()

    resumed = Phase7Trainer(replace(config, training=replace(config.training, resume=True))).train()

    assert resumed.global_step >= result.global_step


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


def _dataset(root: Path) -> Path:
    path = root / "train.jsonl"
    rows = [
        {
            "instruction": "Write add.",
            "input": "",
            "output": "def add(a, b):\n    return a + b",
        },
        {
            "instruction": "Explain a loop.",
            "input": "for item in items: print(item)",
            "output": "It iterates through each item and prints it.",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _phase7_config(root: Path):
    tokenizer_path = _tokenizer(root)
    train_path = _dataset(root)
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
                        "mask_prompt_tokens": True,
                        "batch_size": 1,
                        "dataloader_workers": 0,
                        "pin_memory": False,
                        "shuffle": False,
                    },
                    "training": {
                        "device": "cpu",
                        "mixed_precision": "none",
                        "epochs": 1,
                        "max_steps": 1,
                        "gradient_accumulation_steps": 1,
                        "max_grad_norm": 1.0,
                        "log_every_steps": 1,
                        "save_every_steps": 1,
                        "eval_every_steps": 1,
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
