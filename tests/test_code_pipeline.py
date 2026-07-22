from __future__ import annotations

import gzip
import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from genpy_llm.code_data_download import (
    convert_instruction_records,
    stream_code_records,
)
from genpy_llm.code_filtering import (
    CodeFilterSettings,
    content_hash,
    filter_code_record,
    normalize_license,
    normalize_python_source,
    stable_split,
)
from genpy_llm.code_fine_tuning import (
    CodeInstructionDataset,
    CodeInstructionExample,
    format_instruction_prompt,
)
from genpy_llm.code_generation import extract_code_only, generate_code_text
from genpy_llm.code_sharding import CompressedShardWriter, read_gzip_jsonl
from genpy_llm.code_tokenizer import (
    CodeTokenizer,
    ensure_code_tokenizer,
    train_byte_level_bpe_tokenizer,
)
from genpy_llm.code_training import (
    create_code_dataloader,
    create_code_model,
    load_code_config,
    train_code_steps,
    validate_code_training_artifacts,
)
from genpy_llm.streaming_dataset import StreamingGPTDataset

CODE_SAMPLE = (
    "# தமிழ் comment\n"
    "import math\n\n"
    "def is_prime(number):\n"
    "    if number < 2:\n"
    "        return False\n"
    "    for divisor in range(2, int(math.sqrt(number)) + 1):\n"
    "        if number % divisor == 0:\n"
    "            return False\n"
    "    return True\n"
)


def test_code_filtering_preserves_indentation_newlines_and_unicode() -> None:
    normalized = normalize_python_source(CODE_SAMPLE.replace("\n", "\r\n") + "\x00")

    assert "\r" not in normalized
    assert "    if number" in normalized
    assert "தமிழ்" in normalized
    assert normalized.endswith("True\n")


def test_license_and_python_quality_filtering() -> None:
    record = {"content": CODE_SAMPLE, "path": "src/example.py", "license": "Apache License 2.0"}

    result = filter_code_record(record, settings=CodeFilterSettings(minimum_file_bytes=10))

    assert normalize_license("bsd 3 clause") == "BSD-3-Clause"
    assert result.accepted is True
    assert result.record is not None
    assert result.record.license == "Apache-2.0"
    rejected = filter_code_record(
        {"content": CODE_SAMPLE, "path": "vendor/generated.py", "license": "MIT"},
        settings=CodeFilterSettings(minimum_file_bytes=10),
    )
    assert rejected.accepted is False


def test_hash_dedup_and_stable_split() -> None:
    first = content_hash(CODE_SAMPLE)
    second = content_hash(CODE_SAMPLE.replace("\n", "\r\n"))

    assert first == second
    assert stable_split(first, validation_percent=2) in {"train", "validation"}


def test_compressed_shard_writer_and_atomic_output(tmp_path: Path) -> None:
    writer = CompressedShardWriter(tmp_path, "train", shard_mb=1)
    writer.write({"text": CODE_SAMPLE, "content_hash": content_hash(CODE_SAMPLE)})
    stats = writer.close()

    assert stats.records == 1
    assert stats.shard_paths[0].exists()
    assert not list(tmp_path.glob("*.partial"))
    assert read_gzip_jsonl(stats.shard_paths[0])[0]["text"] == CODE_SAMPLE


def test_download_stream_resume_manifest_validation(tmp_path: Path) -> None:
    records = [
        {"content": CODE_SAMPLE, "path": "a.py", "license": "MIT"},
        {"content": CODE_SAMPLE, "path": "copy.py", "license": "MIT"},
    ]

    summary = stream_code_records(
        records,
        target_bytes=10_000,
        shard_mb=1,
        train_output=tmp_path / "train",
        validation_output=tmp_path / "validation",
        hash_path=tmp_path / "accepted_hashes.txt",
        manifest_path=tmp_path / "manifest.json",
        settings=CodeFilterSettings(minimum_file_bytes=10),
        validation_percent=2,
        seed=42,
        resume=False,
    )

    assert summary.accepted_files == 1
    assert summary.duplicate_files == 1
    assert (tmp_path / "manifest.json").exists()


def test_instruction_conversion_deduplicates(tmp_path: Path) -> None:
    summary = convert_instruction_records(
        [
            {"instruction": "Reverse", "input": "", "output": "print('x')"},
            {"instruction": "Reverse", "input": "", "output": "print('x')"},
            {"instruction": "", "output": "print('skip')"},
        ],
        tmp_path / "instructions.jsonl",
    )

    assert summary.written_records == 1
    assert summary.duplicate_records == 1
    assert summary.skipped_records == 1


def test_tokenizer_training_special_ids_and_round_trip(tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "train" / "python_train_00000.jsonl.gz", [CODE_SAMPLE])
    metadata = train_byte_level_bpe_tokenizer(
        [shard],
        output_path=tmp_path / "code_tokenizer.json",
        metadata_path=tmp_path / "tokenizer_metadata.json",
        vocab_size=128,
        min_frequency=1,
        max_training_bytes=10_000,
    )
    tokenizer = CodeTokenizer.from_file(tmp_path / "code_tokenizer.json")

    assert tokenizer.pad_token_id == 0
    assert tokenizer.unknown_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert metadata.actual_vocab_size == tokenizer.vocab_size
    decoded = tokenizer.decode(tokenizer.encode(CODE_SAMPLE))
    assert "    if number" in decoded
    assert "தமிழ்" in decoded


def test_missing_tokenizer_builds_from_jsonl_with_required_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "data" / "fine_tuning" / "train.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text(
        json.dumps({"instruction": "Add one", "response": "def add_one(x): return x + 1"})
        + "\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "data" / "tokenizer" / "code_tokenizer.json"
    metadata_path = tokenizer_path.with_name("tokenizer_metadata.json")

    tokenizer = ensure_code_tokenizer(
        tokenizer_path=tokenizer_path,
        metadata_path=metadata_path,
        project_root=tmp_path,
        vocab_size=512,
        preferred_corpus_paths=(corpus,),
        train_pattern="data/code_shards/train/*.jsonl.gz",
        min_frequency=1,
    )

    output = capsys.readouterr().out
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample = "def add_one(x):\n    return x + 1\n"
    assert "Tokenizer not found. Building tokenizer..." in output
    assert "✓ Tokenizer built successfully" in output
    assert tokenizer_path.is_file()
    assert tokenizer.vocab_size == 512
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
    assert metadata["vocab_size"] == 512
    assert metadata["special_tokens"] == [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<mask>",
        "<instruction>",
        "<input>",
        "<output>",
    ]
    assert metadata["tokenizer_type"] == "byte_bpe"
    assert metadata["training_corpus"] == ["data/fine_tuning/train.jsonl"]
    assert metadata["creation_timestamp"]


def test_streaming_dataset_shapes_shift_and_padding(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    _write_shard(
        tmp_path / "train" / "python_train_00000.jsonl.gz",
        ["def add(a, b):\n    return a + b\n"],
    )
    dataset = StreamingGPTDataset(
        str(tmp_path / "train" / "*.jsonl.gz"),
        tokenizer,
        context_length=16,
        stride=16,
        incomplete_window_policy="pad",
    )
    sample = next(iter(dataset))

    assert sample["input_ids"].shape == torch.Size([16])
    assert sample["target_ids"].shape == torch.Size([16])
    assert sample["attention_mask"].dtype == torch.bool
    assert sample["target_ids"][0] == sample["input_ids"][1]


def test_worker_shard_partitioning(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    first = _write_shard(
        tmp_path / "train" / "python_train_00000.jsonl.gz",
        ["def a():\n    return 1\n"],
    )
    second = _write_shard(
        tmp_path / "train" / "python_train_00001.jsonl.gz",
        ["def b():\n    return 2\n"],
    )
    dataset = StreamingGPTDataset(
        str(tmp_path / "train" / "*.jsonl.gz"),
        tokenizer,
        context_length=8,
        stride=8,
        incomplete_window_policy="pad",
    )

    assert dataset.shard_paths == (first, second)


def test_gpt_logits_use_actual_tokenizer_vocabulary(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    config = _tiny_code_config(tmp_path, tokenizer)
    model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)

    logits = model(torch.zeros((1, 8), dtype=torch.long))

    assert logits.shape == (1, 8, tokenizer.vocab_size)


def test_base_training_two_step_smoke(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    config = _tiny_code_config(tmp_path, tokenizer)
    model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    train_loader = create_code_dataloader(config, tokenizer, split="train", batch_size=1)
    validation_loader = create_code_dataloader(config, tokenizer, split="validation", batch_size=1)

    result = train_code_steps(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=torch.device("cpu"),
        max_steps=2,
        max_batches=2,
    )

    assert result.global_step >= 1
    assert result.latest_checkpoint is not None and result.latest_checkpoint.exists()


def test_code_training_debug_logging_marks_startup_and_steps(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    config = _tiny_code_config(tmp_path, tokenizer)
    artifacts = validate_code_training_artifacts(config)
    model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    train_loader = create_code_dataloader(config, tokenizer, split="train", batch_size=1)
    validation_loader = create_code_dataloader(config, tokenizer, split="validation", batch_size=1)
    logger = logging.getLogger("genpy_llm.test.code_training")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = train_code_steps(
            model=model,
            tokenizer=tokenizer,
            config=config,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=torch.device("cpu"),
            max_steps=1,
            max_batches=1,
            validation_batches=1,
            logger=logger,
        )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert artifacts.train_shards
    assert result.global_step == 1
    assert "Entered train_code_steps" in messages
    assert "optimizer created" in messages
    assert "scheduler created" in messages
    assert "entering training loop" in messages
    assert "training step start" in messages
    assert "validation step complete" in messages
    assert "checkpoint save complete" in messages


def test_code_training_resume_and_generation_snapshot(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    config = _tiny_code_config(tmp_path, tokenizer)
    config = replace(
        config,
        generation=replace(config.generation, max_new_tokens=1, do_sample=False),
    )
    first_model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    first_result = train_code_steps(
        model=first_model,
        tokenizer=tokenizer,
        config=config,
        train_loader=create_code_dataloader(config, tokenizer, split="train", batch_size=1),
        validation_loader=create_code_dataloader(
            config,
            tokenizer,
            split="validation",
            batch_size=1,
        ),
        device=torch.device("cpu"),
        max_steps=1,
        max_batches=1,
        validation_batches=1,
        evaluation_dir=tmp_path / "evaluation",
        generation_prompts=("def tiny():",),
    )
    resumed_model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)
    second_result = train_code_steps(
        model=resumed_model,
        tokenizer=tokenizer,
        config=config,
        train_loader=create_code_dataloader(config, tokenizer, split="train", batch_size=1),
        validation_loader=create_code_dataloader(
            config,
            tokenizer,
            split="validation",
            batch_size=1,
        ),
        device=torch.device("cpu"),
        max_steps=2,
        max_batches=1,
        validation_batches=1,
        checkpoint_path=first_result.latest_checkpoint,
        evaluation_dir=tmp_path / "evaluation",
        generation_prompts=("def tiny():",),
    )

    assert first_result.global_step == 1
    assert second_result.global_step == 2
    assert second_result.tokens_processed > first_result.tokens_processed
    assert (tmp_path / "evaluation" / "step_0001_generation.txt").exists()
    assert (tmp_path / "evaluation" / "step_0002_generation.txt").exists()
    assert (tmp_path / "evaluation" / "training_metrics.csv").exists()
    assert (tmp_path / "evaluation" / "loss_curve.png").exists()
    assert "gradient_norm" in (tmp_path / "evaluation" / "training_metrics.csv").read_text(
        encoding="utf-8"
    )


def test_instruction_formatting_and_response_only_masking(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    dataset = CodeInstructionDataset(
        [CodeInstructionExample("Reverse text", "", "print(text[::-1])")],
        tokenizer,
        max_sequence_length=32,
        response_only_loss=True,
        ignore_index=tokenizer.pad_token_id,
    )

    sample = dataset[0]

    assert format_instruction_prompt("Do it").startswith("<instruction>")
    assert (sample["target_ids"] == tokenizer.pad_token_id).any()
    assert sample["input_ids"].shape == torch.Size([32])


def test_code_generation_smoke(tmp_path: Path) -> None:
    tokenizer = _tiny_tokenizer(tmp_path)
    config = _tiny_code_config(tmp_path, tokenizer)
    model = create_code_model(config, tokenizer.vocab_size, tokenizer.pad_token_id)

    result = generate_code_text(
        model=model,
        tokenizer=tokenizer,
        prompt="def",
        device=torch.device("cpu"),
        max_new_tokens=2,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        do_sample=False,
        stop_on_eos=True,
        instruction_mode=False,
        code_only=False,
        context_length=8,
    )

    assert len(result.generated_token_ids) >= 1
    assert isinstance(extract_code_only("### Response:\nprint(1)"), str)


def test_cuda_skip_behavior() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    assert torch.device("cuda").type == "cuda"


def test_gradio_constructs_with_code_controls() -> None:
    pytest.importorskip("gradio")
    from genpy_llm.web_interface import GenPyWebInterface

    assert GenPyWebInterface is not None


def _write_shard(path: Path, texts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as file:
        for text in texts:
            json.dump({"text": text, "content_hash": content_hash(text)}, file, ensure_ascii=False)
            file.write("\n")
    return path


def _tiny_tokenizer(tmp_path: Path) -> CodeTokenizer:
    tokenizer_path = tmp_path / "code_tokenizer.json"
    if tokenizer_path.exists():
        return CodeTokenizer.from_file(tokenizer_path)
    shard = _write_shard(
        tmp_path / "train" / "python_train_00000.jsonl.gz",
        [CODE_SAMPLE, "def add(a, b):\n    return a + b\n"],
    )
    _write_shard(tmp_path / "validation" / "python_validation_00000.jsonl.gz", [CODE_SAMPLE])
    train_byte_level_bpe_tokenizer(
        [shard],
        output_path=tokenizer_path,
        metadata_path=tmp_path / "tokenizer_metadata.json",
        vocab_size=128,
        min_frequency=1,
        max_training_bytes=20_000,
    )
    return CodeTokenizer.from_file(tokenizer_path)


def _tiny_code_config(tmp_path: Path, tokenizer: CodeTokenizer):
    config_data = yaml.safe_load(Path("configs/code_small.yaml").read_text(encoding="utf-8"))
    config_data["tokenizer"]["path"] = str(tmp_path / "code_tokenizer.json")
    config_data["tokenizer"]["metadata_path"] = str(tmp_path / "tokenizer_metadata.json")
    config_data["tokenizer"]["vocab_size"] = tokenizer.vocab_size
    config_data["streaming_dataset"].update(
        {
            "train_pattern": str(tmp_path / "train" / "*.jsonl.gz"),
            "validation_pattern": str(tmp_path / "validation" / "*.jsonl.gz"),
            "context_length": 8,
            "stride": 8,
            "shuffle_shards": False,
            "shuffle_buffer_records": 0,
            "incomplete_window_policy": "pad",
            "pin_memory": False,
        }
    )
    config_data["model"].update(
        {
            "embedding_dim": 32,
            "num_heads": 4,
            "num_layers": 2,
            "context_length": 8,
            "dropout": 0.0,
        }
    )
    config_data["training"].update(
        {
            "max_steps": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "none",
            "save_every_steps": 100,
            "validate_every_steps": 100,
            "validation_steps": 1,
        }
    )
    config_data["scheduler"].update({"warmup_steps": 1})
    config_data["checkpoint"]["directory"] = str(tmp_path / "checkpoints" / "code_base")
    config_path = tmp_path / "code_small.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return load_code_config(config_path)
