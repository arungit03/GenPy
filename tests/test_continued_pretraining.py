from __future__ import annotations

import json
from pathlib import Path

import yaml

from genpy_llm.code_filtering import CodeFilterSettings
from genpy_llm.continued_pretraining import (
    CorpusTargetConfig,
    assess_pretraining_corpus,
    load_phase61_config,
    write_phase61_report,
)
from genpy_llm.validation_report import FinalValidationConfig, validate_manifest_record


def test_assesses_phase61_corpus_balance_and_targets(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "stored_path": "code/a.py",
                        "content_type": "python_code",
                        "token_count": 60,
                    }
                ),
                json.dumps(
                    {
                        "stored_path": "docs/guide.md",
                        "content_type": "technical_text",
                        "token_count": 40,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    targets = CorpusTargetConfig(
        minimum_tokens=100,
        maximum_tokens=500,
        min_python_code_fraction=0.5,
        max_python_code_fraction=0.7,
        min_technical_text_fraction=0.3,
        max_technical_text_fraction=0.5,
        allow_under_target=False,
    )

    assessment = assess_pretraining_corpus(manifest, targets)

    assert assessment.total_tokens == 100
    assert assessment.content_type_tokens == {"python_code": 60, "technical_text": 40}
    assert assessment.python_code_fraction == 0.6
    assert assessment.technical_text_fraction == 0.4
    assert assessment.ready_for_training is True


def test_phase61_config_and_report_are_written(tmp_path: Path) -> None:
    config_path = tmp_path / "phase6_1.yaml"
    payload = {
        "version": 1,
        "project_root": ".",
        "phase6_1": {
            "corpus": {
                "minimum_tokens": 10,
                "maximum_tokens": 100,
                "allow_under_target": True,
            },
            "stages": {
                "build_final_corpus": False,
                "train": False,
                "evaluate": False,
            },
            "paths": {
                "pretraining_corpus_config": "configs/pretraining.yaml",
                "training_config": "configs/training.yaml",
                "model_config": "configs/model.yaml",
                "optimizer_config": "configs/optimizer.yaml",
                "generation_config": "configs/generation.yaml",
                "report_directory": "reports/phase6_1",
                "log_file": "logs/phase6_1.jsonl",
            },
            "training": {"resume_from": "checkpoints/last_checkpoint.pt"},
            "evaluation": {"commands": [["python", "--version"]]},
        },
        "logging": {"level": "INFO"},
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"stored_path": "docs/guide.md", "content_type": "technical_text", "token_count": 12}
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_phase61_config(config_path)
    assessment = assess_pretraining_corpus(manifest, config.targets)
    json_path, markdown_path = write_phase61_report(assessment, config)

    assert config.evaluation.commands == (("python", "--version"),)
    assert config.training.resume_from == tmp_path / "checkpoints" / "last_checkpoint.pt"
    assert json_path.is_file()
    assert markdown_path.is_file()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["assessment"]["total_tokens"] == 12
    assert "Ready for continued training" in markdown_path.read_text(encoding="utf-8")


def test_final_validation_accepts_technical_text_when_enabled(tmp_path: Path) -> None:
    corpus_root = tmp_path / "raw"
    path = corpus_root / "docs" / "guide.md"
    path.parent.mkdir(parents=True)
    content = (
        "# Training Guide\n\n"
        "Tokenization, validation, sampling, checkpoints, and evaluation reports "
        "are reviewed before continued pretraining.\n"
    )
    path.write_text(content, encoding="utf-8")
    record = {
        "stored_path": "docs/guide.md",
        "source_path": "guide.md",
        "content_sha256": _sha256(path.read_bytes()),
        "content_type": "technical_text",
        "source": {"id": "docs", "type": "local"},
    }
    config = FinalValidationConfig(
        minimum_file_bytes=1,
        maximum_file_bytes=10_000,
        cleaner=CodeFilterSettings(require_known_license=False),
        allow_technical_text=True,
    )

    validated, reason = validate_manifest_record(record, corpus_root=corpus_root, config=config)

    assert reason is None
    assert validated is not None
    assert validated.text == content


def _sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()
