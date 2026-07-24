from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from genpy_llm.checkpointing import LoadedCheckpoint
from genpy_llm.evaluation import evaluation_metrics
from genpy_llm.evaluation_benchmark import (
    AutomaticCheck,
    EvaluationBenchmarkError,
    PromptEvaluationResult,
    build_evaluation_summary,
    extract_python_code,
    load_evaluation_prompts,
    resolve_evaluation_checkpoint,
    run_automatic_check,
    write_evaluation_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase8_dataset_contains_all_required_prompts() -> None:
    prompts = load_evaluation_prompts(PROJECT_ROOT / "data/evaluation/prompts.json")

    assert len(prompts) == 20
    assert prompts[0].prompt == "Write bubble sort."
    assert any(item.prompt == "Write a FastAPI CRUD API." for item in prompts)
    assert any("def reverse(lst):" in item.prompt for item in prompts)
    assert len({item.id for item in prompts}) == len(prompts)


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            [
                {"id": "same", "prompt": "First"},
                {"id": "same", "prompt": "Second"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationBenchmarkError, match="Duplicate"):
        load_evaluation_prompts(path)


def test_static_check_extracts_fenced_code_without_executing_it() -> None:
    answer = "Here is the solution:\n```python\ndef bubble_sort(items):\n    return items\n```"
    check = AutomaticCheck(
        python_syntax=True,
        required_terms=("def", "return"),
        required_any=(("bubble", "swapped"),),
    )

    result = run_automatic_check(answer, check)

    assert result.passed is True
    assert extract_python_code(answer).startswith("def bubble_sort")


def test_static_check_reports_failures() -> None:
    result = run_automatic_check(
        "def broken(:\n    pass",
        AutomaticCheck(python_syntax=True, required_terms=("return",)),
    )

    assert result.passed is False
    assert "missing required terms" in result.details
    assert "syntax error" in result.details


def test_latest_checkpoint_uses_phase7_last_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "fine_tuned"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "step_00020.pt").touch()
    canonical = checkpoint_dir / "last_checkpoint.pt"
    canonical.touch()
    config = SimpleNamespace(
        project_root=tmp_path,
        checkpoint=SimpleNamespace(
            output_dir=checkpoint_dir,
            last_filename="last_checkpoint.pt",
            step_prefix="step",
        ),
    )

    assert resolve_evaluation_checkpoint("latest", config=config) == canonical.resolve()


def test_writes_json_csv_and_markdown_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    loaded = LoadedCheckpoint(
        epoch=1,
        global_step=42,
        best_metric=1.25,
        training_loss=1.5,
        validation_loss=1.25,
        checkpoint_path=checkpoint,
        extra_state={},
    )
    result = PromptEvaluationResult(
        id="bubble_sort",
        prompt="Write bubble sort.",
        generated_answer="def bubble_sort(values):\n    return values",
        generation_time_seconds=0.5,
        generated_tokens=10,
        tokens_per_second=20.0,
        passed=True,
        check_details="All configured static checks passed.",
    )
    summary = build_evaluation_summary(
        checkpoint_path=checkpoint,
        loaded_checkpoint=loaded,
        device=torch.device("cpu"),
        validation=evaluation_metrics(1.25, 100, 2),
        results=(result,),
    )

    artifacts = write_evaluation_artifacts(summary, tmp_path / "evaluation")

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    with artifacts.csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    report = artifacts.report_path.read_text(encoding="utf-8")
    assert payload["metadata"]["checkpoint_step"] == 42
    assert payload["results"][0]["pass_fail"] == "Pass"
    assert rows[0]["tokens_per_second"] == "20.000000"
    assert "Generated answer" in report
    assert "Validation loss: 1.250000" in report
    assert "Pass/Fail: **Pass**" in report
