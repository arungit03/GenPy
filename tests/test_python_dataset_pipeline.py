from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

import genpy_llm.python_dataset_pipeline as pipeline_module
from genpy_llm.python_dataset_pipeline import (
    DatasetPipelineError,
    JsonlValidationError,
    ValidationSettings,
    build_dataset,
    clean_python_dataset,
    collect_python_data,
    count_jsonl_records,
    deduplicate_dataset,
    generate_instruction_pairs,
    iter_jsonl,
    load_pipeline_config,
    run_stage_cli,
    split_dataset,
    validate_dataset,
    validate_instruction_record,
)


def test_end_to_end_pipeline_uses_real_docstrings_and_source(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)

    result = build_dataset(config, resume=False)

    records = [
        *iter_jsonl(result.train_path),
        *iter_jsonl(result.validation_path),
        *iter_jsonl(result.test_path),
    ]
    assert len(records) == 3
    assert {record["instruction"] for record in records} == {
        "Add two integer values.",
        "Store and update a numeric counter.",
        "Return a value from an asynchronous function.",
    }
    assert all("output" in record and "provenance" in record for record in records)
    assert any("def add" in record["output"] for record in records)
    assert count_jsonl_records(result.train_path) == 1
    assert count_jsonl_records(result.validation_path) == 1
    assert count_jsonl_records(result.test_path) == 1


def test_stage_statistics_cover_cleaning_generation_and_deduplication(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)

    collected = collect_python_data(config, resume=False)
    cleaned = clean_python_dataset(config, resume=False)
    generated = generate_instruction_pairs(config, resume=False)
    deduplicated = deduplicate_dataset(config, resume=False)

    assert collected.written_records == 6
    assert cleaned.reason_counts == {"invalid_python_syntax": 1}
    assert generated.reason_counts["missing_docstring"] == 2
    assert generated.written_records == 4
    assert deduplicated.duplicate_records == 1
    assert deduplicated.written_records == 3


def test_require_docstring_true_keeps_existing_rejection_behavior(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path, require_docstring=True)
    collect_python_data(config, resume=False)
    clean_python_dataset(config, resume=False)

    generated = generate_instruction_pairs(config, resume=False)
    records = list(iter_jsonl(config.paths.generated))

    assert config.pair_generation.require_docstring is True
    assert generated.written_records == 4
    assert generated.reason_counts["missing_docstring"] == 2
    assert all(record["provenance"]["instruction_source"] == "docstring" for record in records)


def test_require_docstring_false_infers_grounded_instructions(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path, require_docstring=False)
    collect_python_data(config, resume=False)
    clean_python_dataset(config, resume=False)

    generated = generate_instruction_pairs(config, resume=False)
    records = list(iter_jsonl(config.paths.generated))
    inferred = [
        record
        for record in records
        if record["provenance"]["instruction_source"] == "inferred"
    ]
    transform = next(
        record for record in inferred if record["provenance"]["symbol"] == "transform_value"
    )

    assert config.pair_generation.require_docstring is False
    assert generated.written_records == 6
    assert generated.rejected_records == 0
    assert "missing_docstring" not in generated.reason_counts
    assert "instruction_too_short" not in generated.reason_counts
    assert len(inferred) == 2
    assert "`transform_value`" in transform["instruction"]
    assert "`value: int`" in transform["instruction"]
    assert "`scale: float`" in transform["instruction"]
    assert "return type `float`" in transform["instruction"]
    assert "Scale the value before returning it" in transform["instruction"]
    assert "Return the computed result" in transform["instruction"]


def test_undocumented_class_instruction_uses_fields_constructor_and_methods(
    tmp_path: Path,
) -> None:
    config = _pipeline_config(tmp_path, require_docstring=False)
    (tmp_path / "approved" / "typed_cache.py").write_text(
        "class TypedCache:\n"
        "    # Cache integer values by string key.\n"
        "    values: dict[str, int]\n\n"
        "    def __init__(self, limit: int):\n"
        "        self.limit = limit\n"
        "        self.values = {}\n\n"
        "    def clear(self) -> None:\n"
        "        self.values.clear()\n",
        encoding="utf-8",
    )
    collect_python_data(config, resume=False)
    clean_python_dataset(config, resume=False)

    generate_instruction_pairs(config, resume=False)
    typed_cache = next(
        record
        for record in iter_jsonl(config.paths.generated)
        if record["provenance"]["symbol"] == "TypedCache"
    )

    instruction = typed_cache["instruction"]
    assert typed_cache["provenance"]["instruction_source"] == "inferred"
    assert "`TypedCache` class" in instruction
    assert "`limit: int`" in instruction
    assert "`values: dict[str, int]`" in instruction
    assert "`clear`" in instruction
    assert "Cache integer values by string key" in instruction


def test_resume_reuses_only_current_completed_stages(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    first = build_dataset(config, resume=True)

    second = build_dataset(config, resume=True)

    assert all(not stage.resumed for stage in first.stages)
    assert all(stage.resumed for stage in second.stages)
    config.paths.train.write_text(
        config.paths.train.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    rebuilt_split = split_dataset(config, resume=True)
    assert rebuilt_split.resumed is False
    assert count_jsonl_records(config.paths.train) == 1
    changed_source = tmp_path / "approved" / "a.py"
    changed_source.write_text(
        changed_source.read_text(encoding="utf-8") + "\n# approved source update\n",
        encoding="utf-8",
    )
    third_collect = collect_python_data(config, resume=True)
    assert third_collect.resumed is False


def test_split_keeps_each_source_file_in_only_one_split(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    collect_python_data(config, resume=False)
    clean_python_dataset(config, resume=False)
    generate_instruction_pairs(config, resume=False)
    deduplicate_dataset(config, resume=False)
    validate_dataset(config, resume=False)

    split_dataset(config, resume=False)

    paths_by_split = {
        "train": _source_paths(config.paths.train),
        "validation": _source_paths(config.paths.validation),
        "test": _source_paths(config.paths.test),
    }
    assert paths_by_split["train"].isdisjoint(paths_by_split["validation"])
    assert paths_by_split["train"].isdisjoint(paths_by_split["test"])
    assert paths_by_split["validation"].isdisjoint(paths_by_split["test"])


def test_jsonl_reader_reports_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"valid": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(JsonlValidationError, match=r"invalid\.jsonl:2"):
        list(iter_jsonl(path))


def test_record_validation_rejects_schema_syntax_and_provenance() -> None:
    errors = validate_instruction_record(
        {
            "record_id": "record",
            "instruction": "short",
            "input": "",
            "output": "def broken(:",
        },
        ValidationSettings(),
    )

    assert "instruction_too_short" in errors
    assert "invalid_python_syntax" in errors
    assert "missing_provenance" in errors


def test_strict_validation_writes_rejection_report_and_fails(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    config.paths.deduplicated.parent.mkdir(parents=True, exist_ok=True)
    config.paths.deduplicated.write_text(
        json.dumps(
            {
                "record_id": "bad",
                "instruction": "Broken Python example.",
                "input": "",
                "output": "def broken(:",
                "provenance": {
                    "source_id": "approved",
                    "repository": "Fixture",
                    "source_path": "bad.py",
                    "content_hash": "abc",
                    "approval": "test fixture",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetPipelineError, match="validation rejected"):
        validate_dataset(config, resume=False)

    rejection = next(iter_jsonl(config.paths.rejected))
    assert rejection["errors"] == ["invalid_python_syntax"]
    assert not (config.paths.manifests / "validate.json").exists()


def test_build_cli_accepts_custom_config(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)

    exit_code = run_stage_cli("build", ["--config", str(config.config_path), "--force"])

    assert exit_code == 0
    assert config.paths.train.is_file()
    assert config.paths.validation.is_file()
    assert config.paths.test.is_file()


def test_parallel_ast_processing_is_deterministic(tmp_path: Path) -> None:
    sequential_root = tmp_path / "sequential"
    parallel_root = tmp_path / "parallel"
    sequential_root.mkdir()
    parallel_root.mkdir()
    categories = [
        "code_generation",
        "explanation",
        "bug_fixing",
        "code_completion",
    ]
    sequential = _pipeline_config(
        sequential_root,
        workers=1,
        require_docstring=False,
        enabled_categories=categories,
        maximum_examples_per_file=4,
    )
    parallel = _pipeline_config(
        parallel_root,
        workers=2,
        require_docstring=False,
        enabled_categories=categories,
        maximum_examples_per_file=4,
    )

    for config in (sequential, parallel):
        collect_python_data(config, resume=False)
        clean_python_dataset(config, resume=False)
        generate_instruction_pairs(config, resume=False)

    assert list(iter_jsonl(sequential.paths.generated)) == list(
        iter_jsonl(parallel.paths.generated)
    )


def test_downstream_stages_use_manifests_and_remove_disk_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _pipeline_config(tmp_path)
    collect_python_data(config, resume=False)

    def unexpected_count(_path: Path) -> int:
        raise AssertionError("stage performed an avoidable full JSONL count pass")

    monkeypatch.setattr(pipeline_module, "count_jsonl_records", unexpected_count)
    clean_python_dataset(config, resume=False)
    generate_instruction_pairs(config, resume=False)
    deduplicate_dataset(config, resume=False)
    validate_dataset(config, resume=False)
    split_dataset(config, resume=False)

    index_directory = config.paths.workspace / "indexes"
    assert not list(index_directory.glob("*.sqlite3*"))


def test_phase4_generates_all_grounded_categories_and_statistics(tmp_path: Path) -> None:
    categories = [
        "code_generation",
        "explanation",
        "bug_fixing",
        "refactoring",
        "documentation",
        "unit_testing",
        "optimization",
        "complexity_analysis",
        "type_hints",
        "code_completion",
        "api_usage",
    ]
    config = _pipeline_config(
        tmp_path,
        require_docstring=False,
        enabled_categories=categories,
        templates={
            "explanation": "Describe {kind} `{qualified_name}` using its AST evidence."
        },
    )
    (tmp_path / "approved" / "phase4.py").write_text(
        "import math\n\n"
        "@staticmethod\n"
        "def normalize(value: float) -> int:\n"
        '    """Normalize a numeric value."""\n'
        "    if value == 0:\n"
        "        return 0\n"
        "    return math.floor(value + 1)\n\n"
        "def squares(values: list[int]) -> list[int]:\n"
        "    result = []\n"
        "    for value in values:\n"
        "        result.append(value * value)\n"
        "    return result\n\n"
        "def test_normalize() -> None:\n"
        "    assert normalize(0.0) == 0\n",
        encoding="utf-8",
    )

    result = build_dataset(config, resume=False)
    records = [
        *iter_jsonl(result.train_path),
        *iter_jsonl(result.validation_path),
        *iter_jsonl(result.test_path),
    ]
    by_category: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_category.setdefault(str(record["category"]), []).append(record)

    assert set(by_category) == set(categories)
    assert by_category["explanation"][0]["instruction"].startswith("Describe ")
    assert by_category["bug_fixing"][0]["input"] != by_category["bug_fixing"][0]["output"]
    assert any(
        "Normalize a numeric value" in str(record["output"])
        for record in by_category["documentation"]
    )
    assert any(
        "return [value * value for value in values]" in str(record["output"])
        for record in by_category["optimization"]
    )
    assert any("test_normalize" in str(record["output"]) for record in by_category["unit_testing"])
    assert any("float" not in str(record["input"]) for record in by_category["type_hints"])
    assert all("..." in str(record["input"]) for record in by_category["code_completion"])
    assert any(record["input"] == "import math" for record in by_category["api_usage"])
    rich_provenance = next(
        record["provenance"]
        for record in records
        if record["provenance"]["source_path"] == "phase4.py"
        and record["provenance"]["symbol"] == "normalize"
    )
    assert rich_provenance["imports"] == ["import math"]
    assert rich_provenance["decorators"] == ["staticmethod"]
    assert rich_provenance["type_hints"] == {"return": "int", "value": "float"}

    statistics = json.loads(result.statistics_path.read_text(encoding="utf-8"))
    expected_counts = dict(sorted(Counter(record["category"] for record in records).items()))
    assert statistics["category_counts"] == expected_counts
    combined_split_counts = Counter()
    for split_counts in statistics["split_category_counts"].values():
        combined_split_counts.update(split_counts)
    assert dict(sorted(combined_split_counts.items())) == expected_counts


def _pipeline_config(
    tmp_path: Path,
    *,
    require_docstring: bool = True,
    workers: int = 1,
    enabled_categories: list[str] | None = None,
    templates: dict[str, str] | None = None,
    maximum_examples_per_file: int = 0,
):
    source = tmp_path / "approved"
    source.mkdir()
    (source / "a.py").write_text(
        'def add(left, right):\n    """Add two integer values."""\n    return left + right\n',
        encoding="utf-8",
    )
    (source / "duplicate.py").write_text(
        'def add(left, right):\n    """Add two integer values."""\n    return left + right\n',
        encoding="utf-8",
    )
    (source / "counter.py").write_text(
        'class Counter:\n    """Store and update a numeric counter."""\n\n'
        "    def __init__(self):\n        self.value = 0\n",
        encoding="utf-8",
    )
    (source / "async_value.py").write_text(
        'async def get_value(value):\n'
        '    """Return a value from an asynchronous function."""\n'
        "    return value\n",
        encoding="utf-8",
    )
    (source / "nodoc.py").write_text(
        "def transform_value(value: int, scale: float) -> float:\n"
        "    # Scale the value before returning it.\n"
        "    result = value * scale\n"
        "    return result\n",
        encoding="utf-8",
    )
    (source / "invalid.py").write_text("def invalid(:\n", encoding="utf-8")
    config_path = tmp_path / "dataset_pipeline.yaml"
    payload = {
        "version": 1,
        "project_root": ".",
        "progress": False,
        "paths": {
            "workspace": "work",
            "final_directory": "final",
            "log_file": "logs/pipeline.log",
        },
        "collection": {
            "approved_sources": [
                {
                    "id": "approved",
                    "path": "approved",
                    "repository": "Fixture",
                    "approval": "Explicit unit-test fixture",
                    "license": None,
                    "include": ["**/*.py"],
                    "exclude": [],
                }
            ]
        },
        "cleaning": {
            "minimum_file_bytes": 1,
            "maximum_file_bytes": 10000,
            "require_python_definitions": True,
        },
        "instruction_generation": {
            "require_docstring": require_docstring,
            "minimum_instruction_characters": 8,
            "maximum_output_bytes": 10000,
            "enabled_categories": enabled_categories or ["code_generation"],
            "templates": templates or {},
            "maximum_examples_per_file": maximum_examples_per_file,
        },
        "validation": {
            "require_python_syntax": True,
            "require_provenance": True,
            "fail_on_invalid": True,
        },
        "split": {
            "train_ratio": 0.34,
            "validation_ratio": 0.33,
            "test_ratio": 0.33,
            "seed": 7,
            "group_by": "source_file",
        },
        "performance": {
            "workers": workers,
            "max_pending_tasks_per_worker": 2,
            "sqlite_batch_size": 2,
            "verify_output_hashes_on_resume": False,
        },
        "logging": {"level": "INFO"},
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_pipeline_config(config_path)


def _source_paths(path: Path) -> set[str]:
    return {record["provenance"]["source_path"] for record in iter_jsonl(path)}
