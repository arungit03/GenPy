from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml

from genpy_llm.python_corpus_population import (
    CorpusPopulationError,
    load_corpus_population_config,
    populate_python_corpus,
    run_corpus_population_cli,
    search_python_corpus,
)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_population_imports_supported_sources_reports_and_searches(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    algorithm = "def binary_search(values, target):\n    return target in values\n"
    (local / "algorithm.py").write_text(algorithm, encoding="utf-8")
    (local / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "models.py").write_text(
        "class Service:\n    def run(self):\n        return True\n",
        encoding="utf-8",
    )
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "population@example.invalid")
    _git(repository, "config", "user.name", "Population Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "approved fixture")

    archive = tmp_path / "approved.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "numeric/numpy_tool.py",
            "import numpy as np\ndef array(values):\n    return np.array(values)\n",
        )
        output.writestr("duplicate.py", algorithm)
    config = _config(tmp_path)

    result = populate_python_corpus(config)

    assert result.python_files_imported == 3
    assert result.total_python_files == 3
    assert result.functions_discovered == 3
    assert result.classes_discovered == 1
    assert result.duplicate_files == 1
    assert result.estimated_instruction_pairs == 4
    assert result.categories["Algorithms"] == 1
    assert result.categories["OOP"] == 1
    assert result.categories["NumPy"] == 1

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["python_files_imported"] == 3
    assert report["python_files_rejected"] == 2
    assert report["duplicate_files"] == 1
    assert report["total_repositories"] == 3
    assert report["python_files_scanned"] == 5
    assert report["accepted_files"] == 3
    assert report["rejected_files"] == 2
    assert report["rejection_reasons"] == {
        "duplicate_content": 1,
        "invalid_python_syntax": 1,
    }
    assert report["category_distribution"] == report["categories"]
    assert report["license_metadata"] == {
        "declared_files": 3,
        "unspecified_files": 0,
        "distribution": {"Apache-2.0": 1, "BSD": 1, "MIT": 1},
    }
    assert {source["id"] for source in report["approved_sources"]} == {
        "local_source",
        "local_repository",
        "zip_source",
    }

    symbol_results = search_python_corpus(config, query="binary_search")
    oop_results = search_python_corpus(config, category="OOP")

    assert len(symbol_results) == 1
    assert symbol_results[0].source_id == "local_source"
    assert symbol_results[0].matching_symbols == ("binary_search",)
    assert len(oop_results) == 1
    assert oop_results[0].source_type == "git"
    assert oop_results[0].license == "Apache-2.0"

    with sqlite3.connect(result.index_path) as database:
        indexed_sources = {
            row[0] for row in database.execute("SELECT DISTINCT source_id FROM files")
        }
    assert indexed_sources == {"local_source", "local_repository", "zip_source"}

    second = populate_python_corpus(config)
    assert second.python_files_imported == 0
    assert second.python_files_unchanged == 3
    assert second.duplicate_files == 1


def test_population_rejects_individual_file_source_type(tmp_path: Path) -> None:
    (tmp_path / "single.py").write_text("value = 1\n", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        [{"id": "single", "type": "file", "location": "single.py", "license": None}],
    )

    with pytest.raises(CorpusPopulationError, match="unsupported type"):
        load_corpus_population_config(config_path)


def test_population_requires_explicit_source_approval(tmp_path: Path) -> None:
    source = tmp_path / "local"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        [{"id": "local_source", "type": "local", "location": "local", "license": "MIT"}],
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del payload["corpus_collection"]["sources"][0]["approval"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(CorpusPopulationError, match="approval statement"):
        load_corpus_population_config(config_path)


def test_population_cli_searches_existing_index(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "local"
    source.mkdir()
    (source / "stack.py").write_text(
        "class Stack:\n    pass\n",
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        [{"id": "local_source", "type": "local", "location": "local", "license": "MIT"}],
    )
    config = load_corpus_population_config(config_path)
    populate_python_corpus(config)

    exit_code = run_corpus_population_cli(
        ["--config", str(config_path), "--search", "Stack", "--limit", "5"]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"stored_path": "local_source/stack.py"' in output
    assert "Search results: 1" in output


def _config(root: Path):
    config_path = _write_config(
        root,
        [
            {"id": "local_source", "type": "local", "location": "local", "license": "MIT"},
            {
                "id": "local_repository",
                "type": "git",
                "location": "repository",
                "license": "Apache-2.0",
            },
            {"id": "zip_source", "type": "zip", "location": "approved.zip", "license": "BSD"},
        ],
    )
    return load_corpus_population_config(config_path)


def _write_config(root: Path, sources: list[dict[str, object]]) -> Path:
    config_path = root / "dataset_pipeline.yaml"
    for source in sources:
        source.setdefault("approval", "Explicitly approved unit-test source")
    payload = {
        "version": 1,
        "project_root": ".",
        "corpus_collection": {
            "output_directory": "raw",
            "provenance_manifest": "raw/collection_manifest.jsonl",
            "report": "raw/collection_report.json",
            "log_file": "logs/collector.log",
            "minimum_file_bytes": 1,
            "maximum_file_bytes": 10000,
            "sources": sources,
        },
        "corpus_expansion": {
            "index": "raw/corpus_index.sqlite3",
            "report": "raw/corpus_expansion_report.json",
        },
        "corpus_population": {"report": "raw/corpus_population_report.json"},
        "logging": {"level": "INFO"},
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
