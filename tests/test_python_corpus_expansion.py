from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from genpy_llm.python_corpus_collector import collect_python_corpus
from genpy_llm.python_corpus_expansion import (
    CATEGORY_NAMES,
    expand_python_corpus,
    load_corpus_expansion_config,
    run_corpus_expansion_cli,
)


def test_expansion_imports_classifies_and_indexes_approved_collection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "approved"
    _write_category_fixture(source)
    (source / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    config = _config(tmp_path)

    result = expand_python_corpus(config)

    assert result.collection is not None
    assert result.collection.files_rejected == 1
    assert result.total_repositories == 1
    assert result.total_python_files == 10
    assert result.total_functions == 10
    assert result.total_classes == 2
    assert result.estimated_instruction_pairs == 12
    assert set(result.category_files) == set(CATEGORY_NAMES)
    assert all(result.category_files[category] >= 1 for category in CATEGORY_NAMES)

    with sqlite3.connect(result.index_path) as database:
        file_row = database.execute(
            "SELECT source_id, license, language, content_sha256, "
            "collection_timestamp FROM files WHERE stored_path = ?",
            ("approved_collection/oop.py",),
        ).fetchone()
        categories = {
            row[0] for row in database.execute("SELECT DISTINCT category FROM file_categories")
        }
        symbols = database.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        metadata = dict(database.execute("SELECT key, value FROM metadata"))

    assert file_row is not None
    assert file_row[0:3] == ("approved_collection", "MIT", "Python")
    assert len(file_row[3]) == 64
    assert file_row[4]
    assert categories == set(CATEGORY_NAMES)
    assert symbols == 12
    assert metadata["estimated_instruction_pairs"] == "12"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["total_python_files"] == 10
    assert report["total_repositories"] == 1
    assert report["estimated_instruction_pairs"] == 12


def test_index_revalidates_hash_before_accepting_manifest_record(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    (source / "module.py").write_text("def original():\n    return 1\n", encoding="utf-8")
    config = _config(tmp_path)
    collect_python_corpus(config.collector)
    collected = config.collector.output_directory / "approved_collection" / "module.py"
    collected.write_text("def changed():\n    return 2\n", encoding="utf-8")

    result = expand_python_corpus(config, collect=False)

    assert result.total_python_files == 0
    assert result.rejected_index_records == 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["index_rejection_reasons"] == {"hash_mismatch": 1}


def test_import_restores_tracked_file_from_approved_source_before_indexing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    approved_content = "def approved():\n    return 1\n"
    (source / "module.py").write_text(approved_content, encoding="utf-8")
    config = _config(tmp_path)
    collect_python_corpus(config.collector)
    collected = config.collector.output_directory / "approved_collection" / "module.py"
    collected.write_text("def untracked_change():\n    return 2\n", encoding="utf-8")

    result = expand_python_corpus(config)

    assert result.total_python_files == 1
    assert result.rejected_index_records == 0
    assert collected.read_text(encoding="utf-8") == approved_content


def test_index_rebuild_removes_stale_rows_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    (source / "first.py").write_text("def first():\n    return 1\n", encoding="utf-8")
    config = _config(tmp_path)
    first = expand_python_corpus(config)
    first_report = json.loads(first.report_path.read_text(encoding="utf-8"))
    stored = config.collector.output_directory / "approved_collection" / "first.py"
    stored.unlink()
    config.collector.provenance_manifest.unlink()
    (source / "first.py").unlink()
    (source / "second.py").write_text("class Second:\n    pass\n", encoding="utf-8")

    second = expand_python_corpus(config)

    with sqlite3.connect(second.index_path) as database:
        indexed_paths = [row[0] for row in database.execute("SELECT stored_path FROM files")]
    assert first_report["total_python_files"] == 1
    assert indexed_paths == ["approved_collection/second.py"]
    assert second.total_python_files == 1
    assert second.total_functions == 0
    assert second.total_classes == 1


def test_expansion_cli_supports_index_only_rebuild(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    (source / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    config = _config(tmp_path)
    collect_python_corpus(config.collector)

    exit_code = run_corpus_expansion_cli(
        ["--config", str(config.collector.config_path), "--index-only"]
    )

    assert exit_code == 0
    assert config.index_path.is_file()
    assert config.report_path.is_file()


def _write_category_fixture(root: Path) -> None:
    (root / "algorithms").mkdir(parents=True)
    (root / "structures").mkdir()
    files = {
        "core.py": "def add(left, right):\n    return left + right\n",
        "oop.py": "class Greeter:\n    def greet(self):\n        return 'hello'\n",
        "algorithms/search.py": (
            "def binary_search(values, target):\n"
            "    return target in values\n"
        ),
        "structures/stack.py": (
            "class Stack:\n"
            "    def push(self, value):\n"
            "        self.values.append(value)\n"
        ),
        "file_errors.py": (
            "from pathlib import Path\n"
            "def load(path):\n"
            "    try:\n"
            "        return Path(path).read_text()\n"
            "    except OSError:\n"
            "        raise\n"
        ),
        "stdlib_tools.py": "import json\ndef serialize(value):\n    return json.dumps(value)\n",
        "numpy_tools.py": "import numpy as np\ndef array(values):\n    return np.array(values)\n",
        "pandas_tools.py": "import pandas as pd\ndef frame(data):\n    return pd.DataFrame(data)\n",
        "plot.py": (
            "import matplotlib.pyplot as plt\n"
            "def plot(values):\n"
            "    return plt.plot(values)\n"
        ),
        "test_feature.py": "import pytest\ndef test_value():\n    assert pytest.approx(1) == 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _config(root: Path):
    config_path = root / "dataset_pipeline.yaml"
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
            "sources": [
                {
                    "id": "approved_collection",
                    "type": "local",
                    "location": "approved",
                    "license": "MIT",
                }
            ],
        },
        "corpus_expansion": {
            "index": "raw/corpus_index.sqlite3",
            "report": "raw/corpus_expansion_report.json",
            "collect_before_index": True,
        },
        "logging": {"level": "INFO"},
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_corpus_expansion_config(config_path)
