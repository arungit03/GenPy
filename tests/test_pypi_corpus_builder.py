from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from genpy_llm.code_tokenizer import CodeTokenizer, train_byte_level_bpe_tokenizer
from genpy_llm.pypi_corpus_builder import (
    DownloadedSdist,
    PyPICorpusError,
    PyPIDeduplicationSettings,
    PyPIDownloadSettings,
    PyPIDuplicateIndex,
    PyPIExtractionSettings,
    PyPIPackage,
    _download_sdist,
    _extract_sdist,
    build_pypi_corpus,
    discover_packages,
    load_pypi_corpus_config,
)
from genpy_llm.streaming_dataset import StreamingGPTDataset


class _FixtureClient:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.metadata_calls: list[tuple[str, str | None]] = []

    def package_metadata(self, name: str, version: str | None = None):
        self.metadata_calls.append((name, version))
        return self.metadata

    def top_packages(self, _limit: int, _minimum: int):
        return [("Demo_Package", 1_000_000)]

    def simple_package_names(self, _limit: int):
        return ["demo-package"]


def test_pypi_pipeline_reuses_collector_tokenizer_reports_and_resume(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    config = _config(tmp_path, archive)
    client = _FixtureClient(_metadata(archive))

    first = build_pypi_corpus(config, api_client=client)

    assert first.packages_discovered == 1
    assert first.packages_downloaded == 1
    assert first.packages_failed == 0
    assert first.collection.files_accepted == 1
    assert first.collection.files_rejected == 4
    assert first.collection.rejection_reasons == {
        "cleaner_low_python_signal": 1,
        "generated_file": 1,
        "invalid_python_syntax": 1,
        "normalized_duplicate": 1,
    }
    assert first.documents == 1
    assert first.token_count > 1
    assert first.shard_count == 1
    assert first.shard_index_path.name == "index.json"
    assert (config.tokens.output_directory / "shard_00000.bin").is_file()
    assert (config.tokens.output_directory / "pypi_document_index.jsonl").is_file()

    records = [
        json.loads(line)
        for line in config.collector.provenance_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    pypi = [record for record in records if record["source"]["type"] == "pypi"]
    assert len(pypi) == 1
    assert pypi[0]["source"]["package"] == "demo-package"
    assert pypi[0]["source"]["version"] == "1.2.3"
    assert pypi[0]["source"]["archive_sha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    assert pypi[0]["license"] == "MIT"
    assert all("tests" not in record["source_path"] for record in pypi)

    document = json.loads(
        (config.tokens.output_directory / "pypi_document_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert document["package"] == "demo-package"
    assert document["token_count"] > 1
    assert document["byte_size"] > 0
    for report in (
        config.paths.pypi_report,
        config.paths.package_statistics,
        config.paths.license_report,
        config.paths.quality_report,
        config.paths.duplicate_report,
        config.paths.token_statistics,
    ):
        assert report.is_file()
    with sqlite3.connect(config.corpus_manager.index_path) as database:
        indexed = database.execute(
            "SELECT package_name, package_version, archive_sha256 FROM files "
            "WHERE source_type='pypi'"
        ).fetchone()
    assert indexed == (
        "demo-package",
        "1.2.3",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )

    tokenizer = CodeTokenizer.from_file(config.tokens.tokenizer_path)
    (config.tokens.output_directory / "shard_index.json").write_text(
        json.dumps({"format": "genpy_uint16_token_shards", "shards": []}),
        encoding="utf-8",
    )
    dataset = StreamingGPTDataset(
        config.tokens.output_directory / "shard_*.bin",
        tokenizer,
        context_length=4,
        stride=4,
        incomplete_window_policy="pad",
    )
    assert list(dataset)

    second = build_pypi_corpus(config, api_client=client)
    assert second.resumed is True
    assert second.collection.files_unchanged == 1


def test_discovery_combines_selectors_and_chooses_only_sdist(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("demo-package==1.2.3\n", encoding="utf-8")
    config = _config(tmp_path, archive)
    config = replace(
        config,
        selection=replace(
            config.selection,
            top_downloaded=True,
            keywords=("demo",),
            categories={"approved": ("Demo.Package",)},
            enabled_categories=("approved",),
            requirements_files=(requirements,),
            manual_packages=("demo_package",),
        ),
    )
    client = _FixtureClient(_metadata(archive))

    packages = discover_packages(config, client)

    assert len(packages) == 1
    assert packages[0].canonical_name == "demo-package"
    assert packages[0].filename.endswith(".tar.gz")
    assert not packages[0].filename.endswith(".whl")


def test_downloader_verifies_checksum_and_resumes(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    package = _package(archive)
    settings = PyPIDownloadSettings(tmp_path / "downloads", 1, 0, 10, 0, True)

    first = _download_sdist(package, settings, None, False, "GenPy-Test")
    second = _download_sdist(
        package, settings, ("downloaded", first.archive_path), False, "GenPy-Test"
    )

    assert first.resumed is False
    assert second.resumed is True
    with pytest.raises(PyPICorpusError, match="Checksum mismatch"):
        _download_sdist(
            replace(package, sha256="0" * 64),
            replace(settings, directory=tmp_path / "bad"),
            None,
            False,
            "GenPy-Test",
        )


def test_extractor_supports_zip_filters_directories_and_rejects_traversal(
    tmp_path: Path,
) -> None:
    settings = PyPIExtractionSettings(
        tmp_path / "extract", 1, ("tests", "docs"), True, 100, 1_000_000
    )
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("root/pkg/main.py", "value = 1\n")
        output.writestr("root/tests/test_main.py", "assert True\n")
        output.writestr("root/readme.txt", "ignore")
    item = DownloadedSdist(_package(archive), archive, False)

    extracted = _extract_sdist(item, tmp_path / "safe", settings)

    assert extracted.python_files == 1
    assert (extracted.extraction_path / "root/pkg/main.py").is_file()
    assert not (extracted.extraction_path / "root/tests/test_main.py").exists()

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as output:
        output.writestr("../escape.py", "value = 1\n")
    with pytest.raises(PyPICorpusError, match="Unsafe archive path"):
        _extract_sdist(
            DownloadedSdist(_package(unsafe), unsafe, False),
            tmp_path / "unsafe-output",
            settings,
        )


@pytest.mark.parametrize(
    ("mode", "suffix"),
    [("w:gz", ".tar.gz"), ("w:bz2", ".tar.bz2"), ("w:xz", ".tar.xz")],
)
def test_extractor_supports_all_tar_sdist_compressions(
    tmp_path: Path, mode: str, suffix: str
) -> None:
    archive = tmp_path / f"package{suffix}"
    content = b"def value():\n    return 1\n"
    with tarfile.open(archive, mode) as output:
        member = tarfile.TarInfo("package/pkg/main.py")
        member.size = len(content)
        output.addfile(member, io.BytesIO(content))
    settings = PyPIExtractionSettings(tmp_path, 1, ("tests",), True, 10, 10_000)

    result = _extract_sdist(
        DownloadedSdist(_package(archive), archive, False),
        tmp_path / f"output-{suffix.removeprefix('.').replace('.', '-')}",
        settings,
    )

    assert result.python_files == 1
    assert (result.extraction_path / "package/pkg/main.py").read_bytes() == content


def test_optional_duplicate_index_detects_normalized_and_near_duplicates(
    tmp_path: Path,
) -> None:
    settings = PyPIDeduplicationSettings(True, True, 4, tmp_path / "duplicates.sqlite3")
    candidate = object()
    with PyPIDuplicateIndex(settings) as index:
        assert index.check(candidate, "def add(a, b):\n    return a + b\n", "a" * 64) is None
        assert (
            index.check(candidate, "def add(a,b): return a+b\n", "b" * 64)
            == "normalized_duplicate"
        )
    near_settings = PyPIDeduplicationSettings(
        False, True, 4, tmp_path / "near-duplicates.sqlite3"
    )
    with PyPIDuplicateIndex(near_settings) as index:
        assert index.check(candidate, "def add(a, b):\n    return a + b\n", "c" * 64) is None
        assert (
            index.check(candidate, "def sum_values(x, y):\n    return x + y\n", "d" * 64)
            == "near_duplicate"
        )


def _config(root: Path, archive: Path):
    corpus = root / "tokenizer_corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {"instruction": "Implement addition.", "output": "def add(a, b): return a + b"}
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
    payload = {
        "version": 1,
        "project_root": ".",
        "progress": False,
        "corpus_collection": {
            "output_directory": "raw",
            "provenance_manifest": "raw/collection_manifest.jsonl",
            "report": "raw/collection_report.json",
            "minimum_file_bytes": 5,
            "maximum_file_bytes": 10000,
            "sources": [],
        },
        "corpus_expansion": {
            "index": "raw/corpus_index.sqlite3",
            "report": "raw/corpus_expansion_report.json",
            "collect_before_index": False,
        },
        "pypi_corpus": {
            "enabled": True,
            "approval": "Approved test source",
            "progress": False,
            "selection": {
                "manual_packages": ["demo-package==1.2.3"],
                "maximum_packages": 10,
            },
            "download": {
                "directory": "downloads",
                "workers": 1,
                "retries": 0,
                "timeout_seconds": 10,
                "retry_backoff_seconds": 0,
                "resume": True,
            },
            "extraction": {
                "directory": "extraction",
                "workers": 1,
                "ignored_directories": ["tests", "docs", "vendor"],
                "ignore_migrations": True,
                "maximum_members": 100,
                "maximum_expanded_bytes": 1000000,
            },
            "deduplication": {
                "normalized": True,
                "near_duplicate": False,
                "index": "state/duplicates.sqlite3",
            },
            "tokenization": {
                "tokenizer": "tokenizer/tokenizer.json",
                "output_directory": "pretraining",
                "shard_prefix": "shard",
                "index": "index.json",
                "statistics": "statistics.json",
                "document_index": "pypi_document_index.jsonl",
                "max_tokens_per_shard": 1000,
                "workers": 1,
            },
            "paths": {
                "checkpoint": "state/checkpoint.sqlite3",
                "report_directory": "reports",
                "log_file": "logs/pypi.jsonl",
            },
        },
        "logging": {"level": "INFO"},
    }
    path = root / "pypi.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_pypi_corpus_config(path)


def _source_archive(root: Path) -> Path:
    archive = root / "demo-package-1.2.3.tar.gz"
    files = {
        "demo-package-1.2.3/pkg/main.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
        "demo-package-1.2.3/pkg/copy.py": "def add(a:int,b:int)->int: return a+b\n",
        "demo-package-1.2.3/pkg/generated.py": "# generated by fixture\nvalue = 1\n",
        "demo-package-1.2.3/pkg/broken.py": "def broken(:\n",
        "demo-package-1.2.3/pkg/low_signal.py": "VALUE = 1\nNAME = 'demo'\n",
        "demo-package-1.2.3/tests/test_main.py": "assert True\n",
        "demo-package-1.2.3/docs/example.py": "value = 1\n",
    }
    with tarfile.open(archive, "w:gz") as output:
        for name, content in files.items():
            encoded = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(encoded)
            member.mtime = 0
            output.addfile(member, io.BytesIO(encoded))
    return archive


def _metadata(archive: Path) -> dict:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return {
        "info": {
            "name": "demo-package",
            "version": "1.2.3",
            "home_page": "https://example.test/demo",
            "project_urls": {"Source": "https://github.com/example/demo"},
            "author": "Example Author",
            "license_expression": "MIT",
            "summary": "Demo Python package",
            "keywords": "demo python",
        },
        "urls": [
            {
                "packagetype": "bdist_wheel",
                "filename": "demo_package-1.2.3-py3-none-any.whl",
                "url": "https://example.test/demo.whl",
                "digests": {"sha256": "0" * 64},
                "yanked": False,
            },
            {
                "packagetype": "sdist",
                "filename": archive.name,
                "url": archive.as_uri(),
                "digests": {"sha256": digest},
                "upload_time_iso_8601": "2026-01-01T00:00:00Z",
                "yanked": False,
            },
        ],
    }


def _package(archive: Path) -> PyPIPackage:
    return PyPIPackage(
        name="demo-package",
        canonical_name="demo-package",
        version="1.2.3",
        release_date="2026-01-01T00:00:00Z",
        homepage="https://example.test/demo",
        project_url="https://pypi.org/project/demo-package/1.2.3/",
        repository_url="https://github.com/example/demo",
        author="Example Author",
        license="MIT",
        summary="Demo",
        keywords="demo",
        download_url=archive.as_uri(),
        filename=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
