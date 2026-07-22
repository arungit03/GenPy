# Phase 5.5B: PyPI Corpus Builder

The PyPI builder turns approved source distributions into validated GenPy
pretraining shards. It is an adapter around the existing corpus collector,
Corpus Manager, SHA-256 index, provenance manifest, ByteLevel BPE tokenizer,
binary shard writer, progress bar, logging, and statistics code. It does not
download wheels and it never trains or changes the tokenizer.

## Installation and approval

Install the project and development tools:

```bash
python -m pip install -e '.[dev]'
```

Review `configs/pypi.yaml`, add only approved package selectors, review the
license policy, and then set:

```yaml
pypi_corpus:
  enabled: true
```

The checked-in configuration is deliberately disabled. Package popularity is
a discovery signal, not evidence that a package's license is suitable for a
particular training use.

## Package selection

Selectors can be combined. Canonical package names are deduplicated before any
metadata or archive is downloaded.

```yaml
pypi_corpus:
  selection:
    top_downloaded: true
    top_downloaded_limit: 1000
    minimum_downloads: 10000
    keywords: [scientific, parser]
    keyword_scan_limit: 2000
    categories:
      scientific: [numpy, scipy]
      tooling: [black, pytest]
    enabled_categories: [scientific]
    requirements_files: [approved-requirements.txt]
    manual_packages: [attrs, pydantic==2.11.7]
    maximum_packages: 10000
    ignored_licenses: []
```

Requirements files accept package names and exact `==` pins. Options, nested
requirements, VCS URLs, and direct URLs are skipped because they are not PyPI
package discovery records. Keyword mode uses the PyPI Simple API as a bounded
candidate list and matches package name, summary, and keyword metadata. The
top-download list URL is configurable because PyPI itself does not expose a
ranked-download endpoint.

For every selected release, the builder records package name, version, release
date, homepage, PyPI project URL, source repository URL when declared, author,
license, summary, keywords, archive URL, archive filename, and PyPI SHA-256.
Only non-yanked `sdist` entries ending in `.tar.gz`, `.tar.bz2`, `.tar.xz`, or
`.zip` are eligible. Wheel and binary-distribution entries are ignored.

## Running and resuming

```bash
python scripts/build_pypi_corpus.py
python scripts/build_pypi_corpus.py --max-packages 100
python scripts/build_pypi_corpus.py --force
```

The SQLite checkpoint at `data/pypi/checkpoint.sqlite3` stores resolved package
metadata, download state, and completed shard fingerprints. Archives are
downloaded in parallel to `.partial` files, resumed with HTTP Range requests
when supported, checked against the published SHA-256, and atomically renamed.
Running the same configuration again reuses valid archives and token shards.
`--force` refreshes downloads and rebuilds the shard generation.

Extraction runs in isolated per-run directories under `data/pypi/extraction/`.
Archive paths are checked for traversal, symlinks are not extracted, member and
expanded-size limits guard against archive bombs, and only `.py` members are
written. Temporary extraction trees are removed even if processing fails.

## Validation, deduplication, and provenance

Extracted files are submitted as `pypi` sources to the existing collector. That
single validation path applies:

- minimum and maximum byte size;
- UTF-8 decoding;
- Python AST syntax parsing;
- generated-file name and content markers;
- ignored virtual environment, build, cache, vendor, test, fixture, docs, demo,
  example, benchmark, and optional migration directories;
- exact SHA-256 deduplication across the full raw corpus.

The existing code cleaner is enabled by default for binary-like, minified,
low-Python-signal, generated-content, and path checks. Its known-license mode
and accepted-license list are configurable under `cleaner`; the separate
discovery `ignored_licenses` policy still applies before download.

AST-normalized and token-SimHash near-duplicate detection are optional in
`deduplication`. Exact SHA-256 checking is always active. Accepted files retain
their package/release/archive fields in `data/raw/collection_manifest.jsonl`.
The existing Corpus Manager is then rebuilt at `data/raw/corpus_index.sqlite3`,
where package fields are searchable alongside categories and function/class
symbols. This keeps `data/raw/` fully rebuildable by the existing dataset tools.

## Tokenization and binary format

The builder loads `data/tokenizer/tokenizer.json`, verifies its identity, and
encodes accepted PyPI documents in deterministic manifest order. It never calls
the tokenizer trainer. Multiprocessing is bounded so a 500,000-file corpus does
not accumulate all source text or futures in memory.

Outputs are:

```text
data/pretraining/
  shard_00000.bin
  shard_00001.bin
  ...
  index.json
  statistics.json
  pypi_document_index.jsonl
```

Shards contain little-endian uint16 token IDs. Documents end in `<eos>` and do
not cross shard boundaries. `index.json` pins the tokenizer SHA-256, vocabulary
size, EOS ID, per-shard hashes, counts, and build fingerprint. The existing
`StreamingGPTDataset` accepts either `index.json` or its earlier
`shard_index.json` name, so the files can feed the future GPT pretraining loader
directly:

```yaml
streaming_dataset:
  train_pattern: data/pretraining/shard_*.bin
```

## Reports and logs

The requested reports are written under `reports/`:

- `pypi_report.json` — aggregate packages, files, tokens, shards, and paths;
- `package_statistics.json` — package metadata and stage errors;
- `license_report.json` — package distribution by recorded license;
- `quality_report.json` — validation acceptance and rejection details;
- `duplicate_report.json` — exact, normalized, and near-duplicate counts;
- `token_statistics.json` — document, token, byte, line, and shard metrics.

Structured JSONL logs are written to `logs/pypi_corpus_builder.jsonl`; console
logs and progress bars remain human-readable. One package failure is recorded
and does not discard successfully processed packages.

## Troubleshooting

- **Pipeline is disabled:** review package/license selectors and set `enabled`.
- **Tokenizer not found:** run `python scripts/train_code_tokenizer.py --force`.
  The PyPI builder intentionally will not train one.
- **No supported sdist:** the release publishes only wheels or an unsupported
  archive type; choose another release or package.
- **Checksum mismatch:** remove only that package's cached archive and rerun, or
  use `--force`. The mismatched file is never imported.
- **Rate or network failures:** rerun normally; valid archives and completed
  checkpoints are reused.
- **Unexpected rejection:** inspect `reports/quality_report.json` and the
  structured log for the exact collector reason.

## Tests

The tests are offline and use local tar/ZIP sdists plus a fake PyPI API:

```bash
pytest tests/test_pypi_corpus_builder.py \
  tests/test_python_corpus_collector.py \
  tests/test_python_corpus_expansion.py \
  tests/test_github_corpus_builder.py
```

Coverage includes all discovery selectors, sdist-only selection, metadata,
checksum enforcement, download resume, tar/ZIP extraction, path traversal,
directory filtering, shared validation, exact/normalized/near deduplication,
provenance, Corpus Manager indexing, existing-tokenizer integration, binary
shards, reports, streaming-loader compatibility, and full-pipeline resume.
