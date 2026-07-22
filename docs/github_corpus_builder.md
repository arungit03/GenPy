# Phase 5.5A: GitHub Corpus Builder

The GitHub corpus builder adds remote discovery and binary pre-training shards
around GenPy's existing corpus infrastructure. It deliberately reuses the raw
corpus collector, AST syntax validation, generated-file filters, SHA-256
deduplication, provenance manifest, SQLite corpus index, 32K tokenizer, progress
bar, and streaming training dataset.

## Approval and configuration

Review `github_corpus` in `configs/dataset_pipeline.yaml` before enabling it.
The checked-in configuration is disabled to prevent unreviewed downloads. Its
license allowlist is part of the approval boundary; repository popularity is not
a substitute for license review.

```yaml
github_corpus:
  enabled: true
  search:
    language: Python
    minimum_stars: 100
    updated_after: 2020-01-01
    updated_before: null
    include_archived: false
    include_forks: false
    allowed_licenses: [MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC]
    queries: [""]
    maximum_repositories: 10000
```

Every query is combined with language, stars, updated-date, archived, fork, and
license qualifiers. GitHub exposes only the first 1,000 search results for one
query, so the client recursively partitions large result sets into deterministic
date windows. Multiple query terms and license partitions are deduplicated by
case-insensitive `owner/repository` name.

Set a token in the configured environment variable for higher API limits:

```bash
export GITHUB_TOKEN=github_pat_...
```

The token is sent as an API authorization header. It is never written to logs,
checkpoints, reports, Git configuration, clone URLs, or provenance.

## Running and resuming

```bash
python scripts/build_github_corpus.py
python scripts/build_github_corpus.py --max-repositories 250
python scripts/build_github_corpus.py --force
```

Repository state lives in `data/github_corpus/checkpoint.sqlite3`. Completed
clones are reused when repository metadata is unchanged. Interrupted clone
partials are isolated and retried; failed repositories are recorded individually
without discarding successful work. `--force` refreshes checkouts and rebuilds
the binary shards.

The pipeline stages are:

1. Search GitHub with the configured quality and license policy.
2. Clone repositories concurrently into `data/github_cache/owner/repository`.
3. Resolve and record the exact checked-out commit SHA.
4. Submit each checkout to the existing corpus collector as a GitHub source.
5. Preserve directory structure under `data/raw/github-.../`.
6. Reuse UTF-8, file-size, generated-code, AST syntax, and SHA-256 checks.
7. Encode accepted GitHub files with `data/tokenizer/tokenizer.json` in bounded,
   input-order multiprocessing batches.
8. Write atomic, document-aligned uint16 binary shards and reports.

The default source exclusions cover virtual environments and build output from
the existing collector, plus vendored/third-party trees and test fixtures from
the GitHub settings. Test-fixture exclusion is configurable. Extremely small or
large file thresholds continue to come from `corpus_collection`, keeping one
validation policy across local, ZIP, Git, and GitHub material.

## Provenance and indexing

Every accepted file retains:

- repository URL and `owner/repository` source ID;
- GitHub stars and detected SPDX license;
- repository creation, update, and push timestamps;
- default branch and exact downloaded commit SHA;
- original source path, collected path, SHA-256, file size, approval, and
  collection timestamp.

These values are stored in `data/raw/collection_manifest.jsonl`. Rebuilding the
existing corpus index also places the GitHub fields in searchable columns while
retaining function/class duplicate detection.

## Binary format and training integration

The default output directory is `data/pretraining/github/`:

- `github_tokens_00000.bin`, ... — little-endian uint16 token IDs;
- `document_index.jsonl` — source provenance and token offset for every document;
- `shard_index.json` — format contract, tokenizer hash, shard hashes/counts, and
  build fingerprint;
- `token_statistics.json` — token, source-size, compression, rejection, and shard
  statistics.

Documents never cross shard boundaries and each ends with `<eos>`. A single file
larger than the target shard size receives its own shard rather than being split.
The index pins the tokenizer SHA-256, vocabulary size, and EOS ID.

`StreamingGPTDataset` reads either existing gzip JSONL shards or the new `.bin`
files. It validates the binary index and tokenizer identity before yielding GPT
input/target windows. To use GitHub shards for base training, point the training
glob at them while retaining an independent validation corpus:

```yaml
streaming_dataset:
  train_pattern: data/pretraining/github/github_tokens_*.bin
  validation_pattern: data/code_shards/validation/*.jsonl.gz
```

## Reports and logs

Reports under `reports/github_corpus/` include:

- `repositories.json` — discovery, clone status, commit, stars, and timestamps;
- `licenses.json` — discovered/downloaded repository counts by license;
- `quality.json` — accepted, unchanged, rejected, and tokenization quality;
- `rejected_files.json` — file rejection reasons and repository download errors;
- `statistics.json` — aggregate repositories, files, tokens, shards, and paths.

Structured JSONL logs are written to `logs/github_corpus_builder.jsonl`, while a
human-readable stream and progress bars are shown in the terminal.

## Tests

The test suite is offline: it uses local Git repositories and a fake GitHub API.

```bash
pytest tests/test_github_corpus_builder.py \
  tests/test_python_corpus_collector.py \
  tests/test_python_corpus_expansion.py
```

Coverage includes search-window partitioning, filters, cloning, commit metadata,
syntax rejection, generated/vendor/test filtering, file deduplication, resume,
all reports, binary indexes, tokenizer pinning, and direct streaming-dataset use.
