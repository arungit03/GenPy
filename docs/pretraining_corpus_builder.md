# Phase 5.5C Pretraining Corpus Builder

Phase 5.5C builds the final deterministic Python pretraining corpus for Phase 6.
It does not download source code and it does not retrain the tokenizer. It reads
the existing `data/raw/collection_manifest.jsonl`, revalidates the files under
`data/raw/`, performs a final global deduplication pass, tokenizes with the
existing GenPy tokenizer, packs fixed-length GPT sequences, and writes binary
training shards.

## Run

```bash
python scripts/build_pretraining_corpus.py
```

Use `--force` to intentionally rebuild existing final shards:

```bash
python scripts/build_pretraining_corpus.py --force
```

## Configuration

Settings live in `configs/pretraining.yaml`.

Important fields:

- `pretraining_corpus.source_types`: source types to merge from the shared raw
  provenance manifest. Current values include `github`, `pypi`, `local`, `git`,
  `zip`, `file`, and `manual_raw`.
- `pretraining_corpus.validation`: final UTF-8, syntax, size, generated-code,
  vendored-code, and cleaner settings.
- `pretraining_corpus.deduplication`: exact SHA-256 deduplication plus optional
  whitespace, comment, newline, and AST normalization.
- `pretraining_corpus.tokenization.tokenizer`: path to the existing Phase 5
  tokenizer. The builder loads this tokenizer and never retrains it.
- `pretraining_corpus.packing.context_length`: GPT context length. Stored
  sequences contain `context_length + 1` token IDs so Phase 6 can form shifted
  input and target tensors.
- `pretraining_corpus.shards.max_tokens_per_shard`: target maximum token IDs per
  binary shard.

## Outputs

Final training files are written under `data/pretraining/`:

- `shard_00000.bin`, `shard_00001.bin`, ...
- `shard_00000.metadata.json.gz`, `shard_00001.metadata.json.gz`, ...
- `index.json`
- `manifest.json`
- `statistics.json`
- `corpus_manifest.jsonl`

Each `.bin` shard stores little-endian uint16 token IDs in fixed-length packed
sequence order. Each gzipped sidecar records sequence offsets, document offsets,
padding counts, and source provenance for inspection and debugging.

Reports are written under `reports/pretraining/`:

- `corpus_report.json`
- `statistics.json`
- `quality_report.json`
- `duplicate_report.json`
- `validation_report.json`
- `license_report.json`
- `source_report.json`
- `token_statistics.json`
- `shard_statistics.json`

## Resume

Resume is enabled by default. The builder computes a deterministic fingerprint
from the selected manifest records, deduplication settings, packing settings,
and tokenizer hash. If existing shards, metadata sidecars, `index.json`, and
`statistics.json` match that fingerprint and their checksums still validate, the
builder reuses them and regenerates reports from the current manifest.

## Source Policy

Phase 5.5C only consumes files that already passed into `data/raw/` through the
approved corpus collector. To add material, use the GitHub builder, PyPI builder,
or approved local import workflow first, then rerun this final builder.
