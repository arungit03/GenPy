# Phase 6.2 Corpus V2

Phase 6.2 builds a large local corpus for continued pretraining. It does not
start training, modify the Transformer, retrain the tokenizer, touch
checkpoints, change LoRA, change quantization, or modify API/frontend code.

## Supported Sources

Corpus V2 only reads local, approved material:

- local Python projects;
- local Markdown documentation;
- local reStructuredText documentation;
- local technical `.txt` files;
- already-cloned and approved Git repositories;
- already-downloaded and approved datasets.

It does not scrape websites and does not download random copyrighted material.
Add new sources to `configs/corpus_v2.yaml` only after source and license review.
The checked-in `approved_imports` source is present but disabled; enable it only
after confirming the local imports are intended for the Phase 6.2 corpus run.

## Build

```bash
python scripts/build_corpus_v2.py
```

Force shard regeneration:

```bash
python scripts/build_corpus_v2.py --force
```

Analyze existing reports:

```bash
python scripts/analyze_corpus_v2.py
```

## Pipeline

The build is modular:

1. `collector.py` recursively scans configured local roots, applies
   include/exclude patterns, skips build/env/cache folders, archives, binary
   files, and unsupported extensions.
2. `cleaner.py` normalizes UTF-8 text, line endings, tabs, whitespace, repeated
   blank lines, and rejects empty, short, generated, corrupted, or minified files.
3. `validator.py` verifies Python syntax with `ast.parse` and validates
   technical text through quality filters.
4. `quality.py` rejects low entropy text, base64 blobs, hex dumps, repeated
   sequences, and nontechnical documentation.
5. `deduplicator.py` removes exact, normalized, and near duplicates at the
   document/file level.
6. `tokenizer.py` reuses `data/tokenizer/tokenizer.json`; the vocabulary is not
   modified.
7. `packer.py` writes fixed-length Phase-6-compatible uint16 binary shards.
8. `statistics.py` and `quality.py` write readiness reports.

## Outputs

Corpus artifacts:

- `data/corpus_v2/document_manifest.jsonl`
- `data/corpus_v2/index.json`
- `data/corpus_v2/statistics.json`
- `data/corpus_v2/corpus_v2_00000.bin`, ...
- `data/corpus_v2/corpus_v2_00000.metadata.json.gz`, ...

Reports:

- `reports/corpus_v2/quality_report.md`
- `reports/corpus_v2/quality_report.json`
- `reports/corpus_v2/statistics.csv`
- `reports/corpus_v2/manifest.json`

## Readiness Gate

The build passes readiness only when:

- total tokens are at least `corpus_v2.readiness.minimum_tokens`;
- Python token ratio is within the configured range;
- technical text token ratio is within the configured range;
- duplicate percentage is below the configured threshold;
- validation has no fatal failures.

If readiness fails, the command still writes corpus/reports and stops. Phase 6.3
can decide whether to consume the shard index after reviewing the readiness
report.

## Scaling Notes

Corpus V2 performs two deterministic passes. The first pass computes
deduplication, token counts, manifest records, statistics, and the build
fingerprint. If existing shards match that fingerprint, they are reused. If not,
the second pass streams token IDs directly into the shard writer so packed token
arrays are not retained across the whole corpus.
