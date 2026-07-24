# Final Corpus Builder

The final corpus builder combines every local GenPy corpus into one train-ready dataset for continued pretraining. It does not download data, alter checkpoints, modify the tokenizer, or change training code.

## Pipeline

1. Discover supported local files from `python_corpus/github`, `docs`, `peps`, `tutorials`, `cleaned`, and `cleaned_docs`.
2. Skip missing folders, ignored build folders, binary artifacts, empty files, tiny files, and oversized files.
3. Validate UTF-8, normalize whitespace, and clean documentation noise while preserving Python code, Markdown/RST structure, tables, headings, examples, and code fences.
4. Deduplicate normalized documents across all source folders.
5. Tokenize with the existing GenPy tokenizer from `data/tokenizer/tokenizer.json`.
6. Pack with the existing `SequencePacker` by default, then write uint16 sequence shards.
7. Write cleaned documents, metadata, shard index, and statistics.

## Folder Layout

```text
python_corpus/
    github/
    docs/
    peps/
    tutorials/
    cleaned/
    cleaned_docs/
    final_corpus/
        cleaned/
        packed/
        metadata/
        statistics/
```

## Configuration

The default config is `configs/final_corpus.yaml`.

```yaml
final_corpus:
  corpus_root: python_corpus
  output_directory: final_corpus
  source_directories: [github, docs, peps, tutorials, cleaned, cleaned_docs]
  tokenizer: data/tokenizer/tokenizer.json
  minimum_size: 80
  maximum_size: 2000000
  deduplication: true
  comment_removal: false
  packing:
    sequence_length: 1025
    overlap: 0
    packing_strategy: packed
```

`packing_strategy: packed` reuses the existing `SequencePacker`. Use `packing_strategy: sliding_window` only when a nonzero `overlap` is needed.

## CLI

```bash
python scripts/build_final_corpus.py --config configs/final_corpus.yaml --force
```

`--force` removes only `python_corpus/final_corpus/` before rebuilding.

## Statistics

The builder writes:

- `python_corpus/final_corpus/statistics/final_statistics.json`
- `python_corpus/final_corpus/statistics/final_statistics.md`

Reports include total input files, processed files, skipped files, duplicates removed, code/documentation file counts, characters, words, tokens, packed sequences, average sequence length, largest files, smallest files, and estimated epochs.

## Outputs

- `cleaned/`: normalized text files used for packing.
- `packed/`: uint16 shard files, gzipped shard metadata, and `index.json`.
- `metadata/final_manifest.jsonl`: accepted source records.
- `metadata/final_rejected.jsonl`: skipped source records and reasons.
- `statistics/`: JSON and Markdown reports.
