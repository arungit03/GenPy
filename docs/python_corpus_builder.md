# Python Corpus Builder

The Python Corpus Builder prepares a local-only GenPy pretraining corpus from
files you manually place on disk. It never downloads repositories, never calls
GitHub, and never modifies model checkpoints.

## Input Layout

```text
python_corpus/
  github/
    repo1/
    repo2/
  docs/
  peps/
  tutorials/
```

Only these file types are read:

- `.py`
- `.pyi`
- `.md`
- `.rst`
- `.txt`

The scanner ignores common generated, binary, environment, and VCS paths such as
`.git`, `.github`, `node_modules`, `venv`, `build`, `dist`, `__pycache__`,
images, PDFs, archives, native libraries, and binaries.

## Run

```bash
python scripts/build_python_corpus.py --config configs/python_corpus.yaml --force
```

## Example Configuration

```yaml
version: 1
project_root: ..

python_corpus:
  input_directory: python_corpus
  output_directory: data/python_corpus
  tokenizer: data/tokenizer/tokenizer.json
  min_file_size: 80
  max_file_size: 2000000
  allowed_extensions: [.py, .pyi, .md, .rst, .txt]
  deduplication: true
  preserve_comments: true

  packing:
    context_length: 1024
    max_tokens_per_shard: 10000000
    shard_prefix: python_corpus
    add_bos: false
    add_eos: true
    document_boundary: eos
    pad_final_sequence: false
```

## Outputs

```text
data/python_corpus/
  cleaned/
  packed/
    index.json
    python_corpus_00000.bin
    python_corpus_00000.metadata.json.gz
  statistics/
    statistics.json
    statistics.md
  metadata/
    manifest.jsonl
    rejected.jsonl
```

## Example Statistics

```json
{
  "number_of_repositories": 3,
  "number_of_files": 125000,
  "python_files": 98000,
  "documentation_files": 27000,
  "total_characters": 850000000,
  "total_words": 92000000,
  "total_tokens": 210000000,
  "average_file_size": 6800.0,
  "duplicate_count": 1842
}
```

## Known Limitations

- Deduplication is exact after normalization, not fuzzy near-duplicate detection.
- Python syntax validation is intentionally not enforced; broken but useful
  training snippets can remain in local corpora.
- Comment removal is optional and applies only to Python files.
- Output shards are uint16 and require a tokenizer with fewer than 65,536 IDs.
