# Python Dataset Generation Pipeline

GenPy's dataset builder converts explicitly approved local Python source into a
provenance-preserving instruction-following dataset. It does not download data
or substitute unapproved code responses. Phase 4 transformations are
deterministic, AST-derived, and recorded in provenance.

## Architecture

The pipeline is configured by `configs/dataset_pipeline.yaml`. Relative paths in
that file are resolved from the configuration file directory. Every approved
source must provide a stable ID, filesystem root, repository name, and an
explicit approval statement. An optional license value is carried into every
output record without being inferred.

The processing flow is:

```text
approved .py files
  -> collect -> clean -> generate -> deduplicate -> validate -> split
  -> train.jsonl / validation.jsonl / test.jsonl
```

Intermediate files and per-stage manifests live under
`data/dataset_pipeline/`. A stage manifest records its input/configuration
fingerprint, counters, output hashes, file size/mtime signatures, and completion
timestamp. Resume checks use the inexpensive signatures by default, avoiding a
full read of every large artifact. Set
`performance.verify_output_hashes_on_resume: true` for a full SHA-256 check on
each resumed stage. `--force` rebuilds instead.

Writes use `.partial` files followed by atomic replacement, so interrupted runs
do not expose half-written JSONL outputs.

## Scripts

### `collect_python_data.py`

Walks only configured approved roots. Include/exclude patterns, symlink
rejection, UTF-8 decoding, source identity, approval, license, byte size, and
SHA-256 content hashes are recorded in `01_collected.jsonl`.

### `clean_python_dataset.py`

Normalizes line endings and control characters while preserving indentation.
It rejects files outside configured size bounds, files with invalid Python ASTs,
and files without Python definitions. Rejection reasons are counted.
Parsing can run in a bounded process pool and results are written in original
input order.

### `generate_instruction_pairs.py`

Uses Python's AST to locate functions, async functions, classes, methods,
nested definitions, imports, decorators, docstrings, and type hints. Phase 4 can
emit multiple grounded tasks for each element:

- code generation and completion retain the real source definition;
- explanation and complexity analysis use explicit AST signals;
- bug fixing applies one recorded AST mutation to the input and retains the
  approved source as the correction;
- refactoring uses behavior-preserving AST normalization;
- documentation uses an existing source docstring;
- unit testing uses existing `test_*` definitions;
- optimization is limited to equivalent append-loop-to-comprehension rewrites;
- type-hint tasks remove annotations from the input and retain the typed source;
- API usage pairs an observed import with a definition that references it.

Records carry a category plus transformation and AST metadata in provenance.
When `maximum_examples_per_file` is set, a successfully materialized example is
reserved for each applicable category before remaining slots are selected by a
stable content hash. Expensive transformations run only after selection. AST
work uses the same bounded, deterministic process pool.

### `deduplicate_dataset.py`

Computes a canonical SHA-256 hash over normalized instruction, input, and code.
Only the first exact semantic pair is retained. Exact hashes are indexed in a
temporary SQLite database instead of an unbounded Python set. Deduplication
happens before any split, preventing duplicate leakage.

### `validate_dataset.py`

Strictly parses JSONL, validates required field types and bounds, reparses every
code-category response as Python, checks categories, provenance, and
record/pair uniqueness.
Rejected records and stable error codes are written to
`validation_rejections.jsonl`. Strict mode stops the build if any record fails.
The uniqueness indexes are disk-backed and committed in configurable batches.

### `split_dataset.py`

Groups records by approved source file (or content hash when configured), then
assigns whole groups deterministically using the configured seed and ratios.
This prevents definitions from the same source file appearing in multiple
splits. All three files are created, including empty files for very small input
corpora. Group discovery, record spooling, and final ordered writes are separate
streaming passes backed by a disposable SQLite index; the corpus is never loaded
into a Python list.

### `build_dataset.py`

Runs all six stages, aggregates their statistics, and writes
`data/fine_tuning/dataset_statistics.json` with counts, sizes, and output hashes.
Counts and hashes already captured by stage manifests are reused, avoiding
redundant full-file scans.

## Scaling and resource bounds

The architecture is intended to keep Python heap use bounded as approved data
grows beyond 100,000 instruction records. This change improves capacity; it does
not synthesize or add 100,000 examples.

| Work | Large-state location | Python memory bound |
| --- | --- | --- |
| Cleaning and AST generation | Worker processes | Configured pending tasks plus one source file per task |
| Exact deduplication | Temporary SQLite primary-key index | One record plus SQLite's bounded cache |
| ID/pair validation | Temporary SQLite primary-key indexes | One record plus SQLite's bounded cache |
| Group-safe split and ordering | Temporary SQLite tables/index | One record batch plus SQLite's bounded cache |

Temporary indexes live under `data/dataset_pipeline/indexes/` only while their
stage is running and are removed on success or failure. Allow working disk space
for the intermediate JSONL artifacts and SQLite indexes.

The `performance` configuration controls throughput and resource usage:

```yaml
performance:
  workers: 0                       # auto, capped at 8; use 1 to disable processes
  max_pending_tasks_per_worker: 4  # bounds queued source records
  sqlite_batch_size: 1000          # bounds write batches
  verify_output_hashes_on_resume: false
```

Output is deterministic across worker counts because completed AST tasks are
consumed in input order. Each stage takes its progress total from the verified
upstream manifest and falls back to a streaming count only when that metadata is
unavailable.

Phase 4 generation is configured under `instruction_generation`:

```yaml
instruction_generation:
  enabled_categories:
    - code_generation
    - explanation
    - bug_fixing
    - refactoring
    - documentation
    - unit_testing
    - optimization
    - complexity_analysis
    - type_hints
    - code_completion
    - api_usage
  maximum_examples_per_file: 16
  selection_seed: 42
  templates:
    explanation: "Explain the behavior of the {kind} `{qualified_name}`."
```

Supported template fields are `api`, `base_instruction`, `description`,
`kind`, `name`, `qualified_name`, and `signature`. Unknown categories, template
fields, and empty templates fail configuration loading.

## Running

Run the complete pipeline:

```bash
python scripts/build_dataset.py
```

Run one stage:

```bash
python scripts/collect_python_data.py
python scripts/clean_python_dataset.py
python scripts/generate_instruction_pairs.py
python scripts/deduplicate_dataset.py
python scripts/validate_dataset.py
python scripts/split_dataset.py
```

Completed current stages resume automatically. Rebuild with:

```bash
python scripts/build_dataset.py --force
```

Use another configuration with `--config PATH`. Each CLI also accepts
`--log-level DEBUG|INFO|WARNING|ERROR`.

## Final JSONL schema

Each final line is a JSON object with:

- `schema_version`
- `record_id`
- `category` — the Phase 4 instruction category
- `instruction` — a real docstring or a source-grounded inferred instruction
- `input` — optional code or API context for transformations and analysis
- `output` — approved code or a deterministic AST-grounded text response
- `deduplication_hash`
- `provenance` — source ID, repository, relative path, source hash, approval,
  optional license, symbol name/type, category, imports, decorators, type hints,
  instruction source, transformation metadata, and source line range

Adding external material requires adding a reviewed entry to
`collection.approved_sources`; the collector never searches unapproved paths.
