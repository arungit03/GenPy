# Python Corpus Expansion

The expansion framework imports approved source collections through GenPy's
existing corpus collector, classifies the validated Python files, and builds a
reproducible SQLite index. It prepares the corpus for future 100K+ instruction
datasets without generating synthetic examples or changing the dataset pipeline.

## Run the framework

```bash
python scripts/expand_python_corpus.py
```

This command performs two operations:

1. Runs the configured corpus collector, importing approved local directories,
   Git repositories, ZIP archives, individual files, and approved files placed
   directly under `data/raw/`.
2. Atomically rebuilds `data/raw/corpus_index.sqlite3` from the collector's
   provenance manifest.

To rebuild the index without fetching or rescanning configured sources:

```bash
python scripts/expand_python_corpus.py --index-only
```

Both modes accept `--config PATH` and
`--log-level DEBUG|INFO|WARNING|ERROR`.

## Configuration

Source collections remain configured under `corpus_collection.sources`; see
`docs/python_corpus_collector.md` for local, Git, ZIP, and file examples.

Expansion artifacts are configured separately in the same
`configs/dataset_pipeline.yaml`:

```yaml
corpus_expansion:
  index: data/raw/corpus_index.sqlite3
  report: data/raw/corpus_expansion_report.json
  collect_before_index: true
```

Only sources whose licensing and provenance have been reviewed should be added.
The framework never searches for or downloads unconfigured collections.

## Validation boundary

The existing collector validates source size, UTF-8, Python syntax, generated
artifacts, archive paths, and content duplicates before copying anything into
the corpus. The expansion framework then validates every manifest record again
before indexing:

- the stored path must remain inside `data/raw/`;
- the file must still exist;
- its SHA-256 must match the provenance manifest;
- its size must remain within configured bounds;
- it must still decode as UTF-8 and parse as Python;
- source, license, and collection timestamp fields must be present.

Failures are excluded from SQLite and counted by reason in the expansion report.

## Automatic classification

Classification is deterministic and based only on file paths, imports, symbol
names, and Python AST structure. Files can have multiple categories, while one
primary category is stored for grouping.

| Category | Typical signals |
| --- | --- |
| Core Python | Valid Python without a more specific signal |
| OOP | Class definitions |
| Algorithms | Algorithm/search/sort/traversal names and paths |
| Data Structures | Stack, queue, tree, graph, heap, trie, and related imports |
| File Handling | File APIs and modules such as `pathlib`, `io`, `csv`, and `pickle` |
| Exception Handling | `try` or `raise` nodes |
| Standard Library | Imports present in Python's standard-library module set |
| NumPy | `numpy` imports |
| Pandas | `pandas` imports |
| Matplotlib | `matplotlib` imports |
| Pytest | `pytest` imports or test paths/filenames |

The report includes both multi-label category counts and mutually exclusive
primary-category counts.

## SQLite corpus index

The index is rebuilt through a temporary database followed by atomic replacement.
Its tables are:

- `files`: stored/original paths, source ID/type/location/revision, license,
  language, SHA-256, byte size, timestamps, primary category, function/class
  counts, and estimated instruction pairs;
- `file_categories`: all category assignments for each file;
- `symbols`: every function, async function, method, nested definition, and
  class with qualified name, line span, definition hash, and duplicate-symbol
  linkage;
- `metadata`: index version, creation timestamp, and aggregate totals.

The collector prevents duplicate files by SHA-256. The symbol index additionally
links repeated function/class definitions without discarding provenance.

## Expansion statistics

`data/raw/corpus_expansion_report.json` records:

- total repositories/source collections represented in accepted files;
- total indexed Python files;
- total functions, including methods and nested functions;
- total classes;
- estimated instruction pairs, currently one per function/async function/class;
- duplicate symbols;
- rejected index records and reasons;
- category and primary-category file counts;
- the associated import/collection result.

The estimate follows the current dataset generator's AST extraction unit. It is
a capacity estimate, not generated training data; later deduplication and dataset
policy can reduce the final number of examples.

## Dataset compatibility

The dataset-generation code is unchanged. Approved files remain in `data/raw/`,
which is already configured as an approved source root. Therefore the complete
dataset can be rebuilt at any time with:

```bash
python scripts/build_dataset.py --force
```

SQLite and expansion-report files are metadata only and are ignored by the
pipeline's `.py` discovery.
