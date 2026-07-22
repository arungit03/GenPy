# Python Corpus Population

Corpus Population is the operational entry point for filling GenPy's approved
Python corpus. It combines the existing collector, classifier, and SQLite index
without changing model, tokenizer, training, or dataset-generation code.

## Populate the corpus

```bash
python scripts/populate_python_corpus.py
```

The default configuration imports the two GenPy source roots that were already
approved for dataset use. Additional collections must be reviewed before adding
them under `corpus_collection.sources` in `configs/dataset_pipeline.yaml`.

Approved repositories can also be added without editing YAML: copy each local
repository into its own directory below `data/imports/`, then rerun this command.
The collector automatically registers each directory, detects local Git clones,
and applies the same validation, deduplication, provenance, classification, and
indexing workflow. See `docs/python_corpus_collector.md` for optional per-source
license metadata.

Population accepts these source types:

- `local`: an approved local directory;
- `git`: an approved local Git repository;
- `zip`: an approved local ZIP archive.

Individual-file and remote-Git inputs remain available to lower-level collector
tools but are deliberately rejected by the population command. This milestone's
population boundary is local and explicitly configured.

Example:

```yaml
corpus_collection:
  sources:
    - id: algorithms_library
      type: local
      location: approved/algorithms
      license: MIT
      approval: Reviewed internal algorithms collection

    - id: local_project_checkout
      type: git
      location: approved/project-repository
      revision: v1.2.0
      license: Apache-2.0
      approval: Reviewed local project release

    - id: data_tools_release
      type: zip
      location: approved/data-tools.zip
      license: BSD-3-Clause
      approval: Reviewed data tools release archive

corpus_population:
  report: data/raw/corpus_population_report.json
```

Relative paths are resolved from the configured `project_root`. Every population
source requires a non-empty approval statement. License values are explicit
metadata and are never inferred.

## Population workflow

For every run, the system:

1. validates that all configured inputs satisfy the local population policy;
2. discovers and validates Python files through the existing corpus collector;
3. rejects invalid UTF-8, syntax errors, size violations, generated artifacts,
   unsafe ZIP entries, and SHA-256 duplicate files;
4. preserves source path, source type/location/revision, license, content hash,
   and collection timestamps;
5. classifies every accepted file into one or more supported categories;
6. atomically rebuilds the searchable SQLite file/symbol index;
7. writes a consolidated population report.

Repeated runs are incremental at collection time. Unchanged files are not
rewritten, while the index is rebuilt from the current provenance manifest so it
cannot retain stale corpus rows.

## Categories

The supported categories are:

- Core Python
- Algorithms
- Data Structures
- OOP
- File Handling
- Exception Handling
- Standard Library
- NumPy
- Pandas
- Matplotlib
- Pytest

Classification uses imports, paths, symbol names, and AST structure. Files can
have multiple categories and also receive a deterministic primary category.

## Search

Search paths, source IDs, original paths, and qualified function/class names:

```bash
python scripts/populate_python_corpus.py --search binary_search
```

Filter by category, with or without text:

```bash
python scripts/populate_python_corpus.py --category Algorithms
python scripts/populate_python_corpus.py \
  --search graph \
  --category "Data Structures" \
  --limit 50
```

Results are JSON objects containing paths, source and license metadata,
categories, function/class counts, and matching qualified symbols. SQLite also
remains directly queryable at `data/raw/corpus_index.sqlite3`.

## Reports and rebuilds

`data/raw/corpus_population_report.json` contains:

- Python files imported and unchanged in the current run;
- total indexed Python files;
- functions and classes discovered;
- all category counts;
- duplicate files detected by SHA-256;
- estimated instruction pairs;
- rejection reasons, approved source declarations, index location, and
  provenance-manifest location.

The source files and provenance live under `data/raw/`. The existing dataset
pipeline already treats that directory as an approved source root, so the
dataset can be reproduced at any time:

```bash
python scripts/build_dataset.py --force
```

Corpus SQLite/JSON metadata is ignored by Python-file discovery.
