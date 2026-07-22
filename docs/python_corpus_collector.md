# Automatic Python Corpus Collector

The corpus collector is the ingestion layer before GenPy's instruction-dataset
pipeline. It copies only validated, explicitly approved Python material into
`data/raw/`; it does not generate examples or alter downstream cleaning,
instruction generation, deduplication, validation, or splitting.

Run it with:

```bash
python scripts/collect_python_corpus.py
python scripts/build_dataset.py
```

Both commands use `configs/dataset_pipeline.yaml` by default. Pass
`--config PATH` to use another configuration.

## Drop-in repository imports

The default configuration watches `data/imports/`. To add approved local
collections, copy each repository into its own first-level directory and rerun
the population command:

```text
data/imports/
├── approved-algorithms/
│   ├── .git/
│   └── src/
└── approved-data-tools/
    └── package/
```

```bash
python scripts/populate_python_corpus.py
```

Every first-level directory is registered under a stable `import-...` source ID.
A directory containing `.git` is handled as a local Git clone and its exact
commit SHA is recorded; other directories are handled as local source trees.
Top-level ZIP archives are also discovered when `discover_zip_archives` is
enabled. Hidden entries, loose files, and symbolic links in `data/imports/` are
not registered.

Placement in this configured directory is the explicit approval action. The
collector never guesses a license. To preserve a known per-repository license
or override selection rules, include an optional `.genpy-corpus.yaml` in the
repository root:

```yaml
license: MIT
approval: Reviewed and approved internal collection
include:
  - "src/**/*.py"
exclude:
  - "**/tests/**"
```

Without that file, the configured `default_license` is recorded (`null` in the
default project configuration). A ZIP can use an adjacent metadata file named
`archive.zip.genpy-corpus.yaml`. Explicit entries under
`corpus_collection.sources` take precedence when they point to the same path.

Automatic discovery is configured independently from the dataset pipeline:

```yaml
corpus_collection:
  automatic_imports:
    enabled: true
    directory: data/imports
    approval: Explicitly approved by placement in the GenPy imports directory
    default_license:
    metadata_file: .genpy-corpus.yaml
    discover_zip_archives: true
```

## Adding data directly

Place an approved Python file or directory anywhere below `data/raw/`, preserving
whatever organization is useful:

```text
data/raw/
└── my_approved_dataset/
    ├── package_a/
    │   └── module.py
    └── utilities.py
```

The next collector run validates and registers previously unseen `.py` files.
Set `corpus_collection.manual_license` when all manually placed files share a
known license; otherwise it remains `null`. License values are explicit metadata
and are never guessed.

The existing dataset pipeline has `data/raw` as an approved source root, so a
subsequent `python scripts/build_dataset.py` includes accepted raw Python files.

## Configured sources

Add reviewed entries under `corpus_collection.sources`. Every source needs a
unique ID, type, location, and explicit license value or `null`.

### Local directory

```yaml
corpus_collection:
  sources:
    - id: approved_local_library
      type: local
      location: datasets/approved_library
      license: MIT
      include:
        - "src/**/*.py"
      exclude:
        - "**/tests/**"
```

### Git repository

`location` may be a local repository path or a Git URL. A configured `revision`
is checked out before scanning. The exact resulting commit SHA is recorded for
every accepted file.

```yaml
    - id: approved_git_project
      type: git
      location: https://example.com/organization/project.git
      revision: v2.1.0
      license: Apache-2.0
```

Git sources require the `git` executable. Remote repositories are accessed only
when this collector command is explicitly run.

### ZIP archive

```yaml
    - id: approved_archive
      type: zip
      location: datasets/approved_release.zip
      license: BSD-3-Clause
```

ZIP files are read without extracting the archive wholesale. Absolute paths,
parent traversal, and archived symbolic links are rejected.

### Individual Python file

```yaml
    - id: approved_single_file
      type: file
      location: datasets/approved_tool.py
      license: MIT
```

Relative locations are resolved from `project_root` in the YAML file. Each
configured source is stored below its own namespace, for example
`data/raw/approved_archive/package/module.py`, preserving the path inside the
source.

## Validation and filtering

Every discovered candidate must:

- have a `.py` filename;
- be within `minimum_file_bytes` and `maximum_file_bytes`;
- decode as UTF-8 (a UTF-8 BOM is supported);
- parse successfully with Python's AST parser.

Directory traversal automatically prunes virtual environments, VCS metadata,
`__pycache__`, build/distribution directories, package caches, and configured
generated directories. Generated filename patterns such as `*_generated.py`,
`*_pb2.py`, and `*_ui.py` are rejected, as are files with common generated-code
markers in their opening lines. Ignored directories, filename patterns, and
header markers are configurable in `dataset_pipeline.yaml`.

## Incremental behavior

The collector records SHA-256 content hashes and stable destination paths.
Subsequent runs:

- leave byte-identical files untouched;
- atomically replace a tracked destination when its configured source changes;
- reject content already collected at another path as `duplicate_content`;
- preserve entries from sources not involved in the current run;
- detect manual changes to tracked files instead of silently trusting stale metadata.

Files are written through `.partial` files followed by atomic replacement. ZIP
archives and Git checkouts use temporary working directories outside the raw
corpus.

## Provenance and report

`data/raw/collection_manifest.jsonl` contains one record per collected file:

- content SHA-256 and byte size;
- source ID, type, configured location, and Git revision when applicable;
- original source path and stored path;
- explicit license field;
- initial collection and most recent observation timestamps.

`data/raw/collection_report.json` records total and per-source counts for files
scanned, accepted, unchanged, and rejected, plus rejection reasons such as
`invalid_utf8`, `invalid_python_syntax`, `too_small`, `too_large`,
`generated_file`, and `duplicate_content`.
