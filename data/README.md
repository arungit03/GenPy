# Data Directory

This folder contains local data artifacts for GenPy LLM.

- `raw/` stores original text and approved Python corpus files.
- `processed/` stores cleaned text.
- `tokenized/` stores token strings in JSONL files.
- `vocabulary/` stores vocabulary mappings and encoded token IDs.
- `datasets/` stores prepared train, validation, and test tensors.
- `tokenizer/` stores the reproducible code tokenizer and its training metadata.

Run `python scripts/train_code_tokenizer.py --force` to rebuild
`tokenizer/code_tokenizer.json` and `tokenizer/tokenizer_metadata.json` from the
available local corpus. Both files are intentionally kept in the repository so
code training and fine-tuning work without copying an external tokenizer.

`fine_tuning/train.jsonl`, `fine_tuning/validation.jsonl`, and
`fine_tuning/test.jsonl` are generated from approved Python sources by
`python scripts/build_dataset.py`. Aggregate counts and hashes are stored in
`fine_tuning/dataset_statistics.json`; resumable intermediate artifacts live in
the ignored `dataset_pipeline/` workspace.

Run `python scripts/collect_python_corpus.py` to validate and register approved
Python files placed in `raw/`, or to collect configured local, Git, ZIP, and
individual-file sources. Per-file provenance and the latest collection report
are written inside `raw/`; configuration examples are documented in
`docs/python_corpus_collector.md`.

Run `python scripts/expand_python_corpus.py` to import configured collections,
classify the validated files, and rebuild `raw/corpus_index.sqlite3` plus
`raw/corpus_expansion_report.json`. These metadata artifacts make corpus
statistics and provenance queryable without changing dataset generation.

`python scripts/populate_python_corpus.py` is the production entry point for
approved local directory, local Git, and ZIP imports. It additionally writes
`raw/corpus_population_report.json`; use its `--search` and `--category` options
to query the corpus index.

`train.pt` contains training samples, `validation.pt` contains validation samples, and `test.pt` contains test samples. `dataset_metadata.json` records preparation settings and statistics.

Prepared datasets contain token ID tensors, shifted target tensors, and padding masks. They do not contain embeddings, trained weights, or model outputs. Step 6 will convert token IDs into embeddings.

Large generated datasets should remain excluded from Git. Only small sample artifacts should be committed intentionally.

Before using any text for training, make sure you have permission to use it.
