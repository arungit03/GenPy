# Phase 6.1 Continued Pretraining

Phase 6.1 expands the Phase 6 corpus target from the local 10M-token scale to a
200M-500M-token range, keeps aggressive global deduplication enabled, mixes
Python code with technical documentation, resumes from an existing checkpoint,
and writes benchmark artifacts for before/after comparison.

## Corpus Expansion

Review and approve sources first. The checked-in collector now supports
technical text only when explicitly configured through `allowed_extensions`.
`configs/dataset_pipeline.yaml` includes the GenPy `docs/` directory as a local
technical-text source for the Phase 6.1 mix, while GitHub and PyPI imports remain
policy-gated.

```bash
python scripts/collect_python_corpus.py
python scripts/build_pretraining_corpus.py --force
```

The final builder keeps exact SHA-256, whitespace-normalized,
comment-normalized, and newline-normalized duplicate detection. Technical text
is validated separately from Python code, then tokenized with the existing GenPy
tokenizer and packed into the same binary shard format.

## Run Phase 6.1

```bash
python scripts/run_phase6_1.py
```

Useful controls:

```bash
python scripts/run_phase6_1.py --skip-training --skip-evaluation
python scripts/run_phase6_1.py --force-corpus
```

By default, Phase 6.1 refuses to train until the packed corpus has at least
200M tokens, no more than 500M tokens, and the configured code/text balance. The
readiness artifacts are written to `reports/phase6_1/`.

## Continued Training

Training uses the normal Phase 6 trainer and config files. The Phase 6.1 config
sets:

```yaml
phase6_1:
  training:
    resume_from: checkpoints/last_checkpoint.pt
```

Set `phase6_1.training.max_steps` to the desired new absolute global step. The
Phase 6 trainer resumes optimizer, scheduler, scaler, RNG state, global step,
and best metric from the checkpoint.

## Benchmarks

`phase6_1.evaluation.commands` contains benchmark commands. Use `{python}` as
the command executable when a benchmark should run under the same interpreter as
the Phase 6.1 runner. The default compares `checkpoints/best_model.pt` and the
continued `checkpoints/last_checkpoint.pt` with the existing coding prompt
benchmark:

```bash
{python} scripts/evaluate_gpt.py --checkpoint checkpoints/best_model.pt \
  --output-dir evaluation/phase6_1/baseline
{python} scripts/evaluate_gpt.py --checkpoint checkpoints/last_checkpoint.pt \
  --output-dir evaluation/phase6_1/continued
```

Each command writes JSON, CSV, and Markdown reports. The Phase 6.1 run summary
also records command stdout, stderr, and return codes under
`reports/phase6_1/run_summary.json`.
