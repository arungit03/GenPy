# Phase 6.3 Continued Pretraining

Phase 6.3 resumes GPT pretraining from an existing Phase 6 checkpoint using the
packed Corpus V2 shards produced by Phase 6.2. It never starts from random
initialization and does not modify the Transformer architecture, tokenizer,
vocabulary, LoRA, quantization, API, frontend, or inference stack.

## Readiness Gate

Before training, `scripts/run_phase6_3.py` verifies:

- Corpus V2 manifest, quality report, statistics, document manifest, and shard
  index exist.
- Corpus V2 readiness passed.
- The configured token target was reached.
- Packed shard checksums match the Corpus V2 build fingerprint.
- Corpus tokenizer hash matches the tokenizer file.
- A previous Phase 6 checkpoint exists.
- Checkpoint tokenizer metadata is compatible.

If any check fails, Phase 6.3 aborts before constructing a training loop.

## Run

```bash
python scripts/run_phase6_3.py
```

Useful smoke-test override:

```bash
python scripts/run_phase6_3.py --max-steps 1 --device cpu --skip-benchmark
```

Resume uses `phase6_3.training.source_checkpoint` when set; otherwise it resolves
`checkpoints/last_checkpoint.pt`, then the latest `checkpoints/step_*.pt`.

## Outputs

Checkpoints:

- `checkpoints/pretraining_v2/epoch_001.pt`
- `checkpoints/pretraining_v2/epoch_002.pt`
- `checkpoints/pretraining_v2/last_checkpoint.pt`
- `checkpoints/pretraining_v2/best_checkpoint.pt`

Reports:

- `reports/pretraining_v2/training_log.csv`
- `reports/pretraining_v2/training_log.json`
- `reports/pretraining_v2/training_curves.json`
- `reports/pretraining_v2/summary.md`
- `reports/pretraining_v2/checkpoint_history.json`
- `reports/pretraining_v2/comparison_report.json`
- `reports/pretraining_v2/comparison_report.md`

## Benchmark

After training, Phase 6.3 compares the previous checkpoint and continued
checkpoint on:

- validation loss;
- perplexity;
- static Python syntax correctness;
- Python benchmark score;
- repetition rate;
- generation speed.

Run only the comparison:

```bash
python scripts/benchmark_phase6_3.py
```

## Safety

The trainer aborts on non-finite loss, non-finite gradients, exploding gradients,
corpus mismatch, tokenizer mismatch, missing/corrupt checkpoints, or backend OOM
exceptions. Diagnostics are emitted to stderr and the structured log.
