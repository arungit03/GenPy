# Continued Pretraining (CPT)

Continued pretraining resumes an existing GenPy GPT checkpoint and keeps training it on the Final Corpus (`python_corpus/final_corpus/packed/`). It never starts from random initialization, and it reuses the existing model, tokenizer, dataset loader (`PackedSequenceDataset`), training loop (`Phase6Trainer`), and checkpoint format (`save_checkpoint`/`load_checkpoint`) unchanged.

## Pipeline

1. Validate readiness: the Final Corpus shard index exists, its `tokenizer_sha256` matches `data/tokenizer/tokenizer.json`, its `sequence_length` matches the configured value, and the source checkpoint's tokenizer hash matches.
2. Resolve the source checkpoint (`--resume latest` or an explicit path).
3. Restore model weights, optimizer state, gradient-scaler state, RNG state, epoch, and global step from the checkpoint.
4. Build a fresh warmup + cosine-decay learning-rate schedule for the CPT leg (leg-relative, so warmup actually happens even when the global step is already large). When resuming an interrupted CPT run, the leg schedule continues exactly where it stopped.
5. Train for `max_steps` additional optimizer steps (bounded by `epochs`), with gradient accumulation, gradient clipping, optional mixed precision, periodic validation (loss + perplexity), periodic checkpoints, optional early stopping, and structured logging.
6. After training, benchmark the previous checkpoint against the new one (validation loss, perplexity, generation checks) and write comparison reports.

## Resume

- `--resume latest` picks, in order: the newest `checkpoints/continued_pretraining/checkpoint_step_*/model.pt`, then `checkpoints/continued_pretraining/last_checkpoint.pt`, then the newest base checkpoint in `checkpoints/` (`last_checkpoint.pt` or `step_*.pt`).
- `--resume <path>` accepts either a checkpoint `.pt` file or a `checkpoint_step_*` directory (its `model.pt` is used).
- Resuming a checkpoint written by CPT continues the same leg exactly: `start_step`, target step, optimizer state, scheduler position, early-stopping state, and RNG state are all restored, so an interrupted run loses nothing.
- Resuming a base (Phase 6/SFT-era) checkpoint starts a new CPT leg at that checkpoint's global step.

## Configuration

Default config: `configs/continued_pretraining.yaml`.

| Key | Meaning |
| --- | --- |
| `training.learning_rate` | Peak learning rate for the CPT leg (null inherits `configs/optimizer.yaml`) |
| `training.batch_size` | Micro-batch size (null inherits `configs/training.yaml`) |
| `training.gradient_accumulation_steps` | Micro-batches per optimizer step |
| `training.epochs` | Maximum passes over the corpus for this leg |
| `training.max_steps` | Additional optimizer steps for this leg |
| `training.checkpoint_interval_steps` | Steps between checkpoints |
| `training.validation_interval_steps` | Steps between validation runs |
| `training.log_interval_steps` | Steps between console log lines |
| `training.warmup_steps` | Linear warmup steps at the start of the leg |
| `training.weight_decay` | AdamW weight decay (null inherits) |
| `training.sequence_length` | Packed sequence length; must equal model context + 1 (1025) |
| `training.device` | `auto`, `cpu`, `cuda`, or `mps` |
| `training.precision` | `fp32`, `fp16`, or `bf16` (falls back safely per device) |
| `training.keep_last_checkpoints` | `checkpoint_step_*` directories kept after rotation |
| `early_stopping.*` | Optional: `enabled`, `patience`, `min_improvement` |
| `benchmark.*` | Post-training before/after comparison settings |

## CLI

```bash
python3 scripts/continued_pretraining.py \
    --config configs/continued_pretraining.yaml \
    --resume latest
```

- `--resume latest` or `--resume <checkpoint path>`
- `--max-steps N` overrides the leg length
- `--device`, `--skip-benchmark`

## Expected Outputs

```text
checkpoints/continued_pretraining/
    checkpoint_step_xxxxx/
        model.pt            # full checkpoint (existing GenPy format: model, optimizer,
                            # scheduler, scaler, RNG, metadata) — resumable on its own
        optimizer.pt        # optimizer state dict
        scheduler.pt        # scheduler state dict
        trainer_state.json  # step, epoch, losses, best metric, early-stopping state
        config.json         # resolved CPT configuration
    last_checkpoint.pt      # copy of the newest model.pt (canonical single-file form)
    best_checkpoint.pt      # copy at the best validation loss of the leg

reports/continued_pretraining/
    training_log.csv / training_log.json / training_curves.json
    checkpoint_history.json
    summary.md
    comparison_report.json / comparison_report.md   # previous vs. new checkpoint
    training_metrics.jsonl                          # per-step and validation records

logs/continued_pretraining.jsonl
```

Because `model.pt` (and its `last_checkpoint.pt` copy) uses the unchanged GenPy checkpoint format, every downstream consumer — evaluation, LoRA, quantization, the API — loads CPT checkpoints without modification.

## Logging

Each step records loss, learning rate, tokens/sec, examples/sec, epoch, global step, gradient norm, GPU memory, and estimated remaining time; validation records loss and perplexity; every checkpoint save is logged with its path and size.

## Training Notes

- `max_steps` counts *additional* optimizer steps for the leg, not an absolute target; the run trains from `start_step` to `start_step + max_steps`.
- Validation uses a held-out fraction of the packed sequences (`validation_fraction`), bounded to `validation_steps` batches per evaluation.
- Loss spikes to NaN/inf abort the run before a corrupt checkpoint can be written.
