# Scheduler Fix

## Every scheduler implementation found

| File | Scheduler | `max_steps` source | Status before this fix |
|---|---|---|---|
| `src/genpy_llm/pretraining.py` | `CosineWarmupScheduler` (canonical implementation) | `config.training.max_steps` — a required, explicit YAML field (`configs/training.yaml: pretraining.training.max_steps: 100000`) | **Correct** — no fallback, always explicit |
| `src/genpy_llm/cpt.py` | `CPTCosineScheduler(CosineWarmupScheduler)` | `start_step + config.training.max_steps`, where CPT's `max_steps` is its own required leg-length field | **Correct** — leg-relative, no fallback bug |
| `src/genpy_llm/lora_training.py` | `CosineWarmupScheduler` (Phase 9 LoRA) | `config.training.max_steps or (epochs * len(train_dataset) // batch_size // gradient_accumulation_steps)` | **Already correct** — properly derives steps from dataset size when unset |
| `src/genpy_llm/fine_tuning.py` | `CosineWarmupScheduler` (Phase 7 SFT) | **was:** `config.training.max_steps or max(1, config.training.epochs)` | **Buggy — fixed here** |
| `src/genpy_llm/code_training.py` | custom `build_scheduler()` (Phase 5, legacy, standalone) | `max_steps` passed in explicitly by the caller from a required YAML field (`configs/code_small.yaml`); no null-coalescing fallback | Correct, unaffected — separate legacy code path, not part of the CPT/SFT pipeline this task targets |

## The bug

`src/genpy_llm/fine_tuning.py`, `Phase7Trainer.__init__` (before this fix):

```python
max_steps = config.training.max_steps or max(1, config.training.epochs)
self.scheduler = CosineWarmupScheduler(self.optimizer, max_steps=max_steps, ...)
```

`configs/finetuning.yaml` leaves `training.max_steps` unset (`null`) and `training.epochs: 1`.
`None or max(1, 1)` evaluates to **`1`** — the scheduler is built with a 1-step horizon. From
optimizer step 2 onward, `CosineWarmupScheduler._factor()`'s progress term
`(step - warmup) / (max_steps - warmup)` is clamped to `1.0`, so the cosine factor is permanently
`0` and the learning rate is pinned at its floor (`minimum_learning_rate_ratio: 0.1`, i.e. 10% of
peak) for the rest of the run. This was confirmed in the forensic audit: logged SFT learning rates
never exceeded `5.09e-6` against a configured peak of `5e-5`.

**Root cause:** `epochs` alone was used as a stand-in for "how many optimizer steps will this run
take," which is only true if the dataset has exactly one example per epoch. With 45,572 SFT
records, one epoch is 45,572 steps, not 1.

## The fix

Added `compute_scheduler_total_steps()` to `src/genpy_llm/pretraining.py` (next to the canonical
`CosineWarmupScheduler`, and exported from the module) as the single, tested source of truth for
this computation:

```python
def compute_scheduler_total_steps(
    *, dataset_size: int, batch_size: int, gradient_accumulation_steps: int,
    epochs: int, max_steps: int | None = None,
) -> int:
    if max_steps is not None:
        return int(max_steps)
    return max(1, epochs * dataset_size // batch_size // gradient_accumulation_steps)
```

- An explicit `max_steps` in config always wins (unchanged behavior for anyone who sets it).
- Otherwise, the horizon is derived from **dataset size, batch size, gradient accumulation, and
  epochs** — exactly the four inputs the objective specifies — using the same formula already
  proven correct in `lora_training.py`'s Phase 9 trainer.

**`fine_tuning.py`** (`Phase7Trainer.__init__`) now:
1. Builds `self.train_dataset` (moved earlier in `__init__`, since the scheduler needs its length —
   verified `_datasets()` depends only on `self.tokenizer`/`self.config`, not on the optimizer or
   scheduler, so this reordering is safe).
2. Computes `max_steps = compute_scheduler_total_steps(dataset_size=len(self.train_dataset),
   batch_size=config.data.batch_size, gradient_accumulation_steps=config.training
   .gradient_accumulation_steps, epochs=config.training.epochs, max_steps=config.training
   .max_steps)`.

**`lora_training.py`** (`Phase9LoRATrainer.__init__`) was refactored to call the same shared
function instead of its own inline (already-correct, identical-formula) copy — pure deduplication,
no behavior change: the formula is byte-for-byte the same or with a proper `max_steps or ...`
short-circuit either way as before.

## Mathematical verification

```
old buggy value, epochs=1, max_steps=None:                         1
new value, dataset=45,572, batch=1, accum=1, epochs=1:          45,572   (matches: 45,572 records ÷ 1 ÷ 1 × 1 epoch)
dataset=1,000, batch=4, accum=2, epochs=3:                          375   (= 3 × 1,000 // 4 // 2, exact match with reference lora formula)
explicit max_steps=50 override (any dataset/batch/epoch combo):      50   (explicit config always wins)
dataset_size=1, batch=8, accum=4, epochs=1 (degenerate case):         1   (floors at 1, never 0)
```

All four cases are asserted in `tests/test_training_pipeline_fixes.py::test_compute_scheduler_total_steps_*`.

## Verified: warmup, cosine decay, resume, remaining steps, checkpoint resume

- **Warmup**: `CosineWarmupScheduler._factor()`'s linear-warmup branch
  (`step / warmup_steps` for `step < warmup_steps`) is untouched and was already correct — the bug
  was purely in the *horizon* (`max_steps`) fed to the scheduler, not the warmup or decay math
  themselves.
- **Cosine decay**: `0.5 * (1 + cos(pi * progress))`, floor-adjusted by
  `minimum_learning_rate_ratio` — untouched, already correct; it simply decayed almost instantly
  because `max_steps` was wrong.
- **Resume / checkpoint resume**: `Phase7Trainer._resume()` already correctly calls
  `load_checkpoint(..., scheduler=self.scheduler, ...)`, which restores `step_count` and
  `base_lrs` via `CosineWarmupScheduler.load_state_dict()`. This is unchanged and was already
  correct — resuming a checkpoint continues the LR trajectory from the right step. What changes is
  that the trajectory a resumed run continues is now the *correct* one (a real multi-thousand-step
  cosine schedule) instead of the collapsed one-step schedule.
- **Remaining steps**: for Phase 6/7, "remaining steps" is implicit in the scheduler's own progress
  fraction (`(step − warmup) / (max_steps − warmup)`); CPT (`cpt.py`) additionally computes and logs
  an explicit `remaining_seconds` ETA from `target_step − global_step`, which was already correct
  and is unaffected by this fix (CPT's `max_steps` was never derived from `epochs`).

## Files modified

- `src/genpy_llm/pretraining.py` — added `compute_scheduler_total_steps()`, exported in `__all__`.
- `src/genpy_llm/fine_tuning.py` — fixed `Phase7Trainer.__init__` (reordered dataset construction,
  uses the shared helper).
- `src/genpy_llm/lora_training.py` — refactored to use the same shared helper (no behavior change).
