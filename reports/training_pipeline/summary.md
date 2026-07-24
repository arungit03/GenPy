# Training Pipeline Remediation — Summary

Fixes every confirmed issue from the forensic audit (`reports/model_audit/`) so the **next**
continued-pretraining and SFT runs are technically correct. Nothing was retrained; no checkpoints,
tokenizer, or benchmark results were touched.

## What was fixed

| # | Audit finding | Fix | Detail |
|---|---|---|---|
| 3 (scheduler bug) | `max_steps = config.training.max_steps or max(1, epochs)` collapsed to `1`, pinning LR at ~10% of peak (5.09e-6 vs 5e-5) for the whole SFT run | Added `compute_scheduler_total_steps()` (derives steps from dataset size × epochs ÷ batch size ÷ grad-accum, matching the already-correct LoRA formula); fixed `Phase7Trainer` to use it; refactored `Phase9LoRATrainer` onto the same shared function | `scheduler.md` |
| 5 (context length) | SFT `context_length: 256` vs. model's real `1024`, truncating 15.3% of examples with a forced mid-answer `<eos>` | `configs/finetuning.yaml`: `context_length: 256 → 1024`. No dataset-loader/padding/mask code changes were needed — already correctly parametrized | `dataset.md` |
| 4 (dataset duplication) | "≈32% duplicated samples" | Built a dedup/validation module + script; found the real picture is more precise than the audit's headline number: **0 exact (instruction, input, output) duplicates** — the 32% figure is legitimate instructions sharing a generic templated output (e.g. 355 different `complexity_analysis` instructions correctly sharing one boilerplate answer), which removal-by-output-alone would have wrongly deleted | `dataset.md` |
| 1, 2 (undertraining, interrupted SFT) | Base/CPT saw 2.37% of one corpus epoch; SFT saw 4.4% of one epoch across two interrupted runs | **Not retrained, per instructions.** The scheduler and context-length fixes mean the *next* full run will behave correctly; actually completing that run is out of this task's scope (no retraining) | `scheduler.md`, `dataset.md` |

## Every modified file

- `src/genpy_llm/pretraining.py` — added `compute_scheduler_total_steps()`.
- `src/genpy_llm/fine_tuning.py` — `Phase7Trainer` now computes scheduler `max_steps` correctly
  from dataset size/batch/accumulation/epochs instead of falling back to `epochs` alone.
- `src/genpy_llm/lora_training.py` — refactored to use the same shared helper (no behavior change,
  it was already correct).
- `configs/finetuning.yaml` — `context_length: 256 → 1024`; `train_path` now points to the
  deduplicated dataset file.

## Every new file

- `src/genpy_llm/sft_dataset_cleaning.py` — SFT dataset loading, validation, duplicate detection
  (instructions/outputs/pairs), deduplication, and writing.
- `scripts/prepare_sft_dataset.py` — CLI that runs the above against the real dataset.
- `data/fine_tuning/train.deduplicated.jsonl` — output of the above (identical to `train.jsonl`
  here: 0 duplicates found; original file untouched).
- `tests/test_training_pipeline_fixes.py` — 21 new tests across scheduler, context length,
  deduplication, dataset validation, and resume.
- `reports/training_pipeline/{scheduler,dataset,validation,summary}.md`,
  `reports/training_pipeline/dedup_run.json` — this report set.

## Scheduler fix, in one line

`max_steps` is no longer a proxy for `epochs`; it's computed from how many optimizer steps a full
run actually takes. Verified: the old formula gave `1` for the real SFT config; the new formula
gives `45,572` (dataset=45,572, batch=1, accum=1, epochs=1) — and a simulated LR trace with the
fixed scheduler now decays smoothly from `4.9999e-5` (step 1) through `2.75e-5` (the true midpoint,
step 22,786) down to the `5e-6` floor only at the final step (45,572), instead of hitting that floor
after step 1.

## Dataset statistics: before vs. after deduplication

| Metric | Before | After |
|---|---|---|
| Total records | 45,572 | 45,572 |
| Duplicate (instruction, input, output) pairs removed | — | **0** |
| Duplicate instruction groups (informational, not removed) | 2,367 groups / 12,665 records | unchanged |
| Duplicate output groups (informational, not removed) | 7,209 groups / 21,825 records | unchanged |
| Empty / broken / malformed records | 0 / 0 / 0 | 0 / 0 / 0 |
| Truncated at context_length=256 | 15.30% | n/a (config now uses 1024) |
| Truncated at context_length=1024 | 1.93% | 1.93% (unchanged by dedup, since 0 records were removed) |

**Category distribution (unchanged, since nothing was removed):** code_completion 7,492 ·
explanation 7,455 · code_generation 7,324 · refactoring 6,540 · api_usage 4,788 · type_hints 3,485 ·
bug_fixing 3,482 · unit_testing 2,238 · documentation 1,536 · complexity_analysis 1,228 ·
optimization 4.

## Context-length verification

- **No avoidable truncation**: 15.30% → 1.93% (the residual 1.93% is inherent — those examples
  genuinely exceed 1,024 tokens, the model's real capacity).
- **Memory**: empirically measured (peak process RSS) at `batch_size: 1` (the configured value):
  256 → 833.6 MB peak, 1024 → 1,664.9 MB peak (+831 MB). Comfortably safe for this 35.8M-parameter
  model on any modern machine; no batch-size change required or made.

## Training pipeline validation results

Gradient accumulation, optimizer state, scheduler resume, checkpoint loading/saving, mixed
precision, gradient clipping, and validation frequency: **all verified correct**, each backed by
either code inspection of the shared `checkpointing.py`/`performance.py` utilities (already
hardened by earlier CPT work) or by running the existing `tests/test_phase7_finetuning.py`
end-to-end train→save→resume test. One gap found: **Phase 7 SFT has no early-stopping mechanism**
(unlike CPT and LoRA); this was not among the audit's 5 confirmed root causes and was not added,
per the "fix every *confirmed* issue" scope — flagged as a recommendation in `validation.md`.

## Quality gates

- **Ruff**: `All checks passed!` (whole repo).
- **pytest**: **803 passed** (782 prior + 21 new), 18 skipped, **0 failed** — no regressions,
  including targeted re-runs of `test_phase7_finetuning.py`, `test_lora.py`,
  `test_phase6_pretraining.py`, and `test_cpt.py` (the three other files touched or scheduler-
  adjacent).

## Readiness confirmation

The training pipeline is now ready for:
1. **A complete continued-pretraining run** against the full Final Corpus (unaffected by this
   remediation — CPT's scheduler was already correct; simply needs to actually run to completion,
   which is a retraining action outside this task's scope).
2. **A full, uninterrupted SFT run** — the learning-rate schedule will now behave correctly for the
   entire run regardless of how many steps it takes; `context_length: 1024` means the model sees
   full, untruncated answers; the dataset is confirmed to contain zero exact-duplicate training
   pairs; every other training mechanic (gradient accumulation, checkpointing, mixed precision,
   clipping, validation) was independently verified working.

No retraining, fine-tuning, checkpoint modification, tokenizer modification, or benchmark-result
modification was performed — only the training pipeline itself was corrected.
