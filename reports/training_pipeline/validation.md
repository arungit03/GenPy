# Training Pipeline Validation (Task 5)

Read-only verification of the Phase 7 SFT trainer's mechanics (`src/genpy_llm/fine_tuning.py`,
`Phase7Trainer`), cross-checked against the shared, already-hardened `checkpointing.py` used by
every phase (pretraining, CPT, SFT, LoRA). No training was run to completion; the existing
`tests/test_phase7_finetuning.py` end-to-end test (train → save → resume) was executed to confirm
these mechanics work in practice, not just on paper.

| Check | Result | Evidence |
|---|---|---|
| **Gradient accumulation** | Correct | `_train_micro_batch`: loss is scaled by `1 / gradient_accumulation_steps` before `backward()`; `optimizer.step()` / `scheduler.step()` / `zero_grad()` only fire when `micro_step % gradient_accumulation_steps == 0`. Matches the same pattern used (and already verified) in `Phase6Trainer` and `Phase63Trainer`. |
| **Optimizer state** | Correct | `save_checkpoint(...)` stores `optimizer.state_dict()`; `load_checkpoint(..., optimizer=self.optimizer, ...)` restores it, then moves per-parameter tensors to the model's device (`_move_optimizer_state_to_model_device`). `tests/test_phase7_finetuning.py::test_phase7_trainer_checkpoint_resume_evaluation_and_generation` exercises a real save→resume round trip and passes. |
| **Scheduler resume** | Correct | `_resume()` passes `scheduler=self.scheduler` to `load_checkpoint`, which calls `scheduler.load_state_dict(...)`, restoring `step_count` and `base_lrs`. Confirmed the *fixed* scheduler's LR trajectory is smooth and monotonically decaying across the full step range (see `scheduler.md`), so resuming now continues a correct trajectory instead of a collapsed one. |
| **Checkpoint loading** | Correct | Delegates to the shared `checkpointing.load_checkpoint` (format-versioned, validated payload, RNG-state restore) — the same function already exercised extensively by the CPT and benchmark work; not modified here. |
| **Checkpoint saving** | Correct | Delegates to the shared `checkpointing.save_checkpoint` (atomic temp-file + rename); `_save_checkpoint` passes model, optimizer, scheduler, scaler, `model_config`, `vocabulary_metadata`, and `extra_state` — nothing missing. |
| **Mixed precision** | Correct | `create_grad_scaler(self.mixed_precision, self.device)` creates a scaler only for CUDA fp16; `autocast_context(...)` wraps the forward+loss; when a scaler is present, `scale().backward()` → `unscale_()` → clip → `step()` → `update()` in the correct order (unscale *before* computing/clipping the gradient norm, matching PyTorch's documented AMP pattern). |
| **Gradient clipping** | Correct | `_clip_gradients()` uses `torch.nn.utils.clip_grad_norm_` when `max_grad_norm` is configured (`1.0` in `configs/finetuning.yaml`), else reports the raw norm without clipping. Applied after `scaler.unscale_()`, so the clip threshold is compared against true (not loss-scaled) gradients. |
| **Validation frequency** | Correct, and effective | `_should_evaluate()`: fires when `global_step > 0` and `global_step % eval_every_steps == 0`, gated on a validation dataset actually being configured. `configs/finetuning.yaml` sets `validation_path: data/fine_tuning/validation.jsonl` (5,824 records) and `eval_every_steps: 100` / `evaluation_steps: 10` — the forensic audit's own metrics (`metrics/phase7/fine_tuning_metrics.jsonl`) show this already produced 21 real evaluation records with a genuinely decreasing loss (7.87 → 3.71) during the truncated run, confirming the mechanism itself works; it was just never given enough total steps to matter. |
| **Early stopping** | **Not implemented in Phase 7** | Unlike `cpt.py` (`EarlyStoppingConfig`/`EarlyStoppingState`) and `lora_training.py`, `Phase7Trainer` has no early-stopping mechanism — it only tracks `best_validation_loss` for checkpoint selection (`_is_best`), with no patience/stop condition. **This was not among the audit's 5 confirmed root causes** (the SFT run's problem was manual interruption after ~4% of an epoch, not a missing stop condition — early stopping would not have prevented or changed that failure). Per the objective's scope ("fix every *confirmed* training issue," not add new features), this gap is reported here as a recommendation, not fixed. |

## Summary

Every mechanical property Task 5 asks about — gradient accumulation, optimizer/scheduler state
persistence, checkpoint load/save, mixed precision, gradient clipping, and validation frequency —
is implemented correctly and was already working before this remediation; the forensic audit's
findings were about *training completeness and configuration* (undertraining, interrupted runs, the
scheduler `max_steps` bug, dataset duplication claims, context length), not about these mechanics
being broken. The one gap found (no early stopping in SFT) is real but out of the confirmed-issue
scope for this pass.
