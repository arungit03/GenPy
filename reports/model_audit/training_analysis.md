# Training Analysis

**Files inspected:** `metrics/training_metrics.jsonl` (Phase 6 pretraining),
`reports/continued_pretraining/training_log.json` (CPT), `metrics/phase7/fine_tuning_metrics.jsonl`
(SFT), `checkpoints/last_checkpoint.pt`, `checkpoints/continued_pretraining/last_checkpoint.pt`,
`checkpoints/fine_tuned/best_checkpoint.pt` (metadata only — weights not modified),
`configs/finetuning.yaml`, `src/genpy_llm/fine_tuning.py`, `src/genpy_llm/instruction_dataset.py`,
`src/genpy_llm/pretraining.py`, `src/genpy_llm/conversation_formatter.py`,
`src/genpy_llm/code_tokenizer.py`.

## 1. Pretraining sufficiency

| Metric | Value |
|---|---|
| Steps completed | 5,000 |
| Total training tokens consumed | 12,505,088 |
| Final Corpus size | ~527,482,033 tokens (53 packed shards) |
| Corpus fraction seen | **2.37%** of one epoch |
| First train loss | 10.416 |
| Last train loss | 3.443 (checkpoint metadata: training_loss=3.443) |
| First validation loss (step 1000) | 4.9539 |
| Last validation loss (step 5000) | 3.7220 (perplexity ≈ 41.3) |
| Continued pretraining steps | +40 (steps 5001–5040) |
| CPT tokens consumed | ~20,480 |

Loss was still descending at the last logged validation point with no sign of convergence — the
run stopped for reasons unrelated to the loss curve (a configured `max_steps: 100000` in
`configs/training.yaml` was never reached; the actual stop was an external interruption, matching
the pattern found in the SFT logs below). **The base model is undertrained by roughly two orders
of magnitude relative to its own corpus.** Continued pretraining's 40 steps / ~20K tokens is
statistically negligible against a 527M-token corpus and does not materially change this.

**Was the model undertrained?** Yes, severely. **Was CPT stopped too early?** Yes — 40 steps is a
smoke-test-sized run, not a training run. **Is validation loss still too high?** Yes — perplexity
≈41 on held-out Final Corpus data means the model is, on average, meaningfully uncertain between
~41 equally-likely next tokens; for reference, a well-trained small code model typically achieves
single-digit-to-low-teens perplexity on in-domain code.

## 2. SFT: was it interrupted?

`configs/finetuning.yaml` sets `epochs: 1`, `max_steps: null`, `batch_size: 1`. With 45,572 records
in `data/fine_tuning/train.jsonl`, **one full epoch requires 45,572 steps.**

| Evidence | Value |
|---|---|
| Best checkpoint (`best_checkpoint.pt`) global_step | 1,900 |
| Last logged training step | 2,000 |
| Records needed for 1 full epoch (batch_size=1) | 45,572 |
| **Fraction of dataset seen** | **4.39%** |

Timestamps in `metrics/phase7/fine_tuning_metrics.jsonl` show the run happened in **two separate
process invocations**:

- Run 1: starts, reaches step 100, timestamp `2026-07-22 22:20:53 UTC`.
- **Gap of 3.48 hours** with no logged steps.
- Run 2: starts over from step 1 (not resumed — `configs/finetuning.yaml` sets `resume: false`) at
  `2026-07-23 01:49:25 UTC`, reaches step 2,000 by `01:56:38 UTC` (≈7 minutes of active training),
  then stops. At the observed throughput (~4.7 steps/sec), a full epoch would take ~2.7 hours; the
  run stopped after ~7 minutes.

Both runs terminated well short of a full epoch, with no `max_steps` or natural end-of-epoch
condition to explain the stop. This is consistent with the process being killed externally
(interactive session ended, `Ctrl+C`, or similar) rather than any code-level stopping condition —
**the SFT model has only ever seen 4.4% of its intended training data, once each, never
completing epoch 1.**

## 3. SFT dataset quality
See `dataset_analysis.md` for full statistics: 32.1% duplicate outputs, one category
(`complexity_analysis`) at 98.9% duplication, 5.83% trivial stub outputs, 22.3% non-Python-parsing
outputs. Combined with only seeing 4.4% of records, the *specific* 2,000 examples drawn (in
whatever shuffle order `seed=42` produced) disproportionately determine what the model learned —
there was no opportunity for repeated exposure across an epoch to average out the noisy/templated
minority.

## 4. Learning-rate scheduler bug

`configs/finetuning.yaml`:
```yaml
training:
  epochs: 1
  max_steps:        # null
learning_rate: 0.00005
```

`src/genpy_llm/fine_tuning.py` (`Phase7Trainer.__init__`):
```python
max_steps = config.training.max_steps or max(1, config.training.epochs)
self.scheduler = CosineWarmupScheduler(self.optimizer, max_steps=max_steps, warmup_steps=..., ...)
```

Because `config.training.max_steps` is `None`, this evaluates to `max_steps = max(1, 1) = 1`. The
cosine-warmup scheduler is therefore configured for a **1-step schedule** — after step 1, its
internal progress is clamped to 1.0 and the cosine factor collapses to the configured floor
(`minimum_learning_rate_ratio: 0.1`, i.e. 10% of peak) for every subsequent step.

**Confirmed empirically** from `metrics/phase7/fine_tuning_metrics.jsonl`: the logged
`learning_rate` field never rises above ~5.09e-6 across all 2,000 steps, against a configured peak
of `5e-5` — **the entire SFT run trained at ≈10% of its intended learning rate.** This is a
genuine, previously-undocumented bug in the Phase 7 config-to-scheduler wiring: `max_steps` (the
*optimizer step budget*, used correctly by the training loop's own stopping check) is silently
reused as the *scheduler's* horizon when unset, instead of being derived from dataset size and
`epochs` as the training loop itself does.

## 5. Training labels / loss masking — verified correct

Checked `src/genpy_llm/instruction_dataset.py::InstructionDataset._encode_record` against 300
random training records:

- **Prompt/full tokenization prefix consistency**: 0/300 mismatches. Tokenizing the prompt alone
  and tokenizing prompt+output produce token sequences where the former is an exact prefix of the
  latter in every sampled case — i.e., no BPE merge crosses the prompt/assistant boundary in a way
  that would corrupt the mask boundary.
- **Assistant tokens trained, user/system tokens masked**: confirmed by construction
  (`labels[:ignore_until] = [-100] * ignore_until` where `ignore_until = len(prompt_ids) - 1`) and
  empirically — 0/300 sampled examples had all-`-100` labels (which would silently contribute zero
  loss and be skipped).
- **Padding labels ignored**: confirmed — `labels.extend([-100] * pad)`.
- **EOS labels present**: confirmed — `<eos>` is appended if the tokenized output doesn't already
  end with one, and is part of the (unmasked) label sequence, so the model is correctly trained to
  predict `<eos>` at the true end of an answer.
- **Trainable label-token counts** (300-sample check, `context_length=256`): min 5, median 47, max
  209 tokens actually contributing to loss per example — reasonable for genuinely short answers,
  but note this is *before* accounting for truncation (next section).

**Verdict: label masking is implemented correctly.** This is not a contributing cause.

## 6. Context-length truncation (real, quantified issue)

`configs/finetuning.yaml` sets `data.context_length: 256`. The model's actual
`context_length` (from `configs/model.yaml`) is **1024**. Running the full 45,572-record dataset
through `InstructionDataset` at both values:

| context_length | Usable | Skipped | Truncated | Truncation rate |
|---|---|---|---|---|
| 256 (as configured) | 41,681 | 3,891 | 6,379 | **15.3%** |
| 1024 (model's real capacity) | 45,190 | 382 | 872 | 1.9% |

When an example is truncated, `InstructionDataset` forcibly overwrites the last surviving token
with `<eos>` (`instruction_dataset.py:124-125`) so the sequence still "ends" cleanly — meaning
15.3% of the examples the (tiny fraction of the) SFT run saw were **not their real answer**, but an
arbitrary mid-answer cut-off relabeled as if it were the correct stopping point. This directly
reinforces "stop generating soon" as a learned association, on top of the already-limited training
signal.

## 7. Tokenizer — verified clean

`data/tokenizer/tokenizer.json` via `CodeTokenizer`:

| Check | Result |
|---|---|
| `vocab_size` | 32,000 |
| `<bos>` id | 2 |
| `<eos>` id | 3 |
| `<pad>` id | 0 |
| `<unk>` id | 1 |
| Special tokens | `<pad>, <unk>, <bos>, <eos>, <mask>, <instruction>, <input>, <output>` |
| Roundtrip: newlines, tabs, nested indentation, blank lines, trailing/leading whitespace | 4/4 exact roundtrip |
| Silent BOS/EOS insertion on plain `encode()` | None — `add_special_tokens` defaults to `False` and is honored |

**Verdict: tokenizer is not a contributing cause.**

## 8. Prompt formatting parity — verified identical

Compared byte-for-byte:

- **Training** (`ConversationTemplate.format_prompt`, used by `InstructionDataset`):
  ```
  <|system|>
  You are GenPy, a Python coding assistant.

  <|user|>
  {instruction}

  <|assistant|>
  ```
- **API** (`api/inference.py::_chat_prompt`): produces the identical string, built from the same
  `configs/finetuning.yaml` `template:` section via `ConversationTemplate.from_mapping`.
- **Release chat example** (`GenPy-v1.0/examples/chat.py::format_turn`): hardcodes the identical
  string with the identical system prompt text.

**Verdict: no prompt-template mismatch.** This was already ruled out in the prior inference audit
and is reconfirmed here.
