# Dataset — Context Length, Deduplication, and Validation

## Task 2: Context length (256 → 1024)

`configs/finetuning.yaml`'s `data.context_length` was `256` while the model's actual
`context_length` (`configs/model.yaml`) is `1024`. Updated to `1024`.

**Code audit — what needed to change:**

| Layer | Needed a change? | Why |
|---|---|---|
| `configs/finetuning.yaml` | **Yes** — changed `256` → `1024` | The only actual defect; everything downstream already reads this value |
| Dataset loader (`src/genpy_llm/instruction_dataset.py`) | No | `InstructionDataset` takes `context_length` as a parameter and already paddens/truncates/masks correctly for *any* value — verified by constructing it at both 256 and 1024 against the real dataset |
| Sequence packing | N/A | SFT does not use sequence packing — each example is padded to a fixed length independently. Packing (`SequencePacker`) is used by the pretraining/CPT pipeline over the Final Corpus, a separate, already-verified-correct code path this task does not touch |
| Padding | No | `input_ids.extend([pad_token_id] * pad)` already keys off `self.context_length`, which now receives 1024 |
| Attention mask | No | `attention.extend([0] * pad)` already keys off the same padding computation |
| Loss mask | No | `labels.extend([-100] * pad)` already keys off the same padding computation; prompt-token masking (`ignore_until`) is independent of `context_length` |

`Phase7Trainer._dataset_context_length()` already correctly clamps
`min(config.data.context_length, config.model.context_length)` and falls back to the model's
context length if unset — so the fix is a one-line config change, not a code change.

### Verification: no (avoidable) truncation

Ran `InstructionDataset` over the (deduplicated — see Task 3 below) dataset at both values:

| context_length | Usable | Skipped | Truncated | Truncation rate |
|---|---|---|---|---|
| 256 (old) | 41,681 | 3,891 | 6,379 | 15.30% |
| **1024 (new)** | **45,190** | **382** | **872** | **1.93%** |

The remaining 1.93% at 1024 is **not a bug** — those are examples whose formatted conversation
(prompt + answer + special tokens) genuinely exceeds 1024 tokens, the model's real capacity; they
are still truncated with a forced trailing `<eos>` exactly as designed, just far less often.
15.3% → 1.93% is a real, verified reduction; zero avoidable truncation remains.

### Verification: memory usage

Measured empirically on this machine (peak process RSS, `resource.getrusage(...).ru_maxrss`, one
forward+backward pass, model from `configs/model.yaml`, 35,823,616 parameters / ~143 MB fp32):

| context_length | batch_size | Peak RSS | Delta over baseline (model loaded, no batch) |
|---|---|---|---|
| 256 | 1 | 833.6 MB | +400.2 MB |
| **1024 (configured batch_size)** | **1** | **1,664.9 MB** | **+1,231.6 MB** |
| 1024 | 2 | 2,194.7 MB | +1,761.3 MB |
| 1024 | 4 | 3,352.3 MB | +2,919.0 MB |

At the currently configured `batch_size: 1`, moving to `context_length: 1024` costs roughly
**+830 MB** of peak activation/gradient memory versus 256 — well within the budget of any modern
machine (this model is 35.8M parameters; even the batch_size=4 case stays under 3.5 GB). No
`batch_size` change is required for the next SFT run to fit comfortably; this was not requested and
was not made.

## Task 3: Deduplication

Built `src/genpy_llm/sft_dataset_cleaning.py` (new module) and
`scripts/prepare_sft_dataset.py` (new script) to find and remove exact duplicates from
`data/fine_tuning/train.jsonl` without touching the original file. Output written to
`data/fine_tuning/train.deduplicated.jsonl`; `configs/finetuning.yaml`'s `data.train_path` now
points at it.

### What was found

| Check | Groups | Records involved (sum of group sizes) | Excess if collapsed to 1 per group (group size − 1, summed) |
|---|---|---|---|
| Duplicate **instructions** (same instruction text, ≥2 records) | 2,367 | 12,665 | 10,298 |
| Duplicate **outputs** (same output text, ≥2 records) | 7,209 | 21,825 | 14,616 |
| Duplicate **(instruction, input, output) pairs** (fully identical records) | **0** | **0** | **0** |

The "excess if collapsed" column is what `reports/training_pipeline/dedup_run.json` reports as
`duplicate_instruction_records` (10,298) and `duplicate_output_records` (14,616) — i.e., how many
records *would* be removed if deduplicating on that field alone. The pair-level check is the one
actually used to remove records (0, for the reason explained below), so `dedup_run.json`'s
`duplicate_pair_records_removed` is 0 and matches `new_size == original_size`.

### Original size, duplicates removed, new size

```
Original size:        45,572
Duplicates removed:        0
New size:              45,572
```

### Why the forensic audit's "32% duplicate outputs" does not mean 32% redundant training pairs

The audit's 14,616-record / 32.07% figure is **numerically reproduced here exactly**
(21,825 records across 7,209 duplicate-output groups → 21,825 − 7,209 = 14,616 removable if
deduplicating by output text alone). But cross-referencing against instructions shows why removing
on that basis would be wrong:

```
top duplicated output (appears 427 times):
  "From the visible AST, the structural time bound is O(n), based on a maximum
   visible loop nesting depth of 1. ..."
paired with 355 DISTINCT instructions, e.g.:
  - "Analyze the time and auxiliary-space complexity of `PydanticModelTransformer.adjust_validators`."
  - "Analyze the time and auxiliary-space complexity of `PromptBase.__call__`."
  - "Analyze the time and auxiliary-space complexity of `test_async_for`."
```

These are 355 **legitimately different training examples** (different function/class named in each
instruction) that happen to share a generic, template-derived answer because the complexity
analysis only depends on AST shape, not the symbol's name. Removing them because their *output*
text matches would delete distinct instruction→behavior associations and **change the dataset's
semantic content** — exactly what the task instructs not to do. The same reasoning applies to the
2,367 duplicate-instruction groups (same instruction, different output — e.g. two different correct
implementations for the same generic instruction).

**Conclusion: this dataset contains zero exact-duplicate training pairs.** Its real, separate
quality issue — heavy reliance on a small number of generic templates for certain categories,
especially `complexity_analysis` — is a *diversity/information-density* problem, not a
*duplication* problem, and is not fixable by exact-match deduplication without violating the
semantic-preservation constraint. It's documented here as a known limitation for future dataset
regeneration, out of scope for "remove exact duplicates."

### Category statistics (before = after, since 0 records were removed)

| Category | Count |
|---|---|
| code_completion | 7,492 |
| explanation | 7,455 |
| code_generation | 7,324 |
| refactoring | 6,540 |
| api_usage | 4,788 |
| type_hints | 3,485 |
| bug_fixing | 3,482 |
| unit_testing | 2,238 |
| documentation | 1,536 |
| complexity_analysis | 1,228 |
| optimization | 4 |

## Task 4: Dataset validation

`analyze_sft_dataset()` (in `sft_dataset_cleaning.py`) over `data/fine_tuning/train.jsonl`:

| Metric | Value |
|---|---|
| Total non-blank JSONL lines | 45,572 |
| Usable records (non-empty instruction and output) | 45,572 |
| Broken JSON lines | 0 |
| Empty instructions | 0 |
| Empty outputs | 0 |
| Malformed records (non-string instruction/output) | 0 |

**Length statistics (characters):**

| Field | Mean | Median | Min | Max |
|---|---|---|---|---|
| Instruction | 81.0 | 64.0 | 8 | 7,998 |
| Output | 477.7 | 194.5 | 1 | 68,699 |

**Category distribution:** see table above.

**Truncated samples:** see the context-length table above (1.93% at the corrected `context_length:
1024`, tokenizer- and template-aware, computed via the real production `InstructionDataset`).

**Malformed conversations:** 0 — every record has a non-empty string `instruction` and `output`;
`input` defaults to `""` when absent (by design, matching `InstructionRecord`'s optional-input
schema) and is never required to be non-empty.

**Cross-split leakage check** (not explicitly requested, but relevant to dataset integrity):
`record_id` is disjoint across `train.jsonl` (45,572), `validation.jsonl` (5,824), and
`test.jsonl` (5,551) — 0 overlapping IDs in any pair. A minor 140-record (instruction, output)
content overlap exists between train and validation (generic templated answers coincidentally
appearing in both splits), consistent with the templating issue described above; this does not
affect the exact-duplicate-removal scope of Task 3 and was left as-is.

## Files modified / created

- `configs/finetuning.yaml` — `context_length: 256 → 1024`; `train_path` now points to
  `data/fine_tuning/train.deduplicated.jsonl`.
- `src/genpy_llm/sft_dataset_cleaning.py` — **new**: loading, validation, duplicate-finding,
  deduplication, and writing for Alpaca-style SFT JSONL data.
- `scripts/prepare_sft_dataset.py` — **new**: CLI that runs the above against the real dataset and
  writes `reports/training_pipeline/dedup_run.json`.
- `data/fine_tuning/train.deduplicated.jsonl` — **new**: deduplicated dataset (identical content to
  `train.jsonl` in this case, since 0 duplicates were found; the pipeline is now wired to use
  whichever file `prepare_sft_dataset.py` produces going forward).
