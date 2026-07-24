# Root Cause — Ranked

## Primary reason

**GenPy generates poor-quality Python because the model was never trained enough, at any stage,
to acquire the capability — not because of a decoding, prompt-formatting, or tokenizer defect.**
The evidence is that the base pretraining checkpoint, before any fine-tuning or chat formatting
enters the picture, already produces collapsed/repetitive garbage on plain code prompts
(`"""\n"""\n"""...`, `self.append(self.append(self.append(...`). A model with that little language
capacity cannot be made to "generate complete Python code when capable" through prompt or decoding
changes, because it is not, in fact, capable yet — training stopped at roughly 2% of the exposure
a model at this scale needs.

## Ranked causes

### 1. Undertrained base / continued-pretraining model — Critical, Primary
- **Evidence:** 5,000 pretraining steps consumed 12,505,088 tokens against a 527,482,033-token
  corpus — **2.37% of one epoch**. Validation loss was still 3.72 (perplexity ≈41) and still
  descending when training stopped, with `configs/training.yaml`'s own `max_steps: 100000` never
  reached. Continued pretraining added 40 more steps (~20,480 tokens) — statistically negligible.
- **Proves:** base and continued checkpoints generate degenerate, repetition-collapsed text on
  plain, non-templated code prompts (`generation_analysis.md` §2) — this is a raw language-modeling
  deficiency, present before SFT or any chat formatting.
- **Fix (not performed — out of scope):** resume pretraining/CPT with a real step budget against
  the full Final Corpus until validation loss/perplexity plateau at a materially lower value.

### 2. SFT ran for a negligible, interrupted fraction of one epoch — Critical
- **Evidence:** `global_step=1900–2000` of the 45,572 steps one epoch requires at `batch_size: 1`
  — **4.39% of the dataset, each example seen exactly once.** Timestamps show two separate process
  starts (a 3.48-hour gap; the second run restarts from step 1, not resumed) — consistent with the
  process being killed externally each time, not any code-level stop condition.
- **Proves:** SFT's only observable effect is making the chat-template pathway emit *something*
  (short stubs) instead of nothing; it did not meaningfully change the model's plain-prompt
  behavior, which remained as degenerate as the base checkpoint's.
- **Fix (not performed):** run at least one full, uninterrupted epoch (ideally several, with real
  validation) before evaluating quality.

### 3. SFT learning-rate scheduler bug — High
- **Evidence:** `configs/finetuning.yaml` leaves `training.max_steps: null`. In
  `src/genpy_llm/fine_tuning.py`, `Phase7Trainer.__init__` computes
  `max_steps = config.training.max_steps or max(1, config.training.epochs)`, which evaluates to
  `1` — feeding the cosine-warmup scheduler a 1-step horizon. From step 2 onward the schedule is
  fully decayed to its 10% floor. Logged learning rates in
  `metrics/phase7/fine_tuning_metrics.jsonl` never exceed ≈5.09e-6 against a configured peak of
  5e-5 — **the entire run trained at roughly a tenth of its intended learning rate.**
- **Fix (not performed):** derive the scheduler's `max_steps` from `len(train_dataset) // batch_size
  * epochs` (matching what the training loop itself uses to decide when to stop), not from
  `epochs` alone.

### 4. SFT dataset duplication and templated stubs — High
- **Evidence:** 32.1% of all 45,572 outputs are exact duplicates. `complexity_analysis` (1,228
  records) collapses to 14 unique outputs (98.9% duplicated). 5.83% of all outputs are trivial
  `pass`-only stubs. 22.3% of outputs are not valid Python at all (expected in prose categories,
  but not filtered/labeled distinctly for training).
- **Interacts with #2:** because only 4.4% of the dataset was ever seen, whichever narrow,
  non-representative slice was drawn (including possibly a disproportionate share of the templated
  `complexity_analysis`/`explanation` boilerplate) had an outsized influence on what little the
  model learned, with no chance for a full epoch to average it out against the dataset's
  genuinely-good 40.6%+ "Good"-rated majority (see `dataset_analysis.md`).
- **Fix (not performed):** deduplicate exactly and near-exactly, especially the
  `complexity_analysis` category; filter or explicitly separate prose-only categories from
  code-generation categories.

### 5. SFT context_length truncation — Medium
- **Evidence:** `configs/finetuning.yaml` sets `data.context_length: 256` against the model's real
  `context_length: 1024`. At 256, 15.3% of usable examples are truncated with a forcibly-inserted
  `<eos>` mid-answer (vs. 1.9% at 1024) — training the model, for a meaningful minority of its
  already-scarce exposure, that answers end wherever they happen to be cut off.
- **Fix (not performed):** set `data.context_length` to match `configs/model.yaml`'s 1024 (or the
  smallest value that keeps truncation negligible).

### 6. Small parameter count (35.8M) — Contextual, not primary
- Sets a real ceiling on ultimate quality achievable even with perfect training, but does **not**
  explain the specific failure modes observed (empty output, repetition collapse, templated stub
  answers) — those are fully explained by causes 1–5. A well-trained 35.8M model would still likely
  produce simple, sometimes-wrong code — but not the docstring-repetition loops and empty
  completions seen here.

### Ruled out (with evidence)
- **Tokenizer** — correct special-token IDs, perfect whitespace/indentation roundtrip, no silent
  BOS/EOS insertion.
- **Prompt-template mismatch** — training, SFT, API, and the release chat example all produce the
  byte-for-byte identical formatted prompt.
- **Label masking / loss masking** — 0/300 sampled records show prompt-tokenization prefix
  mismatch; 0 zero-label examples; assistant tokens trained, user/system/padding tokens correctly
  masked with `-100`.
- **Decoding pipeline (EOS handling, sampling, repetition penalty)** — already audited and fixed in
  the prior session; confirmed here to be working (0% empty/stub/repetition-collapse outputs in the
  100-prompt distribution) but **not sufficient**, because the underlying capability was never
  trained. This rules out decoding as the *primary* cause, though the earlier fixes remain
  correct and should be kept.

## One-sentence summary

**GenPy's poor output quality is a training-completeness problem, not an inference problem:** the
base model saw 2.4% of one pretraining epoch, continued pretraining added a negligible 40 steps,
and SFT — hit by both a learning-rate scheduler bug and two interrupted runs — saw only 4.4% of its
dataset once, so no stage of training ever gave the model enough signal to reliably produce correct
Python; the previously-fixed decoding pipeline is necessary but cannot compensate for that.
