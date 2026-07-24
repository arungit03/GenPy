# GenPy Model Forensic Audit — Summary

**Scope:** read-only investigation into why GenPy generates short, low-quality, or garbled Python
responses. No files were retrained, no checkpoints or tokenizer were modified. This audit
supersedes the prior inference-pipeline audit's conclusion: that audit fixed real decoding bugs
(early EOS, no sampling, no repetition penalty) but those were **not** the primary cause. The
primary cause is upstream, in training.

## Headline finding

**The pretraining base checkpoint itself was already producing degenerate output** — before SFT,
before continued pretraining, before any decoding settings were touched. Fed the plain prompt
`"Write bubble sort."` at low-temperature decoding, the **base checkpoint** (step 5000) generates:

```
append(f)
        assert len(f, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b,
```

This is a collapsed, repeating degenerate loop — not a fine-tuning artifact. Continued pretraining
(40 steps) and SFT (≤2000 steps) inherited this weak base and could not repair it; SFT additionally
learned a template-specific "stop after a stub" behavior from severely limited, low-quality,
truncated training exposure.

## Ranked root causes (full detail in `root_cause.md`)

| # | Cause | Severity | One-line evidence |
|---|---|---|---|
| 1 | Base + continued-pretraining model is severely undertrained | **Critical** | 5000 pretraining steps × ~2500 tokens/step ≈ 12.5M tokens consumed — **2.4% of one epoch** of the 527M-token Final Corpus; CPT added only 40 more steps (~20K tokens) |
| 2 | SFT ran for a negligible fraction of one epoch, twice interrupted | **Critical** | Checkpoint `global_step=1900/2000` vs. 45,572 records needed for 1 epoch at batch_size=1 → **model saw 4.4% of the SFT dataset**, each example exactly once; training logs show two separate process starts (a 3.48-hour gap, second run starting at step 1 again) |
| 3 | SFT learning-rate scheduler bug pins LR near its floor | **High** | `configs/finetuning.yaml` leaves `max_steps: null`; `fine_tuning.py` computes the scheduler's `max_steps = max_steps or epochs = 1`, so the cosine schedule fully decays after step 1 — logged LR stays at ≈5.0–5.1e-6 (≈10% of the configured 5e-5 peak) for essentially the whole run |
| 4 | SFT dataset has heavy templated duplication and stub answers | **High** | 32.1% of all 45,572 outputs are exact duplicates; the `complexity_analysis` category (1,228 records) has only **14 unique outputs (98.9% duplicated)**; 5.83% of all outputs are trivial `pass`-only stubs; 22.3% of outputs are not valid Python at all |
| 5 | SFT `context_length: 256` truncates answers the model's 1024 context could hold | **Medium** | At context_length=256, 15.3% of usable examples are truncated with a forcibly inserted `<eos>` mid-answer, vs. 1.9% at the model's real context_length=1024 |
| 6 | Small model capacity (35.8M parameters) | **Contextual** | Sets a ceiling on achievable quality; does not by itself explain garbage/repetition — causes 1–5 do |
| — | Decoding pipeline (EOS/sampling/repetition penalty) | Already fixed, **not primary** | Even with the fix applied, 100-prompt generation distribution is 63% partial, 28% hallucination/off-topic, only 9% correctly structured — the bottleneck is upstream |
| — | Prompt-template mismatch (train vs. SFT vs. API vs. chat) | **Ruled out** | Byte-for-byte identical across all four call sites |
| — | Loss-masking / label correctness | **Ruled out** | 0/300 sampled records show prompt/full tokenization prefix mismatch; 0 zero-label examples |
| — | Tokenizer (EOS/PAD/UNK/whitespace) | **Ruled out** | Correct special-token IDs; perfect roundtrip on tabs, indentation, blank lines; no silent BOS/EOS insertion |

## Checkpoint comparison — which checkpoint introduced the degradation?

**None did — it was never fixed.** All three checkpoints were run on the same 4 prompts, both as
plain completions and through the chat template:

- **Base** and **continued**: plain-prompt completions are already degenerate (`"""` loops,
  `self.append(self.append(...` loops); chat-templated completions are **empty** (the model has
  never seen the `<|system|>/<|user|>/<|assistant|>` markers meaningfully and cannot continue past
  them).
- **SFT**: plain-prompt completions are *also* degenerate in the same way (`c.c.c.c...`,
  `"items": "items": "items"...`); chat-templated completions are the reported short stubs
  (`def f(): pass`) or off-topic test boilerplate.

SFT's only measurable contribution was making the chat-template pathway emit *something* instead
of nothing — a side effect of exposure to ~2000 template-formatted examples, not a quality
improvement. The underlying language-modeling weakness is present at the base checkpoint and
propagates unchanged through continued pretraining and SFT.

## What this means

Every finding in the prior inference-pipeline audit was real and worth keeping (min_new_tokens,
sampling, repetition penalty all measurably help). But no combination of decoding-time fixes can
produce "complete Python code when capable" from a model whose training never gave it that
capability. The fix has to happen in training: complete a real pretraining/CPT run against the
Final Corpus, complete at least one full, uninterrupted SFT epoch at the intended learning rate,
deduplicate and clean the SFT dataset, and raise `context_length` to the model's real 1024 — none
of which this audit performed, per instructions.

See `dataset_analysis.md`, `training_analysis.md`, `generation_analysis.md`, and `root_cause.md`
for full evidence, statistics, and every file inspected.
