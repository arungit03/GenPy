# Generation Analysis

**Method:** loaded the currently-deployed checkpoint (`checkpoints/fine_tuned/best_checkpoint.pt`,
the same one the API serves) with the *already-fixed* decoding settings from the prior inference
audit (`do_sample=True`, `temperature=0.7`, `top_p=0.95`, `repetition_penalty=1.15`,
`min_new_tokens=24`, `max_new_tokens=96`). Ran the first 100 prompts from the benchmark's 21-category
Python prompt set (`src/genpy_llm/benchmark_prompts.py`), each wrapped in the exact chat template
used by the API. No weights or checkpoints were modified; this is pure inference.

## 1. Generation distribution (100 prompts)

| Category | Count | % |
|---|---|---|
| Partial (parses or code-like, but incomplete/wrong) | 63 | 63.0% |
| Hallucination (off-topic, non-code prose, unrelated boilerplate) | 28 | 28.0% |
| Correct structure (parses, non-trivial body, no more than 2 `pass`/`...`) | 9 | 9.0% |
| Stub (≤6 tokens before stopping) | 0 | 0.0% |
| Repetition (bigram-repetition ratio > 0.5) | 0 | 0.0% |
| EOS-immediate (empty output) | 0 | 0.0% |

**Reading this correctly:** the decoding fix from the prior audit worked exactly as intended —
zero outputs are now empty, trivially short, or degenerate-repetition loops within this 100-prompt
run (all three of those categories, which dominated the *pre-fix* behavior, are now 0%). But fixing
decoding only moved the failure mode downstream: 91% of outputs are still "partial" or outright
"hallucination" — plausible-looking Python-ish text that doesn't implement what was asked. This is
a training-capability ceiling, not a decoding artifact.

**Examples:**

- *Partial* — prompt "Write a Python function `binary_search(items, target)`...":
  ```python
  def func():
      return 1

      # long 8:
          return (a, 3, 2, 3)
      return 2
  ```
  Syntactically plausible-shaped but has no relation to binary search.

- *Hallucination* — prompt "Write a Python function `is_prime(number)`...":
  ```python
  defrb(self,map: list[str, str = 0) -> int:
          """Return a Python type of the operations.
          if isinstance(1, type is not a bug and int):
  ```
  Malformed signature, nonsensical docstring, no relation to primality.

- *Correct structure* (best of the 100) — prompt "Implement a Python `Queue` class using
  `collections.deque`":
  ```python
  class Model(BaseModel):
      pass

      with pytest.raises(Base):
          print(Integer, data, primary_data=True)
  ```
  Parses, has a non-trivial body — but is a Pydantic/pytest fragment, not a Queue implementation.
  Counted as "correct structure" by the mechanical categorizer (valid Python, no stub markers) but
  is **not actually correct** by task semantics — the categorizer measures structural validity, not
  correctness. Manual review of all 9 "correct_structure" outputs confirms none are genuinely
  correct implementations of their prompts; all are structurally well-formed but semantically
  unrelated fragments echoing patterns from the SFT dataset's testing/API-usage categories.

**Revised, semantically-honest read:** effectively **0% of the 100 generations correctly implement
their prompt.** The 9% "correct structure" bucket reflects the categorizer's necessarily mechanical
definition (parses + non-trivial + few stop-markers), not task success.

## 2. Checkpoint comparison (base / continued / SFT, same 4 prompts, plain + chat-templated)

Full transcript in `training_analysis.md` §8 and raw JSON at the audit's working data; summarized
here:

| Checkpoint | Plain-prompt completion | Chat-templated completion |
|---|---|---|
| base | `"""\n"""\n"""...` (docstring-token loop) / `append(f)\n assert len(f, b, b, b, b...` (degenerate repetition) | **empty** |
| continued | `"""\n"""\n"""...` / `0.0.0.0.0.0...` (degenerate repetition) | **empty** |
| sft | `It accepts \`None\`.\n def __init__...self._dict[str] = self._dict[str]` (degenerate) / `c.c.c.c.c.c...` (degenerate repetition) | `def f():\n pass\n\n def f():\n pass\n pass\n pass` (short stub) |

**Base and continued checkpoints produce empty output on the chat template entirely** — they were
never trained on `<|system|>/<|user|>/<|assistant|>` markers with any real signal, so the
"assistant turn" is undefined behavior for them (the generation loop's `min_new_tokens` floor
suppresses `<eos>`, but the model's own next-marker prediction gets stripped by
`clean_assistant_response`, leaving nothing). **Base and continued checkpoints also produce
degenerate, repetition-collapsed completions on plain prompts with no template involved at all** —
proof the weak, repetitive generation behavior predates any fine-tuning or chat formatting.

SFT's chat-template completions are short but at least non-empty and vaguely code-shaped — the only
measurable effect of SFT visible in this comparison. SFT's plain-prompt completions remain just as
degenerate as base/continued's, confirming SFT did not meaningfully alter the model's underlying
language-modeling ability outside the narrow template context it was (barely) trained on.

## 3. Which checkpoint introduced the degradation?

**None of them — it was present from the base checkpoint onward and was never repaired.** See
`root_cause.md` for the full ranked explanation.
