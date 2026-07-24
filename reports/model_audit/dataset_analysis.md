# SFT Dataset Analysis

**File inspected:** `data/fine_tuning/train.jsonl` (45,572 records, schema: `instruction`,
`input`, `output`, `category`, `deduplication_hash`, `provenance`, `record_id`, `schema_version`).

## 1. Size and length statistics

| Metric | Value |
|---|---|
| Total samples | 45,572 |
| Output length — min | 1 char |
| Output length — max | 68,699 chars |
| Output length — mean | 477.7 chars |
| Output length — median | 194.5 chars |
| Output length — p25 | 94 chars |
| Output length — p75 | 397 chars |
| Output length — p95 | 1,685 chars |
| Empty outputs | 0 (0.00%) |

Most answers are *not* tiny stubs by raw length (median 194.5 chars ≈ 30–40 tokens is a short but
non-trivial function). The stub problem is concentrated, not universal — see below.

## 2. Code validity, stubs, duplicates, truncation

| Check | Count | Rate |
|---|---|---|
| Parses as valid Python (`ast.parse`) | 35,421 | 77.73% |
| — of which trivial stubs (every function body is only `pass`/docstring) | 2,659 | 5.83% of all |
| Does not parse as valid Python (prose, broken code) | 10,151 | 22.27% |
| — of which look truncated (unbalanced brackets/quotes) | 19 | 0.04% |
| Exact duplicate outputs (post-dedup would remove) | 14,616 | 32.07% |

22.27% non-parsing is expected in part — `explanation`, `documentation`, and
`complexity_analysis` categories are prose by design, not code. But the duplicate rate is a defect
regardless of category.

## 3. Category breakdown and duplication hot spots

| Category | Count | Duplicate outputs removed if deduped | Duplicate rate |
|---|---|---|---|
| code_completion | 7,492 | 112 | 1.5% |
| explanation | 7,455 | 1,212 | 16.3% |
| code_generation | 7,324 | 42 | 0.6% |
| refactoring | 6,540 | 95 | 1.5% |
| api_usage | 4,788 | 17 | 0.4% |
| type_hints | 3,485 | 52 | 1.5% |
| bug_fixing | 3,482 | 9 | 0.3% |
| unit_testing | 2,238 | 1 | 0.0% |
| documentation | 1,536 | 84 | 5.5% |
| **complexity_analysis** | **1,228** | **1,214** | **98.9%** |
| optimization | 4 | 0 | 0.0% |

`complexity_analysis` is almost entirely templated: **1,228 records collapse to just 14 unique
outputs.** Example of the dominant template (appears 427 times):

> "From the visible AST, the structural time bound is O(n), based on a maximum visible loop
> nesting depth of 1. ..."

`explanation`'s top duplicate (69 occurrences):

> "`f` is a Python function. Perform the operations defined by the function body."

These are auto-generated, generic, low-information answers that teach the model a
fill-in-the-template response style rather than real reasoning — and because they're duplicated
so heavily, they are over-represented relative to their information content in whatever fraction
of the dataset SFT actually sees (see `training_analysis.md`: only 4.4% of the dataset was seen at
all, so which ~2,000 examples were drawn matters a great deal, and templated categories inflate
the chance of the model latching onto stub/templated patterns).

## 4. Manual quality categorization (500 random samples, seed=7)

Heuristic categorization (parses as Python + non-trivial body + `def`/`class` present + ≥20 words
→ Good; templated-prose or too-short → Poor; empty → Broken):

| Category | Count | % |
|---|---|---|
| Excellent | 0 | 0.0% |
| Good | 203 | 40.6% |
| Average | 274 | 54.8% |
| Poor | 23 | 4.6% |
| Broken | 0 | 0.0% |

Sub-issues found within the 500-sample pull:
- **Templated prose**: 88/500 (17.6%) — matches the `"`X` is a Python function..."` pattern.
- **Trivial stub** (`pass`-only bodies): 17/500 (3.4%).
- **Too short** (<8 words for a prose-expected category): 6/500 (1.2%).
- **Hallucinations**: not separately measured here (see `generation_analysis.md` for
  generation-time hallucination rate); dataset-level hallucination would require per-sample ground
  truth verification, out of scope for a read-only audit at this volume.
- **Truncated samples**: 19/45,572 (0.04%) show unbalanced brackets/quotes suggestive of
  mid-generation truncation at *data-collection* time — separate from the *training-time*
  truncation caused by `context_length: 256` (quantified in `training_analysis.md`).
- **Instruction/output relevance**: of 494/500 samples whose instruction names a specific
  identifier (e.g. `` `reverse_string` ``), 469 (94.9%) mention that identifier in the output —
  i.e., instruction-following at the *dataset* level is generally good; degradation happens at
  training and inference time, not because the dataset teaches wrong associations.

**Example — Good:**
```
IN:  Complete the missing body of `test_digestauth_401_only_sent_once.digest_failed_response_handler`.
OUT: def digest_failed_response_handler(sock):
        # Respond to initial GET with a challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert requ...
```

**Example — Poor:**
```
IN:  Fix the bug in the provided implementation of `test_get_name_with_age_pass_int`.
OUT: def test_get_name_with_age_pass_int():
    assert get_name_with_age("John", 30) == "John is this old: 30"
```

## 5. Verdict

The dataset is not broken, but it is **not clean enough to survive a training run that only sees
4.4% of it once**: roughly a third is duplicated, one entire category is 98.9% templated
boilerplate, and ~6% of code answers are trivial stubs. In a full, multi-epoch, deduplicated
training run these issues would be diluted by exposure to the genuinely good 40.6%+ "Good"-rated
majority. In the run that actually happened, they were not diluted — they were part of whatever
narrow, non-representative slice the interrupted run happened to see.
