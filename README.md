# GenPy LLM

GenPy LLM is an educational GPT-style language model project with a Python-code
pretraining pipeline. The current code path includes a byte-level BPE code
tokenizer, gzip JSONL streaming shards, a GPT decoder, mixed-precision training,
checkpointing, code generation, and evaluation tooling.

## Current Stage

Phase 5 adds a production ByteLevel BPE tokenizer trained on the complete
instruction dataset. The model architecture is unchanged.

Completed pieces:

- Base GPT-style decoder architecture
- Code tokenizer training and loading
- Streaming code-shard dataset pipeline
- Mixed-precision code pretraining
- Checkpoint save/load and resume
- Code generation from checkpoints
- Evaluation, loss-history artifacts, and checkpoint reporting
- Supervised code fine-tuning from JSONL, JSON, or TXT datasets
- Policy-controlled GitHub corpus discovery and binary pre-training shards
- Policy-controlled PyPI sdist ingestion and binary pre-training shards
- Final merged pretraining corpus builder with global deduplication and packed
  binary sequence shards
- Official Phase 6 GPT pretraining engine for packed binary corpora
- Phase 6.1 continued pretraining orchestration with corpus readiness gates,
  code/text balance checks, checkpoint resume, and benchmark commands

## GitHub Corpus Builder

Phase 5.5A can discover approved-license public Python repositories, resume
parallel clones, pass their source through the existing corpus validator and
SHA-256 index, and encode it with the Phase 5 tokenizer:

```bash
# Review github_corpus in configs/dataset_pipeline.yaml, then set enabled: true.
export GITHUB_TOKEN=github_pat_...  # optional, recommended for API rate limits
python scripts/build_github_corpus.py
```

The feature is disabled by default so repository and license policy must be
reviewed explicitly. See [GitHub Corpus Builder](docs/github_corpus_builder.md).

## PyPI Corpus Builder

Phase 5.5B discovers approved PyPI packages from ranked lists, keyword searches,
category lists, requirements files, or manual lists. It downloads only source
distributions, verifies published SHA-256 checksums, safely extracts Python,
passes files through the existing collector and Corpus Manager, and encodes them
with the existing 32K tokenizer:

```bash
# Review configs/pypi.yaml, add approved packages, then set enabled: true.
python scripts/build_pypi_corpus.py
```

Downloads, extraction, validation, deduplication, reports, and binary shards are
resumable and deterministic. See [PyPI Corpus Builder](docs/pypi_corpus_builder.md)
for configuration, outputs, approval guidance, and troubleshooting.

## Final Pretraining Corpus

Phase 5.5C merges all approved files already collected into `data/raw`, runs one
global deduplication and validation pass, tokenizes with the existing 32K GenPy
tokenizer, packs fixed-length GPT sequences, and writes Phase 6-ready binary
shards:

```bash
python scripts/build_pretraining_corpus.py
```

Configuration lives in `configs/pretraining.yaml`. Outputs are written to
`data/pretraining/`, including `shard_00000.bin`, `index.json`, `manifest.json`,
`statistics.json`, and `corpus_manifest.jsonl`; reports are written under
`reports/pretraining/`. See
[Pretraining Corpus Builder](docs/pretraining_corpus_builder.md).

## GPT Pretraining

Phase 6 trains the GenPy base GPT model from the final packed binary corpus:

```bash
python scripts/pretrain_gpt.py
```

Configuration is split across `configs/model.yaml`, `configs/training.yaml`,
`configs/optimizer.yaml`, and `configs/generation.yaml`. The trainer supports
packed binary shard loading, deterministic sampling, AdamW, cosine warmup/decay,
gradient accumulation, gradient clipping, AMP precision modes, checkpoint
resume, checkpoint rotation, validation, generated samples, CSV/JSON metrics,
structured logs, and TensorBoard when installed. See
[Phase 6 Pretraining](docs/phase6_pretraining.md).

Precision is configured with `pretraining.training.mixed_precision`, using
`none`, `fp16`, or `bf16`. Legacy `training.precision: fp32` is still accepted
and maps to `mixed_precision: none`. Apple MPS defaults to full precision; `bf16`
requests fall back to full precision with a warning, while `fp16` is used only
when the installed PyTorch/MPS stack supports it.

## Continued Pretraining

Phase 6.1 expands the corpus target to 200M-500M tokens, admits approved
technical text alongside Python code, verifies the packed-corpus balance, resumes
from an existing Phase 6 checkpoint, and can run before/after coding benchmarks:

```bash
python scripts/run_phase6_1.py
```

The default readiness gate refuses to train until the corpus target and balance
requirements are met. See
[Phase 6.1 Continued Pretraining](docs/phase6_1_continued_pretraining.md).

## Corpus V2 Expansion

Phase 6.2 builds a scalable, local-only corpus expansion pipeline for continued
pretraining readiness. It collects approved Python and technical-text sources,
cleans, validates, deduplicates, tokenizes with the existing tokenizer, packs
Phase-6-compatible binary shards, writes reports, and stops before training:

```bash
python scripts/build_corpus_v2.py
python scripts/analyze_corpus_v2.py
```

Configuration lives in `configs/corpus_v2.yaml`. See
[Phase 6.2 Corpus V2](docs/phase6_2_corpus.md).

## Phase 6.3 Continued Pretraining

Phase 6.3 resumes GPT pretraining from the latest Phase 6 checkpoint using the
validated Corpus V2 packed shards. It verifies corpus readiness, tokenizer and
checkpoint compatibility, then writes continued checkpoints under
`checkpoints/pretraining_v2/`:

```bash
python scripts/run_phase6_3.py
python scripts/benchmark_phase6_3.py
```

If Corpus V2 readiness fails, Phase 6.3 aborts before training. See
[Phase 6.3 Continued Pretraining](docs/phase6_3_continued_pretraining.md).

## Supervised Instruction Fine-Tuning

Phase 7 fine-tunes a pretrained GenPy GPT checkpoint into a Python coding
assistant from Alpaca-style JSONL:

```bash
python scripts/finetune_gpt.py --device mps --max-steps 100
```

The trainer reuses the Phase 5 tokenizer and Phase 6 checkpoint/model path,
supports assistant-only loss masking, checkpoint resume, evaluation, metrics,
and generated coding samples. See
[Phase 7 Fine-Tuning](docs/phase7_finetuning.md).

## Evaluation and Benchmarking

Phase 8 evaluates the latest fine-tuned checkpoint on a fixed 20-prompt Python
assistant benchmark, measures generation latency and throughput, calculates
validation loss and perplexity, and writes JSON, CSV, and Markdown reports:

```bash
python scripts/evaluate_gpt.py
```

Use `--checkpoint`, `--device`, or `--output-dir` to override the defaults. The
automatic pass/fail checks are safe static syntax and keyword heuristics; generated
code is never executed.

## LoRA / Parameter-Efficient Fine-Tuning

Phase 9 attaches low-rank weight parametrizations to the fused QKV and attention
output projections. All original model weights remain frozen, and the effective
weights are visible to GenPy's direct `F.linear` attention path:

```bash
python scripts/lora_train.py
python scripts/evaluate_lora.py
```

Adapters support rank, alpha, dropout, merge/unmerge, adapter-only save/load, and
CPU, CUDA, and Apple MPS execution. See [Phase 9 LoRA](docs/phase9_lora.md).

## Code Training

Build the repository-local code tokenizer from the available corpus:

```bash
python scripts/train_code_tokenizer.py --force
```

The YAML-driven trainer reads `train.jsonl`, `validation.jsonl`, and `test.jsonl`
from `data/fine_tuning/`, creates `data/tokenizer/`, and writes `tokenizer.json`,
`vocab.json`, `merges.txt`, runtime configuration, special-token metadata, and a
statistics report. A byte-identical `code_tokenizer.json` compatibility artifact
is also written for older commands. Code pretraining and fine-tuning invoke the
same builder automatically when the canonical tokenizer is missing. See
[Phase 5 Custom Tokenizer](docs/custom_tokenizer.md) for configuration, artifact,
validation, and checkpoint details.

Train the code model from `configs/code_small.yaml`:

```bash
python scripts/train_code_model.py
```

Run a small debug step:

```bash
python scripts/train_code_model.py --debug --max-steps 1
```

The script validates the tokenizer, shard globs, checkpoint directory, and mixed
precision settings before entering the training loop. In debug mode it logs each
major stage, including dataloader waits, forward/backward passes, validation,
and checkpoint saves.

## Resume Training

Resume from the latest managed checkpoint:

```bash
python scripts/train_code_model.py --resume latest
```

Resume from the best checkpoint:

```bash
python scripts/train_code_model.py --resume best
```

Resume from a specific checkpoint path:

```bash
python scripts/train_code_model.py --resume checkpoints/code_base/genpy_code_best.pt
```

The older form is still accepted:

```bash
python scripts/train_code_model.py --resume --checkpoint checkpoints/code_base/genpy_code_best.pt
```

Resume restores model weights, optimizer state, scheduler state, gradient scaler
state when present, global step, and best validation metric.

## Validation Logging

Every validation prints:

- Step
- Train loss
- Validation loss
- Perplexity
- Learning rate
- Gradient norm
- Tokens/sec
- Elapsed time
- ETA

Training metrics are appended to:

```text
evaluation/training_metrics.csv
```

The CSV records step, training loss, validation loss when available, perplexity,
learning rate, gradient norm, throughput, total tokens, elapsed time, and ETA.

After validation, training also writes a generation snapshot with five prompts:

```text
evaluation/step_0001_generation.txt
```

## Evaluate A Checkpoint

Evaluate the best checkpoint:

```bash
python scripts/evaluate_code_model.py --checkpoint best
```

Evaluate the latest checkpoint:

```bash
python scripts/evaluate_code_model.py --checkpoint latest
```

Evaluate an explicit checkpoint:

```bash
python scripts/evaluate_code_model.py --checkpoint checkpoints/code_base/genpy_code_best.pt
```

The script loads config, tokenizer, checkpoint, and validation data, then prints
a compact metrics table with validation loss, perplexity, average generation
length, and generation speed.

## Generation Benchmark

Evaluation automatically generates Python from prompts including:

```text
def factorial(n):
class Student:
for i in range(10):
import numpy as np
def quicksort(arr):
class LinkedList:
def fibonacci(n):
try:
with open("file.txt"):
```

Outputs are saved to:

```text
evaluation/generated_examples.txt
```

## Loss History

Evaluation extracts available checkpoint metadata and writes:

```text
evaluation/loss_history.csv
evaluation/loss_curve.png
```

If `evaluation/training_metrics.csv` exists, the curve is generated from the
recorded training metrics. Otherwise it falls back to checkpoint metadata.

The PNG writer is dependency-free, so it works on Windows and Google Colab
without requiring matplotlib.

## Checkpoint Management

`scripts/evaluate_code_model.py` automatically reports:

- Latest checkpoint
- Best checkpoint
- Total checkpoint count
- Checkpoint sizes

Default code checkpoints live under:

```text
checkpoints/code_base/
```

## Preparing Fine-Tuning Data

Collect approved Python files from configured local directories, Git
repositories, ZIP archives, individual files, or files placed directly in
`data/raw/`:

```bash
python scripts/collect_python_corpus.py
```

The collector validates UTF-8 and Python syntax, filters environments/build and
generated artifacts, preserves source paths, deduplicates incrementally, and
writes provenance and a collection report. See
`docs/python_corpus_collector.md` for source configuration examples.

Import configured collections, classify the validated corpus, and rebuild its
SQLite source/symbol index:

```bash
python scripts/expand_python_corpus.py
```

The expansion report tracks repository, file, function, class, category, and
estimated instruction-pair capacity. See `docs/python_corpus_expansion.md`.

Populate the production corpus from approved local directories, local Git
repositories, and ZIP archives, then query its index:

```bash
python scripts/populate_python_corpus.py
python scripts/populate_python_corpus.py --search binary_search
```

See `docs/python_corpus_population.md` for source policy, reports, and category
search examples.

Build a validated instruction-following dataset from explicitly approved Python
source material:

```bash
python scripts/build_dataset.py
```

The modular collector, cleaner, AST instruction extractor, deduplicator,
validator, and deterministic source-group splitter are configured in
`configs/dataset_pipeline.yaml`. Instructions use real docstrings when present
and otherwise use deterministic AST/source inference; responses are always exact
source spans. Bounded worker queues and temporary disk-backed SQLite indexes let
deduplication, validation, and leakage-safe splitting scale without holding the
full instruction corpus in memory. See
`docs/python_dataset_pipeline.md` for the architecture and individual stage
commands.

Fine-tuning accepts JSONL, JSON, and TXT files.

JSONL example:

```jsonl
{"instruction": "Write factorial in Python.", "input": "", "output": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"}
{"instruction": "Create a class.", "input": "Student with name and grade", "output": "class Student:\n    def __init__(self, name, grade):\n        self.name = name\n        self.grade = grade"}
```

JSON can be a list of records or an object containing `examples`, `data`, or
`records`. TXT files are treated as output-only examples split on blank lines.

Each structured record supports:

- `instruction` optional
- `input` optional
- `output` required

The loader also accepts legacy `response` records for older instruction datasets.

## Fine-Tuning

Start supervised fine-tuning from the best pretrained code checkpoint:

```bash
python scripts/fine_tune_code_model.py
```

Override common settings:

```bash
python scripts/fine_tune_code_model.py \
  --checkpoint checkpoints/code_base/genpy_code_best.pt \
  --dataset data/fine_tuning/code_instructions.jsonl \
  --output-dir checkpoints/code_fine_tune \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 0.00005 \
  --gradient-accumulation 8
```

Fine-tuning uses AdamW, mixed precision when supported by the device, gradient
accumulation, gradient clipping, cosine learning-rate decay with warmup, early
stopping, validation, generation snapshots, CSV/JSON logs, and TensorBoard logs
when TensorBoard is installed.

Fine-tuning artifacts are written by default to:

```text
checkpoints/code_fine_tune/
logs/fine_tune/
evaluation/fine_tune/
```

## Resume Fine-Tuning

Resume from the latest fine-tuning checkpoint:

```bash
python scripts/fine_tune_code_model.py --resume latest
```

Resume from the best fine-tuning checkpoint:

```bash
python scripts/fine_tune_code_model.py --resume best
```

Resume from an explicit path:

```bash
python scripts/fine_tune_code_model.py --resume checkpoints/code_fine_tune/latest.pt
```

Resume restores model weights, optimizer state, scheduler state, gradient scaler,
RNG state, epoch, global step, and best validation loss.

## Fine-Tuning Evaluation

During fine-tuning, every evaluation computes validation loss and perplexity, then
generates code for:

```text
def factorial(n):
class Student:
def fibonacci(n):
class LinkedList:
import numpy as np
```

Snapshots are saved as:

```text
evaluation/fine_tune/step_00000500_generation.txt
```

A loss curve is saved at:

```text
evaluation/fine_tune/loss_curve.png
```

You can still evaluate any checkpoint with:

```bash
python scripts/evaluate_code_model.py --checkpoint checkpoints/code_fine_tune/best.pt
```

## Common Commands

```bash
python scripts/train_code_tokenizer.py
python scripts/build_pypi_corpus.py  # after approving and enabling configs/pypi.yaml
python scripts/train_code_model.py --debug --max-steps 1
python scripts/train_code_model.py --resume latest
python scripts/fine_tune_code_model.py --checkpoint checkpoints/code_base/genpy_code_best.pt
python scripts/fine_tune_code_model.py --resume latest
python scripts/generate_code.py --checkpoint checkpoints/code_base/genpy_code_best.pt --prompt "def factorial(n):"
python scripts/evaluate_code_model.py --checkpoint best
pytest
ruff check .
```

## Notes

Early checkpoints can generate gibberish because the model has seen very few
optimizer steps. Continue pretraining from `checkpoints/code_base/genpy_code_best.pt`
or `latest` to improve generation quality over time.
