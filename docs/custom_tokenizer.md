# Phase 5 Custom Tokenizer

GenPy uses one tokenizer implementation for tokenizer training, model pretraining,
fine-tuning, generation, and inspection. Phase 5 extends the existing
`CodeTokenizer` adapter; it does not introduce a competing tokenizer stack.

## Build

From the repository root, run:

```bash
python scripts/train_code_tokenizer.py --force
```

The command loads `configs/tokenizer.yaml`, streams these datasets in a stable
order, validates every JSONL record while reading it, trains the tokenizer, and
then validates Python and Unicode round trips:

1. `data/fine_tuning/train.jsonl`
2. `data/fine_tuning/validation.jsonl`
3. `data/fine_tuning/test.jsonl`

Use `--config` for another YAML file. `--vocab-size`, `--min-frequency`,
`--max-training-bytes`, `--seed`, `--train-pattern`, and `--output` are intended
for controlled experiments. The checked-in defaults train on the complete
configured corpus with a vocabulary of 32,000 and `min_frequency: 2`.

## Architecture

`scripts/train_code_tokenizer.py` is the CLI and logging layer.
`src/genpy_llm/code_tokenizer.py` owns configuration validation, deterministic
corpus streaming, ByteLevel BPE training, atomic artifact writes, statistics,
and encode/decode checks. `configs/tokenizer.yaml` is the single training
configuration. Existing training code continues to consume the `CodeTokenizer`
adapter.

Training uses the Rust-backed `tokenizers` library with:

- Unicode NFC normalization;
- byte-level pre-tokenization and decoding;
- deterministic, lexically ordered corpus input;
- BPE with a minimum frequency of 2;
- the complete byte alphabet, ensuring arbitrary UTF-8 code is representable;
- fixed special-token ordering and IDs;
- a BOS/EOS post-processor when `add_special_tokens=True`.

The Phase 5 token IDs are stable:

| ID | Token |
|---:|---|
| 0 | `<pad>` |
| 1 | `<unk>` |
| 2 | `<bos>` |
| 3 | `<eos>` |
| 4 | `<mask>` |
| 5 | `<instruction>` |
| 6 | `<input>` |
| 7 | `<output>` |

Fine-tuning prompts use the instruction, input, and output tokens directly. The
loader remains able to read legacy GenPy tokenizer files containing uppercase
`<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>` tokens.

## Artifacts

The build writes all files atomically under `data/tokenizer/`:

- `tokenizer.json`: complete tokenizer graph loaded by GenPy;
- `vocab.json`: BPE token-to-ID vocabulary;
- `merges.txt`: learned BPE merge rules;
- `tokenizer_config.json`: runtime and training settings;
- `special_tokens.json`: special-token roles, strings, and IDs;
- `tokenizer_statistics.json`: corpus, compression, vocabulary-use, validation,
  and artifact SHA-256 statistics;
- `tokenizer_metadata.json`: backward-compatible GenPy build metadata;
- `code_tokenizer.json`: byte-identical compatibility copy of `tokenizer.json`.

Do not edit one artifact independently. Re-run the build so the vocabulary,
merges, configuration, hashes, and compatibility copy remain synchronized.

## Training integration

`configs/code_small.yaml` points at `data/tokenizer/tokenizer.json` and declares
the 32,000-token Phase 5 vocabulary. Both `train_code_model.py` and
`fine_tune_code_model.py` automatically call the shared builder when that file
is absent.

A checkpoint's embedding and output matrices are tied to the tokenizer's exact
vocabulary size and token IDs. A checkpoint created with the legacy 16,000-token
tokenizer cannot be fine-tuned with the new 32,000-token tokenizer without an
explicit embedding migration. Train a Phase 5 base checkpoint before continuing
fine-tuning; compatible checkpoints need no workflow changes.

## Verification and tests

Inspect a built tokenizer with:

```bash
python scripts/inspect_code_tokenizer.py
```

Run the focused tests with:

```bash
pytest tests/test_custom_tokenizer.py tests/test_code_pipeline.py \
  tests/test_code_fine_tuning_pipeline.py
```

The tests cover artifact creation, deterministic builds, fixed special-token
IDs, BOS/EOS processing, Python indentation, decorators, type hints, f-strings,
non-ASCII source, NFC behavior, and legacy adapter compatibility.
