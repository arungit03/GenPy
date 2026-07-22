# Phase 7 Supervised Fine-Tuning

Phase 7 turns a pretrained GenPy base checkpoint into a Python coding assistant
using Alpaca-style supervised instruction fine-tuning.

## Dataset Format

Training data is JSONL. Each line is one object:

```json
{"instruction":"Write bubble sort.","input":"","output":"def bubble_sort(items):\n    ..."}
```

`input` may be an empty string.

## Conversation Format

Records are formatted as:

```text
<|system|>
You are GenPy, a Python coding assistant.

<|user|>
{instruction}

{optional_input}

<|assistant|>
{output}
```

The template is configurable in `configs/finetuning.yaml`.

## Training

Run Phase 7 with:

```bash
python scripts/finetune_gpt.py --device mps --max-steps 100
```

Configuration lives in `configs/finetuning.yaml`. The trainer reuses the
existing Phase 5 tokenizer, Phase 6 GPT model construction, checkpoint loader,
optimizer utilities, AMP helpers, logging, and sample generation.

## Loss Masking

Set:

```yaml
mask_prompt_tokens: true
```

to ignore system/user prompt tokens and train loss only on assistant response
tokens. Set it to `false` to train on the full conversation.

## Checkpoints

Phase 7 writes:

- `checkpoints/fine_tuned/step_00001.pt`
- `checkpoints/fine_tuned/last_checkpoint.pt`
- `checkpoints/fine_tuned/best_checkpoint.pt`

Use `--resume` to resume from `last_checkpoint.pt`, or `--checkpoint` to choose
a custom pretrained base checkpoint.

## Metrics And Samples

Metrics are written as CSV and JSONL under `metrics/phase7/`.

Generated Python-assistant samples are written under `generated_samples/`.
Default prompts include bubble sort, linked-list reversal, binary search, and
reading CSV data with pandas.

## Device Support

Phase 7 supports CPU, CUDA, and Apple MPS. On MPS, DataLoader multiprocessing is
forced to zero workers, prefetching is disabled when workers are zero, and pinned
memory is disabled because it is CUDA-specific.
