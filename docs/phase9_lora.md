# Phase 9: LoRA / Parameter-Efficient Fine-Tuning

Phase 9 adds low-rank adapters to GenPy's existing attention projections without
changing the transformer architecture. The base checkpoint is loaded normally,
all original parameters are frozen, and only adapter matrices are optimized.

## Weight parametrization

GenPy attention calls `F.linear(input, layer.weight, layer.bias)` through
`_linear_preserve_input_dtype`. A wrapper that adds LoRA only in its `forward()`
would therefore be bypassed. Phase 9 registers `LoRAWeightParametrization` with
`torch.nn.utils.parametrize`, making every access to `layer.weight` return:

```text
W_effective = W_original + (alpha / rank) * (B @ A)
```

The active targets in every transformer block are:

```text
blocks.*.attention.qkv_projection
blocks.*.attention.output_projection
```

The fused QKV adapter has an update shaped `(3D, D)`. Its row ranges correspond
to Q, K, and V in that order. The output adapter has shape `(D, D)`.

`lora.dropout` is applied on the low-rank `A` path while training and disabled in
evaluation and merging. This weight-space dropout remains compatible with the
direct `layer.weight` attention path on CPU, CUDA, and Apple MPS.

## Configuration

Phase 9 is configured in `configs/lora.yaml`. Important fields include:

- `adapter.rank`, `adapter.alpha`, and `adapter.dropout`
- `adapter.target_modules`
- `training.base_checkpoint`, device, precision, learning rate, and step limits
- adapter-only checkpoint names and retention
- full-fine-tuning versus LoRA comparison settings

The default base is the final Phase 6 checkpoint. This ensures the LoRA run and
the existing Phase 7 full fine-tuning start from the same pretrained model.

## Training

Run configured LoRA instruction tuning:

```bash
python scripts/lora_train.py
```

Run a bounded check or select a device:

```bash
python scripts/lora_train.py --device mps --max-steps 10
```

Resume adapter values without loading or modifying full-model optimizer state:

```bash
python scripts/lora_train.py --resume-from checkpoints/lora/last_adapter.pt
```

Training writes adapter-only checkpoints under `checkpoints/lora`, metrics under
`metrics/phase9`, and structured logs to `logs/phase9_lora.jsonl`.

## Saving, loading, merging, and unmerging

The public adapter API is:

```python
from genpy_llm.lora import (
    apply_lora,
    load_lora_adapters,
    merge_lora_weights,
    save_lora_adapters,
    unmerge_lora_weights,
)

apply_lora(model, rank=8, alpha=16.0, dropout=0.05)
save_lora_adapters(model, "checkpoints/lora/adapter.pt")

load_lora_adapters(model, "checkpoints/lora/adapter.pt")
merge_lora_weights(model)
unmerge_lora_weights(model)
```

Merging adds the deterministic adapter delta to the frozen original weight and
temporarily disables the parameterized addition. Unmerging subtracts the exact
stored delta and restores parameterized operation. Both operations are idempotent.

Adapter files contain only `A`/`B` tensors, dimensions, hyperparameters, and
small reconstruction metadata. A matching base checkpoint is still required.

## Full fine-tuning comparison

After training an adapter, evaluate it against the existing full Phase 7
checkpoint on identical prompts and validation batches:

```bash
python scripts/evaluate_lora.py
```

The command writes separate Phase 8-style artifacts for both models and a
comparison JSON, CSV, and Markdown report under `evaluation/lora_comparison`.
The report includes validation loss, perplexity, generation speed, automatic
checks, trainable parameters, and checkpoint storage.

For a quick comparison:

```bash
python scripts/evaluate_lora.py --max-new-tokens 8 --validation-batches 1
```

## Compatibility verification

Unit tests explicitly set non-zero adapter tensors, access the effective
parameterized weight, and run the real attention/GPT forward path. They verify
that output changes despite attention calling `F.linear` directly, that base
weights remain frozen during optimization, and that merged, unmerged, saved,
and reloaded adapters produce equivalent outputs.
