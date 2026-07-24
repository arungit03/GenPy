# GenPy Phase 10: Quantization

Phase 10 adds inference-only checkpoint conversion and benchmarking. It does not change the
Transformer architecture, does not retrain the model, and leaves the source checkpoint untouched.

## Methods

- `fp16`: deep-copies the loaded model and converts floating-point weights to `torch.float16`.
- `bf16`: deep-copies the loaded model and converts floating-point weights to `torch.bfloat16`.
- `dynamic_int8`: deep-copies the CPU model and applies PyTorch dynamic quantization to
  `nn.Linear` layers.

Quantized checkpoints are model-only inference artifacts saved separately under
`checkpoints/quantized` by default. They include source checkpoint metadata and a SHA256 digest so
the original artifact can be traced without being modified.

## Backend Support

Phase 10 detects runtime backend capabilities before benchmarking:

- `fp16` runs on CUDA and Apple MPS.
- `bf16` runs on CPU and CUDA devices with BF16 support.
- `dynamic_int8` runs on CPU only.

Unsupported methods are reported as `skipped` instead of failing the whole benchmark. This is
especially important on Apple MPS, where `dynamic_int8` and `bf16` are skipped for inference.

## Usage

Create quantized checkpoints from the latest fine-tuned checkpoint:

```bash
python scripts/quantize_model.py --config configs/quantization.yaml
```

Run the smoke benchmark and write Phase 10 reports:

```bash
python scripts/benchmark_quantization.py --config configs/quantization.yaml
```

Override the source checkpoint or device:

```bash
python scripts/benchmark_quantization.py \
  --checkpoint checkpoints/fine_tuned/last_checkpoint.pt \
  --device cpu
```

## Outputs

The benchmark writes:

- `evaluation/quantization_results.json`
- `evaluation/quantization_results.csv`
- `evaluation/quantization_report.md`

Each result includes:

- checkpoint size
- model state memory
- load time
- inference speed
- validation loss
- perplexity
- backend skip or failure reason when a method cannot run

## Configuration

`configs/quantization.yaml` controls the Phase 7 config to reuse, source checkpoint, quantized
checkpoint directory, benchmark prompts, validation batch limit, generation length, and device.
The default source checkpoint is `latest`, resolved through the existing Phase 8 fine-tuned
checkpoint resolver.
