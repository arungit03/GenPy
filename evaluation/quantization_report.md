# GenPy Phase 10 Quantization Report

- Source checkpoint: `/Users/macbook/Downloads/GenPy/checkpoints/fine_tuned/last_checkpoint.pt`
- Device: `mps`
- Evaluated at: 2026-07-23T07:59:43.048562+00:00

| Method | Status | Size MiB | Load s | Memory MiB | Tokens/sec | Loss | Perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | ok | 472.57 | 1.220083 | 199.156250 | 57.248900 | 5.479552 | 239.739230 |
| fp16 | ok | 99.60 | 0.091451 | 99.578125 | 43.430788 | 5.480469 | 239.959162 |
| bf16 | skipped | 99.60 | N/A | N/A | N/A | N/A | N/A |
| dynamic_int8 | skipped | 98.45 | N/A | N/A | N/A | N/A | N/A |

## Skipped or Failed Methods

- `bf16`: bf16 inference is supported on CPU or CUDA with bf16 support.
- `dynamic_int8`: dynamic INT8 quantization requires CPU and an available quantized engine.

Original checkpoints are read-only inputs. Phase 10 writes separate model-only quantized checkpoints under the configured quantized checkpoint directory.
