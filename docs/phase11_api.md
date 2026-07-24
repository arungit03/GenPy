# Phase 11: Offline API Serving

Phase 11 adds a local FastAPI server for GenPy inference. The server loads the tokenizer and model
once at startup, then reuses the existing checkpoint, LoRA, quantization, and generation utilities
for every request.

## Run

```bash
python scripts/run_api.py --config configs/api.yaml --host 127.0.0.1 --port 8000
```

Interactive docs are available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## Configuration

`configs/api.yaml` controls serving:

```yaml
device: auto
phase7_config: configs/finetuning.yaml
checkpoint: checkpoints/fine_tuned/best_checkpoint.pt
quantized_checkpoint: checkpoints/quantized/last_checkpoint_fp16.pt
lora_adapter:
generation:
  temperature: 0.7
  top_p: 0.95
  max_new_tokens: 256
```

`device: auto` chooses Apple MPS first, then CUDA, then CPU. Set `quantized_checkpoint` to `null`
to serve the base checkpoint. Set `lora_adapter` to an adapter path to apply LoRA on top of the
loaded base or floating quantized checkpoint. Dynamic INT8 checkpoints are CPU-only and cannot load
adapter-only LoRA files.

## Endpoints

### `GET /health`

```json
{
  "status": "healthy",
  "device": "mps",
  "model_loaded": true
}
```

### `GET /model`

Returns model name, parameter count, checkpoint path, quantization method, LoRA status, tokenizer,
context length, vocabulary size, loaded timestamp, and device.

### `POST /generate`

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a Python function that reverses a string.","max_new_tokens":80,"temperature":0.7,"top_p":0.9}'
```

```json
{
  "generated_text": "def reverse_string(value):\n    return value[::-1]",
  "tokens_generated": 42,
  "generation_time": 0.83,
  "tokens_per_second": 50.6
}
```

### `POST /chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Show a binary search implementation in Python."}]}'
```

The response shape matches `/generate`.
