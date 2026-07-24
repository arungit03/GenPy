# GenPy Phase 9: Full Fine-Tuning vs LoRA

| Metric | Full fine-tuning | LoRA |
|---|---:|---:|
| Trainable parameters | 35,823,616 | 147,456 |
| Checkpoint size | 472.57 MiB | 0.57 MiB |
| Validation loss | 5.479552 | 5.354034 |
| Perplexity | 239.739230 | 211.459697 |
| Generation speed | 65.932 tokens/sec | 62.141 tokens/sec |
| Automatic checks | 0/20 | 0/20 |

- Trainable-parameter reduction: 99.5884%
- Checkpoint-size reduction: 99.8791%

Both methods use the same prompts, generation settings, and validation batch limit.
