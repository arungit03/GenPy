# Base vs Continued Checkpoint Comparison

- Base: `/Users/macbook/Desktop/GenPy/checkpoints/last_checkpoint.pt` (step 5000)
- Continued: `/Users/macbook/Desktop/GenPy/checkpoints/continued_pretraining/checkpoint_step_05040/model.pt` (step 5040)

| Metric | Base | Continued | Delta |
| --- | --- | --- | --- |
| Validation loss | 5.574553 | 5.476783 | -0.097770 (+1.754%) |
| Perplexity | 263.632 | 239.076 | -24.555 (+9.314%) |
| Next-token accuracy | 0.2209 | 0.2265 | +0.55 pp |
| Generation speed (tok/s) | 78.71 | 67.60 | -14.110% |
| Mean latency (s) | 0.2033 | 0.2367 | -16.428% |
| Parameter memory (MB) | 136.7 | 136.7 | n/a |
| Checkpoint size | 472.57 MiB | 472.57 MiB | n/a |
| Load time (s) | 0.63 | 0.60 | n/a |
| Python pass rate | 0.143 | 0.238 | +9.52 pp |
| Python execution rate | 0.105 | 0.238 | n/a |
| Documentation QA score | 0.000 | 0.000 | +0.00 pp |
| Instruction following | 0.042 | 0.083 | +4.17 pp |
| Repetition rate | 0.884 | 0.936 | lower is better |

## Python benchmark pass rate by category

| Category | Base | Continued |
| --- | --- | --- |
| algorithms | 0.200 | 0.200 |
| asyncio | 0.400 | 0.200 |
| cli | 0.200 | 0.400 |
| csv | 0.200 | 0.400 |
| data_structures | 0.200 | 0.200 |
| decorators | 0.000 | 0.200 |
| django | 0.200 | 0.200 |
| fastapi | 0.200 | 0.200 |
| file_handling | 0.000 | 0.000 |
| flask | 0.000 | 0.000 |
| generators | 0.000 | 0.200 |
| json | 0.000 | 0.200 |
| logging | 0.200 | 0.200 |
| numpy | 0.200 | 0.200 |
| oop | 0.000 | 0.400 |
| pandas | 0.400 | 0.600 |
| pytorch | 0.200 | 0.200 |
| regex | 0.000 | 0.000 |
| sql | 0.200 | 0.400 |
| testing | 0.200 | 0.200 |
| typing | 0.000 | 0.400 |

## Documentation QA score by source

| Source | Base | Continued |
| --- | --- | --- |
| django | 0.000 | 0.000 |
| fastapi | 0.000 | 0.000 |
| flask | 0.000 | 0.000 |
| numpy | 0.000 | 0.000 |
| pandas | 0.000 | 0.000 |
| peps | 0.000 | 0.000 |
| python_docs | 0.000 | 0.000 |

**Overall improvement: +4.219%**
