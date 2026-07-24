# GenPy Benchmark Summary

- Generated at: 2026-07-24T08:30:37.405227+00:00
- Base checkpoint: `/Users/macbook/Desktop/GenPy/checkpoints/last_checkpoint.pt` (step 5000)
- Continued checkpoint: `/Users/macbook/Desktop/GenPy/checkpoints/continued_pretraining/checkpoint_step_05040/model.pt` (step 5040)

| Metric | Base | Continued |
| --- | --- | --- |
| Validation loss | 5.574553 | 5.476783 |
| Perplexity | 263.632 | 239.076 |
| Next-token accuracy | 0.2209 | 0.2265 |
| Generation speed (tok/s) | 78.71 | 67.60 |
| Validation throughput (tok/s) | 6076.9 | 6733.2 |
| Mean generation latency (s) | 0.2033 | 0.2367 |
| Parameter memory | 136.7 MB | 136.7 MB |
| Checkpoint size | 472.57 MiB | 472.57 MiB |
| Checkpoint load time (s) | 0.63 | 0.60 |
| Python benchmark pass rate | 0.143 | 0.238 |
| Python execution rate | 0.105 | 0.238 |
| Documentation QA score | 0.000 | 0.000 |
| Instruction following | 0.042 | 0.083 |
| Repetition rate | 0.884 | 0.936 |

**Overall improvement: +4.219%**

Overall improvement averages relative loss/perplexity improvements with percentage-point deltas for next-token accuracy, Python pass rate, documentation QA, and instruction following.

Plots: `plots/loss.png`, `plots/perplexity.png`, `plots/speed.png`, `plots/memory.png`, `plots/latency.png` (series legend in `plots/plots.json`).
