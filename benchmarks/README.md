# Benchmark evidence

Raw `adtc-profiler` participant-mode reports backing the model-selection argument
in [`../REPORT.md`](../REPORT.md). Committed so the comparison can be checked
rather than taken on trust.

| File | Model | TPS | Peak RSS | S_perf | S_eff |
|---|---|---|---|---|---|
| `qwen2.5-1.5b-q4_k_m.json` | Qwen2.5-1.5B-Instruct Q4_K_M | 5.87 | 1,711 MB | 39.13 | 76.13 |
| `qwen2.5-0.5b-q4_k_m.json` | Qwen2.5-0.5B-Instruct Q4_K_M | 14.40 | 543 MB | 96.00 | 92.42 |

Both runs used `adtc-profiler 0.1.0`, schema 1.1.0, participant mode,
`--skip-accuracy`, seed 42, on an Intel Core i5-8365U (4C/8T) with 15.8 GB RAM
under Windows 11. `llama-bench` is pinned to `-ngl 0` by the profiler, so these
are CPU-only numbers.

These profile the *stock* base models, not the fine-tuned Kilimo checkpoint. LoRA
does not change parameter count or quantization, so throughput and memory carry
over; the fine-tune moves accuracy, which these runs do not measure
(`--skip-accuracy`, so the `accuracy` array is empty by design).

Regenerate with:

```powershell
pwsh scripts/run_profiler.ps1 -SkipAccuracy -Output benchmarks/<name>.json
```
