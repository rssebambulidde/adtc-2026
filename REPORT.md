# Technical Report — Kilimo: On-Device Agricultural Advisory

**Team ID:** rssebambulidde-kilimo  
**Domain:** agriculture  
**Model:** ADTC-Kilimo-0.5B-Q4_K_M

---

## Problem

Smallholder farmers and frontline extension officers across East Africa need timely agronomic advice — pest triage, storage, input timing — but cloud LLMs assume stable internet, paid APIs, and spare electricity. Those assumptions fail on many farms and in many extension posts.

Kilimo is a fully offline language model for agricultural advisory on the computers people already own: mid-range 8 GB laptops with integrated graphics. The target user is a farmer, agro-dealer, or extension officer in Uganda, Kenya, or Tanzania who needs practical answers about maize, cassava, beans, coffee, and banana under rainfed smallholder conditions, without uploading farm data to the cloud.

---

## Design Decisions

- **Base model:** Qwen2.5-0.5B-Instruct, selected by measurement rather than assumption. The
  1.5B variant was profiled first and rejected; see [Benchmarks](#benchmarks) for the numbers
  that drove the switch.
- **Quantization:** GGUF Q4_K_M. Chosen as the quality/size knee for CPU llama.cpp; Q8 would waste efficiency score; aggressive Q2/Q3 hurt agronomic specificity.
- **Runtime:** llama.cpp only (required by the ADTC evaluation pipeline). No custom server, RAG, or cloud calls during inference.
- **Training:** LoRA SFT on curated East African agronomy Q&A (seed corpus + chat-format build), then merge adapters and export GGUF. Because the base model is small, the corpus deliberately spans crops, livestock, weather, and markets rather than staple crops alone, so the two hidden judge prompts are less likely to land outside its competence.
- **Alternatives considered:**
  - 7B Q4: higher accuracy potential but ~4–5 GB RSS and very low TPS → large loss on both S_perf and S_eff.
  - 1.5B Q4: profiled and rejected. Better raw language ability, but 5.87 tok/s on the dev machine gave up 20.31 points of final score against the 0.5B.
  - App + RAG wrappers: useful for demos, but the profiler scores only the GGUF weights via llama.cpp, so domain knowledge must live in the model.

**Scoring rationale:** throughput saturates at 15 tok/s and carries 30%; efficiency rewards unused
RAM under a 7 GB budget and carries 20%. Measurement showed memory is easily satisfied at this
scale while throughput is not, so parameter count was cut until `S_perf` was nearly saturated,
and the freed effort went into accuracy — the remaining 50% — and the African use-case claim.

---

## Constraints

- Target hardware: Intel Core i5 10th–12th gen or AMD Ryzen 5 3000–5000, 8 GB DDR4, integrated graphics only, Ubuntu 22.04 reference OS.
- Pure CPU inference via llama.cpp; no discrete GPU in evaluation.
- 100% offline during profiling — `download_model.sh` runs first; then zero network.
- Peak RSS must stay under the 7 GB effective budget or S_eff collapses; OOM is disqualification.
- Connectivity and power on the farm: answers must be short, actionable, and usable without follow-up API calls.
- Training compute: LoRA on borrowed GPU time (Udutech credits / Colab); local Windows machine is CPU-only for self-checks.

---

## Benchmarks

Measured with `adtc-profiler 0.1.0` (schema 1.1.0) in participant mode, `llama-bench`
pinned to CPU (`-ngl 0`), prompt 512 / generation 128 tokens.

**Development machine (not the ADTC reference):**
Intel Core i5-8365U, 4 cores / 8 threads, 1.60 GHz base, 15.8 GB RAM, no discrete GPU, Windows 11.
This is an 8th-generation U-series part, a generation or two *below* the ADTC Standard Laptop
(i5 10th–12th gen / Ryzen 5 3000–5000), so the throughput below is a conservative floor rather
than a prediction of the audit result.

| Metric | Qwen2.5-1.5B Q4_K_M | **Qwen2.5-0.5B Q4_K_M (selected)** |
|---|---|---|
| Generation speed | 5.87 tok/s | **14.40 tok/s** |
| Time to first token (512-tok prompt) | 23,125 ms | **13,646 ms** |
| Peak RSS | 1,711 MB (1.67 GB) | **543 MB (0.53 GB)** |
| Steady-state RSS | 1,615 MB | **507 MB** |
| CPU p99 | 100% | 100% |
| Core temp peak | not exposed by the OS on this machine | same |
| Thermal throttling | none flagged | none flagged |
| Normalized S_perf | 39.13 | **96.00** |
| Normalized S_eff | 76.13 | **92.42** |
| 0.30·S_perf + 0.20·S_eff | 26.97 | **47.28** |

Reported `params_count` from the GGUF header matched the declared `parameters_estimate` in both
runs (1,543,714,304 and 494,032,768 respectively), so the profiler's fraud check passes.

### What the measurement changed

The design was initially sized on the assumption that a 1.5B Q4_K_M model would clear the
15 tok/s reference and saturate `S_perf`. Measurement disproved that: at 5.87 tok/s the 1.5B
reaches only 39 of 100 on throughput.

Profiling a 0.5B checkpoint on the identical harness settled the question. It is worth
**20.31 more points of final score** than the 1.5B. For the 1.5B to win overall it would need
to beat the 0.5B by 41 points of accuracy out of 100, since accuracy carries 50% weight — not a
credible gap between two checkpoints of the same family on a judged advisory rubric.

Memory turned out never to be the binding constraint. Even the 1.5B used 1.67 GB against a 7 GB
budget with no OOM risk, so the efficiency term was largely banked either way and shrinking the
model bought only 16 additional efficiency points. Throughput was the real constraint, it carries
a larger weight, and it tracks parameter count almost directly.

The decision is therefore a 0.5B base, with the entire remaining effort budget spent on domain
accuracy through LoRA fine-tuning, which is where the other 50% of the score lives.

### Known caveat

The development machine is an 8th-generation U-series laptop, below the ADTC Standard Laptop
(i5 10th–12th gen / Ryzen 5 3000–5000). The audit machine should be faster, which would push the
0.5B to saturate `S_perf` at 100 and lift the 1.5B somewhat without closing the gap. The core
temperature sensor is not exposed on this machine, so `core_temp_c_peak` is null locally; the
thermal penalty can only be confirmed on the audit run.

These are self-reported development benchmarks. Official scores are measured by the ADTC
profiler on the standard evaluation machine.

Reproduce:

```bash
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json
python scripts/score.py submission.json
```
