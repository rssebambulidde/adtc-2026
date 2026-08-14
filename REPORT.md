# Technical Report — Kilimo: On-Device Agricultural Advisory

**Team ID:** TODO-REGISTERED-TEAM-ID  
**Domain:** agriculture  
**Model:** ADTC-Kilimo-1.5B-Q4_K_M

---

## Problem

Smallholder farmers and frontline extension officers across East Africa need timely agronomic advice — pest triage, storage, input timing — but cloud LLMs assume stable internet, paid APIs, and spare electricity. Those assumptions fail on many farms and in many extension posts.

Kilimo is a fully offline language model for agricultural advisory on the computers people already own: mid-range 8 GB laptops with integrated graphics. The target user is a farmer, agro-dealer, or extension officer in Uganda, Kenya, or Tanzania who needs practical answers about maize, cassava, beans, coffee, and banana under rainfed smallholder conditions, without uploading farm data to the cloud.

---

## Design Decisions

- **Base model:** Qwen2.5-1.5B-Instruct. Small enough to clear the ADTC throughput and efficiency ceilings on 4 vCPUs / 8 GB RAM, large enough that domain LoRA can carry accuracy.
- **Quantization:** GGUF Q4_K_M. Chosen as the quality/size knee for CPU llama.cpp; Q8 would waste efficiency score; aggressive Q2/Q3 hurt agronomic specificity.
- **Runtime:** llama.cpp only (required by the ADTC evaluation pipeline). No custom server, RAG, or cloud calls during inference.
- **Training:** LoRA SFT on curated East African agronomy Q&A (seed corpus + chat-format build), then merge adapters and export GGUF.
- **Alternatives considered:**
  - 7B Q4: higher accuracy potential but ~4–5 GB RSS and low TPS → large loss on S_perf and S_eff.
  - 0.5B Q4: slightly better mechanical scores, but weaker on multi-step advisory answers.
  - App + RAG wrappers: useful for demos, but the profiler scores only the GGUF weights via llama.cpp, so domain knowledge must live in the model.

**Scoring rationale (planning estimates — replace with profiler numbers):** throughput saturates at 15 tok/s; efficiency rewards unused RAM under a 7 GB budget. A ~1.5B Q4 model targets ~100 S_perf and high S_eff, so remaining effort goes into S_acc and the African use-case claim.

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

| Metric | Value |
|---|---|
| Machine | TBD — fill from participant laptop |
| RAM at peak | TBD — from `submission.json` / `scripts/score.py` |
| Time to first token | TBD |
| Generation speed | TBD (target ≥ 15 tok/s) |
| Thermal throttling | TBD (must stay under 85°C) |
| Normalized S_perf | TBD |
| Normalized S_eff | TBD |

These are self-reported development benchmarks. Official scores are measured by the ADTC profiler on the standard evaluation machine.

After a local run:

```bash
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json
python scripts/score.py submission.json
```

Paste the printed S_perf / S_eff and peak RSS into this table before Gate 1 submit.
