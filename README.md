# ADTC 2026 — Kilimo: On-Device Agricultural Advisory

Submission for the [Africa Deep Tech Challenge 2026](https://adtc-2026.devpost.com) Laptop LLM track, domain **agriculture**.

A fine-tuned 0.5B-parameter language model (Qwen2.5-0.5B-Instruct → LoRA → GGUF Q4_K_M) that answers smallholder agronomy questions entirely offline on an 8 GB commodity laptop.

---

## Why this shape

Leaderboard score:

`S_total = 0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal`

- **S_perf** saturates at 15 tokens/sec (`min(TPS/15, 1.0) × 100`) — no reward for going faster.
- **S_eff** pays `(7 − peak_RAM_GB) / 7 × 100` — every unused gigabyte counts.
- **S_acc** (50%) is judged from your two prompts + two hidden domain prompts.

The plan started at 1.5B on the assumption it would clear 15 tok/s and saturate `S_perf`.
**Measurement disproved that**, and the corrected numbers picked the model.

Both candidates profiled with the real ADTC profiler on an Intel Core i5-8365U (4C/8T, 2019
U-series — a generation below the ADTC reference, so these are conservative floors):

| Candidate | Peak RSS | TPS | S_perf | S_eff | 0.3·perf + 0.2·eff |
|---|---|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M | 1.67 GB | 5.87 | 39.1 | 76.1 | 26.97 |
| **Qwen2.5-0.5B Q4_K_M** | **0.53 GB** | **14.40** | **96.0** | **92.4** | **47.28** |

The 0.5B is worth **20.3 more points of final score**. For the 1.5B to win overall it would have
to score 41 points higher on accuracy out of 100, which is not a realistic gap between these two
checkpoints on a judged rubric. On the faster reference laptop the 0.5B should saturate `S_perf`
outright while the 1.5B still would not.

Memory was never the binding constraint — even the 1.5B sat at 1.67 GB against a 7 GB budget
with no OOM risk. Throughput was, and throughput tracks parameter count. Hence a 0.5B base with
all remaining effort spent on domain accuracy, which is the other 50%.

---

## Repository layout

```
metadata.json          Required. Team, model, 2 domain test prompts.
download_model.sh      Required. Idempotent GGUF fetch into model/.
REPORT.md              Required. Technical writeup for judges.
LICENSE                GPL-3.0 (template inheritance).

configs/               LoRA training config.
data/seed/             Curated East African agronomy Q&A.
scripts/               Dataset build, train, merge/quantize, score.
eval/                  Held-out prompts for local quality checks.
model/                 Weights land here (gitignored).
```

Weights are never committed. The evaluator runs `download_model.sh`, then cuts network before profiling.

---

## Runbook

### 0. Prerequisites

Local machine: CPU-only self-checks. Training needs a Linux GPU host (Udutech credits or Colab).

Use Python 3.12 (profiler / `llama-cpp-python` are happier than on 3.14):

```bash
uv venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Install the ADTC profiler separately:

```bash
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
```

You also need `llama-bench` on PATH (from [llama.cpp](https://github.com/ggerganov/llama.cpp)) for the profiler.

### 1. Build the dataset

```bash
python scripts/build_dataset.py --out data/build/train.jsonl
# optional paraphrase expansion:
python scripts/build_dataset.py --out data/build/train.jsonl --augment
```

### 2. Fine-tune (GPU host)

```bash
python scripts/train_lora.py --config configs/kilimo-0.5b.yaml
```

### 3. Merge and quantize

```bash
bash scripts/merge_and_quantize.sh
```

Produces `model/adtc-kilimo-0.5b-q4_k_m.gguf` (must match `_runtime.model_path` in `metadata.json`).

### 4. Benchmark locally

The profiler needs `llama-bench` on PATH. On Windows, grab the prebuilt CPU build
(`llama-*-bin-win-cpu-x64.zip` from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases))
and unzip it to `tools/llama/`, then:

```powershell
bash download_model.sh
pwsh scripts/run_profiler.ps1
```

`run_profiler.ps1` puts `tools/llama` on PATH, runs the profiler in participant
mode, and prints the normalized scores. On Linux/macOS:

```bash
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json
python scripts/score.py submission.json
```

Paste S_perf / S_eff into `REPORT.md` and the Devpost form.

Note: `llama-bench` is pinned to `-ngl 0` by the profiler, so numbers are CPU-only
by design and should reconcile with the audit VM.

### 5. Publish & submit

1. Upload the `.gguf` to a public Hugging Face repo.
2. Set `MODEL_REPO` and `EXPECTED_SHA256` in `download_model.sh`.
3. Fill every `TODO` in `metadata.json`.
4. Push this repo public on GitHub.
5. Submit the repo URL on [Devpost](https://adtc-2026.devpost.com).
6. Attach the 2-minute demo video.

---

## Before you submit

- [ ] Every `TODO` in `metadata.json` replaced
- [ ] `MODEL_REPO` and `EXPECTED_SHA256` set in `download_model.sh`
- [ ] `bash download_model.sh` succeeds from a clean clone
- [ ] `adtc-profiler run` completes with `"measured_on": "participant_laptop"`
- [ ] `REPORT.md` benchmark numbers match the profiler run
- [ ] Repository is public
- [ ] 2-minute demo video recorded
- [ ] Eligibility: team ≤ 3, venture < 12 months, < $25k raised, African residence

---

## License

GPL-3.0, inherited from the [ADTC 2026 submission template](https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template).
