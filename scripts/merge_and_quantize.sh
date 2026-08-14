#!/usr/bin/env bash
# Merge Kilimo LoRA adapters into the base model, convert to GGUF, quantize Q4_K_M.
# Run on a Linux host with Python merge deps and llama.cpp convert/quantize tools.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/kilimo-0.5b.yaml}"
ADAPTER_DIR="${ADAPTER_DIR:-checkpoints/kilimo-0.5b-lora}"
MERGED_DIR="${MERGED_DIR:-merged/kilimo-0.5b}"
OUT_GGUF="${OUT_GGUF:-model/adtc-kilimo-0.5b-q4_k_m.gguf}"
FP16_GGUF="${FP16_GGUF:-model/adtc-kilimo-0.5b-f16.gguf}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

# Paths to llama.cpp helpers (override if installed elsewhere).
CONVERT_HF="${CONVERT_HF:-convert_hf_to_gguf.py}"
QUANTIZE_BIN="${QUANTIZE_BIN:-llama-quantize}"

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "ERROR: adapter dir not found: ${ADAPTER_DIR}" >&2
  echo "Train first: python scripts/train_lora.py --config ${CONFIG}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_GGUF}")" "${MERGED_DIR}"

echo "==> Merging LoRA into base (${BASE_MODEL})"
python - <<PY
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "${BASE_MODEL}"
adapter = "${ADAPTER_DIR}"
merged = "${MERGED_DIR}"

tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

Path(merged).mkdir(parents=True, exist_ok=True)
model.save_pretrained(merged, safe_serialization=True)
tokenizer.save_pretrained(merged)
print(f"Merged model written to {merged}")
PY

if ! command -v "${CONVERT_HF}" >/dev/null 2>&1 && [[ ! -f "${CONVERT_HF}" ]]; then
  echo "ERROR: convert script not found: ${CONVERT_HF}" >&2
  echo "Clone llama.cpp and set CONVERT_HF=/path/to/convert_hf_to_gguf.py" >&2
  exit 1
fi

if ! command -v "${QUANTIZE_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${QUANTIZE_BIN} not on PATH" >&2
  exit 1
fi

echo "==> Converting HF -> GGUF F16"
python "${CONVERT_HF}" "${MERGED_DIR}" --outfile "${FP16_GGUF}" --outtype f16

echo "==> Quantizing -> Q4_K_M"
"${QUANTIZE_BIN}" "${FP16_GGUF}" "${OUT_GGUF}" Q4_K_M

echo "==> Done: ${OUT_GGUF}"
echo "Upload this file to Hugging Face, then set MODEL_REPO / EXPECTED_SHA256 in download_model.sh"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${OUT_GGUF}"
fi
