#!/usr/bin/env bash
# Fetches the quantized GGUF weights for the ADTC 2026 agriculture submission.
# Runs before the profiler starts; no network access is used after this point.
set -euo pipefail

# Override MODEL_URL to point the script at a fork or a mirror without editing it.
MODEL_REPO="${MODEL_REPO:-rssebambulidde/adtc-kilimo-1.5b-gguf}"
MODEL_FILE="${MODEL_FILE:-adtc-kilimo-1.5b-q4_k_m.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}?download=true}"

# Must match _runtime.model_path in metadata.json exactly.
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model"
DEST="${DEST_DIR}/${MODEL_FILE}"

# Populate after uploading the weights: shasum -a 256 model/<file>
EXPECTED_SHA256="${EXPECTED_SHA256:-}"

mkdir -p "${DEST_DIR}"

verify() {
  [ -s "$1" ] || return 1
  # A GGUF file starts with the magic bytes "GGUF".
  [ "$(head -c 4 "$1")" = "GGUF" ] || return 1
  if [ -n "${EXPECTED_SHA256}" ] && command -v sha256sum >/dev/null 2>&1; then
    [ "$(sha256sum "$1" | cut -d' ' -f1)" = "${EXPECTED_SHA256}" ] || return 1
  fi
  return 0
}

if verify "${DEST}"; then
  echo "Model already present and valid: ${DEST}"
  exit 0
fi

echo "Downloading ${MODEL_FILE} from ${MODEL_REPO}"
TMP="${DEST}.part"
rm -f "${TMP}"

if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 5 --retry-delay 3 --retry-connrefused -o "${TMP}" "${MODEL_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=5 --waitretry=3 -O "${TMP}" "${MODEL_URL}"
else
  echo "ERROR: neither curl nor wget is available" >&2
  exit 1
fi

if ! verify "${TMP}"; then
  echo "ERROR: downloaded file failed validation (not GGUF, empty, or checksum mismatch)" >&2
  rm -f "${TMP}"
  exit 1
fi

mv -f "${TMP}" "${DEST}"
echo "Model ready: ${DEST} ($(du -h "${DEST}" | cut -f1))"
