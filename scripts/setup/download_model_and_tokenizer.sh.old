#!/usr/bin/env bash
set -euo pipefail

REPO="HuggingFaceTB/SmolLM2-135M-Instruct"
ONNX_FOLDER="onnx"
MODEL_FILE="model.onnx"
TOKENIZER_FILE="tokenizer.json"
BASE_URL="https://huggingface.co"

# Raw download URLs
MODEL_URL="${BASE_URL}/${REPO}/resolve/main/${ONNX_FOLDER}/${MODEL_FILE}"
TOKENIZER_URL="${BASE_URL}/${REPO}/resolve/main/${TOKENIZER_FILE}"

echo "Downloading model file: $MODEL_FILE"
wget --show-progress "$MODEL_URL" -O "./${MODEL_FILE}"

echo "Downloading tokenizer file: $TOKENIZER_FILE"
wget --show-progress "$TOKENIZER_URL" -O "./${TOKENIZER_FILE}"

echo "Done. Files saved"
