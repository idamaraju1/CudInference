#!/usr/bin/env bash
set -euo pipefail

# Runs the full setup flow:
# 1) dependency check
# 2) model + tokenizer download
# 3) ONNX proto + SentencePiece setup
# 4) CMake configure and build

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SETUP_DIR="$PROJECT_ROOT/scripts/setup"

cd "$PROJECT_ROOT"

echo "=== Checking dependencies ==="
"$SETUP_DIR/check_dependencies.sh"

echo
echo "=== Downloading model and tokenizer ==="
python3 "$SETUP_DIR/download_model.py"

echo
echo "=== Setting up ONNX protobufs and SentencePiece ==="
"$SETUP_DIR/setup_onnx_proto.sh"

echo
echo "=== Configuring and building project ==="
BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ..
cmake --build . -j"$(command -v nproc >/dev/null && nproc || sysctl -n hw.ncpu || echo 4)"

echo
echo "[+] Setup complete. Binary located at: $BUILD_DIR/onnx_gpu_engine"
