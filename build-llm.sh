#!/bin/bash
# Build the llama-go C/C++ library (libbinding.a).
#
# Run this once after cloning, or after updating the llama-go submodule.
#
# Prerequisites: gcc, g++, make
#   sudo apt install build-essential
#
# Usage:
#   ./build-llm.sh              # build libbinding.a (CPU)
#   ./build-llm.sh clean        # remove build artifacts
#   ./build-llm.sh openblas     # build with OpenBLAS acceleration
#
# After this script completes, build with:
#   go build -tags llm

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/third_party/llama-go"

# Edithor pins llama-go to the snapshot agippy's shipped archive was built
# from (2026-05-08, last commit on Go 1.26.2, static linkage by default). third_party/ is gitignored — this script clones it.
# Moving the pin forward moves the Go toolchain too: llama-go HEAD requires
# Go 1.27 (docs/PRO_EDITION.md, "Vendoring gollum").
LLAMA_REPO="https://github.com/tcpipuk/llama-go"
LLAMA_PIN="b8a6878"

if [ ! -d "$LLAMA_DIR" ]; then
  echo "Cloning llama-go @ ${LLAMA_PIN} into ${LLAMA_DIR}..."
  git clone --quiet "$LLAMA_REPO" "$LLAMA_DIR" || exit 1
  git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_PIN" || exit 1
  git -C "$LLAMA_DIR" submodule update --init --recursive --quiet || exit 1
fi

actual="$(git -C "$LLAMA_DIR" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)"
if [ "$actual" != "$LLAMA_PIN" ]; then
  echo "WARNING: llama-go checkout is at ${actual}, pin is ${LLAMA_PIN}"
  echo "  git -C ${LLAMA_DIR} checkout ${LLAMA_PIN} && git -C ${LLAMA_DIR} submodule update --init --recursive"
fi

if [ ! -d "${LLAMA_DIR}/llama.cpp" ] || [ -z "$(ls -A "${LLAMA_DIR}/llama.cpp" 2>/dev/null)" ]; then
  echo "llama.cpp submodule not initialized. Fetching..."
  cd "$LLAMA_DIR"
  git submodule update --init --recursive
  cd "$SCRIPT_DIR"
fi

if [ "$1" = "clean" ]; then
  echo "Cleaning build artifacts..."
  cd "$LLAMA_DIR"
  make clean
  echo "Done."
  exit 0
fi

echo "Building libbinding.a in ${LLAMA_DIR}..."

cd "$LLAMA_DIR"

BUILD_TYPE=""
if [ -n "$1" ]; then
  BUILD_TYPE="$1"
  echo "Build type: ${BUILD_TYPE}"
fi

BUILD_TYPE="${BUILD_TYPE}" make libbinding.a -j$(nproc)

if [ $? -ne 0 ]; then
  echo ""
  echo "Build failed. Make sure you have the required tools:"
  echo "  sudo apt install build-essential"
  exit 1
fi

echo ""
echo "libbinding.a built successfully."
echo "You can now build with LLM support:"
echo "  go build -tags llm"
