#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPP_DIR="${GEMMA_CPP_DIR:-/home/jamie/1tb/dev/cpp/gemma.cpp}"
CPP_BUILD_DIR="${GEMMA_CPP_BUILD_DIR:-/tmp/gemma-cpp-build-rust-parity}"
CPP_BIN="${GEMMA_CPP_BIN:-$CPP_BUILD_DIR/gemma}"
CPP_BUILD_IF_MISSING="${CPP_BUILD_IF_MISSING:-1}"
CPP_BUILD_JOBS="${CPP_BUILD_JOBS:-4}"
PROMPTS_FILE="${PROMPTS_FILE:-$ROOT_DIR/scripts/parity/prompts.txt}"
SBS_PATH="${GEMMA_SBS_PATH:-$ROOT_DIR/gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-1}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$ROOT_DIR/reports/parity"
OUT_FILE="$OUT_DIR/text-parity-${STAMP}.md"

mkdir -p "$OUT_DIR"

if [[ ! -f "$SBS_PATH" ]]; then
  echo "Missing SBS file: $SBS_PATH" >&2
  exit 1
fi
if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "Missing prompts file: $PROMPTS_FILE" >&2
  exit 1
fi

if [[ ! -x "$CPP_BIN" ]]; then
  if [[ "$CPP_BUILD_IF_MISSING" != "1" ]]; then
    echo "Missing gemma.cpp binary: $CPP_BIN" >&2
    echo "Set GEMMA_CPP_BIN to a prebuilt binary or set CPP_BUILD_IF_MISSING=1." >&2
    exit 3
  fi
  echo "Building gemma.cpp CLI target at $CPP_BIN ..."
  cmake -S "$CPP_DIR" -B "$CPP_BUILD_DIR" \
    -DSPM_ENABLE_SHARED=OFF \
    -DSPM_ABSL_PROVIDER=module \
    -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS=-DHWY_BROKEN_AVX10_2=HWY_AVX10_2
  spm_header="$CPP_BUILD_DIR/_deps/sentencepiece-src/src/sentencepiece_processor.h"
  if [[ -f "$spm_header" ]] && ! grep -q '^#include <cstdint>$' "$spm_header"; then
    tmp_file="${spm_header}.tmp"
    awk '
      { print }
      $0 == "#include <vector>" { print "#include <cstdint>" }
    ' "$spm_header" > "$tmp_file"
    mv "$tmp_file" "$spm_header"
  fi
  cmake --build "$CPP_BUILD_DIR" --target gemma -j"$CPP_BUILD_JOBS"
fi

EVAL_CMD=(
  "$ROOT_DIR/target/release/gemma-evals"
  --weights "$SBS_PATH"
  --prompts "$PROMPTS_FILE"
  --cpp-bin "$CPP_BIN"
  --report "$OUT_FILE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --top-p "$TOP_P"
)

if [[ -n "$SEED" ]]; then
  EVAL_CMD+=(--seed "$SEED")
fi

# Build the release evaluator once, then execute it directly.
cargo build -q --release -p gemma-evals

"${EVAL_CMD[@]}"
