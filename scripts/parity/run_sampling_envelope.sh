#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPP_DIR="${GEMMA_CPP_DIR:-/home/jamie/1tb/dev/cpp/gemma.cpp}"
CPP_BUILD_DIR="${GEMMA_CPP_BUILD_DIR:-/tmp/gemma-cpp-build-rust-parity}"
CPP_BIN="${GEMMA_CPP_BIN:-$CPP_BUILD_DIR/gemma}"
CPP_BUILD_IF_MISSING="${CPP_BUILD_IF_MISSING:-1}"
CPP_BUILD_JOBS="${CPP_BUILD_JOBS:-4}"
SBS_PATH="${GEMMA_SBS_PATH:-$ROOT_DIR/gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs}"
PROMPTS_FILE="${PROMPTS_FILE:-$ROOT_DIR/scripts/parity/sampling_prompts.txt}"
CONFIGS_FILE="${CONFIGS_FILE:-$ROOT_DIR/scripts/parity/sampling_configs.tsv}"
MAX_TOKENS="${MAX_TOKENS:-32}"
MIN_EXACT_MATCH_RATE="${MIN_EXACT_MATCH_RATE:-0.60}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$ROOT_DIR/reports/parity"
OUT_FILE="${OUT_DIR}/sampling-envelope-${STAMP}.md"
LATEST_FILE="${OUT_DIR}/sampling-envelope-latest.md"

mkdir -p "$OUT_DIR"

if [[ ! -f "$SBS_PATH" ]]; then
  echo "Missing SBS file: $SBS_PATH" >&2
  exit 1
fi
if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "Missing prompts file: $PROMPTS_FILE" >&2
  exit 1
fi
if [[ ! -f "$CONFIGS_FILE" ]]; then
  echo "Missing configs file: $CONFIGS_FILE" >&2
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

mapfile -t PROMPTS < <(grep -v '^[[:space:]]*$' "$PROMPTS_FILE")
if [[ "${#PROMPTS[@]}" -eq 0 ]]; then
  echo "No prompts loaded from $PROMPTS_FILE" >&2
  exit 1
fi

CPP_HAS_SEED=0
if "$CPP_BIN" --help 2>&1 | grep -q -- '--seed'; then
  CPP_HAS_SEED=1
fi
CPP_HAS_TOP_P=0
if "$CPP_BIN" --help 2>&1 | grep -q -- '--top_p'; then
  CPP_HAS_TOP_P=1
fi

# Build the release CLI once, then execute it directly in the loop.
cargo build -q --release -p gemma-cli
RUST_BIN="$ROOT_DIR/target/release/gemma-cli"

{
  echo "# Sampling Compatibility Envelope Report ${STAMP}"
  echo
  echo "Settings"
  echo "- Rust repo: $ROOT_DIR"
  echo "- gemma.cpp repo: $CPP_DIR"
  echo "- SBS: $SBS_PATH"
  echo "- prompts: ${#PROMPTS[@]}"
  echo "- max tokens: $MAX_TOKENS"
  echo "- min exact-match rate: $MIN_EXACT_MATCH_RATE"
  echo
} > "$OUT_FILE"

total=0
matches=0
mismatches=0

while IFS=$'\t' read -r name temp top_k top_p seed || [[ -n "${name:-}" ]]; do
  [[ -z "${name:-}" ]] && continue
  [[ "$name" =~ ^# ]] && continue

  cfg_total=0
  cfg_match=0
  cfg_mismatch=0

  {
    echo "## Config: $name"
    echo
    echo "- temperature: $temp"
    echo "- top_k: $top_k"
    echo "- top_p: $top_p"
    echo "- seed: $seed"
    echo
  } >> "$OUT_FILE"

  for prompt in "${PROMPTS[@]}"; do
    cfg_total=$((cfg_total + 1))
    total=$((total + 1))

    rust_top_p="$top_p"
    rust_seed="$seed"
    # Keep Rust sampling knobs aligned with what the C++ CLI can apply.
    if [[ "$CPP_HAS_TOP_P" != "1" ]]; then
      rust_top_p="1.0"
    fi
    if [[ "$CPP_HAS_SEED" != "1" ]]; then
      rust_seed="0"
    fi

    rust_cmd=(
      "$RUST_BIN"
      --weights "$SBS_PATH"
      --prompt "$prompt"
      --max-tokens "$MAX_TOKENS"
      --temperature "$temp"
      --top-k "$top_k"
      --top-p "$rust_top_p"
      --seed "$rust_seed"
    )
    rust_out="$("${rust_cmd[@]}")"

    cpp_cmd=(
      "$CPP_BIN"
      --weights "$SBS_PATH"
      --prompt "$prompt"
      --max_generated_tokens "$MAX_TOKENS"
      --temperature "$temp"
      --top_k "$top_k"
      --top_p "$top_p"
      --deterministic 1
      --verbosity 0
    )
    if [[ "$CPP_HAS_SEED" == "1" ]]; then
      cpp_cmd+=(--seed "$seed")
    fi
    cpp_out="$("${cpp_cmd[@]}")"

    if [[ "$rust_out" == "$cpp_out" ]]; then
      status="MATCH"
      cfg_match=$((cfg_match + 1))
      matches=$((matches + 1))
    else
      status="MISMATCH"
      cfg_mismatch=$((cfg_mismatch + 1))
      mismatches=$((mismatches + 1))
    fi

    {
      echo "### Prompt [$status]"
      echo
      echo "Prompt:"
      echo '```text'
      echo "$prompt"
      echo '```'
      echo
      echo "Rust:"
      echo '```text'
      echo "$rust_out"
      echo '```'
      echo
      echo "C++:"
      echo '```text'
      echo "$cpp_out"
      echo '```'
      echo
    } >> "$OUT_FILE"
  done

  cfg_rate="$(awk -v m="$cfg_match" -v t="$cfg_total" 'BEGIN { if (t==0) print "0.000"; else printf "%.3f", (m/t) }')"
  {
    echo "Config summary"
    echo "- total: $cfg_total"
    echo "- matches: $cfg_match"
    echo "- mismatches: $cfg_mismatch"
    echo "- exact-match rate: $cfg_rate"
    echo
  } >> "$OUT_FILE"
done < "$CONFIGS_FILE"

overall_rate="$(awk -v m="$matches" -v t="$total" 'BEGIN { if (t==0) print "0.000"; else printf "%.3f", (m/t) }')"

{
  echo "## Overall Summary"
  echo
  echo "- total: $total"
  echo "- matches: $matches"
  echo "- mismatches: $mismatches"
  echo "- exact-match rate: $overall_rate"
} >> "$OUT_FILE"

cp "$OUT_FILE" "$LATEST_FILE"

echo "Wrote sampling envelope report: $OUT_FILE"
echo "Updated latest report: $LATEST_FILE"

awk -v r="$overall_rate" -v min="$MIN_EXACT_MATCH_RATE" 'BEGIN { if (r + 0 < min + 0) exit 2; }'
