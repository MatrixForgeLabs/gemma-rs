#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="reports/perf"
OUT_FILE="${OUT_DIR}/baseline-${STAMP}.md"
LATEST_FILE="${OUT_DIR}/latest.md"
mkdir -p "$OUT_DIR"

ITERS_CORE="${GEMMA_PERF_ITERS_CORE:-20000}"
ITERS_OPS="${GEMMA_PERF_ITERS_OPS:-500}"
COMPARE_TO="${GEMMA_PERF_COMPARE_TO:-}"
MAX_REGRESSION_PCT="${GEMMA_PERF_MAX_REGRESSION_PCT:-8}"
SUMMARY_FILE="${GEMMA_PERF_SUMMARY_FILE:-${OUT_DIR}/latest-summary.tsv}"

extract_metric() {
  local file="$1"
  local name="$2"
  awk -v key="$name" '
    index($0, key":") == 1 {
      for (i = 1; i <= NF; ++i) {
        if ($i ~ /^ns_per_iter=/) {
          sub("ns_per_iter=", "", $i)
          print $i
          exit
        }
      }
    }
  ' "$file"
}

check_regression() {
  local metric="$1"
  local baseline="$2"
  local current="$3"
  awk -v m="$metric" -v b="$baseline" -v c="$current" -v max_pct="$MAX_REGRESSION_PCT" '
    BEGIN {
      if (b <= 0 || c <= 0) {
        printf("skip %s (invalid baseline/current)\n", m)
        exit 0
      }
      delta = ((c - b) / b) * 100.0
      printf("%s: baseline=%.1f current=%.1f delta=%.2f%% (max=%.2f%%)\n", m, b, c, delta, max_pct)
      if (delta > max_pct) exit 2
    }
  '
}

{
  echo "# Perf Baseline ${STAMP}"
  echo
  echo "Environment"
  echo "- Host: $(uname -a)"
  echo "- Rust: $(rustc --version)"
  echo "- Cargo: $(cargo --version)"
  echo
  echo "## gemma-core attention/decode"
  echo '```'
  GEMMA_PERF_ITERS="$ITERS_CORE" cargo run -p gemma-core --release --bin perf_attention_decode
  echo '```'
  echo
  echo "## gemma-ops matmul/decode"
  echo '```'
  GEMMA_PERF_ITERS="$ITERS_OPS" cargo run -p gemma-ops --release --bin perf_matmul_decode
  echo '```'
} | tee "$OUT_FILE"

flash_ns="$(extract_metric "$OUT_FILE" "flash_attention_causal")"
decode_ns="$(extract_metric "$OUT_FILE" "decode_argmax_surrogate")"
matmul_ns="$(extract_metric "$OUT_FILE" "matmul_f32_64x64x64")"
matvec_ns="$(extract_metric "$OUT_FILE" "matvec_dispatch_f32_4096x2048")"

{
  echo -e "metric\tns_per_iter"
  echo -e "flash_attention_causal\t${flash_ns:-}"
  echo -e "decode_argmax_surrogate\t${decode_ns:-}"
  echo -e "matmul_f32_64x64x64\t${matmul_ns:-}"
  echo -e "matvec_dispatch_f32_4096x2048\t${matvec_ns:-}"
} > "$SUMMARY_FILE"

if [[ -n "$COMPARE_TO" ]]; then
  if [[ ! -f "$COMPARE_TO" ]]; then
    echo "Comparison baseline not found: $COMPARE_TO" >&2
    exit 3
  fi

  base_flash="$(extract_metric "$COMPARE_TO" "flash_attention_causal")"
  base_decode="$(extract_metric "$COMPARE_TO" "decode_argmax_surrogate")"
  base_matmul="$(extract_metric "$COMPARE_TO" "matmul_f32_64x64x64")"
  base_matvec="$(extract_metric "$COMPARE_TO" "matvec_dispatch_f32_4096x2048")"

  echo "Perf regression check against: $COMPARE_TO"
  check_regression "flash_attention_causal" "${base_flash:-0}" "${flash_ns:-0}"
  check_regression "decode_argmax_surrogate" "${base_decode:-0}" "${decode_ns:-0}"
  check_regression "matmul_f32_64x64x64" "${base_matmul:-0}" "${matmul_ns:-0}"
  check_regression "matvec_dispatch_f32_4096x2048" "${base_matvec:-0}" "${matvec_ns:-0}"
fi

cp "$OUT_FILE" "$LATEST_FILE"

echo "Wrote baseline report: $OUT_FILE"
echo "Updated latest report: $LATEST_FILE"
echo "Wrote summary table: $SUMMARY_FILE"
