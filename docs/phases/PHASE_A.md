# Phase A - Text Generation Parity Core

Objective: lock reliable text-generation parity against `gemma.cpp` before API/GPU/multimodal work.
Current status: `complete` (A-P0/A-P1/A-P2 delivered).

Legend:
- Priority: `A-P0` (highest), `A-P1`, `A-P2`
- Effort: `S` (0.5-1 day), `M` (1-3 days), `L` (3-7 days)
- Status: `done`, `in_progress`, `todo`

## A-P0

1. `A-P0-M` Deterministic sampling parity (`temperature/top-k/top-p/seed`)
- Status: `done`
- Scope: core sampling API, CLI flags, deterministic-seed tests
- Key files:
  - `crates/gemma-core/src/gemma.rs`
  - `crates/gemma-core/src/gemma_args.rs`
  - `crates/gemma-cli/src/main.rs`
  - `crates/gemma-core/tests/generation_controls.rs`
- Exit criteria: fixed-seed repeatability + unchanged argmax defaults

2. `A-P0-M` Explicit prefill/decode split + full `prefix_end` semantics
- Status: `done`
- Scope: split flows, position-aware prefix clamping, cache/window interactions
- Key files:
  - `crates/gemma-core/src/gemma.rs`
  - `crates/gemma-core/src/attention.rs`
  - `crates/gemma-core/src/flash_attention.rs`
  - `crates/gemma-core/tests/attention_parity.rs`
- Exit criteria: tests cover `prefix_end > pos`, rolling clamp, window interactions

3. `A-P0-M` Golden comparator harness vs `gemma.cpp`
- Status: `done`
- Scope: one-command Rust-vs-C++ prompt comparison + mismatch report
- Key files:
  - `crates/gemma-evals/*`
  - `scripts/parity/*`
  - `reports/parity/*`
  - `.github/workflows/parity.yml`
- Exit criteria: report artifact + nonzero mismatch gating in CI

## A-P1

4. `A-P1-S` Attention/flash fixture expansion
- Status: `done`
- Scope: richer GQA/window/prefix/cache edge-case matrix
- Key files:
  - `crates/gemma-core/tests/attention_parity.rs`
  - `crates/gemma-core/src/flash_attention.rs`

5. `A-P1-M` Decode/perf instrumentation hardening
- Status: `done`
- Scope: normalize baseline runs, store trend reports, define regression thresholds
- Key files:
  - `crates/gemma-core/src/bench.rs`
  - `crates/gemma-core/src/bin/perf_attention_decode.rs`
  - `crates/gemma-ops/src/bin/perf_matmul_decode.rs`
  - `scripts/perf/run_baseline.sh`
  - `reports/perf/*`

## A-P2

6. `A-P2-M` Sampling quality compatibility envelope
- Status: `done`
- Scope: sampled-sequence compatibility checks against `gemma.cpp`
- Key files:
  - `scripts/parity/run_sampling_envelope.sh`
  - `scripts/parity/sampling_configs.tsv`
  - `scripts/parity/sampling_prompts.txt`
  - `reports/parity/README.md`
  - `.github/workflows/parity.yml`
