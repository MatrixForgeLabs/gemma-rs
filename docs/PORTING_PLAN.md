# gemma.cpp → gemma-rs Porting Plan

Date: 2026-01-31

This document lays out a phased, test-driven plan to port the C++ Gemma
inference engine (`/home/jamie/1tb/dev/gemma.cpp`) to Rust in this repository
(`/home/jamie/1tb/dev/rust/gemma-rs`). The goal is a maintainable, high-performance
Rust implementation with feature parity where practical, preserving file formats
and expected outputs.

## Guiding Principles

- **Parity first, then polish**: prioritize behavioral parity and correctness
  over premature refactors.
- **Test-first migration**: port tests/goldens early; rely on cross-checks with
  the C++ reference.
- **Modular crates**: keep boundaries aligned with the C++ modules to make
  mapping and verification straightforward.
- **Performance conscious**: measure early, enable SIMD and multithreading
  incrementally, and keep performance deltas visible.

## Inventory Summary (C++ Reference)

Key modules in `gemma.cpp`:

- `util/`: allocator, threading, topology, matrices, zones, basics
- `io/`: fields, blob_store, platform IO (mmap/parallel read)
- `ops/`: dot, matmul, sum + mixed-precision kernels
- `compression/`: sfp/nuq/int formats and integration
- `gemma/`: configs, tokenizer, kv_cache, attention, flash_attention, vit,
  model_store, weights
- `paligemma/`: image helpers for PaliGemma
- Frontends: CLI, API server/client, C API bindings
- Tests + benchmarks: `evals/`, `ops/*_test.cc`, etc.

External deps:
- Highway SIMD
- sentencepiece tokenizer
- nlohmann/json
- cpp-httplib
- benchmark

## Progress Update (2026-02-01)

High-level status based on this repo (not the C++ reference):

- Workspace crates exist: `gemma-util`, `gemma-io`, `gemma-ops`,
  `gemma-compression`, `gemma-core`, `gemma-cli`, `gemma-threading`.
- Core IO + serialization are implemented with basic tests
  (`fields`, `blob_store`, `io`).
- Ops are baseline scalar (dot/matmul/sum/nn) with basic tests.
- Compression has placeholder codecs (not bit-compatible yet).
- Model config, tensor registry, model store, and weights stubs exist.
- Tokenizer uses `kitoken` and has a parity test harness; model inference
  is still a stub.
- CLI exists for tokenizer encode/decode and stubbed generation.
- API server/client, C API, benchmarks are placeholders/stubs.

## Phase 1 — Workspace Skeleton + Test Harness

Status:
- [x] Workspace crates created (core/io/ops/compression/util/cli/threading)
- [~] Golden tests/testdata strategy (basic tests exist; no goldens imported yet;
  see `crates/gemma-io/tests/fields_golden.rs`,
  `crates/gemma-io/tests/blob_store_roundtrip.rs`,
  `crates/gemma-ops/tests/ops_basic.rs`,
  `crates/gemma-ops/tests/nn_basic.rs`,
  `crates/gemma-compression/tests/compress_basic.rs`)
- [ ] Shared types crate (if needed)
- [ ] SIMD + tokenizer strategy decisions finalized

**Goal**: Create a workspace mirroring module boundaries and install a test
harness with golden/reference outputs.

Deliverables:
- Cargo workspace layout:
  - `gemma-core`
  - `gemma-ops`
  - `gemma-compression`
  - `gemma-io`
  - `gemma-util`
  - `gemma-cli`
  - `gemma-threading`
  - optional `gemma-capi`, `gemma-api`, `gemma-bench`, `gemma-py`
- Shared types crate if needed (e.g., `gemma-types`)
- Golden tests: copy `goldens/` and minimal `testdata/` references or
  point to the C++ repo path to avoid duplication.
- CI-friendly test entry point (Rust `cargo test` target)

Open questions to resolve early:
- SIMD strategy: `std::arch`, `portable_simd`, or FFI to Highway?
- Tokenizer: Rust crate or FFI to sentencepiece?

## Phase 2 — Utilities + IO Foundation

Status:
- [~] `util/` baseline (see `crates/gemma-util/src/threading.rs`,
  `crates/gemma-util/src/topology.rs`,
  `crates/gemma-util/src/mat.rs`,
  `crates/gemma-util/src/zones.rs`)
- [x] `io/` baseline (`fields`, `blob_store`, platform IO in
  `crates/gemma-io/src/fields.rs`, `crates/gemma-io/src/blob_store.rs`,
  `crates/gemma-io/src/io.rs`)
- [ ] File format compatibility tests vs C++

**Goal**: Establish foundational utilities and binary IO formats.

Deliverables:
- Port `util/`:
  - memory allocator strategy
  - threading context + topology mapping
  - matrix utilities
  - zones/profiling hooks
- Port `io/`:
  - `fields` serialization compatibility
  - `blob_store` and platform IO (mmap/parallel read)
- File format compatibility tests vs C++ (hash checks + sample loads)

## Phase 3 — Numeric Kernels + Mixed Precision

Status:
- [x] Scalar kernels for dot/matmul/sum/nn with tests
  (`crates/gemma-ops/src/dot.rs`, `crates/gemma-ops/src/matmul.rs`,
  `crates/gemma-ops/src/sum.rs`, `crates/gemma-ops/src/nn.rs`,
  `crates/gemma-ops/tests/ops_basic.rs`, `crates/gemma-ops/tests/nn_basic.rs`)
- [ ] Types for bf16/fp8/nuq/i8 aligned with compression formats
- [ ] SIMD path selected and benchmarked
- [ ] Golden comparisons vs C++

**Goal**: Functional parity of `ops/` with correctness tests before speed.

Deliverables:
- Scalar kernels for dot/matmul/sum with correctness tests
- Types for bf16/fp8/nuq/i8 aligned with compression formats
- SIMD implementation path selected and benchmarked
- Golden comparisons vs C++ (per-op and integrated into a micro model)

## Phase 4 — Compression + Weights + Tokenizer

Status:
- [~] Compression codecs scaffolded (placeholders, not bit-compatible;
  `crates/gemma-compression/src/sfp.rs`,
  `crates/gemma-compression/src/nuq.rs`,
  `crates/gemma-compression/src/int_format.rs`,
  `crates/gemma-compression/tests/compress_basic.rs`)
- [~] Weights/model_store minimal (`crates/gemma-core/src/weights.rs`,
  `crates/gemma-core/src/model_store.rs`,
  `crates/gemma-core/tests/model_store_roundtrip.rs`)
- [~] Tokenizer implemented (kitoken) with parity harness
  (`crates/gemma-core/src/tokenizer.rs`,
  `crates/gemma-core/tests/tokenizer_parity.rs`)
- [ ] Compression integrated into matmul path

**Goal**: Enable full model weight loading path.

Deliverables:
- Port `compression/` formats and helpers
- Integrate compression into matmul path
- Port `weights` and `model_store` logic
- Tokenizer implementation or FFI bridge with sentencepiece

## Phase 5 — Model Core

Status:
- [~] Configs/attention/kv_cache/vit stubs present
  (`crates/gemma-core/src/configs.rs`,
  `crates/gemma-core/src/attention.rs`,
  `crates/gemma-core/src/flash_attention.rs`,
  `crates/gemma-core/src/kv_cache.rs`,
  `crates/gemma-core/src/vit.rs`,
  `crates/gemma-core/src/gemma.rs`)
- [ ] Inference loop implemented
- [ ] End-to-end inference tests with goldens

**Goal**: Bring model inference to feature parity.

Deliverables:
- Port `gemma/` core:
  - configs
  - attention/flash_attention
  - kv_cache
  - inference loop
  - vit + paligemma hooks (if in scope)
- End-to-end inference tests using known prompts + golden outputs

## Phase 6 — Frontends + APIs + Benchmarks

Status:
- [~] CLI exists (encode/decode + stub generation;
  `crates/gemma-cli/src/main.rs`)
- [ ] API server/client
- [ ] C API
- [ ] Benchmarks/evals

**Goal**: User-facing parity and perf evaluation.

Deliverables:
- CLI inference (`gemma/run.cc` equivalent)
- API server/client (optional in first pass)
- C API (optional)
- Benchmarks and eval tools

## Phase 7 — Performance + Polish

Status:
- [ ] Profiling + optimization passes
- [ ] Threading/allocator tuning
- [ ] SIMD kernel refinements
- [ ] Docs + build instructions

**Goal**: Closing the performance gap to C++.

Deliverables:
- Profiling + optimization passes
- Threading, allocator tuning
- SIMD kernel refinements
- Docs + build instructions

## Risk & Mitigation

- **SIMD portability**: choose an approach that allows incremental optimization
  with safe fallbacks.
- **Tokenizer parity**: ensure identical tokenization or document mismatch.
- **Binary formats**: strict compatibility tests; fail-fast on version mismatch.
- **Performance regressions**: track benchmarks across phases.

## Immediate Next Actions

1) Replace placeholder compression codecs (SFP/NUQ/int) with bit-compatible
   implementations and add C++ parity tests.
2) Expand weight loading and tensor registry to cover full model layouts.
3) Wire compression-aware matmul and ops into model execution path.
4) Replace the stubbed inference loop with a minimal end-to-end forward pass.
5) Add golden prompt tests against gemma.cpp to lock behavioral parity.

## Strict Parity Matrix Snapshot (2026-02-18)

This snapshot is a strict module-by-module comparison against
`/home/jamie/1tb/dev/cpp/gemma.cpp`.

| C++ module | Status | C++ reference files | Rust mapping files | Gap summary |
|---|---|---|---|---|
| `util/` | Partial | `util/allocator.cc`, `util/mat.cc`, `util/threading.cc` | `crates/gemma-util/src/allocator.rs`, `crates/gemma-util/src/mat.rs`, `crates/gemma-util/src/threading.rs`, `crates/gemma-util/src/topology.rs`, `crates/gemma-util/src/zones.rs` | Core pieces are ported; util test parity is incomplete. |
| `io/` core | Partial | `io/io.cc`, `io/fields.cc`, `io/blob_store.cc` | `crates/gemma-io/src/io.rs`, `crates/gemma-io/src/fields.rs`, `crates/gemma-io/src/blob_store.rs` | Main path works; `migrate_weights`/`blob_compare` equivalents are missing. |
| `ops/` | Partial | `ops/dot-inl.h`, `ops/matmul.cc`, `ops/ops-inl.h` | `crates/gemma-ops/src/dot.rs`, `crates/gemma-ops/src/matmul.rs`, `crates/gemma-ops/src/nn.rs`, `crates/gemma-ops/src/sum.rs` | Correctness path exists; mostly scalar baseline, limited SIMD depth. |
| `compression/` | Partial | `compression/sfp-inl.h`, `compression/nuq-inl.h`, `compression/int-inl.h` | `crates/gemma-compression/src/sfp.rs`, `crates/gemma-compression/src/nuq.rs`, `crates/gemma-compression/src/int_format.rs` | Core codecs are implemented with parity tests; analysis/distortion helpers not ported. |
| `gemma/configs` | Partial | `gemma/configs.cc` | `crates/gemma-core/src/configs.rs` | Structural port present; full behavior parity not complete. |
| `gemma/tokenizer` | Partial | `gemma/tokenizer.cc` | `crates/gemma-core/src/tokenizer.rs`, `crates/gemma-core/tests/tokenizer_parity.rs` | Works; cross-check requires `GEMMA_CPP_DIR` setup. |
| `gemma/model_store + tensor_info + weights` | Partial | `gemma/model_store.cc`, `gemma/tensor_info.cc`, `gemma/weights.cc` | `crates/gemma-core/src/model_store.rs`, `crates/gemma-core/src/tensor_info.rs`, `crates/gemma-core/src/weights.rs` | Load and shape plumbing exists; weights surface remains minimal. |
| `gemma/gemma` inference core | Partial | `gemma/gemma.cc` | `crates/gemma-core/src/gemma.rs`, `crates/gemma-core/tests/minimal_inference.rs` | Scalar forward/generate path exists; not at full feature/perf parity. |
| `gemma/attention` | Partial | `gemma/attention.cc` | `crates/gemma-core/src/attention.rs`, `crates/gemma-core/src/gemma.rs` | Standalone module is minimal; logic is simplified/scalar. |
| `gemma/flash_attention` | Missing | `gemma/flash_attention.cc` | `crates/gemma-core/src/flash_attention.rs` | Placeholder module. |
| `gemma/kv_cache` | Missing | `gemma/kv_cache.cc` | `crates/gemma-core/src/kv_cache.rs` | Placeholder module; only ad-hoc cache inside `gemma.rs`. |
| `gemma/vit` + `paligemma/` | Missing | `gemma/vit.cc`, `paligemma/image.cc` | `crates/gemma-core/src/vit.rs` | ViT placeholder; no Rust `paligemma` module yet. |
| API/C bindings | Missing | `gemma/api_server.cc`, `gemma/api_client.cc`, `gemma/bindings/*` | `crates/gemma-core/src/api_server.rs`, `crates/gemma-core/src/api_client.rs`, `crates/gemma-core/src/c_api.rs` | Placeholder modules. |
| CLI (`run.cc`, args) | Partial | `gemma/run.cc`, `gemma/gemma_args.h` | `crates/gemma-cli/src/main.rs`, `crates/gemma-core/src/gemma_args.rs` | CLI works for basic flow; args/config surface is still minimal. |
| `evals/`, `python/`, `tools/` | Missing | `evals/*`, `python/*`, `tools/tokenizer_dump.cc` | no direct Rust equivalents | No Rust-native eval harness, Python bindings, or tokenizer dump tool. |

## Top 5 Highest-Impact Parity Tasks

1) Implement `flash_attention` and wire it into `gemma.rs` inference path.
   - C++ refs: `gemma/flash_attention.cc`, `gemma/flash_attention_test.cc`
   - Rust target: `crates/gemma-core/src/flash_attention.rs`
   - Exit criteria: parity tests for attention outputs across fixed seeds/prompts.

2) Replace placeholder `kv_cache` module with production implementation.
   - C++ refs: `gemma/kv_cache.cc`, `gemma/kv_cache.h`
   - Rust target: `crates/gemma-core/src/kv_cache.rs` + integration in `crates/gemma-core/src/gemma.rs`
   - Exit criteria: cache correctness tests across multi-token decoding.

3) Promote inference from minimal scalar baseline to parity-checked path.
   - C++ refs: `gemma/gemma.cc`, `gemma/attention.cc`, `gemma/weights.cc`
   - Rust targets: `crates/gemma-core/src/gemma.rs`, `crates/gemma-core/src/weights.rs`
   - Exit criteria: golden prompt outputs match C++ within agreed tolerance.

4) Implement API server/client and C API frontends.
   - C++ refs: `gemma/api_server.cc`, `gemma/api_client.cc`, `gemma/bindings/*`
   - Rust targets: `crates/gemma-core/src/api_server.rs`, `crates/gemma-core/src/api_client.rs`, `crates/gemma-core/src/c_api.rs`
   - Exit criteria: request/response parity smoke tests + C ABI integration test.

5) Build a Rust-native eval harness and golden parity runner.
   - C++ refs: `evals/gemma_test.cc`, `evals/benchmark.cc`, `evals/run_mmlu.cc`
   - Rust targets: new eval crate (recommended `crates/gemma-evals`) + `crates/gemma-core/tests/*`
   - Exit criteria: automated parity report against `gemma.cpp` on a fixed prompt set.
