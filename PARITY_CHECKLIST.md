# gemma-rs Parity Checklist (Against gemma.cpp)

Reference: `/home/jamie/1tb/dev/cpp/gemma.cpp`  
Goal: feature-correct, test-verified parity first; performance parity second.

Legend:
- `[x]` complete
- `[~]` partial/in progress
- `[ ]` not started

## 1) Core Workspace and Architecture
- [x] Rust workspace split into module-aligned crates (`gemma-util`, `gemma-io`, `gemma-ops`, `gemma-compression`, `gemma-core`, `gemma-cli`, `gemma-threading`)
- [x] Offline build/test workflow works (`cargo build --offline`, `cargo test --offline`)
- [~] Central architecture doc synchronized with live status
- [~] CI matrix for parity/perf targets (text parity workflow added)

## 2) Module-by-Module Parity (gemma.cpp)

### `util/`
- [x] allocator/topology/threading/mat/zones baselines ported
- [~] parity-level behavior tests vs C++ util tests
- [ ] high-fidelity threading-context behavior parity

### `io/`
- [x] fields serialization path
- [x] blob_store read/write path
- [x] platform IO path
- [ ] `migrate_weights` equivalent
- [ ] `blob_compare` equivalent

### `compression/`
- [x] SFP/NUQ/I8 codec implementations
- [x] codec parity-style tests in Rust
- [~] deeper bit-level cross-validation against broad C++ golden corpora
- [ ] compression analysis/distortion utilities parity

### `ops/`
- [x] scalar dot/matmul/sum/nn baselines
- [x] dispatch coverage for compressed/f32/bf16 paths
- [~] SIMD/perf depth parity with C++ kernel set
- [ ] full kernel coverage parity (`matmul_static_*` family equivalents)

### `gemma/attention + flash_attention + kv_cache`
- [x] ring-buffer KV cache semantics
- [x] attention span planning (`start/last/clamp` + prefix plumbing)
- [x] GQA head-group KV mapping parity fix
- [x] scalar causal flash attention with soft-cap support
- [x] focused parity fixtures (windowing/rolling/prefix/soft-cap/GQA mapping)
- [~] full C++ attention behavior parity across all edge cases and batch/prefill modes
- [~] flash-attention performance parity (currently correctness-first path)

### `gemma/gemma` generation core
- [x] multi-token autoregressive decode (`max_tokens` now active)
- [x] EOS/secondary-EOS stopping
- [x] `final_cap` soft-cap path
- [x] `generate_with_prefix_end(...)` API plumbing
- [x] deterministic sampling controls (`temperature/top-k/top-p/seed`) with repeatability coverage
- [x] explicit prefill/decode split with clamped `prefix_end` position semantics and coverage

### `gemma/configs + tokenizer + model_store + tensor_info + weights`
- [x] baseline structs and loading path
- [x] tokenizer parity harness (with `gemma.cpp` helper build path)
- [x] model store/tensor shape tests
- [~] exhaustive shape/name/layout parity for all model variants
- [~] weights pipeline parity hardening

### Frontends
- [x] CLI baseline flow
- [ ] API server implementation (`/health`, `/generate`, session KV cache)
- [ ] API client implementation
- [ ] C API implementation (`init/generate/free` ABI-safe path)

### Multimodal (`gemma/vit`, `paligemma/`)
- [ ] ViT forward path
- [ ] image preprocessing helpers
- [ ] multimodal prompt integration

### Evals/Benchmarks/Golden Reports
- [x] perf microbench binaries + baseline script/report output
- [x] dedicated `gemma-evals` crate
- [~] automated Rust-vs-C++ golden prompt report (`gemma-evals` + wrapper script; depends on available/prebuilt `gemma.cpp` binary)
- [x] CI parity summary artifact wiring (GitHub Actions uploads parity markdown report)

## 3) GPU Track (Comprehensive)

### Backend Foundation
- [ ] backend abstraction layer (`CpuBackend`, `CudaBackend`)
- [ ] runtime backend selection + capability checks
- [ ] tensor/storage API that supports device placement

### CUDA/NVIDIA Path (GTX 1050 target)
- [ ] CUDA crate/toolchain integration in workspace
- [ ] device memory allocator + transfer primitives
- [ ] GPU matvec/matmul kernels (f32/bf16/sfp/i8-dequant where practical)
- [ ] GPU attention kernels (QK, softmax, weighted-V, rope)
- [ ] KV cache device representation + rolling semantics
- [ ] end-to-end decode loop on GPU with CPU fallback
- [ ] correctness tests CPU vs GPU (tolerance-based)
- [ ] perf + VRAM profiling on 2GB constraints

### Multi-backend Future (optional after CUDA)
- [ ] Vulkan/wgpu exploration
- [ ] backend-neutral kernel interfaces

## 4) What Is Already Completed (This Project Run History)
- [x] attention refactor scaffolding and reusable planning functions
- [x] head-flash reusable runner API
- [x] GQA mapping parity correction
- [x] rolling KV cache implementation + tests
- [x] flash-attention soft-cap parity fixtures
- [x] generation controls tests (`max_tokens`, EOS, prefix-end API behavior)
- [x] perf hooks + benchmark binaries + baseline report generation
- [x] perf regression-gate plumbing (`scripts/perf/run_baseline.sh` compare/threshold + summary TSV)
- [x] sampling compatibility envelope runner/report (`scripts/parity/run_sampling_envelope.sh`)

## 5) Fast Execution Phasing (Recommended)
- Canonical planning index: `docs/README.md`
- Phase execution boards:
  - `docs/phases/PHASE_A.md`
  - `docs/phases/PHASE_B.md`
  - `docs/phases/PHASE_C.md`
  - `docs/phases/PHASE_D.md`
- Historical long-form plan:
  - `docs/PORTING_PLAN.md`

Current phase status snapshot:
- `Phase A`: complete
- `Phase B`: todo
- `Phase C`: todo
- `Phase D`: todo

Execution convention:
- Track by `Phase A/B/C/D`.
- Use `A-P0`, `A-P1` (or `B-P0`, etc.) only as priorities inside each phase doc.
