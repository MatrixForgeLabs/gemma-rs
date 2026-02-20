# gemma-rs User Guide

Comprehensive instructions for building, running, and benchmarking the Rust port of `gemma.cpp`.

## 1) Requirements
- **Rust**: stable toolchain (2021 edition). Install via `rustup`.
- **CUDA (optional)**: CUDA 12.x+ with NVRTC present at runtime. Works on:
  - Volta+ (compute ≥7.0) via cuBLAS SGEMV.
  - Pascal (compute 6.x) via custom matvec kernel (cuBLAS may be unavailable on recent CUDA).
- **GPU memory**: small demo weights fit in ~1 GB; larger models need more VRAM.
- **Weights**: SBS format. A demo checkpoint is included at the repo root:
  `gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs`.

## 2) Build
- CPU-only (offline capable):
  ```sh
  cargo build --offline
  ```
- CUDA-enabled CLI:
  ```sh
  cargo build -p gemma-cli --features cuda
  ```
  Use `CUDA_VISIBLE_DEVICES` or `--device cuda:<ordinal>` to pick a GPU.

## 3) CLI quickstart
- Single prompt (CPU):
  ```sh
  cargo run -p gemma-cli -- --weights path/to/weights.sbs --prompt "hello" --max-tokens 64
  ```
- GPU decode (ordinal 0):
  ```sh
  cargo run -p gemma-cli --features cuda -- --device cuda --weights path/to/weights.sbs --prompt "hello"
  ```
- Chat loop (optional streaming):
  ```sh
  cargo run -p gemma-cli -- --chat --stream --weights path/to/weights.sbs
  ```
- Encode / decode utilities:
  ```sh
  cargo run -p gemma-cli -- --encode "text here"
  cargo run -p gemma-cli -- --decode "1 2 3"
  ```
- Inspect model:
  ```sh
  cargo run -p gemma-cli -- --weights path/to/weights.sbs --print-config
  cargo run -p gemma-cli -- --weights path/to/weights.sbs --print-backend
  ```

## 3b) Model downloaders (Hugging Face & Kaggle)
Binary: `cargo run -p gemma-fetch -- <options>`

- Download a Kaggle model (latest version):
  ```sh
  # Preferred: API token from Kaggle account page
  echo "KAGGLE_API_TOKEN=your_token_string" > .env   # or export in your shell
  cargo run -p gemma-fetch -- --kaggle google/gemma-3/gguf/gemma-3-1b-it-qat-q4_0 --out models/
  ```
  Use `--version N` to pin a specific version. Legacy creds (`KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`) still work for compatibility.

- Download a Hugging Face file:
  ```sh
  export HF_TOKEN=your_token   # optional for public repos
  cargo run -p gemma-fetch -- --hf-repo google/gemma-3 --hf-file gguf/gemma-3-1b-it-q4_0.gguf --out models/
  ```

- Search:
  ```sh
  cargo run -p gemma-fetch -- --search-hf gemma --limit 10
  cargo run -p gemma-fetch -- --search-kaggle gemma --limit 10
  ```

### Sampling flags
`--temperature`, `--top-k`, `--top-p`, `--seed` control generation. Defaults: temp = 1.0, top‑k = 1, top‑p = 1.0.

### Device selection
`--device cpu` (default) or `--device cuda[:ordinal]`. On GPU, logits matvec and argmax run on device; only the chosen token index is copied back.

### Streaming
`--stream` prints tokens as they are generated in chat mode.

## 4) Benchmarks
Run quick tok/s measurements:
```sh
cargo run -p gemma-cli -- --bench --prompt "hello" --max-tokens 64 --weights path/to/weights.sbs
```
- With CUDA: add `--device cuda`.
- Control runs: `--bench-warmup <n>` (default 1), `--bench-iters <n>` (default 3).
- JSON report: `--bench-json bench.json` writes mean/p50/p90 tok/s plus device info (for CUDA: name, compute capability, free/total MB; for CPU: device=cpu).

## 5) GPU specifics
- **Logits & sampling on device**: logits matvec and argmax execute on GPU; no full-logit downloads are needed. Soft-cap is monotonic, so argmax remains correct without host post-processing.
- **Fallback toggle**: `GEMMA_GPU_CPU_LOGITS=1` forces logits to be downloaded and sampled on CPU (useful for debugging).
- **Device info**: printed at startup and included in CUDA bench JSON.

## 6) Known limitations
- KV cache is per-turn only (no persistence across chat turns yet).
- Advanced sampling (top‑k/top‑p) currently runs on CPU; GPU-side multinomial is not yet wired.
- Compression codecs aim for parity with `gemma.cpp` but are still being validated.

## 7) Testing
- Full suite (offline):
  ```sh
  cargo test --offline
  ```
- Targeted example:
  ```sh
  cargo test -p gemma-core --test tokenizer_parity --offline
  ```

## 8) Repository layout (selected)
- `crates/gemma-cli`: CLI entrypoint and benchmarks.
- `crates/gemma-core`: model logic, tokenizer, GPU planner.
- `crates/gemma-gpu`: CUDA backend (kernels, argmax reduction).
- `crates/gemma-ops`, `gemma-util`, `gemma-io`, `gemma-compression`: math, utilities, IO, codecs.
- `docs/PORTING_PLAN.md`: upstream parity notes.

## 9) Troubleshooting
- **Missing CUDA/NVRTC**: build with `--features cuda` fails → ensure CUDA 12+ runtime is installed and visible on PATH/LD_LIBRARY_PATH.
- **GPU OOM**: reduce `--max-tokens`, switch to CPU, or pick a smaller weights file.
- **Slow CPU perf**: try release build `cargo run -p gemma-cli --release -- ...` or use `--bench` for calibrated numbers.
- **Logits issues**: set `GEMMA_GPU_CPU_LOGITS=1` to force host sampling; re-run with `RUST_BACKTRACE=1` for stack traces.

## 10) Typical workflows
- **Baseline CPU generation**: quick functional check without GPU.
- **GPU perf check**: `--bench --device cuda --bench-json bench.json` to capture device + tok/s.
- **Interactive chat**: `--chat --stream --device cuda` for lowest latency.
