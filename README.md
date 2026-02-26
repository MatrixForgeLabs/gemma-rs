# gemma-rs

`gemma-rs` is a Rust implementation of the `gemma.cpp` inference stack, with CPU execution and optional CUDA acceleration.

This project is being ported for use in [Octant-OS.com](https://octant-os.com).

## What this repo includes

- Multi-crate Rust workspace for model runtime, I/O, kernels, threading, compression, and CLI tooling.
- Local CLI for text generation, chat-style interaction, benchmarking, and optional serving mode.
- Model fetch utility for pulling weights from supported sources.

## Workspace crates

- `gemma-cli`: CLI entrypoint (chat, generation, benchmarking, serve mode).
- `gemma-core`: model logic, tokenizer, weight loading, high-level API surface.
- `gemma-ops`: numeric kernels and neural network primitives.
- `gemma-gpu`: CUDA backend integration.
- `gemma-io`: serialization, field/blob I/O.
- `gemma-compression`: SFP/NUQ/int-format codec work.
- `gemma-threading`: threading abstractions.
- `gemma-util`: low-level utilities.
- `gemma-fetch`: model artifact fetch/download helper.
- `gemma-evals`: evaluation harnesses and related tooling.

## Prerequisites

- Rust toolchain (stable, Rust 2021 edition compatible).
- Optional: CUDA toolkit + compatible GPU driver for CUDA features.
- Local model weights (`.sbs`, `.gguf`, etc.) for inference.

## Build

```bash
cargo build --offline
```

Build CLI with CUDA support:

```bash
cargo build -p gemma-cli --features cuda
```

## Run

CLI help:

```bash
cargo run -p gemma-cli -- --help
```

Chat-style run:

```bash
cargo run -p gemma-cli -- --chat --stream --weights path/to/weights.sbs
```

Benchmark:

```bash
cargo run -p gemma-cli -- --bench --weights path/to/weights.sbs
```

Serve mode:

```bash
cargo run -p gemma-cli --features serve -- --weights path/to/weights.sbs --serve
```

Model fetcher help:

```bash
cargo run -p gemma-fetch -- --help
```

## Test

Run workspace tests:

```bash
cargo test --offline
```

Run a focused test target:

```bash
cargo test -p gemma-core --test tokenizer_parity --offline
```

## Project notes

- Large model artifacts are expected to stay local.
- Generated reports/docs/scripts can be kept locally and are ignored by default.
