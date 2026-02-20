# gemma-rs (port in progress)

This is a Rust port scaffold of `gemma.cpp`.

## Status

- Core IO, serialization, utilities, ops: in place (baseline scalar).
- Compression: placeholder codecs (not bit-compatible yet).
- Model config, tensor registry, model store, weights: in place (minimal).
- Tokenizer: kitoken-based, parity test harness exists (requires gemma.cpp + SBS).
- CLI: supports encode/decode + stubbed generation.
- Frontends (API server/client, C API), benchmarks: placeholders.

## Build

```sh
cargo build --offline
```

## Tests

```sh
cargo test --offline
```

## CLI

```sh
cargo run -p gemma-cli -- "hello world" [vocab_file]
```
