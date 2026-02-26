# gemma-rs (port in progress)

Rust port of `gemma.cpp`, with CPU and optional CUDA backends.

## Quick links
- **User guide (recommended):** `docs/USER_GUIDE.md`
- **Porting notes:** `docs/PORTING_PLAN.md`
- **Model fetcher (HF/Kaggle):** `cargo run -p gemma-fetch -- --help`

## Fast start
- Build CPU: `cargo build --offline`
- Build CUDA CLI: `cargo build -p gemma-cli --features cuda`
- Chat: `cargo run -p gemma-cli -- --chat --stream --weights path/to/weights.sbs`
- Bench: `cargo run -p gemma-cli -- --bench --device cuda --weights path/to/weights.sbs`
- API server: `cargo run -p gemma-cli --features serve -- --weights path/to/weights.sbs --serve`

Run `cargo run -p gemma-cli -- --help` for full flags.
