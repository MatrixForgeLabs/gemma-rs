# Repository Guidelines

## Project Structure & Module Organization
This repository is a Rust workspace for a `gemma.cpp` port in progress. Workspace crates live in `crates/`:
- `gemma-util`: low-level utilities (allocators, args, threading helpers, math helpers).
- `gemma-io`: serialization, field I/O, blob storage.
- `gemma-ops`: core numeric ops and NN primitives.
- `gemma-compression`: SFP/NUQ/int-format codec work.
- `gemma-core`: model logic, tokenizer, weights, API stubs.
- `gemma-cli`: command-line entrypoint.
- `gemma-threading`: threading abstractions.

Integration tests are colocated per crate in `crates/<crate>/tests/`. Large model artifacts (`*.tar.gz`, `.sbs`) are kept at repo root for local parity/testing workflows.

## Build, Test, and Development Commands
- `cargo build --offline`: build all workspace crates without network access.
- `cargo test --offline`: run full workspace tests.
- `cargo test -p gemma-core --test tokenizer_parity --offline`: run one integration test target.
- `cargo run -p gemma-cli -- "hello world" [vocab_file]`: run the CLI.
- `cargo check --workspace --offline`: fast validation during iteration.

## Coding Style & Naming Conventions
Use Rust 2021 idioms and keep formatting `rustfmt`-clean (`cargo fmt --all`). Prefer `snake_case` for functions/modules/files, `PascalCase` for types/traits, and `SCREAMING_SNAKE_CASE` for constants. Keep modules focused and crate boundaries explicit (utility vs IO vs ops vs core).

## Testing Guidelines
Use Rust’s built-in test framework (`cargo test`) with integration tests in `crates/*/tests`. Follow existing descriptive test names such as `tokenizer_parity.rs`, `model_store_roundtrip.rs`, and `matmul_perf.rs`. Add regression tests for bug fixes and parity tests when porting behavior from upstream `gemma.cpp`.

## Commit & Pull Request Guidelines
No commit history exists yet, so follow Conventional Commits (e.g., `feat(core): add kv-cache bounds check`, `fix(io): validate field length`). Keep commits small and scoped to one concern.

PRs should include:
- A concise problem/solution description.
- Linked issue or task reference (if available).
- Test evidence (`cargo test --offline` output summary, or targeted test command).
- Notes on model artifacts or external prerequisites needed to reproduce.

## Security & Configuration Tips
Do not commit secrets, local credentials, or generated `target/` output. Treat large model files as local assets unless explicitly required; prefer documenting download/setup steps over committing new binaries.
