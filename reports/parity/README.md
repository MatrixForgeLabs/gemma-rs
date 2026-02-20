# Parity Reports

This directory stores Rust-vs-`gemma.cpp` comparison outputs.

## Generate text parity report (shell wrapper)

```sh
scripts/parity/run_text_parity.sh
```

Wrapper environment variables:
- `GEMMA_CPP_DIR` (default `/home/jamie/1tb/dev/cpp/gemma.cpp`)
- `GEMMA_CPP_BUILD_DIR` (default `/tmp/gemma-cpp-build-rust-parity`)
- `CPP_BUILD_IF_MISSING` (`1` default, set `0` to require prebuilt C++ binary)
- `GEMMA_CPP_BIN` (default `$GEMMA_CPP_BUILD_DIR/gemma`)
- `GEMMA_SBS_PATH` (default local 270M `.sbs`)
- `PROMPTS_FILE` (default `scripts/parity/prompts.txt`)
- `MAX_TOKENS` (default `32`)
- `TEMPERATURE` (default `1.0`)
- `TOP_K` (default `1`)
- `TOP_P` (default `1.0`)
- `SEED` (optional)

## Generate sampling compatibility envelope

```sh
scripts/parity/run_sampling_envelope.sh
```

Sampling envelope env vars:
- `CONFIGS_FILE` (default `scripts/parity/sampling_configs.tsv`)
- `PROMPTS_FILE` (default `scripts/parity/sampling_prompts.txt`)
- `MAX_TOKENS` (default `32`)
- `MIN_EXACT_MATCH_RATE` (default `0.60`)
- `CPP_BUILD_IF_MISSING`, `GEMMA_CPP_BIN`, `GEMMA_CPP_DIR`, `GEMMA_CPP_BUILD_DIR`
- `GEMMA_SBS_PATH`

Output:
- `reports/parity/sampling-envelope-<UTC timestamp>.md`
- `reports/parity/sampling-envelope-latest.md`

## Run native eval binary directly

```sh
cargo run -p gemma-evals -- \
  --weights /path/to/model.sbs \
  --prompts scripts/parity/prompts.txt \
  --cpp-bin /path/to/gemma \
  --report reports/parity/text-parity-manual.md \
  --max-tokens 32 \
  --temperature 1.0 \
  --top-k 1 \
  --top-p 1.0
```

Exit codes:
- `0` all prompts matched
- `2` one or more prompt mismatches
- `69` dependency missing/unavailable (`gemma.cpp` binary failed/missing)
  - For `run_sampling_envelope.sh`, exit `2` means exact-match rate below `MIN_EXACT_MATCH_RATE`.

## CI wiring

- GitHub Actions workflow: `.github/workflows/parity.yml`
- Behavior:
  - builds `google/gemma.cpp` CLI reference
  - runs `gemma-evals` against `scripts/parity/prompts.txt`
  - runs sampling compatibility envelope against `scripts/parity/sampling_prompts.txt`
  - uploads `reports/parity/ci-text-parity.md` as artifact
  - uploads `reports/parity/ci-sampling-envelope.md` as artifact
  - fails the job on mismatches (nonzero `gemma-evals` exit)
