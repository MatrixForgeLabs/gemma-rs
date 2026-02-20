# Planning Docs Index

This directory contains the execution and parity planning documents.

## Core planning files

- `docs/PORTING_PLAN.md`: long-form historical porting strategy and module parity background.
- `PARITY_CHECKLIST.md`: master status checklist (single source of truth for completion state).
- `reports/perf/README.md`: perf baseline + regression-gate usage.
- `reports/parity/README.md`: text parity + sampling envelope usage.

## Phase execution boards

- `docs/phases/PHASE_A.md`: text-generation parity core (sampling, prefix semantics, comparator harness).
- `docs/phases/PHASE_B.md`: API server/client and C API implementation.
- `docs/phases/PHASE_C.md`: GPU backend plan (CUDA first, GTX 1050 constraints).
- `docs/phases/PHASE_D.md`: ViT/PaliGemma and multimodal eval expansion.

Conventions:
- `Phase X` is the primary tracking level.
- `X-P0/X-P1/...` are priorities inside that phase.
