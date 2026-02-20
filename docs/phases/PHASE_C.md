# Phase C - GPU Backend (GTX 1050 First)

Objective: deliver a practical CUDA-backed inference path for low-VRAM cards while preserving CPU parity.

Legend:
- Priority: `C-P0` (highest), `C-P1`, `C-P2`
- Status: `done`, `in_progress`, `todo`

## C-P0

1. `C-P0-M` Backend abstraction
- Status: `todo`
- Scope: `CpuBackend`/`CudaBackend` split with runtime selection

2. `C-P0-M` Device-aware tensor/storage layer
- Status: `todo`
- Scope: placement-aware buffers and safe host/device transfers

3. `C-P0-M` CUDA toolchain + allocator bring-up
- Status: `todo`
- Scope: build integration, GPU memory manager, error/reporting path

## C-P1

4. `C-P1-L` GPU kernel path for decode-critical ops
- Status: `todo`
- Scope: matvec/matmul + attention substeps (QK/softmax/V) where wins exist

5. `C-P1-M` GPU KV cache representation
- Status: `todo`
- Scope: rolling cache semantics on device with CPU fallback behavior

## C-P2

6. `C-P2-M` CPU-vs-GPU correctness suite
- Status: `todo`
- Scope: tolerance-based parity checks across fixed prompt fixtures

7. `C-P2-M` 2GB VRAM profiling and constraints
- Status: `todo`
- Scope: memory/perf baselines for GTX 1050 class hardware
