# Perf Baseline 20260218T144620Z

Environment
- Host: Linux archbox 6.17.9-arch1-1 #1 SMP PREEMPT_DYNAMIC Mon, 24 Nov 2025 15:21:09 +0000 x86_64 GNU/Linux
- Rust: rustc 1.91.1 (ed61e7d7e 2025-11-07)
- Cargo: cargo 1.91.1 (ea2d97820 2025-10-10)

## gemma-core attention/decode
```
flash_attention_causal: iters=60 total_ms=1.548 ns_per_iter=25798.1
decode_argmax_surrogate: iters=1 total_ms=154.213 ns_per_iter=154212758.0
```

## gemma-ops matmul/decode
```
matmul_f32_64x64x64: iters=6 total_ms=0.316 ns_per_iter=52692.7
matvec_dispatch_f32_4096x2048: iters=1 total_ms=4.395 ns_per_iter=4394943.0
```
