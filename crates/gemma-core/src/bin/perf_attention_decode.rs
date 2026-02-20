use gemma_core::bench::{ns_per_iter, BenchTimer};
use gemma_core::flash_attention::flash_attention_causal;
use gemma_core::kv_cache::KVCache;

fn fill_deterministic(buf: &mut [f32], mut state: u64) {
    for v in buf.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 33) as u32) as f32 / (u32::MAX as f32);
        *v = x * 2.0 - 1.0;
    }
}

fn parse_iters() -> usize {
    std::env::var("GEMMA_PERF_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(20_000)
}

fn main() {
    let iters = parse_iters();
    let seq_len = 1024usize;
    let kv_heads = 8usize;
    let head_dim = 256usize;
    let query_scale = 1.0 / (head_dim as f32).sqrt();
    let mut cache = KVCache::new(seq_len, kv_heads, head_dim);

    let mut q = vec![0.0f32; head_dim];
    let mut k = vec![0.0f32; head_dim];
    let mut v = vec![0.0f32; head_dim];
    let mut out = vec![0.0f32; head_dim];
    fill_deterministic(&mut q, 1);

    for pos in 0..seq_len {
        fill_deterministic(&mut k, 100 + pos as u64);
        fill_deterministic(&mut v, 200 + pos as u64);
        cache.write_k(pos, pos % kv_heads, &k);
        cache.write_v(pos, pos % kv_heads, &v);
    }

    let timer = BenchTimer::start();
    for i in 0..iters {
        let pos = 64 + (i % (seq_len - 64));
        let kv_head = i % kv_heads;
        flash_attention_causal(
            &q,
            &cache,
            kv_head,
            pos - 64,
            pos,
            query_scale,
            Some(30.0),
            &mut out,
        );
        std::hint::black_box(out[0]);
    }
    let elapsed = timer.elapsed();
    println!(
        "flash_attention_causal: iters={iters} total_ms={:.3} ns_per_iter={:.1}",
        elapsed.as_secs_f64() * 1e3,
        ns_per_iter(elapsed, iters),
    );

    let vocab = 32768usize;
    let model_dim = 2048usize;
    let mut hidden = vec![0.0f32; model_dim];
    let mut row = vec![0.0f32; model_dim];
    fill_deterministic(&mut hidden, 42);
    let timer = BenchTimer::start();
    for i in 0..(iters / 100).max(1) {
        let mut best = f32::NEG_INFINITY;
        let mut best_idx = 0usize;
        for tok in 0..vocab {
            fill_deterministic(&mut row, tok as u64 ^ i as u64);
            let mut sum = 0.0f32;
            for j in 0..model_dim {
                sum += hidden[j] * row[j];
            }
            if sum > best {
                best = sum;
                best_idx = tok;
            }
        }
        std::hint::black_box(best_idx);
    }
    let elapsed = timer.elapsed();
    let decode_iters = (iters / 100).max(1);
    println!(
        "decode_argmax_surrogate: iters={decode_iters} total_ms={:.3} ns_per_iter={:.1}",
        elapsed.as_secs_f64() * 1e3,
        ns_per_iter(elapsed, decode_iters),
    );
}
