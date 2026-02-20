use gemma_compression::types::Type;
use gemma_ops::matmul::{matmul_f32, matvec_dispatch};
use std::time::Instant;

fn fill_deterministic(buf: &mut [f32], mut state: u64) {
    for v in buf.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 33) as u32) as f32 / (u32::MAX as f32);
        *v = x * 2.0 - 1.0;
    }
}

fn ns_per_iter(elapsed: std::time::Duration, iters: usize) -> f64 {
    if iters == 0 {
        0.0
    } else {
        elapsed.as_nanos() as f64 / iters as f64
    }
}

fn parse_iters() -> usize {
    std::env::var("GEMMA_PERF_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(500)
}

fn main() {
    let iters = parse_iters();

    let (m, n, k) = (64usize, 64usize, 64usize);
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    fill_deterministic(&mut a, 1);
    fill_deterministic(&mut b, 2);

    let timer = Instant::now();
    for _ in 0..iters {
        matmul_f32(&a, &b, &mut c, m, n, k);
        std::hint::black_box(c[0]);
    }
    let elapsed = timer.elapsed();
    println!(
        "matmul_f32_64x64x64: iters={iters} total_ms={:.3} ns_per_iter={:.1}",
        elapsed.as_secs_f64() * 1e3,
        ns_per_iter(elapsed, iters),
    );

    let rows = 4096usize;
    let cols = 2048usize;
    let mut packed_f32 = vec![0.0f32; rows * cols];
    let mut x = vec![0.0f32; cols];
    let mut y = vec![0.0f32; rows];
    fill_deterministic(&mut packed_f32, 3);
    fill_deterministic(&mut x, 4);
    let packed_bytes = unsafe {
        std::slice::from_raw_parts(
            packed_f32.as_ptr() as *const u8,
            packed_f32.len() * core::mem::size_of::<f32>(),
        )
    };

    let timer = Instant::now();
    for _ in 0..(iters / 4).max(1) {
        matvec_dispatch(Type::F32, packed_bytes, rows, cols, &x, &mut y);
        std::hint::black_box(y[0]);
    }
    let matvec_iters = (iters / 4).max(1);
    let elapsed = timer.elapsed();
    println!(
        "matvec_dispatch_f32_4096x2048: iters={matvec_iters} total_ms={:.3} ns_per_iter={:.1}",
        elapsed.as_secs_f64() * 1e3,
        ns_per_iter(elapsed, matvec_iters),
    );
}
