use std::time::Instant;

use gemma_compression::types::Type;
use gemma_ops::matmul::matmul_f32;
use gemma_ops::matmul::matvec_dispatch;

#[test]
fn matmul_perf_sanity() {
    if std::env::var("GEMMA_PERF_CHECK").ok().as_deref() != Some("1") {
        eprintln!("GEMMA_PERF_CHECK not set; skipping matmul perf test");
        return;
    }

    let m = 128usize;
    let n = 128usize;
    let k = 128usize;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.02).collect();
    let mut out = vec![0.0f32; m * n];
    let max_ms: u128 = std::env::var("GEMMA_PERF_MAX_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000);

    let start = Instant::now();
    matmul_f32(&a, &b, &mut out, m, n, k);
    let elapsed = start.elapsed().as_millis();
    assert!(
        elapsed <= max_ms,
        "matmul_f32 took {elapsed}ms, max {max_ms}ms"
    );
}

#[test]
fn matvec_perf_sanity() {
    if std::env::var("GEMMA_PERF_CHECK").ok().as_deref() != Some("1") {
        eprintln!("GEMMA_PERF_CHECK not set; skipping matvec perf test");
        return;
    }

    let rows = 4096usize;
    let cols = 1024usize;
    let b: Vec<f32> = (0..rows * cols).map(|i| (i % 97) as f32 * 0.01).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i % 89) as f32 * 0.02).collect();
    let mut y = vec![0.0f32; rows];
    let max_ms: u128 = std::env::var("GEMMA_PERF_MATVEC_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000);

    let start = Instant::now();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            b.as_ptr() as *const u8,
            b.len() * core::mem::size_of::<f32>(),
        )
    };
    matvec_dispatch(Type::F32, bytes, rows, cols, &x, &mut y);
    let elapsed = start.elapsed().as_millis();
    assert!(
        elapsed <= max_ms,
        "matvec_f32 took {elapsed}ms, max {max_ms}ms"
    );
}
