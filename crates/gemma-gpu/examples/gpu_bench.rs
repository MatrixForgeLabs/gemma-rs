//! Benchmark: compare matvec performance between CPU and CUDA backends.
//!
//! Run with: cargo run -p gemma-gpu --features cuda --example gpu_bench --release
//!
//! Tests at Gemma 270M dimensions: model_dim=1536, ff_hidden_dim=6144

use gemma_gpu::backend::Backend;
use gemma_gpu::cpu::CpuBackend;
use std::time::Instant;

const WARMUP: usize = 10;
const ITERS: usize = 100;

fn bench_matvec<B: Backend>(backend: &B, name: &str, weight: &B::Wgt, x: &B::Buf, y: &mut B::Buf) {
    // Warmup
    for _ in 0..WARMUP {
        backend.matvec(weight, x, y).unwrap();
    }
    backend.synchronize().unwrap();

    let start = Instant::now();
    for _ in 0..ITERS {
        backend.matvec(weight, x, y).unwrap();
    }
    backend.synchronize().unwrap();
    let elapsed = start.elapsed();

    let per_iter = elapsed / ITERS as u32;
    println!(
        "  {name}: {ITERS} iters in {:.2}ms ({:.1}us/iter)",
        elapsed.as_secs_f64() * 1000.0,
        per_iter.as_secs_f64() * 1e6,
    );
}

fn main() {
    println!("=== gemma-gpu matvec benchmark ===\n");

    // Gemma 270M dimensions
    let configs = [
        ("qkv_ein (1536 -> 1536)", 1536, 1536),
        ("gating_ein (1536 -> 12288)", 12288, 1536),
        ("linear_w (6144 -> 1536)", 1536, 6144),
    ];

    // CPU backend
    let cpu = CpuBackend::new();
    println!("CPU Backend:");

    for (label, rows, cols) in &configs {
        let weight_f32: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 1000) as f32 - 500.0) * 0.001)
            .collect();
        let wgt = cpu
            .upload_weight_f32(&weight_f32, *rows, *cols)
            .unwrap();
        let x_data: Vec<f32> = (0..*cols).map(|i| (i as f32) * 0.001).collect();
        let mut x = cpu.alloc(*cols).unwrap();
        cpu.upload_f32(&x_data, &mut x).unwrap();
        let mut y = cpu.alloc(*rows).unwrap();

        bench_matvec(&cpu, label, &wgt, &x, &mut y);
    }

    // CUDA backend
    #[cfg(feature = "cuda")]
    {
        use gemma_gpu::cuda::CudaBackend;

        match CudaBackend::new(0) {
            Ok(cuda) => {
                let caps = cuda.caps();
                println!(
                    "\nCUDA Backend: {} (compute {}.{}):",
                    caps.name, caps.compute_major, caps.compute_minor
                );

                for (label, rows, cols) in &configs {
                    let weight_f32: Vec<f32> = (0..rows * cols)
                        .map(|i| ((i % 1000) as f32 - 500.0) * 0.001)
                        .collect();
                    let wgt = cuda
                        .upload_weight_f32(&weight_f32, *rows, *cols)
                        .unwrap();
                    let x_data: Vec<f32> = (0..*cols).map(|i| (i as f32) * 0.001).collect();
                    let mut x = cuda.alloc(*cols).unwrap();
                    cuda.upload_f32(&x_data, &mut x).unwrap();
                    let mut y = cuda.alloc(*rows).unwrap();

                    bench_matvec(&cuda, label, &wgt, &x, &mut y);
                }
            }
            Err(e) => {
                println!("\nCUDA Backend: unavailable ({e})");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\nCUDA Backend: not compiled (enable --features cuda)");
    }

    println!("\nDone.");
}
