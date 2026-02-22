//! CUDA backend tests: verify GPU results match CPU reference within tolerance.
//!
//! Run with: cargo test -p gemma-gpu --features cuda --test cuda_backend_basic

#![cfg(feature = "cuda")]

use gemma_compression::types::Type;
use gemma_gpu::backend::Backend;
use gemma_gpu::cuda::CudaBackend;
use gemma_gpu::gemma::upload_gemma_weight;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let rel = if y.abs() > 1e-8 { diff / y.abs() } else { diff };
        assert!(
            diff <= tol || rel <= tol,
            "mismatch at index {}: {} vs {} (diff={}, rel={})",
            i,
            x,
            y,
            diff,
            rel
        );
    }
}

fn get_backend() -> Option<CudaBackend> {
    // cudarc may panic (not return Err) if the driver is too old for the
    // linked CUDA version. Catch that so tests skip instead of failing.
    match std::panic::catch_unwind(|| CudaBackend::new(0)) {
        Ok(Ok(b)) => Some(b),
        Ok(Err(e)) => {
            eprintln!("Skipping CUDA test: {e}");
            None
        }
        Err(_) => {
            eprintln!("Skipping CUDA test: cudarc panicked (driver/toolkit mismatch?)");
            None
        }
    }
}

/// Helper macro to skip test if CUDA backend is unavailable.
macro_rules! require_cuda {
    () => {
        match get_backend() {
            Some(b) => b,
            None => return,
        }
    };
}

#[test]
fn upload_download_roundtrip() {
    let backend = require_cuda!();
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut buf = backend.alloc(5).unwrap();
    backend.upload_f32(&src, &mut buf).unwrap();

    let mut dst = vec![0.0f32; 5];
    backend.download_f32(&buf, &mut dst).unwrap();
    backend.synchronize().unwrap();
    approx_eq(&src, &dst, 1e-7);
}

#[test]
fn matvec_f32() {
    let backend = require_cuda!();

    let rows = 4;
    let cols = 3;
    let weight_f32: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let weight_bytes: Vec<u8> = weight_f32.iter().flat_map(|f| f.to_ne_bytes()).collect();

    let wgt = upload_gemma_weight(&backend, &weight_bytes, Type::F32, rows, cols).unwrap();

    let x_data = vec![1.0, 0.5, -1.0];
    let mut x = backend.alloc(cols).unwrap();
    backend.upload_f32(&x_data, &mut x).unwrap();

    let mut y = backend.alloc(rows).unwrap();
    backend.matvec(&wgt, &x, &mut y).unwrap();
    backend.synchronize().unwrap();

    // CPU reference
    let mut y_ref = vec![0.0f32; rows];
    gemma_ops::matmul::matvec_dispatch(Type::F32, &weight_bytes, rows, cols, &x_data, &mut y_ref);

    let mut y_out = vec![0.0f32; rows];
    backend.download_f32(&y, &mut y_out).unwrap();
    approx_eq(&y_out, &y_ref, 1e-5);
}

#[test]
fn matvec_sfp() {
    let backend = require_cuda!();

    let rows = 4;
    let cols = 8;
    let weight_f32: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.1) - 1.5).collect();
    let mut sfp_packed = vec![0u8; rows * cols];
    gemma_compression::sfp::encode_f32(&weight_f32, &mut sfp_packed);

    let wgt = upload_gemma_weight(&backend, &sfp_packed, Type::SFP, rows, cols).unwrap();

    let x_data: Vec<f32> = (0..cols).map(|i| i as f32 * 0.5).collect();
    let mut x = backend.alloc(cols).unwrap();
    backend.upload_f32(&x_data, &mut x).unwrap();

    let mut y = backend.alloc(rows).unwrap();
    backend.matvec(&wgt, &x, &mut y).unwrap();
    backend.synchronize().unwrap();

    let mut y_ref = vec![0.0f32; rows];
    gemma_ops::matmul::matvec_dispatch(Type::SFP, &sfp_packed, rows, cols, &x_data, &mut y_ref);

    let mut y_out = vec![0.0f32; rows];
    backend.download_f32(&y, &mut y_out).unwrap();
    approx_eq(&y_out, &y_ref, 1e-5);
}

#[test]
fn rms_norm_matches() {
    let backend = require_cuda!();

    let data = vec![1.0, -2.0, 3.0, -4.0];
    let scale_data = vec![0.5, 0.5, 0.5, 0.5];
    let eps = 1e-6;

    let mut buf = backend.alloc(4).unwrap();
    backend.upload_f32(&data, &mut buf).unwrap();
    let mut scale_buf = backend.alloc(4).unwrap();
    backend.upload_f32(&scale_data, &mut scale_buf).unwrap();
    // Gemma uses (1 + scale) convention
    backend.rms_norm(&mut buf, &scale_buf, eps, true).unwrap();
    backend.synchronize().unwrap();

    let mut result = vec![0.0f32; 4];
    backend.download_f32(&buf, &mut result).unwrap();

    let mut ref_data = data.clone();
    gemma_ops::nn::rms_norm(&mut ref_data, &scale_data, eps);

    approx_eq(&result, &ref_data, 1e-5);
}

#[test]
fn softmax_matches() {
    let backend = require_cuda!();

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let mut buf = backend.alloc(4).unwrap();
    backend.upload_f32(&data, &mut buf).unwrap();
    backend.softmax(&mut buf).unwrap();
    backend.synchronize().unwrap();

    let mut result = vec![0.0f32; 4];
    backend.download_f32(&buf, &mut result).unwrap();

    let mut ref_data = data.clone();
    gemma_ops::nn::softmax(&mut ref_data);

    approx_eq(&result, &ref_data, 1e-5);
}

#[test]
fn rope_matches() {
    let backend = require_cuda!();

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let pos = 3.0;

    let mut buf = backend.alloc(8).unwrap();
    backend.upload_f32(&data, &mut buf).unwrap();
    backend.rope_inplace(&mut buf, pos).unwrap();
    backend.synchronize().unwrap();

    let mut result = vec![0.0f32; 8];
    backend.download_f32(&buf, &mut result).unwrap();

    let mut ref_data = data.clone();
    gemma_ops::nn::rope_inplace(&mut ref_data, pos);

    approx_eq(&result, &ref_data, 1e-5);
}

#[test]
fn gelu_gate_matches() {
    let backend = require_cuda!();

    let hidden_dim = 4;
    let src_data = vec![0.5, -0.3, 1.0, -1.0, 2.0, 3.0, 0.5, -0.5];

    let mut src = backend.alloc(8).unwrap();
    backend.upload_f32(&src_data, &mut src).unwrap();
    let mut dst = backend.alloc(hidden_dim).unwrap();
    backend.gelu_gate(&src, hidden_dim, &mut dst).unwrap();
    backend.synchronize().unwrap();

    let mut result = vec![0.0f32; hidden_dim];
    backend.download_f32(&dst, &mut result).unwrap();

    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
    }
    let (gate, up) = src_data.split_at(hidden_dim);
    let reference: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| gelu(g) * u)
        .collect();

    approx_eq(&result, &reference, 1e-5);
}

#[test]
fn add_inplace_works() {
    let backend = require_cuda!();

    let a_data = vec![1.0, 2.0, 3.0];
    let b_data = vec![4.0, 5.0, 6.0];

    let mut a = backend.alloc(3).unwrap();
    let mut b = backend.alloc(3).unwrap();
    backend.upload_f32(&a_data, &mut a).unwrap();
    backend.upload_f32(&b_data, &mut b).unwrap();
    backend.add_inplace(&mut a, &b).unwrap();
    backend.synchronize().unwrap();

    let mut result = vec![0.0f32; 3];
    backend.download_f32(&a, &mut result).unwrap();
    approx_eq(&result, &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn scale_inplace_works() {
    let backend = require_cuda!();

    let data = vec![2.0, 4.0, 6.0];
    let mut buf = backend.alloc(3).unwrap();
    backend.upload_f32(&data, &mut buf).unwrap();
    backend.scale_inplace(&mut buf, 0.5).unwrap();
    backend.synchronize().unwrap();

    let mut result = vec![0.0f32; 3];
    backend.download_f32(&buf, &mut result).unwrap();
    approx_eq(&result, &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn device_caps() {
    let backend = require_cuda!();
    let caps = backend.caps();
    assert!(caps.compute_major > 0, "expected a real GPU");
    println!(
        "GPU: {} (compute {}.{})",
        caps.name, caps.compute_major, caps.compute_minor
    );
}
