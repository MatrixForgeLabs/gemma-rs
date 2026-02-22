//! Tests that the CPU backend (via inference-gpu) produces correct results,
//! including Gemma-specific weight upload through the helper function.

use gemma_compression::types::Type;
use gemma_gpu::backend::{Backend, OpKind};
use gemma_gpu::cpu::CpuBackend;
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
        assert!(
            diff <= tol,
            "mismatch at index {}: {} vs {} (diff={})",
            i, x, y, diff
        );
    }
}

#[test]
fn upload_download_roundtrip() {
    let backend = CpuBackend::new();
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut buf = backend.alloc(5).unwrap();
    backend.upload_f32(&src, &mut buf).unwrap();

    let mut dst = vec![0.0f32; 5];
    backend.download_f32(&buf, &mut dst).unwrap();
    assert_eq!(src, dst);
}

#[test]
fn matvec_f32() {
    let backend = CpuBackend::new();

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

    // Reference: direct gemma_ops call
    let mut y_ref = vec![0.0f32; rows];
    gemma_ops::matmul::matvec_dispatch(Type::F32, &weight_bytes, rows, cols, &x_data, &mut y_ref);

    let mut y_out = vec![0.0f32; rows];
    backend.download_f32(&y, &mut y_out).unwrap();
    approx_eq(&y_out, &y_ref, 1e-6);
}

#[test]
fn matvec_sfp() {
    let backend = CpuBackend::new();

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

    let mut y_ref = vec![0.0f32; rows];
    gemma_ops::matmul::matvec_dispatch(Type::SFP, &sfp_packed, rows, cols, &x_data, &mut y_ref);

    let mut y_out = vec![0.0f32; rows];
    backend.download_f32(&y, &mut y_out).unwrap();
    approx_eq(&y_out, &y_ref, 1e-6);
}

#[test]
fn matvec_bf16() {
    let backend = CpuBackend::new();

    let rows = 4;
    let cols = 8;
    let weight_f32: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.1) - 1.0).collect();
    let weight_bf16: Vec<half::bf16> = weight_f32
        .iter()
        .map(|&f| half::bf16::from_f32(f))
        .collect();
    let weight_bytes: Vec<u8> = weight_bf16.iter().flat_map(|b| b.to_ne_bytes()).collect();

    let wgt = upload_gemma_weight(&backend, &weight_bytes, Type::BF16, rows, cols).unwrap();

    let x_data: Vec<f32> = (0..cols).map(|i| i as f32 * 0.5).collect();
    let mut x = backend.alloc(cols).unwrap();
    backend.upload_f32(&x_data, &mut x).unwrap();

    let mut y = backend.alloc(rows).unwrap();
    backend.matvec(&wgt, &x, &mut y).unwrap();

    let mut y_ref = vec![0.0f32; rows];
    gemma_ops::matmul::matvec_dispatch(Type::BF16, &weight_bytes, rows, cols, &x_data, &mut y_ref);

    let mut y_out = vec![0.0f32; rows];
    backend.download_f32(&y, &mut y_out).unwrap();
    approx_eq(&y_out, &y_ref, 1e-6);
}

#[test]
fn rms_norm_matches() {
    let backend = CpuBackend::new();

    let data = vec![1.0, -2.0, 3.0, -4.0];
    let scale_data = vec![0.5, 0.5, 0.5, 0.5];
    let eps = 1e-6;

    let mut buf = backend.alloc(4).unwrap();
    backend.upload_f32(&data, &mut buf).unwrap();
    let mut scale_buf = backend.alloc(4).unwrap();
    backend.upload_f32(&scale_data, &mut scale_buf).unwrap();
    // Gemma uses (1 + scale) convention
    backend.rms_norm(&mut buf, &scale_buf, eps, true).unwrap();

    let mut result = vec![0.0f32; 4];
    backend.download_f32(&buf, &mut result).unwrap();

    // Reference path
    let mut ref_data = data.clone();
    gemma_ops::nn::rms_norm(&mut ref_data, &scale_data, eps);

    approx_eq(&result, &ref_data, 1e-6);
}

#[test]
fn softmax_matches() {
    let backend = CpuBackend::new();

    let data = vec![1.0, 2.0, 3.0, 4.0];

    let mut buf = backend.alloc(4).unwrap();
    backend.upload_f32(&data, &mut buf).unwrap();
    backend.softmax(&mut buf).unwrap();

    let mut result = vec![0.0f32; 4];
    backend.download_f32(&buf, &mut result).unwrap();

    let mut ref_data = data.clone();
    gemma_ops::nn::softmax(&mut ref_data);

    approx_eq(&result, &ref_data, 1e-6);
}

#[test]
fn rope_matches() {
    let backend = CpuBackend::new();

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let pos = 3.0;

    let mut buf = backend.alloc(8).unwrap();
    backend.upload_f32(&data, &mut buf).unwrap();
    backend.rope_inplace(&mut buf, pos).unwrap();

    let mut result = vec![0.0f32; 8];
    backend.download_f32(&buf, &mut result).unwrap();

    let mut ref_data = data.clone();
    gemma_ops::nn::rope_inplace(&mut ref_data, pos);

    approx_eq(&result, &ref_data, 1e-6);
}

#[test]
fn dot_matches() {
    let backend = CpuBackend::new();

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];

    let mut a = backend.alloc(4).unwrap();
    let mut b = backend.alloc(4).unwrap();
    backend.upload_f32(&a_data, &mut a).unwrap();
    backend.upload_f32(&b_data, &mut b).unwrap();

    let result = backend.dot(&a, &b).unwrap();
    let reference = gemma_ops::dot::dot_f32(&a_data, &b_data);

    assert!((result - reference).abs() < 1e-6);
}

#[test]
fn gelu_gate_matches() {
    let backend = CpuBackend::new();

    let hidden_dim = 4;
    let src_data = vec![0.5, -0.3, 1.0, -1.0, 2.0, 3.0, 0.5, -0.5];

    let mut src = backend.alloc(8).unwrap();
    backend.upload_f32(&src_data, &mut src).unwrap();
    let mut dst = backend.alloc(hidden_dim).unwrap();
    backend.gelu_gate(&src, hidden_dim, &mut dst).unwrap();

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

    approx_eq(&result, &reference, 1e-6);
}

#[test]
fn add_inplace_works() {
    let backend = CpuBackend::new();

    let a_data = vec![1.0, 2.0, 3.0];
    let b_data = vec![4.0, 5.0, 6.0];

    let mut a = backend.alloc(3).unwrap();
    let mut b = backend.alloc(3).unwrap();
    backend.upload_f32(&a_data, &mut a).unwrap();
    backend.upload_f32(&b_data, &mut b).unwrap();

    backend.add_inplace(&mut a, &b).unwrap();

    let mut result = vec![0.0f32; 3];
    backend.download_f32(&a, &mut result).unwrap();
    approx_eq(&result, &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn scale_inplace_works() {
    let backend = CpuBackend::new();

    let data = vec![2.0, 4.0, 6.0];
    let mut buf = backend.alloc(3).unwrap();
    backend.upload_f32(&data, &mut buf).unwrap();

    backend.scale_inplace(&mut buf, 0.5).unwrap();

    let mut result = vec![0.0f32; 3];
    backend.download_f32(&buf, &mut result).unwrap();
    approx_eq(&result, &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn supports_all_ops() {
    let backend = CpuBackend::new();
    let ops = [
        OpKind::Matvec,
        OpKind::MatvecHead,
        OpKind::RmsNorm,
        OpKind::Softmax,
        OpKind::Rope,
        OpKind::Dot,
        OpKind::GeluGate,
        OpKind::SiluGate,
        OpKind::AddInplace,
        OpKind::ScaleInplace,
        OpKind::FlashAttention,
    ];
    for op in ops {
        assert!(backend.supports_op(op));
    }
}
