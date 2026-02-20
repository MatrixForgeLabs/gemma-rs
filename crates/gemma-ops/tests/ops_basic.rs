use gemma_compression::int_format;
use gemma_compression::nuq;
use gemma_compression::sfp;
use gemma_compression::types::{I8Stream, NuqStream, Type};
use gemma_ops::dot::dot_f32;
use gemma_ops::matmul::{matmul_dispatch, matmul_f32, matmul_f32_blocked};
use gemma_ops::sum::sum_f32;

#[test]
fn test_dot() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    assert_eq!(dot_f32(&a, &b), 32.0);
}

#[test]
fn test_sum() {
    let a = [1.0f32, 2.0, 3.0];
    assert_eq!(sum_f32(&a), 6.0);
}

#[test]
fn test_matmul() {
    let a = [1.0f32, 2.0, 3.0, 4.0]; // 2x2
    let b = [5.0f32, 6.0, 7.0, 8.0]; // 2x2
    let mut c = [0.0f32; 4];
    matmul_f32(&a, &b, &mut c, 2, 2, 2);
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_blocked() {
    let m = 3;
    let n = 3;
    let k = 3;
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = [9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut c = vec![0.0f32; m * n];
    let mut ref_c = vec![0.0f32; m * n];
    matmul_f32_blocked(&a, &b, &mut c, m, n, k, 2);
    matmul_f32(&a, &b, &mut ref_c, m, n, k);
    assert_eq!(c, ref_c);
}

#[test]
fn test_matmul_dispatch_compressed() {
    let m = 2;
    let n = 3;
    let k = 4;
    let a = [
        0.5f32, -0.25, 0.75, 0.125, //
        -0.5, 0.25, -0.75, -0.125,
    ];
    let b = [
        0.25f32, -0.5, 0.75, //
        -0.25, 0.5, -0.75, //
        0.125, -0.125, 0.25, //
        -0.5, 0.25, -0.25,
    ];

    let mut expected = vec![0.0f32; m * n];
    matmul_f32(&a, &b, &mut expected, m, n, k);

    // SFP
    let mut sfp_packed = vec![0u8; b.len()];
    sfp::encode_f32(&b, &mut sfp_packed);
    let mut out = vec![0.0f32; m * n];
    matmul_dispatch(Type::SFP, &a, &sfp_packed, &mut out, m, n, k);
    let mut sfp_dec = vec![0.0f32; b.len()];
    sfp::decode_f32(&sfp_packed, &mut sfp_dec);
    let mut sfp_expected = vec![0.0f32; m * n];
    matmul_f32(&a, &sfp_dec, &mut sfp_expected, m, n, k);
    assert_eq!(out, sfp_expected);

    // NUQ
    let nuq_len = NuqStream::packed_end(b.len());
    let mut nuq_packed = vec![NuqStream { byte: 0 }; nuq_len];
    nuq::encode_f32(&b, &mut nuq_packed, 0);
    let nuq_bytes =
        unsafe { std::slice::from_raw_parts(nuq_packed.as_ptr() as *const u8, nuq_packed.len()) };
    matmul_dispatch(Type::NUQ, &a, nuq_bytes, &mut out, m, n, k);
    let mut nuq_dec = vec![0.0f32; b.len()];
    nuq::decompress_and_zero_pad_f32(&nuq_packed, 0, &mut nuq_dec, b.len());
    let mut nuq_expected = vec![0.0f32; m * n];
    matmul_f32(&a, &nuq_dec, &mut nuq_expected, m, n, k);
    assert_eq!(out, nuq_expected);

    // I8
    let i8_len = I8Stream::packed_end(b.len());
    let mut i8_packed = vec![I8Stream { i: 0 }; i8_len];
    int_format::encode_f32(&b, &mut i8_packed, 0);
    let i8_bytes =
        unsafe { std::slice::from_raw_parts(i8_packed.as_ptr() as *const u8, i8_packed.len()) };
    matmul_dispatch(Type::I8, &a, i8_bytes, &mut out, m, n, k);
    let mut i8_dec = vec![0.0f32; b.len()];
    int_format::decompress_and_zero_pad_f32(&i8_packed, 0, &mut i8_dec, b.len());
    let mut i8_expected = vec![0.0f32; m * n];
    matmul_f32(&a, &i8_dec, &mut i8_expected, m, n, k);
    assert_eq!(out, i8_expected);

    // F32 passthrough still matches baseline.
    let b_bytes = unsafe {
        std::slice::from_raw_parts(
            b.as_ptr() as *const u8,
            b.len() * core::mem::size_of::<f32>(),
        )
    };
    matmul_dispatch(Type::F32, &a, b_bytes, &mut out, m, n, k);
    assert_eq!(out, expected);
}
