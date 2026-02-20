//! Scalar dot-product helpers (no SIMD yet).

use gemma_compression::types::Type;

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    assert_eq!(a.len(), b.len());
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;
    let len = a.len();
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut total = tmp.iter().sum::<f32>();
    for j in i..len {
        total += a[j] * b[j];
    }
    total
}

pub fn dot_bf16(a: &[half::bf16], b: &[half::bf16]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += f32::from(a[i]) * f32::from(b[i]);
    }
    sum
}

pub fn condition_number_f32(a: &[f32], b: Option<&[f32]>) -> f64 {
    if let Some(bv) = b {
        let mut sum = 0.0f64;
        let mut sum_abs = 0.0f64;
        for i in 0..a.len() {
            let v = a[i] as f64 * bv[i] as f64;
            sum += v;
            sum_abs += v.abs();
        }
        let denom = sum.abs();
        if denom == 0.0 {
            return f64::INFINITY;
        }
        2.0 * sum_abs / denom
    } else {
        let mut sum = 0.0f64;
        let mut sum_abs = 0.0f64;
        for &v in a {
            let vd = v as f64;
            sum += vd;
            sum_abs += vd.abs();
        }
        let denom = sum.abs();
        if denom == 0.0 {
            return f64::INFINITY;
        }
        2.0 * sum_abs / denom
    }
}

pub fn dot_dispatch(ty: Type, a: &[u8], b: &[u8]) -> f32 {
    match ty {
        Type::F32 => {
            let a = cast_slice::<f32>(a);
            let b = cast_slice::<f32>(b);
            dot_f32(a, b)
        }
        Type::BF16 => {
            let a = cast_slice::<half::bf16>(a);
            let b = cast_slice::<half::bf16>(b);
            dot_bf16(a, b)
        }
        _ => panic!("dot_dispatch: unsupported type"),
    }
}

fn cast_slice<T: Copy>(bytes: &[u8]) -> &[T] {
    assert_eq!(bytes.len() % core::mem::size_of::<T>(), 0);
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const T,
            bytes.len() / core::mem::size_of::<T>(),
        )
    }
}
