//! Scalar matmul kernels (baseline correctness).

use gemma_compression::int_format;
use gemma_compression::nuq;
use gemma_compression::sfp;
use gemma_compression::types::{I8Stream, NuqStream, Type};
use half::bf16;
use rayon::prelude::*;

pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    if m * n * k >= 64 * 64 * 64 {
        matmul_f32_blocked(a, b, c, m, n, k, 64);
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

pub fn matmul_f32_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    c.fill(0.0);
    let bm = block;
    let bn = block;
    let bk = block;
    let mut i0 = 0;
    while i0 < m {
        let i_max = (i0 + bm).min(m);
        let mut k0 = 0;
        while k0 < k {
            let k_max = (k0 + bk).min(k);
            let mut j0 = 0;
            while j0 < n {
                let j_max = (j0 + bn).min(n);
                for i in i0..i_max {
                    for p in k0..k_max {
                        let a_ip = a[i * k + p];
                        let b_row = &b[p * n + j0..p * n + j_max];
                        let c_row = &mut c[i * n + j0..i * n + j_max];
                        for j in 0..b_row.len() {
                            c_row[j] += a_ip * b_row[j];
                        }
                    }
                }
                j0 += bn;
            }
            k0 += bk;
        }
        i0 += bm;
    }
}

pub fn matmul_dispatch(
    ty: Type,
    a: &[f32],
    b_packed: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(c.len(), m * n);
    let b = decode_b_to_f32(ty, b_packed, k, n);
    matmul_f32(a, &b, c, m, n, k);
}

pub fn matvec_dispatch(
    ty: Type,
    b_packed: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    y: &mut [f32],
) {
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    match ty {
        Type::F32 => {
            let b = cast_slice::<f32>(b_packed);
            assert_eq!(b.len(), rows * cols);
            #[cfg(feature = "blas")]
            {
                // C (rows x 1) = A(rows x cols) * B(cols x 1)
                let m = rows as isize;
                let k = cols as isize;
                // matrixmultiply is row-major: row stride is number of columns.
                matrixmultiply::sgemm(m, k, 1, 1.0, b, k, 1, x, 1, k, 0.0, y, 1, rows as isize);
            }
            #[cfg(not(feature = "blas"))]
            {
                if rows >= 64 {
                    y.par_iter_mut().enumerate().for_each(|(r, out)| {
                        let row = &b[r * cols..(r + 1) * cols];
                        *out = dot_row(row, x);
                    });
                } else {
                    for r in 0..rows {
                        let row = &b[r * cols..(r + 1) * cols];
                        y[r] = dot_row(row, x);
                    }
                }
            }
        }
        Type::BF16 => {
            let b = cast_slice::<bf16>(b_packed);
            assert_eq!(b.len(), rows * cols);
            if rows >= 64 {
                y.par_iter_mut().enumerate().for_each(|(r, out)| {
                    let row = &b[r * cols..(r + 1) * cols];
                    let mut sum = 0.0f32;
                    for i in 0..cols {
                        sum += f32::from(row[i]) * x[i];
                    }
                    *out = sum;
                });
            } else {
                for r in 0..rows {
                    let row = &b[r * cols..(r + 1) * cols];
                    let mut sum = 0.0f32;
                    for i in 0..cols {
                        sum += f32::from(row[i]) * x[i];
                    }
                    y[r] = sum;
                }
            }
        }
        Type::SFP => {
            assert!(b_packed.len() >= rows * cols);
            let mut row_buf = vec![0.0f32; cols];
            for r in 0..rows {
                let start = r * cols;
                let end = start + cols;
                sfp::decode_f32(&b_packed[start..end], &mut row_buf);
                y[r] = dot_row(&row_buf, x);
            }
        }
        Type::NUQ => {
            let packed = cast_slice::<NuqStream>(b_packed);
            let mut row_buf = vec![0.0f32; cols];
            for r in 0..rows {
                nuq::decompress_and_zero_pad_f32(packed, r * cols, &mut row_buf, cols);
                y[r] = dot_row(&row_buf, x);
            }
        }
        Type::I8 => {
            let packed = cast_slice::<I8Stream>(b_packed);
            let mut row_buf = vec![0.0f32; cols];
            for r in 0..rows {
                int_format::decompress_and_zero_pad_f32(packed, r * cols, &mut row_buf, cols);
                y[r] = dot_row(&row_buf, x);
            }
        }
        _ => panic!("matvec_dispatch: unsupported type"),
    }
}

fn decode_b_to_f32(ty: Type, b_packed: &[u8], k: usize, n: usize) -> Vec<f32> {
    let num = k * n;
    match ty {
        Type::F32 => {
            let s = cast_slice::<f32>(b_packed);
            assert_eq!(s.len(), num);
            s.to_vec()
        }
        Type::BF16 => {
            let s = cast_slice::<bf16>(b_packed);
            assert_eq!(s.len(), num);
            s.iter().map(|v| f32::from(*v)).collect()
        }
        Type::SFP => {
            assert!(b_packed.len() >= num);
            let mut out = vec![0.0f32; num];
            sfp::decode_f32(&b_packed[..num], &mut out);
            out
        }
        Type::NUQ => {
            let packed = cast_slice::<NuqStream>(b_packed);
            let mut out = vec![0.0f32; num];
            nuq::decompress_and_zero_pad_f32(packed, 0, &mut out, num);
            out
        }
        Type::I8 => {
            let packed = cast_slice::<I8Stream>(b_packed);
            let mut out = vec![0.0f32; num];
            int_format::decompress_and_zero_pad_f32(packed, 0, &mut out, num);
            out
        }
        _ => panic!("matmul_dispatch: unsupported type"),
    }
}

fn dot_row(row: &[f32], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature check.
            return unsafe { dot_row_avx2(row, x) };
        }
    }
    let mut sum = 0.0f32;
    for i in 0..row.len() {
        sum += row[i] * x[i];
    }
    sum
}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_row_avx2(row: &[f32], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;
    let len = row.len();
    while i + 8 <= len {
        let vr = _mm256_loadu_ps(row.as_ptr().add(i));
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(vr, vx));
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut total = tmp.iter().sum::<f32>();
    for j in i..len {
        total += row[j] * x[j];
    }
    total
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
