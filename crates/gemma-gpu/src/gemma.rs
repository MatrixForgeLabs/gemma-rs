//! Gemma-specific helpers for bridging `gemma_compression::types::Type`
//! to `inference_gpu::backend::WeightFormat` and uploading Gemma weights.

use gemma_compression::types::Type;
use inference_gpu::backend::{Backend, Result, WeightFormat};

/// Convert a Gemma compression `Type` to the model-agnostic `WeightFormat`.
pub fn weight_format_from_gemma(ty: Type) -> WeightFormat {
    match ty {
        Type::F32 => WeightFormat::F32,
        Type::BF16 => WeightFormat::BF16,
        Type::SFP => WeightFormat::SFP,
        Type::NUQ => WeightFormat::NUQ,
        Type::I8 => WeightFormat::I8,
        other => panic!("unsupported Gemma type for GPU upload: {:?}", other),
    }
}

/// Upload a Gemma weight matrix to any `Backend`.
///
/// Handles decompression of Gemma-specific formats (SFP, NUQ, I8) to f32
/// on the host before uploading. F32 and BF16 weights are uploaded via
/// `upload_weight_packed` which handles the conversion internally.
pub fn upload_gemma_weight<B: Backend>(
    backend: &B,
    packed: &[u8],
    ty: Type,
    rows: usize,
    cols: usize,
) -> Result<B::Wgt> {
    match ty {
        // Standard formats: delegate to upload_weight_packed
        Type::F32 | Type::BF16 => {
            let fmt = weight_format_from_gemma(ty);
            backend.upload_weight_packed(packed, fmt, rows, cols)
        }
        // Gemma compressed formats: decompress to f32 on host, then upload
        Type::SFP | Type::NUQ | Type::I8 => {
            let f32_data = decompress_gemma_to_f32(packed, ty, rows, cols);
            backend.upload_weight_f32(&f32_data, rows, cols)
        }
        other => panic!("upload_gemma_weight: unsupported type {:?}", other),
    }
}

/// Decompress Gemma weight data to f32 on the host.
///
/// This is the decompression path originally in `gemma-gpu`'s CUDA backend,
/// now extracted as a standalone function so both CPU and CUDA backends
/// can share it.
fn decompress_gemma_to_f32(packed: &[u8], ty: Type, rows: usize, cols: usize) -> Vec<f32> {
    let n = rows * cols;
    let mut out = vec![0.0f32; n];
    match ty {
        Type::F32 => {
            let src = unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const f32, n) };
            out.copy_from_slice(src);
        }
        Type::BF16 => {
            let src =
                unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const half::bf16, n) };
            for (i, &v) in src.iter().enumerate() {
                out[i] = f32::from(v);
            }
        }
        Type::SFP => {
            for r in 0..rows {
                let start = r * cols;
                let end = start + cols;
                gemma_compression::sfp::decode_f32(&packed[start..end], &mut out[start..end]);
            }
        }
        Type::NUQ => {
            let packed_nuq = unsafe {
                std::slice::from_raw_parts(
                    packed.as_ptr() as *const gemma_compression::types::NuqStream,
                    packed.len(),
                )
            };
            for r in 0..rows {
                gemma_compression::nuq::decompress_and_zero_pad_f32(
                    packed_nuq,
                    r * cols,
                    &mut out[r * cols..(r + 1) * cols],
                    cols,
                );
            }
        }
        Type::I8 => {
            let packed_i8 = unsafe {
                std::slice::from_raw_parts(
                    packed.as_ptr() as *const gemma_compression::types::I8Stream,
                    packed.len(),
                )
            };
            for r in 0..rows {
                gemma_compression::int_format::decompress_and_zero_pad_f32(
                    packed_i8,
                    r * cols,
                    &mut out[r * cols..(r + 1) * cols],
                    cols,
                );
            }
        }
        _ => panic!("decompress_gemma_to_f32: unsupported type {:?}", ty),
    }
    out
}
