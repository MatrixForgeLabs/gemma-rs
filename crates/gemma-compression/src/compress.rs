//! Minimal compression helpers (placeholder implementations).

use half::bf16;

use crate::types::Type;

pub fn compress_f32(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

pub fn decompress_f32(src: &[f32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

pub fn compress_bf16(src: &[bf16], dst: &mut [bf16]) {
    assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

pub fn decompress_bf16(src: &[bf16], dst: &mut [bf16]) {
    assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

pub fn compress_dispatch(ty: Type, src: &[u8], dst: &mut [u8]) {
    match ty {
        Type::F32 => {
            let s = cast_slice::<f32>(src);
            let d = cast_slice_mut::<f32>(dst);
            compress_f32(s, d);
        }
        Type::BF16 => {
            let s = cast_slice::<bf16>(src);
            let d = cast_slice_mut::<bf16>(dst);
            compress_bf16(s, d);
        }
        _ => panic!("compress_dispatch: unsupported type"),
    }
}

pub fn decompress_dispatch(ty: Type, src: &[u8], dst: &mut [u8]) {
    match ty {
        Type::F32 => {
            let s = cast_slice::<f32>(src);
            let d = cast_slice_mut::<f32>(dst);
            decompress_f32(s, d);
        }
        Type::BF16 => {
            let s = cast_slice::<bf16>(src);
            let d = cast_slice_mut::<bf16>(dst);
            decompress_bf16(s, d);
        }
        _ => panic!("decompress_dispatch: unsupported type"),
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

fn cast_slice_mut<T: Copy>(bytes: &mut [u8]) -> &mut [T] {
    assert_eq!(bytes.len() % core::mem::size_of::<T>(), 0);
    unsafe {
        std::slice::from_raw_parts_mut(
            bytes.as_mut_ptr() as *mut T,
            bytes.len() / core::mem::size_of::<T>(),
        )
    }
}
