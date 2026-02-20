use half::bf16;

use gemma_compression::int_format::{decompress_and_zero_pad_f32, encode_f32};
use gemma_compression::types::I8Stream;

fn read_bf16(bytes: &[u8], offset: usize) -> bf16 {
    let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    bf16::from_bits(bits)
}

#[test]
fn i8_layout_and_basic_quantization() {
    let mut input = [0.0f32; I8Stream::K_GROUP_SIZE];
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as f32;
    }

    let packed_len = I8Stream::packed_end(input.len());
    let mut packed = vec![I8Stream { i: 0 }; packed_len];
    encode_f32(&input, &mut packed, 0);

    let bytes = unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len()) };

    let scale_f = 255.0f32 / 127.0;
    let inv_scale = bf16::from_f32(1.0 / scale_f);
    let zeropoint = bf16::from_f32(-128.0);

    assert_eq!(read_bf16(bytes, 0), inv_scale);
    assert_eq!(read_bf16(bytes, 2), zeropoint);

    let data_offset = 4;
    let q0 = bytes[data_offset] as i8 as i32;
    let q1 = bytes[data_offset + 1] as i8 as i32;
    let q2 = bytes[data_offset + 2] as i8 as i32;
    let q127 = bytes[data_offset + 127] as i8 as i32;
    assert_eq!(q0, -128);
    assert!(q1 >= -126 && q1 <= -125);
    assert!(q2 >= -124 && q2 <= -123);
    assert_eq!(q127, 127);
}

#[test]
fn i8_decompress_and_zero_pad() {
    let input = [0.0f32; I8Stream::K_GROUP_SIZE];
    let packed_len = I8Stream::packed_end(input.len());
    let mut packed = vec![I8Stream { i: 0 }; packed_len];
    encode_f32(&input, &mut packed, 0);

    let mut out = vec![1.0f32; I8Stream::K_GROUP_SIZE + 8];
    decompress_and_zero_pad_f32(&packed, 0, &mut out, I8Stream::K_GROUP_SIZE);
    assert!(out[I8Stream::K_GROUP_SIZE..].iter().all(|&v| v == 0.0));
}
