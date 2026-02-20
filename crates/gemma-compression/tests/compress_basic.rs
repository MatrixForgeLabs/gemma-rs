use gemma_compression::compress::{compress_f32, decompress_f32};
use gemma_compression::int_format::{decompress_and_zero_pad_f32, encode_f32};
use gemma_compression::types::I8Stream;

#[test]
fn test_f32_roundtrip() {
    let src = [1.0f32, -2.0, 3.5];
    let mut dst = [0.0f32; 3];
    compress_f32(&src, &mut dst);
    let mut out = [0.0f32; 3];
    decompress_f32(&dst, &mut out);
    assert_eq!(src, out);
}

#[test]
fn test_i8_roundtrip() {
    let src = [0.0f32, 1.0, -1.0, 0.5];
    let packed_len = I8Stream::packed_end(src.len());
    let mut packed = vec![I8Stream { i: 0 }; packed_len];
    encode_f32(&src, &mut packed, 0);
    let mut out = vec![0.0f32; src.len()];
    decompress_and_zero_pad_f32(&packed, 0, &mut out, src.len());
    for i in 0..src.len() {
        assert!((src[i] - out[i]).abs() <= 0.2);
    }
}
