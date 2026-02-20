//! Scalar SFP8 codec compatible with gemma.cpp.

use half::bf16;

use crate::types::SfpStream;

pub fn sfp8_from_f32(mut f: f32) -> u8 {
    assert!((-SfpStream::K_MAX..=SfpStream::K_MAX).contains(&f));

    let mut bits = f.to_bits();
    let sign = (bits & 0x8000_0000) >> 24;
    bits &= 0x7FFF_FFFF;
    f = f.abs();

    let mut large_e = f >= 0.007568359375f32;
    let mantissa_mask = 0x007F_FFFFu32;
    let m32 = bits & mantissa_mask;
    let mut m_bits = if large_e { 3 } else { 2 };
    let is_odd = (m32 >> (23 - m_bits)) & 1;
    let round = is_odd + (1u32 << (23 - m_bits - 1)) - 1;
    let rounded = bits + round;

    if f >= 0.00732421875f32 {
        large_e = true;
        m_bits = 3;
    }

    let mut m = (mantissa_mask & rounded) >> (23 - m_bits);
    let e = ((rounded >> 23) as i32) - 127;

    if e <= -23 {
        if e < -23 {
            return 0;
        }
        if m == 0 {
            m = 1;
        }
    }

    let e_sfp = (e + if large_e { 15 } else { 23 }) as u32;
    let encoded = (e_sfp << m_bits) | m | sign;
    encoded as u8
}

pub fn f32_from_sfp8(encoded: u8) -> f32 {
    assert_ne!(encoded, 0x80, "sfp8: -0 is reserved");
    let mut sfp = encoded as u32;
    let sign32 = (sfp & 0x80) << 24;
    sfp &= 0x7F;
    if sfp == 0 {
        return 0.0;
    }

    let large_e = sfp >= 64;
    let m_bits = if large_e { 3 } else { 2 };
    let m = sfp & ((1u32 << m_bits) - 1);
    let e = sfp >> m_bits;
    let e_bias = if large_e { 15 } else { 23 };
    let exp32 = ((127 + e as i32 - e_bias) as u32) << 23;
    let mnt32 = m << (23 - m_bits);
    f32::from_bits(sign32 | exp32 | mnt32)
}

pub fn encode_f32(src: &[f32], dst: &mut [u8]) {
    assert!(dst.len() >= src.len());
    for (out, &v) in dst.iter_mut().zip(src.iter()) {
        *out = sfp8_from_f32(v);
    }
}

pub fn decode_f32(src: &[u8], dst: &mut [f32]) {
    assert!(dst.len() >= src.len());
    for (out, &v) in dst.iter_mut().zip(src.iter()) {
        *out = f32_from_sfp8(v);
    }
}

pub fn encode_bf16(src: &[bf16], dst: &mut [u8]) {
    assert!(dst.len() >= src.len());
    for (out, &v) in dst.iter_mut().zip(src.iter()) {
        *out = sfp8_from_f32(f32::from(v));
    }
}

pub fn decode_bf16(src: &[u8], dst: &mut [bf16]) {
    assert!(dst.len() >= src.len());
    for (out, &v) in dst.iter_mut().zip(src.iter()) {
        *out = bf16::from_f32(f32_from_sfp8(v));
    }
}
