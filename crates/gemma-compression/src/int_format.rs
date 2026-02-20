//! Int8 quantization helpers compatible with gemma.cpp layout.

use half::bf16;

use crate::types::I8Stream;

const BF16_BYTES: usize = core::mem::size_of::<bf16>();
const GROUP_BYTES: usize = (2 * BF16_BYTES) + I8Stream::K_GROUP_SIZE;

fn group_byte_offset(packed_ofs: usize) -> usize {
    (packed_ofs / I8Stream::K_GROUP_SIZE) * GROUP_BYTES
}

fn packed_bytes(packed: &[I8Stream]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len()) }
}

fn packed_bytes_mut(packed: &mut [I8Stream]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(packed.as_mut_ptr() as *mut u8, packed.len()) }
}

fn read_bf16(bytes: &[u8], offset: usize) -> bf16 {
    let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    bf16::from_bits(bits)
}

fn write_bf16(bytes: &mut [u8], offset: usize, value: bf16) {
    let bits = value.to_bits().to_le_bytes();
    bytes[offset] = bits[0];
    bytes[offset + 1] = bits[1];
}

fn round_ties_to_even(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let floor = x.floor();
    let frac = x - floor;
    if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

pub fn quantize_group_f32(raw: &[f32], packed: &mut [I8Stream], packed_ofs: usize) {
    assert!(packed_ofs % I8Stream::K_GROUP_SIZE == 0);
    assert!(!raw.is_empty());

    let bytes = packed_bytes_mut(packed);

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &v in raw {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    let mut range = max_val - min_val;
    if range == 0.0 {
        range = 1.0;
    }

    let scale_f = 255.0 / range;
    let zeropoint_f = ((-scale_f * min_val - 128.0).trunc() as i32) as f32;

    let _scale = bf16::from_f32(scale_f);
    let inv_scale = bf16::from_f32(1.0 / scale_f);
    let zeropoint = bf16::from_f32(zeropoint_f);

    let base = group_byte_offset(packed_ofs);
    write_bf16(bytes, base, inv_scale);
    write_bf16(bytes, base + BF16_BYTES, zeropoint);

    let current_offset = base + 2 * BF16_BYTES + (packed_ofs % I8Stream::K_GROUP_SIZE);
    let g_num = raw.len().min(I8Stream::K_GROUP_SIZE);
    for i in 0..g_num {
        let q = round_ties_to_even(scale_f * raw[i] + zeropoint_f)
            .clamp(i8::MIN as f32, i8::MAX as f32);
        bytes[current_offset + i] = q as i8 as u8;
    }
}

pub fn encode_f32(raw: &[f32], packed: &mut [I8Stream], packed_ofs: usize) {
    assert!(packed_ofs % I8Stream::K_GROUP_SIZE == 0);
    if raw.is_empty() {
        return;
    }

    let mut current_ofs = packed_ofs;
    let mut idx = 0;
    while idx < raw.len() {
        let g_num = (raw.len() - idx).min(I8Stream::K_GROUP_SIZE);
        quantize_group_f32(&raw[idx..idx + g_num], packed, current_ofs);
        current_ofs += g_num;
        idx += g_num;
    }
}

pub fn dequantize_group_f32(packed: &[I8Stream], packed_ofs: usize, dst: &mut [f32]) {
    if dst.is_empty() {
        return;
    }
    let bytes = packed_bytes(packed);
    let base = group_byte_offset(packed_ofs);
    let inv_scale = read_bf16(bytes, base).to_f32();
    let zeropoint = read_bf16(bytes, base + BF16_BYTES).to_f32();
    let zeroscale = -zeropoint * inv_scale;
    let current_offset = base + 2 * BF16_BYTES + (packed_ofs % I8Stream::K_GROUP_SIZE);

    let g_num = dst.len().min(I8Stream::K_GROUP_SIZE);
    for i in 0..g_num {
        let q = bytes[current_offset + i] as i8 as f32;
        dst[i] = inv_scale * q + zeroscale;
    }
}

pub fn dequantize_group_bf16(packed: &[I8Stream], packed_ofs: usize, dst: &mut [bf16]) {
    if dst.is_empty() {
        return;
    }
    let bytes = packed_bytes(packed);
    let base = group_byte_offset(packed_ofs);
    let inv_scale = read_bf16(bytes, base).to_f32();
    let zeropoint = read_bf16(bytes, base + BF16_BYTES).to_f32();
    let zeroscale = -zeropoint * inv_scale;
    let current_offset = base + 2 * BF16_BYTES + (packed_ofs % I8Stream::K_GROUP_SIZE);

    let g_num = dst.len().min(I8Stream::K_GROUP_SIZE);
    for i in 0..g_num {
        let q = bytes[current_offset + i] as i8 as f32;
        dst[i] = bf16::from_f32(inv_scale * q + zeroscale);
    }
}

pub fn decompress_and_zero_pad_f32(
    packed: &[I8Stream],
    packed_ofs: usize,
    dst: &mut [f32],
    num: usize,
) {
    if num == 0 {
        return;
    }
    assert!(dst.len() >= num);
    dst[num..].fill(0.0);

    let mut current_ofs = packed_ofs;
    let mut out_idx = 0usize;
    let mut remaining = num;

    if current_ofs % I8Stream::K_GROUP_SIZE != 0 {
        let within_group = current_ofs % I8Stream::K_GROUP_SIZE;
        let remaining_in_group = I8Stream::K_GROUP_SIZE - within_group;
        let num_in_first = remaining.min(remaining_in_group);
        dequantize_group_f32(
            packed,
            current_ofs,
            &mut dst[out_idx..out_idx + num_in_first],
        );
        current_ofs += num_in_first;
        out_idx += num_in_first;
        remaining -= num_in_first;
    }

    while remaining >= I8Stream::K_GROUP_SIZE {
        dequantize_group_f32(
            packed,
            current_ofs,
            &mut dst[out_idx..out_idx + I8Stream::K_GROUP_SIZE],
        );
        current_ofs += I8Stream::K_GROUP_SIZE;
        out_idx += I8Stream::K_GROUP_SIZE;
        remaining -= I8Stream::K_GROUP_SIZE;
    }

    if remaining > 0 {
        dequantize_group_f32(packed, current_ofs, &mut dst[out_idx..out_idx + remaining]);
    }
}
