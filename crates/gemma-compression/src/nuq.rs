//! Scalar NUQ codec compatible with gemma.cpp.

use half::bf16;

use crate::sfp::{f32_from_sfp8, sfp8_from_f32};
use crate::types::NuqStream;

pub const K_CLUSTERS: usize = NuqStream::K_CLUSTERS;
pub const K_GROUP_SIZE: usize = NuqStream::K_GROUP_SIZE;

const fn table_byte_offset(packed_ofs: usize) -> usize {
    let bytes_per_group = K_CLUSTERS + K_GROUP_SIZE / 2;
    (packed_ofs / K_GROUP_SIZE) * bytes_per_group
}

fn packed_bytes(packed: &[NuqStream]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len()) }
}

fn packed_bytes_mut(packed: &mut [NuqStream]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(packed.as_mut_ptr() as *mut u8, packed.len()) }
}

#[inline]
fn float_payload_clear(bits: u32) -> u32 {
    bits & !(K_GROUP_SIZE as u32 - 1)
}

#[inline]
fn float_payload_set(f: f32, bits: usize) -> f32 {
    let mut v = f.to_bits();
    v = float_payload_clear(v);
    v |= bits as u32;
    f32::from_bits(v)
}

#[inline]
fn float_payload_get(f: f32) -> usize {
    (f.to_bits() & (K_GROUP_SIZE as u32 - 1)) as usize
}

fn cost_prefix(
    sorted: &[f32; K_GROUP_SIZE],
) -> (
    [f32; K_GROUP_SIZE + 1],
    [f32; K_GROUP_SIZE + 1],
    [f64; K_GROUP_SIZE + 1],
) {
    let mut cumsum = [0.0f32; K_GROUP_SIZE + 1];
    let mut cumsum2 = [0.0f32; K_GROUP_SIZE + 1];
    let mut dcumsum = [0.0f64; K_GROUP_SIZE + 1];
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for i in 0..K_GROUP_SIZE {
        let v = f32::from_bits(float_payload_clear(sorted[i].to_bits()));
        sum += v as f64;
        sum2 += (v as f64) * (v as f64);
        dcumsum[i + 1] = sum;
        cumsum[i + 1] = sum as f32;
        cumsum2[i + 1] = sum2 as f32;
    }
    (cumsum, cumsum2, dcumsum)
}

#[inline]
fn range_cost(
    cumsum: &[f32; K_GROUP_SIZE + 1],
    cumsum2: &[f32; K_GROUP_SIZE + 1],
    first: usize,
    last: usize,
) -> f32 {
    let sum = cumsum[last + 1] - cumsum[first];
    let sum2 = cumsum2[last + 1] - cumsum2[first];
    let len = (last - first + 1) as f32;
    let mu = sum / len;
    let l2 = sum2 + mu * (mu * len - 2.0 * sum);
    if l2 < 0.0 {
        0.0
    } else {
        l2
    }
}

#[inline]
fn range_sum(dcumsum: &[f64; K_GROUP_SIZE + 1], first: usize, last: usize) -> f64 {
    dcumsum[last + 1] - dcumsum[first]
}

fn cluster_exact_l2(
    x: &[f32],
    num: usize,
    centers: &mut [f32; K_CLUSTERS],
    indices: &mut [u16; K_GROUP_SIZE],
) -> usize {
    assert!(num <= K_GROUP_SIZE);

    let mut sorted_and_i = [0.0f32; K_GROUP_SIZE];
    for i in 0..num {
        sorted_and_i[i] = float_payload_set(x[i], i);
    }
    if num != K_GROUP_SIZE {
        let mut max_val = -1.0e38f32;
        for i in 0..num {
            if x[i] > max_val {
                max_val = x[i];
            }
        }
        for i in num..K_GROUP_SIZE {
            sorted_and_i[i] = float_payload_set(max_val, i);
        }
    }

    sorted_and_i.sort_by(|a, b| a.total_cmp(b));

    let (cumsum, cumsum2, dcumsum) = cost_prefix(&sorted_and_i);

    let mut costs = [[0.0f32; K_GROUP_SIZE]; K_CLUSTERS];
    let mut argmin = [[0usize; K_GROUP_SIZE]; K_CLUSTERS];

    for last in 0..K_GROUP_SIZE {
        costs[0][last] = range_cost(&cumsum, &cumsum2, 0, last);
        argmin[0][last] = 0;
    }

    for cluster_idx in 1..K_CLUSTERS {
        for last in 0..K_GROUP_SIZE {
            let mut min_cost = costs[cluster_idx - 1][last];
            let mut arg = argmin[cluster_idx - 1][last];
            for first in 1..=last {
                let c =
                    costs[cluster_idx - 1][first - 1] + range_cost(&cumsum, &cumsum2, first, last);
                if c < min_cost {
                    min_cost = c;
                    arg = first;
                }
            }
            costs[cluster_idx][last] = min_cost;
            argmin[cluster_idx][last] = arg;
        }
    }

    let mut last = K_GROUP_SIZE - 1;
    let mut unused_clusters = 0usize;
    for k in (0..K_CLUSTERS).rev() {
        let start = argmin[k][last];
        let sum = range_sum(&dcumsum, start, last);
        let size = (last - start + 1) as f32;
        centers[k] = (sum / size as f64) as f32;

        for i in start..=last {
            let idx = float_payload_get(sorted_and_i[i]);
            indices[idx] = k as u16;
        }

        if start == 0 {
            unused_clusters = k;
            for c in 0..unused_clusters {
                centers[c] = 0.0;
            }
            break;
        }
        last = start - 1;
    }

    unused_clusters
}

fn pack_indices(indices: &[u16], dst: &mut [u8]) {
    let mut i = 0usize;
    let mut out = 0usize;
    while i < indices.len() {
        let lo = (indices[i] & 0xF) as u8;
        let hi = if i + 1 < indices.len() {
            (indices[i + 1] & 0xF) as u8
        } else {
            0
        };
        dst[out] = lo | (hi << 4);
        out += 1;
        i += 2;
    }
}

fn unpack_index(indices: &[u8], idx: usize) -> usize {
    let byte = indices[idx / 2];
    if idx % 2 == 0 {
        (byte & 0xF) as usize
    } else {
        ((byte >> 4) & 0xF) as usize
    }
}

pub fn encode_f32(raw: &[f32], packed: &mut [NuqStream], packed_ofs: usize) -> usize {
    assert!(packed_ofs % K_GROUP_SIZE == 0);
    if raw.is_empty() {
        return 0;
    }

    let bytes = packed_bytes_mut(packed);
    let mut current_offset = packed_ofs;
    let num_groups = (raw.len() + K_GROUP_SIZE - 1) / K_GROUP_SIZE;
    let mut unused_clusters_total = 0usize;

    for g in 0..num_groups {
        let g_start = g * K_GROUP_SIZE;
        let g_num = (raw.len() - g_start).min(K_GROUP_SIZE);
        let g_in = &raw[g_start..g_start + g_num];

        let mut centers = [0.0f32; K_CLUSTERS];
        let mut idx = [0u16; K_GROUP_SIZE];
        let unused = cluster_exact_l2(g_in, g_num, &mut centers, &mut idx);
        unused_clusters_total += unused;

        let table_off = table_byte_offset(current_offset);
        for i in 0..K_CLUSTERS {
            bytes[table_off + i] = sfp8_from_f32(centers[i]);
        }
        let packed_start = table_off + K_CLUSTERS;
        let bytes_needed = (g_num + 1) / 2;
        pack_indices(
            &idx[..g_num],
            &mut bytes[packed_start..packed_start + bytes_needed],
        );

        current_offset += g_num;
    }

    unused_clusters_total
}

fn decode_group_f32(
    table_bytes: &[u8],
    indices_bytes: &[u8],
    start: usize,
    num: usize,
    dst: &mut [f32],
) {
    let mut centers = [0.0f32; K_CLUSTERS];
    for i in 0..K_CLUSTERS {
        centers[i] = f32_from_sfp8(table_bytes[i]);
    }
    let max_indices = indices_bytes.len() * 2;
    for i in 0..num {
        let idx_pos = start + i;
        if idx_pos >= max_indices {
            break;
        }
        let idx = unpack_index(indices_bytes, idx_pos);
        dst[i] = centers[idx];
    }
}

fn decode_group_bf16(
    table_bytes: &[u8],
    indices_bytes: &[u8],
    start: usize,
    num: usize,
    dst: &mut [bf16],
) {
    let mut centers = [0.0f32; K_CLUSTERS];
    for i in 0..K_CLUSTERS {
        centers[i] = f32_from_sfp8(table_bytes[i]);
    }
    let max_indices = indices_bytes.len() * 2;
    for i in 0..num {
        let idx_pos = start + i;
        if idx_pos >= max_indices {
            break;
        }
        let idx = unpack_index(indices_bytes, idx_pos);
        dst[i] = bf16::from_f32(centers[idx]);
    }
}

pub fn decompress_and_zero_pad_f32(
    packed: &[NuqStream],
    packed_ofs: usize,
    dst: &mut [f32],
    num: usize,
) {
    if num == 0 {
        return;
    }
    assert!(dst.len() >= num);
    dst[num..].fill(0.0);

    let bytes = packed_bytes(packed);
    let mut current_offset = packed_ofs;
    let mut out_idx = 0usize;
    let mut remaining = num;

    let within_group = current_offset % K_GROUP_SIZE;
    if within_group != 0 {
        let group_start = current_offset - within_group;
        let table_off = table_byte_offset(group_start);
        let table = &bytes[table_off..table_off + K_CLUSTERS];
        let num_in_first = remaining.min(K_GROUP_SIZE - within_group);
        let indices_len = (within_group + num_in_first + 1) / 2;
        let indices = &bytes[table_off + K_CLUSTERS..table_off + K_CLUSTERS + indices_len];
        decode_group_f32(
            table,
            indices,
            within_group,
            num_in_first,
            &mut dst[out_idx..out_idx + num_in_first],
        );
        current_offset += num_in_first;
        out_idx += num_in_first;
        remaining -= num_in_first;
    }

    while remaining > 0 {
        let table_off = table_byte_offset(current_offset);
        let table = &bytes[table_off..table_off + K_CLUSTERS];
        let num_in_group = remaining.min(K_GROUP_SIZE);
        let indices_len = (num_in_group + 1) / 2;
        let indices = &bytes[table_off + K_CLUSTERS..table_off + K_CLUSTERS + indices_len];
        decode_group_f32(
            table,
            indices,
            0,
            num_in_group,
            &mut dst[out_idx..out_idx + num_in_group],
        );
        current_offset += num_in_group;
        out_idx += num_in_group;
        remaining -= num_in_group;
    }
}

pub fn decompress_and_zero_pad_bf16(
    packed: &[NuqStream],
    packed_ofs: usize,
    dst: &mut [bf16],
    num: usize,
) {
    if num == 0 {
        return;
    }
    assert!(dst.len() >= num);
    dst[num..].fill(bf16::from_f32(0.0));

    let bytes = packed_bytes(packed);
    let mut current_offset = packed_ofs;
    let mut out_idx = 0usize;
    let mut remaining = num;

    let within_group = current_offset % K_GROUP_SIZE;
    if within_group != 0 {
        let group_start = current_offset - within_group;
        let table_off = table_byte_offset(group_start);
        let table = &bytes[table_off..table_off + K_CLUSTERS];
        let num_in_first = remaining.min(K_GROUP_SIZE - within_group);
        let indices_len = (within_group + num_in_first + 1) / 2;
        let indices = &bytes[table_off + K_CLUSTERS..table_off + K_CLUSTERS + indices_len];
        decode_group_bf16(
            table,
            indices,
            within_group,
            num_in_first,
            &mut dst[out_idx..out_idx + num_in_first],
        );
        current_offset += num_in_first;
        out_idx += num_in_first;
        remaining -= num_in_first;
    }

    while remaining > 0 {
        let table_off = table_byte_offset(current_offset);
        let table = &bytes[table_off..table_off + K_CLUSTERS];
        let num_in_group = remaining.min(K_GROUP_SIZE);
        let indices_len = (num_in_group + 1) / 2;
        let indices = &bytes[table_off + K_CLUSTERS..table_off + K_CLUSTERS + indices_len];
        decode_group_bf16(
            table,
            indices,
            0,
            num_in_group,
            &mut dst[out_idx..out_idx + num_in_group],
        );
        current_offset += num_in_group;
        out_idx += num_in_group;
        remaining -= num_in_group;
    }
}
