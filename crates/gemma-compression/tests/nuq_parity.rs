use gemma_compression::nuq::{decompress_and_zero_pad_f32, encode_f32, K_CLUSTERS, K_GROUP_SIZE};
use gemma_compression::types::NuqStream;

#[test]
fn nuq_flat_cluster() {
    let input = vec![0.5f32; K_GROUP_SIZE];
    let packed_len = gemma_compression::types::NuqStream::packed_end(input.len());
    let mut packed = vec![NuqStream { byte: 0 }; packed_len];
    let unused = encode_f32(&input, &mut packed, 0);
    assert_eq!(unused, K_CLUSTERS - 1);

    let mut out = vec![0.0f32; K_GROUP_SIZE];
    decompress_and_zero_pad_f32(&packed, 0, &mut out, K_GROUP_SIZE);
    for v in out {
        assert_eq!(v.to_bits(), 0.5f32.to_bits());
    }
}

#[test]
fn nuq_plateaus_exact() {
    let mut input = vec![0.0f32; K_GROUP_SIZE];
    for i in 0..K_GROUP_SIZE {
        let idx_cluster = i / (K_GROUP_SIZE / K_CLUSTERS);
        input[i] = (idx_cluster as f32 / K_CLUSTERS as f32) - 0.5f32;
    }

    // Deterministic shuffle.
    let mut seed = 0x1234_5678u64;
    for i in (1..K_GROUP_SIZE).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let j = (seed as usize) % (i + 1);
        input.swap(i, j);
    }

    let packed_len = gemma_compression::types::NuqStream::packed_end(input.len());
    let mut packed = vec![NuqStream { byte: 0 }; packed_len];
    let unused = encode_f32(&input, &mut packed, 0);
    assert_eq!(unused, 0);

    let mut out = vec![0.0f32; K_GROUP_SIZE];
    decompress_and_zero_pad_f32(&packed, 0, &mut out, K_GROUP_SIZE);
    for (i, v) in input.iter().enumerate() {
        assert_eq!(v.to_bits(), out[i].to_bits());
    }
}
