//! Scalar KV cache for autoregressive decoding.

pub struct KVCache {
    /// Interleaved storage: K followed by V for each (position, head).
    data: Vec<f32>,
    max_seq_len: usize,
    kv_heads: usize,
    head_dim: usize,
    filled_len: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, kv_heads: usize, head_dim: usize) -> Self {
        assert!(max_seq_len > 0, "max_seq_len must be > 0");
        let size = max_seq_len
            .checked_mul(kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .and_then(|n| n.checked_mul(2))
            .expect("KV cache size overflow");
        Self {
            data: vec![0.0; size],
            max_seq_len,
            kv_heads,
            head_dim,
            filled_len: 0,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn filled_len(&self) -> usize {
        self.filled_len
    }

    pub fn write_k(&mut self, pos: usize, kv_head: usize, src: &[f32]) {
        self.write(pos, kv_head, src, true);
    }

    pub fn write_v(&mut self, pos: usize, kv_head: usize, src: &[f32]) {
        self.write(pos, kv_head, src, false);
    }

    pub fn read_k(&self, pos: usize, kv_head: usize) -> &[f32] {
        self.read(pos, kv_head, true)
    }

    pub fn read_v(&self, pos: usize, kv_head: usize) -> &[f32] {
        self.read(pos, kv_head, false)
    }

    fn write(&mut self, pos: usize, kv_head: usize, src: &[f32], dst_is_k: bool) {
        assert!(kv_head < self.kv_heads, "KV head out of bounds");
        assert_eq!(src.len(), self.head_dim, "input vector dim mismatch");

        self.filled_len = self
            .filled_len
            .max(pos.saturating_add(1))
            .min(self.max_seq_len);
        let pos_mod = pos % self.max_seq_len;
        let base = (pos_mod * self.kv_heads + kv_head) * self.head_dim * 2;
        let offset = if dst_is_k { 0 } else { self.head_dim };
        let dst = &mut self.data;
        dst[base + offset..base + offset + self.head_dim].copy_from_slice(src);
    }

    fn read(&self, pos: usize, kv_head: usize, src_is_k: bool) -> &[f32] {
        assert!(kv_head < self.kv_heads, "KV head out of bounds");

        let pos_mod = pos % self.max_seq_len;
        let base = (pos_mod * self.kv_heads + kv_head) * self.head_dim * 2;
        let offset = if src_is_k { 0 } else { self.head_dim };
        let src = &self.data;
        &src[base + offset..base + offset + self.head_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::KVCache;

    #[test]
    fn writes_and_reads_per_position_and_head() {
        let mut cache = KVCache::new(4, 2, 3);

        cache.write_k(0, 0, &[1.0, 2.0, 3.0]);
        cache.write_v(0, 0, &[4.0, 5.0, 6.0]);
        cache.write_k(1, 1, &[7.0, 8.0, 9.0]);

        assert_eq!(cache.read_k(0, 0), &[1.0, 2.0, 3.0]);
        assert_eq!(cache.read_v(0, 0), &[4.0, 5.0, 6.0]);
        assert_eq!(cache.read_k(1, 1), &[7.0, 8.0, 9.0]);
        assert_eq!(cache.read_v(1, 1), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn wraps_positions_by_sequence_length() {
        let mut cache = KVCache::new(2, 1, 2);

        cache.write_k(0, 0, &[1.0, 2.0]);
        cache.write_k(1, 0, &[3.0, 4.0]);
        cache.write_k(2, 0, &[5.0, 6.0]);

        assert_eq!(cache.read_k(0, 0), &[5.0, 6.0]);
        assert_eq!(cache.read_k(1, 0), &[3.0, 4.0]);
        assert_eq!(cache.read_k(2, 0), &[5.0, 6.0]);
    }
}
