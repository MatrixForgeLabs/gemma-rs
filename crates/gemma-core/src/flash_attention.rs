//! Scalar causal flash-attention helpers.

use crate::kv_cache::KVCache;
use gemma_ops::dot::dot_f32;
use gemma_ops::nn::rope_inplace;

pub fn flash_attention_causal(
    q: &[f32],
    cache: &KVCache,
    kv_head: usize,
    start_pos: usize,
    last_pos: usize,
    query_scale: f32,
    logits_soft_cap: Option<f32>,
    out: &mut [f32],
) {
    assert!(start_pos <= last_pos);
    assert_eq!(q.len(), out.len());
    out.fill(0.0);

    let mut m = f32::NEG_INFINITY;
    let mut d = 0.0f32;

    for pos in start_pos..=last_pos {
        let k = cache.read_k(pos, kv_head);
        let mut x = dot_f32(q, k) * query_scale;
        if let Some(cap) = logits_soft_cap {
            if cap > 0.0 {
                x = cap * (x / cap).tanh();
            }
        }

        let m_new = m.max(x);
        let scale_old = if d == 0.0 { 0.0 } else { (m - m_new).exp() * d };
        let contrib = (x - m_new).exp();
        let d_new = scale_old + contrib;
        let inv_d = 1.0 / d_new.max(1e-20);

        let v = cache.read_v(pos, kv_head);
        for i in 0..out.len() {
            out[i] = (out[i] * scale_old + contrib * v[i]) * inv_d;
        }

        m = m_new;
        d = d_new;
    }
}

pub struct HeadFlashParams {
    pub pos: usize,
    pub kv_head: usize,
    pub start_pos: usize,
    pub last_pos: usize,
    pub query_scale: f32,
    pub logits_soft_cap: Option<f32>,
}

pub fn run_head_flash_attention(
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    cache: &mut KVCache,
    params: HeadFlashParams,
    out: &mut [f32],
) {
    rope_inplace(q, params.pos as f32);
    rope_inplace(k, params.pos as f32);
    cache.write_k(params.pos, params.kv_head, k);
    cache.write_v(params.pos, params.kv_head, v);
    flash_attention_causal(
        q,
        cache,
        params.kv_head,
        params.start_pos,
        params.last_pos,
        params.query_scale,
        params.logits_soft_cap,
        out,
    );
}

#[cfg(test)]
mod tests {
    use super::{flash_attention_causal, run_head_flash_attention, HeadFlashParams};
    use crate::kv_cache::KVCache;
    use gemma_ops::nn::rope_inplace;

    fn naive_attention(
        q: &[f32],
        cache: &KVCache,
        kv_head: usize,
        start_pos: usize,
        last_pos: usize,
        query_scale: f32,
        logits_soft_cap: Option<f32>,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; last_pos - start_pos + 1];
        for t in start_pos..=last_pos {
            let k = cache.read_k(t, kv_head);
            let mut s = 0.0f32;
            for i in 0..q.len() {
                s += q[i] * k[i];
            }
            let mut s = s * query_scale;
            if let Some(cap) = logits_soft_cap {
                if cap > 0.0 {
                    s = cap * (s / cap).tanh();
                }
            }
            scores[t - start_pos] = s;
        }

        let mut max = f32::NEG_INFINITY;
        for &v in &scores {
            if v > max {
                max = v;
            }
        }
        let mut denom = 0.0f32;
        for s in &mut scores {
            *s = (*s - max).exp();
            denom += *s;
        }
        let inv = 1.0 / denom.max(1e-20);
        for s in &mut scores {
            *s *= inv;
        }

        let mut out = vec![0.0f32; q.len()];
        for t in start_pos..=last_pos {
            let v = cache.read_v(t, kv_head);
            for i in 0..q.len() {
                out[i] += scores[t - start_pos] * v[i];
            }
        }
        out
    }

    #[test]
    fn flash_attention_matches_naive_softmax() {
        let mut cache = KVCache::new(4, 2, 4);
        cache.write_k(0, 1, &[0.2, -0.1, 0.4, 0.0]);
        cache.write_v(0, 1, &[1.0, 0.0, 0.5, -1.0]);
        cache.write_k(1, 1, &[0.1, 0.3, -0.2, 0.7]);
        cache.write_v(1, 1, &[0.2, 0.4, -0.3, 0.1]);
        cache.write_k(2, 1, &[-0.5, 0.6, 0.2, -0.4]);
        cache.write_v(2, 1, &[0.8, -0.2, 0.9, 0.3]);

        let q = vec![0.3, -0.7, 0.5, 0.2];
        let scale = 1.0 / (q.len() as f32).sqrt();

        let expected = naive_attention(&q, &cache, 1, 0, 2, scale, None);
        let mut actual = vec![0.0f32; q.len()];
        flash_attention_causal(&q, &cache, 1, 0, 2, scale, None, &mut actual);

        for i in 0..q.len() {
            assert!(
                (actual[i] - expected[i]).abs() < 1e-5,
                "index {i} mismatch: actual={} expected={}",
                actual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn flash_attention_respects_start_pos_and_cache_wrap() {
        let mut cache = KVCache::new(2, 1, 2);
        cache.write_k(0, 0, &[1.0, 0.0]);
        cache.write_v(0, 0, &[1.0, 0.0]);
        cache.write_k(1, 0, &[0.0, 1.0]);
        cache.write_v(1, 0, &[0.0, 1.0]);
        cache.write_k(2, 0, &[1.0, 1.0]); // overwrites slot for pos 0
        cache.write_v(2, 0, &[0.5, 0.5]);

        let q = vec![1.0, 0.0];
        let scale = 1.0;
        let expected = naive_attention(&q, &cache, 0, 1, 2, scale, None);
        let mut actual = vec![0.0f32; 2];
        flash_attention_causal(&q, &cache, 0, 1, 2, scale, None, &mut actual);

        for i in 0..2 {
            assert!((actual[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn flash_attention_matches_naive_with_logits_softcap() {
        let mut cache = KVCache::new(4, 1, 4);
        cache.write_k(0, 0, &[1.0, 0.0, 0.0, 0.0]);
        cache.write_v(0, 0, &[1.0, 2.0, 3.0, 4.0]);
        cache.write_k(1, 0, &[0.0, 1.0, 0.0, 0.0]);
        cache.write_v(1, 0, &[4.0, 3.0, 2.0, 1.0]);
        cache.write_k(2, 0, &[0.0, 0.0, 1.0, 0.0]);
        cache.write_v(2, 0, &[0.5, 0.5, 0.5, 0.5]);

        let q = vec![20.0, -10.0, 5.0, 1.0];
        let scale = 1.0;
        let cap = Some(2.0);

        let expected = naive_attention(&q, &cache, 0, 0, 2, scale, cap);
        let mut actual = vec![0.0f32; q.len()];
        flash_attention_causal(&q, &cache, 0, 0, 2, scale, cap, &mut actual);

        for i in 0..q.len() {
            assert!(
                (actual[i] - expected[i]).abs() < 1e-5,
                "index {i} mismatch: actual={} expected={}",
                actual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn run_head_flash_attention_writes_cache_and_matches_causal() {
        let mut cache_a = KVCache::new(8, 1, 4);
        let mut cache_b = KVCache::new(8, 1, 4);
        let mut q = vec![0.1, 0.2, 0.3, 0.4];
        let mut k = vec![0.0, 1.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let params = HeadFlashParams {
            pos: 0,
            kv_head: 0,
            start_pos: 0,
            last_pos: 0,
            query_scale: 1.0,
            logits_soft_cap: None,
        };

        let mut out_a = vec![0.0; 4];
        run_head_flash_attention(&mut q, &mut k, &v, &mut cache_a, params, &mut out_a);

        let mut q_b = vec![0.1, 0.2, 0.3, 0.4];
        let mut k_b = vec![0.0, 1.0, 0.0, 1.0];
        rope_inplace(&mut q_b, 0.0);
        rope_inplace(&mut k_b, 0.0);
        cache_b.write_k(0, 0, &k_b);
        cache_b.write_v(0, 0, &v);
        let mut out_b = vec![0.0; 4];
        flash_attention_causal(&q_b, &cache_b, 0, 0, 0, 1.0, None, &mut out_b);

        assert_eq!(cache_a.read_k(0, 0), cache_b.read_k(0, 0));
        assert_eq!(cache_a.read_v(0, 0), cache_b.read_v(0, 0));
        for i in 0..4 {
            assert!((out_a[i] - out_b[i]).abs() < 1e-6);
        }
    }
}
