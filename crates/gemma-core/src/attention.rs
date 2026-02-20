//! Attention range helpers aligned with gemma.cpp behavior.

pub fn start_pos(pos: usize, attention_window_size: usize) -> usize {
    if attention_window_size == 0 {
        return 0;
    }
    pos.saturating_sub(attention_window_size - 1)
}

pub fn last_pos(pos: usize, prefix_end: Option<usize>) -> usize {
    let mut out = pos;
    if let Some(prefix_end) = prefix_end {
        if prefix_end > 0 {
            out = out.max(prefix_end - 1);
        }
    }
    out
}

pub fn clamp_start_to_cache_window(
    start_pos: usize,
    last_pos: usize,
    cache_seq_len: usize,
) -> usize {
    if cache_seq_len == 0 {
        return start_pos;
    }
    let rolling_start = (last_pos + 1).saturating_sub(cache_seq_len);
    start_pos.max(rolling_start)
}

pub fn attention_span(
    pos: usize,
    attention_window_size: usize,
    prefix_end: Option<usize>,
    cache_seq_len: usize,
) -> (usize, usize) {
    let last = last_pos(pos, prefix_end);
    let start =
        clamp_start_to_cache_window(start_pos(pos, attention_window_size), last, cache_seq_len);
    (start, last)
}

pub fn kv_head_for_query_head(head: usize, heads: usize, kv_heads: usize) -> usize {
    assert!(kv_heads > 0, "kv_heads must be > 0");
    assert!(heads > 0, "heads must be > 0");
    assert!(
        heads % kv_heads == 0,
        "heads must be divisible by kv_heads for grouped-query attention"
    );
    let head_groups = heads / kv_heads;
    head / head_groups
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AttentionPlan {
    pub start_pos: usize,
    pub last_pos: usize,
}

pub fn build_attention_plan(
    pos: usize,
    attention_window_size: usize,
    prefix_end: Option<usize>,
    cache_seq_len: usize,
) -> AttentionPlan {
    let (start_pos, last_pos) =
        attention_span(pos, attention_window_size, prefix_end, cache_seq_len);
    AttentionPlan {
        start_pos,
        last_pos,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        attention_span, build_attention_plan, clamp_start_to_cache_window, kv_head_for_query_head,
        last_pos, start_pos,
    };

    #[test]
    fn start_pos_matches_window_rule() {
        assert_eq!(start_pos(0, 32), 0);
        assert_eq!(start_pos(5, 4), 2);
        assert_eq!(start_pos(5, 1), 5);
    }

    #[test]
    fn last_pos_respects_prefix_end() {
        assert_eq!(last_pos(7, None), 7);
        assert_eq!(last_pos(7, Some(0)), 7);
        assert_eq!(last_pos(7, Some(6)), 7);
        assert_eq!(last_pos(7, Some(12)), 11);
    }

    #[test]
    fn start_pos_clamps_to_rolling_window() {
        assert_eq!(clamp_start_to_cache_window(0, 3, 8), 0);
        assert_eq!(clamp_start_to_cache_window(0, 10, 4), 7);
        assert_eq!(clamp_start_to_cache_window(9, 10, 4), 9);
    }

    #[test]
    fn attention_span_uses_prefix_end_and_cache_clamp() {
        assert_eq!(attention_span(3, 10, Some(8), 4), (4, 7));
        assert_eq!(attention_span(7, 4, None, 16), (4, 7));
    }

    #[test]
    fn kv_head_mapping_matches_gqa_groups() {
        // 8 query heads, 2 KV heads => groups of 4 query heads per KV head.
        assert_eq!(kv_head_for_query_head(0, 8, 2), 0);
        assert_eq!(kv_head_for_query_head(3, 8, 2), 0);
        assert_eq!(kv_head_for_query_head(4, 8, 2), 1);
        assert_eq!(kv_head_for_query_head(7, 8, 2), 1);
    }

    #[test]
    fn build_attention_plan_matches_span_helpers() {
        let plan = build_attention_plan(6, 4, Some(12), 5);
        assert_eq!(plan.start_pos, 7);
        assert_eq!(plan.last_pos, 11);
    }
}
