use gemma_core::attention::{attention_span, kv_head_for_query_head};

#[test]
fn gqa_head_group_mapping_fixture() {
    // heads=16, kv_heads=4 => each KV head owns 4 query heads.
    let expected = [
        0usize, 0, 0, 0, // 0..3
        1, 1, 1, 1, // 4..7
        2, 2, 2, 2, // 8..11
        3, 3, 3, 3, // 12..15
    ];
    for (h, &kv) in expected.iter().enumerate() {
        assert_eq!(kv_head_for_query_head(h, 16, 4), kv, "head {h}");
    }
}

#[test]
fn gqa_mapping_identity_when_heads_equal_kv_heads() {
    for h in 0..8usize {
        assert_eq!(kv_head_for_query_head(h, 8, 8), h, "head {h}");
    }
}

#[test]
fn gqa_mapping_collapses_when_single_kv_head() {
    for h in 0..12usize {
        assert_eq!(kv_head_for_query_head(h, 12, 1), 0, "head {h}");
    }
}

#[test]
fn attention_span_window_roll_prefix_interaction() {
    // pos=6, window=4 => raw start=3; prefix_end=10 => last=9.
    // cache_seq_len=5 => rolling start = (9+1)-5 = 5, so final span is [5,9].
    let (start, last) = attention_span(6, 4, Some(10), 5);
    assert_eq!((start, last), (5, 9));
}

#[test]
fn attention_span_prefix_end_extends_last_when_window_is_small() {
    // pos=2, window=2 => raw start=1; prefix_end=8 => last=7.
    let (start, last) = attention_span(2, 2, Some(8), 64);
    assert_eq!((start, last), (1, 7));
}

#[test]
fn attention_span_prefix_end_is_clamped_by_rolling_cache_window() {
    // pos=2, window=2 => raw start=1; prefix_end=20 => last=19.
    // cache_seq_len=4 => rolling start=(19+1)-4=16, so final span [16,19].
    let (start, last) = attention_span(2, 2, Some(20), 4);
    assert_eq!((start, last), (16, 19));
}

#[test]
fn attention_span_matrix_covers_prefix_window_and_cache_edges() {
    let cases = [
        // (pos, window, prefix_end, cache_seq_len, expected_start, expected_last)
        (0usize, 4usize, None, 32usize, 0usize, 0usize),
        (5, 1, None, 32, 5, 5),
        (5, 4, Some(0), 32, 2, 5),
        (5, 4, Some(3), 32, 2, 5),
        (5, 4, Some(8), 32, 2, 7),
        (5, 4, Some(8), 3, 5, 7),
        (10, 0, Some(14), 4, 10, 13),
    ];
    for (pos, window, prefix_end, cache, exp_start, exp_last) in cases {
        let (start, last) = attention_span(pos, window, prefix_end, cache);
        assert_eq!(
            (start, last),
            (exp_start, exp_last),
            "case pos={pos} window={window} prefix_end={prefix_end:?} cache={cache}"
        );
    }
}
