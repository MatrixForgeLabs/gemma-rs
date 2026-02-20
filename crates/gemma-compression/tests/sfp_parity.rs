use gemma_compression::sfp::{f32_from_sfp8, sfp8_from_f32};

#[test]
fn sfp_golden_values() {
    let golden: &[(f32, f32)] = &[
        (0.46875, 0.46875),
        (0.9375, 0.9375),
        (0.484375, 0.5),
        (0.96875, 1.0),
        (0.28125, 0.28125),
        (0.5625, 0.5625),
        (0.296875, 0.3125),
        (0.59375, 0.625),
        (0.279296875, 0.28125),
        (0.55859375, 0.5625),
        (0.265625, 0.25),
        (0.53125, 0.5),
        (0.0068359375, 0.0068359375),
        (0.00732421875, 0.0078125),
        (0.007568359375, 0.0078125),
        (1.0, 1.0),
        (1.0625, 1.0),
        (2.384185791015625E-7, 2.384185791015625E-7),
        (1.49011611938E-7, 1.49011611938E-7),
        (1.19209289551E-7, 1.49011611938E-7),
        (5.96046447754E-8, 0.0),
        (8.94069671631E-8, 0.0),
        (1.11758708954E-7, 1.49011611938E-7),
        (0.013841, 0.013671875),
    ];

    for sign in [-1.0f32, 1.0f32] {
        for (input, expected) in golden {
            let in_val = input * sign;
            let out_val = expected * sign;
            let encoded = sfp8_from_f32(in_val);
            let decoded = f32_from_sfp8(encoded);
            if out_val == 0.0 {
                assert!(
                    decoded == 0.0,
                    "in {in_val} encoded {encoded} decoded {decoded} expected 0"
                );
            } else {
                assert!(
                    decoded.to_bits() == out_val.to_bits(),
                    "in {in_val} encoded {encoded} decoded {decoded} expected {out_val}"
                );
            }
        }
    }
}

#[test]
fn sfp_round_trip_all_encodings() {
    for encoded in 0u32..=255 {
        if encoded == 0x80 {
            continue;
        }
        let decoded = f32_from_sfp8(encoded as u8);
        let reencoded = sfp8_from_f32(decoded);
        assert_eq!(encoded as u8, reencoded, "encoded {encoded}");
    }
}
